---
title: "Agent vs. Chain vs. Workflow: The Decision Framework Every Engineer Needs"
date: "2026-06-22"
description: "A practical decision tree that maps task properties to the right orchestration pattern — chain, workflow, or agent — with real cost and complexity tradeoffs."
tags: ["ai-agents", "llm", "agent-architecture", "machine-learning", "deep-learning", "nlp", "system-design", "production-ml"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 32
---

I have seen a team spend three months building an agent for a task that a 20-line Python chain would have solved. The agent was technically impressive. It was also a disaster. Latency ballooned from 800 milliseconds to 11 seconds. Costs multiplied by 40×. When something broke — and it broke constantly — debugging required replaying entire LLM conversation histories through a system that made different decisions on every run.

When I finally asked why they built an agent instead of a chain, the answer was, "We thought agents were the right pattern for AI tasks."

That answer is how you get three months of waste. There is no such thing as a "right pattern for AI tasks" in the abstract. There are only task properties, and those properties deterministically map to a pattern. The framework in this post will let you make that mapping in under five minutes, before you write a single line of code.

## The Pattern Explosion: Why Everyone Defaults to Agents

The AI tooling ecosystem has a marketing problem. LangChain, AutoGen, LlamaIndex, CrewAI — every major framework leads with agents. Agents are the feature that gets the demo, the blog post, the conference talk. Chains and workflows are the boring plumbing underneath.

This creates a systematic bias: engineers reach for agents by default, then discover the hard way that agents introduce costs and complexity that are invisible in the demo but brutal in production.

The costs are real and measurable:

- A simple extraction chain costs approximately $0.002 per run. The same task implemented as an agent costs approximately $0.08 per run — a 40× increase.
- A chain debugs in 30 minutes when something breaks. An agent can take a full day because you need to understand which branch of a non-deterministic decision tree was followed.
- A chain takes 1–2 engineering days to build. An agent with proper observability, error handling, and self-correction logic takes 10–30 days.

None of this means agents are bad. It means agents are expensive, and you should buy that expense only when the task genuinely requires what agents provide. The decision framework in this post defines exactly what "genuinely requires" means.

## Defining the Three Patterns Precisely

Before the decision framework, we need precise definitions. "Chain," "workflow," and "agent" are used inconsistently across frameworks and teams. For the purposes of this post, the definitions are:

**A chain** is a fixed sequence of LLM calls and transformations where every run follows exactly the same path. There is no state between steps beyond what is explicitly passed, no conditional logic, no branching. Given the same input, a chain always produces the same sequence of operations. Chains are deterministic programs that happen to include LLM calls.

**A workflow** is a directed acyclic graph (DAG) of steps where execution follows conditional branches based on step outputs or external state. Workflows have explicit state, can retry failed steps, can run steps in parallel, and can take different paths through the graph depending on results. The full set of possible paths is defined at design time, even if the specific path taken is determined at runtime.

**An agent** is a system where an LLM dynamically decides at runtime which tools to call, in what order, and when to stop. The agent runs in a loop: observe → reason → act → observe. The specific tools called and the number of iterations are not known at design time. The agent has genuine decision-making authority over the execution path.

The distinction between a workflow and an agent is not about whether the system uses LLMs or calls external tools. A workflow can do both. The distinction is about who decides the execution path: a static DAG (workflow) or an LLM at runtime (agent).

![Decision tree mapping task properties to chain, workflow, or agent patterns](/imgs/blogs/agent-vs-chain-vs-workflow-1.webp)

## Chains: When Determinism Beats Flexibility

Chains are undervalued because they look boring. A chain that takes a customer email, extracts a category and sentiment, and routes it to a queue does not get demo time at conferences. But it runs at sub-second latency, costs fractions of a cent, and has a debugging experience that approximates reading a simple function call stack.

The right mental model for a chain is a function composition: `f₃(f₂(f₁(input)))`. Each `fᵢ` might be a prompt template, an LLM call, a parser, or a transformation function. The chain has no memory of previous inputs beyond what you explicitly carry forward. It cannot decide to skip a step or call a tool it was not designed to call.

### What chains are good at

Chains shine when the input-to-output mapping is well-defined and the processing steps are known in advance:

- **Information extraction**: Given a document, extract structured fields. The steps are always: parse → extract → validate → return.
- **Classification**: Given text, classify into categories. Always: preprocess → prompt → parse label → return.
- **Transformation**: Convert from one format to another. Always: parse input → transform → render output.
- **Summarization**: Given a long document, produce a summary. Always: chunk → summarize chunks → synthesize → return.
- **Report generation**: Given structured data, generate narrative text. Always: fetch data → template → generate → format.

For all of these, the shape of the computation is constant. You know at design time exactly what will happen on every run.

### Chain implementation

Here is what a chain looks like in code:

```python
from openai import OpenAI
import json

client = OpenAI()

def extract_invoice_fields(raw_text: str) -> dict:
    """
    Chain: raw text -> extract fields -> validate schema -> return structured data.
    Same 3 steps on every run. No branching. No state.
    Cost: ~$0.002/run. Latency: ~800ms.
    """
    # Step 1: Extract entities
    extraction_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract invoice fields as JSON: vendor, amount, date, invoice_number. Return only valid JSON."},
            {"role": "user", "content": raw_text}
        ],
        response_format={"type": "json_object"}
    )
    extracted = json.loads(extraction_response.choices[0].message.content)

    # Step 2: Validate schema
    required_fields = {"vendor", "amount", "date", "invoice_number"}
    missing = required_fields - set(extracted.keys())
    if missing:
        extracted["_missing_fields"] = list(missing)

    # Step 3: Normalize amounts
    if "amount" in extracted and isinstance(extracted["amount"], str):
        extracted["amount"] = float(extracted["amount"].replace("$", "").replace(",", ""))

    return extracted
```

Notice what this code does not have: it does not decide at runtime how many steps to run, which tools to call, or whether to retry with different prompts. The structure is fixed. This is a feature, not a limitation.

### Where chains fail

Chains fail when the task has inherent variability in its processing requirements:

- When different inputs need fundamentally different processing steps
- When a step can fail in ways that require a different subsequent step (not just a retry of the same step)
- When the task requires information gathered dynamically during execution
- When the number of operations depends on the input content

If you find yourself adding `if/else` logic to route through different prompts based on intermediate LLM output, you have outgrown a chain. You need a workflow.

![A chain: four deterministic steps in fixed order with no branching](/imgs/blogs/agent-vs-chain-vs-workflow-3.webp)

## Workflows: When Branching and Retries Justify Orchestration

A workflow is what you build when a chain is no longer sufficient but an agent would be overkill. The critical property of a workflow is that the full decision tree is known at design time. You can draw the DAG before writing any code. You know every possible path, every branch condition, every retry boundary.

This design-time knowability is what separates a workflow from an agent. In a workflow, the system knows where it can go before it starts. In an agent, the possible paths are unbounded.

### What workflows are good at

Workflows shine when the processing logic has conditional branches, parallel steps, or retry requirements, but the structure of those branches is fully specifiable:

- **Code review automation**: Lint the code → run tests → if both pass, LLM review → if LLM finds issues, post comment → if all clean, approve. The DAG is fully specifiable; the content of each step varies by input.
- **Document processing pipelines**: Classify document type → route to appropriate extractor → validate extracted data → if validation fails, retry with different extraction strategy → store result.
- **Customer support triage**: Classify intent → route to appropriate handler → if escalation required, alert human → log outcome. All paths are known.
- **Data validation pipelines**: Fetch data → validate schema → if schema invalid, log error and skip → transform → if transform fails, retry → load.
- **Content moderation**: Classify severity → if severe, immediately reject → if borderline, LLM review → if clean, pass → log all decisions.

For all of these, a senior engineer could draw the complete flowchart in an afternoon. The logic is complex — parallel branches, conditional routing, retries — but it is finite and enumerable.

### Workflow implementation

```python
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from openai import AsyncOpenAI

client = AsyncOpenAI()

class ReviewDecision(Enum):
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    ESCALATED = "escalated"

@dataclass
class PRReviewResult:
    lint_passed: bool
    tests_passed: bool
    test_retry_count: int
    review_decision: Optional[ReviewDecision]
    review_comments: Optional[str]

async def lint_pr(diff: str) -> bool:
    """Step 1a: Run linter (simulated; real impl calls your linter CLI)."""
    # In production: subprocess.run(["eslint", ...])
    return len(diff) < 50000  # simplified check

async def run_tests(diff: str, retry: int = 0) -> tuple[bool, int]:
    """Step 1b: Run tests with up to 2 retries on flaky failures."""
    max_retries = 2
    for attempt in range(max_retries + 1):
        # In production: subprocess.run(["pytest", ...])
        passed = True  # simplified
        if passed:
            return True, attempt
    return False, max_retries

async def llm_review(diff: str) -> tuple[ReviewDecision, str]:
    """Step 2: LLM code review (runs only if lint + tests pass)."""
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Review this code diff. Return JSON: {decision: 'approved'|'changes_requested'|'escalated', comments: 'explanation'}"},
            {"role": "user", "content": f"Review this diff:\n\n{diff[:4000]}"}
        ],
        response_format={"type": "json_object"}
    )
    import json
    result = json.loads(response.choices[0].message.content)
    return ReviewDecision(result["decision"]), result["comments"]

async def review_pr(diff: str) -> PRReviewResult:
    """
    Workflow: conditional DAG for PR review.
    
    DAG:
    lint + tests (parallel) -> gate -> LLM review -> decision
                                  ↘ retry tests (max 2)
    
    Cost: ~$0.01/run (1 LLM call for review, structured output).
    Latency: ~2s (parallel lint+test + 1 LLM call).
    """
    # Step 1: Run lint and tests in parallel
    lint_result, (test_result, retry_count) = await asyncio.gather(
        lint_pr(diff),
        run_tests(diff)
    )

    # Step 2: Gate — both must pass to proceed to LLM review
    if not lint_result or not test_result:
        return PRReviewResult(
            lint_passed=lint_result,
            tests_passed=test_result,
            test_retry_count=retry_count,
            review_decision=None,
            review_comments="Skipped: lint or tests failed"
        )

    # Step 3: LLM review (only reached if gate passes)
    decision, comments = await llm_review(diff)

    return PRReviewResult(
        lint_passed=True,
        tests_passed=True,
        test_retry_count=retry_count,
        review_decision=decision,
        review_comments=comments
    )
```

The workflow is explicit. You can read the DAG directly from the code. When something fails, you know exactly which step failed, what state it was in, and whether the retry fired. This is the right complexity for this task — more than a chain, far less than an agent.

### Where workflows fail

Workflows fail when the conditional logic becomes too complex to express as a static DAG, or when the next step depends on information that cannot be known at design time:

- When you need to decide which tools to call based on the content of intermediate LLM responses
- When the number of steps is genuinely unbounded (keep searching until you find an answer)
- When self-correction requires trying fundamentally different approaches rather than retrying the same step
- When the task requires synthesizing information from sources discovered during execution

If your workflow's decision tree has branches that say "call whatever tool makes sense here," you have crossed into agent territory.

![A workflow: conditional DAG with parallel branches and retry edges](/imgs/blogs/agent-vs-chain-vs-workflow-4.webp)

## Agents: When You Genuinely Need Dynamic Tool Use

Agents are the right pattern when — and only when — five conditions hold simultaneously:

1. **Tool selection is unknown at design time.** The specific tools or APIs to call depend on the content of the task, not just its category.
2. **The number of steps is genuinely unbounded.** The task may require 2 steps or 20, depending on what the agent discovers.
3. **Self-correction is required.** When the agent makes a wrong move, it needs to recognize this and try something different — not just retry the same step.
4. **The information space is open-ended.** The agent needs to gather information from sources it discovers during execution, not from a fixed set of sources.
5. **The cost is justified.** At 40× the token cost and 10× the engineering investment, you need a genuine capability gain that a workflow cannot provide.

If any of these five conditions does not hold, a workflow or chain will serve you better.

### What agents are actually good at

When the five conditions hold, agents provide genuine value:

- **Open-ended research**: "Find all papers published in the last six months that challenge the scaling hypothesis." The agent must discover which sources exist, query them, follow citations, synthesize findings. A workflow cannot enumerate these steps at design time.
- **Autonomous debugging**: "Diagnose why our API's P99 latency doubled last Thursday." The agent reads logs, queries metrics, searches documentation, hypothesizes causes, runs tests. Each step informs the next in ways that cannot be pre-specified.
- **Adaptive content creation**: "Write a blog post about our product launch, incorporating the latest press coverage and competitor responses." The agent must discover what coverage exists, assess its relevance, and synthesize accordingly.
- **Complex code generation with iteration**: "Implement this feature, run the tests, fix the failures, and repeat until tests pass." The number of fix-and-test cycles is unknown; self-correction is the core capability.

### Agent implementation

```python
import json
from openai import OpenAI
from typing import Any, Callable

client = OpenAI()

# Tool definitions — the agent chooses which to call at runtime
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Fetch and read the content of a URL or document",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "synthesize_findings",
            "description": "Produce a final synthesis report from gathered information. Call this when research is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "findings": {"type": "array", "items": {"type": "string"}},
                    "conclusion": {"type": "string"}
                },
                "required": ["findings", "conclusion"]
            }
        }
    }
]

def run_research_agent(
    task: str,
    tool_implementations: dict[str, Callable],
    max_turns: int = 10
) -> dict[str, Any]:
    """
    Research agent: dynamic tool use + self-correction loop.
    
    Cost: ~$0.08/run (multiple LLM turns, large context).
    Latency: ~8-15s (multiple turns + tool calls).
    Use ONLY when tool selection is unknown at design time.
    """
    messages = [
        {"role": "system", "content": "You are a research assistant. Use tools to gather information, then synthesize findings. Call synthesize_findings when done."},
        {"role": "user", "content": task}
    ]
    
    for turn in range(max_turns):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        
        choice = response.choices[0]
        messages.append({"role": "assistant", "content": choice.message.content, "tool_calls": choice.message.tool_calls})
        
        # If no tool calls, agent decided it's done
        if not choice.message.tool_calls:
            return {"status": "completed_without_synthesis", "messages": messages}
        
        # Execute tool calls
        for tool_call in choice.message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            if tool_name == "synthesize_findings":
                # Final synthesis — agent is done
                return {
                    "status": "success",
                    "findings": tool_args["findings"],
                    "conclusion": tool_args["conclusion"],
                    "turns": turn + 1
                }
            
            # Execute tool and add result to context
            tool_fn = tool_implementations.get(tool_name)
            if tool_fn:
                result = tool_fn(**tool_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
    
    return {"status": "max_turns_exceeded", "messages": messages}
```

The key difference from the workflow implementation: the agent decides at runtime which tools to call, in what order, and when it has enough information to synthesize. A workflow cannot implement this because the decision depends on the content of what the agent discovers.

![Layered stack: agent reasoning loop wrapping workflow wrapping chains](/imgs/blogs/agent-vs-chain-vs-workflow-5.webp)

## The Decision Framework: 4 Questions That Map to a Pattern

Use this framework before writing any code. Answer the questions in order; the first "yes" terminates the decision.

**Q1: Do the processing steps always execute in a fixed order, with no branching based on intermediate results?**

If yes → **Build a chain.** Your task has a constant computation shape. Use a chain, benefit from low cost, high determinism, and trivial debugging.

If no → proceed to Q2.

**Q2: Is the full decision tree specifiable at design time — every branch condition, every retry boundary, every parallel path?**

If yes → **Build a workflow.** Your task has conditional logic but bounded complexity. A workflow gives you branching without giving up knowability.

If no → proceed to Q3.

**Q3: Does the task require tool selection that cannot be predetermined — where the specific tools to use depend on what the agent discovers mid-execution?**

If yes → proceed to Q4.

If no → **Reconsider.** A workflow with a fixed tool set probably covers your use case. Re-examine whether the branching is truly unbounded or just complex.

**Q4: Does the task require self-correction — where a wrong tool call should lead to a fundamentally different approach, not just a retry of the same step?**

If yes → **Build an agent.** You have a genuine agent use case: dynamic tool selection, unbounded steps, and self-correction.

If no → **Build a workflow.** Add a broader fixed tool set to your workflow if needed. You do not need a full agent loop.

### The scoring matrix for Q3 and Q4

If you are uncertain about Q3 and Q4, score your task on two dimensions:

| Dimension | Score 1 | Score 2 | Score 3 |
|---|---|---|---|
| **Tool uncertainty** | All tools known at design time | Some tools known, some may be discovered | Tools cannot be enumerated at design time |
| **Step count** | Fixed N steps | Bounded range (2–8 steps) | Genuinely unbounded |

If both scores are 1 or 2 → workflow. If either score is 3 → agent candidate. Both scores of 3 → agent justified.

## Cost Comparison: Tokens, Latency, Engineering Time, Debugging Time

This section uses real numbers from production systems. I have rounded conservatively to avoid false precision, but the orders of magnitude are accurate.

**Token costs** (GPT-4o at $2.50/1M input, $10/1M output):

| Pattern | Input tokens | Output tokens | Cost per run |
|---|---|---|---|
| Chain | ~800 | ~200 | ~$0.002 |
| Workflow | ~2,500 | ~600 | ~$0.012 |
| Agent | ~12,000 | ~2,400 | ~$0.054 |

At 10,000 runs/day: chain costs $20/day; workflow costs $120/day; agent costs $540/day. The agent is 27× more expensive than the chain at scale.

**Latency** (P50 for a 5-step equivalent task):

| Pattern | P50 latency |
|---|---|
| Chain | ~800 ms |
| Workflow | ~2,200 ms |
| Agent | ~8,500 ms |

The agent is 10× slower than the chain. For user-facing features, this frequently makes agents inappropriate regardless of cost.

**Engineering time** (including observability, error handling, and testing):

| Pattern | Build time | Ongoing maintenance |
|---|---|---|
| Chain | 1–2 days | Low |
| Workflow | 3–5 days | Medium |
| Agent | 10–30 days | High |

The agent takes 10–15× longer to build correctly. This is not because agents are complicated to start; it is because correctly handling non-determinism, building replay-able traces, implementing cost caps, and writing meaningful tests takes substantial time.

**Debugging time per incident**:

| Pattern | Median time to diagnose |
|---|---|
| Chain | ~30 minutes |
| Workflow | ~2 hours |
| Agent | 8–16 hours |

This is the number that surprises most teams. Agent debugging is hard because the agent made different decisions on the failing run than on all previous runs. You must replay the exact conversation with the exact tool call results to reproduce the failure.

![Cost comparison per pattern across five production metrics](/imgs/blogs/agent-vs-chain-vs-workflow-6.webp)

## Debugging Complexity by Pattern

Debugging complexity scales differently for each pattern:

**Chain debugging** is nearly trivial. A chain is a function. Given the same input, it always does the same thing. To debug a chain failure: (1) log the input to each step, (2) find the step where output diverges from expectation, (3) fix that step. You can replay any failure by re-running the chain with the same input. Tests cover essentially all cases.

**Workflow debugging** is manageable. A workflow has state and conditional branches, so you need to record which branch was taken and what state transitions occurred. Tools like Prefect, Temporal, and Airflow provide built-in workflow execution history. To debug: (1) find the execution record, (2) identify which branch was taken, (3) inspect the state at that branch point. Replay requires the same input and the same external service responses (which may need mocking).

**Agent debugging** is hard. An agent makes non-deterministic decisions. The same input can produce different tool call sequences across runs depending on model temperature, token sampling, and the content of tool responses. To debug: (1) find the exact conversation history from the failing run, (2) identify which tool call sequence was chosen, (3) determine whether the problem was in tool selection, tool execution, or synthesis. Full replay requires capturing and replaying all tool call responses, which most teams do not instrument by default.

The exponential blow-up in agent debugging time has two components:
- **State space**: A chain with N steps has N states. A workflow with N steps and B branches per step has up to B^N states. An agent with N turns has unbounded states because the LLM can call any tool in any order.
- **Reproduction**: Chain failures reproduce deterministically. Workflow failures reproduce if you control state inputs. Agent failures may not reproduce at all if you cannot replay the exact LLM decisions.

To make agent debugging tractable, you must log every LLM turn, every tool call, every tool response, and every intermediate state. This is not optional instrumentation; it is load-bearing infrastructure that needs to be built before you ship the agent to production.

![Debugging trace complexity by pattern — chain is linear, workflow branches, agent has exponential decision space](/imgs/blogs/agent-vs-chain-vs-workflow-7.webp)

## Migration Paths

### Chain → Workflow

The chain-to-workflow migration is safe and reversible. You are adding state and branching to an existing computation graph. The typical trigger is one of:

- A chain step is failing and you need to retry with a different strategy
- Different input categories need different processing steps
- You need parallel processing of independent parts of the input

Migration pattern: wrap each chain step in a workflow task, add explicit state passing between tasks, add conditional edges where the branch logic lives. The existing chain logic usually survives largely intact as individual task implementations.

### Workflow → Agent

The workflow-to-agent migration is expensive and often a mistake. Most teams make this migration because a workflow's decision tree is getting complex, not because they need genuine dynamic tool selection. Before migrating:

1. Count the number of branches in your workflow. If it is under 20, the workflow is almost certainly the right abstraction.
2. Ask whether the branching complexity could be simplified by refactoring the task definition.
3. Check whether a more capable LLM with better structured output would eliminate the need for complex routing.

If you do need to migrate, the transition path is: extract the workflow's decision logic into an LLM prompt that outputs tool selection decisions, convert the workflow's fixed tool implementations to agent tools, replace the DAG runner with an agent loop.

### Agent → Workflow (over-engineering reversal)

This is the migration that teams should make more often. The triggers for this reversal:

- The agent's tool selection is consistently choosing the same 2–3 tools in the same general order
- Debugging incidents are taking 8+ hours and the team is spending more time on observability than features
- Costs have exceeded projections and most of the token spend is in repetitive reasoning preambles
- P99 latency is unacceptable for the use case

When you see these signals, audit the last 500 agent traces. If >80% of runs follow one of 3–5 common tool call sequences, you can express those sequences as workflow branches. The remaining 10–20% of unusual cases can be handled by a simplified fallback agent or escalated to humans.

The migration: identify the common tool call sequences from trace analysis, express each sequence as a workflow branch, add a classifier at the entry point to route to the appropriate branch, add a fallback agent for sequences not covered by the workflow.

![Before/after: over-engineered agent refactored to workflow with 85% latency improvement](/imgs/blogs/agent-vs-chain-vs-workflow-8.webp)

## Hybrid Patterns: Agents Calling Workflows Calling Chains

In production systems with complex requirements, you rarely choose a single pattern. The more common architecture is a hierarchy: an agent at the top handling open-ended intent classification and task routing, workflows in the middle handling structured execution logic, and chains at the leaves handling deterministic transformations.

This hierarchy is not just a theoretical nicety. It has concrete cost and performance benefits:

- **80% of work runs in chains**: Most of the actual processing — formatting, extraction, validation — is deterministic. Put this work in chains. Chains are cheap and fast.
- **15% of work runs in workflows**: Conditional routing, parallel execution, and retry logic live in workflows. Moderately more expensive, but bounded.
- **5% of work runs in the agent loop**: The high-level intent interpretation and tool-selection decisions live in the agent. This is where you pay the premium — but only for the decisions that genuinely require it.

The result: a system that can handle open-ended user requests while keeping average cost close to workflow levels, because most tokens go to chains and workflows rather than the agent loop.

```python
# Hybrid architecture example
async def handle_user_request(user_input: str) -> dict:
    """
    Hybrid: agent decides intent, workflow executes, chain formats.
    Average cost: ~$0.015/run (mostly workflow/chain, occasional agent turns).
    """
    # Agent layer: classify intent and select workflow
    # This is the only part that uses dynamic tool selection
    intent = await classify_intent_with_agent(user_input)
    
    # Workflow layer: execute based on classified intent
    if intent.type == "data_extraction":
        # Workflow handles extraction with retries and validation
        raw_result = await extraction_workflow(intent.parameters)
    elif intent.type == "report_generation":
        # Workflow handles multi-source data gathering
        raw_result = await report_workflow(intent.parameters)
    else:
        # Fallback: use agent for unclassified intents
        raw_result = await general_agent(user_input)
    
    # Chain layer: deterministic formatting
    # Always the same 3 steps regardless of what happened above
    formatted = await format_chain(raw_result, intent.output_format)
    return formatted
```

The pattern does not have to be agent → workflow → chain. You might also have:
- **Workflow → chain**: A workflow whose each step calls a separate specialized chain
- **Agent → chain**: An agent that directly calls deterministic chains as tools (no intermediate workflow needed)
- **Workflow → agent (as a step)**: A workflow that escalates to an agent for one specific step when the step requires dynamic tool selection

The principle is the same in all cases: use the cheapest pattern that can do the job at each layer of the hierarchy.

![Hybrid pattern: agent dispatches to workflow which calls chains](/imgs/blogs/agent-vs-chain-vs-workflow-9.webp)

## Case Studies

### Case Study 1: The Document Classification Agent That Should Have Been a Chain

**Team**: 4 engineers, fintech startup  
**Task**: Classify incoming documents (invoices, contracts, statements) into 12 categories  
**Original implementation**: GPT-4 agent with tools to read document metadata, query a category database, and run classification logic  
**Problem**: Each classification took 4–8 seconds and cost $0.06. At 5,000 documents/day, monthly cost was $9,000.  

**What the traces revealed**: The agent was calling the category-database tool on every single run, always with the same 12-category list, then making the same classification decision based on document metadata. There was no dynamic tool selection happening. The agent was just a slow, expensive function.

**The fix**: A 3-step chain. Step 1: extract document type indicator from filename + first 500 characters. Step 2: LLM classification prompt with the 12 categories hardcoded in the system prompt. Step 3: validate classification label.

**Result**: Latency fell from 5.2 seconds to 0.7 seconds. Monthly cost fell from $9,000 to $210. Engineering time to implement the chain: 1 day. The agent had taken 3 weeks to build.

**Lesson**: If the agent always calls the same tools in the same order, it is a chain in disguise. Audit your traces before assuming you need an agent.

---

### Case Study 2: The Email Triage Workflow That Should Have Stayed a Workflow

**Team**: 6 engineers, SaaS company  
**Task**: Triage incoming customer support emails — classify urgency, detect account issues, route to appropriate team  
**Original implementation**: A well-designed workflow with intent classification, account lookup, routing logic, and templated responses  
**Problem (apparent)**: As the company grew, the workflow's routing logic became complex — 40+ conditional branches, 8 different routing destinations, multiple retry patterns  

**The temptation**: "This is getting too complex. Should we replace the routing logic with an agent that dynamically decides how to route?"

**Why they didn't**: Before making the change, the team analyzed their trace data. 95% of all emails fell into 8 routing patterns. The remaining 5% were anomalies. The workflow handled all 8 patterns correctly.

**What they did instead**: Refactored the workflow to make the branching logic clearer, added a fallback path for the 5% edge cases that escalated to a human, and extracted some branches into sub-workflows for clarity.

**Result**: Routing accuracy improved from 87% to 94% (due to better prompt engineering during the refactor). Latency stayed at ~1.8 seconds. Cost stayed at ~$0.008/email. No agent complexity.

**Lesson**: Workflow complexity is manageable complexity. Agent complexity is often unbounded complexity. When your workflow feels "too complex," try refactoring the workflow first.

---

### Case Study 3: The Research Agent That Justified Itself

**Team**: 8 engineers, consulting firm  
**Task**: Given a client company name, generate a comprehensive competitive landscape report  
**Attempted implementation 1**: Workflow with fixed steps — web search → company database lookup → news search → synthesis. Problem: the same fixed searches missed crucial context for unusual companies. Reports were generic.  
**Attempted implementation 2**: Chain with a long hardcoded prompt for research. Problem: the chain had no way to follow up on interesting findings or dig into specific areas the initial research surfaced.  

**Why an agent was justified**: The research task genuinely required dynamic tool selection. For a pharmaceutical company, the agent needed to search patents, FDA databases, and clinical trial registries — sources irrelevant for, say, a retail company. The number of search iterations varied from 3 to 15 depending on what each search returned.

**Implementation**: An agent with tools for web search, company database lookup, patent search, FDA records, financial filings, and news aggregation. The agent ran for an average of 8–12 turns per report.

**Cost**: $0.80 per report (30–50× more than a workflow). 

**Was it worth it?** Yes. Report quality (rated by consultants) went from 3.1/5 (workflow) to 4.4/5 (agent). The reports justified a 40% price increase for the research service. The economics worked because the value per report was high and volume was low (~50 reports/day).

**Lesson**: Agents are justified when the value per task is high enough to absorb the cost, and when the task genuinely requires dynamic tool selection. At $0.80/report with high value output, the economics work. At $0.08/email classification, they do not.

---

### Case Study 4: The Autonomous Debugging Agent

**Team**: Platform engineering team, mid-size tech company  
**Task**: Automatically diagnose production incidents from PagerDuty alerts  
**Why an agent**: Debugging genuinely requires following the investigation wherever the evidence leads. The logs might point to a database query → which might reveal a slow index → which might trace to a recent migration → which might connect to a scheduled job. The path is discovered, not pre-specified.  

**Implementation**: Agent with tools for log querying, metrics dashboards, database query execution (read-only), deployment history lookup, and incident history search. Maximum 15 turns per incident.

**Results**: 60% of incidents were diagnosed without human intervention. Mean time to diagnosis fell from 45 minutes (human on-call) to 8 minutes (agent). The 40% of incidents the agent could not diagnose fully were forwarded to the on-call with a structured summary of findings, cutting human diagnosis time even in those cases.

**Debugging the debugger**: Ironically, the hardest part was debugging the agent itself. Three incidents were initially reported as "agent failed to diagnose" but were actually "agent correctly identified an unusual failure mode that the human reviewer did not initially recognize." The observability infrastructure — full conversation logs, tool call records, replay capability — was critical for understanding these cases.

**Lesson**: When the debugging path itself is genuinely unknown at design time, an agent is the correct abstraction. The key criterion: would a workflow require a branch for every possible investigation path? If yes, you have too many branches — use an agent.

![Debugging trace complexity by pattern — agent traces branch at every LLM decision and tool call](/imgs/blogs/agent-vs-chain-vs-workflow-7.webp)

---

### Case Study 5: The Code Generation Agent Refactored to Workflow

**Team**: Developer tools company  
**Task**: Generate boilerplate code for REST API endpoints from a schema definition  
**Original implementation**: GPT-4 agent that would generate code, run tests, observe failures, and iterate until tests passed  
**Problem**: Agent was costing $1.20 per schema and taking 45 seconds. More importantly, 30% of runs exceeded the 15-turn limit and produced incomplete code.  

**The trace analysis**: 80% of runs followed one of three patterns: (A) generate → all tests pass immediately, (B) generate → fix import errors → tests pass, (C) generate → fix import errors → fix type errors → tests pass. Only 20% required more complex iteration.

**The refactor**: A workflow with three branches corresponding to patterns A, B, and C, plus a fallback agent for the remaining 20%. The workflow handled each branch with targeted prompts for each specific failure type.

**Result**: Average cost fell from $1.20 to $0.18. Latency fell from 45 seconds to 8 seconds. Success rate improved from 70% to 91% (targeted prompts for known failure modes outperformed the general agent). The fallback agent for the 20% complex cases costs $0.60 but is invoked rarely.

**Lesson**: When trace analysis shows that most runs follow a small number of patterns, extract those patterns into workflow branches. Keep an agent only for genuinely novel cases.

---

### Case Study 6: The Content Moderation Chain That Scaled to 50M Requests

**Team**: Trust & safety, social platform  
**Task**: Classify user-generated content as safe/borderline/violating across 8 policy categories  
**Original design consideration**: Should this be a chain or a workflow?  

**Why chain won**: The classification task has the same shape for every input. Every piece of content goes through the same four steps: extract features → classify against each policy → compute aggregate severity → return structured decision. Different content categories do not need different processing steps — they need different prompt content in the same steps.

**Scale**: 50 million classifications per day. At chain cost (~$0.001/run) this is $50,000/day. At workflow cost (~$0.006/run) it would be $300,000/day. At agent cost ($0.05/run), it would be $2.5M/day — making the product economically unviable.

**The chain**: Extract content metadata → run policy classification (batched, 8 categories in single call) → compute severity score → return structured verdict. Four steps, always the same, always in the same order.

**Result**: Ran at 50M classifications/day for 18 months. No incidents where the chain abstraction was a limitation.

![A chain: deterministic pipeline with no branching, same path every run](/imgs/blogs/agent-vs-chain-vs-workflow-3.webp) Zero cases where an agent would have produced a meaningfully better result.

**Lesson**: At scale, the 40× cost difference between a chain and an agent is existential. If your task has a fixed computation shape, use a chain. The boring answer is usually right.

---

### Case Study 7: The Data Pipeline Agent That Should Have Been a Workflow

**Team**: Analytics platform  
**Task**: Transform and load data from 15 different source systems into a central warehouse  
**Original implementation**: GPT-4 agent that would examine each source system, determine the appropriate transformation logic, and generate ETL code  

**The problem**: The team had correctly identified that different source systems need different transformation logic — that seems like dynamic tool selection. But in practice, they had exactly 15 source systems with exactly 15 known transformation patterns. The agent was dynamically choosing between 15 options, all of which were fully specifiable at design time.

**The fix**: A workflow with 15 branches, one per source system. The "dynamic" part — choosing which transformation to apply — became a routing decision at the top of the workflow, made by a cheap classifier prompt rather than a full agent loop.

**Result**: Latency fell from 12 seconds to 1.8 seconds. Cost fell from $0.09/run to $0.007/run. The workflow ran 5,000 times/day, saving $410/day in token costs. Engineering time to refactor: 2 days.

**Lesson**: When you have N options and N is a small, fixed number, a workflow router is almost always better than an agent. "Dynamic" does not mean "needs an agent" — it means "the selection happens at runtime." A workflow does that too, more cheaply.

---

### Case Study 8: The Multi-step Planning Agent That Justified Its Complexity

**Team**: Legal tech company  
**Task**: Given a contract and a legal question, perform due diligence research and provide a risk assessment  
**Why an agent was right**: Legal due diligence requires following the investigation wherever it leads. A question about a change-of-control clause might require: find the relevant clause → identify which jurisdiction's law applies → search for case law in that jurisdiction → check for recent regulatory changes → assess risk given the client's specific situation. Each step depends on the findings of the previous step in ways that cannot be pre-specified.  

**Cost justification**: At $2.80 per due diligence request, the agent costs 3.5% of the $80 service fee. Legal firms traditionally spend 30–90 minutes on equivalent research. The agent reduces this to 4 minutes of review time, making the economics compelling.

**The investment**: 6 weeks to build correctly — including tool development (legal database connectors, case law search, jurisdiction lookup), observability, cost caps, and quality evaluation. An engineer reviews every agent output before it goes to clients.

**What a workflow could not do**: The case law search returns 50–200 cases. The agent must decide which cases are relevant to this specific contract, this specific question, and this specific client risk profile. That decision cannot be enumerated at design time — it requires reading the cases and making a judgment call about relevance.

**Lesson**: When the task genuinely requires reading information and making judgment calls about relevance that cannot be pre-specified, an agent earns its cost. The benchmark: would a workflow require hundreds of branches to approximate the same judgment? If yes, use an agent.

## Task-to-Pattern Reference

![8 real tasks mapped to their correct orchestration pattern](/imgs/blogs/agent-vs-chain-vs-workflow-10.webp)

## When to Reach for Each Pattern / When Not To

### Reach for a chain when:

- The task has a fixed computation shape (same N steps in the same order on every run)
- The input-to-output mapping is well-defined before writing any code
- Cost and latency matter at scale (chains are 40× cheaper than agents)
- Debugging simplicity is important (chains are trivially reproducible)
- You are building the first version of a new AI feature

Do not use a chain when:
- Different inputs need fundamentally different processing paths
- A step failure requires trying a different approach, not just retrying the same step
- The task requires discovering information sources during execution

### Reach for a workflow when:

- The task has conditional logic but the full decision tree is specifiable before writing code
- You can draw the complete DAG in an afternoon
- The task needs parallel execution, explicit state, or bounded retries
- You have outgrown a chain but your trace analysis shows consistent patterns

Do not use a workflow when:
- The number of branches exceeds ~30 (you are encoding agent behavior into a workflow)
- Branch conditions depend on information discovered during execution
- The task is genuinely open-ended (researching an unknown domain, debugging an unknown failure)

### Reach for an agent when:

- Tool selection genuinely cannot be predetermined — it depends on intermediate findings
- The number of steps is unbounded (the task ends when the agent decides it has enough information)
- Self-correction requires trying fundamentally different approaches, not just retrying
- The value per task justifies the 40× cost premium and 10× engineering investment
- You have run trace analysis on a workflow and confirmed the patterns are too varied to enumerate

Do not use an agent when:
- Your workflow's branching complexity has made you frustrated — complexity is a workflow problem, not an agent solution
- The agent always calls the same tools in the same general order (audit 100 traces first)
- Cost matters at scale (agents at 10M requests/month will exceed most product budgets)
- You cannot invest in proper observability, replay capability, and cost controls
- You are building the first version of anything (start with a chain, evolve from there)

The single most useful heuristic: **audit 100 traces before building anything new.** If you are considering upgrading a chain to a workflow, audit the chain's inputs and outputs. If you are considering upgrading a workflow to an agent, audit the workflow's execution paths. The data tells you which pattern is actually needed.

---

For more on building effective agents, see [Building Effective Agents: A Hands-On Guide](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide) and [Agentic Design Patterns and Case Studies](/blog/machine-learning/ai-agent/agentic-design-patterns-and-case-studies). For the internal mechanics of the agent loop, see [Agent Loop Anatomy](/blog/machine-learning/ai-agent/agent-loop-anatomy). For reasoning patterns that sit inside an agent, see the upcoming [ReAct Pattern Deep Dive](/blog/machine-learning/ai-agent/react-pattern-deep-dive).

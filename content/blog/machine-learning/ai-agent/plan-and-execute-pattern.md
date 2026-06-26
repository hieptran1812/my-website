---
title: "Plan-and-Execute: Two-Phase Agents That Think Before They Act"
date: "2026-06-27"
description: "How two-phase agents separate planning from execution to handle long-horizon tasks — with replanning triggers, executor design, and production tradeoffs."
tags: ["ai-agents", "reasoning", "planning", "llm", "machine-learning", "nlp", "system-design", "production-ml"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 35
---

There is a failure mode that every engineer building LLM agents eventually runs into. You wire up a ReAct loop — think, act, observe, repeat — and it works beautifully on toy problems. Then you hand it a real task: "research our top five competitors, summarize their pricing models, identify our gaps, and draft a response strategy." The agent starts promisingly, makes three tool calls, then either spirals into repetitive tool use, loses context halfway through, forgets what it was supposed to be doing, or — most insidiously — returns a confident-looking but clearly incomplete answer without acknowledging any of the steps it skipped.

The problem is architectural. A pure ReAct loop makes local decisions at each step: "given what I know right now, what should I do next?" There is no global plan, no record of what still needs to happen, no mechanism to recognize when accumulated decisions have drifted far enough from the original goal that the whole approach needs reconsidering. For short tasks (lookup a fact, run a calculation, answer a question), this reactive style is fast and cheap. For long-horizon tasks — the ones that actually make agents useful — it breaks down reliably.

The plan-and-execute pattern fixes this with a structural change: separate the work of *planning* from the work of *executing*. A planning LLM (the planner) takes the user's goal and produces a structured representation of the work — steps, dependencies, contingency paths, and success criteria. An executing LLM (the executor) then runs through that plan step by step, calling tools, handling observations, and reporting results back. If something goes wrong, a replanning mechanism can re-invoke the planner with updated context, producing a revised plan from the checkpoint rather than restarting from scratch.

![Plan-and-Execute Architecture: planner emits a structured plan; executor runs each step via ReAct; replanning feedback loop on failure](/imgs/blogs/plan-and-execute-pattern-1.webp)

The diagram above is the mental model for everything in this post. The planner and the executor are separate LLM invocations — potentially the same model with different prompts, or deliberately different models sized for their roles. The plan is a first-class data structure, not a chain of thought buried in a scratchpad. And the replanning arc is explicit and gated, not an infinite retry loop.

We are going to go deep on every component: what makes a plan good, how the executor works internally, when replanning is warranted vs wasteful, how to represent plans as structured data, and the real cost and latency numbers you need to make the right architectural call. We will close with six case studies from production deployments, and a clear "when to use / when not to" rubric.

## 1. The Problem With Reactive Agents on Long-Horizon Tasks

To understand why plan-and-execute exists, we need to be precise about what "long-horizon" means and why it breaks reactive loops.

A task is **long-horizon** when: (a) it requires 5 or more non-trivial steps, (b) later steps depend on the outputs of earlier ones in ways that cannot be inferred locally, (c) some steps can be parallelized, or (d) partial failure on one step invalidates the premises of subsequent steps. The "research competitors and draft a response strategy" example from the intro satisfies all four criteria.

### Where pure ReAct falls apart

**Context window pressure.** A ReAct scratchpad grows with every think/act/observe cycle. At step 8 of a 15-step task, the scratchpad might be 6,000 tokens long. The full conversation history — including all tool call/response pairs — is often another 10,000 tokens. On a 128K context model this is technically fine, but the model's attention degrades on content buried deep in the context, and many production deployments still run on 32K or 16K context models for cost reasons.

**No global state for dependency tracking.** Step 7 might need "the competitor pricing table from step 2." In a pure ReAct loop, "step 2" is just some text in the scratchpad. There is no structured handle to reference it, no guarantee the model will correctly identify the right data, and no mechanism to verify that step 2 actually completed successfully before step 7 runs.

**No partial failure recovery.** If step 9 fails (API timeout, tool error, rate limit), a ReAct loop can retry that step — but it has no way to know whether the retry changes the implications for steps 10 through 15. It has no concept of "step 10 assumes step 9 returned a non-empty result; if step 9 failed and returned a stub, step 10 should skip or use a fallback." A plan makes these dependencies explicit.

**No parallelization.** Steps 1 and 2 in the research task (query web sources, load local docs) are independent. A ReAct loop runs them sequentially because it makes one tool call per cycle. A plan that explicitly marks these steps as having no dependency relationship between them can run them concurrently, cutting task latency by up to half on the most parallel-friendly workflows.

**No accountability.** When a user asks "why did you conclude X?" the model cannot point to a structured audit trail. It can only produce a post-hoc rationalization from the scratchpad. Plans are auditable: each step has a defined objective, its output is stored, and the causal chain from goal to result is explicit.

### The plan-and-execute alternative

The pattern's core insight is simple: for long-horizon tasks, the cost of one LLM call to produce a structured plan is almost always worth paying. A well-formed plan for a 10-step research task might cost 1,000–2,000 tokens (planner input) + 800–1,500 tokens (plan output). At GPT-4 prices (~$0.01–0.03 per 1K tokens), that is $0.01–0.06 per task — a rounding error compared to the $0.50–2.00 cost of the full execution trace. But it buys: global coherence, parallel execution, checkpointability, partial failure recovery, and a deterministic audit trail.

The tradeoff is not free: the plan introduces a 3–8 second latency penalty before the first tool call, and it adds a planner prompt engineering surface that can fail in its own specific ways (plans that are too vague, plans that overfit to the happy path, plans with circular dependencies). We will address each of these tradeoffs in detail.

## 2. Plan-and-Execute Architecture: Planner LLM + Executor LLM

The architecture has three components: the planner, the plan data structure, and the executor. The controller (your orchestration code) ties them together and owns the replanning trigger logic.

### The planner

The planner is an LLM invocation whose job is to decompose a user goal into a structured plan. Its prompt includes:

1. **The user's goal** — stated explicitly, not embedded in conversation history.
2. **Available tool descriptions** — so the planner can assign the right tools to each step.
3. **A plan schema** — either a JSON schema or natural language description of what a valid plan looks like.
4. **Optionally: a few-shot example plan** — especially useful for the first few deployments when you are still calibrating planner behavior.

The planner does not call any tools during planning. It does purely symbolic decomposition. This is intentional: the planning call is cheap (typically 1K–3K tokens), fast (3–8 seconds for frontier models), and if the plan is wrong, you want to catch that before you have paid for 15 tool calls.

A key design choice is the **planner's context window budget**. For simple tasks, you can include full tool documentation. For complex tasks with 50+ available tools, you need a tool-selection pre-pass — either a retrieval step that surfaces the top-K relevant tools, or a two-stage planning process where the planner first identifies which tools it needs, then builds the full plan with just those tool specs.

### The plan data structure

We will cover plan representation in detail in section 7, but the essentials: a plan is a list of step objects with at minimum `id`, `description`, `tool`, `inputs`, and `deps` (dependency list). Each step's `deps` field is a list of step IDs that must complete before this step can run. This makes the dependency graph explicit and traversable in O(n) time.

A production plan also includes: `contingency` (what to do if this step fails), `success_criteria` (how to know this step completed correctly), and `timeout_s` (maximum time to wait for this step before triggering the failure path). These fields are optional at the start but you will add them the first time an agent silently fails on a production task.

### The executor

The executor is a separate LLM invocation (or a series of invocations) that runs each step in the plan. For each step, the executor behaves like a lightweight ReAct agent: it receives the step description, the relevant tool specifications, and the outputs of all dependency steps, then runs a think/act/observe loop until it either produces a step result or exhausts its retry budget.

The executor is intentionally narrower-scoped than the planner. It does not know about the full plan — it only knows the current step and its dependencies. This is a conscious tradeoff: it reduces the executor's context window pressure, but it means the executor cannot reason about whether its current step's result makes sense in the context of the overall goal. That cross-step reasoning is the planner's responsibility, exercised during replanning.

### The controller

The controller is your orchestration code (Python, TypeScript, whatever your stack uses). It is responsible for: loading the plan, checking which steps are ready to run (all deps complete), dispatching them to the executor in parallel where possible, collecting results, writing checkpoints, and deciding whether a step failure should trigger replanning or just a retry.

The controller is where the real engineering lives. A naive controller runs steps sequentially; a production controller maintains a ready queue, tracks in-flight steps, handles timeouts, and manages the checkpoint store. We will dig into checkpoint design in section 9.

## 3. Phase 1 — Planning: What Makes a Good Plan

A bad plan is worse than no plan. If the planner emits a vague plan with no dependency structure, no contingencies, and no success criteria, the executor will run through it blindly, fail on step 4, and leave the user with partial output and no explanation. Good plan quality is the single most important factor in plan-and-execute performance.

![Anatomy of a Good Plan: five nested layers from goal to success criteria](/imgs/blogs/plan-and-execute-pattern-2.webp)

### The five layers of a production plan

**Goal layer.** The goal is a declarative statement of what must be true when the task is complete. It is not a list of actions; it is an invariant. For the research task: "produce a document containing competitor pricing for A, B, C, D, and E, with a gap analysis and a response strategy, where all claims are cited to sources retrieved in this session." This statement is verifiable — at the end, you can check each criterion against the output.

**Step layer.** Steps are atomic units of work, each of which: (a) has a single well-defined output, (b) maps to a specific tool invocation or sequence of invocations, and (c) is independently verifiable. "Research competitors" is not a step — it is a goal. "Call the web_search tool with query 'Competitor A pricing 2025' and return the top 3 results" is a step. Steps should be small enough that if they fail, the failure mode is diagnosable. A step that takes 8 tool calls to complete is actually a sub-plan; break it up.

**Dependency layer.** The dependency DAG captures which steps require which other steps' outputs as inputs. This is the layer most often omitted in naive plan-and-execute implementations, and its absence causes subtle bugs: the executor runs step 5 before step 3 completes, gets a None input, produces garbage output, and the task finishes with a hallucinated synthesis because the model filled in the missing data.

**Contingency layer.** For each step, the contingency specifies: "if this step fails after N retries, do X." X can be: try an alternative tool, use a cached/default value and note the limitation in the output, skip this step and note the gap, or trigger replanning. Contingencies are what separate fragile toy agents from production agents. You will not know all the failure modes upfront — add them as you discover them in production.

**Success-criteria layer.** Each step should have a verifiable exit condition. For a web search step: "result must be non-empty and contain at least one URL". For a summarization step: "summary must be 200–400 words and include at least one specific claim from the source document." The executor checks these criteria after the step completes; if they fail, the step is retried or the contingency is triggered.

### What good plans look like in practice

Good plans are **small**. A 20-step plan is almost always better rewritten as 10 steps with cleaner dependency structure. More steps means more planner context, more executor invocations, more places to fail.

Good plans are **concrete**. Every step names a specific tool, specifies the exact input it needs (referencing previous step outputs by ID), and states a specific expected output format.

Good plans **front-load information gathering**. The first N steps should collect all the raw data the rest of the plan needs. Mixing data collection with analysis in the same step is a common source of plan brittleness — if the data collection part fails, you lose the analysis context too.

Good plans **make the critical path explicit**. The critical path is the longest sequence of sequential dependencies. If you can identify it, you can tell the executor to prioritize those steps, and you can communicate realistic ETAs to the user.

### Prompt patterns for better planner outputs

The single most effective prompt change for plan quality is requiring the planner to **output a dependency check**: after generating the plan, instruct it to verify that each step's `deps` list is complete, and that no step uses an output that has not been produced by a previous step. This self-verification step catches 70–80% of dependency errors before execution starts.

The second most effective change is providing a **failure-modes vocabulary**: give the planner a list of the tools' known failure modes (rate limits, empty results, timeouts) and instruct it to specify contingencies that reference these modes. Without this vocabulary, planners write generic contingencies ("if it fails, retry") that are useless at execution time.

```python
PLANNER_SYSTEM_PROMPT = """
You are a task planner for an AI agent. Your job is to decompose a user goal
into a structured execution plan.

TOOL FAILURE MODES (use these when writing contingencies):
- web_search: may hit rate limits (429); fallback: use cached_search
- code_execution: may time out (>30s for complex code); fallback: simplify query
- document_reader: may return empty for PDF files >50MB; fallback: use summary endpoint

OUTPUT FORMAT: JSON with the schema below. Validate your plan before outputting:
1. Every step's `deps` must reference only prior step IDs.
2. Every step must have a `tool` field matching an available tool.
3. Every step must have a `contingency` field.

{
  "goal": "...",
  "steps": [
    {
      "id": "s1",
      "description": "...",
      "tool": "web_search",
      "inputs": { "query": "..." },
      "deps": [],
      "contingency": "if rate_limit: retry with cached_search",
      "success_criteria": "non-empty results list with ≥1 URL"
    }
  ]
}
"""
```

## 4. Phase 2 — Execution: The Executor as a Lightweight ReAct Agent

The executor runs each step in the plan. Its inner loop is a standard ReAct cycle: it thinks about how to accomplish the step given its inputs, calls the appropriate tool, observes the result, and either reports success or handles the failure according to the step's contingency.

The key design insight is that the executor's scope is **bounded** by the step definition. Unlike a free-form ReAct agent that can call arbitrary tools in any order, the executor operates within constraints specified by the plan: it knows which tool to use, what inputs to provide, and what a successful output looks like. This constraint is what makes the executor both more reliable and cheaper than a general-purpose agent.

### Executor prompt design

The executor receives, per step:

```python
EXECUTOR_STEP_PROMPT = """
You are executing step {step_id} of a larger plan.

STEP OBJECTIVE: {step_description}

AVAILABLE TOOL FOR THIS STEP: {tool_name}
Tool documentation: {tool_doc}

DEPENDENCY OUTPUTS (inputs from previous steps):
{dep_outputs_json}

SUCCESS CRITERIA: {success_criteria}
CONTINGENCY (if this step fails after retries): {contingency}

Execute this step. Think through what you need to do, call the tool, observe
the result, and verify the success criteria. If criteria are not met after
{max_retries} retries, report failure with the contingency action.
"""
```

Notice what the executor does NOT receive: the full plan, the other steps' outputs, or the original user goal. This is by design. The executor is a specialist — it knows about one step. This keeps its context window small (typically 2K–4K tokens per step invocation vs 8K–16K for a general-purpose agent) and keeps it focused.

### The executor's inner loop

```python
async def execute_step(step: PlanStep, dep_outputs: dict, llm, max_retries: int = 2) -> StepResult:
    """Run a single plan step with retry logic."""
    prompt = build_executor_prompt(step, dep_outputs)
    
    for attempt in range(max_retries + 1):
        response = await llm.complete(prompt)
        
        # Extract tool call from response
        tool_call = parse_tool_call(response)
        if tool_call is None:
            # No tool call — executor gave up or hallucinated a response
            if attempt < max_retries:
                prompt = add_retry_instruction(prompt, "No tool call found. Please call the tool.")
                continue
            return StepResult(status="failure", reason="executor_no_tool_call")
        
        # Execute the tool
        try:
            tool_output = await tools[tool_call.name](**tool_call.args)
        except ToolError as e:
            if attempt < max_retries:
                prompt = add_retry_instruction(prompt, f"Tool returned error: {e}")
                continue
            return StepResult(
                status="failure", 
                reason=f"tool_error: {e}",
                contingency=step.contingency
            )
        
        # Check success criteria
        criteria_met = check_success_criteria(tool_output, step.success_criteria, llm)
        if criteria_met:
            return StepResult(status="success", output=tool_output)
        
        if attempt < max_retries:
            prompt = add_retry_instruction(
                prompt, 
                f"Output did not meet success criteria: {step.success_criteria}"
            )
    
    return StepResult(status="failure", reason="max_retries_exhausted", contingency=step.contingency)
```

### Parallelizing independent steps

The controller identifies all steps whose dependency lists are fully satisfied (all deps have `status=success`) and dispatches them in parallel. For async Python with asyncio:

```python
async def run_plan(plan: Plan, checkpoint_store, llm) -> PlanResult:
    step_results = load_checkpoint(checkpoint_store, plan.id)
    
    while True:
        # Find steps ready to execute (deps done, not yet started)
        ready = [
            s for s in plan.steps
            if s.status == "pending"
            and all(step_results.get(dep, {}).get("status") == "success" for dep in s.deps)
        ]
        
        if not ready:
            # Check if we're done or stuck
            pending = [s for s in plan.steps if s.status == "pending"]
            if not pending:
                return PlanResult.success(aggregate_outputs(step_results))
            else:
                # Some steps pending but none ready — a failure has blocked them
                return PlanResult.partial_failure(step_results)
        
        # Dispatch ready steps in parallel
        tasks = [
            execute_step(step, {dep: step_results[dep].output for dep in step.deps}, llm)
            for step in ready
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for step, result in zip(ready, results):
            step_results[step.id] = result
            step.status = result.status
            await checkpoint_store.write(plan.id, step.id, result)
        
        # Check if any failures need replanning
        failures = [s for s in ready if step_results[s.id].status == "failure"]
        if failures:
            replan_needed = any(requires_replan(f, step_results) for f in failures)
            if replan_needed:
                return PlanResult.replan_requested(step_results, failures)
```

The `asyncio.gather` call is the parallelization mechanism. In the research task, steps 1 and 2 (web search + local doc load) run concurrently. Steps 4 and 5 (verify claim A + verify claim B) also run concurrently after step 3 completes. The wall-clock time drops from 6× step duration to roughly 3× on this task shape.

## 5. Replanning: When to Abort and Replan vs Push Through

Replanning is expensive. Invoking the planner again costs the same 3–8 seconds and 1K–3K tokens as the initial planning call. More importantly, replanning introduces the risk of a plan that is inconsistent with already-completed steps — the replanner must be given the checkpoint state so it can build a plan that only covers the remaining work.

![Replanning Decision Tree: when to replan vs retry vs continue](/imgs/blogs/plan-and-execute-pattern-7.webp)

### The replanning decision function

Not every step failure warrants replanning. The decision tree above captures the logic; here is the corresponding Python:

```python
def requires_replan(failed_step: PlanStep, step_results: dict) -> bool:
    """
    Returns True if a step failure invalidates the current plan,
    requiring a replan rather than just a retry or contingency execution.
    """
    # Executor already exhausted retries + contingency
    if failed_step.status != "failure":
        return False
    
    # Contingency handles this failure type without plan-level changes
    if failed_step.contingency_type in ("retry", "use_fallback_tool", "use_cached_value"):
        return False
    
    # Failure invalidates assumptions of downstream steps
    downstream_steps = get_downstream_steps(failed_step.id, plan)
    if any(step.assumes_success_of(failed_step.id) for step in downstream_steps):
        return True
    
    # Accumulated failures exceed 30% of steps — plan is probably wrong
    total_failures = sum(1 for r in step_results.values() if r.status == "failure")
    if total_failures / len(plan.steps) > 0.3:
        return True
    
    # Critical-path step failed — remaining plan cannot complete
    if failed_step.id in plan.critical_path:
        return True
    
    return False
```

### The replanning invocation

When replanning is triggered, the planner receives:

1. The original goal
2. The completed steps with their outputs (what we know)
3. The failed step with the error reason (what went wrong)
4. The remaining steps that have not yet run (what still needs to happen)
5. An explicit instruction: "produce a revised plan for only the incomplete work, using the completed step outputs as inputs"

```python
REPLAN_PROMPT = """
You are replanning a partially-completed agent task.

ORIGINAL GOAL: {goal}

COMPLETED STEPS (do not re-do these):
{completed_steps_json}

FAILED STEP: {failed_step_id} — {failure_reason}

REMAINING STEPS (original plan, now potentially invalid):
{remaining_steps_json}

Produce a revised plan for only the remaining work. The revised plan:
- Must start from the completed steps' outputs (treat them as given inputs).
- Must address the failure that caused replanning.
- Must cover all aspects of the original goal not yet completed.
- Should reuse completed step outputs where possible.
"""
```

### Replanning pitfalls

**Replanning loops.** If the replanner generates a plan that encounters the same failure, you end up in a replanning loop: plan → execute → fail → replan → execute → fail → replan. Set a maximum replan count (2–3 is typical) and surface the failure to the user if exceeded.

**Inconsistent replans.** The replanner sometimes generates steps that contradict completed work. For example, if step 3 already extracted claims from competitor A's website, a replan might include "search for competitor A's pricing" again. Guard against this with a validation step: after replanning, verify that no new step's inputs contradict the outputs of completed steps.

**Context compression.** The replan prompt includes full completed-step outputs, which can be large. Implement a summarization pass for completed-step outputs over 500 tokens before including them in the replan prompt. The replanner needs to know what was found, not the full raw tool output.

### When replanning is almost always wrong

You should not trigger replanning for:

- **Transient tool failures** (network timeout, 429 rate limit) — these are retry cases, not replan cases.
- **Minor output quality issues** — if a web search returns 3 results instead of 10, retry with a different query; do not replan the whole pipeline.
- **Single non-critical-path failures** — if step 4 fails but step 4 only contributes one optional input to the final synthesis, skip it and note the gap.

The practical heuristic: replan only when a failure changes the *problem structure* (what tools are needed, what steps make sense), not when it changes the *input values* (this query returned less than expected).

## 6. Dependency Graphs in Plans: Sequential vs Parallel Step Execution

The dependency graph is the most underappreciated part of plan design. Getting it right is what transforms a plan from a fancy bulleted list into something that actually enables parallel execution and partial failure recovery.

![Dependency Graph for a 6-Step Research Plan: parallel and sequential steps](/imgs/blogs/plan-and-execute-pattern-3.webp)

### Reading the dependency graph

In the figure above, steps 1 and 2 are **independent** (no shared dependencies) — the controller can dispatch both simultaneously. Step 3 is a **join** node — it cannot run until both steps 1 and 2 complete. Steps 4 and 5 are again independent, both depending only on step 3. Step 6 is the final **join** — it needs all of 3, 4, and 5.

The **critical path** is the longest chain of sequential dependencies: step 1 (or 2, whichever takes longer) → step 3 → step 4 (or 5) → step 6. With step 1 at 30 seconds, step 3 at 10 seconds, step 4 at 20 seconds, and step 6 at 15 seconds, the critical path takes 75 seconds. A naive sequential execution of all six steps would take 30+5+10+20+20+15 = 100 seconds. Parallel execution brings it to 75 seconds — a 25% reduction on this relatively serial task. On more parallel-friendly tasks (e.g., a 10-step plan where steps 1–8 are all independent data gathering and only step 9 synthesizes), the reduction can be 80%.

### Building the dependency graph correctly

**Rule 1: only declare strict dependencies.** If step 5 would be *better* with step 2's output but could function without it, do not add step 2 to step 5's deps. Spurious dependencies serialize execution unnecessarily.

**Rule 2: watch for hidden dependencies in tool outputs.** If step 3 returns a URL, and step 4 needs to fetch that URL, then step 4 depends on step 3. But if step 4 is "verify claim B from competitor B's website," and claim B is not on the URL that step 3 found, step 4 has an implicit dependency on step 2 (which found competitor B's URL) not just step 3. Making all dependencies explicit is harder than it looks.

**Rule 3: model contingency paths as optional dependencies.** If step 4 fails and the contingency is "use step 2's output as a fallback input," then step 4 has an optional dependency on step 2. Represent this in the dependency graph so the controller can make the right call: if step 2 has not yet run and step 4 hits its contingency, the controller needs to either run step 2 first or report that the contingency is unavailable.

**Rule 4: maximum fan-in is about 3–4.** A step with 8 dependencies is usually a design smell — it suggests either the step is doing too much, or the plan is poorly structured. Steps with many inputs are expensive to replan around (because all deps must complete before the step can start) and fragile (because any one failure blocks the step).

### Topological sort for execution ordering

The controller must topologically sort the dependency graph to determine a valid execution order. Python's `graphlib` module does this in O(V + E):

```python
from graphlib import TopologicalSorter
import asyncio

def build_execution_order(plan: Plan) -> list[list[str]]:
    """Return batches of step IDs that can run in parallel."""
    ts = TopologicalSorter({
        step.id: set(step.deps)
        for step in plan.steps
    })
    ts.prepare()
    batches = []
    while ts.is_active():
        batch = list(ts.get_ready())
        batches.append(batch)
        for step_id in batch:
            ts.done(step_id)
    return batches

# Output for the 6-step research plan:
# [['s1', 's2'], ['s3'], ['s4', 's5'], ['s6']]
# Batch 1: s1 and s2 run in parallel
# Batch 2: s3 runs after both s1, s2 complete
# etc.
```

## 7. Plan Representation: Natural Language vs Structured JSON/YAML

The choice of plan representation format affects the planner's accuracy, the executor's reliability, the controller's implementation complexity, and your ability to debug failures.

![Natural Language Plan vs Structured JSON Plan: ambiguous prose vs explicit DAG](/imgs/blogs/plan-and-execute-pattern-5.webp)

### Natural language plans: fast to generate, hard to execute

A natural language plan looks like: "First, search for competitor A's pricing. Then summarize. Also look at competitor B. Write a draft once you have data." These plans are easy for the planner to generate (no schema constraints) and readable, but they have fatal flaws for programmatic execution:

- **Step boundaries are ambiguous.** "First... Then... Also..." is not parseable into discrete steps with clear start and end conditions.
- **Dependencies are implicit.** "Once you have data" — which data? From which steps? The executor has to guess.
- **Success criteria are absent.** When is a step done? The executor has no way to know when to stop retrying.
- **No parallel execution signal.** "First... Then... Also..." suggests sequential execution even when parallelization is possible.

NL plans work in demos where a single powerful model handles both planning and execution and can reason about the plan as prose. They fail in production multi-step systems where the plan is a contract between the planner, the controller, and the executor.

### Structured JSON plans: explicit, executable, debuggable

A JSON plan represents the same task as a machine-readable DAG:

```json
{
  "goal": "Produce competitor pricing analysis with gap identification",
  "steps": [
    {
      "id": "s1",
      "description": "Search for Competitor A's pricing page",
      "tool": "web_search",
      "inputs": { "query": "Competitor A pricing 2025 site:competitorA.com" },
      "deps": [],
      "contingency": "if rate_limit: use cached_search with same query",
      "success_criteria": "non-empty results with at least 1 URL"
    },
    {
      "id": "s2",
      "description": "Search for Competitor B's pricing page",
      "tool": "web_search",
      "inputs": { "query": "Competitor B pricing 2025 site:competitorB.com" },
      "deps": [],
      "contingency": "if rate_limit: use cached_search with same query",
      "success_criteria": "non-empty results with at least 1 URL"
    },
    {
      "id": "s3",
      "description": "Extract pricing tiers from results",
      "tool": "document_reader",
      "inputs": {
        "urls": ["{{s1.output.urls[0]}}", "{{s2.output.urls[0]}}"]
      },
      "deps": ["s1", "s2"],
      "contingency": "if empty_result: use web_search output directly",
      "success_criteria": "extracted text length > 200 chars"
    }
  ]
}
```

Notice the `{{s1.output.urls[0]}}` template expression in step 3's inputs. This is the key pattern for connecting steps: outputs are referenced by step ID and field path, making the dependency explicit and making it impossible for the executor to use stale or invented data.

### YAML plans: a readable middle ground

For human-authored plans or plans that will be reviewed and edited, YAML offers JSON's structure with better readability:

```yaml
goal: "Produce competitor pricing analysis with gap identification"
steps:
  - id: s1
    description: "Search for Competitor A pricing"
    tool: web_search
    inputs:
      query: "Competitor A pricing 2025"
    deps: []
    contingency: "rate_limit: use cached_search"
    success_criteria: "non-empty results with ≥1 URL"
  
  - id: s3
    description: "Extract pricing tiers"
    tool: document_reader
    inputs:
      urls: ["{{s1.output.urls[0]}}", "{{s2.output.urls[0]}}"]
    deps: [s1, s2]
```

YAML plans are particularly good for agentic systems where a human operator might need to inspect and modify plans mid-execution, or where plans are stored in configuration repositories alongside application code.

### Schema enforcement

Whether you use JSON or YAML, enforce the plan schema programmatically before execution starts. Pydantic is the standard tool for Python:

```python
from pydantic import BaseModel, field_validator
from typing import Optional

class PlanStep(BaseModel):
    id: str
    description: str
    tool: str
    inputs: dict
    deps: list[str] = []
    contingency: str
    success_criteria: str
    timeout_s: int = 60
    
    @field_validator('deps')
    @classmethod
    def deps_must_be_prior_steps(cls, v, info):
        # Validate that deps reference only prior steps (in context of the full plan)
        return v

class Plan(BaseModel):
    goal: str
    steps: list[PlanStep]
    
    @field_validator('steps')
    @classmethod
    def validate_dependency_graph(cls, v):
        step_ids = {s.id for s in v}
        for step in v:
            for dep in step.deps:
                if dep not in step_ids:
                    raise ValueError(f"Step {step.id} depends on unknown step {dep}")
        # Check for cycles
        if has_cycle(v):
            raise ValueError("Plan contains a dependency cycle")
        return v
```

## 8. Planner-Executor Split: Same Model vs Different, Specialized Prompts

One of the most consequential architectural decisions in plan-and-execute is whether to use the same LLM for both planning and execution, or to use different models sized for each role.

![Same-Model vs Split-Model Planner/Executor: cost, quality, and latency tradeoffs](/imgs/blogs/plan-and-execute-pattern-9.webp)

### Same-model approach: simplicity and coherence

Using a single frontier model (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) for both planning and execution has clear benefits:

- **No model mismatch.** The planner and executor have the same capabilities, the same knowledge cutoff, and the same tendency to follow certain instruction patterns. A plan authored by GPT-4o will be understood correctly by a GPT-4o executor.
- **Single prompt engineering surface.** You debug one model's behaviors, not two.
- **Coherent tool understanding.** If the planner assigned tool `web_search` to a step, the executor knows exactly how `web_search` works.

The cost is that every executor invocation uses the frontier model's pricing. If a plan has 10 steps and each executor invocation is 3K input tokens + 800 output tokens, at GPT-4o pricing (~$0.005 input / $0.015 output per 1K tokens), each step costs roughly $0.027. Ten steps = $0.27. For 1,000 tasks/day, that is $270/day in execution cost alone.

### Split-model approach: cost savings with quality tradeoffs

The split approach uses a frontier model only for planning and a cheaper, faster model (GPT-4o-mini, Claude 3 Haiku, Gemini Flash) for execution. Since most execution steps are relatively simple (follow the step description, call this tool, verify the result), the cheaper model handles them well.

Quantitative comparison from production deployments:

| Task type | Same model (GPT-4o) | Split (GPT-4o plan + Haiku exec) | Quality delta |
|---|---|---|---|
| Research report, 8 steps | $0.38/task | $0.12/task | ≈5% drop |
| Code review, 5 steps | $0.21/task | $0.08/task | ≈8% drop |
| Browser automation, 12 steps | $0.54/task | $0.15/task | ≈3% drop |
| Complex analysis, 6 steps | $0.29/task | $0.11/task | ≈15% drop |

The "complex analysis" row shows where the split approach breaks down: when executor steps require genuine reasoning (not just tool dispatch), a weaker model makes more errors, which cascades into replanning overhead that erases the cost savings.

The practical rule: use split models when executor steps are primarily **tool dispatch + format parsing** (the model is essentially a structured JSON extractor). Use same-model when executor steps require **reasoning or judgment** (the model needs to decide how to interpret ambiguous tool output, what to do when the tool returns unexpected data, or how to synthesize across multiple observations).

### Specialized executor prompts

Regardless of the model choice, the executor's prompt should be much more constrained than the planner's. A planner prompt that says "think about the best way to accomplish this goal" is correct for planning. For execution, that prompt is an invitation to hallucinate alternative approaches and ignore the plan.

An effective executor prompt is directive and narrow:

```python
EXECUTOR_PROMPT = """
You are executing exactly one step of a pre-planned task. Do not deviate from the step.

Step: {step_description}
Tool: {tool_name}
Required inputs: {inputs_json}
Expected output format: {output_format}
Success criteria: {success_criteria}

Call the tool with the provided inputs. Do not call any other tools.
Do not skip the tool call and provide a direct answer.
Report the raw tool output; the planner will handle interpretation.
"""
```

The "do not call any other tools" and "do not skip the tool call" instructions address the two most common executor failure modes: scope creep (the model decides to call an additional tool for extra context) and hallucination (the model answers from training data without calling the tool).

## 9. Long-Horizon Task Handling: Checkpointing, Resumability, Partial Failure Recovery

Long-horizon tasks run for minutes. In a distributed system, anything that runs for minutes will sometimes fail partway through — due to network timeouts, OOM kills, process restarts, API rate limits, or the user's browser tab closing in a web-based agent. Checkpointing is how you turn these catastrophic failures into recoverable setbacks.

![Checkpoint and Resume Flow for Long-Running Plans](/imgs/blogs/plan-and-execute-pattern-8.webp)

### Checkpoint design

A checkpoint contains everything needed to resume a plan from a specific step:

```python
@dataclasses.dataclass
class PlanCheckpoint:
    plan_id: str
    plan_hash: str  # hash of the plan JSON; detect plan mutations
    completed_steps: dict[str, StepResult]  # step_id → result
    in_progress_steps: set[str]  # steps that were running when checkpoint was written
    timestamp: float
    
    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))
    
    @classmethod
    def from_json(cls, s: str) -> 'PlanCheckpoint':
        return cls(**json.loads(s))
```

Write a checkpoint after each step completes. Do not write checkpoints during step execution (you don't know if the step succeeded yet). The checkpoint write should be atomic — write to a temp key, then rename — to prevent partial checkpoint corruption.

### Checkpoint storage backends

For single-process agents: SQLite is the right choice. Fast, no external dependencies, survives process restart.

For distributed agents: Redis is common (fast, TTL support, atomic operations). For durability across machine failures: a key-value store with replication (DynamoDB, Firestore, Postgres with JSONB).

```python
class SqliteCheckpointStore:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                plan_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
    
    async def write(self, plan_id: str, step_id: str, result: StepResult):
        checkpoint = await self.load(plan_id) or PlanCheckpoint.empty(plan_id)
        checkpoint.completed_steps[step_id] = result
        checkpoint.timestamp = time.time()
        self.conn.execute(
            "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?)",
            (plan_id, checkpoint.to_json(), checkpoint.timestamp)
        )
        self.conn.commit()
    
    async def load(self, plan_id: str) -> Optional[PlanCheckpoint]:
        row = self.conn.execute(
            "SELECT data FROM checkpoints WHERE plan_id = ?", (plan_id,)
        ).fetchone()
        return PlanCheckpoint.from_json(row[0]) if row else None
```

### Resume logic

When an agent process restarts after a crash, the controller loads the checkpoint and skips completed steps:

```python
async def resume_plan(plan: Plan, checkpoint_store, llm) -> PlanResult:
    checkpoint = await checkpoint_store.load(plan.id)
    
    if checkpoint is None:
        # No checkpoint — start fresh
        return await run_plan(plan, checkpoint_store, llm)
    
    # Validate checkpoint against current plan
    if checkpoint.plan_hash != hash_plan(plan):
        # Plan changed — checkpoint may be incompatible
        # Policy: use checkpoint results where step IDs match, rerun others
        step_results = {
            sid: result for sid, result in checkpoint.completed_steps.items()
            if sid in {s.id for s in plan.steps}
        }
    else:
        step_results = checkpoint.completed_steps
    
    logger.info(f"Resuming plan {plan.id}: {len(step_results)} steps already complete")
    
    # Resume from the first incomplete step
    return await run_plan_with_state(plan, step_results, checkpoint_store, llm)
```

### Partial failure recovery

When a step fails and the contingency is "skip and note the gap," the controller marks the step as `status=skipped_with_gap` and continues execution. The final synthesis step receives a note in its inputs: "Step s3 was skipped due to {failure_reason}; the output may be incomplete in {affected_areas}."

This is different from replanning: the plan structure does not change, but the final output is flagged as partial. For many use cases (research, summarization), a partial result is far more useful than a full failure.

```python
def apply_contingency(step: PlanStep, failure_reason: str, step_results: dict):
    """Apply the step's contingency action and return a synthetic result."""
    if step.contingency_type == "skip":
        return StepResult(
            status="skipped_with_gap",
            output=None,
            gap_description=f"Skipped step {step.id}: {failure_reason}",
        )
    elif step.contingency_type == "use_fallback_tool":
        # Return a "retry with different tool" signal to the controller
        return StepResult(
            status="retry_with_fallback",
            fallback_tool=step.contingency_fallback_tool,
        )
    elif step.contingency_type == "use_cached_value":
        cached = load_cache(step.id)
        return StepResult(
            status="success_from_cache",
            output=cached,
            note="Used cached value from {cache_age_hours} hours ago",
        )
```

## 10. Cost Analysis: Plan-and-Execute vs Pure ReAct vs Chain

The right pattern depends heavily on cost and latency requirements. Here is the first-principles breakdown.

![Execution Trace with One Replan: timing and recovery](/imgs/blogs/plan-and-execute-pattern-4.webp)

### Token cost breakdown

For a representative 8-step research task (web search × 3, document read × 3, summarize × 1, synthesize × 1):

**Fixed chain** (no LLM routing, direct tool calls with fixed templates):
- Per tool call: 400–600 tokens in, 200–400 out
- 8 calls: ~4,800 input tokens, ~2,400 output tokens
- GPT-4o: ~$0.036 total
- GPT-4o-mini: ~$0.005 total

**Pure ReAct** (single LLM in a loop, 2–3 thoughts per tool call):
- Per step: ~1,200 tokens in (history growing), ~500 tokens out
- 8 steps with growing context: ~16,000 input tokens total, ~4,000 output
- GPT-4o: ~$0.14 total
- GPT-4o-mini: ~$0.016 total

**Plan-and-execute (same model)**:
- Planning call: 2,000 in, 1,200 out → ~$0.028
- 8 executor calls, 1,500 in each (smaller context, just step + deps): ~$0.060 execution
- Total: ~$0.088 (37% cheaper than pure ReAct, 144% more expensive than chain)

**Plan-and-execute (frontier plan + fast executor)**:
- Planning call (GPT-4o): 2,000 in, 1,200 out → ~$0.028
- 8 executor calls (GPT-4o-mini): ~$0.012
- Total: ~$0.040 (71% cheaper than pure ReAct, 11% more expensive than chain)

| Pattern | 8-step task cost | Replanning support | Parallelization | Partial recovery |
|---|---|---|---|---|
| Fixed chain | $0.005–0.036 | None | None | None |
| Pure ReAct | $0.016–0.140 | None | None | None |
| P&E same model | $0.088 | Yes | Yes | Yes |
| P&E split model | $0.040 | Yes | Yes | Yes |

### Latency breakdown

**Fixed chain**: sequential, ~2–4 s per step, 16–32 s total for 8 steps (no parallel).

**Pure ReAct**: sequential, ~3–6 s per cycle (includes think time), 24–48 s for 8 steps.

**Plan-and-execute**:
- Planning phase: 3–8 s (one-time cost)
- Execution phase: parallel, so latency ≈ critical-path length × per-step time
- For the 8-step task above (research plan with fan-out structure): critical path ≈ 4 steps × 4 s = 16 s
- Total: 3–8 s plan + 16 s execute = 19–24 s

Plan-and-execute total latency is comparable to pure ReAct on tasks with good parallelism structure, and significantly better on highly parallel tasks (where the ReAct loop's sequential nature dominates).

### The replanning cost

One replanning event adds: 3–8 s latency + planner cost ($0.028 for GPT-4o). Over 100 tasks, if 15% require replanning, expected additional cost is 15 × $0.028 = $0.42 — about 5% of total task cost. This is well worth paying for the reliability improvement.

## 11. Case Studies

### Case study 1: research report generator at a consulting firm

A mid-sized consulting firm built a plan-and-execute agent to automate first-draft competitor research reports. The task: given a target company name, produce a 5-page report covering products, pricing, recent news, key personnel, and financial performance.

**Plan structure**: 12 steps. 8 parallel information-gathering steps (one per source category: LinkedIn, Crunchbase, news search ×3, official website, SEC filings if public, product documentation), followed by 3 analysis steps (extract structured data, cross-reference for consistency, identify gaps), followed by 1 synthesis step.

**Execution**: used GPT-4o for planning, GPT-4o-mini for execution. Planning averaged 6 seconds, total task time averaged 47 seconds vs 4–6 minutes for pure ReAct on the same task.

**Key failure mode encountered**: the SEC filings search step failed on private companies (no filings to retrieve). Initial plan had no contingency for this; 30% of tasks triggered unnecessary replanning. Fixed by adding a company-type pre-check step that set a `is_public` flag, and updating contingencies to skip SEC step if `is_public=false`.

**Result after fix**: 94% of tasks completed without replanning, reports passed quality review 88% of the time (up from 72% with the previous ReAct approach), average cost $0.12/report.

### Case study 2: code review and refactoring agent

A developer tools company integrated plan-and-execute for an automated code review tool. The task: given a pull request diff, produce a structured review covering correctness, security, performance, and style.

**Plan structure**: 6 steps. Step 1: parse the diff and identify changed functions/classes. Steps 2–5 (parallel): analyze one dimension each (correctness checker, security scanner, performance analyzer, style linter). Step 6: synthesize all findings into a structured review with priority ranking.

**Key innovation**: the planner was given the diff statistics (number of files changed, lines changed, programming language) and used them to dynamically set the depth of analysis steps. A 50-line Python diff got lighter analysis steps than a 500-line Go diff — trading cost for appropriate depth.

**Failure mode**: the security scanner (step 3) sometimes returned false positives that caused the synthesizer (step 6) to over-weight security concerns. Fixed by adding a confidence score to each finding and adjusting the synthesizer prompt to weight by confidence.

**Result**: $0.08/review average, turnaround under 25 seconds, 91% of generated reviews accepted by developers without significant modification.

### Case study 3: agentic browser automation for e-commerce

An e-commerce operations team used plan-and-execute to automate competitive price monitoring — a task that requires navigating competitor websites, extracting product pages, and comparing prices across categories.

**Plan structure**: dynamic, generated per run. Planner received the list of competitor sites (5–12 per run) and target product categories, and generated a plan with one fetch step per competitor per category (15–40 parallel steps) followed by category-level comparison steps.

**Checkpointing was critical**: browser automation steps are slow (8–25 seconds each). A single 40-step plan took 3–5 minutes total. Without checkpointing, a network interruption at step 35 would lose all prior work. With checkpointing, a restart from step 36 recovered the task in under 10 seconds.

**Scale**: 800 competitor-product pairs monitored per day. Cost: $0.04/monitoring run (aggressive use of split models and caching).

### Case study 4: long-form document generation at a law firm

A litigation support team used plan-and-execute to draft discovery request documents — highly structured legal documents that require pulling information from case management systems, applying legal templates, and cross-referencing with prior motions.

**Plan structure**: 15 steps, mostly sequential (legal documents have strict ordering requirements). Despite limited parallelism, plan-and-execute was still valuable for its partial failure recovery and auditability.

**Critical benefit**: every step output was logged and checkpointed. When a partner questioned a specific claim in a draft document, the team could trace it back to exactly which step produced it, which tool call was made, and what the raw data source was. This auditability was impossible with the previous ReAct-based approach.

**Lesson learned**: the plan must be generated with domain-specific constraints built into the planner prompt. A legal document plan needs ordering constraints that are not captured in simple dependency graphs (e.g., "recitals must precede operative clauses"). These were expressed as step ordering requirements in the plan schema.

### Case study 5: multi-step data pipeline debugging

A data engineering team deployed a plan-and-execute agent to help debug failed data pipeline runs. The task: given a pipeline run ID and failure log, identify the root cause and produce a remediation plan.

**Plan structure**: 5–8 steps. Step 1: fetch the failure log. Step 2: identify the failure type (schema mismatch, null value, timeout, etc.). Steps 3–N (branch based on failure type): different investigation paths for different failure categories. Final step: generate remediation steps.

**Key design insight**: the planner was specifically prompted to generate different plan shapes based on the failure type identified in step 2. This required a two-phase approach: a lightweight "triage planner" that did step 1 and produced a stub plan, and a "full planner" that saw step 1's output and generated the complete investigation plan. The cost of the second planner call was worth paying for plan quality on ambiguous failure modes.

**Result**: 73% of debugging tasks resolved without human escalation, down from 45% with the previous approach. Mean time to resolution: 2.8 minutes vs 18 minutes for human debugging.

### Case study 6: financial report summarization pipeline

An investment research firm used plan-and-execute to process earnings call transcripts and 10-K filings, producing structured summaries with key metrics extracted, management tone analysis, and comparison to prior quarters.

**Plan structure**: 9 steps. Two parallel extraction steps (earnings call + 10-K), followed by metric normalization, tone analysis, peer comparison, and synthesis. The plan was parameterized by fiscal quarter, allowing the same plan template to be reused across hundreds of companies per quarter.

**Scale challenge**: at peak (earnings season), the system needed to process 200+ companies in a 48-hour window. The plan-and-execute architecture enabled horizontal scaling: plans were stored in a queue, and multiple executor processes consumed steps from the queue in parallel across machines. The checkpoint store (Redis) was shared across all executor processes.

**Cost optimization**: the system learned that for 80% of companies, the earnings call transcript alone was sufficient for the synthesis step (the 10-K added minimal marginal information). Added a plan-level optimization: after step 1 (earnings call extraction) completes, score the completeness of the extraction, and skip the 10-K step if the score exceeds a threshold. This reduced average per-company cost by 35%.

## 12. When to Use Plan-and-Execute, and When Not To

![8 Task Types Rated for Plan-and-Execute Suitability](/imgs/blogs/plan-and-execute-pattern-10.webp)

### The comparison across patterns

![Plan-and-Execute vs ReAct vs Chain: 7 Dimensions](/imgs/blogs/plan-and-execute-pattern-6.webp)

### Use plan-and-execute when:

**The task has 5+ distinct steps with tool calls.** The planning overhead (3–8 seconds, $0.01–0.05) amortizes over execution. For fewer than 3–4 steps, a fixed chain or simple ReAct loop is almost always the better choice.

**Steps have explicit data dependencies.** If step 5 cannot run without step 3's output, you need a dependency graph. A plan makes this explicit and enforces it. Without it, the executor will occasionally run steps out of order or with stale/hallucinated inputs.

**Partial failure recovery matters.** If a 10-step task that fails at step 8 should return 80% of the answer (with the gap noted) rather than returning nothing, plan-and-execute is the only standard pattern that supports this.

**The task benefits from parallelism.** If 4 of 8 steps are independent data gathering, parallel execution cuts wall-clock time by 40–50%. This matters for user-facing agents where latency is a quality dimension.

**Auditability is required.** In regulated industries, AI systems often need to be able to explain every decision and every data source used. A plan with step-level outputs and checkpoints provides this trace natively.

**The task will be repeated many times with different inputs.** A plan template that is validated once and then used for 1,000 task instances is a very different cost/risk profile than a free-form agent that generates a novel approach each time.

### Do not use plan-and-execute when:

**The task is 1–3 steps.** The planning overhead is a significant fraction of total task cost and latency. A fixed chain or direct tool call is 3–10× cheaper and faster.

**The task requires real-time responsiveness.** In a conversational agent where the user expects a response within 1–2 seconds, a 3–8 second planning phase is unacceptable. Pure ReAct or chain is the right choice.

**The tool landscape is dynamic.** If the set of available tools changes frequently, keeping the planner's prompt updated with accurate tool descriptions becomes a maintenance burden. Pure ReAct agents are more tolerant of tool description staleness because the model reasons about tool use at execution time.

**The task is highly exploratory.** Some agent workflows are genuinely open-ended: "research this topic and go wherever the evidence leads." A fixed plan constrains the exploration too much. Pure ReAct is better for tasks where the goal is discovery rather than execution of a known plan shape.

**You don't control the planner's prompt.** Some agent frameworks abstract away the planning prompt in ways that prevent you from enforcing the plan schema you need. In these cases, the "plan" is more like a structured ReAct scratchpad than a proper dependency graph, and you lose most of the benefits.

**Cost is the primary constraint for simple tasks.** For high-volume, low-complexity tasks (classify this email, extract these fields from this document), the planning overhead can add $0.01–0.05 per task. At 100K tasks/month, that is $1,000–5,000 per month in planning cost alone, for no quality improvement.

### The decision flowchart

```
Does the task require ≥ 5 non-trivial steps?
├── No → Use chain (if deterministic) or ReAct (if adaptive)
└── Yes
    Do later steps depend on earlier steps' outputs?
    ├── No → ReAct with memory is probably fine
    └── Yes
        Does partial failure recovery matter?
        ├── No → Evaluate if parallel execution alone justifies planning cost
        └── Yes → Use plan-and-execute
            Does latency budget allow 3–8 s planning phase?
            ├── No → Consider plan caching (generate plan once, reuse)
            └── Yes → Build plan-and-execute with checkpointing
```

### A note on hybrid patterns

The most robust production systems often combine approaches. The outer loop is plan-and-execute; each executor step internally runs a lightweight ReAct loop (2–3 cycles maximum, bounded by the step's timeout); and the replanning trigger connects the executor back to the planner. This gives you the global coherence of a plan and the local adaptability of ReAct within each step, without paying the full cost of an unbounded ReAct loop.

The pattern composites naturally with [multi-agent systems](/blog/machine-learning/ai-agent/designing-multi-agent-systems) where the planner and executor are separate agent instances, and with the [ReAct pattern](/blog/machine-learning/ai-agent/react-pattern-deep-dive) which provides the executor's inner loop mechanics. Understanding the [anatomy of the agent loop](/blog/machine-learning/ai-agent/agent-loop-anatomy) is prerequisite reading for building the controller correctly — the event loop, tool dispatch, and state management concepts all apply directly to the plan-and-execute controller.

## Summary: The One Thing to Carry Away

Plan-and-execute is not a magic bullet. It is a disciplined structural decision: you pay upfront (one planning call, added latency, more complex controller logic) in exchange for global coherence, parallelization, partial failure recovery, and auditability on long-horizon tasks. The pattern earns its overhead whenever the task has ≥ 5 steps with explicit data dependencies and partial failure is unacceptable.

The most important implementation decision is plan quality. A bad plan executed by a perfect executor produces bad results. Invest in your planner prompt — require self-verification, provide failure mode vocabulary, enforce a strict plan schema. The execution layer is almost mechanical by comparison.

Start with a same-model setup. Once you have validated plan quality and execution reliability, switch to a frontier planner + fast executor to cut cost by 50–70% on the execution side. Add checkpointing before you go to production, not after — you will hit a partial failure in the first week and regret not having it.

The replanning loop is your safety net, not your normal path. If more than 15–20% of your tasks are triggering replanning, the planner is generating bad plans. Fix the planner prompt. Do not compensate for a weak planner by lowering the replan threshold — you will end up with a system that plans, replans, re-replans, and never actually completes a task.

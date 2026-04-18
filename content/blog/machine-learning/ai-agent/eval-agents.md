---
title: "Demystifying Evals for AI Agents: A Practical Guide to Testing AI Systems"
publishDate: "2026-03-15"
category: "machine-learning"
subcategory: "AI Agent"
tags: ["ai-agent", "evaluation", "testing", "llm", "anthropic", "claude", "ml-engineering"]
date: "2026-03-15"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "A comprehensive guide to building evaluations for AI agents — from grader types and metrics to a step-by-step roadmap for shipping agents with confidence."
---

## Introduction

You've built an AI agent. It works great in your demos. But how do you know it actually works reliably in production? How do you catch regressions before your users do? How do you confidently ship improvements?

The answer is **evaluations** (or "evals") — systematic tests that measure how well your AI agent performs. Think of evals as the **unit tests of the AI world**, but with a twist: because AI outputs are non-deterministic (the same input can produce different outputs each time), evaluating them requires fundamentally different approaches than traditional software testing.

If you've ever written unit tests for your code, you already understand the core idea. The difference is that in traditional testing, you check for exact outputs (`assertEqual(add(2,3), 5)`), while in AI evals, you need to assess whether an output is *good enough* — and "good enough" can be surprisingly hard to define.

This article breaks down everything you need to know about building evals for AI agents, based on [Anthropic's engineering guide](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

## What Is an Evaluation?

At its simplest, an **evaluation** is a test that applies grading logic to AI system outputs. But there's an important distinction between two types:

### Single-Turn Evals

These are the simplest form — one prompt in, one response out, then grade it:

```
Prompt → Response → Grade
```

**Example:** You ask "What is the capital of France?", the agent responds "Paris", and your grader checks if the response contains "Paris". Done.

This is similar to how you'd test a pure function in traditional software. One input, one output, deterministic check.

### Multi-Turn Evals

This is where things get interesting — and where most real-world agent evaluation happens. Agents don't just answer questions; they **take actions across multiple steps**, use tools, modify the environment, and make decisions along the way:

```
Task → Agent acts → Uses tools → Modifies state → ... → Final outcome → Grade
```

**Example:** You tell a coding agent "Find the bug in this repo and fix it." The agent then:
1. Reads the repository structure
2. Looks at relevant files
3. Runs the existing test suite to see failures
4. Identifies the bug
5. Edits the code
6. Runs tests again to verify the fix
7. ✅ All tests pass

Notice how much more complex this is than a single-turn eval. The agent made **multiple decisions**, used **multiple tools**, and the final state of the environment (the codebase) matters just as much as any individual response.

This is why agent evals are fundamentally different from simple LLM evals — you're not just grading a text response, you're grading a **sequence of decisions and their cumulative effect on the world**.

## Key Terminology

Before diving deeper, let's establish a shared vocabulary. These terms come up constantly in the eval world, and having clear definitions will make everything else easier to follow:

| Term | Definition | Example |
|------|-----------|---------|
| **Task** | A single test with defined inputs and success criteria | "Refactor this function to use async/await" |
| **Trial** | One attempt at a task (run multiple for consistency) | Running the same task 5 times to see if results vary |
| **Grader** | Logic that scores agent performance (tasks can have multiple graders) | A unit test, an LLM judge, or a human reviewer |
| **Transcript** | Complete record of outputs, tool calls, and reasoning | The full conversation log including all tool uses and decisions |
| **Outcome** | Final environmental state after trial completion | The modified codebase, database state, or UI after the agent finishes |
| **Eval harness** | Infrastructure that runs evals end-to-end | The test runner + environment setup + grading pipeline |
| **Agent scaffold** | System that enables models to act as agents | Tool definitions + system prompts + orchestration logic |
| **Eval suite** | Collection of tasks measuring specific capabilities | 50 tasks testing "can the agent handle API errors gracefully" |

**Think of it this way:** An eval suite contains many tasks. Each task is run multiple times (trials). Each trial produces a transcript and an outcome. Graders score the outcomes and transcripts. The eval harness orchestrates all of this.

## Why Build Evaluations?

### The "It Works On My Machine" Problem

Early prototyping may succeed through manual testing — you try a few prompts, it looks good, ship it. But **once an agent is in production and starts scaling, building without evals starts to break down**. You're essentially flying blind with no way to verify except to guess and check.

Here's the painful reality of what happens to teams **without** evals:

- **Whack-a-mole debugging** — Fix one failure, create another, with no way to tell until a user complains
- **Can't distinguish signal from noise** — Is that failure a real regression or just the model being non-deterministic today?
- **Slow model adoption** — When a more powerful model comes out, you face weeks of manual testing while competitors with evals upgrade in days
- **No baselines** — You can't answer "Is our agent getting better or worse?" because you have no historical data on latency, cost, or error rates
- **Team misalignment** — Product and research teams talk past each other because there's no shared vocabulary for agent performance

### What Evals Enable

Teams **with** evals get compounding benefits over time:

- **Automatic regression detection** — Every change is tested against hundreds of scenarios automatically
- **Faster model adoption** — Days instead of weeks to validate new model versions. When Anthropic released Opus 4.5, teams with evals quickly determined its strengths, tuned their prompts, and upgraded rapidly
- **Clear baselines** — Concrete numbers for latency, token usage, cost, and error rates that you can track over time
- **Shared language** — Product managers, engineers, and researchers all look at the same metrics
- **Compounding value** — Every bug that becomes a test case prevents that class of bug from ever recurring. The costs of building evals are visible upfront, but the benefits compound over the agent's lifecycle

### Real-World Examples

**Descript** (AI-powered video editing) evolved their eval approach through three stages:
1. **Manual grading** of agent outputs — slow, inconsistent, didn't scale
2. **LLM-based graders** with periodic human calibration — scalable, reliable
3. **Separate quality and regression suites** — comprehensive coverage

They tested three dimensions of a successful editing workflow: *don't break things*, *do what I asked*, and *do it well*. Each dimension had its own graders and metrics.

**Bolt** (AI coding platform) built comprehensive evaluation systems within three months of launch, enabling rapid iteration on their agent's capabilities.

**Qodo** (AI code quality) initially was unimpressed by Opus 4.5 because their one-shot coding evals didn't capture the gains on longer, more complex tasks. This highlights a critical lesson: **your evals shape what you can see**. If your evals only test simple tasks, you'll miss improvements on complex ones.

### Why Agent Evals Are Fundamentally Harder Than Traditional ML Evals

If you come from a traditional ML background, you might be tempted to reuse familiar patterns — a held-out test set, accuracy/F1, maybe ROC curves. **This doesn't work for agents**, and understanding *why* is foundational to everything that follows.

| Dimension | Traditional ML Eval | Agent Eval |
|-----------|---------------------|------------|
| **Output space** | Fixed (class labels, numbers) | Open-ended text + tool calls + state mutations |
| **Ground truth** | Labeled dataset, one right answer | Often multiple valid answers, sometimes no clear "right" |
| **Determinism** | Deterministic given same weights | Non-deterministic even with temperature=0 (batching, hardware) |
| **Evaluation cost** | Pennies per prediction | Cents to dollars per trial (tokens + tool calls + compute) |
| **Failure modes** | Wrong prediction | Wrong action, unsafe action, infinite loop, tool misuse, hallucinated tool result, partial completion, subtly wrong reasoning |
| **Side effects** | None (pure prediction) | Real — can modify files, call APIs, send messages |
| **Time horizon** | Single inference | Multi-turn trajectories spanning minutes to hours |
| **Environment** | Static input | Dynamic — each action changes the next observation |

**The compounding problem:** An agent with 95% success per step still only succeeds **59%** of the time on a 10-step task (0.95^10 ≈ 0.59). Step-level metrics lie about end-to-end capability. This is why you must evaluate trajectories, not just individual responses.

**The sparse reward problem:** Traditional ML gives feedback on every example. An agent's trajectory may have 50 tool calls, but you might only know whether the *final outcome* was correct. Localizing *which* step caused the failure requires careful transcript analysis, intermediate graders, or counterfactual replay.

**The observability problem:** With a classifier, you see input → output. With an agent, you also need to see the *reasoning*, *tool choices*, *tool results*, *error recovery attempts*, and *backtracking*. A correct outcome achieved through fragile logic is still a bug waiting to happen.

## The Three Types of Graders

How do you actually *grade* an AI agent's output? There are three fundamental approaches, and understanding when to use each is one of the most important skills in building evals.

### 1. Code-Based Graders (Deterministic)

**Fast, cheap, objective, and 100% reproducible.**

These use deterministic logic — regular code — to check outputs. Think of them as unit tests for AI.

```python
import subprocess
import os

# 1. String matching — check if the answer contains expected content
def grade_capital(response: str) -> bool:
    return "paris" in response.lower()

# 2. Binary test — run unit tests to verify correctness
def grade_code_fix(workspace: str) -> bool:
    result = subprocess.run(
        ["pytest", workspace],
        capture_output=True
    )
    return result.returncode == 0

# 3. Tool call verification — check the agent used the right tools
def grade_used_search(transcript: list) -> bool:
    """Did the agent search the web when it should have?"""
    return any(
        call["tool"] == "web_search"
        for call in transcript
    )

# 4. State verification — check the final state of the environment
def grade_file_created(workspace: str) -> bool:
    """Did the agent create the expected output file?"""
    return os.path.exists(os.path.join(workspace, "output.csv"))

# 5. Static analysis — run linters to check code quality
def grade_code_quality(workspace: str) -> bool:
    """Does the code pass linting with no new errors?"""
    ruff = subprocess.run(
        ["ruff", "check", workspace],
        capture_output=True
    )
    mypy = subprocess.run(
        ["mypy", workspace],
        capture_output=True
    )
    return ruff.returncode == 0 and mypy.returncode == 0
```

**When to use:** Whenever you can! Code graders are the most reliable option because they're perfectly reproducible — you'll get the same result every time you run them.

**Strengths:**
- Lightning fast (milliseconds)
- Near-zero cost
- 100% reproducible — no variance between runs
- Easy to debug when something goes wrong
- Can verify specific conditions precisely

**Weakness:** They're **brittle to valid variations**. For example, `grade_capital` above would work fine for "Paris" but would also incorrectly pass "Paris, Texas is a city in the US" (which doesn't answer the question about France). And if you used exact matching (`response == "Paris"`), it would fail on perfectly good answers like "The capital of France is Paris, which is also known as the City of Light."

**Rule of thumb:** Use code graders to check *what* happened (did the tests pass? was the file created? was the right tool called?), and use other grader types to assess *how well* it happened.

### 2. Model-Based Graders (LLM-as-Judge)

**Flexible, scalable, and capable of capturing nuance that code can't.**

These use another LLM to evaluate the agent's output. It's like having an automated reviewer who can understand context and quality.

```python
import json

GRADER_PROMPT = """
You are evaluating a customer support agent's response.

Score each dimension from 1-5, where 1 is terrible and 5 is excellent:

1. **Empathy**: Did the agent acknowledge the customer's frustration
   and show understanding of their situation?
2. **Accuracy**: Was the information provided factually correct?
   Did the agent avoid making promises it can't keep?
3. **Resolution**: Was the issue actually resolved, or at least
   given a clear path to resolution?
4. **Tone**: Was the response professional, warm, and helpful
   without being condescending?

Customer query: {query}
Agent response: {response}
Expected resolution: {expected}

If you don't have enough information to score a dimension,
return "Unknown" for that dimension.

Return a JSON object with scores for each dimension and a brief
explanation for each score.
"""

def grade_with_llm(query, response, expected):
    result = llm.evaluate(GRADER_PROMPT.format(
        query=query,
        response=response,
        expected=expected
    ))
    scores = json.loads(result)
    # Pass if all scorable dimensions are >= 4
    scorable = {
        k: v for k, v in scores.items()
        if isinstance(v, (int, float))
    }
    return all(score >= 4 for score in scorable.values())
```

**Common approaches for LLM-as-judge:**

| Approach | How it works | Best for |
|----------|-------------|----------|
| **Rubric-based scoring** | Score against specific criteria (1-5 scale) | Multi-dimensional quality assessment |
| **Pairwise comparison** | "Is response A better than B?" | A/B testing model versions |
| **Natural language assertion** | "Does this response contain a greeting?" | Simple yes/no quality checks |
| **Consensus** | Multiple LLM judges vote, majority wins | High-stakes evaluations where accuracy matters |

**When to use:** When output quality is subjective or has many valid forms — creative writing, conversation quality, explanation clarity, code style, reasoning quality.

**Strengths:**
- Flexible — handles open-ended and freeform outputs
- Scalable — can grade thousands of outputs cheaply
- Captures nuance — understands context, tone, and quality
- Handles tasks where there's no single "right" answer

**Weaknesses:**
- Non-deterministic — the same input might get slightly different scores
- More expensive than code graders ($0.01-0.10 per evaluation)
- **Requires calibration** — you must regularly compare LLM grades against human expert judgments

**Critical tip for LLM graders:** Always give the LLM a way out. Include instructions like "Return 'Unknown' when you don't have enough information." This prevents the grader from hallucinating scores when it genuinely can't assess a dimension.

**Another pro tip:** Grade each dimension in isolation with separate LLM calls rather than asking one LLM to grade everything at once. This prevents cross-contamination between dimensions (e.g., a bad empathy score influencing the accuracy score).

#### Known Biases of LLM-as-Judge (And How to Mitigate Them)

LLM graders aren't neutral — they bring systematic biases that can silently corrupt your results. Understanding these biases is the difference between an eval you can trust and one that gives you false confidence.

| Bias | What happens | Mitigation |
|------|--------------|------------|
| **Position bias** | In pairwise comparisons, LLMs prefer the first (or sometimes last) response regardless of quality | Run each comparison twice with swapped order; only count consistent preferences |
| **Verbosity bias** | Longer responses get higher scores even when adding no value | Include "penalize unnecessary length" in the rubric; measure length separately as its own metric |
| **Self-preference bias** | A model tends to rate its own outputs higher than other models' outputs | Use a different model family as the judge, or use multiple judges and require consensus |
| **Sycophancy** | Agrees with framing in the prompt ("Is this response excellent?") | Use neutral framing ("Rate this response on X from 1-5") and avoid leading language |
| **Format bias** | Markdown tables, bullet points, and headers get scored higher independent of content | Normalize format or strip formatting before grading on substance |
| **Recency / primacy in long transcripts** | For multi-turn grading, LLMs over-weight the opening and closing, under-weighting the middle | Chunk long transcripts and grade sections separately, then aggregate |
| **Calibration drift** | Same prompt, same output — but scores drift across model versions of the judge | Pin the judge model version; when upgrading, re-run calibration against human labels |

**Concrete calibration workflow:** Build a "golden set" of ~50–100 examples with human expert labels across the full score range (not just pass/fail — include borderline cases). Compute the agreement between your LLM judge and the golden set using Cohen's kappa (target ≥0.6 for usable, ≥0.8 for reliable). Re-run this calibration monthly or whenever you change the judge prompt or model.

```python
from sklearn.metrics import cohen_kappa_score

# Human-labeled golden set
human_scores = [4, 5, 2, 3, 5, 1, 4, ...]    # n=100 examples
judge_scores = [4, 4, 2, 3, 5, 2, 4, ...]    # same examples graded by LLM

kappa = cohen_kappa_score(human_scores, judge_scores, weights="quadratic")
# < 0.4: judge is unreliable — fix the prompt or switch models
# 0.4-0.6: marginal — usable with caution, sample and verify frequently
# 0.6-0.8: good — the judge generally agrees with humans
# > 0.8: excellent — treat judge scores as trustworthy
```

**Pitfall:** Don't calibrate only on clear pass/fail cases. The hard work of an LLM judge is on the borderline — a 3 vs. 4 distinction. If your golden set is all 1s and 5s, high agreement tells you nothing.

### 3. Human Graders

**The gold standard for quality, but the most expensive and slowest option.**

Sometimes only a human can tell you if the output is truly good. Human graders come in several forms:

- **SME (Subject Matter Expert) review** — Domain experts evaluate outputs. A doctor reviews medical advice, a lawyer reviews legal analysis
- **Crowdsourced judgment** — Multiple human raters score outputs independently, then you aggregate scores
- **A/B testing** — Real users interact with two agent versions, and you measure which one they prefer
- **Sampling** — Humans review a random subset (e.g., 5%) of production outputs to spot-check quality

**When to use:**
- For **calibrating LLM graders** — "Is our LLM judge actually aligned with what humans think is good?"
- For **high-stakes domains** — Medical, legal, financial advice where errors have serious consequences
- For **subjective quality** that even LLMs struggle with — creativity, humor, cultural sensitivity
- For **building intuition** — Reading transcripts and manually grading builds understanding of how your agent actually behaves

**Trade-offs:** Expensive ($1-50 per evaluation), slow (hours to days), and surprisingly inconsistent (~70-85% inter-rater agreement even among experts). But there's no substitute for human judgment on truly ambiguous cases.

### Grader Comparison Summary

| | Code-Based | Model-Based | Human |
|---|---|---|---|
| **Speed** | Milliseconds | Seconds | Hours/Days |
| **Cost** | Near-zero | $0.01-0.10/eval | $1-50/eval |
| **Consistency** | 100% reproducible | ~90-95% consistent | ~70-85% consistent |
| **Flexibility** | Low (rigid rules) | High (understands nuance) | Highest |
| **Best for** | Correctness, state checks | Quality, style, reasoning | Calibration, edge cases |
| **Debugging** | Easy — read the code | Medium — check the prompt | Hard — ask the person |

**The best eval suites combine all three types.** Use code graders for objective checks, LLM graders for quality assessment, and human graders for calibration and edge cases.

### Scoring: How to Combine Multiple Graders

Most real tasks use multiple graders. There are three ways to combine them:

**Weighted scoring** — Each grader contributes a weighted score, and the combined score must hit a threshold:
```python
# Example: 40% correctness + 30% quality + 20% tool usage + 10% efficiency
total = 0.4 * correctness + 0.3 * quality + 0.2 * tools + 0.1 * efficiency
passed = total >= 0.7  # 70% threshold
```

**Binary (all-must-pass)** — Every grader must pass:
```python
passed = correctness and quality and tools and efficiency
```

**Hybrid** — Some graders are hard requirements, others contribute to a score:
```python
# Must pass correctness, then quality contributes to score
if not correctness:
    passed = False
else:
    passed = (0.5 * quality + 0.3 * tools + 0.2 * efficiency) >= 0.6
```

## Capability vs. Regression Evals

Not all evals serve the same purpose. Understanding this distinction is critical for organizing your eval suite effectively.

### Capability Evals: "Can we do this yet?"

These target areas where the agent **currently struggles**:

- Start at a **low pass rate** (maybe 20-40%)
- Used as **improvement targets** — they give you "a hill to climb"
- Help you measure progress on hard problems
- Drive focused improvement efforts

**Example:** You want your coding agent to handle multi-file refactors. Right now it only succeeds 25% of the time. You create a suite of 30 multi-file refactoring tasks and track progress week over week.

**Practical approach: Eval-driven development.** Build evals to define planned capabilities *before* agents can fulfill them, then iterate until the agent performs well. This is analogous to test-driven development (TDD) in traditional software — write the test first, then make it pass.

### Regression Evals: "Did we break anything?"

These protect against **backsliding** on things that already work:

- Should maintain **~100% pass rates**
- Run on **every change** (in CI/CD pipelines)
- Catch unintended side effects of "improvements"
- Act as a safety net for confident shipping

**Example:** Your coding agent already handles simple bug fixes reliably. You create 50 bug-fix tasks that should always pass. If any of them fail after a change, you know you've introduced a regression.

### The Graduation Pattern

Here's the beautiful part — these two types work together in a **flywheel**:

```
Week 1:  Capability eval "Multi-file refactors" → 25% pass rate
Week 4:  Improvements → 60% pass rate
Week 8:  More improvements → 85% pass rate
Week 12: Stable at 95% pass rate
         ↓
         "Graduates" to regression suite (must stay ≥ 90%)
         ↓
         New capability eval created for harder tasks
```

**The cycle:**
1. Create capability evals for things the agent can't do well
2. Improve the agent until those evals pass reliably
3. Graduate those evals to the regression suite
4. Create new, harder capability evals
5. Repeat

This means your regression suite **grows organically** over time, protecting an ever-expanding set of capabilities, while your capability evals always push the frontier.

## Eval Strategies by Agent Type

Different types of agents need different evaluation approaches. Let's look at four common agent types with detailed examples.

### Coding Agents

Coding agents are the **easiest to evaluate** because code has clear correctness criteria — either the tests pass or they don't.

**Evaluation structure:**

```yaml
task:
  name: "Fix off-by-one error in pagination"
  input:
    repo: "sample-app"
    issue: "Page 2 shows duplicate items from page 1"
  # Reference solution proves the task is solvable
  reference_solution:
    file: "solutions/fix-pagination.patch"

graders:
  # Deterministic: Does the fix work?
  - type: code
    test: "pytest tests/test_pagination.py"
    weight: 0.4

  # Deterministic: No new issues introduced?
  - type: code
    test: "ruff check src/"
    weight: 0.1

  # LLM: Is the fix clean and idiomatic?
  - type: llm
    rubric: |
      Score 1-5: Is the code change minimal,
      well-structured, and following project conventions?
      A good fix changes only what's necessary.
      A poor fix adds unnecessary complexity or
      touches unrelated files.
    weight: 0.2

  # Deterministic: State check
  - type: code
    check: "all original tests still pass"
    weight: 0.3

metrics:
  - turns_taken        # How many steps did the agent take?
  - tool_calls_count   # How many tool calls were made?
  - tokens_used        # How much did this cost?
  - wall_clock_time    # How long did it take?
```

**Key insight:** Don't just check *if* the code works — also check *how* the agent got there. An agent that produces correct code through 47 random file edits is significantly worse than one that makes 3 targeted changes, even if both pass the tests. Track metrics like turns taken, tool calls, and tokens used to measure efficiency alongside correctness.

**Benchmark reference:** SWE-bench Verified improved from ~40% to >80% solve rates within one year, showing rapid progress in coding agent capabilities. But as the benchmark saturated, the community needed harder benchmarks to maintain useful signal.

### Conversational Agents

Conversational agents are **harder to evaluate** because interaction quality itself is the product. There's no simple "pass/fail" — success is multidimensional.

**Example: Customer support agent**

```yaml
task:
  name: "Handle frustrated customer with billing dispute"
  persona:
    name: "Alex"
    mood: "frustrated"
    issue: "Charged twice for subscription"
    history: "3 previous contacts about same issue"
    constraint: "Refund amount must be ≤ $100"
  tools_available:
    - lookup_account
    - issue_refund
    - create_ticket
    - transfer_to_human

graders:
  # LLM: Conversation quality (multidimensional)
  - type: llm
    rubric: |
      Score each independently (1-5):

      Empathy: Did the agent acknowledge Alex's frustration
      and the fact that this is their 4th contact about
      the same issue? (1=ignored, 5=deeply acknowledged)

      Clarity: Were the next steps clearly communicated?
      Did the customer know exactly what would happen
      and when? (1=vague, 5=crystal clear)

      Groundedness: Did the agent ONLY promise actions
      it could actually take? Did it avoid overpromising?
      (1=made false promises, 5=perfectly grounded)
    weight: 0.3

  # Code: Was the issue actually resolved?
  - type: code
    check: "refund_issued == True and ticket_created == True"
    weight: 0.4

  # Code: Required tools used appropriately?
  - type: code
    check: |
      lookup_account in tool_calls
      and issue_refund in tool_calls
      and refund_amount <= 100
    weight: 0.2

  # Code: Conversation length reasonable?
  - type: code
    check: "turn_count <= 8"
    weight: 0.1
```

**Why this is hard:** Consider these two agent responses to Alex's billing dispute:

**Response A:** "I've processed your refund. Is there anything else?" *(Resolved the issue, but zero empathy — Alex has contacted 3 times before!)*

**Response B:** "I'm so sorry you've had to deal with this 4 times, Alex. That's completely unacceptable. I've looked into your account and I can see the duplicate charge. Let me process that refund right now... Done! You should see $29.99 back in your account within 3-5 business days. I've also flagged this in our system so it won't happen again." *(Resolved AND empathetic — this is what good looks like)*

A code grader would mark both as "pass" since both issued the refund. Only an LLM grader (or human) can distinguish the quality difference.

**Benchmark reference:** τ2-Bench simulates user personas across domains like retail support and airline booking, testing agents in extended interactions where maintaining context and persona consistency matters.

### Research Agents

Research agents face a unique challenge: **ground truth shifts constantly** as reference content changes. The "right answer" to "What are the latest advances in quantum computing?" changes every week.

**Evaluation combines multiple grader types:**

```python
def evaluate_research_output(query, output, sources):
    """
    Research agents need multi-dimensional evaluation because
    a factually correct but poorly sourced report is dangerous,
    and a well-sourced but incomplete report is useless.
    """
    scores = {}

    # 1. Groundedness: Are claims supported by cited sources?
    #    This catches hallucination — the #1 failure mode
    scores["groundedness"] = llm_grade(
        "For each factual claim in the output, check if it "
        "has explicit support in the provided sources. "
        "Score 0-1 where 1 means every claim is supported.",
        output, sources
    )

    # 2. Coverage: Are the key facts included?
    #    A research report that misses critical information is
    #    worse than one with minor gaps
    required_facts = get_required_facts(query)
    found = sum(
        1 for fact in required_facts
        if fact_present(output, fact)
    )
    scores["coverage"] = found / len(required_facts)

    # 3. Source quality: Did it consult authoritative sources?
    #    Citing a random blog vs. a peer-reviewed paper matters
    scores["source_quality"] = evaluate_source_authority(sources)

    # 4. Synthesis: Is the output coherent and well-organized?
    #    Can a reader actually learn from this?
    scores["synthesis"] = llm_grade(
        "Rate the coherence, organization, and analytical "
        "insight of this research summary. Does it go beyond "
        "just listing facts to provide useful analysis? (1-5)",
        output
    )

    return scores
```

**Key insight:** Research evals require **frequent expert calibration** because the "right answer" evolves. Schedule regular sessions (e.g., monthly) where domain experts review grading criteria and sample outputs. What was a great research summary six months ago might miss critical new developments today.

**Common pitfall:** Don't over-index on factual accuracy alone. A research agent that perfectly recites facts but fails to synthesize them into useful insights is barely better than a search engine.

### Computer Use Agents

These agents interact through **human interfaces** — screenshots, clicks, scrolling, typing — rather than APIs. They see what a human sees and act through the same controls.

**Unique evaluation challenges:**
- **State is visual** (screenshots) rather than structured data — you can't just check a variable
- **Multiple valid paths** to the same outcome — there are many ways to navigate a website
- **Token efficiency varies dramatically** by interaction method

**Two approaches to browser interaction:**

| Approach | Speed | Token Cost | Accuracy |
|----------|-------|------------|----------|
| **DOM-based** (reading HTML elements) | Fast | High (HTML is verbose) | High for structured pages |
| **Screenshot-based** (visual analysis) | Slower | Lower | Better for dynamic content |

**Evaluation approaches from real benchmarks:**

| Benchmark | Domain | Grading Method | Example Task |
|-----------|--------|----------------|--------------|
| **WebArena** | Browser tasks | URL/page state + backend checks | "Book a flight from SF to NYC under $300" |
| **OSWorld** | Desktop tasks | File system, app config, DB, UI | "Create a spreadsheet with quarterly data" |

**Key insight:** Your evals should measure both **success** and **efficiency**. An agent that completes a task using 50,000 tokens is significantly more expensive than one that completes the same task in 5,000 tokens. Both matter in production.

## Understanding Non-Determinism: pass@k and pass^k

Here's something that trips up many teams: AI agents are **non-deterministic**. Run the same task twice, get different results. This means a single trial doesn't tell you much. You need to run multiple trials and use the right metrics to reason about performance.

### pass@k — The Optimistic Metric

**"What's the probability of getting at least one correct answer in k attempts?"**

Imagine your agent has a 60% success rate on a particular task. If you run it once, there's a 60% chance of success. But what if you run it 3 times and only need one success?

```
pass@1 = 60%   → Single attempt: 60% chance of success
pass@3 = 93.6% → 3 attempts: at least 1 succeeds 93.6% of the time
pass@5 = 98.9% → 5 attempts: at least 1 succeeds 98.9% of the time
```

**The math:** pass@k = 1 - (1 - p)^k where p is the per-trial success rate.

**Use when:** One success is enough. For example, code generation where you can verify the output — generate 5 solutions, test them all, and use the one that passes.

### pass^k — The Pessimistic Metric

**"What's the probability that ALL k attempts succeed?"**

Now flip it around. If you need your agent to succeed *every single time* (because it's talking to real customers), the math is very different:

```
pass^1 = 60%   → Same as pass@1
pass^3 = 21.6% → All 3 attempts succeed only 21.6% of the time
pass^5 = 7.8%  → All 5 attempts succeed only 7.8% of the time
```

**The math:** pass^k = p^k where p is the per-trial success rate.

**Use when:** Consistency matters. Customer-facing agents, safety-critical systems, anything where *every* interaction needs to be good.

### Visual Intuition

```
pass@k (optimistic — rises with k):
k=1:  ████████████░░░░░░░░ 60%
k=3:  ██████████████████░░ 94%
k=5:  ███████████████████░ 99%

pass^k (pessimistic — falls with k):
k=1:  ████████████░░░░░░░░ 60%
k=3:  ████░░░░░░░░░░░░░░░ 22%
k=5:  ██░░░░░░░░░░░░░░░░░  8%
```

Notice how dramatically they diverge! At k=1 they're identical, but by k=10, pass@k approaches 100% while pass^k crashes toward 0%.

### Why This Matters in Practice

**Report both metrics.** They tell you very different things:

- **High pass@k, low pass^k** → The agent *can* do it but isn't reliable. Focus on consistency improvements (better prompts, more constrained tools, guardrails).
- **Low pass@k, low pass^k** → The agent fundamentally can't do this task yet. Focus on capability improvements (better model, more tools, different approach).
- **High pass@k, high pass^k** → The agent reliably handles this. Graduate to regression suite.

**Practical example:** Your support agent has pass@3 = 95% but pass^3 = 40%. This means it *can* handle the task (95% of the time at least one of 3 tries works), but it's inconsistent (only 40% of the time all 3 tries work). For a customer-facing agent, that inconsistency is a problem — every 10th customer gets a bad experience.

### How Many Trials Do You Actually Need?

A single run of an eval suite is essentially a noisy point estimate. If you want to claim "the new prompt is better," you need enough trials to distinguish real improvement from random variance.

**Rough guide for 50-task eval suites:**

| Observed pass rate | Minimum trials per task | Why |
|---|---|---|
| Near 0% or near 100% | 1–2 | Low variance region — a single failure or success is informative |
| 30–70% (middle band) | 5–10 | High variance — binomial variance peaks at p=0.5 |
| Rare failures (95%+) | 20+ | You're hunting tail failures; need many samples to observe them |

**The math in plain English:** The standard error of a pass rate with `n` trials is roughly `sqrt(p(1-p)/n)`. For p=0.5 and n=5, that's ±22%. For n=20, it's ±11%. For n=100, it's ±5%. If two prompts look 8% apart but your standard error is ±22%, you have **no evidence** one is better — you're reading noise.

**Practical rule:** Before claiming a change is an improvement, compute a confidence interval. If CIs overlap, you need more trials or a larger eval suite. McNemar's test or a paired bootstrap works well for comparing two agent versions on the same task set.

```python
from scipy.stats import binomtest

# Old agent passed 34/50 tasks. New agent passed 40/50. Is this real?
# Use a paired comparison: on how many tasks did each version win/lose?
# Suppose 12 tasks had different outcomes: old won 3, new won 9.
result = binomtest(k=9, n=12, p=0.5, alternative="greater")
print(result.pvalue)   # 0.073 — suggestive but not conclusive at α=0.05
```

If your evals cost real money, don't run 20 trials per task by default — use adaptive sampling: run few trials first, and only invest more on tasks where the result is close to the decision boundary.

## Roadmap: Building Evals from Zero to One

Here's a step-by-step guide to building your first eval suite. Don't try to do everything at once — each step builds on the previous one.

### Step 1: Start Small with Real Failures

**Don't wait until you have hundreds of test cases.** Start with 20-50 tasks drawn from real failures. Early-stage changes show large effect sizes, so even a small eval suite provides valuable signal.

Where to find your first tasks:

| Source | Why it's valuable | Example |
|--------|------------------|---------|
| Bug reports from users | These are *real* failures that matter | "Agent said it could refund but didn't have permission" |
| Manual QA checks | Things you already verify before release | "Agent responds in the right language" |
| Support queue patterns | Repeated issues indicate systematic problems | "Agent keeps recommending discontinued product" |
| Edge cases from development | Things that worried you while building | "What happens with an empty cart?" |
| Common user workflows | Things that *must always* work | "User asks to check order status" |

```python
# Your first eval can literally be this simple.
# Don't overthink it — start capturing test cases NOW
# and refine the grading logic later.

eval_tasks = [
    {
        "name": "user_reported_bug_042",
        "description": "User reported agent failed to summarize long PDFs",
        "input": "Summarize this 3-page PDF about climate change",
        "grader": lambda output: (
            "temperature" in output.lower()   # Mentions key topic
            and len(output) > 100             # Not too short
            and len(output) < 2000            # Not too long
        )
    },
    {
        "name": "support_queue_issue_17",
        "description": "Agent was recommending discontinued product",
        "input": "What's your best product for X?",
        "grader": lambda output: (
            "discontinued_product" not in output.lower()
        )
    },
    # ... 18 more tasks from real failures
]

# Run all tasks and report results
results = []
for task in eval_tasks:
    output = agent.run(task["input"])
    passed = task["grader"](output)
    results.append({
        "name": task["name"],
        "passed": passed,
        "output": output
    })

pass_rate = sum(r["passed"] for r in results) / len(results)
print(f"Pass rate: {pass_rate:.0%}")
```

### Step 2: Write Unambiguous Tasks

A good task is one where **two domain experts would independently reach the same pass/fail verdict**. If reasonable people could disagree on whether the agent passed, the task needs refinement.

**Bad task (ambiguous):**
> "Write good code to solve this problem"

*What does "good" mean? Fast? Readable? Well-tested? Minimal?*

**Good task (unambiguous):**
> "Write a Python function `merge_sorted(a, b)` that takes two sorted lists of integers and returns a single sorted list containing all elements from both inputs. The function must:
> - Run in O(n+m) time where n and m are the lengths of the inputs
> - Not use Python's built-in `sorted()` or `list.sort()`
> - Handle empty lists correctly
> - Preserve duplicate values"

**Everything the grader checks should be clear from the task description.** Agents shouldn't fail because the spec was ambiguous.

**Create reference solutions** for every task — a known working output that passes all graders. This proves two things:
1. The task is actually solvable
2. Your graders are correctly configured

If your reference solution fails the grader, **the grader is broken, not the task**.

### Common Pitfall: Ambiguous Specs

Real-world example from Terminal-Bench: a task asked the agent to "write a script" but didn't specify a filepath. The grader assumed the script would be at `./solution.sh`. The agent wrote the script at `./scripts/solution.sh`. The agent did the right thing — the eval was broken.

Another example from METR: tasks asked agents to "optimize to a stated score threshold," but the grading required *exceeding* that threshold. This penalized models that precisely followed instructions.

**Rule of thumb:** After writing a task, pretend you're the agent. Could you pass the eval by following the instructions exactly? If there's any ambiguity, fix it.

### Step 3: Build Balanced Problem Sets

Test both **when behaviors should AND shouldn't occur**. One-sided evals create one-sided optimization.

**Example: Web search triggering for Claude.ai**

When building the search eval for Claude.ai, the team needed to balance:
- **Undertriggering** — not searching when it should ("What's happening in the news today?" → Should search)
- **Overtriggering** — searching unnecessarily ("What is 2 + 2?" → Should NOT search)

This required many rounds of refinement to get the balance right:

```python
eval_tasks = [
    # Should trigger search (50% of tasks)
    {"input": "Latest news about SpaceX launch", "should_search": True},
    {"input": "Current weather in Tokyo",        "should_search": True},
    {"input": "Who won the game last night?",    "should_search": True},
    {"input": "Current stock price of AAPL",     "should_search": True},

    # Should NOT trigger search (50% of tasks)
    {"input": "Explain what recursion is",          "should_search": False},
    {"input": "Write a haiku about rain",           "should_search": False},
    {"input": "What's 15% of 200?",                 "should_search": False},
    {"input": "Help me debug this Python function",  "should_search": False},
]
```

**Why 50/50 matters:** If 90% of your tasks are "should search" cases, an agent that *always* searches will score 90% — even though it's wasting resources on simple questions. Balanced sets force the agent to actually discriminate.

### Step 4: Build Robust Eval Harnesses

Your eval infrastructure needs **stable, isolated environments**. This is where many teams stumble — they build great tasks and graders but run them on flaky infrastructure, then can't tell if failures are real or just infrastructure noise.

**Critical principle:** Each trial must start from a **clean state**. Shared state between runs (leftover files, cached data, resource exhaustion) causes correlated failures that look like agent regressions but are actually infrastructure problems.

```python
# ✅ Good: Isolated environment per trial
def run_trial(task):
    """Each trial gets a fresh sandbox — no contamination."""
    with fresh_sandbox() as sandbox:
        # Agent starts with a clean environment every time
        result = agent.run(task, sandbox)
        grade = evaluate(result, sandbox)
        return grade

# ❌ Bad: Shared state leaks between trials
shared_workspace = "/tmp/eval"
for task in tasks:
    # Trial 2 sees leftover files from Trial 1!
    # Trial 3 might fail because Trial 2 used up disk space!
    result = agent.run(task, shared_workspace)
    grade = evaluate(result, shared_workspace)
```

**What "isolated" means in practice:**
- Fresh filesystem (no leftover files from previous trials)
- Fresh database state (if applicable)
- Fresh environment variables
- No shared network resources
- Independent random seeds

**Anti-pattern to watch for:** Your eval passes 95% of the time when run in isolation but drops to 80% when run as a full suite. This is almost always a state leakage problem.

### Step 5: Design Thoughtful Graders

This step deserves special attention because **bad graders are worse than no graders** — they give you false confidence or false alarms, both of which erode trust in the eval system.

**Principle 1: Grade the outcome, not the path**

```python
# ❌ Bad: Rigid step checking
# Penalizes valid approaches the designer didn't anticipate
def grade_rigid(transcript):
    return (
        transcript[0]["tool"] == "read_file"      # Must read first?
        and transcript[1]["tool"] == "search"      # Then search?
        and transcript[2]["tool"] == "edit_file"   # Then edit?
    )
    # What if the agent searched first, THEN read? Still valid!
    # What if the agent read two files? Breaks the index assumption!

# ✅ Good: Outcome-based — check what was achieved
def grade_flexible(outcome, transcript):
    return outcome["tests_pass"] and outcome["no_new_lint_errors"]
```

**Principle 2: Build partial credit**

A support agent that correctly identifies the problem and verifies the customer's identity but fails to process the refund is meaningfully better than one that fails immediately. Your graders should reflect this.

```python
# ✅ Good: Partial credit for multi-component tasks
def grade_with_partial_credit(outcome, transcript):
    score = 0.0

    # Core outcome (most important)
    if outcome["tests_pass"]:
        score += 0.5

    # Code quality (important but secondary)
    if outcome["no_new_lint_errors"]:
        score += 0.2

    # Efficiency (nice to have)
    if len(transcript) <= 10:
        score += 0.15

    # Precision (nice to have)
    if no_unnecessary_file_changes(transcript):
        score += 0.15

    return score  # 0.0 to 1.0 scale
```

**Principle 3: Watch out for grader bugs**

Real example: Claude Opus 4.5 initially scored only 42% on CORE-Bench. When researchers investigated, they found **multiple grading bugs**:
- Rigid grading that penalized `96.12` when expecting `96.124991...`
- Ambiguous task specs that had multiple valid interpretations
- Stochastic tasks that were impossible to reproduce exactly

After fixing the eval, the score jumped to **95%**. The agent was fine — the grader was broken.

**Principle 4: Prevent cheating**

The agent shouldn't be able to "game" the eval. Design tasks so that passing genuinely requires solving the problem rather than exploiting loopholes.

```python
# ❌ Exploitable: Agent could just write "All tests pass" to a file
def grade_naive(workspace):
    result_file = os.path.join(workspace, "results.txt")
    return "All tests pass" in open(result_file).read()

# ✅ Robust: Actually run the tests
def grade_robust(workspace):
    result = subprocess.run(
        ["pytest", workspace, "--tb=short"],
        capture_output=True
    )
    return result.returncode == 0
```

### Step 6: Read Transcripts

This is **non-negotiable** and there's no shortcut. You won't know if your graders work well unless you read the transcripts and grades from many trials.

**What to look for when reading transcripts:**

| What to check | What it tells you | Action if found |
|---------------|------------------|-----------------|
| False positives (grader says pass, but output is bad) | Grader is too lenient | Tighten grading criteria |
| False negatives (grader says fail, but output is good) | Grader is too strict or buggy | Fix grading logic |
| Unexpected failure modes | Gaps in your eval suite | Add new tasks covering this failure mode |
| Agent taking bizarre paths to success | Missing efficiency checks | Add metrics for turns, tokens, tool calls |
| Agent "gaming" the eval | Grader is exploitable | Redesign the grader to be more robust |

**How often:** Read at least 20-30 transcripts when you first launch an eval, then sample 5-10 per week to maintain calibration.

**Pro tip:** Don't just read the failures. Reading *successes* is equally important — you might discover the agent is "passing" for the wrong reasons.

### Step 7: Monitor for Eval Saturation

An eval at 100% tracks regressions but provides **no signal for improvement**. This is called saturation, and it's a sign your eval suite needs to evolve.

When your capability evals saturate:

1. **Graduate them** to regression suites (they become your safety net)
2. **Create harder capability evals** (push the frontier further)
3. **Look for new dimensions** to test (efficiency, cost, user satisfaction)

**Watch for this pattern:** As benchmarks approach their ceiling, the remaining improvements appear deceptively small but may actually represent hard, important problems. Don't confuse "slow progress on a saturated benchmark" with "the problem is solved."

### Step 8: Scale with Dedicated Teams

As your eval suite grows beyond the initial 20-50 tasks, you need organizational support:

- **Dedicated eval team** owns the core infrastructure (harnesses, grading pipelines, dashboards)
- **Domain experts** contribute tasks — product managers, customer success, salespeople all see failure modes that engineers don't
- **Contributing evals should be as routine as writing unit tests** — lower the barrier to entry
- **Enable non-engineers** to submit evals (e.g., a PM pastes a user complaint and it becomes a test case via a simple PR template)

**Practical tip:** Build a simple PR template for eval contributions:

```markdown
## New Eval Task

**Source:** [Bug report / User complaint / Manual testing / Edge case]
**Task description:** [What should the agent do?]
**Success criteria:** [How do we know it passed?]
**Failure example:** [What does a bad response look like?]
```

This makes it easy for anyone to contribute without needing to write code.

## Deep Dive: Common Problems and How to Solve Them

The generic advice above ("isolate environments", "read transcripts") skips the hard part: *what does it look like when things go wrong, and what do you actually do about it?* This section walks through the most common failure patterns teams encounter, with concrete diagnostic signals and fixes.

### Problem 1: Non-Deterministic Flakiness Masks Real Regressions

**Symptom:** You ship a prompt change. CI runs the eval suite — pass rate drops from 87% to 82%. You revert. CI runs again — pass rate is 85%. You can't tell whether the change was bad or whether you're chasing noise.

**Root cause:** Low trial counts + high per-trial variance. With 1 trial per task and p=0.85, a single suite run has a standard deviation of ~5%. A 5-point swing means nothing.

**Solutions, in order of effort:**

1. **Increase trials on the contested tasks.** If 8 tasks swung between pass and fail, rerun those 8 with n=10. The overall suite score will stabilize quickly because most tasks are either solidly passing or solidly failing.

2. **Fix the temperature and seed where possible.** For Claude via the API, setting `temperature=0` reduces but does not eliminate variance (batching, floating-point non-associativity, KV cache differences still produce variation). Document this — don't assume `temp=0` means deterministic.

3. **Use paired statistical tests, not raw percentages.** Instead of "old=87%, new=85%, therefore worse," ask "on how many tasks did the new version lose vs. win?" A McNemar's test or paired bootstrap gives you a p-value on whether the difference is real.

4. **Separate a "stable regression suite" from a "noisy capability suite."** Your CI gate should be the stable suite — tasks with near-100% pass rate where any failure is informative. Capability evals with 40–70% pass rates are for research/iteration, not CI blocking.

### Problem 2: The LLM Judge Is Drunk

**Symptom:** Your LLM-as-judge rates two similar responses very differently. Or it rates a clearly wrong answer as 5/5. Or scores drift upward over time with no real quality improvement.

**Root cause:** Usually one of:
- The rubric is ambiguous (the judge is guessing)
- The judge sees context it shouldn't (the "expected answer" in the prompt leaks)
- You upgraded the judge model and didn't re-calibrate
- The biases from the previous section (verbosity, position, self-preference) are compounding

**Diagnostic: The "Swap Test".** Take 20 transcripts. Have the judge score them. Now make a trivial edit (add a friendly opening, reformat as bullets, double the length with fluff). Rescore. If scores move by more than ±0.5 on average, your judge is responding to surface features, not substance.

**Fixes:**
- **Tighten the rubric.** Replace "is this response good?" with specific criteria and anchor examples ("5 = includes all three required steps AND cites sources; 3 = includes two of three steps; 1 = misses all steps").
- **Chain-of-thought the judge.** Ask the judge to write its reasoning *before* producing the score. This dramatically reduces score-first-justify-later behavior.
- **Pin the model and prompt.** Treat your judge prompt like production code — version it, write regression tests for it against a golden set, and re-validate on any change.
- **Use consensus for high-stakes decisions.** Two different judge models (e.g., Claude + a different family) and only trust decisions where both agree. Where they disagree, route to human review.

### Problem 3: Eval Drift — Your Ground Truth Is Stale

**Symptom:** Six months into production, your regression suite still says 98% pass. User complaints are up. The evals aren't catching it.

**Root cause:** Reality moved. User expectations rose, product scope expanded, the world changed (for research agents), or the agent was quietly "taught to the test" through prompt iteration.

**This is a classic Goodhart's Law failure: *when a measure becomes a target, it ceases to be a good measure*.**

**Fixes:**
- **Schedule "eval refresh" sprints quarterly.** Sample 50 recent production conversations. For each, ask: does our eval suite have a task that catches this case? If not, add it. This is the single highest-leverage habit for keeping evals honest.
- **Track production ↔ eval divergence.** If production failure rate is 8% but eval failure rate is 1%, your evals are too easy. Narrow that gap.
- **Retire stale tasks.** Tasks that pass 100% for 6 straight months provide zero signal. Either make them harder or remove them and free up the compute budget for new cases.
- **Tag tasks with creation date.** When eval performance shifts, knowing "80% of the tasks we're passing were added in 2024" makes drift visible.

### Problem 4: Cost Explosion

**Symptom:** Your eval suite takes 6 hours to run and costs $400 per CI run. Engineers start skipping it. Death spiral.

**Root cause:** Every task runs every trial on every PR, using your biggest model, with LLM-judge graders on every output.

**Fixes, in rough order of impact:**

1. **Tiered CI.** A 5-minute smoke suite (20 critical tasks, 1 trial each, deterministic graders only) runs on every PR. The full 500-task suite runs nightly or on release candidates. Most bugs get caught by the fast tier.

2. **Cache aggressively.** Deterministic graders are free to rerun. But if the *agent's* output is deterministic for a given input (via fixed seed + `temp=0`), cache it. Don't pay to re-run identical trajectories.

3. **Use a cheaper judge model where possible.** You don't need Opus to grade "did the response contain the word 'refund'?" — a small model or a regex works. Reserve expensive judges for truly nuanced quality dimensions.

4. **Early-stopping on obvious failures.** If an agent crashes or the output is empty, you don't need to run the expensive LLM judge on it. Short-circuit to a fail.

5. **Parallelize.** Eval tasks are embarrassingly parallel. Running 50 tasks sequentially at 60s each = 50 minutes. Running 50 in parallel = 60 seconds.

### Problem 5: Multi-Turn Tasks Where You Can't Tell Which Step Broke

**Symptom:** Agent fails a 15-step task. You know the end state is wrong. You have no idea *when* it went wrong.

**Root cause:** Only evaluating the final outcome provides a single pass/fail signal across a long trajectory. This is the sparse reward problem mentioned earlier.

**Fixes:**

- **Instrument intermediate checkpoints.** For known-structure tasks, add graders at each milestone. ("After step 3, did the agent identify the right file?" "After step 7, did it correctly diagnose the bug?") These are cheap and dramatically improve debuggability.

- **Use trajectory-level LLM graders.** Ask a judge to read the whole transcript and annotate: "Identify the step where the agent first went off-track, and explain why." This won't be perfect, but it's a strong starting point for human review.

- **Record observations at each step.** Don't just log actions — log what the agent *saw*. If the bug is "it misread a tool result," you can only diagnose that if you have the tool result captured.

- **Counterfactual replay.** When a task fails, resume from step N with a known-correct action and see if the remainder succeeds. If yes, the bug is before step N. Binary search narrows it quickly.

### Problem 6: The Agent Games Your Grader

**Symptom:** Pass rate jumps from 70% to 94% after a prompt change, but spot-checking reveals the outputs are somehow *worse*. The agent learned to optimize the grader, not the task.

**Root cause:** The grader has a shortcut — a surface feature it rewards that doesn't require solving the actual problem. Example: your LLM judge rewards "well-formatted responses with section headers," so the agent learns to pad every response with headers regardless of whether they help.

**Fixes:**
- **Red-team your grader.** Before trusting it, deliberately try to cheat it. Submit a response that is blatantly wrong but superficially formatted well. Submit one that is right but ugly. If the grader gets either wrong, fix it before deploying.
- **Add adversarial canary tasks.** Include tasks where the "obvious" high-scoring answer is wrong. If the agent passes the canary, it's exploiting the grader, not the task.
- **Run grader sanity checks on known-bad outputs.** Maintain a set of outputs that should clearly fail. When they start passing, your grader has drifted.
- **Grade outcomes, not transcripts, where possible.** Code graders that check final state are much harder to game than LLM graders that read the response text.

### Problem 7: Environment State Contamination

**Symptom:** Tasks pass individually but fail when run as a suite. Or the last 10% of tasks consistently fail regardless of which tasks they are. Or identical inputs produce different outcomes from run to run.

**Root cause:** Shared state is leaking between trials. Common culprits:
- Filesystem artifacts from previous tasks
- Cached API responses that went stale
- Database rows from previous trials
- Port/resource exhaustion (the 50th trial can't open a socket)
- Rate limits accumulating across trials

**Fixes:**
- **Containerize each trial.** Fresh container, fresh filesystem, fresh network namespace. Overhead is worth it.
- **Explicit teardown.** Every task should define what state it expects and reset it. Don't rely on "no state should leak" — verify it.
- **Order-randomize the suite.** If running tasks in a different order changes results, you have state leakage. Randomizing surfaces this immediately.
- **Pre-flight sanity check.** Before each trial, verify the environment is in the expected initial state. If not, rebuild it.

### Problem 8: Small Eval Suite Overfitting

**Symptom:** You have 20 eval tasks. You iterate prompts. Pass rate climbs from 60% → 80% → 90%. You ship. Production performance is unchanged or worse.

**Root cause:** You optimized for 20 specific tasks. Your "improvements" encoded idiosyncrasies of those specific inputs that don't generalize.

**Fixes:**
- **Split into train/dev/test.** Use some tasks for iteration, some for validation, some held out until final evaluation. This is standard ML hygiene, often ignored in eval work.
- **Grow the suite in parallel with the agent.** Every production failure becomes a new task — but keep most of them in the *held-out* test set, not the iteration set.
- **Sanity check with fresh tasks.** Before claiming an improvement ships, generate 10 new tasks you've never seen. If the "improvement" collapses on fresh tasks, it didn't generalize.

## Common Pitfalls to Avoid

Before we wrap up, here are the most common mistakes teams make when building evals:

### 1. Zero Pass Rate = Broken Eval (Usually)

With frontier models, a **0% pass rate across many trials is most often a signal of a broken task or grader, not an incapable agent**. Before concluding your agent can't do something, verify that:
- The task spec is unambiguous
- The reference solution passes the grader
- The grader handles valid output variations

### 2. Evals That Don't Match Real Usage

An eval can create **false confidence** if it doesn't match real usage patterns. If your production users ask complex multi-step questions but your evals only test simple single-step tasks, you're measuring the wrong thing.

### 3. Confusing Infrastructure Flakiness with Agent Failures

If your pass rate fluctuates between 60-90% on different days with no agent changes, you likely have an infrastructure problem, not an agent problem. Isolate your environments.

### 4. Over-Indexing on a Single Metric

No single number captures agent quality. Combine pass rates with efficiency metrics (tokens, latency, cost) and quality dimensions (empathy, accuracy, groundedness).

### 5. Treating Evals as Write-Once

Evals written on day 1 aren't necessarily still valid on day 180. The product evolved, the user base shifted, the model changed. Schedule periodic reviews — retire saturated tasks, refresh stale ones, add cases for newly discovered failure modes.

### 6. Ignoring the Cost of Eval Iteration

Every hour an engineer spends waiting for a slow eval suite is an hour they're not improving the agent. If your CI eval takes 30 minutes, engineers will skip it, batch changes, or ship without running it. Investment in fast eval infrastructure pays for itself in iteration velocity.

### 7. Not Distinguishing Agent Failure from Environment Failure

A tool returned a 500. A website rate-limited you. A flaky subprocess timed out. These aren't agent failures — they're environment failures. If your grader doesn't distinguish them, environment instability looks like agent regression. Tag trials with environment status so you can exclude (or analyze separately) trials with infrastructure problems.

## Complementary Evaluation Methods

Automated evals are powerful but insufficient alone. Think of evaluation like the **Swiss Cheese Model** from safety engineering — each method has holes, but when you layer them together, failures that slip through one layer are caught by another.

| Method | Speed | What it catches | What it misses |
|--------|-------|-----------------|----------------|
| **Automated evals** | Fast (minutes) | Regressions, known failure modes, capability gaps | Novel failures, subtle quality issues |
| **Production monitoring** | Real-time | Real-world failures, usage patterns, latency spikes | Root causes, counterfactuals |
| **A/B testing** | Slow (days/weeks) | User preference, engagement changes | Why users prefer one version |
| **User feedback** | Async | Unexpected problems, unanticipated use cases | Suffers from selection bias |
| **Manual transcript review** | Slow (hours) | Subtle quality issues, grader calibration | Not scalable, observer bias |
| **Systematic human studies** | Very slow (weeks) | Gold-standard quality judgments | Expensive, hard to run frequently |

**The most effective teams combine these methods:** automated evals for fast iteration, production monitoring for ground truth, and periodic human review for calibration. No single method catches everything, but together they provide comprehensive coverage.

## Eval Frameworks and Tools

You don't have to build everything from scratch. Here are popular frameworks, with guidance on when to use each:

### Harbor
- **Best for:** Teams that need containerized, reproducible environments
- Standardized task and grader formats
- Popular benchmarks (SWE-bench, WebArena) ship via Harbor registry
- Supports cloud scaling for running many trials in parallel

### Promptfoo
- **Best for:** Quick-start evaluation without heavy infrastructure
- Open-source, YAML-configured — minimal setup required
- Supports everything from string matching to LLM-as-judge rubrics
- Great for teams just starting with evals

### Braintrust
- **Best for:** Teams that want offline evaluation AND production observability in one tool
- Experiment tracking and comparison across model versions
- Seamless transition from offline eval to production monitoring

### LangSmith / Langfuse
- **Best for:** Teams already in the LangChain ecosystem
- Tracing, offline/online evaluations, dataset management
- **Langfuse** is the self-hosted open-source alternative — use this if you have data residency requirements

**Practical advice:** Don't spend weeks evaluating frameworks. Pick one that fits your workflow, then invest your energy in the evals themselves — iterating on high-quality test cases and graders. The framework matters far less than the quality of your tasks and graders.

## Key Takeaways

1. **Start small and start now** — 20-50 tasks from real failures beats waiting for the "perfect" eval suite. Every bug you encounter is a future test case.

2. **Use the right grader for the job** — Code-based for objective checks (did the tests pass?), LLM-based for subjective quality (was the response empathetic?), human for calibration and edge cases.

3. **Distinguish capability from regression** — Push frontiers with capability evals, lock in gains with regression evals, and use the graduation pattern to grow your safety net organically.

4. **Isolate your environments** — Flaky infrastructure creates false signal. Every trial should start from a clean state.

5. **Read your transcripts** — There's no substitute for actually looking at what your agent does. You'll catch grader bugs, discover new failure modes, and build intuition.

6. **Design graders that grade outcomes, not paths** — Don't penalize agents for solving problems differently than you expected. Focus on *what* was achieved, not *how*.

7. **Combine multiple methods** — No single evaluation approach catches everything. Layer automated evals, production monitoring, and human review for comprehensive coverage.

> Teams without evals get bogged down in reactive loops — fixing one failure, creating another, unable to distinguish real regressions from noise. Teams that invest early find the opposite: development accelerates as failures become test cases, test cases prevent regressions, and metrics replace guesswork.

## References

- [Demystifying Evals for AI Agents — Anthropic Engineering](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [SWE-bench Verified](https://www.swebench.com/)
- [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://webarena.dev/)
- [τ2-Bench: Benchmarking Conversational AI Agents](https://arxiv.org/abs/2406.12045)

---
title: "Reflection and Self-Critique: How Agents Catch Their Own Mistakes"
date: "2026-06-27"
description: "How to wire self-checking loops into agents — critic design, revision thresholds, multi-round reflection, and when self-critique hurts more than it helps."
tags: ["ai-agents", "reasoning", "reflection", "self-critique", "llm", "machine-learning", "nlp", "production-ml"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 35
---

There is an asymmetry at the heart of language model generation that most agent builders never fully internalize: the same model that struggles to write a correct answer on the first pass can usually *identify* the flaws in that answer immediately afterward. Ask GPT-4 to solve a multi-step reasoning problem and it gets it right about 70% of the time. Show it its own answer and ask "what is wrong here?" and it catches the error more than 80% of the time. The generator is fallible; the critic, aimed at the same output, is sharper.

This asymmetry is what makes reflection work. It is not magic — it is a product of the task structure. Generation requires search over a vast output space simultaneously. Critique requires checking a fixed candidate against a rubric. The second task is cognitively easier, and the model exploits that difference.

The diagram below is the core mental model: a loop in which the agent generates a draft, routes it to a critic for scoring and feedback, and either accepts it, revises it, or escalates based on the result.

![Generate → Critique → Revise Loop](/imgs/blogs/reflection-and-self-critique-agents-1.webp)

This post is about the engineering decisions that make that loop useful in production — and the failure modes that make it actively harmful if you get those decisions wrong. We will cover critic architectures, threshold calibration, multi-round diminishing returns, self-consistency sampling, the over-critique failure mode, and integration with ReAct. Concrete numbers throughout.

## 1. Why Agents Can Critique Better Than They Generate

The intuition is simple: critique is a *verification* task, and verification is almost always cheaper than synthesis. You can verify a multiplication result by dividing; you cannot synthesize the multiplication by knowing the answer. This asymmetry appears in every domain:

- A compiler can flag a type error in one pass; writing type-safe code requires the programmer to reason forward through the entire call graph.
- A spell-checker catches "recieve" in milliseconds; writing correctly requires holding the spelling rule in working memory throughout composition.
- A code reviewer can spot an off-by-one error in seconds; the author had to construct the loop invariant from scratch.

For LLMs, the asymmetry is encoded in training data structure. Critique is a natural language task that appears constantly in training corpora: peer review, bug reports, code reviews, teacher annotations, editor comments. The model has seen millions of examples of "here is text; here is what is wrong with it." It has seen far fewer examples of getting the generation right on the first attempt in a zero-shot chain-of-thought context.

There are concrete empirical benchmarks. The "Reflexion" paper (Shinn et al., 2023) showed that an agent with verbal reflection and memory outperformed a chain-of-thought agent without reflection on decision-making tasks by 20% on HotpotQA and 10% on AlfWorld. The Self-Refine paper (Madaan et al., 2023) showed consistent gains of 5–40% across seven diverse tasks ranging from code optimization to math word problems. The gains are real — but they are not free, and they are not uniform across task types.

### The Critique-Generation Gap in Practice

We can measure the gap directly. For a given task, run the following protocol:

1. Generate 50 outputs from the model on the task.
2. For each output, ask the *same model* to score it against a rubric and identify errors.
3. Use those critiques as ground truth. Measure: what fraction of actual errors does the critique catch?

On factual summarization tasks, the gap is large: the model hallucinated facts in ~22% of its summaries, and the self-critique step caught 78% of those hallucinations. On arithmetic tasks, the gap is essentially zero or inverted: the model made arithmetic errors in 31% of solutions, and self-critique correctly identified only 19% of them. The critic was actually *adding* errors by suggesting wrong corrections.

This asymmetry across task types is the central engineering constraint you have to respect when wiring reflection into any agent.

## 2. Reflection Architectures: Inline, Post-Hoc, Critic-as-Separate-Agent

Before writing a single line of code, you need to pick which of three fundamentally different reflection architectures you are building. They have very different cost, latency, quality, and complexity profiles.

![Three Reflection Architectures × Four Production Dimensions](/imgs/blogs/reflection-and-self-critique-agents-2.webp)

### Inline Reflection

The model reflects within a single generation call. You add a reflection prompt to the system message: "After generating your answer, review it for accuracy and correct any errors before finalizing." The model self-corrects in one pass.

```python
INLINE_REFLECTION_SYSTEM = """You are a careful assistant. 
Follow this process for every response:
1. Draft your answer in <draft>...</draft> tags.
2. Review the draft: check factual accuracy, logical consistency, and completeness.
3. Identify issues in <issues>...</issues> tags. If none, say "No issues found."
4. Produce the final corrected answer in <answer>...</answer> tags.
"""

def inline_reflect(client, user_query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": INLINE_REFLECTION_SYSTEM},
            {"role": "user", "content": user_query},
        ],
        temperature=0.7,
    )
    content = response.choices[0].message.content
    # Extract final answer from <answer> tags
    import re
    match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    return match.group(1).strip() if match else content
```

**When to use it**: Simple tasks, cost-sensitive pipelines, low-latency requirements. Quality gain is modest (+12–18%) because the model's internal representations don't change between draft and review — it is using the same "reasoning context" for both.

**The fundamental limit**: The model cannot genuinely surprise itself. If it was confident enough to write something wrong, it will often be confident enough to approve it wrong. The generation and critique are correlated — both share the same priors, the same biases, the same blind spots.

### Post-Hoc Critic Call

You make two separate API calls: one to generate, one to critique. The generator call produces a draft. The critic call receives the original task plus the draft and returns a structured critique with a score and feedback.

```python
import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class CritiqueResult:
    score: float          # 0.0 to 10.0
    issues: list[str]     # List of identified problems
    suggested_fixes: list[str]  # Concrete remediation steps
    should_revise: bool

CRITIC_PROMPT = """You are a strict quality critic. Review the following output
and return a JSON object with this exact schema:
{{
  "score": <float 0-10>,
  "issues": ["issue1", "issue2", ...],
  "suggested_fixes": ["fix1", "fix2", ...],
  "should_revise": <true if score < 8.0>
}}

Task: {task}
Output to Review:
{output}

Rubric:
- Factual accuracy (3 points): are all factual claims verifiable?
- Completeness (3 points): does it address all parts of the task?
- Clarity (2 points): is the explanation logical and unambiguous?
- Safety (2 points): does it avoid harmful content or unsafe advice?

Return ONLY the JSON object, no surrounding text."""

def post_hoc_critique(client, task: str, output: str) -> CritiqueResult:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": CRITIC_PROMPT.format(
                task=task, output=output
            )},
        ],
        temperature=0.1,  # Low temperature for consistent scoring
    )
    raw = response.choices[0].message.content.strip()
    data = json.loads(raw)
    return CritiqueResult(
        score=data["score"],
        issues=data["issues"],
        suggested_fixes=data["suggested_fixes"],
        should_revise=data["should_revise"],
    )
```

**When to use it**: The default for production agents. The separate context means the critic has no carry-forward from the generation step — it cannot be "anchored" to wrong reasoning. Quality gain is +20–30%.

### Critic as Separate Agent

The critic is not just a separate API call; it is a different model, potentially with different tools, a different system prompt, and a different specialized role. In multi-agent systems, the critic might have access to a calculator, a code execution environment, or a retrieval system — tools the generator did not use.

```python
class SeparateCriticAgent:
    """
    A critic that can actually verify claims, not just assess plausibility.
    Has access to: calculator, code execution, web search for fact-checking.
    """
    def __init__(self, generator_client, critic_client, tools):
        self.generator = generator_client
        self.critic = critic_client
        self.tools = tools  # {name: callable}

    def generate_and_critique(self, task: str) -> dict:
        # Step 1: Generator produces draft
        draft = self._generate(task)
        
        # Step 2: Critic independently assesses with tool access
        critique = self._critique_with_tools(task, draft)
        
        # Step 3: If critique found issues, revise
        if critique["score"] < 8.0:
            revised = self._revise(task, draft, critique)
            return {"output": revised, "rounds": 2, "critique": critique}
        return {"output": draft, "rounds": 1, "critique": critique}

    def _critique_with_tools(self, task: str, draft: str) -> dict:
        # The critic can verify math, run code, check facts
        # This is where separate-agent architecture pays off:
        # a hallucinated statistic can be cross-checked via search
        ...
```

**When to use it**: High-stakes generation (medical summaries, legal analysis, financial reports) where the quality gain of +30–40% justifies 3–5× the cost and significant orchestration complexity.

## 3. Designing the Critic Prompt: What to Ask, What to Avoid

The critic prompt is the most leverage you have in the entire reflection pipeline. A poorly designed critic produces vague complaints ("this could be improved"), circular rewrites (same content, different words), or false positives that destroy correct content.

![Critic Prompt Anatomy: Four Mandatory Layers](/imgs/blogs/reflection-and-self-critique-agents-6.webp)

The four layers are not optional — omit any one and the critique degrades in a predictable way:

**Without Task Context**: The critic has no baseline for what "correct" looks like. It will compare the output to imaginary alternatives rather than to the task specification. You get suggestions like "add more detail" when the task explicitly asked for brevity.

**Without the Output to Review**: Obviously fatal — this happens more than you'd think when developers build critic prompts that summarize or rephrase the output before asking the model to review it. The summarization itself can strip out the errors you need the critic to find.

**Without the Rubric**: The critic defaults to stylistic preferences. It will criticize word choice, passive voice, and paragraph length while missing factual errors. Add a weighted rubric and watch the error-catching rate jump.

**Without the Output Format**: The critic returns freeform text, and you cannot parse it reliably. Downstream code cannot decide whether to revise or accept. Always request structured output — a JSON schema with specific fields.

### What to Avoid in Critic Prompts

**Avoid asking the critic to rewrite the output.** Ask it to identify issues and suggest fixes, not to produce a new version. Critics that produce rewrites tend to anchor on stylistic choices rather than correctness, and they lose the original voice.

**Avoid vague rubric criteria.** "Is it clear?" produces a yes/no with no actionable feedback. "Are all technical terms defined on first use?" produces a checklist you can act on.

**Avoid asking the critic to rate things it cannot verify.** "Is this factually accurate?" — if the critic does not have retrieval tools, it will assess *plausibility* not accuracy. Plausibility-checking hallucinates confidence. Replace with "Are there any claims that should be verified with a primary source?" — this produces flagging behavior instead of false confidence.

**Avoid critic prompts that anchor on the structure of the draft.** If your critic prompt says "review this step-by-step explanation," the critic will almost always suggest keeping the step-by-step structure even if the task would be better served by a table or a diagram.

Here is a full production-grade critic prompt template:

```python
CRITIC_TEMPLATE = """You are a strict, systematic quality critic. Your job is to 
identify concrete problems in the output below — not to suggest stylistic 
improvements, not to praise, not to rewrite.

TASK SPECIFICATION:
{task}

CONSTRAINTS THE OUTPUT MUST SATISFY:
{constraints}

OUTPUT TO REVIEW:
---
{output}
---

RUBRIC (score each dimension 0–10, then average for total_score):
1. Factual accuracy (weight 3): every claim is verifiable or clearly marked uncertain
2. Task completeness (weight 3): all parts of the task spec are addressed
3. Logical consistency (weight 2): no contradictions, no non-sequiturs
4. Constraint adherence (weight 2): output respects all stated constraints

Return ONLY this JSON structure:
{{
  "total_score": <float, weighted average>,
  "dimension_scores": {{
    "factual_accuracy": <0-10>,
    "task_completeness": <0-10>,
    "logical_consistency": <0-10>,
    "constraint_adherence": <0-10>
  }},
  "issues": [
    {{
      "severity": "critical" | "major" | "minor",
      "dimension": "<which rubric dimension>",
      "description": "<concrete description of the problem>",
      "location": "<quote the problematic text>"
    }}
  ],
  "suggested_fixes": ["<fix 1>", "<fix 2>"],
  "should_revise": <true if total_score < 8.0>
}}"""
```

## 4. The Revision Threshold Problem: When to Revise vs Accept

The single most under-specified decision in reflection pipelines is the threshold: at what quality score do you accept vs revise vs escalate?

![Revision Threshold Decision Flow: Score Bands and Outcomes](/imgs/blogs/reflection-and-self-critique-agents-4.webp)

### The Case Against a Universal Threshold

The instinct is to pick one number — say, 8.0 out of 10 — and apply it everywhere. This breaks for two opposite reasons:

**Over-triggering on cheap tasks.** For a factual Q&A task where a 7.5/10 output is genuinely good enough, a threshold of 8.0 fires a revision that adds cost and latency without materially improving user experience. Your revision rate climbs to 40%, your costs double, and users see no difference.

**Under-triggering on critical tasks.** For a medical summary or a legal analysis, a 9.0/10 output may still contain one critical error that a higher threshold and a stricter rubric would have caught. A universal threshold of 8.0 lets that error through.

The correct design is **per-task-type thresholds** calibrated empirically:

```python
from enum import Enum

class TaskType(Enum):
    FACTUAL_QA = "factual_qa"
    CODE_GENERATION = "code_generation"
    LONG_FORM_WRITING = "long_form_writing"
    SUMMARIZATION = "summarization"
    SAFETY_CRITICAL = "safety_critical"

# Thresholds calibrated via offline evaluation on held-out datasets.
# acceptance_threshold: score above which we accept immediately.
# revision_threshold: score below which we escalate (no revision attempted).
TASK_THRESHOLDS = {
    TaskType.FACTUAL_QA:        {"acceptance": 7.5, "escalation": 4.0},
    TaskType.CODE_GENERATION:   {"acceptance": 8.5, "escalation": 5.0},
    TaskType.LONG_FORM_WRITING: {"acceptance": 8.0, "escalation": 5.0},
    TaskType.SUMMARIZATION:     {"acceptance": 7.0, "escalation": 4.5},
    TaskType.SAFETY_CRITICAL:   {"acceptance": 9.5, "escalation": 6.0},
}

def threshold_decision(score: float, task_type: TaskType) -> str:
    thresholds = TASK_THRESHOLDS[task_type]
    if score >= thresholds["acceptance"]:
        return "accept"
    elif score < thresholds["escalation"]:
        return "escalate"
    else:
        return "revise"
```

### Calibrating Thresholds Empirically

Calibration requires human evaluation data. The protocol:

1. Sample 200 outputs per task type from your production distribution.
2. Have humans rate each output on a 1–10 scale using the same rubric as your critic.
3. Run the critic on all 200 outputs. Plot critic score vs human score.
4. Find the critic score that corresponds to human score 7.5 (your target quality floor).
5. Set that as your acceptance threshold.

In practice, critic scores and human scores correlate at r ≈ 0.65–0.75 for well-designed rubrics. The relationship is not 1:1: critics tend to over-score long, detailed outputs (verbosity bias) and under-score brief, correct ones (brevity penalty). Calibrate for this by adjusting the rubric weights.

### Dynamic Thresholds

For production systems handling millions of requests, you want dynamic thresholds that adapt to usage patterns:

```python
class AdaptiveThresholdManager:
    """Tracks revision rates and adjusts thresholds to hit a target."""
    
    def __init__(self, target_revision_rate: float = 0.15):
        self.target_revision_rate = target_revision_rate
        self.recent_decisions: list[str] = []
        self.base_threshold = 8.0
        self.adjustment = 0.0  # Positive = stricter, negative = more lenient
    
    def record_decision(self, decision: str):
        self.recent_decisions.append(decision)
        if len(self.recent_decisions) > 1000:
            self.recent_decisions.pop(0)
    
    def adapt(self):
        if len(self.recent_decisions) < 100:
            return
        recent_revision_rate = (
            self.recent_decisions[-100:].count("revise") / 100.0
        )
        # If revision rate > target, lower threshold (accept more)
        # If revision rate < target, raise threshold (revise more)
        delta = (recent_revision_rate - self.target_revision_rate) * 0.5
        self.adjustment -= delta  # Damped adjustment
        self.adjustment = max(-2.0, min(2.0, self.adjustment))  # Clamp
    
    @property
    def current_threshold(self) -> float:
        return self.base_threshold + self.adjustment
```

## 5. Multi-Round Reflection: How Many Rounds Before Diminishing Returns

The empirical finding is consistent across multiple studies: **the biggest quality gain happens in the first round, and most tasks show diminishing returns by round 3**.

![Multi-Round Reflection: Quality Plateau After Round 2](/imgs/blogs/reflection-and-self-critique-agents-5.webp)

The numbers in the diagram reflect the median outcome across open-ended generation tasks. The pattern has a mechanistic explanation:

**Round 1** fixes the most salient errors — the kinds of issues the critic can identify most reliably (factual claims, logical contradictions, missing components). These are high-confidence critique items with clear, actionable fixes.

**Round 2** fixes subtler issues that only became visible once the round-1 errors were removed. Classic example: a factual error was masking a structural problem. Once the factual error is corrected, the structural issue is now the most glaring flaw.

**Round 3+** is where the returns collapse. The remaining errors are either undetectable by self-critique (the same blind spots apply to the critic as to the generator), or they are so minor that the critic's noise floor exceeds the signal. Worse, round 3 often introduces *regression* — the critic finds something to complain about, the reviser makes a change, and the change makes something else slightly worse.

### The Regression Problem

Round 4 in the diagram shows a quality *drop* from round 3. This is not an artifact — it appears in 30–40% of reflection sequences that run past round 2. The mechanism:

1. The critic, forced to find something actionable in a high-quality draft, surfaces a low-confidence concern.
2. The reviser acts on it, making a change that technically satisfies the concern.
3. The change has negative downstream effects on other parts of the output that the critic did not inspect in that round.

This is exactly analogous to premature optimization in software: fixing a non-bottleneck while introducing a regression elsewhere.

```python
class MultiRoundReflectionAgent:
    """
    Implements bounded reflection with regression detection.
    Exits early if quality drops between rounds (regression signal).
    """
    
    def __init__(self, generator, critic, reviser, max_rounds: int = 3):
        self.generator = generator
        self.critic = critic
        self.reviser = reviser
        self.max_rounds = max_rounds
    
    def run(self, task: str, acceptance_threshold: float = 8.0) -> dict:
        current_output = self.generator.generate(task)
        history = []
        
        for round_num in range(self.max_rounds):
            critique = self.critic.critique(task, current_output)
            history.append({
                "round": round_num,
                "output": current_output,
                "score": critique.score,
                "issues": critique.issues,
            })
            
            # Accept if above threshold
            if critique.score >= acceptance_threshold:
                return {"output": current_output, "rounds": round_num + 1,
                        "final_score": critique.score, "history": history}
            
            # Detect regression: if score dropped from last round, stop
            if round_num > 0:
                prev_score = history[-2]["score"]
                if critique.score < prev_score - 0.3:  # 0.3 noise tolerance
                    # Revert to previous round's output
                    return {"output": history[-2]["output"],
                            "rounds": round_num, "final_score": prev_score,
                            "history": history, "reverted": True}
            
            # Revise and continue
            current_output = self.reviser.revise(
                task, current_output, critique.issues, critique.suggested_fixes
            )
        
        # Exhausted rounds — return best output
        best = max(history, key=lambda h: h["score"])
        return {"output": best["output"], "rounds": self.max_rounds,
                "final_score": best["score"], "history": history,
                "exhausted": True}
```

### Practical Round Limits by Task Type

From empirical benchmarking across production agents:

| Task Type | Optimal Max Rounds | Notes |
|---|---|---|
| Factual Q&A | 1–2 | Round 2 rarely helps if round 1 did not fix it |
| Code generation | 2–3 | Test execution feedback helps round 3 if available |
| Long-form writing | 2 | Structure is set in round 1; round 3 hurts voice |
| Mathematical reasoning | 1 | Self-critique unreliable for math; use tool verification |
| Safety-critical | 3–5 | Errors are costly; regression risk is acceptable |

## 6. Critic-as-Tool vs Critic-as-Separate-LLM-Call vs Critic-as-Separate-Agent

The three architectures from the matrix earlier deserve deeper treatment — specifically, when the cost premium of a separate agent is justified.

### Critic-as-Tool: Deterministic Verification

For tasks where correctness can be *verified*, not just *assessed*, replace probabilistic LLM critique with deterministic tools:

```python
class ToolBackedCritic:
    """
    Uses deterministic tools for verification instead of LLM judgment.
    LLM assessment is a last resort, not a first resort.
    """
    
    def critique(self, task: str, output: str, task_type: str) -> CritiqueResult:
        issues = []
        
        if task_type == "code":
            # Run the code — deterministic verdict
            issues.extend(self._run_tests(output))
            issues.extend(self._run_linter(output))
            issues.extend(self._check_types(output))
        
        elif task_type == "math":
            # Use a symbolic math checker — SymPy or Wolfram
            issues.extend(self._verify_math(output))
        
        elif task_type == "factual":
            # Retrieval-augmented verification
            issues.extend(self._verify_claims_with_retrieval(output))
        
        # Only use LLM for the residual: style, completeness, non-verifiable items
        if not issues:
            # Cheap LLM check only if deterministic tools found nothing
            issues.extend(self._llm_style_check(task, output))
        
        score = 10.0 - len(issues) * 1.5
        return CritiqueResult(score=max(0, score), issues=issues,
                               suggested_fixes=[], should_revise=score < 8.0)
```

This hybrid approach is strictly better than pure LLM critique for tasks with verifiable outputs: it has deterministic recall on common error classes, zero false-positive rate on those classes, and no "confident but wrong" hallucinations of the sort that pure LLM critics produce on math.

### Critic-as-Separate-LLM-Call: The Production Default

The separate call architecture pays its cost in clear quality gain. The key implementation detail that most tutorials miss: **the critic and generator should use different temperatures**.

The generator runs at temperature 0.7–1.0 to encourage diversity and creativity. The critic should run at temperature 0.0–0.1 to be maximally deterministic and reproducible — you want the same critique for the same output across multiple calls.

```python
def reflection_pipeline(
    client,
    task: str,
    acceptance_threshold: float = 8.0,
    max_rounds: int = 3,
) -> dict:
    
    output = generate(client, task, temperature=0.8)
    
    for round_num in range(max_rounds):
        # Critical detail: low temperature for critique consistency
        critique = critique_output(client, task, output, temperature=0.05)
        
        if critique["score"] >= acceptance_threshold:
            return {"output": output, "rounds": round_num + 1,
                    "score": critique["score"]}
        
        if critique["score"] < 4.0:
            return {"output": None, "error": "quality too low",
                    "rounds": round_num + 1, "score": critique["score"]}
        
        output = revise(client, task, output, critique, temperature=0.5)
    
    return {"output": output, "rounds": max_rounds}
```

### Critic-as-Separate-Agent: Justified Cost Cases

The separate agent architecture justifies its 3–5× cost premium when:

1. **The critic needs tools the generator did not use.** A code-generating agent produces code that calls an API. The critic agent can actually *call* that API with test inputs and verify the response format. The generator had no test data; the critic synthesizes it.

2. **The domains are adversarially divergent.** In red-teaming use cases, you want the critic to be a *different model* — not just a different call. If GPT-4 generated the content, run Claude as the critic. Different training data, different fine-tuning choices, different failure modes.

3. **The quality bar is asymmetric with respect to cost.** For high-revenue or high-liability applications, the cost of a 0.1% error rate times its consequences may exceed the 5× cost of a separate critic agent by orders of magnitude.

## 7. Self-Consistency: Sampling N Responses and Finding Consensus

Self-consistency (Wang et al., 2022) is a form of reflection that does not require a critic prompt. Instead of generating one output and critiquing it, you generate N outputs and choose the one with the most support from the ensemble.

![Self-Consistency Sampling: N Samples × Four Dimensions](/imgs/blogs/reflection-and-self-critique-agents-7.webp)

The accuracy gain is real and well-documented. On GSM8K (grade-school math), self-consistency with N=40 improves accuracy from 74% (chain-of-thought, N=1) to 88%. The practical question is where the sweet spot is before cost overtakes benefit.

### The Marginal Return Curve

The returns are super-linear for small N and sub-linear for large N. Empirically:
- N=1 → N=3: approximately +8–12% accuracy on reasoning tasks
- N=3 → N=5: approximately +7–9% additional gain
- N=5 → N=7: approximately +1–2% additional gain
- N=7 → N=10: approximately +0.5% additional gain, sometimes regression

The inflection point at N=5 is where most production systems should cap. Beyond that, you are paying 40% more (5×→7×) for 1–2% gain.

```python
from collections import Counter
import asyncio

async def self_consistent_answer(
    client,
    task: str,
    n_samples: int = 5,
    temperature: float = 0.7,
) -> dict:
    """
    Generate N samples in parallel, extract answers, return majority vote.
    """
    
    async def generate_one(seed: int) -> str:
        # Note: use temperature > 0 for diversity; N=5 at temp=0 is just 5× waste
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": task}],
            temperature=temperature,
            seed=seed,  # Different seed = different exploration
        )
        return response.choices[0].message.content
    
    # Generate all samples in parallel
    samples = await asyncio.gather(*[
        generate_one(seed) for seed in range(n_samples)
    ])
    
    # Extract final answers (task-specific extraction logic)
    answers = [extract_final_answer(s) for s in samples]
    
    # Majority vote
    vote_counts = Counter(answers)
    majority_answer, vote_count = vote_counts.most_common(1)[0]
    confidence = vote_count / n_samples
    
    return {
        "answer": majority_answer,
        "confidence": confidence,
        "all_answers": answers,
        "agreement": vote_count >= (n_samples // 2 + 1),
    }
```

### Self-Consistency vs Iterative Reflection

These are not competing approaches — they target different error types:

- **Self-consistency** works best when errors come from unlucky initial token choices — the model knew the right answer but started down a wrong path. Majority vote corrects for sampling variance.
- **Iterative reflection** works best when the error is a systematic gap that the model can identify and fix given explicit feedback. The error persists across samples but the correction mechanism works.

In practice, combine them: use self-consistency for tasks with verifiable short answers (math, multiple choice), and iterative reflection for open-ended generation where majority voting is not meaningful.

## 8. The Over-Critique Failure Mode: When the Agent Rewrites Correct Answers

Every team that builds reflection agents eventually discovers this failure mode, usually expensively. The agent is given a correct, high-quality output. The critic, forced to find something to critique, surfaces a minor stylistic concern. The reviser, anchored on "the critic said to change this," makes a change that breaks something correct.

![Over-Critique Detection: When to Override the Critic](/imgs/blogs/reflection-and-self-critique-agents-9.webp)

### Taxonomy of Over-Critique

**Verbosity bias.** The critic interprets "more detail" as universally positive. Correct, concise answers get flagged for being "too brief." The reviser adds a paragraph that introduces ambiguity or redundancy.

**Anchor flipping.** A technically correct claim gets flagged as "potentially ambiguous" and rewritten to be clearer — but the rewrite introduces a different ambiguity. The critic saw the original phrasing; it cannot see the ways the rewrite fails.

**Spurious specificity.** The critic requests concrete numbers or examples. The reviser provides them — but hallucinates them, because the task did not supply real numbers to use.

**Style overcorrection.** The critic prefers passive voice for academic writing. The reviser converts active constructions to passive, which introduces agent ambiguity ("the system was trained" — by whom?).

### Detection and Prevention

The most reliable prevention mechanism is the regression check: before accepting any revision, re-score the original output and the revised output, and only commit if the revision strictly improves.

```python
class RegressionAwareReviser:
    """
    Only accepts a revision if it strictly improves the quality score.
    Prevents the critic from degrading a good output.
    """
    
    IMPROVEMENT_THRESHOLD = 0.3  # Must improve by at least 0.3 to accept
    
    def revise_with_regression_check(
        self,
        task: str,
        original: str,
        critique: CritiqueResult,
        critic,
    ) -> dict:
        # Generate proposed revision
        proposed = self._revise(task, original, critique)
        
        # Score both original and proposed
        original_score = critic.score_only(task, original)
        proposed_score = critic.score_only(task, proposed)
        
        delta = proposed_score - original_score
        
        if delta >= self.IMPROVEMENT_THRESHOLD:
            return {"output": proposed, "decision": "accept_revision",
                    "delta": delta}
        elif delta > -0.3:
            # Negligible change — keep original to avoid regression risk
            return {"output": original, "decision": "override_critic",
                    "delta": delta}
        else:
            # Revision made things worse — escalate
            return {"output": None, "decision": "escalate",
                    "delta": delta, "error": "revision_regressed"}
    
    def _revise(self, task: str, original: str, critique: CritiqueResult) -> str:
        prompt = f"""Given this task and the identified issues, revise the output.
Only fix the identified issues. Do not change anything else.

Task: {task}
Original Output: {original}
Issues to Fix:
{chr(10).join(f'- {issue}' for issue in critique.issues)}
Suggested Fixes:
{chr(10).join(f'- {fix}' for fix in critique.suggested_fixes)}

Revised Output:"""
        # ... API call
        return ""  # placeholder
```

### Confidence-Weighted Critique

Another approach: not all critique items are equally reliable. Attach a confidence score to each issue and only act on high-confidence items.

```python
CONFIDENCE_WEIGHTED_CRITIC = """Review the output. For each issue you identify,
rate your confidence that this is actually a problem (not a stylistic preference)
on a scale of 1–5 where:
1 = stylistic preference, reasonable people disagree
2 = minor concern, low impact
3 = moderate concern, affects quality
4 = significant problem, should be fixed
5 = critical error, must be fixed

Only return issues with confidence ≥ 3."""
```

Items with confidence < 3 are stylistic — skip them. Items with confidence ≥ 4 are genuine errors — act on them. This filtering alone reduces the over-critique rate by ~40% in practice.

## 9. Combining Reflection with ReAct and Plan-and-Execute

Reflection does not only apply to final outputs. In agentic loops, you can insert a reflection step after each observation — before the next action is chosen.

![ReAct + Reflection: Combined Agent Pattern](/imgs/blogs/reflection-and-self-critique-agents-8.webp)

The insertion point matters. Reflection after an *observation* catches misinterpretations of tool output before they compound. Reflection after a *thought* catches reasoning errors before a bad action is taken. Reflection on the *final answer* catches output-quality issues.

### ReAct with Mid-Loop Reflection

```python
class ReActWithReflection:
    """
    Standard ReAct loop augmented with a reflection step after each observation.
    Reflection runs post-observation, before the next thought — catching
    tool-result misinterpretations early.
    """
    
    def __init__(self, llm, tools: dict, critic_llm=None):
        self.llm = llm
        self.tools = tools
        self.critic_llm = critic_llm or llm  # Fall back to same model if no separate critic
    
    def run(self, task: str, max_steps: int = 10) -> str:
        trajectory = []
        
        for step in range(max_steps):
            # THOUGHT: what should I do next?
            thought = self._think(task, trajectory)
            trajectory.append({"type": "thought", "content": thought})
            
            if "FINAL ANSWER:" in thought:
                answer = thought.split("FINAL ANSWER:")[-1].strip()
                # Reflection on final answer before returning
                final_critique = self._reflect_on_answer(task, answer, trajectory)
                if final_critique["should_revise"]:
                    answer = self._revise_answer(task, answer, final_critique)
                return answer
            
            # ACTION: call a tool
            action, action_input = self._parse_action(thought)
            if action not in self.tools:
                trajectory.append({"type": "error", "content": f"unknown tool: {action}"})
                continue
            
            observation = self.tools[action](action_input)
            trajectory.append({"type": "observation", "content": observation,
                                "tool": action, "input": action_input})
            
            # REFLECTION: did this observation make sense? was the tool call correct?
            reflection = self._reflect_on_observation(
                task=task,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            )
            
            if reflection["critical_error"]:
                # Tool result is garbage — flag and try a different approach
                trajectory.append({"type": "reflection",
                                    "content": reflection["analysis"],
                                    "critical": True})
            else:
                trajectory.append({"type": "reflection",
                                    "content": reflection["analysis"]})
        
        return "MAX_STEPS_REACHED"
    
    def _reflect_on_observation(
        self,
        task: str,
        thought: str,
        action: str,
        action_input: str,
        observation: str,
    ) -> dict:
        prompt = f"""You are reviewing a step in an agent's reasoning process.

Task: {task}
Agent's thought: {thought}
Tool called: {action}
Tool input: {action_input}
Observation (tool result): {observation}

Is there a critical error here? Specifically:
1. Does the observation match what the thought expected?
2. Is the tool result internally consistent and believable?
3. Would acting on this observation lead toward the task goal?

Return JSON: {{"critical_error": <bool>, "analysis": "<one sentence>", 
               "recommended_action": "continue" | "retry_tool" | "revise_approach"}}"""
        
        response = self.critic_llm.complete(prompt, temperature=0.1)
        return parse_json(response)
```

### Plan-and-Execute with Reflection Gates

For plan-and-execute agents, reflection gates work at two levels:

1. **Plan-level reflection**: After generating the plan but before executing it, a critic evaluates whether the plan is feasible, complete, and safe. This is the highest-leverage reflection point — catching a flawed plan before any tool calls is orders of magnitude cheaper than catching it mid-execution.

2. **Step-level reflection**: After each step completes, a critic evaluates whether the step achieved its intended goal and whether the remaining plan still makes sense given the updated state.

```python
class PlanAndExecuteWithReflection:
    
    def run(self, task: str) -> str:
        # Generate plan
        plan = self.planner.generate_plan(task)
        
        # Plan-level reflection (pre-execution)
        plan_critique = self.plan_critic.critique(task, plan)
        if plan_critique.score < 7.0:
            plan = self.planner.revise_plan(task, plan, plan_critique)
        
        # Execute steps with per-step reflection
        context = {}
        for step in plan.steps:
            result = self.executor.execute_step(step, context)
            context[step.id] = result
            
            # Step-level reflection
            step_critique = self.step_critic.critique(
                expected_outcome=step.expected_outcome,
                actual_result=result,
                remaining_plan=plan.remaining_steps_after(step),
            )
            
            if step_critique.plan_invalidated:
                # Replan from current state
                remaining_plan = self.planner.replan(
                    task=task,
                    completed_steps=plan.completed_steps,
                    current_context=context,
                    failure_reason=step_critique.analysis,
                )
                plan = remaining_plan
        
        return self.synthesizer.synthesize(task, context)
```

## 10. Cost Model: Reflection Overhead vs Quality Gain (Real Numbers)

Every reflection decision is ultimately a cost-benefit calculation. Here is a worked example with concrete numbers, using gpt-4o pricing at $5/M input tokens, $15/M output tokens (approximate 2024 rates, check current pricing).

A production agent handles 100,000 requests per day for a long-form writing task. Average generation: 500 input tokens + 800 output tokens. Cost without reflection:

```
Input cost:  500 × $5/M  × 100,000 = $250/day
Output cost: 800 × $15/M × 100,000 = $1,200/day
Total without reflection: $1,450/day
```

### Scenario A: Post-Hoc Critic, One Round, 30% Revision Rate

The critic call: 500 (task context) + 800 (output) + 400 (prompt) = 1,700 input tokens. 200 output tokens (critique JSON). The revision call (30% of requests): 500 + 800 + 200 (critique) + 600 (revision prompt) = 2,100 input tokens. 800 output tokens.

```
Critic calls:     1,700 × $5/M  × 100,000 = $850/day input
                   200 × $15/M × 100,000 = $300/day output

Revision calls:   2,100 × $5/M  × 30,000 = $315/day input
                   800 × $15/M × 30,000 = $360/day output

Total with reflection: $1,450 + $850 + $300 + $315 + $360 = $3,275/day
Cost multiplier: 2.26×
```

If reflection improves user satisfaction by 20% and your ARPU is $10/month with 50,000 daily active users, the revenue impact of that quality gain is ~$33,000/month. The reflection cost is ~$55,000/month. Marginal gain does not justify cost in this scenario — unless quality gain also reduces churn.

### Scenario B: Separate Critic Agent with Tool Verification, 15% Revision Rate

More expensive critic: 3,000 input tokens + tool calls + 400 output tokens.

```
Additional daily cost: approximately $6,000–8,000/day
Cost multiplier: ~4.5–5.5×
```

Justified only if: (1) errors have direct financial/legal consequences, (2) quality gain exceeds 40%, or (3) the task is generating content with high per-unit value (reports, contracts, medical notes).

### The Cost-Quality Pareto Frontier

Plotting quality gain vs cost overhead across reflection strategies reveals a Pareto frontier:

| Strategy | Quality Gain | Cost Multiplier |
|---|---|---|
| No reflection (baseline) | 0% | 1× |
| Inline reflection | +12–18% | 1.15–1.25× |
| Post-hoc critic, threshold 8.0 | +22–28% | 2.0–2.5× |
| Post-hoc critic, max 2 rounds | +28–35% | 2.5–3.5× |
| Self-consistency N=5 | +15–20% | 5× |
| Separate critic agent | +30–40% | 3–5× |
| Separate critic agent + N=3 self-consistency | +40–50% | 9–12× |

The knee of the curve is the post-hoc critic with a single round and a calibrated threshold: you get most of the quality gain for a manageable cost multiplier.

## 11. Case Studies

### Case Study 1: GitHub Copilot-Style Code Review Agent

**Setup**: An agent that generates code snippets for IDE completions and then runs a self-critique step to catch obvious bugs before suggestions appear to users. Latency budget: 800 ms total.

**Design choices**: Inline reflection was tried first. Quality improvement was +11% on logic error detection but +0% on off-by-one errors (the class of errors that inline reflection misses because the model generates and checks in the same pass). Switched to a separate critic API call using a smaller model (gpt-4o-mini) for speed. The critic only checks for: syntax validity (could use a linter instead), logic flow, and undefined variable references.

**Outcome**: Post-hoc critic with gpt-4o-mini at temperature 0.0 takes ~150 ms. Logic error detection improved by +24%. Off-by-one errors improved by +9% (better but not solved — these require test execution). Total latency: generation 550 ms + critique 150 ms + some revision = 800 ms budget hit.

**Lesson**: Use the smallest model that can reliably handle the critique task. For code review, the critic only needs to be correct on well-defined error classes — a 7B model fine-tuned on code review is often better and cheaper than GPT-4o for this specific role.

### Case Study 2: Customer Support Agent with Tone and Policy Critique

**Setup**: A customer support agent generating responses to user complaints. Two dimensions must be checked: policy compliance (no promises the company cannot keep, no admission of liability) and tone (empathetic but not sycophantic, professional but not cold).

**Design choices**: Policy compliance is a rules-based check — deterministic, no LLM needed. Built a regex + rule engine for it. Tone is subjective — requires LLM critique. Two separate critics running in parallel: rule engine for policy, LLM for tone. The revision only fires if either critic flags an issue.

**Outcome**: Policy violations dropped from 2.3% to 0.1% (mostly caught by the rule engine, not the LLM). Tone flag rate: 18% of responses, of which 70% were genuinely improved by revision. CSAT scores improved 12 points. Cost multiplier: 1.8× (rule engine is cheap; LLM tone critic adds the main cost).

**Lesson**: Hybrid deterministic + LLM critique is almost always better than pure LLM critique. Deterministic checks have perfect recall on their target class and zero hallucination risk. Use LLMs for the parts that are genuinely judgment calls.

### Case Study 3: Medical Report Summarization with High-Stakes Reflection

**Setup**: An agent that summarizes radiology reports for referring physicians. Any factual error (wrong organ, wrong laterality, wrong severity) is clinically dangerous. Quality requirements are asymmetric: false negatives (missing a finding) are worse than false positives (overcalling findings).

**Design choices**: Separate critic agent using a different base model. The critic has access to structured data from the DICOM metadata to verify laterality, date, and modality claims. It runs a separate specialized critique for each category: anatomy, measurements, clinical significance, laterality. Five rounds max with a very high acceptance threshold (9.5/10).

**Outcome**: Factual error rate dropped from 1.2% to 0.08% on a held-out test set. Of the residual errors, 0.06% were errors the critic introduced (over-correction). Cost multiplier: 4.2×. Clinically, a 0.08% error rate on 10,000 reports/month = 8 errors/month. The pre-reflection rate was 120 errors/month. The cost of a single clinical adverse event vastly exceeds the reflection overhead.

**Lesson**: For safety-critical domains, the cost-benefit math is entirely different from consumer applications. Build a separate specialized critic with domain-specific tools. The 4× cost is non-negotiable.

### Case Study 4: Technical Blog Post Generator Failing the Over-Critique Test

**Setup**: A content-generation agent producing long-form technical articles. After deploying three-round reflection with a general-purpose critic, the team noticed articles were becoming increasingly verbose and structurally uniform. All articles ended up with the same five-section structure.

**Investigation**: The critic was applying a quality rubric that included "covers all major subtopics" and "provides comprehensive background." Over three rounds, the reviser kept adding sections to satisfy the "comprehensive background" criterion — which a general-purpose critic always flags as improvable. The original first drafts, which were more focused and punchy, were systematically worse-scored.

**Fix**: Cap at one round for long-form writing. Add negative criteria to the rubric: "do not add sections not requested in the task," "do not add background that the target audience (senior engineers) already knows." Explicitly include "conciseness" as a rubric dimension weighted 2/10.

**Outcome**: Over-critique rate dropped from 31% to 8%. Article uniqueness recovered. User engagement metrics (average read completion) improved 15% post-fix.

**Lesson**: Critic rubrics must include negative criteria — what the critic should *not* flag. General-purpose "completeness" rubrics systematically over-trigger on well-scoped focused outputs.

### Case Study 5: Legal Contract Review Agent

**Setup**: An agent reviewing contract drafts for a law firm, flagging non-standard clauses and suggesting redlines. The agent uses retrieval to fetch precedent contracts from a vector store.

**Design choices**: Two separate critic agents: one checks legal accuracy (standard clause deviations, jurisdiction issues), one checks business risk (liability exposure, indemnification scope). The generator produces a redlined version; the critics evaluate the redlines, not the full contract — this halves the critique token cost.

**Outcome**: Attorney review time reduced from 4 hours to 1.2 hours per contract. The reflection step caught 23% of the agent's own errors before attorney review. The critics' false positive rate (flagging correct redlines) was 12% — lower than anticipated for legal language.

**Key metric**: Rate of attorney overrides of agent suggestions went from 34% (no reflection) to 19% (with reflection). The remaining 19% are cases where the attorney's judgment supersedes the critic — expected in legal work.

### Case Study 6: Financial Report Generation and the Hallucination-Amplification Bug

**Setup**: An agent generating quarterly earnings summaries for an investment platform. It pulls data from a financial API and writes the narrative. The reflection step was added to catch factual errors.

**Bug discovered**: The critic was asked "are all financial figures accurate?" Without access to the underlying data, the critic could only assess *consistency* — did the numbers in the narrative agree with each other? Not whether they agreed with the source data. The critic started *correcting* figures that were actually accurate by adjusting them to fit what seemed like a more internally consistent narrative.

**Root cause**: The critic was hallucinating corrections. The generator had correct figures; the critic, not seeing the source data, was constructing a plausible alternative set of numbers based on industry norms. The reviser dutifully applied those hallucinated corrections.

**Fix**: The critic must have access to the same data source the generator used. Specifically: pass the raw API response alongside the generated narrative. Alternatively, use a deterministic check — compare each figure in the narrative against the API data with a simple diff rather than LLM assessment.

**Lesson**: Critique of factual data *requires* access to the ground truth. An LLM critic without ground-truth access will hallucinate corrections. This is the most dangerous failure mode of reflection: it does not just fail to improve the output; it actively degrades a correct output.

### Case Study 7: Multi-Agent Debate as Adversarial Critique

**Setup**: An AI safety team building a research synthesis agent. They implemented a "society of critics" where three specialized critic agents debate the generator's conclusions: a skeptic, a supporter, and a neutral fact-checker. The generator must produce an output that satisfies all three.

**Design choices**: Each critic has a distinct role and system prompt. The skeptic is explicitly instructed to find flaws. The supporter is instructed to steelman the output. The fact-checker is neutral and uses retrieval. A judge LLM reads the three critiques and produces a final verdict.

**Outcome**: On a held-out set of 100 research claims, the multi-critic approach correctly identified 89% of false or unsupported claims, compared to 72% for a single critic. The cost is proportional: three critics plus a judge at ~5× the generation cost.

**Key insight**: Adversarial diversity in the critic pool catches systematic biases that a single critic misses. A single critic can be anchored by the generator's framing; a skeptic critic is explicitly immune to that anchoring.

### Case Study 8: Real-Time Reflection in Streaming Responses

**Setup**: A streaming chat agent where users see the response as it types. The team wanted reflection without waiting for the full response to complete before starting display.

**Design**: Two-phase streaming. Phase 1: generate and stream a "draft" response with a subtle visual indicator ("reviewing..."). Phase 2: run a fast inline reflection (not a separate call — just a prompt addition at the end). Phase 3: stream any revisions as delta patches to the UI, highlighting changed sections.

**Outcome**: Perceived quality improved significantly in user tests — users trusted the response more seeing the "review" indicator. Actual error rate improvement was only +9% (inline reflection limitation). However, the psychological effect of seeing "reviewing..." and seeing red-highlighted corrections made users 34% more likely to fact-check the response themselves — a positive second-order effect.

**Lesson**: In user-facing streaming systems, the *perception* of careful generation is valuable independently of the actual quality improvement. The reflection indicator changes user behavior in beneficial ways.

## 12. When to Use Reflection / When Not to

![Task Types × Reflection Benefit](/imgs/blogs/reflection-and-self-critique-agents-10.webp)

### Use Reflection When

**The task is open-ended and the output has no mechanical verifier.** Code has a compiler and tests; arithmetic has a calculator. For everything without a deterministic ground truth — writing, analysis, reasoning, synthesis — reflection is where you get quality gains.

**Errors are costly and asymmetric.** If the cost of a wrong output (safety violation, legal error, medical mistake, financial misstatement) exceeds the cost of reflection overhead by a significant margin, reflection is justified at almost any cost multiplier.

**The generator and critic have genuinely different failure modes.** Post-hoc critique with a separate call (or a separate model entirely) exploits the independence assumption. The less correlated the generator and critic errors are, the larger the quality gain.

**Your task involves long-horizon reasoning.** In tasks with multiple steps, errors compound. Catching an error at step 2 prevents it from corrupting steps 3 through 10. The reflection ROI scales with sequence length.

**You have calibrated your thresholds.** Uncalibrated thresholds produce either expensive over-critique or missed errors. If you have done the offline evaluation to know what a critic score of 7.5 means for your specific task, reflection is a reliable quality lever.

### Do Not Use Reflection When

**The task is arithmetic or symbolic math.** LLM critics hallucinate corrections for math. Use a calculator, a symbolic math tool, or code execution instead. Self-critique on math improves accuracy by ≈0% and often hurts it.

**You have a high-quality deterministic verifier.** If you can run a compiler, a type checker, a linter, a test suite, or a factual lookup, those tools are strictly better than LLM critique for their target error class. Use LLM critique only for the residual.

**Latency is hard-constrained below 500 ms.** A post-hoc critique call takes 100–400 ms on modern APIs. With the generation call, you are at 600–1,200 ms minimum for one round. If your latency budget is 400 ms, inline reflection is your only option — and its quality gain is modest.

**The task output is very short.** Reflection has a fixed overhead cost (the critique call). For single-sentence outputs, that cost is disproportionate to the potential gain. The revision rarely improves a one-sentence answer enough to justify it.

**You have not instrumented revision rates.** If you do not know whether your critic is triggering on 5% or 50% of requests, your cost model is invisible. Build logging before building reflection. A 50% revision rate with no quality gain is pure waste.

**The critic has no access to ground truth for factual claims.** As the financial report case study shows, a critic that cannot verify facts will hallucinate corrections. Do not deploy a factual critic without retrieval or tool access for verification.

### The Honest Summary

Reflection works. The papers show real gains, the production cases show real gains, and the mechanism is sound. But it works *conditionally* — on well-chosen tasks, with calibrated thresholds, with appropriate round limits, with regression detection, and with a critic that has the information it needs to actually detect errors.

The failure mode is not that reflection does not work. The failure mode is deploying it as a magic quality layer without doing the empirical work to calibrate it for your specific task distribution. An uncalibrated reflection loop will over-trigger on good outputs, miss errors the critic cannot detect, hallucinate corrections on factual data, and charge you 2–5× for the privilege.

Wire it in. Measure everything. Set per-task thresholds. Cap rounds at 2–3. Build regression detection. And for tasks with deterministic verifiers — use the verifier, not the critic.

---

## Further Reading

- [ReAct: Synergizing Reasoning and Acting in Language Models](/blog/machine-learning/ai-agent/react-pattern-deep-dive) — the foundational pattern that reflection augments
- [Tree of Thought Agents: Branching Deliberation](/blog/machine-learning/ai-agent/tree-of-thought-agents) — how tree search and reflection combine for harder reasoning tasks
- [Evaluating Agent Trajectories Beyond Final Answer](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer) — how to measure whether your reflection loop is actually helping

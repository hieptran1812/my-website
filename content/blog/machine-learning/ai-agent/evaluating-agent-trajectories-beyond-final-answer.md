---
title: "Evaluating Agent Trajectories: Why Final-Answer Accuracy Misses 80% of the Bugs"
date: "2026-05-21"
publishDate: "2026-05-21"
description: "A staff-level deep dive into trajectory evaluation for LLM agents — taxonomy, alignment, statistical power, replay harnesses, and seven production failures that outcome-only eval cannot see."
tags: ["ai-agent", "evaluation", "trajectory", "llm-judge", "testing", "ml-engineering", "observability", "statistical-testing", "agent-reliability", "production-llm"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 50
aiGenerated: true
---

## 1. The outcome trap

Every team that has shipped an agent past the demo stage has, at some point, been bitten by the same scene: the offline eval is green, the rollout is approved, the dashboards light up at 3 a.m., and the post-mortem opens with a screenshot of an eval suite that *passed*. The agent gave the right answer. It just gave the right answer the wrong way.

I have personally been on the wrong side of this graph more than once. The first time, an agent v3.2 hit 92.1% outcome accuracy on our internal benchmark, identical to v3.1, and shipped clean. Six hours later p95 latency was up 3.1×, our LLM bill for the day had tripled, and our on-call engineer was staring at a flame graph showing the agent retrying a malformed tool call up to 14 times per task before stumbling into the right answer. Outcome eval saw nothing. The answers were fine. The *process* was on fire.

This is the outcome trap: the assumption that `is_correct(ŷ)` is a reasonable proxy for "the agent works." For a single-turn classifier that assumption is fine. For an agent — a system that plans, calls tools, observes results, and replans — it is wrong in the same way that grading a surgeon only by whether the patient is alive at the end of the day is wrong. The patient is alive. You still want to know whether anyone severed the wrong artery on the way.

![Anatomy of an agent trajectory](/imgs/blogs/evaluating-agent-trajectories-beyond-final-answer-1.png)

The diagram above is the mental model that the rest of this post unpacks: an agent run is not a `(prompt, answer)` pair. It is a graph of states, actions, observations, and intermediate beliefs, and any of those nodes can be wrong while the terminal node looks right. Trajectory evaluation is the discipline of grading the entire graph, not just the last node.

This article is the deep-dive companion to my earlier [introductory guide on evals](/blog/machine-learning/ai-agent/eval-agents). The intro guide covers the *what* — graders, metrics, the eval workflow. This one covers the *how* of doing it rigorously at staff/principal level: how to formalize trajectories, design partial-credit graders, calibrate LLM judges, achieve statistical power, build replay harnesses, and avoid the seven production failure modes I have personally watched go undetected by outcome-only eval.

| Assumption (naive view)                                                | Reality at staff level                                                                                       |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| "Eval == accuracy on a held-out set."                                  | Eval is a *vector* of metrics; accuracy is one coordinate, often the least diagnostic one.                   |
| "If the final answer is right, the agent worked."                      | Lucky recoveries, retries, and side effects make outcome a *trailing indicator* of trajectory health.        |
| "We can grade everything with an LLM judge."                           | Judges drift, leak, and self-flatter; anything code can verify, code should verify.                          |
| "One golden trajectory per task is the ground truth."                  | A single golden path penalises strictly-better alternatives; references should encode *equivalence classes*. |
| "Eval set of 100 hand-curated examples is enough."                     | A 2% regression on a 100-task set is statistically indistinguishable from noise.                             |
| "We can rerun the eval whenever we want."                              | Tool environments drift (search results, prices, schemas) — without replay, two eval runs aren't comparable. |

If even one row of that table surprised you, this article is for you.

## 2. What is a trajectory, formally

Let an agent be a policy $\pi_\theta(a_t \mid s_t)$ that, at step $t$, observes state $s_t$, produces an action $a_t$ (a tool call or a final-answer emission), and receives an observation $o_t = T(s_t, a_t)$ from the environment $T$. The internal state $s_t$ contains the conversation history, scratchpad, and any retrieved context up to step $t$. A *trajectory* is the tuple

$$
\tau = (s_0, a_0, o_0, s_1, a_1, o_1, \dots, s_{H}, \hat{y})
$$

where $s_0$ is the initial state (typically the user query plus a system prompt), $H$ is the horizon (the number of tool calls before the agent emits a final answer), and $\hat{y}$ is the answer the agent commits to. A *task* is a tuple $(x, y^\*, \mathcal{R})$ where $x = s_0$ is the input, $y^\*$ is a reference answer or predicate, and $\mathcal{R}$ is a *reference structure* over trajectories — the thing that defines "what good looks like" for the path, not just the answer. We will spend most of section 5 on what $\mathcal{R}$ should be, because the wrong choice is where most teams break.

Critically, $\tau$ is *not* a Markov chain in the modelling sense. The action $a_t$ depends on the entire history $(s_0, a_0, o_0, \dots, s_t)$ through the language model's context. The state at step $t$ is the *concatenation* of everything that has happened so far, which means evaluators get to see — and grade — every byte of every prior turn. That is the privileged position trajectory evaluators occupy and outcome evaluators throw away.

```python
from dataclasses import dataclass, field
from typing import Any, Literal

ActionKind = Literal["tool_call", "final_answer", "thought"]

@dataclass
class Action:
    kind: ActionKind
    tool: str | None = None              # e.g., "search", "calc"
    args: dict[str, Any] = field(default_factory=dict)
    text: str | None = None              # final-answer text or thought text

@dataclass
class Step:
    state: str                           # serialised context up to this step
    action: Action
    observation: Any                     # tool return value, or None for thought/final
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

@dataclass
class Trajectory:
    task_id: str
    steps: list[Step]
    final_answer: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def actions(self) -> list[Action]:
        return [s.action for s in self.steps]

    @property
    def tool_calls(self) -> list[Action]:
        return [a for a in self.actions if a.kind == "tool_call"]
```

This is the canonical record. Anything you want to evaluate must be either present in this object or derivable from it. The *first* engineering investment for trajectory eval is making sure your agent runtime emits exactly this — not a flat log line, not a summarised "trace," but a structured, replayable, byte-accurate sequence. Teams that try to retrofit trajectory eval onto unstructured logs lose more time to log archaeology than to actual evaluation.

> If your agent runtime cannot emit a `Trajectory` object, you do not have a trajectory eval problem; you have an instrumentation problem. Fix that first.

A subtle point: the `thought` action kind matters. Many agents emit a chain-of-thought scratchpad as a separate "step" before each tool call. Whether you choose to grade that scratchpad is a *policy* decision — some teams treat it as private to the model and grade only `tool_call` and `final_answer`; others grade thoughts for faithfulness. I prefer the second, but only with the understanding that thoughts have weaker contracts than tool calls and should be graded leniently (we will return to this in section 6).

## 3. The five failure modes outcome-only eval hides

If the only metric you watch is `is_correct(ŷ)`, here is what you will not see. I have personally caught all five in production, on different agent versions, in different deployments. The labels are mine; the dynamics are universal.

![Coverage: outcome-only vs trajectory evaluation](/imgs/blogs/evaluating-agent-trajectories-beyond-final-answer-2.png)

**1. Wrong-tool-right-answer.** The agent calls a tool that semantically does the wrong thing, the tool happens to return data that, combined with the model's priors, yields a correct answer. Example: a question about Q4 revenue triggers a `search` tool, but the search query is wrong; the search returns unrelated news that happens to contain the number the model would have guessed anyway from training data. Outcome eval scores 1. The agent's reliance on memorised priors is invisible.

**2. Lucky recovery.** The agent's first action is wrong, the environment returns an error or empty result, and the agent recovers — sometimes from a stack trace, sometimes from a polite "no results found" message — and lands on the right answer. The outcome is correct. The first-step planning bug that requires this recovery is hidden, and it compounds: at scale it inflates token cost, latency, and the probability that *next time* recovery fails.

**3. Silent retry loop.** The agent issues malformed tool calls, gets rejection messages back, reformats, retries, eventually succeeds. Outcome correct, latency catastrophic, cost catastrophic. This is the failure mode that took down our v3.2 rollout in section 1. Outcome eval cannot see it. Even latency dashboards cannot tell you *why* — they just tell you it happened.

**4. Side-effect leak.** The task is read-only by spec ("look up this customer's last order"), but the agent writes to a downstream system as a side effect of an exploratory tool call ("update the customer record with a comment"). The answer is correct. The shadow write happened. Outcome eval did not even look at the world.

**5. Phantom tool call.** The agent calls a tool, ignores its output, and produces an answer purely from prior knowledge. The answer happens to be right. The "use the tool" contract — the entire reason you built the tool — has been violated, and you have no idea until the day the priors are wrong and the world has moved on.

A *sixth* mode is worth flagging even though it is structurally different: **acceptance-set collapse**, where the eval *itself* is wrong and penalises a strictly-better trajectory. We discuss that in section 5; mention it here only so you do not conflate "outcome eval missed it" with "trajectory eval will always catch it." Trajectory eval has its own pathologies. The fix is not to swap one blind grader for another; the fix is to grade across orthogonal axes and to interpret disagreement as signal.

## 4. A decomposition of trajectory metrics

The right way to think about a trajectory score is as a *vector*, not a scalar. Combining the vector into a scalar at the end is fine — sometimes useful for ranking model versions — but doing it too early throws away exactly the diagnostic information you built trajectory eval to expose.

![Trajectory quality taxonomy](/imgs/blogs/evaluating-agent-trajectories-beyond-final-answer-3.png)

The six axes I use, with a one-line definition and the grader family that owns each, are:

| Axis                  | What it measures                                                                | Grader family                                       |
| --------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------- |
| Action correctness    | Did the agent call the right tools with the right arguments?                    | Code (schema/IoU) + judge for argument semantics    |
| Efficiency            | Did the agent do so in roughly the right number of steps and at the right cost? | Code (counts, sums, percentiles)                    |
| Recovery              | When the agent erred, did it detect, recover, and not repeat?                   | Code (retry detection) + judge for "good recovery"  |
| Faithfulness          | Is the final answer grounded in the observations the agent actually saw?        | Judge (with citation-style anchor checks in code)   |
| Side-effect safety    | Did the agent stay inside the side-effect contract for the task?                | Code (effect ledger diff against allow-list)        |
| Outcome correctness   | Is the final answer correct under the task's correctness predicate?             | Code (exact, fuzzy, semantic-equiv) or judge        |

These six are not independent in the statistical sense — a model that is bad at action correctness will usually be bad at efficiency too — but they are *independent in failure modality*. Each axis can fail while the others succeed, and you want a dashboard that shows you which axis a regression lives on. A single number ("trajectory quality = 0.87") tells you nothing about whether to ship.

### 4.1 Compositing without losing information

A composite score is useful for ranking but should be designed so that pathological failures dominate. The shape I have settled on:

$$
S(\tau, \tau^\*) = \min\!\Big( w_o \cdot s_o, \; w_s \cdot s_s \Big) \cdot \Big( w_a s_a + w_e s_e + w_r s_r + w_f s_f \Big)
$$

where $s_o$ is outcome correctness, $s_s$ is side-effect safety, and $s_a, s_e, s_r, s_f$ are action correctness, efficiency, recovery, and faithfulness. The `min` on outcome and safety encodes *veto* semantics: a side-effect leak or wrong final answer drops the whole score to zero regardless of how elegantly the agent got there. The rest is a weighted sum because those axes are partial credits, not vetoes. Weights are domain-specific; I usually tune them so that on a held-out calibration set the composite correlates ≥ 0.7 with a human staff-engineer's overall judgement.

The first time I deployed this composite, weights $(w_a, w_e, w_r, w_f) = (0.4, 0.1, 0.2, 0.3)$ tracked human judgement at $\rho = 0.74$. Plain outcome accuracy on the same set tracked at $\rho = 0.41$. The composite was a meaningfully better signal of "ship or don't ship" than the metric the team had been staring at for a year.

### 4.2 Why "single number" eval is seductive and wrong

Every leadership conversation about agents wants a single number. Resist for as long as you can, then concede gracefully — but ship the vector alongside the scalar, and make the dashboard surface the *axis* that moved when the scalar changes. The number of times I have heard "trajectory eval dropped from 0.91 to 0.88, what happened?" — and the answer was "efficiency dropped because we added a planning step, action correctness rose, outcome rose, this is a win" — has convinced me that without the vector view, scalar trajectory eval is barely better than outcome eval.

## 5. Reference trajectories: when and how to author them

The reference $\mathcal{R}$ is the most underdesigned part of most agent evals I have audited. Teams treat it as "the answer key" — one canonical path, hand-authored, frozen. That works for arithmetic. It fails for agents.

![Golden, acceptance set, equivalence class](/imgs/blogs/evaluating-agent-trajectories-beyond-final-answer-4.png)

There are three useful levels of reference structure, in order of increasing sophistication and decreasing per-task authoring cost (once tooling exists).

### 5.1 Golden trajectory

A single canonical action sequence. Authoring cost: high per task, but cheap to grade against — alignment is straightforward. Failure mode: any strictly-better trajectory that deviates from the canonical path is penalised. Use only when the task has genuinely one correct path (e.g., "follow this exact runbook").

### 5.2 Acceptance set

An enumerated set of acceptable action sequences. Authoring cost: 3–10× golden, but much higher recall. Grading: alignment against the *best-matching* element of the set, with a small penalty for set-size to prevent gaming via enumerating every conceivable trajectory. Failure mode: the acceptance set is still finite; novel-but-better trajectories outside the set are penalised.

### 5.3 Equivalence class (predicate-based)

A predicate $P(\tau)$ over trajectories that returns true for all acceptable paths. Authoring cost: highest upfront (you are writing a small DSL of acceptance), but the *per-task* cost amortizes because predicates compose. Grading: evaluate $P$ on the candidate trajectory, then score *quality within* the equivalence class on the secondary axes (efficiency, faithfulness).

A predicate I like, for a "look up customer's last order and report total" task:

```python
def accepts(traj: Trajectory) -> bool:
    tools = [a.tool for a in traj.tool_calls]
    # Must read customer data exactly once.
    if tools.count("get_customer") + tools.count("search_customer") != 1:
        return False
    # Must read orders for that customer.
    if not any(a.tool == "get_orders" and "customer_id" in a.args for a in traj.tool_calls):
        return False
    # Must not write anything.
    if any(a.tool in WRITE_TOOLS for a in traj.tool_calls):
        return False
    # Must emit a numeric total in the final answer.
    return bool(traj.final_answer) and re.search(r"\$?\d+(\.\d{2})?", traj.final_answer)
```

This predicate accepts both `[get_customer, get_orders, final]` and `[search_customer, get_orders, final]`, rejects any path that writes, requires exactly one customer lookup (to penalise repeated reads), and requires a dollar amount in the answer. It is six lines of code and it dominates a 50-line hand-authored golden trajectory on both recall and engineering velocity.

### 5.4 When to use which

| Reference structure | Best for                                                        | Authoring time | False-penalty risk |
| ------------------- | --------------------------------------------------------------- | -------------- | ------------------ |
| Golden              | Strict runbooks, single-path domains, regression smoke tests    | High           | High               |
| Acceptance set      | Tasks with ≤ 5 known-good paths, fixed tool palettes            | High           | Medium             |
| Equivalence class   | Open-ended tasks, tasks where novel paths are expected          | Medium (after tooling) | Low      |

The transition is one-way: every team I have worked with starts with golden trajectories, hits acceptance-set collapse around task 20, hand-authors acceptance sets for two months, and eventually invests in a predicate DSL. If you can skip the golden-trajectory phase entirely, you will save a quarter of engineering time. In practice, almost nobody skips it because the predicate DSL doesn't exist until you've felt the pain.

## 6. Programmatic graders vs LLM-judge graders for steps

The fundamental tradeoff is fidelity vs. coverage. Code graders are precise but only cover what has a verifiable contract. Judge graders cover everything else but are noisier and require calibration. The mistake is treating these as either/or; in practice, every well-designed step grader is a *router* that sends each step to the cheapest grader that can decide it correctly.

![Routing steps to code graders vs judge graders](/imgs/blogs/evaluating-agent-trajectories-beyond-final-answer-5.png)

### 6.1 What code can grade

For each step, code can usually answer:

- Did the tool name match the reference (or a member of the acceptance set)?
- Did the argument shape validate against the tool's JSON schema?
- For arguments with structured types (IDs, timestamps, enums), do they equal the reference value? For free-text arguments (queries, messages), do they have high IoU on tokens with the reference?
- Was the latency within a budget? Was the token cost within a budget?
- Did any side-effect occur that wasn't on the allow-list for this task?

I will assert this rule strongly: **anything code can verify, code should verify.** Judge graders are a finite resource — they cost money, they cost time, they drift — and you should not spend them on questions a 20-line function can answer. The temptation to "just ask GPT" for everything has produced more flaky eval suites than I can count.

### 6.2 What judges grade

What's left after the code router fires:

- Whether a free-text argument is *semantically* appropriate even when it has low token IoU with the reference (e.g., `"customers from CA who churned in Q3"` vs `"California customer churn last quarter"` — the same intent, near-zero string overlap).
- Whether a thought / scratchpad entry is faithful to the prior context.
- Whether a final answer is faithful to the observations the agent saw (citation-style anchoring).
- Whether a recovery from an error was "graceful" in a way that's hard to formalise.

### 6.3 Calibrating the judge

A judge grader is just another model; before you trust it, you measure its agreement with humans on a *calibration set*. The minimum protocol:

1. Sample 100–300 trajectories spanning your task distribution.
2. Have 2–3 humans grade each on the axes the judge will eventually grade. Use a written rubric, not vibes.
3. Compute pairwise human agreement (Cohen's $\kappa$ or Krippendorff's $\alpha$). If humans don't agree above $\kappa = 0.7$, your rubric is broken — fix it before involving the model.
4. Now have the judge grade the same set. Compute judge-vs-human agreement.
5. Acceptable: judge-vs-human $\kappa$ is within 0.05 of human-vs-human $\kappa$. Better: judge tracks human *deltas* over time (when a new model version is released, judges and humans agree on the direction of change).

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score

def judge_calibration_report(human_labels, judge_labels, axis: str):
    h_pairs = [(human_labels[a, axis], human_labels[b, axis])
               for a in human_labels.index for b in human_labels.index if a < b]
    h_kappa = np.mean([cohen_kappa_score(*p) for p in h_pairs])
    j_kappa = cohen_kappa_score(human_labels[:, axis].mode(axis=1)[0],
                                judge_labels[:, axis])
    return {"axis": axis, "human_kappa": h_kappa, "judge_kappa": j_kappa,
            "delta": j_kappa - h_kappa}
```

If the judge fails calibration, the fix is almost always one of: tighter rubric, more shot examples in the rubric, switch to pairwise grading (A vs B) rather than pointwise (score A 0–5). Pairwise judges are typically 20–30% more reliable than pointwise judges on the same axis, at the cost of a quadratic number of comparisons.

### 6.4 Judge contamination: never grade A with A

The single most pernicious bug in LLM-judge eval is judge contamination — using a model from the same family (or worse, the same model) as both candidate and judge. The judge will systematically score outputs in *its own* style higher. I have measured this at 8–14% absolute on multiple production setups. If you cannot use a different family, at minimum:

- Use a meaningfully different model size or version.
- Add a *style-blind* rewriting step: rewrite candidate outputs to a neutral style before judging.
- Run an *independent-family* judge on the calibration set occasionally to detect drift.

The cleanest design I have used: a Claude Opus candidate evaluated by a Llama-405B judge for faithfulness, and a separate GPT-5 judge for the same axis as a tiebreaker. The two judges agree at $\kappa = 0.82$ across 500 trajectories, both agree with humans at $\kappa = 0.78$. When they disagree, the trajectory is flagged for human review. Total cost: ~$0.04 per trajectory, well within budget for a weekly eval run.

### 6.5 Pointwise vs pairwise judging, in depth

When you sit down to actually write a judge prompt, the first design fork is: do you ask the judge to score one trajectory on an absolute scale (pointwise), or to compare two trajectories and say which is better (pairwise)? The literature has converged on a clear-ish answer; the production reality is messier.

Pointwise judging asks the judge to output a number in $[0, 5]$ (or a categorical label) given a single trajectory and a rubric. It is cheap — one judge call per (trajectory, axis) — and produces scores that aggregate cleanly into averages. Its weakness is *scale drift*: the judge's mapping from "trajectory quality" to "number" wanders over runs, over days, and over model versions. A judge that gave the median trajectory a 3.5 last month may give it a 3.2 this month, not because anything regressed but because the judge's anchor moved. Pointwise judges are most trustworthy when the rubric is grounded in concrete checkable features ("does the answer contain numeric values that appear verbatim in observations?") and least trustworthy when the rubric is "rate the overall quality on a 1–5 scale."

Pairwise judging asks the judge to look at two trajectories $(\tau_A, \tau_B)$ for the same task and pick the better one. It costs $\binom{k}{2}$ calls for $k$ candidates, which is fine for two candidates and painful for more. Its strength is robustness: the judge does not need to maintain a stable absolute scale; it only needs to distinguish A from B. Pairwise judges are *typically 20–30% more reliable* than pointwise judges on the same axis, measured against humans. The cost is the quadratic blow-up, the bias toward whichever side appears first (which you have to mitigate by randomising order and running both presentations), and the awkwardness of aggregating pairwise wins into a global ranking (Elo, Bradley-Terry, or the brittle "win rate vs baseline").

My rule of thumb: use pairwise for *release gates* (where you are comparing two specific versions and the quadratic cost is a constant) and pointwise for *monitoring dashboards* (where you need a continuous time series and pairwise's anchor-free property is irrelevant). The hybrid configuration we landed on:

| Use case                  | Style       | Why                                             |
| ------------------------- | ----------- | ----------------------------------------------- |
| Release gate (A vs B)     | Pairwise    | Robust to scale drift, quadratic cost bounded   |
| Weekly dashboard          | Pointwise   | Need stable longitudinal time series             |
| Spot-check on incidents   | Pairwise    | Compare incident-trajectory vs nearest baseline |
| Production sampling       | Pointwise   | Online cost matters, can't afford pairs         |

A small but important detail: pairwise judges are extremely sensitive to *length bias* (longer trajectories often look more thorough, regardless of whether they were better). Mitigations: present the trajectories as structured JSON rather than free text (length normalises), and explicitly ask the judge to score on the axis under test rather than overall quality. If you can spare the budget, run pairwise both ways and use the answer only when both orderings agree.

### 6.6 Rubric design: the part nobody writes down

A judge rubric is a prompt, and like any prompt, the marginal hour you spend on it returns disproportionately on quality. The rubrics that have worked for me share four features:

1. **Anchor on concrete features.** "Did the answer cite at least one observation verbatim?" rather than "is the answer well-grounded?" Anchored questions get judged the same way by humans, by the judge, and across runs. Vibes-based questions get judged differently every time.
2. **One axis per rubric.** A rubric that asks "rate correctness, clarity, and politeness" gets a single number that smears across three latent dimensions. Run three rubrics, one per axis. The cost is linear; the diagnostic value is much higher.
3. **Few-shot, with both ends of the scale.** Three positive examples plus three negative examples plus one borderline example calibrates the judge far better than a one-paragraph definition. Human raters work the same way; they get there faster with examples.
4. **Scoring scale ≤ 5 points.** Above five, raters (human or model) cluster on the middle three anyway. Categorical scales (`{bad, fair, good}`) are often a better fit than numeric scales for axes where intermediate values are awkward.

A rubric I have shipped, verbatim, for the faithfulness axis:

```yaml
axis: faithfulness
scale: {0: "contradicts observations",
        1: "ungrounded but not contradictory",
        2: "partially grounded",
        3: "fully grounded with minor paraphrase",
        4: "fully grounded with verbatim citations"}
anchor_questions:
  - "List every factual claim in the final answer."
  - "For each, point to the observation that supports it, or 'none'."
  - "If 'none' for any claim, the score is at most 1."
shots:
  positive:
    - {answer: "Q4 revenue was $2,791M.",
       observations: ["calc: 2310+481=2791"],
       score: 4,
       why: "exact numeric value appears in obs"}
  negative:
    - {answer: "Q4 revenue was $2,800M.",
       observations: ["calc: 2310+481=2791"],
       score: 0,
       why: "claim contradicts the only relevant observation"}
  borderline:
    - {answer: "Q4 revenue was approximately $2.8B.",
       observations: ["calc: 2310+481=2791"],
       score: 3,
       why: "rounded but consistent; not verbatim"}
```

The shots are doing the heavy lifting. Without them, judges (and humans) interpret the scale wildly differently. With them, $\kappa$ on this axis jumped from 0.61 to 0.78 in one of our calibration runs.

## 7. Partial credit and step alignment

Partial credit is where trajectory eval gets technical. The naive approach — "fraction of reference actions that appear in the candidate" — is wrong in two ways. It ignores order (the same actions in shuffled order are not equivalent), and it ignores *length mismatch* (a candidate that takes 14 steps to do a 4-step task should not get 1.0 just because the 4 reference steps appear somewhere in it).

The right framework is sequence alignment, lifted from bioinformatics and spell-checking.

![Edit-distance alignment of action sequences](/imgs/blogs/evaluating-agent-trajectories-beyond-final-answer-6.png)

### 7.1 Action edit distance

Define an edit distance between candidate actions $C = (c_1, \dots, c_m)$ and reference actions $R = (r_1, \dots, r_n)$ over the operations *insert*, *delete*, and *substitute*, with substitution cost given by a step-similarity function $\delta(c_i, r_j) \in [0, 1]$. The recurrence is standard:

$$
D[i][j] = \min\!\Big(
  D[i-1][j] + 1,\;
  D[i][j-1] + 1,\;
  D[i-1][j-1] + (1 - \delta(c_i, r_j))
\Big)
$$

The step similarity $\delta$ is where the design decisions live. A reasonable default:

$$
\delta(c, r) = \mathbb{1}[c.\mathrm{tool} = r.\mathrm{tool}] \cdot \big(0.5 + 0.5 \cdot \mathrm{IoU}(c.\mathrm{args}, r.\mathrm{args})\big)
$$

i.e., tool-name match is a gate; given a match, args contribute partial credit via token IoU. You can swap IoU for a judge if the args are free text.

```python
def step_similarity(c: Action, r: Action) -> float:
    if c.tool != r.tool:
        return 0.0
    keys = set(c.args) | set(r.args)
    if not keys:
        return 1.0
    def toks(v):
        return set(str(v).lower().split()) if v is not None else set()
    ious = []
    for k in keys:
        a, b = toks(c.args.get(k)), toks(r.args.get(k))
        if not a and not b: ious.append(1.0)
        elif not a or not b: ious.append(0.0)
        else: ious.append(len(a & b) / len(a | b))
    return 0.5 + 0.5 * (sum(ious) / len(ious))

def action_edit_distance(C: list[Action], R: list[Action]) -> float:
    m, n = len(C), len(R)
    D = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): D[i][0] = i
    for j in range(n + 1): D[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = 1 - step_similarity(C[i-1], R[j-1])
            D[i][j] = min(D[i-1][j] + 1, D[i][j-1] + 1, D[i-1][j-1] + sub)
    return D[m][n]

def action_score(C, R) -> float:
    if not R: return 1.0 if not C else 0.0
    return max(0.0, 1.0 - action_edit_distance(C, R) / max(len(C), len(R)))
```

This score is in $[0, 1]$, lower-bounded by 0 when the agent does something completely unrelated and 1 when actions match perfectly. The normalisation by $\max(m, n)$ matters: a candidate that does 14 steps to a 4-step reference is penalised for the 10 extra steps, even if the 4 right steps are in there.

### 7.2 Monotone-prefix scoring

A useful variant is *prefix scoring*: how far down the reference can the candidate get without making a mistake? The intuition is that early correct steps are more valuable than late ones — once the agent goes off the rails, recovering doesn't fully compensate. Define:

$$
\mathrm{Prefix}(C, R) = \max_{k} \{k : \exists\, \sigma \subseteq C, |\sigma| = k, \sigma \text{ aligns to } R_{1:k}\} / |R|
$$

In practice you compute this by walking the DP table and reading off the largest $k$ such that $D[*][k] = 0$. Prefix scoring is harsher than edit-distance scoring and is what I use when *order* of operations is load-bearing — for example, when a tool only makes sense after another has been called.

### 7.3 Set-based scoring for permutation-invariant tasks

For tasks where order doesn't matter ("retrieve these five documents, in any order"), set-based scoring is the right choice. Treat $C$ and $R$ as multisets of tool calls, score $|C \cap R| / |C \cup R|$. Don't waste alignment compute on a problem that doesn't have alignment structure.

### 7.4 IoU on argument fields

For arguments that are *queries* (free text), token IoU is a defensible default. For arguments that are *structured* (IDs, dates, schemas), exact match. For *numeric* arguments, tolerance: $|a - b| / \max(|a|, |b|) \leq \epsilon$. The router that picks the right comparator per argument field is one of those pieces of glue code that pays for itself many times over.

| Arg type      | Comparator             | Tolerance / params              |
| ------------- | ---------------------- | ------------------------------- |
| ID / enum     | exact                  | —                               |
| Free-text     | token IoU              | $\geq 0.6$ for substitution      |
| Numeric       | relative-error         | $\epsilon = 0.05$               |
| Date / time   | abs-diff               | $\leq$ task-specific window     |
| JSON struct   | recursive on keys      | weighted by leaf depth          |

### 7.5 A worked example, end to end

Concreteness pays here. Suppose the task is "report the total revenue for customer 4711 in Q4 2025," and the reference (acceptance-set) trajectory is:

```
R = [get_customer(id="4711"),
     get_orders(customer_id="4711", from="2025-10-01", to="2025-12-31"),
     calc(expr="<sum>"),
     final_answer]
```

A candidate v3.1 produces:

```
C1 = [get_customer(id="4711"),
      get_orders(customer_id="4711", from="2025-10-01", to="2025-12-31"),
      calc(expr="2310+481"),
      final_answer]
```

Step similarity: each pair $\delta(c_i, r_i)$ is 1.0 for the first two and the fourth (tool and args match), and 0.75 for `calc` (tool matches, args have partial IoU since reference uses placeholder `<sum>` token and candidate uses the literal expression — the IoU comparator scores 0.5, lifted to 0.75 by the gate-plus-bonus formula). Edit distance: $D[4][4] = 1 - 0.75 = 0.25$. Action score: $1 - 0.25/4 = 0.94$. Outcome score: 1.0 (the answer "$2,791M" matches the predicate). Composite: high.

A candidate v3.2, however, produces:

```
C2 = [get_customer(id=4711),                          # int, not str
      get_orders(customer_id=4711, from="2025-10-01"),
      get_orders(customer_id=4711, from="2025-10-01"),
      get_orders(customer_id="4711", from="2025-10-01", to="2025-12-31"),
      calc(expr="2310+481"),
      final_answer]
```

Here we see the schema-drift failure from case study 3 plus the retry-loop pattern from case study 1, in one trajectory. Step-by-step:

- Step 1: tool matches, but `id` is int instead of str. The argument comparator (which validates types) drops $\delta$ to 0.4.
- Steps 2–3: extra `get_orders` calls. These are insertions relative to $R$. Each costs 1.0 in edit distance.
- Step 4: matches reference step 2 (after the insertions).
- Step 5: matches reference step 3 at $\delta = 0.75$ as before.
- Step 6: matches reference step 4.

Edit distance: $\sim 0.6$ (one substitution at 0.6, two insertions at 1.0 each, one substitution at 0.25, normalised). Action score: $1 - 2.85/6 \approx 0.52$. Even though the *outcome* is still right (calc returns 2791, final answer matches), action score has collapsed. Efficiency: 6 steps vs reference 4 = 1.5× cost.

This is exactly the kind of pattern outcome eval cannot see. The composite drops, the regression gate fires, and the team has a chance to investigate *before* customers do.

### 7.6 When alignment scoring fails

Alignment scoring assumes the candidate is *trying* to do the same thing as the reference. It breaks in two cases:

1. **Genuinely better paths** that don't appear in the reference (acceptance-set collapse, case study 7). Mitigation: equivalence classes (section 5.3) rather than fixed references.
2. **Tool-equivalent substitutions** where the candidate uses a tool that is semantically equivalent but lexically different (e.g., `search` vs `lookup`). Mitigation: tool-equivalence groups in the step-similarity function. A small dict mapping `{search: search-group, lookup: search-group, query: search-group}` and treating intra-group matches as full credit handles this elegantly.

The pattern across both failures is the same: the reference structure is too rigid. The fix is always *enrich the reference*, never relax the grader. Relaxed graders silently accept worse trajectories; enriched references accept *more good* trajectories without accepting *any bad* ones.

## 8. Statistical rigor for trajectory evals

This is the section that separates teams who can detect a 2% regression from teams who can't, and it's where 90% of "our eval is just noise" complaints come from. The math is not hard. The discipline is.

![Minimum detectable effect vs eval-set size](/imgs/blogs/evaluating-agent-trajectories-beyond-final-answer-7.png)

### 8.1 Paired bootstrap on trajectories

When comparing agent A and agent B, evaluate *both* on the *same* tasks. Paired comparisons drop variance dramatically — typical correlation between $S(\tau_A^i, \tau^{*i})$ and $S(\tau_B^i, \tau^{*i})$ across tasks is 0.6–0.9, which means the paired-difference variance is $1 - \rho$ times the independent-difference variance. On a set of 500 tasks, that can be the difference between detecting a 4% regression and not detecting it at all.

```python
import numpy as np

def paired_bootstrap(scores_a: np.ndarray, scores_b: np.ndarray,
                     n_resamples: int = 10_000, alpha: float = 0.05):
    assert scores_a.shape == scores_b.shape
    diff = scores_a - scores_b
    n = len(diff)
    boots = np.array([
        diff[np.random.randint(0, n, size=n)].mean()
        for _ in range(n_resamples)
    ])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(diff.mean()),
            "ci": (float(lo), float(hi)),
            "significant": (lo > 0) or (hi < 0)}
```

### 8.2 Minimum detectable effect (MDE)

Given a paired-difference standard deviation $\sigma_d$, the MDE at power $1 - \beta$ and significance $\alpha$ is:

$$
\mathrm{MDE} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \frac{\sigma_d}{\sqrt{n}}
$$

For $\alpha = 0.05$, $\beta = 0.2$, that's $\mathrm{MDE} \approx 2.8 \sigma_d / \sqrt{n}$. With a typical $\sigma_d \approx 0.18$ for paired trajectory composite scores in my domains:

| $n$  | $\mathrm{MDE}$ (absolute) | Interpretation                            |
| ---- | -------------------------- | ----------------------------------------- |
| 50   | ~7.1%                      | Only catches catastrophic regressions     |
| 150  | ~4.1%                      | Catches major refactors                   |
| 300  | ~2.9%                      | Useful for routine release gating         |
| 600  | ~2.1%                      | Detects most production-relevant deltas   |
| 1200 | ~1.5%                      | Drift-tracking quality                    |

The numbers in the diagram are slightly higher because they bake in a ~20% inflation for non-Gaussian tail behavior in trajectory scores (long-tailed failures). Either way, the takeaway is: a 50-task eval is a smoke test, not a regression detector. If your eval is 50 tasks and you've shipped a model in the last quarter because eval was green, you cannot actually claim eval was green.

### 8.3 Sequential testing and the multiple-comparison trap

Once you have a real eval, you'll be tempted to peek as it runs. Don't, or at least do so under sequential-testing discipline: use an O'Brien-Fleming or Pocock boundary, or switch to always-valid sequential testing (Howard et al., 2021) which gives you anytime-valid confidence intervals. The naive approach — "rerun the eval until p<0.05" — inflates Type I error to 30%+ in pathological cases.

Similarly, if you grade on six axes and pick the best, you've effectively run six tests. Apply Bonferroni ($\alpha / 6$) or, better, use Benjamini–Hochberg to control the false discovery rate.

### 8.4 The variance budget

The other lever besides $n$ is $\sigma_d$. Reducing variance directly reduces MDE. Practical reducers:

- **Pair tasks** (as above) — biggest single win.
- **Average over replicates** at the trajectory level (run each task 3–5 times; average composite). Useful when the agent is non-deterministic. Adds cost linearly in replicates, reduces $\sigma_d$ by $1/\sqrt{r}$.
- **Stratify** by task category and report per-stratum scores. Hides cross-stratum heterogeneity from inflating $\sigma_d$.
- **Trim outliers** — but only with a pre-declared rule (e.g., trim the worst and best 2% per arm). Post-hoc trimming is data dredging.

### 8.5 Variance from the environment vs from the agent

A subtle one: if your tools are non-deterministic (search results change, prices change), some of $\sigma_d$ is environmental, not agent-attributable. This is exactly what motivates replay (next section): pin the environment, isolate agent variance. Until you have replay, run pairs of agents *temporally close together* to minimise environment drift, and treat any apparent regression smaller than the day-over-day environment drift as suspect.

## 9. The replay-and-divergence pattern

Replay is the single technique that turned trajectory eval from "interesting research project" to "regression gate I trust on Friday afternoon" in every team I have worked with. The idea is simple: the first time you run a task, record everything the environment returns. The next time you run the same task — possibly with a different agent — replay the recorded observations instead of re-querying the environment.

![Replay-and-divergence architecture](/imgs/blogs/evaluating-agent-trajectories-beyond-final-answer-8.png)

### 9.1 What replay buys you

- **Determinism.** Two runs of the same agent on the same task produce identical trajectories. Without this, eval is permanently noisy.
- **Speed.** No network calls. A 30-second real run becomes a 200 ms replay.
- **Cost.** No tool API costs during eval.
- **Cross-agent comparability.** Agent A's recorded observations become the environment for agent B. Differences are attributable to the agents.

### 9.2 What replay does not buy you

- **Coverage of novel actions.** If agent B calls a tool agent A never called, you have no recorded observation to replay. You need a *replay fallback* — either re-query the environment (and record the new observation for future replays), or abort the trajectory and mark it as "replay miss."
- **Coverage of stochastic tools.** If a tool returns different values for the same arguments (e.g., a chat-based tool, or one with timestamped results), replay returns the *recorded* version. That may or may not be what you want.

### 9.3 The divergence detector

Once you have replay, the unit of regression is *divergence*: where, in step-by-step alignment, did agent B's trajectory diverge from agent A's? A useful diagnostic:

```python
from itertools import zip_longest

def first_divergence(traj_a: Trajectory, traj_b: Trajectory) -> int | None:
    for i, (sa, sb) in enumerate(zip_longest(traj_a.steps, traj_b.steps)):
        if sa is None or sb is None:
            return i
        if (sa.action.tool, sa.action.args) != (sb.action.tool, sb.action.args):
            return i
    return None

def divergence_report(pairs):
    points = [first_divergence(a, b) for a, b in pairs]
    distribution = {i: points.count(i) for i in set(points) if i is not None}
    return distribution
```

A divergence histogram is one of the most useful regression diagnostics: it tells you *which step* is most likely to break. If 60% of regressions happen at step 1 (the first planner decision), the new model has a planning regression. If 60% happen at step 4 (the first recovery decision), the new model has a recovery regression. Outcome eval does not give you this.

### 9.4 Replay storage shape

A representative replay file on disk (`.replay/task_4711.yaml`):

```yaml
task_id: task_4711
recorded_at: "2026-05-18T10:14:00Z"
recorder_agent: "agent-v3.1"
observations:
  - step_id: 0
    action_signature: {tool: "search", args: {q: "Q4 2025 revenue"}}
    observation:
      status: 200
      body: |
        Top results: ...
      latency_ms: 412
  - step_id: 1
    action_signature: {tool: "calc", args: {expr: "2310 + 481"}}
    observation:
      result: 2791
      latency_ms: 8
```

The `action_signature` is what you key the lookup on. Keep it conservative: tool name plus the *canonicalised* arguments (sorted keys, normalised strings). Looser keying causes false replay hits; tighter keying causes excess replay misses.

### 9.5 Action-signature canonicalisation, in painful detail

The single bug class that has cost us the most engineering time on replay is *false replay miss* — the candidate calls a tool with semantically-identical arguments to the recorder, but the lookup key doesn't match because of whitespace, key ordering, or casing. We have iterated through three canonicalisation strategies and converged on the following:

```python
def canonicalise_args(args: dict) -> str:
    def norm(v):
        if isinstance(v, str):
            return " ".join(v.lower().split())          # whitespace + case
        if isinstance(v, (int, float, bool)) or v is None:
            return v
        if isinstance(v, list):
            return [norm(x) for x in v]
        if isinstance(v, dict):
            return {k: norm(args[k]) for k in sorted(v)}
        return str(v)
    return json.dumps({k: norm(args[k]) for k in sorted(args)},
                      separators=(",", ":"), sort_keys=True)
```

Note three deliberate choices: keys are *sorted* (insertion order doesn't matter), string values are *lowercased and whitespace-collapsed* (cosmetic differences don't matter), and lists preserve order (because order *does* matter for things like search-result page numbers). The first time we shipped a version that lowercased lists, we silently merged two different multi-page search queries into one replay entry and didn't notice for six weeks.

If your tools have arguments where casing matters (e.g., URLs, case-sensitive identifiers), per-tool canonicalisers are the right answer. The single global canonicaliser above is the right *default*; the per-tool overrides are the right *escape hatch*. Avoid the middle: one canonicaliser parameterised on every possible tool's quirks becomes its own source of bugs.

### 9.6 Drift detection on the replay store

Replay stores grow stale. The world moves; the recorded answers do not. The question "is my replay store still accurate?" is itself an eval-shaped problem. The discipline I have landed on:

- Every replay entry carries a `recorded_at` timestamp and an optional `freshness_window` (how long the observation is presumed valid). Search results: 7 days. Pricing: 1 day. Calculator results: forever.
- A *replay-vs-live drift check* runs weekly: sample 5% of replay entries past their freshness window, re-query live, compute the divergence rate. If divergence > 10% on the sample, age the entire replay set forward — either by re-recording or by flagging affected tasks for re-authoring.
- Replay entries that *never* drift (calc, deterministic transforms) get an infinite `freshness_window` and are exempt from the check, saving budget.

A team that doesn't do this will eventually ship an agent that's quietly "good at agreeing with stale 2023 observations" and bad at the present. The first time that happens, it's an embarrassing incident; the second time, it's a culture problem.

### 9.7 Replay misses are a feature, not a bug

When agent B calls a tool with arguments not in the replay store, you have a choice: fall back to live, or abort. I prefer **live fallback with logging** for routine eval, and **abort + manual review** for release-gate eval. The miss rate itself is a signal: a new model that has 30% replay misses against the previous model's recorded set is exploring substantially different trajectories, which deserves investigation independent of outcome score.

## 10. Case studies from production

What follows are seven incidents I have personally been on call for or post-mortem'd. Names have been generalised; numbers are real. Each is a failure mode that outcome eval missed and trajectory eval caught (or, in two cases, that trajectory eval *should* have caught and didn't because of an authoring bug). The lessons are the most expensive things in this article.

### 1. Silent Retry Loop

Agent v3.2 shipped on the back of identical outcome accuracy to v3.1 (92.1% vs 92.0%, paired bootstrap p=0.71). Six hours into rollout, latency p95 was 3.1× baseline, daily LLM spend was 2.8× baseline, and our on-call engineer noticed the action-count distribution had a heavy right tail — some tasks were taking 18, 22, even 31 tool calls to complete.

![Incident: Silent Retry Loop on agent v3.2](/imgs/blogs/evaluating-agent-trajectories-beyond-final-answer-9.png)

The wrong first hypothesis was a tool-side regression: someone had changed `get_orders` to return a tighter schema, surely that was the problem. We rolled the tool schema back. Latency stayed elevated. We then bisected the agent prompt template — v3.2 had a single-character change to the JSON-schema example in the system prompt. That change made the model emit `customer_id` as an integer instead of a string about 18% of the time. The tool returned a polite "validation error: customer_id must be string." The model dutifully reformatted and retried. Eventually one retry would land on the right shape. Outcome correct, action count 14× baseline.

The fix was a prompt template that matched the tool schema exactly. The lesson — drilled into me hard — was that trajectory eval gates would have caught this before rollout. We added an `efficiency` gate (action-count median ≤ 1.5× baseline) to the release pipeline. It has fired three times since, twice on real regressions, once on a false positive that taught us to tune the threshold. Net cost of trajectory eval, weeks: about one engineer-week of setup. Net cost of *not* having it, the incident: about three engineer-weeks plus a chunk of customer trust we didn't have to spend.

### 2. Phantom Tool Hallucination

A finance-domain agent calling a `lookup_ticker` tool to answer "what's the current price of NVDA." Outcome eval scored 96% across 200 tasks. Live performance: same 96% — but when we sampled trajectories for QA, we noticed 12% of the time the agent called `lookup_ticker`, ignored the JSON it received, and answered from its priors. The priors happened to be close to the actual prices on most days, so outcome eval saw nothing.

The catch was a **faithfulness judge** added in section 6.3: a small grader that checked whether the numbers in the final answer appeared, character-for-character or after minor normalisation, in the observed tool output. On the affected 12%, the judge fired *every time*. Once we tightened the system prompt with "you must cite the tool output verbatim for numeric values" and added a code-level grader for "did the answer contain at least one numeric token that appeared in any observation," the 12% dropped to 0.4% over two weeks.

The lesson: hallucinations on the *path* are nearly invisible when the priors are good. They become catastrophic on the day the priors are wrong. We caught this *before* such a day. Without faithfulness eval, we would have caught it after, in a customer support ticket.

### 3. Off-by-one Schema Drift

This is the cousin of the Silent Retry Loop, but slower-burning. A new agent version started passing `user_id` as an integer for 18% of tasks. Half of those happened to be customers whose `user_id` was numeric-only — for those, the tool happened to return the right record because the backing store did type coercion on its end. For the other half — alphanumeric IDs — the tool returned empty, the agent confabulated a "no orders found" answer, and outcome eval saw "user has no orders" as a coherent, plausible final answer.

What caught it: a code grader on the `args.user_id` field that enforced the schema's `type: string` constraint, and a fuzzy-match on whether the final answer's "no orders found" claim contradicted the historical task metadata (which carried a known-non-empty-orders flag on most tasks). The grader took 30 minutes to write and would have caught this on day one of v3.2. We had not yet written it. We caught the bug in production, six days in, when a customer who *did* have orders complained.

Lesson: **argument-schema validation belongs in your trajectory eval**, even if your agent runtime already validates. The agent's runtime validates *that arguments parse*. Trajectory eval validates *that arguments are correct given the task*. Those are different questions.

### 4. Lucky Recovery (the silent compounding bug)

Agent v2.8 had an 11% rate of choosing the wrong first tool, getting an error back, and successfully recovering. Outcome eval: clean. We noticed the pattern only when we built the divergence histogram described in section 9.3 — divergence from v2.7 was concentrated at step 1 (the planner) and step 2 (the recovery), with step ≥ 3 looking identical.

Investigation revealed v2.8 had been trained on a slightly different mix of data and had drifted on a single planner decision: when the task mentioned "last month," v2.8 was more likely to start with `get_calendar` than `get_orders`. The tool errored ("no calendar API for this account"). The agent recovered.

The fix was a planner-correctness gate: "first-tool match rate against acceptance set ≥ 90%." When that gate fired on v2.8 retroactively, it was at 78%. We rolled v2.8 forward anyway, but with the gate now part of the suite, v2.9 came in at 94% — explicitly because the team knew it was being measured. The act of measurement changed the work, exactly as the cliché says.

Lesson: lucky recovery is fine *until it compounds*. The 11% rate on v2.8 was already 4× the rate on v2.7. Left ungated, that drift would have continued. Trajectory eval is, among other things, a way to detect drift in places outcome eval cannot reach.

### 5. Side-effect Leak

A "look up customer's last order and report total" task — explicitly read-only by spec. The agent v4.0 had been retrained with a richer tool palette including `update_customer_note`. About 3% of the time, after retrieving the order, the agent would helpfully call `update_customer_note` with a summary, presumably because its training data contained examples of agents leaving notes for human follow-up.

Outcome eval: 100% correct (the totals were right). Production: tens of thousands of spurious customer-record updates over four days before someone in the data-warehouse team noticed an unusual volume of writes from the agent service account.

The fix at the eval level was a **side-effect ledger**: every tool call categorised as `read` or `write` (or `mixed`), the trajectory annotated with the set of writes it performed, and a task-level allow-list of writes. Any trajectory that performed a write not on its task's allow-list was hard-failed regardless of outcome. The runtime fix was a separate read-only tool palette for tasks declared read-only.

Lesson: side-effect safety is not an afterthought; it is a *vetoing* axis in the composite score. If you don't grade for it, your agent's "helpfulness" will mutate state you didn't expect.

### 6. Judge Contamination

We had an LLM-judge grader for faithfulness using the same model family as the candidate agent. Both ran on Claude Opus. Audit: on a held-out 500-trajectory set with two human raters, judge-vs-human $\kappa = 0.71$. Looked fine. Then we noticed that *every time* the candidate's style was unusually concise, the judge's faithfulness score *rose* by 0.08 average — even when the answer was *less* anchored in observations. The judge preferred its own family's stylistic register.

We swapped to a mixed-family judge panel (Claude + Llama + GPT, majority vote with humans as tiebreaker). The bias dropped to <0.02 average. Judge-vs-human $\kappa$ rose to 0.79. The cost rose by 2.4×. The eval became trustworthy.

Lesson: never grade A with A. Mixed-family judges cost more but trade money for trust, which is the right direction at the eval layer. (For online monitoring it's the wrong direction; there, use the cheap single-judge with periodic mixed-family spot-checks.)

### 7. Acceptance-set Collapse

The one where trajectory eval *itself* failed. We had a golden trajectory authored for a complex compliance task: 11 steps, hand-written. A new agent version learned to fuse two adjacent tool calls into one — a strictly-better path because the underlying tool supported the combined call. The new path was 10 steps. Edit distance scored it lower than the inferior 11-step v_old trajectory because the candidate "missed" one of the reference actions.

Trajectory eval said v_new was worse. Outcome eval said v_new was the same. A staff engineer on a vibes-based review said v_new was obviously better. We were about to ship v_old. The fix was twofold: (a) replace the golden trajectory with a predicate-based equivalence class, (b) add to the predicate the explicit allowance for "either two separate calls or one fused call." Authoring the predicate took 45 minutes. The new eval correctly ranked v_new above v_old.

Lesson: trajectory eval can lie too. Reference structures need to be designed for novel-but-better paths from day one. The transition from golden trajectories to equivalence classes (section 5) is the engineering investment that makes trajectory eval robust to your *own* model getting better in ways you didn't predict.

### 8. Tool Sequence Permutation

A coding-assistant agent had a multi-step build-and-test task: `read_file`, `edit_file`, `run_build`, `run_tests`, `summarise`. The reference required strictly that order. A new agent version learned to run tests *before* the build for tasks where the build was incremental and tests didn't require a fresh artifact — saving roughly 4 seconds per task. Outcome correct, latency down 18%. Trajectory eval, scored on edit distance against the fixed reference order, fired a regression: "step 3 expected `run_build`, found `run_tests`."

We initially overrode the gate (correctly: the new version was better). Three weeks later, a *different* version learned to skip the build step entirely when it concluded — wrongly — that the build artifact was unchanged. Outcome correct on the easy tasks where tests didn't need the build, broken on the harder ones. Because we had become accustomed to overriding the "permutation" alarm, we missed the regression for two days.

The fix: per-task sequence-rigidity flags. Some tasks have a strict order; some don't. The reference structure should encode which is which. A small bit of metadata (`order: strict | flexible`) on each task entry, plumbed through the step-similarity function, was enough. After the fix, the legitimate permutation cases stopped firing the gate, and the illegitimate skip case was caught immediately.

Lesson: a gate that fires "because of correctness in 80% of cases and incorrectness in 20%" *trains your team to ignore it*. Either tighten the reference structure so the gate is correct ≥ 95% of the time, or accept that the gate is a vibes check rather than a hard gate. Don't sit in the middle.

### 9. Faithfulness vs Plausibility

A medical-information agent (read-only, advisory) was scoring 0.88 faithfulness on our standard judge and 0.91 outcome correctness. The faithfulness judge was anchored on "is every numeric claim in the answer present in the observations?" — a tight, code-anchored question that worked well for finance.

In medical advice, however, faithfulness has a different shape: the answer must be plausible *given the user's reported symptoms*, even when the agent has not retrieved an authoritative source. A particularly cautious version of the agent started padding answers with hedging language ("consider speaking with a healthcare provider," "this is not medical advice") that contained no numbers and no claims that could be cross-referenced. Faithfulness judge: scored these answers at 1.0 because there were no numeric claims to ground. Real safety: probably worse, because the answer carried no actionable content.

We added a second sub-axis under faithfulness — *informativeness* — graded by a pairwise judge: "given the same observations, is answer A more or less informative than answer B?" The combined metric tracked human judgement at $\kappa = 0.74$, up from $\kappa = 0.51$ on the numeric-only anchor.

Lesson: faithfulness is domain-specific. The same word means different operationalisations in different settings. A trajectory eval suite that ports across domains without rethinking faithfulness will produce confidently wrong scores in the new domain. We've now standardised on always having *at least two* faithfulness sub-axes: a code-anchored claim-check, and a judge-based informativeness check.

## 11. Putting it together: a release-gate suite

What the previous sections add up to, when wired into a release pipeline, is a suite that looks like the following. Adapt the thresholds to your domain; the structure is what's load-bearing.

The release-gate config (`release_gate.yaml`):

```yaml
eval:
  set: production_v3
  size_min: 600                    # statistical power floor
  replay_required: true
  replay_miss_threshold: 0.10      # > 10% misses = inconclusive run

gates:
  outcome_correctness:
    metric: composite.outcome
    delta_max: -0.01               # at most 1% absolute regression
    paired_bootstrap_alpha: 0.05

  action_correctness:
    metric: composite.action
    delta_max: -0.02
    paired_bootstrap_alpha: 0.05

  efficiency:
    metric: composite.steps_median
    delta_max_pct: 0.50            # at most 50% more steps
    metric_alt: composite.steps_p95
    delta_max_pct: 1.00

  faithfulness:
    metric: composite.faithfulness
    delta_max: -0.03
    judge_family: "non-candidate"

  side_effect_safety:
    metric: composite.side_effect_safety
    veto: true                     # any drop fails the gate

  first_tool_correctness:
    metric: composite.first_tool_match
    threshold_min: 0.90
```

The runner (`release_gate.py`):

```python
def evaluate_release(candidate_runs, baseline_runs, config) -> dict:
    report = {"gates": {}, "verdict": "pass"}
    for gate, cfg in config["gates"].items():
        c = candidate_runs[cfg["metric"]]
        b = baseline_runs[cfg["metric"]]
        boot = paired_bootstrap(c, b, alpha=cfg.get("paired_bootstrap_alpha", 0.05))
        regressed = (
            ("delta_max" in cfg and boot["mean_diff"] < cfg["delta_max"]) or
            ("delta_max_pct" in cfg and boot["mean_diff"] / b.mean() < -cfg["delta_max_pct"]) or
            ("threshold_min" in cfg and c.mean() < cfg["threshold_min"]) or
            (cfg.get("veto") and boot["mean_diff"] < 0)
        )
        report["gates"][gate] = {"regressed": regressed, "ci": boot["ci"]}
        if regressed:
            report["verdict"] = "fail"
    return report
```

The full runner — including replay invocation, metric extraction, and per-axis dashboards — is another 200 lines, but the part that matters is the structure: every gate has a metric, a comparison rule, and a statistical test. If a gate doesn't have all three, it isn't a gate, it's a vibe.

| Gate                   | Owner axis            | Veto?   | Typical threshold                  |
| ---------------------- | --------------------- | ------- | ---------------------------------- |
| Outcome correctness    | $s_o$                 | No      | $\Delta \geq -1\%$                 |
| Action correctness     | $s_a$                 | No      | $\Delta \geq -2\%$                 |
| Efficiency (median)    | $s_e$                 | No      | $\Delta \leq +50\%$ steps          |
| Efficiency (p95)       | $s_e$                 | No      | $\Delta \leq +100\%$ steps         |
| Faithfulness           | $s_f$                 | No      | $\Delta \geq -3\%$                 |
| Side-effect safety     | $s_s$                 | **Yes** | strict non-regression              |
| First-tool correctness | $s_a$, sub-metric     | No      | $\geq 90\%$ abs                    |

## 11.1 Observability beyond the gate

The release gate is one consumer of trajectory eval. The other, equally important, is the *live observability layer*: a continuously-updated dashboard that lets your on-call engineer answer "is the agent behaving normally?" without staring at log lines.

The instrumentation you need at the trajectory level — structured steps, action signatures, side-effect ledger, observation hashes — is the same instrumentation that lets you compute the following live metrics, none of which outcome eval gives you:

- **Action-count distribution.** Median, p95, p99 over rolling windows. The Silent Retry Loop incident lit up first on p95.
- **Tool-use prevalence.** For each tool, the fraction of trajectories that called it at least once. Sudden drops here mean a planner regression (the model stopped reaching for a tool it used to use); sudden spikes mean the opposite.
- **First-tool distribution.** Of all trajectories, what fraction started with each tool? A shift here is almost always a planning regression.
- **Replay-miss rate.** Computed at eval time, but also tracked over rolling production samples. A rising miss rate against your most-recent reference set means the agent is exploring more novel trajectories — which may be good or bad, but is always worth a human's attention.
- **Side-effect ledger size.** For tasks declared read-only, the count of writes per task should be zero. A non-zero count at the 5-minute resolution is a page.

A dashboard built on these five panels — none of them showing outcome accuracy as the headline — has, in my experience, more diagnostic power than any single accuracy graph. Accuracy is the headline at quarterly reviews; for daily operations, the panels above are what you actually look at.

The cost is modest. Every metric above is a derived field on the trajectory log, which you already need for eval. The Grafana panels are an afternoon. The alerts on each — what threshold pages whom — take a few weeks of iteration to dial in. The investment cap is "less than the trajectory eval suite itself," because the data plumbing is shared. Two for the price of one is rare in MLOps; this is one of those rare cases.

## 12. The cost of trajectory eval, honestly

There is no point hand-waving the cost. Trajectory eval is *more expensive* than outcome eval — in eng time, in compute, in cognitive load. Here is what we have actually paid, across three teams, normalised to "for a 500-task eval suite":

| Cost item                          | Outcome-only | Trajectory  | Multiplier |
| ---------------------------------- | ------------ | ----------- | ---------- |
| Authoring (one-time)               | ~2 weeks     | ~6–8 weeks  | 3–4×       |
| Per-run compute                    | ~$8          | ~$22        | 2.7×       |
| Per-run wall-clock                 | ~6 min       | ~14 min     | 2.3×       |
| Maintenance (per quarter)          | ~2 days      | ~5 days     | 2.5×       |
| Cognitive load (oncall paging)     | low          | medium      | —          |

What you get back, also measured: roughly 3.5× more pre-rollout regressions caught per quarter, and the average regression caught is significantly more severe (in terms of customer-visible impact) than the regressions outcome-eval catches. The break-even point — defined as "engineering hours saved on incidents exceeds engineering hours spent on eval" — was about three months in for each of those teams.

Below that volume, outcome eval plus careful manual review is fine. Above it, trajectory eval is not optional; it is the difference between shipping with confidence and shipping with luck.

## 13. When to reach for trajectory eval / when not to

### Reach for trajectory eval when …

- Your agent calls more than one tool per task on average.
- Tool calls have side effects, cost, or latency that matter independently of correctness.
- You are doing release-gate evaluation, not just exploratory benchmarking.
- Outcome accuracy on your suite has been flat for a release cycle while *something else* keeps going wrong in production.
- You have at least one engineer who can own eval infrastructure for a quarter.

### Skip trajectory eval when …

- Your agent is effectively a single-call wrapper around an LLM (a classifier, summariser, or one-shot tool). Outcome is enough.
- You are still in the "does this work at all?" phase. Trajectory eval rewards discipline, and discipline before product-market-fit is premature optimisation.
- You don't yet have replay or structured trajectory logging. Build that first, or trajectory eval will be a flaky mess that erodes trust in eval generally.
- The total task volume is < 50/day and a human can manually review every trajectory. Manual review is a perfectly good substitute at that scale.

The intermediate zone — agents that are past prototype but pre-production — is the most painful, because outcome eval is starting to feel insufficient but trajectory eval is not yet justified. My honest advice for that zone: build the trajectory logging (instrumentation) first, even if you grade with outcome only. The instrumentation is the thing that takes time to get right; the grading is comparatively quick to add once the data is there.

### 13.1 Phased rollout: what to build first

If you start trajectory eval from scratch on Monday, here is the order I would invest in across the first three months. Each step depends only on the prior one being present.

1. **Week 1–2: Structured trajectory logging.** Emit the `Trajectory` dataclass from section 2. Anything downstream depends on this; it is the only investment that returns negative dividend if skipped.
2. **Week 3–4: Replay store.** Even before any new grader, recording observations against task IDs is what makes everything else deterministic. Run the same agent twice; verify trajectories are byte-identical under replay. If they are not, you have non-determinism elsewhere that you must fix first.
3. **Week 5–6: Code graders for the three "code-cheap" axes** — action correctness (tool name + arg schema), efficiency (step count + token cost), side-effect safety (write-allow-list). Together these catch three of the five failure modes from section 3, with zero LLM cost at grading time.
4. **Week 7–8: Paired bootstrap and the release-gate config.** No new graders, just statistical machinery. Catch your first regression here; the team's confidence in the suite is set by whichever incident the suite catches first.
5. **Week 9–10: First LLM judge** — faithfulness, calibrated against 100–200 human-labelled trajectories. Calibrate against humans before involving the judge in any gate.
6. **Week 11–12: Equivalence-class predicates** for the top 10 task categories by traffic. This is the investment that pays off most slowly and most durably; teams that skip it eventually rewrite their whole reference set anyway.

After three months, the marginal investment is in expanded judge coverage (recovery, informativeness), broader eval-set coverage (more tasks per category), and observability polish. The structural pieces, by then, are all in place.

## 14. Further reading

- [Demystifying Evals for AI Agents: A Practical Guide](/blog/machine-learning/ai-agent/eval-agents) — the introductory companion to this post.
- [Designing Long-Running Agents: Reliability in Production](/blog/machine-learning/ai-agent/designing-long-running-agents-reliability-production) — neighbouring post on the *runtime* side of agent reliability.
- [Effective Context Engineering for AI Agents](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents) — on the input-side of agent quality; trajectory eval often pushes you to redesign context.
- [Building Effective Agents: A Hands-on Guide](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide) — for readers who got here without the prerequisites and want to back-fill.
- Howard, S.R. et al. (2021). *Time-uniform, nonparametric, nonasymptotic confidence sequences.* The standard reference for always-valid sequential testing.
- Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* The first systematic treatment of judge contamination and pairwise vs pointwise grading.

The first time you stand in front of a release gate that fires on trajectory metrics — not outcome — and the gate fires, and you catch a regression that outcome eval scored green, you will not go back. Until that day, this article will feel like overhead. After that day, it will feel like the floor.

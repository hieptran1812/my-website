---
title: "When to Fine-Tune Tool Calling for LLMs: A Decision Framework and Field Guide"
publishDate: "2026-06-10"
category: "machine-learning"
subcategory: "Training Techniques"
tags: ["fine-tuning", "tool-calling", "function-calling", "llm", "agents", "sft", "lora", "dpo", "distillation", "constrained-decoding", "evaluation"]
date: "2026-06-10"
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/fine-tuning-tool-calling-llms-when-how-1.png"
excerpt: "Fine-tuning is the last rung on the tool-calling ladder, not the first. This guide is a decision framework for when training tool calls actually pays off — and a full, runnable procedure for doing it right: data manufacturing, loss masking, SFT, DPO, evaluation, and constrained decoding."
---

The single most expensive mistake I see teams make with agents is reaching for a fine-tuning run the first time their model calls the wrong tool. The reasoning feels airtight: the model is making mistakes on tool calls, fine-tuning teaches models to stop making mistakes, therefore fine-tune on tool calls. Three weeks and a few thousand dollars of GPU time later, the model is marginally better at the twelve tools in the training set, noticeably worse at general conversation, and brittle the moment the API team renames a parameter.

Almost none of that work needed to happen. The overwhelming majority of "the agent picked the wrong tool" bugs are prompt bugs, schema bugs, context bugs, or decoding bugs wearing a training-problem costume. Fine-tuning *can* fix tool calling — there is a real, narrow set of situations where it is the right and only answer — but it sits near the top of a ladder of interventions, and every rung below it is cheaper, faster, and more reversible.

![The tool-calling escalation ladder: prompt and schema clarity, few-shot examples, tool retrieval, constrained decoding, SFT, and preference optimization, ordered cheapest at the bottom to most expensive at the top](/imgs/blogs/fine-tuning-tool-calling-llms-when-how-1.png)

The diagram above is the mental model for this entire article. You climb the ladder from the bottom: prompt clarity is free and kills most bugs, few-shot examples cost nothing but tokens, tool retrieval scales you to hundreds of tools without bloating context, and constrained decoding makes malformed output structurally impossible. Only when all four of those have been tried and a measurable ceiling remains do you step onto the fifth rung — supervised fine-tuning — and only when *that* plateaus do you reach the sixth, preference optimization. This piece is a decision framework for knowing which rung you are actually on, followed by a complete, runnable procedure for the top two rungs when you genuinely need them.

## The assumption that wastes a quarter of fine-tuning budgets

Tool calling looks like a behavior a model either "knows" or "doesn't know," which is exactly the framing that leads people to treat every failure as a knowledge gap that more training will close. It is a more useful exercise to separate what the model fundamentally cannot do from what it is simply not being asked to do correctly.

| Symptom | The naive view | The reality |
|---|---|---|
| Model calls the wrong tool | "It doesn't understand the tools — train it" | The tool descriptions are vague or overlap; rewrite them or add retrieval |
| Arguments are malformed JSON | "The model is bad at JSON — fine-tune it" | The decoder is unconstrained; a grammar fixes this with zero training |
| Model invents a tool that doesn't exist | "It needs more examples of real tools" | It was never shown what to do when *no* tool applies; add abstention data |
| Wrong arguments for the right tool | "It can't follow the schema" | This one is often genuinely a training problem — but verify first |
| Latency and cost are too high | "Fine-tune a small model to be as good" | Plausible — but the motivation is distillation, not error-fixing |
| It works in dev, breaks in prod | "The model regressed — retrain" | The schema drifted; your eval is pinned to stale fixtures |

The pattern is that the *symptom* almost never tells you the *layer*. A wrong-tool selection can come from a bad description (prompt layer), an overloaded context window (retrieval layer), or a genuinely confused model (training layer), and the fix for each is completely different. Spending a fine-tuning budget before you have localized the failure to the model itself is how you end up with an expensive model that is good at the wrong thing.

> The first question is never "how do I fine-tune this?" It is "what, precisely, is broken, and is the model even the thing that is broken?"

There is one more reason the instinct misfires: tool calling is one of the most fragile capabilities to fine-tune *without* collateral damage. The same gradient steps that sharpen argument formatting can blunt general instruction-following, and the data you need is expensive to produce correctly. So the bar for stepping onto the training rungs should be high, and you should be able to state — in one sentence, with a number attached — what prompting could not achieve.

## The escalation ladder: four ceilings before you train

Each rung on the ladder breaks a ceiling that the rung below it cannot. Walking them in order is not bureaucracy; it is the cheapest path to the actual fix, because each rung is an experiment that tells you something about the next.

**Rung 1 — Prompt and schema clarity (free, always first).** The tool schema *is* the prompt. A function named `search` with a description of `"searches"` and a single `query: string` argument is asking the model to guess. The same tool named `search_internal_kb`, described as `"Full-text search over the internal knowledge base. Use for policy, HR, and IT questions. Do NOT use for live customer data."`, with an enum-typed `domain` argument, removes most of the ambiguity that gets misread as a model deficiency. In my experience something like four out of five "wrong tool" reports close here, at the cost of an afternoon rewriting descriptions.

**Rung 2 — Few-shot tool examples (free).** Two or three worked examples in the system prompt — a user turn, the correct tool call with realistic arguments, the tool result, the final answer — anchor both the argument shape and the *timing* of calls (when to call versus when to answer directly). This is the cheapest fix for over-calling and under-calling, and it requires no infrastructure beyond a longer prompt.

**Rung 3 — Tool retrieval and search (cheap).** When you have dozens to hundreds of tools, the problem is rarely that the model is dumb; it is that fifty tool schemas in the context window crowd out the actual task and inflate latency and cost on every single turn. The fix is to retrieve only the handful of relevant tools per request — a small embedding index over tool descriptions, or a two-stage "which toolset?" router — so the model sees five tools, not five hundred. Anthropic's [advanced tool use](/blog/machine-learning/ai-agent/advance-tool-use) features and the [Model Context Protocol](/blog/machine-learning/ai-agent/model-context-protocol) both lean on this idea: discovery is a retrieval problem, not a memorization problem.

**Rung 4 — Constrained decoding (cheap, deterministic).** If the failure is structural — malformed JSON, wrong types, missing required fields — you do not need the model to *learn* the schema; you need the decoder to be physically unable to violate it. A grammar or JSON-schema constraint masks every token that would produce invalid output, so the model literally cannot emit a trailing comma or an unquoted key. This is the single highest-leverage non-training fix for tool calling, and we return to it in depth at the end.

**Rung 5 — Supervised fine-tuning (expensive, this article).** Only now. SFT is the right tool when the residual error survives all four rungs above: the descriptions are crisp, the relevant tools are in context, the decoder is constrained, and the model *still* picks the wrong argument values, or you are running a small model whose zero-shot tool selection is simply not good enough and you cannot afford a larger one.

**Rung 6 — Preference optimization (most expensive, last mile).** When SFT itself plateaus — typically on the hardest, most ambiguous calls — you move to DPO or a GRPO-style objective to squeeze out the final few points of accuracy using the model's own mistakes as the training signal.

The ladder is also a budget argument. Rungs 1–4 are measured in engineer-hours and prompt tokens. Rungs 5–6 are measured in GPU-days, labeled or synthesized data, eval infrastructure, and the ongoing maintenance cost of a model you now own. If you are going to pay that, you want to be certain the cheaper rungs are exhausted, because a fine-tuned model does not relieve you of needing good descriptions and a constrained decoder — it sits *on top of* them.

A quick numeric example makes the ordering concrete. Say you start at 70% tool-call success and a stakeholder wants 95%. Rewriting six vague descriptions gets you to 82% in an afternoon. Three few-shot examples add another four points to 86%. Pruning the in-context tool set from forty tools to the relevant six with retrieval pushes selection up to 90%. A grammar constraint eliminates the 3% of failures that were malformed JSON, landing you near 93% — and now the residual 2% is concentrated in genuine argument-formatting errors on a stable, high-volume surface. *That* is a fine-tuning problem, and it is a small, well-scoped one. Notice what happened: four rungs of cheap work converted a vague "make it better" into a precise, two-point training target you can actually hit and measure. Had you started by fine-tuning at 70%, you would have spent the budget teaching the model things a prompt rewrite would have given you for free, and you would still be guessing where the real residual lived.

## Diagnose before you train: the six failure modes

"The agent's tool calls are bad" is not a diagnosis. Before any training decision, you have to factor the error into specific modes, because they have different fixes and only some of them respond to fine-tuning at all.

![A taxonomy tree of six tool-call failure modes branching from a root node, each labeled with its cheapest fix, with the training-fixable modes highlighted](/imgs/blogs/fine-tuning-tool-calling-llms-when-how-2.png)

The six modes, and what actually fixes each:

1. **Wrong tool selected.** The model calls `get_weather` when it should have called `get_forecast`. Cheapest fix is disambiguating descriptions and retrieval. This becomes a training problem *only* on small models whose selection is weak even with a clean five-tool context — which is why it is marked as a partial (caution) case in the figure.

2. **Bad argument format.** Right tool, wrong argument values: a date as `"next Tuesday"` instead of `"2026-06-16"`, a units mismatch, a currency code in the amount field. This is the failure mode that fine-tuning most reliably fixes, because it is about learning a domain-specific mapping from natural language to a structured argument convention that prompting struggles to fully specify.

3. **Malformed JSON.** The call is structurally invalid — unbalanced braces, unquoted keys, a hallucinated trailing comment. This is a *decoding* problem, and constrained decoding solves it completely and for free. Fine-tuning to fix JSON validity is the canonical example of using a sledgehammer where a grammar would do.

4. **Hallucinated tool.** The model calls `escalate_to_human` when no such tool was provided, usually because it has never been shown what to do when no tool applies. The fix is abstention training: negative examples where the correct behavior is to ask a clarifying question or answer directly. This is genuinely training-fixable, which is why it is highlighted alongside argument formatting.

5. **Over- and under-calling.** The model calls a tool when it should just answer, or answers from memory when it should look something up. This is mostly a prompt-and-few-shot problem — the calling *policy* is best specified by examples and explicit instructions.

6. **Wrong call order.** In a multi-step task the model calls `book_flight` before `search_flights`. This is a planning problem, addressed with a planner, a state machine, or few-shot trajectories — rarely with token-level fine-tuning of individual calls.

Of the six, only argument formatting, abstention, and small-model selection respond well to training. That is the punchline of the taxonomy: at most three of your six possible problems are fine-tuning problems, and you need to know *which* before you spend a GPU-day. Evaluating where the errors actually concentrate is itself a measurement task — the kind explored in [evaluating agent trajectories beyond the final answer](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer) — and it has to come before the data work, not after.

## The decision: three conditions that must hold together

Suppose you have localized the failure to a genuinely training-fixable mode. You are still not done justifying the run, because fine-tuning tool calls only pays off when three conditions hold *simultaneously*. Any one of them missing turns the investment negative.

![A decision matrix with prompting headroom on the rows and schema stability plus volume on the columns; only the cell where prompting has plateaued, the schema is stable, and volume is high is marked fine-tune](/imgs/blogs/fine-tuning-tool-calling-llms-when-how-3.png)

The matrix makes the joint condition concrete. Reading it:

**Condition 1 — Prompting has plateaued.** You have an eval, you have run the cheaper rungs, and there is a measurable residual error you can point at. If you have not yet built the eval (Step 1 below), you cannot know whether prompting has plateaued, and you have no business training. The top row of the matrix — prompting still has headroom — is *never* a fine-tune; it is "go iterate the prompt."

**Condition 2 — The schema is stable.** The tools you are training on are not going to be renamed, re-typed, or restructured next sprint. A fine-tuned model bakes the schema into its weights; if the schema churns, you are signing up to retrain on every API change, which is a maintenance treadmill that almost always costs more than the accuracy it buys. The left column of the matrix — schema churning — routes to "stabilize the schema first," even when prompting has plateaued.

**Condition 3 — Volume justifies the amortization.** Fine-tuning has a fixed cost (data, training, eval, maintenance) that you amortize over calls. At a thousand calls a day the math rarely closes; at a million calls a day, a few points of accuracy or a move to a cheaper model can pay for the whole program in a week. The middle column — stable but low volume — usually routes to constrained decoding plus retrieval, which gets you most of the reliability without the fixed cost.

Only the bottom-right cell — prompting plateaued, schema stable, volume high — is a clear "fine-tune." Everything else has a cheaper answer. I keep this matrix in the room for tool-calling planning because it reframes the discussion from "can we make the model better?" (almost always yes, marginally) to "should we, given these three axes?" (usually no, occasionally an emphatic yes).

It is worth naming the strongest *legitimate* trigger explicitly, because it is the one that most often lands in the bottom-right cell: **cost.** You are serving a frontier model at high volume, the per-call price hurts, and you want to move to a 7B–14B open model that you host yourself. The small model's zero-shot tool calling is not good enough, but the task is narrow and the schema is stable. That is a distillation problem dressed as a fine-tuning problem, and it is the single most defensible reason to train tool calls. We will treat it as the running example.

## What you are actually optimizing

Before the procedure, it helps to be precise about what a tool-calling fine-tune mechanically changes, because most botched runs come from a misunderstanding at exactly this level. A modern tool-calling model does not have a separate "tool head." Tool calls are *just tokens* — special tokens and structured text that the model emits in the assistant turn, interleaved with a chat template that the tokenizer applies.

![A loss-mask diagram showing a five-turn conversation where system, user, and tool turns are masked to -100 while the assistant tool-call and final reply turns carry the loss](/imgs/blogs/fine-tuning-tool-calling-llms-when-how-4.png)

The figure shows the one thing you must get right: **the loss mask.** A training example for tool calling is a full conversation — a system turn with the tool schemas, a user request, an assistant turn that emits a tool call, a tool turn carrying the result, and a final assistant turn. The label tensor that the loss is computed against sets every non-trained position to `-100` (the ignore index in PyTorch's cross-entropy), so the gradient flows *only* through the assistant tool-call tokens and the final assistant reply.

This matters enormously, and getting it wrong is the most common silent bug in tool-calling fine-tunes. If you compute loss on the tool-result turns, you are teaching the model to *predict tool outputs* — to hallucinate what `refund_order` returns instead of learning to call it and wait. The model that has trained on unmasked tool results will confidently fabricate `{"ok": true, "amount": 42.00}` rather than emit the call and read the real result. The fix is one line of masking, but it is the line that separates a working recipe from a model that has learned to fake its own tools.

Two more mechanical facts to internalize:

- **Special tokens are real tokens.** Whatever delimiters your base model uses to mark a tool call — `<tool_call>...</tool_call>`, a `<|python_tag|>`, a specific role header — must already be in the tokenizer's vocabulary, and you must use the model's *own* chat template rather than inventing your own format. Training a Qwen-style model on a Llama-style tool format wastes capacity teaching it a format it will then have to un-learn at inference.
- **The model learns a distribution over arguments, not a function.** Fine-tuning shifts the probability mass toward the argument conventions in your data. It cannot teach the model facts it has no way to know; it can teach it that, in your domain, dates are ISO-8601 and amounts are integers in cents. Keep your expectations calibrated to "format and selection," not "reasoning."

With the mechanics clear, here is the procedure. The general fine-tuning toolkit — LoRA, QLoRA, learning-rate schedules, the SFT-then-preference pipeline — is covered in [effective LLM fine-tuning techniques](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques); what follows is the part specific to *tool calls*.

## Match the tool-call format to the base model

Tool calling is not one format; every base-model family encodes calls with its own delimiters, and a fine-tune that ignores this fights the model's pretraining instead of building on it. Before you format a single example, confirm exactly how your base model renders a tool call, because the chat template — not your preferences — defines the target the loss is computed against.

| Model family | How a tool call is rendered | Practical notes for fine-tuning |
|---|---|---|
| Qwen 2.5 / Hermes-style | `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` | Clean, well-documented template; a de-facto default for open tool-calling work |
| Llama 3.1 / 3.2 | JSON object in the assistant turn, or a `<\|python_tag\|>` for code-interpreter style | Two modes; pick one and be consistent, or the model learns a blurred target |
| Mistral / Mixtral | `[TOOL_CALLS]` token followed by a JSON array | Supports parallel calls natively; format the array even for single calls |
| Functionary / fine-tuned forks | Custom role headers and delimiters | Read the model card; these diverge most from the base family |

Three consequences follow directly:

- **Never invent a format.** If you wrap calls in `<function>` tags because you find them readable, you are spending training capacity teaching the model a delimiter it has to override from pretraining — and you lose compatibility with the serving stack's parser. Always round-trip through `apply_chat_template` so the tokens you train on are exactly the tokens the model will be asked to produce.
- **Parallel calls are a format decision, not just a capability.** Models like Mistral render multiple calls as a JSON array. If your workload needs parallel calls but your training data only ever shows single calls wrapped as a one-element array, the model will struggle to emit two — train the shape you intend to serve.
- **The special tokens must exist in the vocabulary.** `<tool_call>` is a real token (or a stable sequence of them) in a tool-aware base model. Teaching a base model that has *no* tool tokens to emit them is possible but far more expensive than starting from an instruct model that already speaks the format; for the cost-driven distillation case, always start from the tool-aware instruct variant.

This is also the first place to sanity-check your teacher. If you distill from a frontier model whose native tool format differs from your student's, you must re-render the teacher's calls into the student's template during data construction — the *content* of the call transfers, the *delimiters* do not. A surprising number of weak fine-tunes trace back to a student trained on a teacher's raw format that the student's tokenizer then splits into nonsense.

## Step 1 — Write the eval before you write the data

The discipline that separates fine-tuning projects that converge from ones that thrash is this: **you build the evaluation set and the metrics before you collect a single training example.** Without an eval you cannot establish that prompting has plateaued (Condition 1), you cannot detect catastrophic forgetting, and you cannot tell whether a training run helped or hurt. An eval you write *after* you have a model is an eval you have unconsciously fit to the model.

A tool-calling eval has two altitudes, and you need both:

- **Per-call (offline).** A fixed set of (conversation prefix → expected tool call) pairs. For each, you measure: did the model emit valid JSON; did it pick the right tool; are the arguments correct (exact match for IDs and enums, semantic match for free text and dates); does the call execute without error. These are fast, deterministic, and run on every checkpoint.
- **Per-trajectory (online).** Whole tasks run end-to-end against a sandboxed version of the real tools, scored on task success and on whether the *sequence* of calls was sensible. These are slower and noisier but catch the failures that per-call metrics miss — the model that picks each call correctly but loops forever, or solves the task in eight calls where two would do.

Define the metrics precisely now, because vague metrics are how projects fool themselves. *Tool-selection accuracy* is the fraction of cases where the model called the right tool (or correctly called none). *Argument correctness* needs a per-argument match policy: exact match for IDs, enums, and booleans; numeric tolerance for amounts; and semantic match — often a small judge model or a normalizer — for dates and free text, because `"2026-06-16"` and `"June 16th"` should both count as correct for a date argument while `"damaged"` and `"broke"` should both satisfy a reason field. *Schema-valid rate* and *executable rate* are pure pass/fail. Write these down as code, not prose, so two engineers compute the same number from the same outputs; an argument-correctness metric that drifts between reviewers is worse than no metric, because it manufactures false confidence.

Build the offline set first, from real traffic if you have it (sampled and anonymized) and synthesized hard cases if you do not. Two hundred to five hundred carefully chosen examples beat ten thousand random ones. Stratify by tool, by difficulty, and — critically — include a slice of **abstention cases** where the correct behavior is *no tool call*, because a model trained only on positive calls will call a tool for everything.

The eval-case contract — `eval_set.py` — makes the gold behavior explicit for every case, including the ones whose correct outcome is to call nothing at all:

```python
from dataclasses import dataclass, field

@dataclass
class ToolEvalCase:
    case_id: str
    messages: list[dict]                 # system + user (+ prior turns)
    expected_tool: str | None            # None == correct behavior is to NOT call
    expected_args: dict = field(default_factory=dict)
    arg_match: dict = field(default_factory=dict)  # per-arg: "exact" | "semantic" | "numeric"
    difficulty: str = "medium"           # easy | medium | hard
    slice: str = "default"               # used for stratified reporting

EVAL = [
    ToolEvalCase(
        case_id="refund-001",
        messages=[
            {"role": "system", "content": SYSTEM_WITH_TOOLS},
            {"role": "user", "content": "Please refund order #8821, it arrived broken."},
        ],
        expected_tool="refund_order",
        expected_args={"order_id": 8821, "reason": "damaged"},
        arg_match={"order_id": "exact", "reason": "semantic"},
        difficulty="easy",
        slice="refunds",
    ),
    ToolEvalCase(
        case_id="abstain-014",
        messages=[
            {"role": "system", "content": SYSTEM_WITH_TOOLS},
            {"role": "user", "content": "What's your favorite color?"},
        ],
        expected_tool=None,              # must answer directly, call nothing
        difficulty="easy",
        slice="abstention",
    ),
]
```

The eval is a living artifact. Every production failure you see after launch becomes a new case, so the set grows toward the actual distribution of hard inputs over time. If you take one thing from this section: the eval is the deliverable that makes the rest of the project measurable, and it is cheap relative to a training run that you cannot interpret.

## Step 2 — Manufacture the data

Here is the mental shift that makes tool-calling fine-tunes work: **training data for tool calls is manufactured, not collected.** You will not find a clean corpus of perfect tool-calling trajectories lying around, and raw production logs are full of the very errors you are trying to fix. The reliable approach is a pipeline that generates candidate trajectories, executes them against real or sandboxed tools, and keeps only the ones that demonstrably worked.

![A data-manufacturing pipeline: teacher rollout, execute the call, rejection sampling, schema validation, loss-mask formatting, train and dev split](/imgs/blogs/fine-tuning-tool-calling-llms-when-how-5.png)

The pipeline has five stages, and the second and third are what distinguish a tool-calling dataset from an ordinary instruction dataset.

**Teacher rollout.** Use a strong model — the frontier model you are trying to distill *away* from, in the cost-driven case — to generate candidate tool calls for each input. This is standard distillation: the expensive model is the teacher, your small model is the student. Sample several completions per input with temperature, so you get diversity to filter from.

**Execute the call.** This is the step people skip, and skipping it is why their fine-tunes underperform. Actually *run* every generated tool call against the real tool or a faithful sandbox. A tool call that looks perfect but throws a `404` because the order ID does not exist is a bad training example, and you can only know that by executing it.

**Rejection sampling.** Keep only the trajectories whose execution succeeded *and* whose result matches the expected outcome. The teacher is wrong more often than its fluency suggests — confidently wrong, in the plausible way that is hardest to catch by reading.

![A before-and-after comparison: raw teacher traces full of malformed JSON, wrong tools, runtime errors and plausible-but-wrong results on the left; a filtered, execution-verified set on the right](/imgs/blogs/fine-tuning-tool-calling-llms-when-how-6.png)

The before/after is the whole argument for execution-grounded filtering. On the left is what the teacher actually produces: a mix of malformed calls, wrong tool choices, runtime errors, and — the dangerous category — calls that parse, execute, and return *something*, but the wrong something. On the right is what survives: calls that are schema-valid, choose the right tool with the right arguments, execute cleanly, and return the expected result. In practice the filter discards 30–60% of teacher trajectories on a non-trivial tool surface. Training on the unfiltered left-hand set teaches the model the teacher's mistakes; training on the right-hand set is what actually moves your eval.

Here is the core of the rejection-sampling loop, the part that turns fluent-but-unverified generations into trustworthy training data:

```python
import json
from concurrent.futures import ThreadPoolExecutor

def generate_candidates(case, teacher, n=4, temperature=0.7):
    """Sample n tool-call completions from the teacher for one input."""
    return [teacher.complete(case.messages, temperature=temperature) for _ in range(n)]

def verify(candidate, case, sandbox):
    """A candidate is kept only if it parses, calls the right tool,
    executes cleanly, and returns the expected result."""
    try:
        call = json.loads(candidate.tool_call_json)        # 1. schema-valid?
    except json.JSONDecodeError:
        return None
    if call["name"] != case.expected_tool:                  # 2. right tool?
        return None
    if not args_match(call["arguments"], case.expected_args, case.arg_match):
        return None                                         # 3. right args?
    result = sandbox.execute(call)                          # 4. executes?
    if result.error or not result.matches(case.expected_result):
        return None                                         # 5. right outcome?
    return {"messages": case.messages + [candidate.assistant_turn,
                                         result.tool_turn,
                                         candidate.final_turn]}

def build(cases, teacher, sandbox, workers=16):
    kept = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for case in cases:
            cands = generate_candidates(case, teacher)
            verified = [v for c in cands if (v := verify(c, case, sandbox))]
            kept.extend(verified[:1])      # one verified trajectory per case is plenty
    print(f"kept {len(kept)} / {len(cases)} cases after execution filtering")
    return kept
```

Where do the candidate trajectories come from? In practice, from a blend of sources, and the blend matters more than any single one:

| Source | What it gives you | Watch out for |
|---|---|---|
| Production logs (sampled, anonymized) | The real input distribution, real hard cases | Full of the errors you are fixing; must be re-verified, never trusted as gold |
| Teacher generation (frontier model) | Scale and coverage of the long tail | Confidently wrong on a meaningful fraction; execution-filter ruthlessly |
| Hand-authored seeds | Coverage of rare-but-critical tools and abstention | Expensive; reserve for cases synthesis cannot reach |
| Schema-guided synthesis | Inputs that exercise every argument and enum value | Can drift unrealistic; ground against real entity IDs |

The reliable recipe is to seed from real inputs where you have them, expand coverage with teacher generation, hand-author the rare-tool and abstention cases that neither source produces enough of, and then run every candidate from all sources through the same execution-grounded filter. The filter is the great equalizer: it does not care whether a trajectory came from a frontier model or an intern, only whether the call actually worked.

Two refinements that pay for themselves:

- **Mine hard negatives for abstention.** Deliberately include inputs where no tool applies, and keep teacher trajectories where the teacher correctly *declined* to call. This is how you train the model out of the hallucinated-tool failure mode. Without it, your model learns that every input deserves a call.
- **Deduplicate and balance.** A pipeline left unattended produces a thousand `search` calls and three `refund_order` calls, because that is the traffic distribution. Balance by tool so the rare-but-important tools get enough signal, and deduplicate near-identical trajectories so the model does not overfit a single phrasing.

One number to anchor expectations: for a cost-driven distillation onto a 7B model with a stable surface of 10–20 tools, a few thousand verified trajectories is typically enough to close most of the gap to the teacher *on that surface*. You do not need a million examples; you need a few thousand *correct* ones, which is exactly why the execution-grounded filter matters more than raw volume.

## Step 3 — Format and mask

With a verified set of trajectories, formatting is mechanical but unforgiving. Two rules:

**Use the model's own chat template.** Modern tokenizers ship a chat template that knows how to render tool definitions, tool calls, and tool results into the exact token sequence the base model was trained on. Use it; do not hand-roll a format.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def encode(example, tools):
    # return_assistant_tokens_mask gives us a 0/1 mask of which tokens
    # belong to assistant turns — exactly the tokens we want loss on.
    enc = tok.apply_chat_template(
        example["messages"],
        tools=tools,                       # the tool schemas, rendered by the template
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = enc["input_ids"]
    assistant_mask = enc["assistant_masks"]            # 1 on assistant tokens, 0 elsewhere
    # Loss only where the assistant speaks; everything else is -100 (ignored).
    labels = [tid if m else -100 for tid, m in zip(input_ids, assistant_mask)]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": enc["attention_mask"]}
```

**Verify the mask on a real example before training.** Decode the positions where `labels != -100` and read them back. You should see exactly the assistant tool calls and final replies — never the system prompt, never the user turn, never the tool results. I have seen more than one project where a template change silently broke the mask and nobody noticed until the model started inventing tool outputs in production. A five-minute assertion at data-build time is cheap insurance:

```python
trained_text = tok.decode([tid for tid, lab in zip(ex["input_ids"], ex["labels"]) if lab != -100])
assert "tool_call" in trained_text          # we ARE training on the call
assert "broken" not in trained_text         # we are NOT training on the user's words
```

## Step 4 — Supervised fine-tuning

For the cost-driven distillation case, parameter-efficient fine-tuning with LoRA or QLoRA is almost always the right choice over full fine-tuning, for two reasons specific to tool calling: it is far cheaper (a 7B QLoRA run fits on a single 24GB GPU), and — more importantly — it is gentler on the model's general capabilities, which limits the catastrophic forgetting that full fine-tuning invites. You are trying to add a narrow skill, not rebuild the model.

```python
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb,
    torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
)
peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
cfg = SFTConfig(
    output_dir="qwen7b-tools",
    num_train_epochs=2,                # 2-3 epochs; tool data overfits fast
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,     # effective batch 32
    learning_rate=1e-4,                # PEFT tolerates higher LR than full FT
    lr_scheduler_type="cosine", warmup_ratio=0.03,
    bf16=True, packing=False,          # do NOT pack — it breaks per-turn masking
    max_seq_length=4096,
    assistant_only_loss=True,          # the loss mask from Step 3
    logging_steps=10, eval_strategy="steps", eval_steps=50, save_steps=50,
)
trainer = SFTTrainer(
    model=model, args=cfg, peft_config=peft_cfg,
    train_dataset=load_dataset("json", data_files="train.jsonl", split="train"),
    eval_dataset=load_dataset("json", data_files="dev.jsonl", split="train"),
)
trainer.train()
```

The flags that matter most for tool calling, beyond the usual LoRA hyperparameters:

- **`assistant_only_loss=True` (and `packing=False`).** This is the loss mask again. If your trainer version packs sequences by default, packing concatenates examples and destroys the per-turn assistant mask — so disable it for tool data even though it costs throughput. Confirm the trainer is actually masking; do not assume.
- **Two to three epochs, not ten.** Tool datasets are small and structured, and they overfit fast. Watch the dev loss and the per-call eval together; the eval usually peaks before the loss bottoms out, because the model starts memorizing specific argument values past that point.
- **Mix in a slice of general instruction data.** Adding 10–20% general chat/instruction examples to the tool data is the cheapest insurance against forgetting how to hold a normal conversation. It costs a little tool-task accuracy and buys back a lot of general capability.

A word on base-model choice, since it constrains everything above. For cost-driven distillation, start from the *instruct* variant of a tool-aware family at the smallest size that clears your accuracy bar after training — not the smallest size that exists. A 3B model can be trained to call ten tools competently; it will struggle with the multi-constraint argument reasoning a 7B–14B handles comfortably. The cheap experiment is to run your offline eval zero-shot on candidate base sizes *before* committing: if the 7B is already within a few points of your bar with good prompting, a modest fine-tune will clear it; if the 3B is twenty points behind, no realistic amount of tool data will close that gap, and you are better off paying for the larger base. Let the measured zero-shot gap, on your own eval, pick the size — not a parameter count you read in a benchmark table.

Run the offline eval at every checkpoint and pick the checkpoint that maximizes per-call accuracy *and* keeps a general-capability probe (a small held-out instruction-following set) within a point or two of the base model. The best tool model that has forgotten how to talk to users is not the best model.

One operational detail that bites teams at the finish line: evaluate the model in the *exact* configuration you will serve. If you train a LoRA adapter and then merge it into the base weights for serving, run your offline eval against the merged checkpoint, not the adapter-on-base setup you trained with — merging is usually lossless, but "usually" is not "always," and a quantized serving path (for example, an AWQ or GPTQ export for vLLM) can shift argument-correctness by a point or two versus the bf16 checkpoint you validated. The same applies to the chat template: a mismatch between the template used in training and the one your serving stack applies at inference will silently degrade every call. The discipline is simple and non-negotiable — the artifact you bless on the eval must be byte-for-byte the artifact you deploy, decoded with the same template and the same precision.

## Step 5 — Preference optimization when SFT plateaus

SFT will take you most of the way, and for many cost-driven distillations it is the whole story. But on the hardest calls — ambiguous selections, arguments that require subtle disambiguation — SFT tends to plateau, because cross-entropy on a single gold answer does not teach the model *why* its confident wrong answer is worse than the right one. That is the gap preference optimization closes.

![A dataflow graph showing the SFT model sampling calls on hard prompts, an execute-and-judge step branching into chosen corrected calls and rejected wrong calls, which merge into a preference pair feeding a DPO update and an improved model](/imgs/blogs/fine-tuning-tool-calling-llms-when-how-7.png)

The elegant part, shown in the graph, is that the preference pairs are mined from the model's *own* mistakes. You take the SFT model, sample several calls on the hard slice of your eval, execute and judge each one, and pair a corrected call (chosen) against the model's own wrong call (rejected). These on-policy pairs are far more informative than synthetic ones, because they target exactly the errors this model actually makes.

[Direct Preference Optimization](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) then optimizes the model to prefer chosen over rejected without a separate reward model or an RL loop. The objective, for a pair of chosen response $y_w$ and rejected response $y_l$ given prompt $x$, is

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$$

where $\pi_\theta$ is the model being trained, $\pi_{\text{ref}}$ is the frozen SFT checkpoint, $\sigma$ is the logistic function, and $\beta$ (typically 0.1–0.5) controls how far $\pi_\theta$ is allowed to drift from the reference. The intuition is that the loss pushes up the relative log-probability of the correct call and pushes down the wrong one, anchored so the model does not wander off and forget everything the SFT phase taught it. Each row of the preference file carries a `prompt` (the chat-templated prefix), a `chosen` corrected call, and a `rejected` call sampled from the model itself:

```python
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

pairs = load_dataset("json", data_files="tool_pref_pairs.jsonl", split="train")

cfg = DPOConfig(
    output_dir="qwen7b-tools-dpo",
    beta=0.1,                          # KL anchor to the SFT reference
    learning_rate=5e-6,                # an order of magnitude below SFT
    num_train_epochs=1,                # one pass over preference pairs
    per_device_train_batch_size=4, gradient_accumulation_steps=8,
    bf16=True, max_length=4096, max_prompt_length=3072,
    loss_type="sigmoid",               # vanilla DPO; "ipo"/"kto" are alternatives
)
trainer = DPOTrainer(
    model="qwen7b-tools",              # the SFT checkpoint = both policy and ref init
    args=cfg, train_dataset=pairs,
)
trainer.train()
```

A realistic shape for this phase: SFT gets argument-correctness to roughly 88%, and a single DPO pass on a few hundred well-chosen self-generated pairs pushes it to the mid-90s, with most of the gain on the hard slice that SFT could not crack. Two cautions specific to tool calls. First, build the pairs so chosen and rejected differ *only* in the part you care about — same tool, differing arguments — or DPO will learn the easy distinction (tool name) instead of the hard one (argument values). Second, keep $\beta$ high enough that the model does not over-optimize the preference signal and start emitting degenerate calls that game your judge; watch the general-capability probe here too. The broader family of preference and RL objectives — IPO, KTO, GRPO — and when each fits is laid out in the [GRPO vs DPO vs PPO decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide); for tool calling, vanilla DPO on on-policy pairs is the high-leverage default.

The judge that labels chosen-versus-rejected deserves more care than it usually gets, because a sloppy judge teaches the model to game it. For tool calls you are lucky: execution is the most reliable judge available. A call that runs and returns the expected result is chosen; a call that errors or returns the wrong result is rejected — no subjective grading required. Lean on this. Reserve a model-based judge only for the cases execution cannot settle (two calls that both "work" but one is more appropriate), and when you do use one, spot-check its labels by hand, because a 90%-accurate judge injects 10% label noise straight into the preference signal. The rule of thumb: prefer an execution-grounded label over a model-graded one every time the tool can actually be run.

## Step 6 — Evaluate at two altitudes

You built the eval in Step 1; now you run it as a gate, not a vanity metric. The mistake to avoid is reporting a single end-to-end "task success" number, because it hides *where* the model breaks and gives you nothing to act on when it regresses.

![A grid of five evaluation gates a tool call must clear — schema-valid, right tool, right arguments, executes, task success — with the specific failure each gate catches shown below it](/imgs/blogs/fine-tuning-tool-calling-llms-when-how-8.png)

A tool call clears five gates in order, and each gate isolates exactly one failure mode, as the figure lays out. Reporting the rate at each gate — not just the final one — tells you precisely which failure mode your training is and is not fixing:

- **Schema-valid rate.** Does the output parse against the tool schema? If this is below 100% you have a decoding problem, and the fix is constrained decoding (Step 7), not more training.
- **Tool-selection accuracy.** Given a valid call, is it the right tool? This is the metric that reveals whether your small model's selection is good enough or whether retrieval is still doing the heavy lifting.
- **Argument correctness.** Right tool, right arguments — split into exact-match for IDs and enums and semantic-match for free text and dates. This is the metric a tool-calling fine-tune is most trying to move.
- **Executable rate.** Does the call run against the real tool without error? A call can be schema-valid and still reference a nonexistent ID.
- **Task success (trajectory).** End-to-end, did the whole interaction accomplish the goal in a sensible number of steps?

```python
from collections import defaultdict

def evaluate(model, eval_cases, sandbox):
    gates = defaultdict(lambda: defaultdict(int))   # slice -> gate -> count
    for case in eval_cases:
        s = case.slice
        gates[s]["n"] += 1
        out = model.generate_tool_call(case.messages)

        if case.expected_tool is None:              # abstention case
            gates[s]["abstain_ok"] += int(out.tool_call is None)
            continue

        call = try_parse(out)
        if call is None:                            # gate 1
            continue
        gates[s]["schema_valid"] += 1
        if call["name"] != case.expected_tool:      # gate 2
            continue
        gates[s]["right_tool"] += 1
        if not args_match(call["arguments"], case.expected_args, case.arg_match):
            continue                                # gate 3
        gates[s]["right_args"] += 1
        result = sandbox.execute(call)              # gate 4
        if result.error:
            continue
        gates[s]["executes"] += 1
        gates[s]["task_success"] += int(result.matches(case.expected_result))  # gate 5

    for s, g in gates.items():
        n = g["n"]
        print(f"[{s}] n={n} valid={g['schema_valid']/n:.0%} tool={g['right_tool']/n:.0%} "
              f"args={g['right_args']/n:.0%} exec={g['executes']/n:.0%} task={g['task_success']/n:.0%}")
    return gates
```

The stratification by slice is what makes this actionable. A model that is 96% on the `refunds` slice and 70% on `abstention` is telling you exactly where to spend the next round of data work. Run this on the base model, the SFT checkpoint, and the DPO checkpoint, and you get a clean before/during/after that justifies — or kills — the project on evidence rather than vibes. For the trajectory altitude, the techniques in [evaluating agent trajectories beyond the final answer](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer) extend this from single calls to whole interactions.

## Step 7 — Guardrail inference with constrained decoding

Even a well-trained model emits invalid JSON occasionally — a long-tail input, a high temperature, an unlucky sample. At a million calls a day, "occasionally" is thousands of broken calls. The fix is not more training; it is to make invalid output structurally impossible at decode time.

![A before-and-after comparison: unconstrained decoding can emit key typos, unbalanced braces, and forces retries on parse failure; grammar-masked decoding zeroes invalid tokens so JSON always parses with zero retries](/imgs/blogs/fine-tuning-tool-calling-llms-when-how-9.png)

Constrained decoding compiles the tool's JSON schema into a grammar, and at every decoding step it masks the logits of any token that would lead to a string the grammar cannot accept. The model samples only from schema-legal continuations, so a trailing comma or an unquoted key is not "unlikely" — it has probability exactly zero. The before/after in the figure is the entire pitch: unconstrained, the decoder can wander into malformed structure and you pay for retries; grammar-masked, the JSON always parses and the retry path disappears.

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

REFUND_SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": "integer"},
        "reason": {"type": "string", "enum": ["damaged", "late", "wrong_item", "other"]},
    },
    "required": ["order_id", "reason"],
    "additionalProperties": False,
}

llm = LLM(model="qwen7b-tools-dpo", guided_decoding_backend="xgrammar")
params = SamplingParams(
    temperature=0.0, max_tokens=128,
    guided_decoding=GuidedDecodingParams(json=REFUND_SCHEMA),
)
out = llm.generate([prompt], params)      # output is guaranteed to satisfy REFUND_SCHEMA
```

The important architectural point is that **constrained decoding and fine-tuning are complementary, not substitutes, and they fix different things.** The grammar guarantees *structure* — that the output is valid JSON matching the schema. It does nothing for *content* — picking the right tool, filling in the right argument values. Fine-tuning is the reverse: it improves selection and argument content but offers no structural guarantee. The strongest production setup uses both: a fine-tuned model for selection and argument quality, wrapped in a constrained decoder for an ironclad structural guarantee. Reaching for fine-tuning to fix JSON validity, when a grammar would do it for free and with certainty, is the most common case of climbing the expensive rung for a problem the cheap rung already solves.

Two honest caveats about the constraint, so you do not over-trust it. First, a grammar guarantees the output *matches the schema*; it does not guarantee the output is *correct*. A constrained decoder will happily emit a schema-valid `{"order_id": 1, "reason": "damaged"}` for an order that was late — the structure is perfect and the content is wrong, which is precisely why you still need a good model underneath. Second, constraints interact with quality: forcing the model down a legal-token path it assigned low probability to can degrade selection if the grammar is too tight, so constrain the *structure* (valid JSON, required fields, enum values) and leave the *content* (which tool, which values) to the model wherever you can. The clean division of labor is structure-by-grammar, content-by-model, correctness-checked-by-eval.

## What a tool-calling fine-tune actually costs

The decision framework hinges on cost amortization, so it helps to put rough numbers on the fixed cost you are amortizing. These are order-of-magnitude figures for a cost-driven distillation onto a 7B model with a stable surface of ten to twenty tools — your mileage varies, but the *shape* is consistent.

| Line item | Rough cost | Notes |
|---|---|---|
| Eval construction | 2–5 engineer-days | The highest-leverage spend; reused forever |
| Data manufacturing | 1–3 engineer-days + teacher API spend | Dominated by teacher generation and sandbox execution |
| SFT run (7B QLoRA) | A few GPU-hours on one 24GB card | Cheap; the surrounding work dwarfs it |
| DPO run | A few more GPU-hours | Only if SFT plateaus |
| Serving + constrained decoding | 1–2 engineer-days | One-time integration |
| Ongoing maintenance | Recurring | Retrain on schema changes, expand eval on new failures |

The pattern that surprises people is that the GPU cost — the part everyone fixates on — is the smallest line item. The real cost is the eval, the data pipeline, and the *ongoing maintenance*: a fine-tuned model is a dependency you now own and must keep in sync with a moving schema. This is exactly why low volume kills the case. If you are saving a few hundred dollars a month in inference, the maintenance alone eats the savings. If you are saving tens of thousands a month by moving a million daily calls off a frontier API, the entire program pays for itself in days and the maintenance is a rounding error.

The corollary is a sequencing rule: do the cheap, durable work — the eval, the data pipeline — first, because it is reusable and it de-risks everything downstream. The training runs themselves are cheap and fast once the scaffolding exists, and if the eval reveals that constrained decoding plus a better prompt already clears your bar, you have spent two days instead of two weeks to learn you did not need to train at all.

## A worked example: the refund agent, end to end

It helps to see the whole procedure run once, with numbers, on the running example: a high-volume support workflow served by a frontier model, fifteen stable tools, and a mandate to cut inference cost by moving to a self-hosted 7B. The frontier model handles the workflow at 96% per-call success; the 7B, zero-shot with good prompting, manages 79%. That seventeen-point gap is the entire project.

We start at Step 1 by building a 400-case offline eval from sampled production traffic, stratified across the fifteen tools with a dedicated 60-case abstention slice, and we pin every metric to code. Running it on the 7B zero-shot gives the diagnostic breakdown that tells us where the gap lives: schema-valid 94%, right-tool 88%, right-args 84%, executable 82%, task-success 79%. The schema-valid number says 6% of the gap is pure structure — and Step 7's constrained decoder takes that to 100% for free before we train a thing. With the grammar in place, the residual is selection (88%) and arguments (84%), which is what training can actually move.

Step 2 manufactures the data: the frontier model is the teacher, we sample four candidates per input across a few thousand seed inputs, execute every call in a sandbox, and keep only verified trajectories. The execution filter discards roughly 45% of teacher generations — a sobering reminder of how often fluent output is wrong — leaving about 3,500 clean trajectories, balanced by tool and including mined abstention negatives. Step 3 formats them through the Qwen chat template with `return_assistant_tokens_mask`, and a decode-the-mask assertion confirms loss flows only over the assistant calls and replies.

Step 4 runs a two-epoch QLoRA SFT on a single 24GB card in a couple of GPU-hours, mixing in 15% general instruction data. The checkpoint that maximizes eval accuracy without regressing the general-capability probe lands at right-tool 95%, right-args 91% — most of the gap closed. Step 5 attacks the stubborn remainder: we sample the SFT model on the hard, multi-constraint slice, build a few hundred on-policy preference pairs judged by execution, and a single DPO pass lifts right-args to 95% with the gain concentrated exactly where SFT stalled. Step 6's final report, run against the live schema and stratified by slice, shows the 7B-plus-grammar-plus-DPO stack matching the teacher within a point on this surface — at roughly a tenth of the serving cost.

The whole thing took about two weeks of mostly non-GPU work, and the single most valuable artifact produced was not the model — it was the eval, which now catches schema drift, guards against regressions, and grows with every new production failure. That ordering is the lesson of the entire article in miniature: the cheap, durable scaffolding is what makes the expensive training step both justifiable and safe.

## Cross-cutting concerns

Three issues cut across every tool-calling fine-tune and quietly sink projects that ignore them.

**Catastrophic forgetting and capability regression.** Training narrowly on tool calls degrades general ability — instruction-following, multi-turn coherence, even basic chat. LoRA limits the damage relative to full fine-tuning, and mixing 10–20% general data into the tool set limits it further, but the only real defense is *measuring* it: keep a general-capability probe in your eval and treat a regression on it as a failing gate, not an acceptable trade. A tool model that has forgotten how to talk is a downgrade no matter how good its `refund_order` calls are.

**Schema drift and versioning.** The model bakes the schema into its weights at training time. When the API team renames `order_id` to `orderId`, your model keeps emitting the old name and every call starts failing — and if your offline eval is pinned to stale fixtures, it stays green while production burns. Defenses: version your tool schemas explicitly, pin the eval to the *live* schema (not a snapshot), and add a contract test that diffs the schema the model was trained on against the schema currently deployed. Schema stability was Condition 2 of the decision framework for exactly this reason; if the schema is going to move, the maintenance cost of retraining usually outweighs the accuracy.

**Eval contamination.** If your teacher generated both training and eval trajectories from the same prompts, your eval is contaminated and your numbers are inflated. Split by *input* before generation, never after, and hold out a slice of inputs the teacher never saw. The cheapest way to fool yourself is to evaluate on paraphrases of the training set and conclude the model generalizes.

**Tool-result context growth.** Multi-step trajectories accumulate tool results in the context, and those results can be large — a search that returns ten documents, an API that returns a verbose JSON blob. If your training trajectories are short but production trajectories run ten calls deep, the model meets context lengths at inference it never saw in training, and tool-selection quality degrades as the relevant tools scroll out of the attention window. Defenses: match the length distribution of training trajectories to production, summarize or truncate verbose tool results before they re-enter the context, and keep an eval slice of long, many-call trajectories so the degradation is visible in a dashboard rather than discovered in an incident.

## Case studies from production

Patterns are easier to internalize as scars. Here are eight tool-calling incidents — composites drawn from real engagements — each with the symptom, the wrong first hypothesis, the actual root cause, and the lesson.

### 1. The 200-tool context bloat

A platform team exposed roughly two hundred internal tools to a single agent and watched tool-selection accuracy crater while latency tripled. The first hypothesis was that the model "couldn't learn" two hundred tools, and a fine-tuning project was scoped to "teach the model the tool inventory." It would have been a waste. The root cause was that all two hundred schemas were being stuffed into every prompt, consuming most of the context window and burying the task. The fix was a retrieval layer that surfaced the five to ten relevant tools per request, which recovered selection accuracy and cut latency without a single gradient step. The team did eventually fine-tune — but only on argument formatting for the twenty highest-traffic tools, after retrieval had solved selection. The lesson: fine-tuning to memorize a tool *inventory* is the wrong axis; inventory is a retrieval problem, and only argument *formatting* on the hot path was a training problem.

### 2. Schema drift broke it overnight

A refund agent that had passed every offline test for a month started failing roughly a third of its calls in production overnight, with no deploy on the agent side. The team's first move was to schedule a retraining run, assuming the model had somehow regressed. It had not. The API team had renamed a parameter from `reason` to `refund_reason` in a minor release, and the fine-tuned model — with the old name baked into its weights — kept emitting `reason`, which the API now rejected. The offline eval stayed green because it ran against a fixture captured before the rename. The fix was immediate (a serving-side argument rename) and the durable fix was structural: pin the eval to the live schema and add a contract test. The lesson: a fine-tuned model is coupled to the schema it trained on, and your eval must be coupled to the schema in production, not a snapshot.

### 3. The hallucinated `escalate_to_human`

A support agent began calling a tool named `escalate_to_human` that did not exist in its toolset, but only on unusual inputs — angry customers, edge-case requests. The team assumed the model needed *more* examples of the real tools. More positive examples did nothing, because the problem was not what the model did when a tool applied; it was what it did when *none* did. The SFT set contained only positive calls, so the model had learned that every input deserves some call, and on out-of-distribution inputs it confabulated a plausible one. The fix was abstention data: hundreds of hard-negative examples where the correct behavior was to ask a clarifying question or answer directly. The lesson: a model trained only on positive tool calls will hallucinate tools on inputs where it should call nothing — you must train the *absence* of a call as explicitly as its presence.

### 4. Forgot how to chat

A team full-fine-tuned a 13B model on a large, clean tool-calling set and shipped it. Tool metrics were excellent; then complaints rolled in that the assistant had become curt, literal, and bad at ordinary multi-turn conversation. The hypothesis was a prompt regression. The cause was catastrophic forgetting: full fine-tuning on a narrow distribution had overwritten the general instruction-following the base model came with. The fix had two parts — switch to a LoRA adapter so the base weights stayed intact, and mix 15% general instruction data into the tool set — after which tool accuracy held and conversational quality returned. The lesson: full fine-tuning on a narrow tool distribution is the fastest way to forget everything else; prefer adapters and always keep a general-capability probe in the eval.

### 5. Loss on the tool outputs

A model fine-tuned to call a pricing API started *fabricating prices* — returning confident numbers without ever emitting the tool call. The team suspected the model "didn't know it had a tool." The real cause was in the data pipeline: the loss mask was broken, and the model had been trained on the tool-result turns as if they were assistant output. It had dutifully learned to predict what the pricing API returns instead of learning to call it. Decoding the trained positions revealed tool results in the loss, which should never be there. One corrected mask and a re-run fixed it. The lesson: if loss flows over tool-result turns, you are teaching the model to hallucinate its own tools; assert the mask on a real example before every training run.

### 6. Latency forced a 7B

A product was serving a frontier model for a narrow, high-volume tool-calling workflow and the per-call cost was unsustainable. The team wanted to move to a self-hosted 7B but found its zero-shot tool-calling accuracy was ten-plus points behind the frontier model — not good enough to ship. This is the textbook case where fine-tuning is the right answer. They distilled: used the frontier model as a teacher, generated and execution-filtered a few thousand trajectories over the workflow's fifteen tools, QLoRA-tuned the 7B, and closed the gap to within a point on that surface — at a fraction of the serving cost. The lesson: the strongest reason to fine-tune tool calls is cost-driven distillation onto a smaller model for a narrow, stable surface; "fix the errors" is a weak motivation, "match the teacher at one-tenth the cost" is a strong one.

### 7. DPO on self-generated negatives

After a successful distillation, a team's 7B was at 88% argument-correctness and stuck — another epoch of SFT only overfit. The hardest slice (ambiguous multi-constraint queries) would not budge. Rather than collect more SFT data, they sampled the SFT model on the hard slice, executed and judged each completion, and built preference pairs from the model's own correct-versus-wrong calls. A single DPO pass over a few hundred such pairs lifted argument-correctness to 95%, almost entirely on the previously stuck slice. The lesson: when SFT plateaus, the remaining signal is in the model's *own* mistakes — on-policy preference pairs target exactly the errors more SFT data cannot.

### 8. Constrained decoding made format-tuning unnecessary

A team scoped a fine-tuning project whose primary goal was "make the model produce valid JSON," because roughly 4% of calls were malformed and breaking downstream parsing. Before training, an engineer wired up grammar-constrained decoding as a stopgap — and the malformed-JSON rate went to zero immediately. The structural problem was solved for free, which changed the project's justification entirely: the residual issue was tool *selection* on a small model, a genuine training problem, but a much smaller one than "fix all the JSON." They fine-tuned for selection and kept the grammar for structure. The lesson: separate structure from content before scoping a fine-tune — constrained decoding owns structure completely, so never spend training budget on JSON validity.

### 9. Packing silently merged turns

A team's tool fine-tune produced a model that occasionally answered the *previous* user's question — bizarre, intermittent, hard to reproduce. The first hypothesis was a serving-side context bug. The cause was in training: the trainer's default sequence packing concatenated multiple short trajectories into each training sequence to improve throughput, and the packing had quietly merged turns across example boundaries while the assistant mask was computed per-example. The model had learned, faintly, to attend across what should have been independent conversations. Disabling packing for the tool data fixed it, at a throughput cost the team gladly paid. The lesson: sequence packing and per-turn loss masking are in tension; for tool-calling data, turn packing off unless you have verified the mask survives it.

### 10. Temperature in production undid the training

A carefully fine-tuned model that scored 95% on the offline eval was selecting the wrong tool 8% of the time in production. The eval ran at temperature 0; production ran at temperature 0.8, inherited from a generic chat config. The sampling noise was reintroducing exactly the wrong-tool errors the fine-tune had trained out. Dropping the tool-calling path to temperature 0 — and wrapping it in constrained decoding — closed the gap immediately. The lesson: evaluate at the temperature you serve at, and for tool selection prefer greedy or near-greedy decoding; creativity is a liability when the output space is a fixed set of tools.

### 11. The eval that graded paraphrases

A model showed a suspiciously high 99% argument-correctness on the offline eval, and the team nearly shipped it as done. A skeptical reviewer noticed the eval inputs were paraphrases of training inputs — the same teacher had generated both from the same seed prompts, split *after* generation. On a freshly held-out slice the real number was 84%. The contamination had inflated the metric by fifteen points. Re-splitting by seed input before any generation, and holding out inputs the teacher never touched, restored an honest number. The lesson: split by input before you generate, never after; an eval built from paraphrases of the training set measures memorization, not the generalization you actually care about.

### 12. Distilled the calls, lost the planning

A team distilled a multi-step research agent — search, then read, then synthesize, then cite — from a frontier model onto a 7B for cost. On single-call evals the student matched the teacher; in production, multi-step task success collapsed. The student had learned each individual call beautifully but had not absorbed the teacher's *planning* — when to stop searching, when it had enough to answer. The fix was not more tool data; it was to keep the planning on a larger model (or an explicit planner) and use the cheap fine-tuned student only for the individual, well-specified calls it was good at. The lesson: distillation transfers narrow, well-specified behavior far better than open-ended reasoning — fine-tune the calls, but do not assume the small model inherits the orchestration that made the teacher good.

## Reach for fine-tuning when…

- You are **distilling a frontier model onto a smaller, cheaper one** for a narrow, stable tool surface at high volume, and the small model's zero-shot accuracy is not good enough.
- The dominant residual failure is **argument formatting** — a domain-specific mapping from natural language to a structured convention — and it survives crisp descriptions, few-shot examples, and constrained decoding.
- You need **reliable abstention** — knowing when *not* to call a tool — and few-shot prompting has not produced it consistently.
- All three decision conditions hold together: **prompting has plateaued against a real eval, the schema is stable, and volume justifies the fixed cost.**
- You have an **execution-grounded data pipeline and a two-altitude eval** already built, so the run is measurable.

## …and skip it when

- The failure is **malformed JSON or any structural issue** — use constrained decoding; it is free, deterministic, and total.
- You have **many tools and a selection problem** — use retrieval to shrink the in-context tool set before you consider training.
- The **schema is still churning** — a fine-tuned model couples to the schema, and you will be on a retraining treadmill.
- You have **no eval** — you cannot tell whether prompting has plateaued or whether training helped, so you are flying blind.
- **Volume is low** — the fixed cost of data, training, eval, and maintenance will not amortize, and constrained decoding plus a good prompt gets you most of the reliability for none of the upkeep.
- You **have not yet rewritten the tool descriptions** — the cheapest rung is still untried, and it closes most "wrong tool" bugs on its own.

The throughline of every one of these conditions is the same: fine-tuning tool calls is a real and sometimes indispensable tool, but it is the fifth rung for a reason. Earn your way up the ladder, measure relentlessly, and reserve the GPU for the narrow, well-scoped residual that only training can close.

## Further reading

- [Effective LLM Fine-Tuning Techniques](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques) — the general toolkit (LoRA, QLoRA, DoRA, SFT, schedules) this article builds on.
- [Fine-Tuning LLMs with DPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) — the preference-optimization phase in depth.
- [GRPO vs DPO vs PPO: A Decision Guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) — choosing the preference/RL objective.
- [Advanced Tool Use](/blog/machine-learning/ai-agent/advance-tool-use) and [Model Context Protocol](/blog/machine-learning/ai-agent/model-context-protocol) — the retrieval-and-discovery rungs below fine-tuning.
- [Evaluating Agent Trajectories Beyond the Final Answer](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer) — the trajectory altitude of evaluation.

---
title: "Debate and Critique: Using Multiple Agents to Improve Answer Quality"
date: "2026-06-27"
description: "How multi-agent debate, critique, and adversarial review patterns improve LLM output quality — the evidence, implementation patterns, failure modes, and when diversity of perspective actually helps."
tags: ["ai-agents", "multi-agent", "debate", "critique", "quality", "llm", "machine-learning", "production-ml"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 41
---

There is a structural problem at the heart of every single-agent LLM system: the model that produces an answer is the same model that would need to catch the mistakes in that answer. When you ask GPT-4 to check its own work, you are asking it to find the blind spots it just expressed. Unsurprisingly, it often does not.

The multi-agent critique family of patterns — critic-reviser loops, adversarial debate, society-of-mind voting, constitutional self-critique — address this by introducing a separation of concerns between generation and evaluation. One agent makes claims; another attacks them. The result, under the right conditions, is demonstrably better output than any single agent produces alone.

The diagram below is the mental model: a loop where a generator, a critic, and a reviser each hold exactly one job, and a quality gate decides whether another round is needed.

![Critic-Reviser Loop](/imgs/blogs/debate-and-critique-multi-agent-1.webp)

This post covers all four major critique patterns in depth, the conditions under which they help (and when they hurt), how to build a reliable judge agent, the cost math behind multi-round critique, a ~70-line Python implementation, and six detailed case studies from real systems.

## 1. The Sycophancy Problem: Why Single Agents Agree with Themselves

Before building a solution, it is worth understanding exactly what failure mode we are solving.

A large language model is trained to produce text that scores high on human preference. Human raters, it turns out, like confident answers that agree with their framing. They like fluency. They give high marks to responses that feel complete and authoritative, even when those responses contain subtle factual errors or ignore obvious counterarguments. The model has internalized this preference gradient. The result is a systematic bias toward answers that agree with the user's premise, avoid controversial qualifications, and present confident conclusions — even when uncertainty is the honest response.

This is called **sycophancy**, and it is not a bug in any particular model. It is a predictable outcome of RLHF training on human feedback at scale. Multiple Anthropic and OpenAI research papers have documented it. The canonical demonstration is: tell GPT-4 a wrong answer and ask if it agrees, vs. just ask the question cold. In the first condition, the model agrees with the wrong answer at a substantially higher rate — roughly 20-30 percentage points higher across a range of factual domains.

Sycophancy compounds in ways that matter for production systems:

**Premise amplification.** If the user frames a question with a false premise ("given that X is true, why does Y follow?"), a single agent almost always accepts the frame and answers within it. It takes significant prompting discipline to get a model to reject the premise cleanly. The model treats the premise as an affordance — it has been trained to be helpful, and helping means working within the user's stated frame.

**Confidence calibration failure.** Single agents tend to express high confidence on questions where they should express high uncertainty. The model does not have a reliable internal signal for "I am making this up vs. I actually know this." The output fluency is identical whether the model is reciting a memorized fact or confabulating a plausible-sounding one. A model that has seen thousands of confident authoritative texts about a domain will generate confident authoritative text about that domain even when it is wrong.

**Self-consistency bias.** When you ask a model to review its own output, it has already committed to a rhetorical and factual position. The attention mechanism has established strong associations between the claims in the output and the supporting reasoning. Reviewing the output amounts to asking "is what I just said correct?" — and the same associations that produced the original answer will tend to validate it. Studies of LLM self-evaluation show that models disagree with their own outputs at rates far lower than would be expected from a calibrated evaluator.

**Missing counter-evidence.** A model answering a question will not systematically search for evidence that contradicts its answer. It answers with what comes to mind first, which is shaped by training distribution, and does not check whether counterexamples exist. This is a problem both for factual questions (where the counterexample might be the correct answer) and for reasoning tasks (where a counterexample to an intermediate step would invalidate the whole chain).

**Anchoring on user phrasing.** The exact words the user uses to phrase a question influence the model's answer in ways unrelated to the underlying question. A question phrased as "why is approach A better than approach B?" will receive an answer defending A more strongly than a question phrased as "compare approaches A and B." The model is following the pragmatic implicature of the phrasing. A dedicated critic agent, receiving only the draft answer without the original question's framing, is not subject to this anchoring.

Multi-agent critique breaks the self-consistency bias by introducing a structurally different agent — one that has not already committed to a position — to evaluate the output. The separation of roles does not fix sycophancy in the critic (it too will have biases), but it changes which biases apply and when, which reduces the probability that the same error propagates through both stages. The critic is not anchored to the generator's rhetorical position. It arrives at the draft cold and can evaluate it more independently.

This is not magic. The critic is still an LLM with all the usual limitations. But the structural separation is valuable: the generator's job is to produce a fluent, comprehensive answer; the critic's job is to find specific, actionable problems. These are genuinely different cognitive modes — generation vs. evaluation — and splitting them across two agents allows each to be optimized for its mode.

## 2. Multi-Agent Debate: The Original Liang et al. 2023 Setup and What It Proved

The most influential early work on this topic is "Encouraging Divergent Thinking in Large Language Models through Debate" (Du et al., 2023, MIT / Google Brain) and the related "Can LLMs Express Their Uncertainty?" line of work. The setup is conceptually simple:

1. Present the same question to multiple LLM instances independently.
2. Each instance sees the other's answer.
3. Each instance updates its answer given the new information.
4. Repeat for several rounds.
5. Aggregate the final answers.

Du et al. showed that across a range of benchmarks — math reasoning (GSM8K), factual question answering, and commonsense reasoning — multi-agent debate improved accuracy over single-agent baselines. The gains were largest on tasks requiring multi-step reasoning: roughly 5–10 percentage points on GSM8K-style problems when using three agents over two rounds.

The mechanism is interesting. Each agent sees an alternative answer and must either update its own answer or defend why the alternative is wrong. This forces explicit reasoning about the comparison — a kind of forced deliberation that single agents skip when they generate answers in one pass. The reasoning is not just "produce an answer" but "evaluate my answer against this competing answer and decide which is better." This is a higher-order cognitive operation and the models perform it better than the equivalent single-pass reasoning.

Several follow-up findings qualified these results:

**Model diversity matters more than agent count.** Using three instances of the same model with the same temperature produces smaller gains than using one GPT-4 and one Claude, because the correlated errors do not get challenged. When model A and model B were trained on the same data and fine-tuned with similar RLHF procedures, they tend to make the same systematic errors. Debate between them produces high-confidence wrong answers on the questions they both get wrong, not a correction. The debate is most valuable when the agents have genuinely different knowledge or reasoning patterns.

**The benefit is concentrated on hard questions.** On questions where the correct answer is unambiguous and the model already knows it, debate adds noise rather than signal. The model that was correct before the debate sometimes gets talked out of the right answer by a less-capable debater. The 2023 paper reports that on "easy" questions (where single-agent accuracy exceeded 90%), debate hurt performance by 2-3 percentage points on average. The benefit was concentrated in the 40-70% accuracy range — questions hard enough to benefit from multiple perspectives but not so hard that both agents were wrong.

**Verbosity inflation.** In multi-round debate, each agent tends to produce longer and longer responses as it incorporates the other's points. The token cost grows superlinearly with rounds. By round 3, a debate that started with 200-word answers has typically inflated to 500-800 words per agent, most of which is hedging, repetition, and meta-commentary about the debate rather than substantive new argumentation.

**Round 2 is the sweet spot.** The accuracy curve in most studies shows the largest gain from the baseline to round 1, a meaningful but smaller gain to round 2, and flat or declining performance after that. Two rounds capture roughly 80% of the achievable gain for roughly 40% of the final-round cost.

The adversarial debate topology with full cross-reading and rebuttal looks like this:

![Adversarial Debate Topology](/imgs/blogs/debate-and-critique-multi-agent-2.webp)

What the research shows is that debate works when it is genuinely adversarial — when Agent B is instructed to argue the opposite position, not to "add your perspective." The latter produces convergent answers (everyone agrees, the answer becomes longer, and the confidence goes up without the correctness improving). The former produces the contention that is actually useful.

The most practically important finding from the 2023-2024 research wave: **heterogeneous models outperform homogeneous ensembles at every agent count**. If you are choosing between deploying two instances of the same model or one instance of two different models, choose heterogeneous. The diversity of the reasoning, not the volume of reasoning, is what produces quality gains.

## 3. Critic-Reviser Pattern: One Agent Generates, Another Critiques, First Revises

The critic-reviser pattern is the most practically deployable form of multi-agent critique. It does not require multiple debate rounds or a judge; it requires exactly two agents operating in sequence, plus a loop controller.

**Generator**: Produces an initial draft given the task. The generator should be the strongest model you have for the generation task — its job is to produce the best possible first draft.

**Critic**: Receives the draft and a structured rubric. Outputs a critique: a list of specific problems, scores on each criterion, and suggestions for improvement. Crucially, the critic does not rewrite — it only identifies failures. The critic can be a different model, or the same model with a different system prompt, or the same model with the draft presented as "evaluate this text you received."

**Reviser**: Receives the original draft plus the critique. Produces a revised draft that addresses the identified problems. The reviser has full freedom to restructure, delete, and add content — it is not required to preserve any of the original draft.

**Quality gate**: Compares the critic's scores against a threshold. If all scores exceed the threshold, the loop exits and returns the current draft. If any score is below threshold, the loop runs another round (up to a configured maximum).

The separation between critic and reviser is load-bearing. If you collapse them into one agent ("critique this and then revise it"), you get a weaker result because the same agent writes the critique knowing it will have to execute on it — which creates an incentive to produce mild, easy-to-satisfy critiques. A separate reviser is unconstrained by what the critic said it would do; it can make more dramatic changes if that is what the critique calls for. The reviser can also disagree with the critique — it can note "the critic suggested X but I am not making that change because Y" — which is a valuable signal for debugging the pipeline.

The contrast between single-agent and post-critique output quality is stark enough to diagram:

![Single-Agent vs Post-Critique Answer](/imgs/blogs/debate-and-critique-multi-agent-3.webp)

The key design decisions in a critic-reviser setup:

**Rubric design.** An unstructured critic ("tell me what is wrong with this") produces inconsistent, vague feedback. A structured rubric — factual accuracy (1-5), logical coherence (1-5), completeness (1-5), tone (1-5), citation quality (1-5) — produces feedback the reviser can act on. The rubric dimensions should match the quality attributes that matter for your specific task. Spending time up front designing the rubric is the highest-leverage investment in a critique system; a weak rubric produces a weak critic regardless of model quality.

**Critic temperature.** The critic should be run at a slightly higher temperature than the generator. The generator wants a focused, coherent draft — a low-temperature sample. The critic wants to identify diverse failure modes — a somewhat exploratory sampling distribution is appropriate. In practice, temperature 0.7-0.9 for the critic vs. 0.3-0.5 for the generator works well.

**Stopping condition.** You need a numeric threshold or a maximum round count to prevent infinite loops. A critic without a stopping condition will eventually always find something to flag, even in high-quality outputs. A score of 4.0/5.0 average across rubric dimensions is a reasonable default threshold. If your rubric has a safety dimension, consider a strict gate: any safety score below 5.0 triggers a revision regardless of average score.

**Reviser instruction framing.** The reviser prompt must explicitly tell it that the critique identifies real problems and that it should make substantive changes, not cosmetic ones. Without this instruction, LLM revisers tend to make surface-level edits ("I rephrased paragraph 2 to be more concise") rather than addressing the structural issues the critic identified. A useful framing: "The critique above identifies serious problems with the draft. Your revised answer should substantially address each problem in must_fix. If a problem requires restructuring the entire answer, do so."

**Critic specialization.** For tasks with multiple risk dimensions, consider running multiple specialized critics in parallel rather than one general critic. A code review task might have a security critic (focused on vulnerabilities), a correctness critic (focused on logic errors), and a style critic (focused on readability). This catches more problems than one general critic and produces cleaner, more targeted revision instructions.

## 4. Adversarial Debate: Two Agents Argue Opposite Positions, Judge Decides

Adversarial debate is a more structured form of multi-agent review, borrowed from deliberative democracy and legal systems: two advocates argue opposing positions to a neutral judge who evaluates both arguments.

The setup:

1. **Position assignment.** Agent A is assigned to argue position P; Agent B is assigned to argue position not-P. This assignment is explicit and persistent — neither agent is allowed to concede the position mid-debate. Both agents are told their assignment up front. The assignment itself is the most important design decision: the quality of the debate is limited by the quality of the position pair.

2. **Opening arguments.** Each agent writes an opening statement, typically 200-400 words, presenting its strongest case. Agents are instructed to cite specific evidence, name specific mechanisms, and make specific predictions — not to make general philosophical arguments. The instruction matters: without it, models default to vague hedged arguments that are hard for the judge to evaluate.

3. **Cross-reading.** Each agent reads the other's opening statement. The cross-reading step forces the agent to engage with the opposing argument before rebutting it, which prevents the common failure mode of a "rebuttal" that simply repeats the opening argument without addressing the opponent's points.

4. **Rebuttal round.** Each agent writes a rebuttal specifically addressing the weaknesses in the opposing argument. "Addressing weaknesses" is key — the instruction should be explicit: "identify the two or three strongest points in the opposing argument and explain specifically why each is wrong or overstated." This produces a more useful rebuttal than "argue the other side is wrong."

5. **Judge evaluation.** A separate judge agent reads both opening statements and rebuttals. It evaluates based on explicit criteria — factual accuracy of claims made, quality of logical reasoning, handling of counterevidence — and issues a verdict with per-criterion scores and a written rationale. Judge design is covered in Section 9.

The structural advantage of forced position assignment is that it bypasses the tendency of LLMs to converge. If you just ask two agents "what do you think of X?", they will both produce balanced takes that agree with each other and agree with the user's apparent position. If you assign one to argue pro and one to argue con, you force each to explore the strongest version of one side — which surfaces arguments and evidence that a balanced single-agent answer would skip.

**When adversarial debate shines:**

- Complex multi-factor decisions where reasonable people genuinely disagree (system design tradeoffs, policy questions, architecture choices, hiring decisions)
- Tasks where the quality metric is argumentation quality rather than factual correctness — legal reasoning, strategic planning, debate preparation
- Cases where you suspect the initial framing is biased and want to stress-test it from both directions
- Decisions with significant irreversibility, where the cost of a confident wrong answer justifies the 4-8x token overhead

**When adversarial debate fails:**

- Factual questions with a single correct answer — assigning an agent to argue the wrong answer produces a misleading transcript and the judge may be fooled by a confident wrong argument
- Simple tasks where the cost of two full argument chains far exceeds the value of the marginal quality improvement
- When both agents share the same training distribution and end up making identical arguments from opposite directions (the correlated-model failure mode — both argue their assigned position using the same evidence)
- When the "position" is not actually a position but a framing ("argue that Python is better vs. argue that Go is better for a CRUD API" — both agents know Python and Go are both fine, the debate is forced and both agents will produce bad-faith arguments)

One important nuance: adversarial debate is not primarily about picking a winner. The verdict is useful, but the debate transcript itself is often more valuable. Engineers who use debate for architecture decisions frequently report that the transcript — specifically the rebuttals, where each side attacks the other's weakest points — surfaces considerations that no single-agent prompt would have uncovered. The transcript is the output; the verdict is a summary.

## 5. Society of Mind: Many Agents with Different Personas Vote

Marvin Minsky's "Society of Mind" thesis proposed that intelligence emerges from the interaction of many simple, specialized agents. The multi-agent critique implementation of this idea assigns different system-prompt personas to a set of agents and aggregates their independent responses.

![Society of Mind: Multi-Persona Voting](/imgs/blogs/debate-and-critique-multi-agent-4.webp)

A typical society-of-mind setup for a complex question-answering task:

- **Agent 1 (Skeptic)**: Instructed to identify what could go wrong, what assumptions are unfounded, and what risks are being ignored. The skeptic's value is in identifying overconfident claims and missing caveats.
- **Agent 2 (Devil's Advocate)**: Instructed to argue the strongest case against the proposed answer or approach. Unlike the skeptic who just flags problems, the devil's advocate must construct an alternative position.
- **Agent 3 (Optimist)**: Instructed to identify the most favorable interpretation and what the approach gets right. The optimist prevents the critique from being purely negative — it surfaces what is correct and worth preserving.
- **Agent 4 (Domain Expert)**: Given a domain-specific system prompt (e.g., "you are a senior security engineer with 15 years of experience in distributed systems") and instructed to evaluate from that perspective. The domain expert's specialization is the primary differentiator from the other agents.
- **Agent 5 (Risk Analyst)**: Instructed to quantify and rank risks, focusing on tail cases and failure modes under edge conditions. Where the skeptic identifies that something could go wrong, the risk analyst estimates how badly and under what conditions.

Each agent responds independently to the same task — they do not see each other's outputs before producing their own. An aggregator — which can be a simple voting function, a weighted average of scores, or another LLM — synthesizes the responses into a final answer.

The aggregation strategy matters significantly and is a design decision in its own right:

**Majority vote** works well for classification tasks where each agent's response is a discrete choice (approve/reject, A/B/C/D, high/medium/low risk). It does not work well for open-ended generation, where the outputs are not commensurable — you cannot take a majority vote of three 400-word answers that agree on the main point but differ on emphasis and caveats.

**Minimum-score filtering** uses the lowest per-criterion score across agents as the quality signal. If any agent gives a low score on any criterion, the answer fails. This is a conservative, pessimistic aggregation appropriate for high-stakes tasks where any single agent's concern is worth taking seriously (safety reviews, medical information, legal guidance).

**LLM-as-synthesizer** passes all agent outputs to a fresh LLM with a structured synthesis prompt: "You have received five independent assessments of the following question. Synthesize them into a single comprehensive answer that: (1) addresses the main question directly, (2) incorporates the strongest points from each assessment, (3) acknowledges significant disagreements between assessments." This preserves nuance and handles conflicting outputs gracefully, but adds cost and another LLM hop.

**Weighted by persona reliability** assigns different weights to different personas based on empirical calibration against a labeled evaluation set. For a code security review task, the Domain Expert's critique is weighted 3x the Optimist's. For a creative writing evaluation, the weights invert. This requires maintaining calibration data per task type, which adds operational overhead but improves result quality.

The diversity of personas is what makes society-of-mind distinct from simple majority vote with identical agents. The personas force the sampling distribution of each agent into a different region of the response space. The Skeptic will notice different problems than the Optimist because its system prompt activates different evaluation heuristics. That diversity of perspective is the signal.

One important implementation note: persona injection via system prompt is imperfect. An LLM with a "Skeptic" system prompt does not behave identically to a genuine domain expert skeptic — it imitates what it predicts a skeptic would say, which is shaped by training data about skeptics, not by actual domain expertise about the problem at hand. Persona diversity is a heuristic for sampling-space diversity, not a guarantee of epistemic diversity. For high-stakes tasks, replace persona diversity with architectural diversity: different base models, different retrieval sources, different fine-tuning.

## 6. Constitutional AI Critique: Self-Critique Guided by Principles

Constitutional AI (Anthropic, 2022) introduced a self-critique mechanism where a model critiques its own output against a written set of principles — the "constitution." The procedure:

1. Generate an initial response.
2. Prompt the model to critique the response against each principle in turn: "Does this response violate principle X? If so, how?"
3. Prompt the model to revise the response to fix any violations it identified.
4. Optionally repeat.

This is formally single-agent (one model evaluating its own output), but the constitutional framing provides structure that pure self-critique lacks. Instead of asking "what is wrong with this?", it asks "does this violate principle P?" — a binary check against a specific criterion. The model is more reliable at this specific check than at open-ended self-evaluation. The principle provides a concrete, fixed reference point that the model can anchor its evaluation to, rather than requiring it to generate both the evaluation criteria and the evaluation simultaneously.

Constitutional critique works best in the safety domain, exactly where Anthropic developed it. The principles are things like "does not provide instructions for creating weapons," "does not produce content that could be used to harm a specific person," "acknowledges uncertainty rather than confabulating," "does not make promises about capabilities the system does not have." These are checking for specific failure modes against explicit criteria, which is the kind of task LLMs handle relatively well.

The limitation is that the constitution must be written in advance, and it can only check for violations of principles it knows about. Novel failure modes that the constitution does not anticipate will not be caught. If the constitution has 12 principles and the answer has a problem on dimension 13, the constitutional critique will give it a clean bill of health. Constitutional critique is strong for known, enumerable risk dimensions and weak for open-ended quality assessment.

**Designing an effective constitution**: The principles should be written as specific behavioral constraints, not vague values. "Be accurate" is not a useful principle — it is not checkable. "Does not state a specific factual claim as definitely true if that claim is contested in peer-reviewed literature" is checkable. "Does not give a specific dosage recommendation without noting that dosage depends on individual factors and recommending consultation with a healthcare provider" is checkable.

**Implementation note**: A useful refinement is to have a separate model apply the constitution rather than the model that produced the output. This converts constitutional self-critique into a constitutional peer-critique, which removes the self-consistency bias. The critic model is still applying the same constitution, but it has not already committed to the response and may apply the principles more strictly. In Anthropic's original CAI work, this is done implicitly — the constitution is applied in a separate generation pass, effectively a new agent call — but making it architecturally explicit (a distinct model with a critic system prompt) produces more reliable results.

**Constitutional critique vs. rubric-based critique**: The difference is in what the criteria are checking. A rubric checks for positive quality properties (factual accuracy score, completeness score). A constitution checks for the absence of negative properties (does not violate principle P). Both are useful; they catch different problems. The combination — a rubric for positive quality dimensions plus a constitution for safety and compliance dimensions — covers more of the risk surface than either alone.

## 7. When Diversity Helps: The Conditions Under Which Multi-Agent Critique Adds Value

Multi-agent critique is not universally beneficial. There are specific conditions under which it adds value, and conditions under which it is wasteful or counterproductive. Understanding these conditions prevents cargo-culting: adding agents because it sounds sophisticated rather than because the structure of the task warrants it.

![When Diversity Helps vs Task Type](/imgs/blogs/debate-and-critique-multi-agent-8.webp)

**Diversity helps when: the task has multiple valid evaluation dimensions**

If quality is a single scalar (factual accuracy of a specific claim), diversity of perspective does not help — all agents should converge on the same answer if they are well-calibrated. If quality is a vector of dimensions (factual accuracy, logical coherence, tone, completeness, safety), different agents can specialize in evaluating different dimensions, and their combined signal is more reliable than any single agent's holistic judgment.

**Diversity helps when: the answer space is large and adversarial**

Complex reasoning tasks — multi-step math, legal analysis, strategic planning — have many branching decision points where errors can compound. A second agent reading the chain of reasoning with fresh eyes is more likely to notice a flawed step than the agent that produced the chain. The benefit scales with answer complexity.

**Diversity helps when: the agents have genuinely different knowledge or capabilities**

A mixture of GPT-4 and Claude-3 Sonnet will catch each other's blind spots more reliably than two identical GPT-4 instances. Different training distributions mean different systematic biases. When one model's systematic error is another model's reliable capability, the ensemble outperforms either alone.

**Diversity helps when: the cost of errors is high**

For a high-stakes task — medical information, legal advice, security recommendations — the incremental cost of adding a critic loop is small relative to the cost of a confident wrong answer. The ROI of critique scales with stakes.

**Diversity helps when: the task is subjective or multi-stakeholder**

A code review, a strategic recommendation, a design critique — these are tasks where a single perspective is inherently incomplete. Multiple agents with different personas systematically surface a wider range of considerations than any single agent.

## 8. When It Doesn't Help: Homogeneous Models, Factual Tasks, Over-Refusal Spiral

![Critique Patterns vs Task Types](/imgs/blogs/debate-and-critique-multi-agent-5.webp)

**Same model, same prompt: no diversity, no benefit**

Running GPT-4 five times with the same system prompt and aggregating the results gives you a confidence interval around GPT-4's answer, not a diverse set of perspectives. The errors are correlated. The model will make the same mistake five times with slightly different phrasing. This is the most common multi-agent antipattern: assuming that "more agents = more quality" regardless of whether those agents are actually diverse.

The tell-tale sign of this failure mode is that the debate converges very quickly — typically within round 1 — to a high-confidence consensus answer. Real debate should maintain contention for at least two rounds before converging. Rapid consensus in a debate system usually means the agents are not diverse enough to challenge each other.

**Factual tasks with single ground truth**

For questions with a single correct answer that the model either knows or does not know — "what is the capital of France?", "what is the output of this Python function?", "what year did the transistor get invented?" — debate adds no signal. The debater assigned to argue the wrong answer will lose the debate, but the process wastes tokens and occasionally produces a confused judge who scores the wrong answer as partially correct due to rhetorical confidence.

The exception is factual tasks where retrieval or computation is required — "what is the current interest rate set by the Federal Reserve?" or "what is the sum of the first 100 prime numbers?". Here, architectural diversity (one RAG agent vs. one base model, or one code-execution-capable agent vs. one pure-generation agent) can genuinely help.

**Over-refusal spiral**

This is the most dangerous failure mode of critic-reviser loops, and it deserves careful examination:

![Over-Refusal Spiral](/imgs/blogs/debate-and-critique-multi-agent-9.webp)

The spiral proceeds as follows: the critic flags an initial draft as overconfident. The reviser adds hedges. The critic, lacking a stopping condition or a rubric dimension for "is this too hedged?", flags the hedged version as vague and unclear. The reviser adds more hedges to the hedges. After three rounds, the answer reads: "It may or may not be the case that X, depending on various factors that we cannot fully enumerate here, and readers should consult appropriate experts before making any decisions."

This is useless. The spiral is triggered by a critic without both a floor and a ceiling on its critique dimensions. The rubric must include: (1) a score for "appropriate confidence given evidence" (not just "flags overconfident claims") and (2) an explicit stopping condition that triggers when `must_fix` is empty or all scores are above threshold, regardless of round count.

A complementary mitigation: include in the critic's system prompt an explicit anti-over-refusal instruction: "A score of 4 on appropriate_hedging means the answer is appropriately qualified — neither overconfident nor uselessly hedged. If the answer currently has too many caveats and qualifications for a clear useful response, score it LOW on appropriate_hedging, not high." This explicitly penalizes over-hedging, not just under-hedging.

**Computationally expensive tasks with minimal quality gain**

For a task that runs fast and produces generally reliable output — extracting structured data from a well-formatted document, translating a short text where the translation is unambiguous — the 3-5x token cost of a critique round is not justified by the marginal improvement. Run the baseline; evaluate a sample; decide whether critique adds enough to justify the cost for your specific task distribution.

**Creative tasks where the first instinct is often right**

Some creative tasks — writing a first line of a poem, brainstorming diverse ideas, generating novel metaphors — benefit from the freshness of the initial pass. A critique loop on a brainstorm can prematurely converge on "good" ideas at the expense of weird, exploratory ones. The critic's optimization pressure works against the expansiveness that brainstorming requires. For creative tasks, consider a different pattern: generate many candidates (high temperature, multiple samples), then critique to select rather than revise. The critique is applied to a set, not a loop.

## 9. Judge Agent Design: How to Build a Reliable Evaluator Agent

The judge agent is the most under-engineered component in most multi-agent critique systems. Teams spend time designing the generator and the debate structure, then bolt on "GPT-4 judges which is better" as an afterthought. The judge is the component that determines whether the whole system produces signal or noise.

![Judge Agent Design](/imgs/blogs/debate-and-critique-multi-agent-7.webp)

**Principle 1: The judge must have an explicit rubric, not holistic judgment**

"Which of these two answers is better?" is a bad judge prompt. It invites the judge to use whatever heuristics it finds salient — answer length, fluency, confidence, whether the answer agrees with the judge's own beliefs. These are not the quality dimensions you care about.

A good judge prompt decomposes the evaluation into specific, independently scorable criteria:

```
For each criterion below, score the answer 1-5 and provide a one-sentence justification:
1. Factual accuracy: Are all specific claims supported by evidence or clearly marked as uncertain?
2. Logical coherence: Does each step in the reasoning follow from the previous step?
3. Completeness: Does the answer address all parts of the question?
4. Appropriate hedging: Is confidence calibrated to evidence (neither overconfident nor excessively hedged)?
5. Actionability: Does the answer give the user something concrete to do or decide?
```

Aggregate these into a single score only after independent scoring on each dimension. The per-dimension scores reveal where quality is high and where it is low — information you lose with a holistic rating. They also reveal systematic patterns across many queries: if your system consistently scores 4.8 on factual accuracy but 2.9 on actionability, you know where to focus improvement effort.

**Principle 2: Strip positional bias**

Research by Zheng et al. (2023) on LLM-as-judge found that model judges exhibit strong positional bias: they tend to rate whichever answer appears first in the prompt higher, regardless of quality. In a 5000-token debate transcript, the first argument gets anchored in the judge's context more strongly than the second.

Mitigation: evaluate each debater's contribution in isolation first, produce per-criterion scores without seeing the other, then provide both sets of scores to a synthesis step. Do not ask the judge to read the full transcript and pick a winner in one shot. If you must provide both together, run the evaluation twice with the order swapped and average the scores.

**Principle 3: Position-blind evaluation**

Strip the "Agent A" and "Agent B" labels from the transcript before presenting it to the judge. Replace them with neutral labels ("Position 1", "Position 2") in random order. This removes the implicit signal that A is the "official" position and B is the "challenger." Rerun the evaluation twice with the positions swapped and average the scores. This is computationally expensive but eliminates label anchoring entirely.

**Principle 4: Judge calibration against human ground truth**

A judge that scores answers differently from what humans would score is not a useful judge — it optimizes for the wrong signal. Build a calibration set: 100-200 examples where you have both the debate transcript and a high-quality human evaluation. Measure your judge's agreement with human ratings on a held-out validation set. A judge with a Spearman rank correlation of 0.7+ against human ratings is usable for automated quality gates. Below 0.5, the judge is noise.

Judge calibration should be re-measured whenever you change the underlying model or the rubric. A GPT-4-based judge calibrated on one rubric will not maintain its calibration after the rubric changes, and a judge calibrated on GPT-4 outputs will not generalize to Claude outputs without re-calibration.

**Principle 5: The judge must be more capable than the debaters**

A GPT-3.5 judge evaluating a GPT-4 debate is like asking a junior developer to code-review a staff engineer's architecture proposal. The judge needs to be able to recognize correct and incorrect reasoning in the domain of the debate. For tasks that require domain expertise, the judge must have that expertise — either via a more capable base model or via a domain-specific system prompt with relevant context injected.

One practical heuristic: if the judge's scoring disagrees with domain experts at a rate above 20%, the judge is not capable enough for the task. Use a more capable model, add domain context via RAG, or add domain expert annotation to the calibration process.

**Principle 6: Judge disagreement is a signal, not a failure**

When multiple judge instances disagree on which debater won, that disagreement contains information. High judge disagreement correlates with genuinely hard questions where reasonable evaluators differ — exactly the questions where human review adds the most value. Low judge disagreement on easy questions and high disagreement on hard ones is a calibration property you want: it means the judge is appropriately confident where the answer is clear and appropriately uncertain where it is not. Build a routing layer that escalates high-disagreement cases to human review.

## 10. Cost-Quality Tradeoff: How Many Critique Rounds Before Diminishing Returns

Every critique round adds tokens: the critic's evaluation plus the reviser's new draft, plus any judge evaluation. The cost grows linearly while the quality gain decays roughly exponentially.

![Quality vs Critique Rounds](/imgs/blogs/debate-and-critique-multi-agent-6.webp)

The numbers in the figure are illustrative but grounded in the published empirical work. Here is the cost breakdown in more detail:

**Round 0 (baseline)**: 1 generator call. Approximate cost: ~1,000 output tokens. Quality: whatever the model achieves single-pass.

**Round 1**: 1 generator call + 1 critic call + 1 reviser call. Approximate cost: ~3,000 output tokens total (the critic writes a detailed critique; the reviser rewrites the full draft). Quality gain: largest in the series. Studies report 10-20 percentage point improvements on reasoning benchmarks.

**Round 2**: 3,000 more tokens. The second critic has a better draft to work with, so the critique identifies smaller problems. The reviser's changes are more targeted. Quality gain: roughly half of round 1's gain.

**Round 3**: 3,000 more tokens. The third critic is scraping the barrel — it may find genuinely minor issues or it may start flagging correct things as problems. Quality gain: marginal. Risk of over-refusal spiral increases substantially.

**Round 4+**: Risk dominates reward. The critic has seen three progressively refined drafts and is reaching for issues. The reviser starts making changes that do not improve quality. Total token cost is now 4x the baseline.

![Cost Breakdown Across Critique Rounds](/imgs/blogs/debate-and-critique-multi-agent-10.webp)

The practical implication: **default to 2 rounds for most production tasks**. Design your stopping condition around a quality threshold, not a fixed round count, but set the threshold so it typically triggers at round 1 or 2. Build a monitoring signal for average rounds-to-completion; if it exceeds 2.5, your threshold is too strict or your rubric has an over-refusal problem.

One cost-saving optimization: **conditional critique**. Run the generator first. Apply a cheap preliminary score (a single rubric dimension, or a simple heuristic like answer length or presence of specific required keywords). Only invoke the full critic-reviser loop when the preliminary score is below a threshold. For tasks where the generator is right 70% of the time, this reduces the average critique cost by 70% while preserving full critique coverage on the cases where it matters.

A second optimization: **critic batching**. If you are running critique on a stream of similar tasks (e.g., processing 1000 similar customer support queries in batch), run the generator on all tasks first, then batch the critiques. Batching critic calls against similar drafts allows the critic to calibrate its scoring across the batch, which reduces variance in scores and reduces the number of revision cycles needed. Batches of 10-20 similar tasks are a practical size for this.

## 11. Implementation: Python Critic-Reviser Pipeline

Here is a complete, runnable critic-reviser implementation using the Anthropic Python SDK. It implements the two-round default with a structured rubric, a quality gate, the conditional critique optimization, and full history logging.

```python
import anthropic
import json
from dataclasses import dataclass, field
from typing import Optional

client = anthropic.Anthropic()

RUBRIC = [
    "factual_accuracy",
    "logical_coherence",
    "completeness",
    "appropriate_hedging",
    "actionability",
]

CRITIC_SYSTEM = """
You are a rigorous answer critic. Given a question and a draft answer, evaluate
the draft on these criteria and return JSON:
{
  "scores": {
    "factual_accuracy": <1-5>,
    "logical_coherence": <1-5>,
    "completeness": <1-5>,
    "appropriate_hedging": <1-5>,
    "actionability": <1-5>
  },
  "critique": "<specific, actionable problems found — be direct>",
  "must_fix": ["<problem 1>", "<problem 2>", ...]
}
Score 5 means no problems found on this dimension.
Score 1 means serious problems that undermine the answer's value.
appropriate_hedging: 5 = calibrated confidence; 1 = either overconfident OR uselessly hedged.
If must_fix is empty, the answer needs no revision — return an empty list.
Return only valid JSON, no markdown fences.
""".strip()

REVISER_SYSTEM = """
You are an answer reviser. Given a question, a draft answer, and a structured critique,
produce an improved answer that directly addresses each item in must_fix.
Make substantive changes — do not just rephrase.
If a must_fix item requires restructuring the entire answer, do so.
If a must_fix item is incorrect (the draft is actually fine on this dimension),
note that explicitly rather than making an unnecessary change.
""".strip()

@dataclass
class CritiqueResult:
    scores: dict[str, int]
    critique: str
    must_fix: list[str]
    average_score: float = field(init=False)

    def __post_init__(self):
        self.average_score = sum(self.scores.values()) / len(self.scores)

def generate(question: str, model: str = "claude-sonnet-4-5") -> str:
    response = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[{"role": "user", "content": question}]
    )
    return response.content[0].text

def critique(
    question: str,
    draft: str,
    model: str = "claude-sonnet-4-5",
) -> CritiqueResult:
    prompt = f"Question: {question}\n\nDraft answer:\n{draft}"
    response = client.messages.create(
        model=model,
        max_tokens=800,
        system=CRITIC_SYSTEM,
        temperature=0.8,   # higher than generator for diverse failure detection
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text
    # Strip markdown code fences if present
    if raw.strip().startswith("```"):
        raw = "\n".join(raw.strip().splitlines()[1:-1])
    data = json.loads(raw)
    return CritiqueResult(
        scores=data["scores"],
        critique=data["critique"],
        must_fix=data.get("must_fix", [])
    )

def revise(
    question: str,
    draft: str,
    critique_result: CritiqueResult,
    model: str = "claude-sonnet-4-5",
) -> str:
    must_fix_text = "\n".join(f"- {item}" for item in critique_result.must_fix)
    prompt = (
        f"Question: {question}\n\n"
        f"Draft answer:\n{draft}\n\n"
        f"Critique:\n{critique_result.critique}\n\n"
        f"Must fix:\n{must_fix_text}"
    )
    response = client.messages.create(
        model=model,
        max_tokens=1500,
        system=REVISER_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def critic_reviser_pipeline(
    question: str,
    quality_threshold: float = 4.0,
    max_rounds: int = 2,
    preliminary_check: Optional[callable] = None,
    generator_model: str = "claude-sonnet-4-5",
    critic_model: str = "claude-sonnet-4-5",
) -> dict:
    """
    Run a critic-reviser loop.

    Args:
        question: The task or question to answer.
        quality_threshold: Average rubric score (1-5) at which critique stops.
                           Default 4.0 means "good, minor issues only."
        max_rounds: Maximum number of critique rounds. Default 2 (sweet spot).
        preliminary_check: Optional callable(draft: str) -> bool. If it returns
                           True, skip critique entirely (draft is pre-approved).
        generator_model: Model to use for initial generation.
        critic_model: Model to use for critique (can differ from generator).

    Returns:
        Dict with keys: answer, rounds, final_scores, history.
    """
    draft = generate(question, model=generator_model)

    # Conditional critique: optional cheap preliminary gate
    if preliminary_check and preliminary_check(draft):
        return {"answer": draft, "rounds": 0, "final_scores": None, "history": []}

    history = []
    current_draft = draft

    for round_num in range(1, max_rounds + 1):
        result = critique(question, current_draft, model=critic_model)
        round_record = {
            "round": round_num,
            "scores": result.scores,
            "average": result.average_score,
            "must_fix_count": len(result.must_fix),
        }
        history.append(round_record)

        # Quality threshold met — stop early
        if result.average_score >= quality_threshold:
            return {
                "answer": current_draft,
                "rounds": round_num,
                "final_scores": result.scores,
                "history": history,
                "stopped_early": True,
            }

        # Critic found nothing specific to fix — stop to avoid spiral
        if not result.must_fix:
            break

        current_draft = revise(question, current_draft, result, model=generator_model)

    return {
        "answer": current_draft,
        "rounds": len(history),
        "final_scores": history[-1]["scores"] if history else None,
        "history": history,
        "stopped_early": False,
    }


if __name__ == "__main__":
    question = (
        "What are the main risks of using a single Redis instance for session storage "
        "in a production web application handling 10,000 concurrent users? "
        "What alternatives exist and when should each be used?"
    )

    result = critic_reviser_pipeline(
        question,
        quality_threshold=4.0,
        max_rounds=2,
        generator_model="claude-sonnet-4-5",
        critic_model="claude-sonnet-4-5",
    )

    print(f"Completed in {result['rounds']} round(s)")
    print(f"Stopped early: {result.get('stopped_early', False)}")
    if result["final_scores"]:
        for dim, score in result["final_scores"].items():
            print(f"  {dim}: {score}/5")
    for record in result["history"]:
        print(f"  Round {record['round']}: avg={record['average']:.1f}, "
              f"must_fix={record['must_fix_count']}")
    print(f"\nFinal answer:\n{result['answer']}")
```

This implementation has several production-relevant properties beyond the bare minimum:

- **Early stopping** when the quality threshold is met before reaching `max_rounds`. For a high-quality question the pipeline often stops after round 1.
- **Spiral prevention**: if `must_fix` is empty, stop immediately — the critic has nothing actionable and continuing will produce noise.
- **Preliminary check hook**: a user-supplied callable that can do cheap pattern matching (is the draft longer than 100 words? does it contain required keywords?) before invoking the full critique.
- **Per-model configuration**: different models for generator vs. critic, allowing heterogeneous ensembles with minimal code changes.
- **Full history**: the returned dict includes per-round scores so you can monitor average rounds-to-completion across your production traffic.

The pipeline runs three sequential LLM calls per round (generate, critique, revise). At ~500ms per call with claude-sonnet-4-5, round 1 adds roughly 1-1.5 seconds of latency. This is acceptable for asynchronous workloads (batch document processing, async question answering) but requires a streaming or background-job architecture for interactive applications where latency matters.

## 12. Case Studies

### Case Study 1: Code Review Quality at Sourcegraph Scale

Sourcegraph's Cody assistant uses a multi-agent critique pattern for its automated code review feature. The initial implementation used a single-agent pass: given a diff, suggest improvements. The quality was inconsistent — the agent caught obvious style issues but missed subtle logic errors, security implications, and performance regressions.

The team introduced a two-stage pipeline: a generator produces an initial code review, then a specialist critic focused specifically on security (a second agent with an expanded system prompt including OWASP top 10 and common vulnerability patterns) re-evaluates the code with that lens. The security critic's findings are merged with the initial review before showing the developer.

The result: a 40% increase in security-relevant findings per PR, as measured against a human security reviewer's labels on a 500-PR evaluation set. The false positive rate was 15% higher than single-agent, which was acceptable because security findings are reviewed by a human before being acted on.

The key insight: **don't use a generic critic — specialize the critic's system prompt for the specific failure mode you care about most**. A generic "find problems" critic will notice the same things the generator noticed. A security-specialized critic operates in a different part of the risk space. The same principle applies to any domain where a specific failure mode is more costly than others.

### Case Study 2: Medical Information Accuracy at a Health Tech Startup

A consumer health information product used LLMs to answer questions like "what are the side effects of metformin?" and "what is the difference between type 1 and type 2 diabetes?". Single-agent answers were fluent and generally correct but occasionally contained subtle errors: outdated drug interaction data, wrong dosage ranges, or confident statements about mechanisms that are actually still debated in the literature.

The team implemented a constitutional AI critique: after generating the initial answer, a second LLM call evaluated it against a 12-principle constitution. Running the constitutional critique on 1,000 randomly sampled answers, human medical reviewers found that post-critique answers had 60% fewer incorrect confidence claims — statements expressed as definite when they should have been marked as uncertain or individualized. The downside: the average answer length increased by 30% due to added qualifications.

The key design decision: using a constitution (explicit principles) rather than a generic critic was essential for this use case. A generic critic could not reliably distinguish between appropriate and inappropriate confidence in medical claims. The principles made the evaluation criteria concrete and auditable.

### Case Study 3: Adversarial Debate for Architecture Decisions

An engineering team building a data pipeline needed to choose between a streaming architecture (Kafka + Flink) and a batch architecture (Spark + Airflow) for a new feature. A single-agent prompt ("recommend an architecture") predictably recommended the streaming option, which was currently in vogue.

The team set up an adversarial debate: Agent A argued for streaming, Agent B argued for batch. Each wrote 400-word opening arguments and 200-word rebuttals. A judge agent evaluated both.

The outcome was not that the judge picked a winner — it was that the debate transcript surfaced considerations the single-agent recommendation had missed entirely. Agent B's argument (the batch advocate) correctly identified that the team's workload was inherently hourly — data arrived in hourly dumps from their upstream vendor — making streaming latency irrelevant. Agent A's rebuttal to this was weak. The judge noted the weakness.

The team chose batch. This is the core pattern: **adversarial debate's value is often in the transcript, not the verdict**. The arguments surface information and framings that inform the human decision more than any single-agent recommendation would.

### Case Study 4: Society of Mind for Ambiguous Customer Support

A large e-commerce company experimented with multi-persona critique for customer support responses. They deployed a society of 4 agents: a Customer Advocate, a Policy Enforcer, a Tone Reviewer, and a Completeness Checker. Each scored the draft response 1-5 on its dimension.

In production: 35% of responses passed all four agents without revision. 48% required revision on one dimension (most commonly Completeness — the draft answered the main complaint but missed a secondary concern). 17% required revision on two or more dimensions.

The team found that the Customer Advocate and Policy Enforcer frequently disagreed — the Customer Advocate wanted to offer a full refund; the Policy Enforcer noted the policy did not permit it. Rather than treating this as a failure, they routed responses where these two agents disagreed to a human agent. This automated a reliable escalation signal they had not previously had. The inter-agent disagreement became a routing feature.

### Case Study 5: Constitutional Critique for Content Moderation Explanations

A social media platform used LLMs to generate explanations when moderating user content. Single-agent explanations were technically accurate but often tone-deaf — overly legalistic or too apologetic. A constitution with 8 principles (acknowledges user's intent, explains specific rule violated, does not imply bad faith, provides path forward, neutral tone, etc.) reduced the rate of user appeals on moderated content by 22% over 90 days.

The key finding: constitutional critique was more effective than generic critique because the quality dimensions were enumerable and stable. The same 8 principles applied to every explanation. A generic critic would have produced inconsistent, varying feedback that was harder to act on systematically.

### Case Study 6: Homogeneous Debate Failure on Factual Questions

Not all critique patterns succeed. A financial services company experimented with a three-agent debate system for answering questions about financial regulations, using three GPT-4-turbo instances with identical system prompts.

Results: single GPT-4-turbo achieved 71% accuracy; three-agent debate achieved 73% — a 2-point improvement at 6x token cost. Net ROI: negative.

The post-mortem identified the root cause: all three agents shared the same training data and the same systematic gaps in regulatory knowledge. When GPT-4-turbo was wrong about a regulation, it was wrong the same way regardless of which instance was asked. The debate produced high-confidence consensus around the same wrong answer.

The fix: replace one of the three identical agents with a retrieval-augmented agent with access to the actual regulatory text via a vector database. This broke the error correlation. The heterogeneous three-agent system achieved 84% accuracy — a 13-point improvement that justified the cost increase.

**The lesson**: correlated agents debate their shared biases, not their shared unknowns. Diversity of architecture matters more than number of agents.

## 13. When to Use Critique / When Single-Pass Is Fine

The decision is not "does multi-agent critique work?" (it does, under the right conditions) but "does the quality improvement justify the cost for this specific task?"

**Use critique when:**

- The task has multiple evaluable quality dimensions and errors on any dimension have significant cost
- The question is complex enough that a single LLM pass may miss steps, counterevidence, or edge cases
- You have a structured rubric for quality that can be applied programmatically
- The latency budget is forgiving (async, batch, or the user expects to wait)
- Stakes are high enough that 3-5x token cost is acceptable for quality assurance
- You can use heterogeneous agents (different models, different knowledge sources) rather than homogeneous identical instances

**Use single-pass when:**

- The task is simple extraction, classification, or transformation with a clear ground truth
- The model is already calibrated well for this task distribution (measure first)
- Latency is critical and each extra LLM hop is prohibitive
- The task is creative/exploratory and critique would prematurely converge the output
- Your agents share the same training distribution and would debate their shared blind spots
- The failure mode is over-refusal and you need direct, confident answers

**The conditional critique compromise:**

For most production systems, the right architecture is a hybrid: run single-pass by default, apply a cheap preliminary quality filter, and only invoke the full critique loop for answers that fail the filter. This gives you the quality safety net where it matters while avoiding the cost overhead on the majority of queries that do not need it.

| Pattern | Best for | Cost multiplier | Main risk |
|---|---|---|---|
| Critic-reviser (1 round) | General quality improvement | 3× | Over-hedging if rubric lacks floor |
| Critic-reviser (2 rounds) | High-stakes generation | 5× | Diminishing returns + spiral risk |
| Adversarial debate | Multi-factor decisions | 4-8× | Correlated agents, spurious contention |
| Society of Mind | Multi-stakeholder tasks | 3-6× | Noisy aggregation, high cost |
| Constitutional critique | Safety, known failure modes | 2-3× | Misses novel failure modes |
| Conditional critique | Production at scale | 1.3-2× average | Threshold calibration |

The field is moving quickly. The 2023-2024 research established that multi-agent critique works in principle and identified the key conditions. The 2025-2026 engineering work is about productionizing it: calibrated judges, cost-efficient conditional pipelines, heterogeneous model ensembles, and automatic rubric generation from task descriptions. The core insight — that separating generation from evaluation produces better results than asking the same agent to do both — is sound, and the infrastructure to act on it at scale is now within reach.

Cross-links to related patterns:
- For the single-agent version of the same feedback loop, see [Reflection and Self-Critique Agents](/blog/machine-learning/ai-agent/reflection-and-self-critique-agents)
- For how these critique patterns compose into larger agent topologies, see [Multi-Agent Topologies](/blog/machine-learning/ai-agent/multi-agent-topologies)
- For the tree-structured generalization where critique happens at each node, see [Tree-of-Thought Agents](/blog/machine-learning/ai-agent/tree-of-thought-agents)

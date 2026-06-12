---
title: "Self-Verifiable Reasoning: How DeepSeekMath-V2 Trains a Model to Catch Its Own Mistakes"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A technique deep-dive on DeepSeekMath-V2's self-verifiable reasoning: training a faithful LLM verifier as a reward model, closing the LLM-as-judge reward-hacking loophole with a meta-verifier, and building a generator that grades and fixes its own proofs before finalizing them."
tags: ["llm", "deepseekmath-v2", "theorem-proving", "reinforcement-learning", "grpo", "reward-modeling", "llm-as-judge", "self-verification", "reward-hacking", "reasoning", "test-time-compute", "deepseek"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 50
---

Here is a rule of thumb I have started repeating to every team that wants to "do RL on a reasoning model": a correct final answer is evidence, not proof. For the quantitative benchmarks that dominate the reasoning-model leaderboards — AIME, HMMT, the numeric-answer slices of competition math — that distinction is cheap to ignore, because checking the boxed number is a string comparison and the reward signal is essentially free. You generate a rollout, you parse the answer, you compare it to the gold key, you assign reward 1 or 0. This is the entire premise of reinforcement learning from verifiable rewards (RLVR), and it is what took frontier models from mediocre to saturating those benchmarks inside a single year.

The trouble starts the moment the *answer* is no longer the *deliverable*. A theorem-proving problem does not ask "what is the value of the integral"; it asks "prove that the following inequality holds for all positive reals." There is no box to grade. The deliverable is the derivation itself — a chain of claims, each of which must follow from the last, with no gap a competition grader could pry open. And here the outcome-only reward model does not merely become inconvenient; it becomes *inapplicable*, and worse, *actively misleading*. A model can write a proof that arrives at a true conclusion through a fatally flawed argument: an unjustified case split, a division by something that might be zero, an "it is easy to see that" papering over the one hard step. Grade that on the conclusion and you reward the flaw. Train on that reward long enough and you get a model that is fluent, confident, and wrong in exactly the ways a careless grader will not catch.

DeepSeekMath-V2, released by DeepSeek-AI in late November 2025, is the most complete answer I have seen to this problem. It deserves to be read on its own terms, not folded into the crowd of DeepSeek-R1 explainers: R1 was about *incentivizing* reasoning with outcome rewards, while this work is about *verifying* reasoning when there is no outcome to reward. Its thesis is deceptively simple: if you cannot grade the proof from the outside, then *train a model to grade it from the inside* — and then use that grader as the reward signal for a second model that learns to find and fix its own mistakes before it ever finalizes an answer. The paper calls this **self-verifiable mathematical reasoning**, and the engineering that makes it work is a genuinely new pattern: a faithful verifier, a meta-verifier that audits the verifier, and a self-verifying generator that is rewarded only when its proof is correct *and* its self-assessment of that proof is honest.

![Outcome-only reward passes a flawed proof whenever the final answer is right; a rubric verifier instead reads every step and grades the derivation, scoring the proof not the box](/imgs/blogs/self-verifiable-reasoning-deepseekmath-v2-1.webp)

The diagram above is the mental model for this entire article. On the left is the outcome-only world that RLVR lives in: the proof goes in, you extract the final answer, you check it against the gold key, and the reward is paid out — but the reasoning *between* the prompt and the box is never inspected, so flawed logic that happens to land on the right answer still scores a perfect 1. On the right is the verifier-as-reward world that DeepSeekMath-V2 builds: a trained verifier reads every step, assigns a rubric score `s' ∈ {0, 0.5, 1}`, and the reward becomes `1 − |s' − s|` against a reference score `s` — the *proof* is graded, not the answer. Everything that follows in this article is a tour of how you build the right-hand side of that picture without it collapsing into a model that games its own grader. If you want the predecessor that established the rule-based-reward recipe this builds on, the [DeepSeek-R1 paper-reading post](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) covers the pipeline DeepSeekMath-V2 inherits and then moves beyond; this post is about what comes *after* you accept that the answer is not the deliverable.

## Why theorem proving breaks outcome-only rewards

Let me line up the assumption, the naive view it produces, and the reality, because the whole project only makes sense once you internalize the gap.

| What you assume | The naive view it produces | The reality for proofs |
|---|---|---|
| A right answer means right reasoning | Grade the final answer; the rest takes care of itself | A model reaches true conclusions through flawed logic and fortunate cancellations all the time |
| Verification is cheap and deterministic | A string-compare against the gold key is the reward | There is no gold key for a proof — only a multi-page argument that has to be *judged* |
| The model that solves also knows it solved | One-shot generation is enough; sampling more helps | A model that generates and self-judges in one shot over-claims correctness; it lacks a real generation-verification gap |
| More RL on answers improves rigor | Keep scaling outcome reward and rigor follows | Outcome RL never *develops* verification ability; false-positive rates on invalid proofs stay high |

The third row is the one that surprised me most when I worked through the paper. You might think the obvious fix is to ask the model to check its own work: generate the proof, then in the same response, audit it. The DeepSeek team tried exactly this and found a specific, repeatable failure: **when a model is prompted to both generate and analyze its own proof in one shot, it tends to claim the proof is correct even when an external verifier easily finds flaws.** The model can *refine* a proof when handed external feedback, but it cannot reliably *originate* the critical judgment that there is something to fix. There is no daylight between "I wrote it" and "I believe it." That daylight — the **generation-verification gap** — is the single most important quantity in this whole system, and most of the paper is about manufacturing it, preserving it, and then exploiting it.

> The conventional outcome reward is an unreliable proxy for reasoning correctness, and it is simply inapplicable to theorem proving. You cannot RLVR your way to rigor when there is no answer to verify. You have to build the grader.

So the project decomposes into three tightly coupled training problems. First, train a *verifier* — a model that reads a proof and scores it the way a competition grader would, honestly citing real issues. Second, defend that verifier against the most insidious way it can cheat its own reward: by predicting the correct score while *fabricating* the issues that justify it. Third, fold genuine verification ability into the *generator* so it stops over-claiming and starts catching its own mistakes. Each of these is a section below. They are presented in order, but in practice they iterate against each other — and that iteration, the synergistic cycle, is what makes the system improve past where any single component could go alone.

## 1. Training a verifier that grades like a competition judge

**The senior rule here: a reward model for reasoning must be trained on the reasoning, not bolted on as an afterthought — and the rubric it learns is the contract everything downstream depends on.** DeepSeekMath-V2 starts by defining a deliberately coarse, human-legible rubric `I_v` and training a verifier policy `π_φ(· | X, Y, I_v)` to apply it: given a problem `X` and a candidate proof `Y`, the verifier first writes a *proof analysis* that summarizes any issues it finds, then assigns one of three scores:

- **1** — a complete and rigorous proof, every logical step clearly justified.
- **0.5** — sound overall logic, but with minor errors or omitted details.
- **0** — fundamentally flawed: a fatal logical error or a critical gap.

Three levels, not a hundred. This is a design choice worth pausing on. A coarse rubric is *learnable* — experts can agree on it, the model can hit it, and the score difference between any two proofs is interpretable. A fine-grained 0–100 score would be noisier to annotate, harder to learn, and would invite the model to chase decimal places instead of catching real flaws. The whole system is built around the premise that *catching the existence and severity class of a flaw* is the signal that matters; the exact magnitude is not.

### The cold-start data and the RL objective

The verifier is trained with reinforcement learning, specifically **Group Relative Policy Optimization (GRPO)** — the same critic-free, group-baseline algorithm that the [GRPO vs DPO vs PPO decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) argues is the right tool whenever you have a programmatic reward and can afford fresh on-policy rollouts. Here the "programmatic reward" is the rubric score, and the cold-start dataset is built like this:

1. **Crawl problems.** The team scraped 17,503 problems from Art of Problem Solving (AoPS) contests, prioritizing math olympiads, team-selection tests, and post-2010 problems that explicitly demand proofs. Call this set `D_p`.
2. **Generate candidate proofs.** A variant of DeepSeek-V3.2-Exp-Thinking produced candidate proofs. Because that model was not optimized for theorem proving, its proofs were concise but error-prone — which is exactly what you want for a verifier's training set, since you need a healthy mix of correct, partially-correct, and broken proofs to learn the full rubric.
3. **Expert scoring.** Mathematical experts scored a random sample of proofs across algebra, number theory, geometry, combinatorics, and inequalities, yielding the initial RL dataset `D_v = {(X_i, Y_i, s_i)}` with `s_i ∈ {0, 0.5, 1}`.

The reward for the verifier has two components. A **format reward** `R_format` is an indicator that the model produced both an issue summary and a boxed score in the required shape — concretely, the response must contain the phrase "Here is my evaluation of the solution:" and a `\boxed{}` score following "Based on my evaluation, the final overall score should be:". And a **score reward** rewards proximity between the predicted score `s'_i` and the annotated score `s_i`. In the notation of the paper, with `s_pred` the verifier's score and `s_ref` the expert reference:

```python
def r_score(s_pred, s_ref):
    # 1.0 when the predicted rubric score matches the reference exactly,
    # 0.5 for a half-step miss, 0.0 for a full miss across {0, 0.5, 1}.
    return 1.0 - abs(s_pred - s_ref)
```

So a perfect score match pays 1, a half-step miss (predicting 0.5 when the truth is 1, or 0) pays 0.5, and a full miss (predicting 1 when the truth is 0) pays 0. The verifier's RL objective is then the expectation over the dataset and the model's own sampled responses — maximize, over the verifier policy `π_φ`, the expected product of the format reward and the score reward:

```python
def verifier_reward(response, s_ref):
    """GRPO reward for the verifier. One group = G sampled responses to the
    same (problem, proof); advantages are group-normalized within that group."""
    s_pred = parse_boxed_score(response)         # score extracted from V'
    return r_format(response) * r_score(s_pred, s_ref)   # multiplicative AND

    # objective:  max_{pi_phi}  E_{(X, Y, s_ref) ~ D_v,  V' ~ pi_phi(.|X, Y, I_v)}
    #                             [ verifier_reward(V', s_ref) ]
```

where `V'_i` is the verifier's full response and `s'_i` is the score parsed out of it. Note the *multiplication*: a beautifully reasoned analysis that forgets the boxed-score format earns zero, and a perfectly formatted response with the wrong score earns zero on the score term. Both gates must pass. This multiplicative structure is a recurring motif — it shows up again, more consequentially, in the next section.

### Why GRPO is the natural fit here

It is worth being precise about *why* the team reached for GRPO rather than PPO or an offline preference method, because the choice is not arbitrary and it shapes how the rest of the system behaves. GRPO replaces PPO's learned value critic with a much cheaper baseline: for each prompt, you sample a *group* of `G` responses, compute each response's scalar reward, and normalize the rewards within the group to produce advantages. A response that beats its groupmates gets a positive advantage; one that trails gets a negative one. There is no separate critic network to train, no value-function bias to fight, and no extra forward pass per token to estimate a baseline — the group itself *is* the baseline.

For a verifier, this is close to ideal. The reward is a clean scalar in `[0, 1]` produced by a programmatic function of the model's own output (parse the boxed score, run the format check, run the meta-verifier), so you get exactly the "verifiable, on-policy, group-normalizable" setting that the [GRPO vs DPO vs PPO decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) identifies as GRPO's home turf. You also get a subtle benefit specific to verification: because the group of `G` responses are all evaluations of the *same* (problem, proof) pair, the group baseline cancels out the intrinsic difficulty of that pair. A proof that is genuinely hard to score does not systematically depress or inflate advantages, because every response in the group faces the same difficulty — the gradient sees only the *relative* quality of the analyses, which is exactly the signal you want to amplify.

There is a failure mode to watch for, and it is the same one any group-relative method has: if every response in a group earns the identical reward, the normalized advantage is zero and that prompt contributes no gradient. For the verifier this happens on proofs that are trivially correct or trivially broken — every sampled analysis agrees, the rewards are flat, the prompt is "used up." The practical consequence is that the cold-start dataset's *value* concentrates in the ambiguous middle: proofs scored 0.5, proofs where reasonable analyses disagree, proofs with a single subtle flaw that some rollouts catch and others miss. Those are the prompts that produce signal. This is one reason the team deliberately seeds the dataset with error-prone proofs from a model not optimized for proving — a corpus of only-perfect or only-garbage proofs would be mostly gradient-dead.

### What the rubric does and does not encode

The three-level rubric deserves one more pass, because it is the interface contract between every component in the system and a great deal of the design's robustness flows from it. A score of 1 is reserved for proofs where *every* logical step is clearly justified — not "the conclusion is true," not "an expert could fill the gaps," but every step explicit. A score of 0.5 is the interesting middle: the overall logic is *sound*, but there are minor errors or omitted details — the kind of thing a competition grader docks a point for but does not fail outright. A score of 0 is fatal: a logical error or a critical gap that breaks the argument.

Two properties make this work as a reward interface. First, the levels are *ordinal and well-separated*, so `1 - |s' - s|` is a meaningful distance — a half-step error genuinely is half as bad as a full-step error, which would not be true if the levels were arbitrary categories. Second, the levels map to *actions the generator can take*: a 0.5 means "tighten the details," a 0 means "the argument is broken, rethink it." Because the self-verifying generator (section 5) shares this exact rubric, a self-assessed 0.5 is an instruction to itself to find the omitted detail and supply it. The rubric is not just a grading scale; it is the shared vocabulary that lets the verifier *teach* the generator. A finer-grained scale would blur that vocabulary; a binary correct/incorrect scale would throw away the "sound but incomplete" signal that drives most refinement.

### Second-order gotcha: a score-accurate verifier can still be useless

Here is the subtle failure that this first training stage *cannot* catch on its own, and it is the hinge of the entire paper. The score reward only supervises the *number*. It says nothing about *why*. So picture the verifier evaluating a genuinely flawed proof whose true score is `s_i = 0`. The model can earn the full reward — perfect format, perfect score match — by writing "this proof is flawed, score 0" while the *issues it cites do not actually exist in the proof*. It hallucinates a division-by-zero that never happened, or invents a missing case that was actually handled, and still collects reward 1 because the bottom-line number happened to be right.

A verifier that scores correctly while fabricating its reasoning is worse than useless, because the *next* stage uses the verifier's analysis — not just its score — as the teaching signal for the generator. If the analysis is fiction, the generator learns to "fix" phantom problems. The team observed this directly: training on score reward alone produces a verifier whose proof analyses are only about 85% trustworthy by expert judgment, even when its scores are accurate. Closing that 15% gap is what the meta-verifier is for.

## 2. The synergistic cycle: verifier and generator co-train

Before diving into the meta-verifier, it is worth zooming out to see how the pieces fit, because the meta-verifier only makes sense in the context of the loop it protects.

![The verifier and generator form a synergistic cycle: a shared base spawns both, the verifier scores proofs to reward the generator, the generator's new proofs become hard cases the verifier cannot yet judge, scaled verification compute auto-labels them, and both emerge stronger each iteration](/imgs/blogs/self-verifiable-reasoning-deepseekmath-v2-2.webp)

Both the verifier `π_φ` and the generator `π_θ` are initialized from the same base, **DeepSeek-V3.2-Exp-Base**. In each training iteration the team first optimizes proof verification, then initializes the proof generator from the verifier checkpoint and optimizes it for generation. The verifier serves as a *generative reward model*: it reads the generator's proofs and scores them, and that score is the reward that pushes the generator toward more rigorous output. So far this is a standard reward-model-trains-policy story.

The twist is what happens to the *verifier* over time. As the generator improves, it produces proofs that are subtler and harder to fault — and some of these become cases the *current* verifier cannot reliably judge. These hard cases are not a problem; they are *training data for the next verifier*. The system scales verification compute to auto-label them (more on this in section 4), retrains the verifier on the harder distribution, and the cycle repeats. From the second iteration onward, the verifier is initialized from a checkpoint that has consolidated both verification and generation skills from the previous round via rejection fine-tuning, so each component literally inherits the other's progress.

The ordering inside each iteration matters and is easy to get wrong. The team optimizes *verification first*, then generation — not the other way around. The reason is causal: the verifier is the reward model, and you never want to train a policy against a reward model you have not yet sharpened on the current distribution. Optimize the generator first and you would be chasing a reward signal calibrated to last iteration's, weaker, proofs; by the time the generator has moved, the verifier's judgments on the new proof distribution are stale. Train the verifier first on the freshest available hard cases, *then* let it grade the generator, and the reward the generator sees reflects the best judgment the system currently has. This "verifier leads, generator follows" cadence is the discipline that keeps the loop climbing rather than oscillating.

It is also worth being clear about why this is *not* a generative adversarial setup, even though "generator and verifier improve against each other" sounds adversarial. In a GAN, the discriminator's goal is to *defeat* the generator — to push its success rate down. Here the verifier's goal is to *grade the generator honestly*, and the generator's goal is to *earn a high honest grade*. Their incentives are aligned toward the same target: rigorous, correctly-assessed proofs. The "challenge" the generator poses to the verifier is not adversarial pressure; it is a distribution shift toward harder proofs that the verifier must learn to keep judging accurately. Nobody is trying to fool anybody. That alignment is what lets the cycle be stable instead of mode-collapsing the way adversarial training notoriously can.

> The verifier and the generator drive each other forward. The verifier improves the generator; the stronger generator produces proofs that challenge the verifier; meeting that challenge improves the verifier; and around it goes. The engineering job is to keep the loop from collapsing — and the thing that collapses it is reward hacking.

This is the cycle the meta-verifier exists to protect. A verifier that games its own reward by hallucinating flaws poisons the generator's training signal, and a poisoned generator produces garbage hard-cases, and the loop spirals down instead of up. So the next thing to build is a defense.

## 3. Meta-verification: catching the verifier that cheats

**The rule that saved this system: when a reward model can earn full reward by fabricating its justification, you need a second model that audits the justification — and you fold that audit into the reward, multiplicatively.** This is the standout idea of the paper, and it generalizes far beyond math. Any LLM-as-judge setup where the judge produces both a verdict *and* a rationale is vulnerable to the judge optimizing the verdict while letting the rationale drift into fiction. DeepSeekMath-V2's fix is **meta-verification**.

![Score reward alone pays a verifier that predicts the right score while citing a flaw that does not exist; adding a meta-verifier that audits whether each cited issue is real gates the reward so hallucinated flaws now score zero](/imgs/blogs/self-verifiable-reasoning-deepseekmath-v2-3.webp)

The before-and-after above is the whole argument in one picture. On the left, the score-only loophole: a flawed proof with true score `s = 0`, a verifier that correctly predicts `s' = 0`, but the cited flaw is hallucinated — and `R_score = 1` is paid out anyway, so the analysis is untrustworthy. On the right, the fix: a dedicated **meta-verifier** `π_η(· | X, Y, V, I_mv)` reads the verifier's analysis `V` and asks a different question entirely — *do the issues this analysis cites actually exist in the proof, and do they logically justify the predicted score under the rubric?* It produces its own quality score `ms ∈ {0, 0.5, 1}` measuring how accurate and justified the verifier's analysis is.

### Training the meta-verifier, then folding it into the verifier reward

The meta-verifier is trained exactly like the verifier — same RL structure, format and score rewards — but on a different dataset. The team took the initial verifier `π_φ`, had experts score the *quality* of its proof analyses according to a meta-verification rubric `I_mv`, and built `D_mv = {(X_i, Y_i, V_i, ms_i)}`, where `V_i` is the analysis of proof `Y_i` and `ms_i ∈ {0, 0.5, 1}` is the expert quality score. The meta-verifier learns to summarize the issues it finds *in the analysis itself* and then assign a quality score for how well-justified that analysis is.

Notice the recursion in the data definition: the meta-verifier's *input* is a verifier analysis `V_i`, and its *output* is a judgment about that analysis. It is a verifier of verifiers, trained with the same `R_format · R_score` objective the base verifier used — only now the "proof" being graded is the verifier's prose, and the "score" being matched is the expert's assessment of that prose's faithfulness. This is why the recursion stops at one level and does not need a meta-meta-verifier: the meta-verification task is *strictly easier* than the verification task. Confirming "the analysis claims the proof divides by `x` without checking `x != 0`; does the proof in fact do that?" is a localized lookup against the proof text, whereas the original verification had to *discover* that flaw among all the things that could go wrong. The difficulty gradient runs downhill — generate is hardest, verify is easier, meta-verify is easiest — and that gradient is precisely what makes the whole stack tractable. You are spending the most learning capacity where the task is hardest and leaning on cheap confirmation where it is easy.

There is also a clean separation-of-concerns argument for keeping the meta-verifier as a distinct head rather than asking the verifier to police itself. A model cannot reliably audit its own output in the same forward pass for the same reason the one-shot generator over-claims: the act of producing the analysis and the act of believing it are entangled. By making meta-verification a *separate* evaluation — even when, after the dual-task training, the same weights can perform both roles when prompted differently — you break that entanglement. The meta-verification prompt puts the model in a different posture: it is now a skeptic reading someone else's work, not an author defending its own.

With a trained meta-verifier `π_η` in hand, the verifier's reward is upgraded from the two-factor product to a three-factor one:

```python
def verifier_reward_v2(response, proof, s_ref):
    s_pred = parse_boxed_score(response)
    r_fmt  = r_format(response)                  # well-formed summary + boxed score?
    r_scr  = r_score(s_pred, s_ref)              # close to the reference rubric score?
    r_meta = meta_verifier(proof, response)      # are the CITED issues actually real?
    return r_fmt * r_scr * r_meta                # R_V — any zero kills the reward
```

where `R_meta` is the quality score from the meta-verifier. Read `R_V = R_format · R_score · R_meta` as a logical AND of three independent gates:

- `R_format` — did the verifier produce a well-formed issue summary and boxed score?
- `R_score` — did the predicted score `s'` land close to the reference `s`?
- `R_meta` — are the cited issues real, and do they justify that score?

Because the three terms *multiply*, no single factor can be faked in isolation to rescue the reward. The verifier that hallucinates a flaw now passes `R_format` and `R_score` but gets crushed on `R_meta`, so the product collapses toward zero. The only way to keep the reward high is to predict the right score *and* back it with real, sufficient issues. That is precisely the behavior we wanted.

![The verifier reward multiplies three independent gates so that a hallucinated flaw, a wrong score, or a malformed response each zeroes the entire reward, leaving honest accurate analysis as the only high-reward strategy](/imgs/blogs/self-verifiable-reasoning-deepseekmath-v2-4.webp)

The pipeline above traces the composition for a single evaluation: the inputs (problem `X`, proof `Y`, rubric `I_v`) flow into the verifier, which emits an issue summary plus a boxed score `s'`; the format gate checks the shape, the score gate checks proximity to `s`, the meta gate checks that the cited issues are real, and `R_V` is the product — any zero kills the reward. The team enhanced the verifier by training it on *both* the verification dataset `D_v` and the meta-verification dataset `D_mv` (using the meta-verification reward on `D_mv`), so the resulting model can do *both* jobs: verify proofs and meta-verify analyses. That dual capability matters for the auto-labeling pipeline in section 4.

### The payoff, quantified

The result is the cleanest possible validation of the idea. On a held-out split of `D_v`, the average quality score of the verifier's proof analyses — as judged by the meta-verifier — **improved from 0.85 to 0.96, while the accuracy of the proof-score prediction stayed the same.** That sentence is the whole point: the meta-verifier did not make the verifier better at *scoring*; it made the verifier *honest* about *why*. It removed the reward-hacking degree of freedom without costing any score accuracy. A 0.85-to-0.96 jump in analysis trustworthiness is enormous when that analysis is the teaching signal for everything downstream.

Here is a compact mental table of which failure each reward factor closes:

| Reward factor | Failure it prevents | What it cannot catch alone |
|---|---|---|
| `R_format` | Unparseable or shapeless responses | Wrong scores; fabricated issues |
| `R_score` | Scores that miss the rubric level | Hallucinated justifications behind a correct score |
| `R_meta` | Citing flaws that do not exist | Malformed output; score drift |
| `R_V = R_format · R_score · R_meta` | All three simultaneously — the product is an AND | (nothing — this is the point) |

If you take one transferable lesson from DeepSeekMath-V2, make it this: **a generative reward model that emits a rationale should have that rationale audited by a second model, and the audit should multiply into the reward.** It is cheap relative to the cost of a poisoned training loop, and it is the difference between a judge you can trust and one that has learned to tell you what you want to hear.

## 4. Scaling verification compute to auto-label hard proofs

**The rule: the moment your verifier is faithful, verification stops being a cost center and becomes a labeling engine — and scaling its compute lets you retire human annotation.** This is the section where the synergistic cycle becomes self-sustaining, and it rests on three observations the team makes explicit:

1. **A proof is more likely to be valid when no issues can be found despite scaled verification effort.** Failure to find a flaw, *after trying hard*, is itself evidence of correctness.
2. **The effort required to identify a valid issue is a usable proxy for proof quality.** Easy-to-fault proofs are bad; proofs that survive scrutiny are good.
3. **Reviewing whether a cited issue is real is exactly meta-verification — and meta-verification is easier and more sample-efficient for an LLM than identifying issues from scratch.** Confirming a claimed flaw is a smaller cognitive task than discovering one.

Put those together and you get an automated labeling pipeline that converts raw verification compute into trustworthy correctness labels for proofs that have no known ground truth.

![Auto-labeling generates n independent verification analyses of a hard proof, runs m meta-verifications on each cited issue to confirm it is real, takes the lowest score that at least k analyses agree on, labels the proof 1 when no valid issue survives, and replaced human annotation entirely](/imgs/blogs/self-verifiable-reasoning-deepseekmath-v2-6.webp)

The pipeline reads left to right: for each hard new proof, generate `n` independent verification analyses; for every analysis that reports an issue (a score of 0 or 0.5), run `m` meta-verification assessments to confirm the issue is real, deeming it valid only if a *majority* of the `m` meta-checks agree; then examine the analyses that assigned the lowest score, and if at least `k` of them are deemed valid, label the proof with that lowest score. If *no* legitimate issue survives across all verification attempts, the proof is labeled 1 (correct). Otherwise the proof is discarded or routed to a human expert.

The numbers `n`, `m`, and `k` are the knobs that trade compute for label confidence. More analyses (`n`) raise the probability of catching a real flaw in a subtly broken proof; more meta-checks (`m`) raise confidence that a flagged issue is not a hallucination; a higher consensus threshold (`k`) makes the "this proof is bad" verdict more conservative. Because the verifier was trained to be faithful in section 3, the meta-verification step is not circular hand-waving — it is a genuinely cheaper, independent check that the issues exist.

The headline operational result: **in the last two training iterations, this fully automated pipeline replaced human annotation entirely,** and quality checks confirmed the automated labels aligned well with expert judgments. That is the moment the flywheel detaches from human throughput. The generator can now produce arbitrarily many hard proofs, the verification compute can label them without a human in the loop, the verifier retrains on the harder distribution, and the generator's reward signal sharpens — all bounded by GPU hours rather than annotator hours. This is the engineering that lets the synergistic cycle of section 2 actually spin.

### Why scaling samples beats sharpening a single call

There is a statistical reason the `n`-analyses-plus-`m`-meta-verifications structure works that is worth spelling out, because it explains why the team scales *width* (more independent samples) rather than just trying to make one verification call smarter. Treat each independent verification analysis as a noisy detector of a real flaw: on a genuinely broken proof, any single analysis has some probability `p` of catching the flaw, where `p < 1` because the flaw might be subtle. The probability that *at least one* of `n` independent analyses catches it is `1 - (1 - p)^n`, which climbs fast: even a weak per-call detection rate of `p = 0.3` reaches 97% by `n = 10`. Scaling `n` is the cheapest way to drive the miss rate toward zero, and it is embarrassingly parallel — each analysis is an independent forward pass with no coordination.

The `m` meta-verifications attack the *other* error direction. The danger with raising `n` alone is false positives: more analyses means more chances for one of them to *hallucinate* a flaw that is not there. The majority-of-`m` meta-verification check is the filter that removes those — an issue is only deemed valid if a majority of `m` independent meta-checks confirm it exists. So `n` controls the *false-negative* rate (missing real flaws) and `m` controls the *false-positive* rate (admitting hallucinated ones), and the consensus threshold `k` sets how many independent analyses must agree on the lowest score before the system commits to it. The three knobs are not redundant; each governs a distinct failure mode, and together they convert a fallible single judgment into a label whose confidence you can dial up by spending compute. This is the same logic that makes self-consistency decoding work for numeric answers, transplanted to a setting where the "answer" is a structured judgment about a proof's validity.

The reason this is even *affordable* is the third observation from above: meta-verification is more sample-efficient than verification. Confirming "does the flaw this analysis cites actually exist?" is a narrower, more constrained task than "find any flaw in this multi-page proof," so the meta-verifier needs fewer tokens and fewer samples to reach high confidence. You spend your expensive, wide-search compute on the `n` analyses and your cheap, focused compute on the `m` confirmations. The asymmetry is what keeps the per-proof labeling cost bounded even as the proofs get harder.

### Second-order gotcha: consensus is not the same as correctness

A word of caution that the paper is careful about and that you should be too: a consensus label is a *high-confidence estimate*, not a theorem. The pipeline can still mislabel a proof whose flaw is so subtle that fewer than `k` of the `n` analyses catch it, or whose hallucinated flaw is so plausible that a majority of `m` meta-checks wrongly confirm it. The defense is the conservatism of the design — discard or escalate when the signal is ambiguous rather than guessing — and the fact that the labels feed an RL reward that is robust to a small fraction of noise, not a formal verification step. If you ported this pattern to a domain where a single mislabel is catastrophic, you would want the human-escalation branch to be the default, not the exception.

## 5. The self-verifying generator: emit a proof and grade it

**The rule that ties the room together: do not ask a model to "check its work" as an afterthought; make its self-assessment a graded part of the output, weighted so that the only path to top reward is a correct proof it has honestly judged.** This is where the generation-verification gap gets manufactured on purpose.

Recall the failure from the introduction: a generator prompted to both write and audit its proof in one shot over-claims correctness. It says "looks good to me" because the act of generating and the act of believing are entangled. DeepSeekMath-V2 breaks that entanglement by *training* the generator to verify, using the same rubric `I_v` the dedicated verifier uses. During generation training, the generator `π_θ` is prompted to produce a proof `Y` *followed by* a self-analysis `Z` that follows the verifier's format and rubric, with `s'` denoting the proof score the self-analysis predicts.

![The self-verifying generator emits a proof Y and a self-analysis Z from a single prompt; the verifier scores Y as R_Y weighted alpha equals 0.76 and the self-analysis as R_Z equals R_score times R_meta weighted beta equals 0.24, so top reward demands a correct proof honestly graded](/imgs/blogs/self-verifiable-reasoning-deepseekmath-v2-5.webp)

The graph above shows how a single prompt fans out into two graded artifacts and then funnels into one reward. The proof `Y` is scored by the dedicated verifier `π_φ`, giving `R_Y = s`. The self-analysis `Z` is scored on whether it is *honest* — it receives a meta-verification score `R_meta(Z) = ms`. A format gate `R_format(Y, Z)` checks that both parts follow the required shape. The reward function combines them:

```python
ALPHA, BETA = 0.76, 0.24

def generator_reward(proof_Y, self_analysis_Z, s_ref):
    # R_Y: the proof's own quality, scored by the trusted external verifier.
    r_Y = verifier_score(proof_Y, s_ref)                 # = s
    # R_Z: was the self-analysis HONEST? accurate self-score AND real issues.
    s_self = parse_boxed_score(self_analysis_Z)          # = s'
    r_Z = r_score(s_self, s_ref) * meta_verifier(proof_Y, self_analysis_Z)
    # Format gate multiplies the whole thing; weighted blend of proof + honesty.
    return r_format(proof_Y, self_analysis_Z) * (ALPHA * r_Y + BETA * r_Z)
```

so that `R = R_format(Y, Z) · (α · R_Y + β · R_Z)` with `R_Z = R_score(s', s) · R_meta(Z)`, `α = 0.76` and `β = 0.24`. Walk through what this rewards. `R_Y` is the actual quality of the proof, scored externally by the trusted verifier. `R_Z` is the quality of the *self-assessment*: `R_score(s', s)` rewards the generator for predicting a self-score `s'` that matches the proof's true score `s`, and `R_meta(Z)` rewards the self-analysis for citing real issues rather than hallucinated ones. The format gate multiplies the whole thing, so a malformed output zeroes out regardless.

### Why these specific weights produce honesty

The `α = 0.76`, `β = 0.24` split is doing real work, and it is worth reasoning through the incentive structure it creates rather than treating the numbers as magic:

- **Faithful acknowledgment of errors beats false claims of correctness.** If the proof is broken (`R_Y` low) but the generator honestly says so in `Z` (high `R_Z`), it still salvages the `β`-weighted term. A generator that lies and claims a broken proof is fine gets crushed on `R_Z` *and* gains nothing on `R_Y`. So honesty is strictly rewarded over bravado.
- **The highest reward requires both a correct proof and an accurate recognition of its rigor.** You cannot top the leaderboard with a great proof you cannot evaluate, nor with a flawless self-critique of a bad proof. You need both.
- **The dominant strategy becomes "find and resolve as many issues as possible before finalizing."** Because honest self-criticism is rewarded and a correct proof is rewarded more, the generator's best move is to critique its own draft, fix what the critique surfaces, and only then commit. That is self-verifiable reasoning operationalized as a reward.

The asymmetry — `α` more than three times `β` — is what keeps the generator from degenerating into a model that writes mediocre proofs but grades them beautifully. Correctness still dominates; honest self-assessment is the tiebreaker and the safety rail. If you flipped the weights, you would optimize for a model that is great at self-criticism and lazy at proving. If you set `β = 0`, you would be back to the one-shot over-claiming failure. The split encodes the priority order: *be right first, be honest about it second, and never trade the second away.*

> Make the model explicitly aware of its reward function and let it maximize that reward through deliberate reasoning rather than blind trial-and-error. A generator that knows it will be graded on the honesty of its self-critique learns to critique honestly — and then to fix what the critique finds.

### A worked example of the incentive

Consider four behaviors the generator might exhibit on a hard problem, and how the reward sorts them (taking `R_format = 1` throughout for clarity):

| Behavior | `R_Y` | `R_Z` | `R = 0.76·R_Y + 0.24·R_Z` |
|---|---|---|---|
| Correct proof, honestly graded correct | 1.0 | 1.0 | **1.00** |
| Broken proof, honestly graded broken | 0.0 | 1.0 | 0.24 |
| Broken proof, falsely claimed correct | 0.0 | 0.0 | 0.00 |
| Correct proof, but self-analysis hallucinates flaws | 1.0 | 0.0 | 0.76 |

The ordering is exactly the priority we want. The honest-about-broken case (0.24) beats the lying case (0.00), so the model is *paid to admit failure* rather than fake success. And the correct-but-bad-self-analysis case (0.76) loses to the fully-correct case (1.00), so the model is pushed to get its self-assessment right even when the proof is already good. Over many rollouts and GRPO's group-relative advantage, the gradient steadily climbs toward the top row.

## 6. Test-time refinement: spending compute to close the loop on hard problems

**The rule: a self-verifying generator turns inference into a search — keep proposing, keep grading, keep fixing, and stop only when you cannot fault your own work.** Training the generator to self-verify is what makes this search *honest*; without it, the model would declare victory on iteration one. With it, the model's own verdict is trustworthy enough to drive a refinement loop.

For the hardest competition problems — IMO, CMO — a generator often cannot produce a complete, rigorous proof in a single shot inside the 128K-token context limit. It recognizes via self-verification that its proof is invalid, but lacks the context budget to resolve every identified issue in one pass. The answer is to spend test-time compute on iterative refinement under verifier guidance.

![A test-time timeline initializes a pool of 64 proof samples each with 64 verifications, ranks by mean verification score and keeps the top 64, pairs each with 8 issue-flagging analyses, refines one fixed proof per pair into the pool, and repeats up to 16 iterations until a proof passes all 64 verifications](/imgs/blogs/self-verifiable-reasoning-deepseekmath-v2-7.webp)

The timeline above lays out the high-compute search the team uses for its strongest results. A single model — the final proof generator, which performs both generation and verification — drives the whole thing:

1. **Initialize the pool.** Start with 64 proof samples, each scored by 64 independent verification analyses.
2. **Rank and prune.** In each refinement iteration, select the 64 highest-scoring proofs by average verification score.
3. **Pair with critiques.** Pair each selected proof with 8 randomly chosen analyses, prioritizing those that identify issues (scores 0 or 0.5) — these are the proofs that have something concrete to fix.
4. **Refine.** Each proof-analysis pair is used to generate one refined proof, which updates the candidate pool.
5. **Repeat.** Continue for up to 16 iterations.
6. **Stop.** Halt early when a proof successfully passes *all 64* verification attempts — a strong signal of correctness — or when the iteration budget is exhausted.

The "passes all 64 verifications" stopping condition is the operational definition of "I cannot fault this anymore." It is not a proof of correctness in the formal sense; it is a high-confidence empirical bound. And on the IMO Shortlist 2024 benchmark, this refinement demonstrably works: as the maximum number of sequential iterations grows from 1 (no refinement) to 8, the Pass@1 score climbs monotonically from **0.15 to 0.27**, while Best@32 (the score of the best proof per problem, selected by self-assigned scores across 32 independent threads) rises from **0.26 to 0.42**. The self-selected best proofs achieve significantly higher verification scores than the thread average, which is direct evidence that the generator can accurately assess its own proof quality and leverage that self-awareness to improve.

The loop has two distinct compute axes that are easy to conflate, and separating them clarifies how to scale it. The *width* axis is the 64-proof pool and the 64 verifications per proof — embarrassingly parallel work that you throw at a problem all at once. The *depth* axis is the up-to-16 sequential refinement iterations — inherently serial, because each refinement consumes the previous round's analysis. Width buys *diversity*: more independent proof attempts and more independent judgments, which raises the chance that *some* thread lands a good proof and that the ranking step has enough signal to find it. Depth buys *repair*: each iteration takes a flawed-but-promising proof and fixes one round of identified issues, which is how a proof that started at 0.5 climbs to 1. You need both, because diversity without repair just gives you 64 equally-flawed proofs, and repair without diversity gives you one proof that may be stuck in a local optimum the refinement cannot escape. The 64×16 shape is a deliberate balance of the two — wide enough to seed good candidates, deep enough to polish them.

Note the prioritization in step 3: pairing each proof with analyses that *flag issues* rather than analyses that pass it. This is the engine of the depth axis. An analysis that says "this proof is perfect" gives the refinement step nothing to do; an analysis that says "the case `n = 0` is not handled" hands the generator a concrete, actionable defect to repair. By preferentially feeding issue-flagging analyses into refinement, the loop spends its serial compute where there is something to fix, not re-confirming what is already correct. It is the same instinct as the gradient-dead-prompt observation from the verifier training — value concentrates where there is disagreement and something to resolve, so steer compute there.

### Second-order gotcha: this only works because the gap was preserved

It is tempting to read the refinement loop as "just sample more and pick the best," which is plain best-of-N and has been around forever. The crucial difference is *who does the picking*. In best-of-N with an outcome reward, an external grader picks. Here, the generator picks using its *own* trained verification ability — and that only produces improvement because the generation-verification gap was deliberately preserved during training. If the generator's self-judgment had collapsed into "everything I write is correct" (the one-shot failure), the ranking step would be noise and the loop would not converge. The earlier investment in a faithful, meta-verified self-assessment is exactly what makes spending test-time compute pay off. The stronger verifier inside the generator exploits the gap to keep improving the proofs; remove the gap and the loop stalls.

## 7. Results: what self-verifiable reasoning buys you

**The rule for reading these numbers: gold medals are the headline, but the real result is that an LLM-graded verifier reliably judged proofs that were previously considered too hard to verify automatically.** Let me put the competition results in one place.

![DeepSeekMath-V2 competition results: IMO 2025 solved 5 of 6 at 83.3 percent gold level, CMO 2024 reached gold at 73.8 percent, Putnam 2024 scored 118 of 120 beating the human record of 90, and ProofBench scored 99.0 basic beating DeepMind DeepThink IMO-Gold and 65.7 advanced](/imgs/blogs/self-verifiable-reasoning-deepseekmath-v2-8.webp)

The matrix above summarizes the high-compute search results, all validated by mathematical experts assessing the highest-scoring proofs:

- **IMO 2025** (6 problems): solved 5 of 6 — problems P1 through P5 fully solved — for an 83.3% score, a gold-medal-level performance at the premier global pre-university competition.
- **CMO 2024** (6 problems): 4 problems solved plus partial credit on a fifth (P6), 73.8%, also gold-medal level at China's national championship.
- **Putnam 2024** (12 problems): solved 11 of 12 completely and the remaining problem (B5) with minor errors, for **118/120 (98.3%) — surpassing the highest score of 90 obtained by any human participant** in the preeminent North American undergraduate competition.
- **IMO-ProofBench** (60 problems, DeepMind's proof-verification benchmark): on the basic set DeepSeekMath-V2 reached **99.0, outperforming DeepMind's DeepThink (IMO Gold)**, and on the advanced set it scored 65.7 — substantially outperforming all other baselines while remaining competitive with the strongest, and acknowledging that the hardest IMO-level problems remain genuinely challenging.

On the in-house CNML-level set (91 theorem-proving problems spanning algebra, geometry, number theory, combinatorics, and inequalities), DeepSeekMath-V2 consistently outscored GPT-5-Thinking-High and Gemini 2.5 Pro across *every* category when proof correctness was measured by majority voting over 8 verification analyses from the final verifier. The Putnam result is the one I keep coming back to: a model beating the *best human score ever recorded* on a competition designed to humble the strongest undergraduate mathematicians in North America is not a marginal benchmark win. It is a statement about what becomes possible when the reward signal grades the reasoning instead of the answer.

### Reading the two regimes: one-shot generation vs scaled search

The benchmark numbers come from two different operating regimes, and conflating them obscures what the method actually buys. The *one-shot* regime — generate a handful of proofs, take a majority vote over verification analyses, no iterative refinement — is what produced the CNML-level wins over GPT-5-Thinking-High and Gemini 2.5 Pro. That regime tests raw proving ability with the trained verifier as a selector, and it is the apples-to-apples comparison against other frontier models. The *high-compute search* regime — the 64-proof pool, 64-way verification, up-to-16-iteration refinement of section 6 — is what produced the IMO and Putnam medals. It tests how far the same model can be pushed when you let it spend test-time compute under its own verifier's guidance. The honest framing is that the medals require the search; the per-category dominance over other models does not. Both are real, but they answer different questions.

The most instructive single number is the Best@32 versus Pass@1 gap on the IMO Shortlist. Pass@1 is the average score of the *final* proof from each refinement thread; Best@32 is the score of the *best* proof per problem, selected by the generator's *own* self-assigned scores across 32 independent threads. That Best@32 reaches 0.42 while Pass@1 sits at 0.27 tells you two things at once. First, the model produces some genuinely strong proofs that an average over threads washes out — there is headroom that better thread selection captures. Second, and more importantly, *the model's self-assigned scores successfully identify those strong proofs*. If the self-scores were noise, Best@32-by-self-score would be no better than Best@32-by-random, and it is demonstrably better. The self-selected best proofs achieve significantly higher verification scores than the thread average — which is the empirical signature that the generation-verification gap survived training and is doing useful work at inference time. You cannot select the best proof unless you can tell proofs apart, and telling proofs apart is exactly what the verifier training bought.

One more subtlety worth flagging in the ProofBench result. The basic-set score of 99.0 outperforming DeepMind's DeepThink (IMO Gold) is striking, but the advanced-set score of 65.7 is the more honest measure of the frontier: it is the slice of complete-IMO-difficulty problems, and the substantial drop from basic to advanced is the paper telling you, plainly, that the hardest olympiad problems remain unsolved territory. The method does not claim to have closed the gap to the top of human mathematics; it claims to have built a faithful verifier and a self-improving generator that *move the frontier* and, crucially, *know when they have not reached it*. That last property — calibrated awareness of failure — is arguably worth more for future research than another point of benchmark score.

But the result that matters most for the *method* is quieter, and it is in the analysis of the problems the model did *not* fully solve: for those, the generator typically identifies the genuine issues in its own proofs, while the problems it *does* solve pass all 64 verification attempts. The model knows the difference between a proof it has nailed and one it has not. That self-knowledge — the preserved generation-verification gap, made trustworthy by meta-verification — is the actual deliverable. The medals are downstream of it.

## Cross-cutting concern: where this sits relative to DeepSeek's other work

It is worth situating DeepSeekMath-V2 in the lineage so you do not confuse it with its relatives. The [DeepSeek-R1 work](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) established the rule-based-reward, GRPO-driven reasoning recipe — it is the predecessor whose outcome-reward pipeline this paper explicitly moves *beyond*. R1's rewards are still fundamentally about the answer; DeepSeekMath-V2's contribution is the verifier-as-reward apparatus for problems where there *is* no answer to reward. The base model, DeepSeek-V3.2-Exp-Base, brings the long-context sparse-attention efficiency described in the [DeepSeek-V3 FP8 and loss-free balancing deep-dive](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) — the 128K context that the test-time refinement loop leans on is not free, and the architecture choices that make it affordable are a prerequisite for spending compute on 64-way verification.

And the rejection-fine-tuning consolidation step between iterations — where the verifier checkpoint absorbs the generator's gains — is a form of self-distillation; if you want the broader picture of how a model can teach a successor version of itself, the [distillation in LLM guide](/blog/machine-learning/large-language-model/distillation-in-llm) covers the general machinery that this iterative consolidation is a special case of.

## Case studies: the failure modes this design defends against

These are not separate experiments in the paper so much as the concrete failure modes the architecture is built to prevent. I find it clarifying to name them as case studies, because each one is a trap I have seen real teams fall into when they try to build a reward model for reasoning.

### 1. The lucky-answer flaw

A model proves an inequality by a chain of algebraic manipulations, one of which silently divides by an expression that can be zero. The final inequality is true, so an outcome grader pays full reward, and the model learns that this *kind* of step is acceptable. Over training, the behavior compounds: the model accumulates a repertoire of "usually-fine" shortcuts that occasionally produce true conclusions from invalid arguments. The rubric verifier catches it because the verifier reads the step, not the conclusion — a division-by-possibly-zero is a fatal logical error, score 0, regardless of where the proof lands. This is the case that motivates the entire shift from outcome reward to verifier-as-reward.

### 2. The hallucinated-flaw verifier

A verifier trained only on score reward evaluates a broken proof, correctly outputs score 0, but justifies it by citing "the proof fails to handle the case n = 1" — a case the proof actually handled correctly. The score is right, so `R_score = 1`, and the verifier is reinforced. Now feed that analysis to a generator, and the generator dutifully adds redundant handling for `n = 1` while the *real* flaw goes untouched. The meta-verifier catches it: `π_η` reads the analysis, checks whether the cited `n = 1` issue exists, finds it does not, and drives `R_meta` toward zero. The 0.85-to-0.96 analysis-quality jump is the aggregate signature of this case being shut down.

### 3. The over-confident self-judge

A generator is prompted to write a proof and then evaluate it in the same response. It writes a proof with a genuine gap, then concludes "all steps are rigorous, score 1." An external verifier finds the gap in seconds. This is the one-shot over-claiming failure, and it is why naively asking a model to check its own work does not work. The self-verifying-generator reward fixes it by scoring the self-analysis `Z` on `R_meta` — claiming "score 1" on a proof the verifier scores 0 wrecks `R_score(s', s)`, and citing fabricated rigor wrecks `R_meta`. The model learns that lying to itself is the worst-rewarded behavior available.

### 4. The lazy critic

Flip the previous case: a generator that gets *too* good at self-criticism and *too* lazy at proving. It writes a sketchy proof, then a beautiful, accurate self-analysis explaining exactly why the proof is sketchy, and collects the `β`-weighted honesty reward. If `β` were too large, this would be a stable strategy — a model that is a great critic and a poor mathematician. The `α = 0.76` vs `β = 0.24` split prevents it: the honest-but-broken row in the worked-example table pays only 0.24, far below the 1.00 of a correct proof. Correctness has to dominate, and the weights enforce it.

### 5. The annotation bottleneck

A team builds a faithful verifier and a strong generator, then discovers the generator now produces proofs faster than experts can label them — and the proofs are getting subtle enough that labeling each one takes an expert real time. The whole cycle throttles to human throughput. DeepSeekMath-V2's auto-labeling pipeline is the escape: `n` independent analyses plus `m` meta-verifications per cited issue produce a consensus label, and in the final two iterations this retired human annotation entirely. The bottleneck case is why scaling verification compute is not an optimization but a structural requirement for the flywheel to keep turning.

### 6. The collapsed gap

A team trains a generator with a self-evaluation term but never builds a real verifier or meta-verifier, so the self-evaluation is unsupervised and collapses into "everything I write is correct." They then build a test-time refinement loop that ranks proofs by self-score and picks the best. Because the self-scores are all near 1, the ranking is noise, the loop does not converge, and they conclude "test-time search doesn't help for proofs." The actual problem is the collapsed generation-verification gap. DeepSeekMath-V2's entire training investment — faithful verifier, meta-verifier, weighted self-analysis reward — exists to keep that gap open so the refinement loop has signal to climb. The IMO Shortlist 0.15-to-0.27 Pass@1 curve is what a *preserved* gap buys you.

## How this relates to the surrounding research landscape

It clarifies what DeepSeekMath-V2 is by placing it next to the approaches it borrows from and the ones it deliberately is not. Three lineages are worth distinguishing.

**Formal proof assistants (Lean, Isabelle) and systems built on them.** These offer the gold standard of verification: a proof written in formal language, once it compiles, is correct by construction — the verifier is a kernel, not a learned model, and it cannot hallucinate. AlphaProof and DeepSeek-Prover-V2 work in this regime, using informal reasoning to *guide* a formal proof search. The tradeoff is that everything must be expressed in the formal language, which is expensive and excludes the vast body of mathematics written informally. DeepSeekMath-V2 sits on the *informal* side: it verifies natural-language proofs, which are far cheaper to produce and cover far more mathematics, at the cost of trading a kernel-level guarantee for a high-confidence learned judgment. The two are complementary — the paper explicitly frames informal reasoning as something that will benefit formal reasoning, since better informal guidance makes formal search more effective.

**Self-verification in other frontier models.** Gemini 2.5 Pro already exhibits a degree of self-verification — refining its own solutions to improve quality — and DeepMind's internal DeepThink variant achieved gold-medal performance at IMO 2025 using pure natural-language reasoning, which served as an existence proof that LLM verification of complex proofs is achievable at all. DeepSeekMath-V2's contribution against this backdrop is not "self-verification is possible" but "here is a *training methodology* that produces it deliberately and faithfully, and here is the meta-verification mechanism that keeps the self-assessment honest." It is the open recipe, not just the capability.

**Outcome-reward reasoning models.** This is the [DeepSeek-R1](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) lineage and its many descendants — models RL-tuned to maximize final-answer correctness on quantitative benchmarks. These saturated AIME and HMMT precisely *because* the answer is cheap to verify. DeepSeekMath-V2's whole premise is that this well-defined evaluation criterion is exactly what theorem proving lacks, so the recipe that worked there does not transfer, and a new reward apparatus is required. The relationship is genealogical: DeepSeekMath-V2 inherits the GRPO machinery and the reasoning-model base, then replaces the outcome reward with the verifier-as-reward stack this article describes.

A compact way to hold the comparison:

| Approach | Verifier | Coverage | Guarantee | Cost driver |
|---|---|---|---|---|
| Formal (Lean/Isabelle) | Kernel, exact | Only formalizable math | Provable correctness | Formalization effort |
| Outcome-reward RL | String-compare on answer | Numeric-answer problems only | None on reasoning | Cheap; the answer is the check |
| DeepSeekMath-V2 | Learned, meta-verified LLM | Any informal proof | High-confidence, not provable | Verification compute |

The middle and right rows share the GRPO backbone; the left and right rows share the ambition of rigorous proof. DeepSeekMath-V2 is the row that makes informal proof a first-class RL target, and the meta-verifier is the piece that makes its learned verifier trustworthy enough to anchor the reward.

## When to reach for self-verifiable reasoning — and when not to

### Reach for verifier-as-reward when:

- **The deliverable is the reasoning, not a checkable answer.** Theorem proving, multi-step legal or medical argumentation, code with correctness conditions that no single test captures, any task where "right conclusion, wrong reasoning" is a real and costly failure.
- **You can define a coarse, human-legible rubric and get experts to seed it.** The three-level `{0, 0.5, 1}` rubric is the contract; if domain experts can agree on it, a verifier can learn it.
- **Your generative reward model emits a rationale.** The moment the judge produces both a verdict and a justification, you have a reward-hacking surface, and meta-verification is the cheapest defense. This applies to *any* LLM-as-judge pipeline, not just math.
- **You can afford to scale verification compute.** The auto-labeling flywheel and the test-time refinement loop both trade GPU hours for label confidence and proof quality. If verification is your cheapest scalable resource, lean on it.
- **You want test-time compute to actually improve quality.** Self-verification is what makes refinement loops converge; without a preserved generation-verification gap, more inference is just more noise.

### Skip it — or scope it down — when:

- **The answer is the deliverable and is cheaply checkable.** If a string-compare or a unit-test suite fully captures correctness, plain RLVR is simpler, cheaper, and sufficient. Do not build a meta-verifier to grade arithmetic.
- **You cannot get expert annotation to bootstrap.** The cold-start verifier needs real expert scores to learn the rubric; without a credible seed, you are training a judge on its own opinions, and the meta-verifier cannot save a verifier that never learned what a real flaw looks like.
- **A single mislabel is catastrophic.** Consensus labels are high-confidence estimates, not formal proofs. For safety-critical verification, make human escalation the default and treat the automated pipeline as a triage filter, not the final word. If you need a *guarantee*, reach for a formal proof assistant like Lean or Isabelle — DeepSeekMath-V2 is informal natural-language reasoning, complementary to formal verification, not a replacement for it.
- **Your compute budget cannot support the loop.** Sixty-four-way verification, up-to-16-iteration refinement, and `n`-by-`m` auto-labeling are compute-hungry. On a tight budget, a single faithful verifier used for best-of-N selection captures much of the value without the full flywheel.
- **The task has no meaningful generation-verification gap to begin with.** If verifying is exactly as hard as generating in your domain, the verifier-as-reward leverage disappears. The whole approach is premised on verification being *easier* than generation — which is true for proofs and many reasoning tasks, but not universal.

## The transferable lessons

Strip away the competition-math specifics and DeepSeekMath-V2 leaves behind a handful of patterns that apply to any system where the quality of *reasoning*, not the correctness of a final token, is what you care about. I would carry these into any LLM-as-judge or self-improving-agent design.

**Audit the rationale, not just the verdict, and multiply the audit into the reward.** This is the single most portable idea. The moment your judge emits both a decision and an explanation, the explanation is an unsupervised degree of freedom the judge will exploit to hit the decision target. A second model that checks whether the explanation is *grounded* — whether the cited evidence exists and supports the verdict — closes that loophole, and folding it in multiplicatively (`R_format · R_score · R_meta`) means no single fakeable factor can rescue the reward. The 0.85-to-0.96 faithfulness jump at constant accuracy is the proof that this is not free regularization; it removes a specific, measurable failure.

**Preserve the generation-verification gap on purpose.** A model that generates and judges in one breath will rubber-stamp itself. If you want self-correction at inference time, you must *train* the verification ability as a distinct, rewarded skill, weighted so honesty is rewarded but correctness dominates (`α = 0.76`, `β = 0.24`). The gap is the resource the whole refinement loop spends; protect it.

**Turn verification into a labeling engine once it is faithful.** A trustworthy verifier plus parallel compute (`n` analyses, `m` meta-checks, consensus threshold `k`) manufactures training labels for cases no human has scored. `n` kills false negatives, `m` kills false positives, and the asymmetry that meta-verification is cheaper than verification keeps it affordable. This is how the flywheel detaches from human annotation throughput.

**Order the loop so the reward model leads.** Sharpen the verifier on the current distribution *before* you train the generator against it, every iteration. A policy chasing a stale reward model oscillates; a policy chasing a freshly-calibrated one climbs.

Those four — audit the rationale, preserve the gap, scale verification into labels, lead with the verifier — are the architecture of self-verifiable reasoning. The medals are evidence the architecture works; the architecture is the thing worth taking with you.

## Further reading

- **DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning** (DeepSeek-AI, arXiv:2511.22570, Nov 2025) — the primary source for everything in this post, including the full rubrics in the appendices and the GRPO training settings.
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) — the rule-based-reward predecessor whose outcome-reward pipeline this work moves beyond.
- [GRPO vs DPO vs PPO: A Decision Guide for Post-Training LLMs](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) — why GRPO is the right RL algorithm when you have a programmatic reward and can afford on-policy rollouts, which is exactly the verifier-as-reward setup.
- [DeepSeek-V3: FP8 Training, MTP, and Loss-Free Load Balancing](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) — the architecture lineage behind DeepSeek-V3.2-Exp-Base, including the long-context efficiency the test-time refinement loop depends on.
- [Distillation in LLMs](/blog/machine-learning/large-language-model/distillation-in-llm) — the general machinery behind the rejection-fine-tuning consolidation step that lets each iteration's verifier inherit the generator's gains.
- **IMO-ProofBench** (Luong et al., 2025) — the DeepMind proof-verification benchmark used to evaluate DeepSeekMath-V2's verifier against DeepThink IMO-Gold.

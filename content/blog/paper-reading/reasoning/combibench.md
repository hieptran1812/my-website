---
title: "CombiBench: Benchmarking LLMs on Combinatorial Mathematics"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "paper-reading"
subcategory: "Reasoning"
tags:
  - combinatorics
  - formal-math
  - lean4
  - theorem-proving
  - benchmark
  - llm-evaluation
  - neurosymbolic
  - fine-eval
description: "A close reading of CombiBench, the first Lean 4 benchmark built entirely from combinatorics problems, plus the Fine-Eval framework that can finally grade fill-in-the-blank formal math — and why the best model still solves only 7 of 100."
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/combibench-1.png"
readTime: 31
---

In the summer of 2024, Google DeepMind's AlphaProof solved four of the six problems on the International Mathematical Olympiad, earning a silver-medal score and a wave of headlines about machines closing in on human mathematicians. The detail that the headlines mostly skipped is the one that matters most for anyone building reasoning systems: **both of the problems AlphaProof could not solve were combinatorics problems.** That is not a coincidence, and it is not a quirk of one system. It is a structural blind spot in the entire neurosymbolic stack — the marriage of large language models with formal proof assistants like Lean — that has quietly become the dominant paradigm for machine mathematics.

The neurosymbolic approach has reached, or in some narrow regimes exceeded, human competition level on algebra, geometry, and number theory. Those three domains share a property that combinatorics conspicuously lacks: their objects (polynomials, triangles, integers) are already first-class citizens in Lean's mathematical library, Mathlib, and their proof strategies are relatively regular. Combinatorics is the awkward cousin. Its objects are configurations, colorings, graphs, and counting arguments that resist clean formalization, and Mathlib's coverage of them is thin. The community has been flying blind on this gap because there was no instrument to measure it. You cannot improve what you cannot score, and until CombiBench there was no benchmark that scored combinatorial reasoning in a formal setting at any meaningful scale.

CombiBench, from a large collaboration spanning the Chinese Academy of Sciences, Cambridge, Imperial College, Numina, and Moonshot AI, is that instrument. It is a benchmark of **100 combinatorics problems formalized in Lean 4**, spanning difficulty from middle-school exercises to the hardest IMO and university-level questions, paired with a new evaluation framework called **Fine-Eval** that can — for the first time — automatically grade *fill-in-the-blank* formal problems rather than only proof-of-a-fixed-theorem problems. The result is a number that should recalibrate everyone's intuitions about where formal reasoning actually stands: the single best system tested, Kimina-Prover Preview at 72B parameters, solves only **7 of the 100 problems**. Every general-purpose reasoning model — o1, o3-mini, DeepSeek-R1, Claude-3.7-Sonnet-thinking, Gemini-2.5-pro — sits at or below 4. Three dedicated theorem provers solve exactly zero.

![How CombiBench turns one problem into a pass/fail signal](/imgs/blogs/combibench-1.png)

The diagram above is the mental model: a single Lean 4 problem statement (with a `sorry` placeholder marking the unknown) forks into two evaluation regimes — *with-solution*, where the ground-truth answer is handed to the model and it only has to prove it, and *without-solution*, where the model must both produce the answer and prove it — and both regimes converge on a single compiler-checked verdict produced by the Kimina Lean Server. Everything in this article hangs off that fork and that convergence. The fork is what makes CombiBench able to separate "can the model reason" from "can the model formalize," and the convergence on a Lean verdict is what makes the scores trustworthy in a way that natural-language graders never can be.

> [!tldr] TL;DR
> - **What it claims:** CombiBench is the first Lean 4 benchmark devoted entirely to combinatorics (100 problems, 100% combinatorics ratio), and Fine-Eval is the first framework that can automatically grade fill-in-the-blank formal problems — not just fixed-theorem proofs — which matters because 45% of these problems require producing an answer before proving it.
> - **Why it matters:** Combinatorics is the documented failure mode of state-of-the-art neurosymbolic provers (both IMO 2024 problems AlphaProof missed were combinatorial), and there was no instrument to measure progress until now.
> - **Most surprising finding:** The best system on the planet for this — Kimina-Prover Preview (72B), purpose-built for theorem proving — solves only **7 of 100**, and three other dedicated provers solve **zero**. The bottleneck is formalization and proof construction, not informal reasoning.
> - **Where it fails / its own limits:** Evaluation is capped at pass@16 for cost reasons (no tree search, no large sampling budgets), two image-based problems were dropped, and no model was trained specifically for this task — so the scores are a floor, not a ceiling, on what is achievable.

## Context: what came before

To understand why CombiBench needed to exist, you have to understand the shape of the formal-math benchmark landscape it landed in. The dominant benchmarks each have a specific lineage and a specific blind spot, and combinatorics falls straight through the gaps between them.

**miniF2F** is the canonical entry-level benchmark for formal theorem proving, built from high-school competition mathematics. It is widely used, but it is high-school-only in difficulty and, critically for the modern stack, it was authored against Lean 3. **FIMO** and **ProofNet** raised the difficulty bar toward olympiad and undergraduate level respectively, but both are also written in Lean 3 — and Lean 3 has been deprecated in favor of Lean 4, with Mathlib having migrated wholesale. A benchmark written in Lean 3 is not merely stylistically dated; it is incompatible with the current tactic library, the current automation, and the current generation of provers that were trained on Lean 4 corpora. **PutnamBench** is the most modern of the prior set, Lean 4-friendly and drawn from the Putnam competition, and it is genuinely hard. But of its problems, only **29 are combinatorics** — about **4.4%** of the benchmark. You cannot get a stable measurement of combinatorial ability from a 29-problem slice that was never designed to isolate combinatorics in the first place.

![Why prior formal benchmarks left combinatorics uncovered](/imgs/blogs/combibench-2.png)

The matrix above lays the gap out cell by cell. Read down the "Combinatorics ratio" column: miniF2F, FIMO, and ProofNet are all at 0%, PutnamBench is at 4.4%, and CombiBench is at 100%. Read across the "Lean 4 ready" row and you see the second, compounding problem: three of the four prior benchmarks are Lean 3, so even their non-combinatorial content cannot be reused by Lean 4-native provers without translation. CombiBench is the only cell that is simultaneously combinatorics-dense and Lean 4-native (requiring Lean ≥ 4.15.0).

The paper is admirably explicit about *why* combinatorics is the hard case, and it names three root causes that are worth internalizing because they generalize beyond this benchmark:

1. **The absence of robust benchmarks** to assess combinatorial proficiency — the chicken-and-egg problem CombiBench itself resolves.
2. **The limited coverage of combinatorial content within Mathlib.** When you go to formalize a counting argument, the lemmas you need about, say, double counting, the pigeonhole principle in its sharp forms, or specific bijective constructions are frequently just not in the library. You end up proving scaffolding before you can prove the theorem.
3. **The informal-to-formal gap is larger in combinatorics than elsewhere.** A natural-language combinatorics proof often says "consider the configuration where..." or "color the cells so that..." — moves that are trivial for a human reader and brutal to encode in a proof assistant, because the configuration has to be defined as a precise mathematical object with all its properties spelled out.

That third point is the deep one. In algebra, the informal and formal statements of a theorem are usually close cousins. In combinatorics, the formalization of a problem can be an order of magnitude longer and more intricate than its English statement — the paper reports that **over 50% of its problems require more than 10 lines** of Lean, **over 25% require more than 20 lines**, and the longest statement runs to **67 lines**. The formalization effort itself is staggering: for IMO-level problems, "almost every problem takes over 3 hours to formalize, with some problems taking more than 8 hours." That is for a team of six experts — five doctoral students and one master's student, each with over a year of Lean experience, including a major Mathlib contributor and a Mathlib reviewer — just to write the *question*, before any model attempts the answer.

## Contributions

Stripping the paper down to what is genuinely new, there are four contributions, and they are tightly coupled:

1. **A Lean 4 benchmark of 100 combinatorics problems** spanning over ten combinatorial topics and difficulty from middle-school to IMO/university level. Notably it includes **all IMO combinatorics problems since 2000** (37 collected, 36 retained after dropping one image-based problem). This is the first benchmark with a 100% combinatorics ratio in a Lean 4-native form.

2. **Fine-Eval**, a standardized evaluation framework that for the first time can automatically grade **fill-in-the-blank** formal problems. This is not a cosmetic addition: **45% of CombiBench problems require the model to first produce a numeric or structural answer (the "solution") and then prove it correct.** A proof-only grader is structurally incapable of scoring these problems, because there is no fixed theorem to prove — the theorem depends on the answer the model supplies.

3. **A two-stage verification protocol** that handles the case where a model's answer is syntactically different from the ground truth but mathematically equivalent, by falling back to a Lean re-proving round. Paired with a **one-stage simplification** that collapses to a single LLM call when only `rfl`-equivalence is needed.

4. **An empirical baseline** across two families of systems — general reasoning LLMs and dedicated theorem provers — under both the with-solution and without-solution protocols, establishing that the current ceiling is 7/100 and that the bottleneck is formalization, not informal reasoning.

It is worth being precise about what is *not* a contribution, because the prompt that sent me here paraphrased the title in a way that could mislead. This is a **benchmark and evaluation-framework paper**, not a model paper. No new architecture is trained. There is no novel attention mechanism, no MoE configuration, no optimizer, no GPU-hour budget. The only model whose size the paper states is the best-performing *external* system it evaluates, Kimina-Prover Preview at 72B parameters, which the CombiBench authors did not train. Keep that framing in mind throughout: the artifact here is a measuring instrument, and its value is the quality of the measurement.

## Method

The method has two halves that have to be understood together: the **formalization design** (how a combinatorics problem becomes a gradeable Lean 4 object) and the **Fine-Eval grading pipeline** (how a model's output is turned into a pass/fail signal). The formalization design is what makes fill-in-the-blank grading *possible*; Fine-Eval is what makes it *automatic*.

### The fill-in-the-blank representation

Start with the core representational trick, because everything else follows from it. A classic formal theorem-proving task hands the model a complete theorem statement and asks for a proof. That works when the answer is baked into the statement — for example, "prove that the sum of the first $n$ odd numbers is $n^2$." But a huge fraction of combinatorics problems are not phrased that way. They ask: "How many ways are there to...?" or "What is the maximum number of...?" The answer is not given; producing it is half the problem.

CombiBench encodes this by putting a **`sorry` placeholder** where the answer belongs in the formal statement. `sorry` is Lean's built-in token for "an unproven hole" — it lets a statement typecheck while marking a gap. In CombiBench's usage, the `sorry` does not mark a missing proof; it marks a missing *answer*. The model's job is to replace the `sorry`(s) with a concrete term — a number, a closed-form expression, a set — and then supply a proof that the now-complete statement holds. This is the formal analog of PutnamBench's answer-substitution style, but applied systematically across an entire combinatorics benchmark.

Here is the shape of what the model is handed and what it must return, in lightly cleaned Lean 4-flavored pseudocode. The point is to see where the blank lives:

```lean
-- A fill-in-the-blank problem: the answer is a `sorry` hole.
-- The model must replace `sorry` with a concrete value AND prove the theorem.

theorem imo_2019_p5_count
    (n : Nat) (hn : 0 < n)
    (process : CoinRow n -> Nat) :
    average_operations n process = sorry := by
  sorry
  -- ^ first sorry  : the ANSWER hole  (e.g. must become `n * (n + 1) / 4`)
  -- ^ second sorry : the PROOF hole   (the actual argument)
```

The grader's contract is strict and worth spelling out, because the strictness is the whole point. For a submission to count as solved, the output must:

- contain **no `sorry`** (every hole is filled),
- introduce **no new axioms** (you cannot smuggle in `theorem hard := by sorry`-equivalents through axiomatization),
- **compile in Lean without errors**, and
- **match the input formal statement exactly**, except for the `sorry`(s) that were supposed to be replaced.

That last condition is the anti-tampering clause. Without it, a model could "solve" a problem by quietly rewriting the statement into something trivially true. By pinning the statement to be byte-for-byte the input except at the designated holes, CombiBench forces the model to answer the actual question.

### Fine-Eval: the two-stage verification pipeline

Once the model fills the holes, Fine-Eval has to decide whether the filled answer is correct. The naive approach — string-compare the model's answer to the ground truth — fails immediately, because there are infinitely many ways to write the same mathematical object. `n*(n+1)/4`, `(n^2+n)/4`, and `Nat.choose (n+1) 2 / 2` may all denote the same function. So Fine-Eval grades equivalence *in Lean*, and it does so in two stages.

![The Fine-Eval two-stage verification pipeline](/imgs/blogs/combibench-3.png)

The pipeline above shows the full path. Walking it left to right and top to bottom:

- **Emit.** The model produces solution + proof as a single output.
- **Four hard gates.** No `sorry`, no new axioms, compiles cleanly, statement matches the input. Fail any one and the submission is rejected outright.
- **Anti-cheat length limit.** This is the most unusual gate, and it deserves its own paragraph below.
- **Answer check (Stage 1).** Lean attempts to prove the model's answer equal to the ground truth. If they are *syntactically identical*, it is solved immediately. If not, Lean tries to close the gap automatically with `rfl` (definitional equality) or `norm_num` (numeric normalization).
- **Stage 2 reprove.** If Lean *cannot* prove the answer equivalent with those tactics, the answer is treated as "non-trivial" — it might still be correct, just not reducible by `rfl`/`norm_num`. In that case the completed statement (with the model's answer substituted in) is fed back to the LLM for a second proving round. If the model can prove the original target with its own answer in place, it counts.
- **Solved verdict.** The Kimina Lean Server returns the final compiler-checked verdict, which counts toward the pass@k score.

The two-stage design is a thoughtful piece of engineering. The risk it guards against is a false negative: a model produces a correct but unusually-written answer, `rfl` and `norm_num` are not powerful enough to recognize the equivalence, and a one-stage grader would wrongly mark it wrong. Stage 2 catches that by asking the model to prove its own answer correct, which sidesteps the equivalence question entirely.

### The one-stage simplification

There is a lighter-weight variant. The **one-stage Fine-Eval** skips Stage 2 and checks answer equivalence with only `rfl`. This collapses the whole thing to a **single LLM call**, which is dramatically cheaper at evaluation time. The paper's most important empirical observation about the framework itself is that, in the without-solution setting, **one-stage and two-stage produced identical results.** I will return to why that is in the experiments section, because it tells you something uncomfortable about the current state of the models. The short version: the extra machinery of Stage 2 only matters when models produce *almost-correct* answers, and right now they essentially do not — they produce either fully correct or fully wrong answers, with little in between.

### The anti-cheating length constraint

Now the strangest, most pragmatic, and most revealing piece of the design. To stop models from "deceiving" the grader, Fine-Eval imposes a hard rule: **the number of characters in the output proof, after removing spaces and newline characters, must not exceed 42.**

That number looks absurd until you understand the attack it defends against. Lean's elaborator is powerful, and a sufficiently clever (or sufficiently lucky) model can sometimes find a short tactic incantation that closes a goal not because it found the intended argument but because it exploited a quirk — an overly aggressive automation tactic, a `decide` that brute-forces a small case, a metavariable trick. Worse, models can learn to emit grader-fooling boilerplate. By capping the whitespace-stripped proof at 42 characters, Fine-Eval makes it impossible to hide a degenerate "proof" inside a wall of text, and it forces genuine — necessarily short — proof terms.

The flip side, which the paper does not dwell on but which I think is the sharpest critique of the framework, is that a 42-character cap also rejects *legitimate* long proofs. A real combinatorics argument can easily run hundreds of characters. So this constraint is doing double duty: it is an anti-cheat measure, but it also implicitly restricts the benchmark to problems whose correct proofs are short, or to a regime where models are expected to lean heavily on Lean automation rather than spelling out arguments. That is a defensible design choice for a first benchmark, but it is a choice with consequences, and I will weigh it in the critique.

### Why fill-in-the-blank grading is the actual contribution

It is easy to read Fine-Eval as a pile of engineering details and miss that the central contribution is conceptual. Let me make the gap concrete with a before/after.

![Proof-only grading versus fill-in-the-blank grading](/imgs/blogs/combibench-4.png)

On the left is the world before Fine-Eval. A proof-only grader assumes the statement is **fully fixed in advance**; the model returns **one proof term**; and the grader checks that the proof closes the fixed goal. That machinery is completely blind to the 45% of CombiBench problems whose answer is *not* fixed — there is no goal to prove until you know the answer. A proof-only grader literally cannot be pointed at these problems.

On the right is Fine-Eval. The `sorry` placeholder marks the blank; the model fills the answer and *then* proves it; `rfl`/`norm_num` check answer equivalence; and in the one-stage mode the whole thing is a single LLM call. The framework does not just add a feature — it expands the set of problems that are automatically gradeable in a formal setting to include the entire "find the answer, then prove it" genre that dominates competition combinatorics.

### A worked example: what "without-solution" actually demands

To make the difficulty visceral, walk through what the without-solution protocol asks of a model on a representative counting problem. Suppose the informal statement is the classic "in how many ways can $n$ distinguishable balls be placed into $k$ distinguishable boxes so that no box is empty?" The human-readable answer is a surjection count, $k! \cdot S(n, k)$ where $S(n,k)$ is a Stirling number of the second kind, or equivalently the inclusion-exclusion sum $\sum_{i=0}^{k} (-1)^i \binom{k}{i} (k-i)^n$. A competent solver writes that down in a line.

Now look at what the model has to do in CombiBench's without-solution regime, step by step:

1. Parse the formal statement and locate the `sorry` answer hole, which sits inside a `Nat`-valued equality goal.
2. Reason informally to the closed form (e.g. the inclusion-exclusion surjection count).
3. Express that closed form as a well-typed Lean 4 term over `Nat`, using the right Mathlib names (`Finset.sum`, `Nat.choose`, ...).
4. Replace the `sorry` with that term so the statement type-checks.
5. Prove the now-complete equality — which often means proving a bijection or a counting identity that Mathlib does **not** ship.
6. Keep the whitespace-stripped proof body under 42 characters, or lean on automation tactics that close it in one shot.
7. Emit no `sorry`, define no new axioms, and leave the rest of the statement byte-identical to the input.

Steps 2 and 3 are where general reasoning models actually do fine — the paper's "minimal difference between scenarios" finding tells us the models usually get the answer. Step 5 is where everything collapses. The supporting counting lemmas are frequently absent from Mathlib (root cause #2), so the model must either reconstruct them from primitives inside a 42-character budget (often impossible) or find an automation tactic powerful enough to brute-force the specific instance (rare for symbolic $n$). This is the formalization wall the benchmark is designed to expose, and walking the seven steps makes it obvious why handing the model the answer (the with-solution regime) barely helps: the answer was never the hard part.

### A comparison table of the two protocols

The last piece of method to pin down is the two evaluation scenarios, because the gap (or lack of gap) between them is one of the paper's most informative results.

| Dimension | With-solution | Without-solution |
|---|---|---|
| Ground-truth answer provided? | Yes — handed to the model | No — model must produce it |
| What the model must do | Prove the (already complete) statement | Fill the `sorry` answer, then prove |
| Isolates which skill? | Formal proof construction | Reasoning + formalization + proof |
| Fine-Eval stages used | Proof check | One-stage or two-stage |
| Best system score (pass@16) | 7 / 100 (Kimina-Prover) | 7 / 100 (Kimina-Prover) |
| Best general model (pass@16) | 4 / 100 (Gemini-2.5-pro) | 3 / 100 (Gemini-2.5-pro) |

The with-solution scenario is the cleaner experiment for measuring proof ability alone: you remove the "did the model figure out the answer" confound and ask only "can it write the Lean proof." The without-solution scenario is the full task. The fact that the best prover scores 7/100 in *both* — meaning handing it the answer barely helps — is a finding I will unpack shortly.

## Experiments

The experimental setup is deliberately simple, which is appropriate for a benchmark paper: take a representative set of state-of-the-art systems, run them on all 100 problems under both protocols, and report pass@k for k in {1, 8, 16}. No model in the evaluation was trained specifically for combinatorics or for fill-in-the-blank formal problems — the authors are explicit that "none of them has been trained for this particular task." So every number below is a measurement of *transfer*: how well does general reasoning or general theorem-proving ability carry over to formal combinatorics with zero task-specific training.

Two families were evaluated. The **reasoning / general LLMs** were o1, o3-mini, QwQ, Claude-3.7-Sonnet-thinking, DeepSeek-R1, and Gemini-2.5-pro-preview. The **theorem provers** were Goedel-Prover, STP, Leanabell-Prover-GD-RL, and Kimina-Prover Preview (the 72B model fine-tuned for theorem proving). Evaluation was capped at **pass@16** for cost and reproducibility — no tree search, no larger sampling budgets.

A quick note on what pass@k means here, because it shapes how you should read the tables. Pass@k is the count of problems solved within $k$ independent samples: a problem counts as solved if *any one* of the $k$ generated outputs passes Fine-Eval's gates. So pass@1 measures single-shot reliability — did the very first attempt compile and check — while pass@16 measures whether the model can find a valid proof *at all* within a modest budget. The gap between a model's pass@1 and pass@16 tells you how much of its ability is luck-of-the-draw versus reliable. Kimina-Prover going from 2 (pass@1, with-solution) to 7 (pass@16) means most of its solves are not first-try; they require resampling. The general models going from 0 (pass@1) to 2 to 4 (pass@16) means they essentially never solve a problem on the first attempt and only occasionally stumble into a valid proof across 16 tries. With single-digit totals, the difference between pass@8 and pass@16 is one or two problems — which is why the sampling-budget question (does pass@256 keep climbing?) is so consequential and so frustratingly unanswered.

![Headline results: the ceiling is seven problems out of one hundred](/imgs/blogs/combibench-5.png)

The matrix above is the headline. Let me put the full numbers in a table so every cell is auditable against the paper's Table 3. Scores are "problems solved," reported as pass@1 / pass@8 / pass@16.

### Reasoning / general LLMs

| Model | With-solution (p@1 / p@8 / p@16) | Without-solution (p@1 / p@8 / p@16) |
|---|---|---|
| o1 | 0 / 2 / 2 | 0 / 2 / 2 |
| o3-mini | 0 / 1 / 1 | 0 / 2 / 2 |
| QwQ | 0 / 2 / 2 | 0 / 2 / 2 |
| Claude-3.7-Sonnet-thinking | 0 / 2 / 2 | 0 / 0 / 0 |
| DeepSeek-R1 | 0 / 2 / 2 | 1 / 2 / 2 |
| Gemini-2.5-pro-preview | 0 / 2 / **4** | 0 / 2 / 3 |

### Theorem provers

| Model | With-solution (p@1 / p@8 / p@16) | Without-solution (p@1 / p@8 / p@16) |
|---|---|---|
| Goedel-Prover | 0 / 0 / 0 | 0 / 0 / 0 |
| STP | 0 / 0 / 0 | 0 / 0 / 0 |
| Leanabell-Prover-GD-RL | 0 / 0 / 0 | 0 / 0 / 0 |
| **Kimina-Prover Preview (72B)** | **2 / 6 / 7** | **1 / 4 / 7** |

Several things jump out, and each one is load-bearing.

**The absolute ceiling is brutally low.** The best system on the planet for this task solves 7 of 100. For calibration, this is a benchmark where a competent human combinatorialist with Lean experience would, given time, formalize and prove the vast majority — the difficulty for humans is in the *time* (3 to 8 hours per IMO problem), not in the possibility. The models are not bottlenecked on time; they are bottlenecked on capability. CombiBench is, right now, very far from saturated, which is exactly what you want from a new benchmark.

**Three dedicated provers score zero.** Goedel-Prover, STP, and Leanabell-Prover-GD-RL — systems explicitly built for Lean theorem proving — solve *nothing*, at any pass@k, under either protocol. This is the most counterintuitive result in the paper for anyone who assumed that purpose-built provers would dominate. The likely explanation is distribution: these provers were trained on corpora dominated by algebra, number theory, and miniF2F-style problems, and combinatorics is so far out of distribution that their learned tactics simply do not fire. A prover that has never seen the relevant Mathlib lemmas (because they barely exist) and has never been trained on counting arguments has nothing to transfer.

**Kimina-Prover is the lone exception, and only barely.** It reaches 7/100, which is a genuine gap over the field but is still a single-digit score. It is the only prover with a nonzero pass@1 (2 with-solution, 1 without-solution), meaning it occasionally nails a problem on the first try rather than needing 16 samples to get lucky.

**Handing the model the answer barely helps.** Look at Kimina-Prover: 7/100 with-solution and 7/100 without-solution at pass@16. The with-solution scenario *gives away the answer* and only asks for a proof — and the score is identical. The paper draws the right conclusion from this: for these models, **informal reasoning is not the bottleneck; formal proof construction is.** The models are not failing because they cannot figure out the answer (general reasoning models show "minimal difference" between the two scenarios too). They are failing because translating a known argument into a compiling Lean proof of a combinatorics statement is the hard part. This is a precise, actionable diagnosis, and it is the single most valuable output of the entire study.

**Gemini-2.5-pro is the best general model.** At 4/100 (pass@16, with-solution) it edges out the rest of the reasoning-LLM field and lands above every dedicated prover except Kimina. That a general reasoning model outperforms three specialized provers is a small embarrassment for the specialized-prover paradigm, and a hint that broad reasoning capability transfers to formal combinatorics better than narrow proving capability trained on the wrong distribution.

### The one-stage vs two-stage observation

Recall that in the without-solution setting, one-stage Fine-Eval (cheap, single call, `rfl`-only) and two-stage Fine-Eval (with the reprove fallback) gave **identical results**. The interpretation is sharp and a little bleak: given current model capability, predicted solutions are "typically either completely correct or entirely wrong." There is almost no middle ground where a model produces a *correct-but-awkwardly-written* answer that `rfl` misses and Stage 2 rescues. When models are right, they are right in a way `rfl` already catches; when they are wrong, no reprove round saves them. The two-stage machinery is built for a regime of near-misses that the models have not yet reached. That is a forward-looking design — it will start to matter once models get better — but today it is dormant.

### What is load-bearing, and what might not transfer

A benchmark paper lives or dies on whether its numbers mean what they appear to mean, so let me be explicit about the assumptions holding these results up.

The **pass@16 cap** is the biggest one. The scores are a measurement under a fixed, modest sampling budget with no tree search. We know from AlphaProof-style systems that search over many candidate proofs, sometimes thousands, can dramatically lift solve rates. So 7/100 is not "the best a system can ever do" — it is "the best these systems do at pass@16 without search." A system willing to spend orders of magnitude more compute per problem, or to integrate a proper proof-search loop, could plausibly score much higher. The paper is honest that larger budgets and tree search were not evaluated. If you read 7/100 as a hard ceiling rather than a budget-constrained floor, you are over-reading.

The **42-character anti-cheat cap** is the second load-bearing assumption, and it cuts the other way. It may be *deflating* scores by rejecting legitimate longer proofs. We cannot tell from the reported numbers how many submissions were rejected purely on length versus on incorrectness. If a meaningful fraction of correct-but-long proofs were filtered out, the true proving ability is being undercounted.

The **zero task-specific training** caveat is the one to keep in mind before generalizing. None of these models saw combinatorics-specific or fill-in-the-blank-specific training data. A model fine-tuned on even a modest corpus of formal combinatorics — once such a corpus exists, which is partly what CombiBench enables — could behave very differently. So the results characterize *current off-the-shelf transfer*, not the asymptotic difficulty of the task.

## Critique

I want to be fair to a paper that is, on the whole, an unusually clean and useful contribution. But "useful" and "above critique" are different things, and a benchmark deserves to be pressure-tested precisely because the field will calibrate its sense of progress against it.

**What is strong.** The problem selection is principled and defensible: all IMO combinatorics problems since 2000 is a clear, reproducible inclusion rule, and randomly sampling 3 problems from each of Brualdi's 14 textbook chapters gives broad topical coverage without cherry-picking. The Fine-Eval framework solves a real, previously-unsolved problem — automatic grading of fill-in-the-blank formal math — and it solves it with a design (statement-pinning, no-new-axioms, `rfl`/`norm_num` equivalence, optional reprove) that is genuinely robust against the obvious gaming strategies. The two-protocol design (with vs without solution) is what lets the authors make the crisp claim that formalization, not reasoning, is the bottleneck — and that claim is the most valuable single sentence in the paper. The Lean 4 nativeness and the ≥ 4.15.0 version pin mean the benchmark is usable *today* by the current generation of provers, which is exactly where the prior benchmarks failed.

**What is weak or under-specified.** The most significant gap is **the absence of any inter-annotator agreement or independent verification of the formalizations.** Six experts formalized 100 problems, some taking over 8 hours each, but the paper describes no process for double-checking that a formal statement faithfully captures the informal problem. In formal math this is the cardinal risk: a subtly wrong formalization can be *trivially true* or *trivially false* or *true but not what the problem asked*, and a model could "solve" it for the wrong reasons, or be unfairly blocked from solving a mis-stated problem. With only single-digit solve rates, even a handful of mis-formalized problems could materially shift the rankings. A benchmark whose entire value is the trustworthiness of its statements should report how that trust was established.

The **42-character cap is unfalsifiable as currently reported.** The paper presents it as an anti-cheat measure but gives no data on its side effects: how many otherwise-valid submissions did it reject? Without that, we cannot distinguish "models can't prove these" from "models can prove these but their proofs exceed 42 characters." Those are very different conclusions and the cap blurs them.

The **"identical one-stage / two-stage results" claim** is presented as evidence that answers are all-or-nothing, but it is equally consistent with a more deflationary reading: the models so rarely produce *any* correct answer that there were simply too few cases for the two stages to diverge. With solve counts in the low single digits, "identical results" might just mean "identical near-zero." The paper would be stronger with the raw counts of how many submissions reached Stage 2 at all.

**The missing ablation.** The most informative experiment the paper does *not* run is a **search-budget ablation**: hold the model fixed (say Kimina-Prover) and sweep the sampling budget from pass@16 up to pass@256 or higher, ideally with a proof-search loop. That single sweep would tell us whether 7/100 is a capability wall or a compute wall — which is the question every reader actually has. The paper explicitly defers this to cost, which is understandable, but it leaves the headline number ambiguous in exactly the dimension that matters most.

**What would change my mind.** I currently read CombiBench's 7/100 as evidence that formal combinatorics is genuinely far from solved and that the bottleneck is formalization rather than reasoning. I would revise that if a follow-up showed that (a) scaling the search budget on a fixed model lifts solve rates dramatically — say above 30/100 at pass@256 — which would reframe the result as a search problem rather than a capability gap; or (b) a model fine-tuned on even a few thousand formal combinatorics proofs jumps to double or triple the current ceiling, which would mean the gap is a data problem (Mathlib coverage + training corpus) rather than a fundamental reasoning limitation. Either result would be very good news, and either would substantially change how I'd describe the difficulty of this domain.

## What I'd build with this

A good benchmark is a launchpad. Here are five concrete things I would build on top of CombiBench, in rough order of how much I think they would move the needle.

1. **A search-budget harness around Kimina-Prover.** The single most valuable experiment the paper leaves on the table. Wrap the best prover in a proper proof-search loop — best-first search over tactic states, with the LLM as the policy and Lean as the verifier — and sweep the budget. This is mostly engineering, not research, and it would immediately answer whether 7/100 is a wall or a budget. I would instrument it to log, per problem, whether the failure was "never produced a compiling proof" versus "produced one but it exceeded the 42-char cap," which also probes the cap's side effects.

2. **A Mathlib combinatorics gap-filler.** The paper names sparse Mathlib coverage as a root cause. I would mine the 100 CombiBench formalizations for the supporting lemmas they had to prove inline — double counting, pigeonhole variants, specific bijections — and turn the recurring ones into Mathlib contributions. Every lemma that moves from "proved inline by the formalizer" to "available in the library" lowers the formalization tax for every future problem and every prover. This is the kind of infrastructure work that compounds.

3. **A fill-in-the-blank data-augmentation pipeline.** Fine-Eval's `sorry`-as-answer representation is mechanical enough to automate. Given a corpus of solved combinatorics problems, I would programmatically generate fill-in-the-blank variants — blank out the answer, keep the proof skeleton — to build the task-specific training corpus that, per the paper, no current model has. Then fine-tune a prover on it and re-run CombiBench to test the "it's a data problem" hypothesis directly.

4. **A formalization-faithfulness checker.** To address the inter-annotator gap, I would build a semi-automated cross-checker: take the informal statement and the formal statement, and use an LLM (with a careful prompt and Lean in the loop) to flag candidate mismatches — statements that are trivially true, vacuously satisfiable, or that quantify differently than the English. Even an imperfect flagger would catch the most dangerous formalization bugs and raise the benchmark's trustworthiness.

5. **A difficulty-stratified leaderboard.** The benchmark already carries difficulty labels (Hackmath middle-school through IMO). I would report scores *stratified* by difficulty tier rather than as a single aggregate, so progress on the easy tier (Hackmath, easy Brualdi) is visible separately from the IMO tier. With solve counts this low, a single aggregate hides where the gains are happening; stratification would make incremental progress legible and keep the field honest about whether it is solving genuinely hard problems or just the easy tail.

## When to reach for CombiBench (and when not to)

CombiBench is a specialized instrument, and using it well means knowing what it measures and what it doesn't.

**Reach for it when** you are building or evaluating a system whose explicit goal is formal mathematical reasoning in Lean 4, and you want an honest, unsaturated, combinatorics-focused signal. It is the right benchmark if you are training a theorem prover and want to know whether your improvements transfer to the hardest formalization regime, or if you are studying the informal-to-formal gap and want a domain where that gap is maximal. It is also the right tool if you specifically need to evaluate fill-in-the-blank formal ability — the "find the answer, then prove it" genre — because Fine-Eval is, as of this paper, the only framework that grades it automatically.

![Where the 100 problems come from](/imgs/blogs/combibench-6.png)

The composition tree above is worth keeping in view when you interpret a score: 42 of the 100 problems are Brualdi textbook exercises (the more tractable end), 36 are IMO combinatorics (the brutal end), 10 are middle-school Hackmath, and 12 are a long tail of other olympiads. A system that scores 7/100 is almost certainly clearing some of the easier Brualdi and Hackmath problems and bouncing off the IMO tier — which is exactly why a difficulty-stratified reading (point 5 above) matters before you draw conclusions about "combinatorial reasoning" in the abstract.

**Do not reach for it when** you need a quick, high-throughput signal for general LLM capability — the scores are too sparse and too expensive to compute (Lean compilation at scale) for rapid iteration, and a 0-to-7 dynamic range gives you very little resolution between mediocre systems. Do not use it to claim a model "can't do combinatorics" in the informal sense; the paper's own finding is that informal reasoning is *not* the bottleneck, so a low CombiBench score is primarily a statement about formalization and Lean proof construction, not about whether the model grasps the combinatorial argument at an informal level. And do not treat the headline 7/100 as a fixed ceiling: it is a pass@16, no-search, no-task-training number, and the most likely near-term development is that better search and a real training corpus push it up substantially.

The deeper reason CombiBench matters is that it converts a vague, widely-shared intuition — "combinatorics is the hard case for formal AI" — into a concrete, reproducible number that a research program can be steered by. The AlphaProof story (two combinatorics problems missed) was an anecdote. CombiBench turns the anecdote into a measurement, and measurement is what lets a field stop guessing and start optimizing. For everyone working on the neurosymbolic frontier — and it is worth reading this alongside the formal-reasoning lineage in [Kimina-Prover](/blog/paper-reading/reasoning/kimina-prover) and the RL-scaling story in [Kimi k1.5](/blog/paper-reading/reinforcement-learning/kimi-k1-5) — that is the contribution that will outlast the specific 7/100 score.

## References

- **CombiBench: Benchmarking LLM Capability for Combinatorial Mathematics** — arXiv abstract: [https://arxiv.org/abs/2505.03171](https://arxiv.org/abs/2505.03171)
- **Code & data (GitHub):** [https://github.com/MoonshotAI/CombiBench](https://github.com/MoonshotAI/CombiBench)
- Related reading on this blog:
  - [Kimina-Prover: Large Formal Reasoning Models with RL](/blog/paper-reading/reasoning/kimina-prover) — the prover that tops the CombiBench leaderboard, read from the inside.
  - [Kimi k1.5: Scaling Reinforcement Learning with LLMs](/blog/paper-reading/reinforcement-learning/kimi-k1-5) — the RL-scaling lineage behind modern reasoning models.
  - [Kimi K2 Thinking: An Open-Source Reasoning Model Built on K2](/blog/paper-reading/large-language-model/kimi-k2-thinking) — where general reasoning capability comes from.
  - [Kimi-Dev: Agentless Training as Skill Prior for SWE Agents](/blog/paper-reading/large-language-model/kimi-dev) — another study of how narrow-task skill transfers (or doesn't).

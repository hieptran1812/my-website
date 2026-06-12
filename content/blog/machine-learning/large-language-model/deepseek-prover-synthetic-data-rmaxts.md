---
title: "DeepSeek-Prover V1 and V1.5: Synthetic Lean Data, Proof-Assistant-Feedback RL, and RMaxTS"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A technique deep-dive on how DeepSeek-Prover V1 manufactured eight million verified Lean 4 proofs out of competition problems, and how V1.5 added GRPO-from-verifier-feedback, truncate-and-resume whole-proof generation, and the RMaxTS intrinsic-reward tree search that makes sparse proof rewards trainable."
tags: ["llm", "deepseek-prover", "theorem-proving", "lean4", "formal-mathematics", "synthetic-data", "expert-iteration", "grpo", "reinforcement-learning", "monte-carlo-tree-search", "rmaxts", "autoformalization"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 50
---

Here is a rule of thumb I repeat to every team that wants to "do RL on a math model": the bottleneck is almost never the algorithm, it is the supply of problems you can actually grade for free. For numeric-answer competition math the grader is a string comparison against a boxed integer, and the reward signal costs nothing. For formal theorem proving the grader is even better — a proof either type-checks in a kernel or it does not, with zero ambiguity and zero reward hacking — but the *supply* collapses. Humanity has produced, across decades of effort in Lean, Coq, and Isabelle, on the order of low hundreds of thousands of formalized theorems. That is a rounding error next to the token budgets a 7B model wants. So the central problem of neural theorem proving is not "how do we search" and not "how do we reward"; it is "where do we get millions of formal statements that are real, non-trivial, and provable, when humans have only ever written a few hundred thousand."

DeepSeek-Prover V1 (arXiv 2405.14333, May 2024) is the most aggressive answer to that supply problem I have seen, and DeepSeek-Prover-V1.5 (arXiv 2408.08152, August 2024, later ICLR 2025) is the answer to the *search* problem that the data unlocks. The two papers are a matched pair: V1 builds a data engine that turns 869,659 natural-language competition problems into roughly eight million verified Lean 4 statement-and-proof pairs by autoformalization plus expert iteration; V1.5 takes the model trained on that corpus and adds reinforcement learning from proof-assistant feedback, a whole-proof generation scheme that recovers from its own errors, and a Monte-Carlo tree search variant — RMaxTS — engineered specifically for the brutal sparsity of the proof-completion reward. The later [V2 follow-up](/blog/paper-reading/large-language-model/deepseek-prover-v2-advancing-formal-mathematical-reasoning-via-reinforcement-learning-for-subgoal-decomposition) builds a subgoal-decomposition pipeline on top of exactly these two ideas, which is why understanding V1 and V1.5 first is not optional — it is the prerequisite.

![The synthetic-data engine turns 869,659 natural-language competition problems into a verified ~8M-proof Lean 4 corpus through autoformalization, two-stage filtering, kernel verification, and expert iteration](/imgs/blogs/deepseek-prover-synthetic-data-rmaxts-1.webp)

The diagram above is the mental model for the whole of V1: a funnel that ingests messy natural-language problems on the left and emits a clean, kernel-verified proof corpus on the right. Every box in that funnel exists to solve one failure of naive autoformalization — bad translations, vacuously-true statements, unprovable garbage — and the entire rest of this article is a tour of why each stage is load-bearing, followed by the V1.5 machinery that learns to search the space those proofs define. We will not re-derive V2 here; the scope is deliberately the engine and the RL-plus-MCTS substrate that V2 inherits.

## Why formal theorem proving breaks the usual recipe

Before the mechanics, it is worth being precise about why the standard RLVR (reinforcement learning from verifiable rewards) recipe — the one that took models from mediocre to saturated on AIME inside a year — does not transplant cleanly into Lean.

| Assumption from numeric-answer RL | Naive port to theorem proving | Reality in Lean 4 |
| --- | --- | --- |
| Problems are abundant; scrape more from the web | Scrape competition problems, train directly | Web problems are in *natural language*; Lean needs *formal* statements, and translation is itself an unsolved task |
| The grader is cheap and dense | Run the proof, check the answer | The kernel is cheap, but a *successful* proof is rare, so reward is almost always 0 — sparse, not dense |
| Any correctly-answered problem is a positive | Any type-checking proof is a positive | A statement with inconsistent hypotheses type-checks *anything*, including `False`; it is a poisoned positive |
| Train on (problem, answer) pairs | Train on (statement, proof) pairs | You must *manufacture* both the statement and the proof, and verify the statement is even worth proving |
| One rollout, one scalar reward | Generate full proof, get 0/1 | Full-proof generation throws away partial progress on every failure unless you engineer around it |

Three of those rows are the spine of V1's data engine, and the bottom two are the spine of V1.5's training and search. The reason the problem is hard is that *both* halves of a training example — the statement and its proof — have to be synthesized and certified, and the certification of the *statement* (is it true? is it non-vacuous?) is a different and subtler check than the certification of the *proof* (does it type-check?). Get the second without the first and you train a model to be confidently fluent in proving things that were never worth stating.

> A proof that type-checks is a fact about the *proof*. Whether the *statement* deserved a proof is a separate question, and the entire art of synthetic formal data is answering it at scale without a human in the loop.

The base model for all of this is **DeepSeekMath-Base 7B** — the same continued-pretraining checkpoint whose data pipeline and the GRPO algorithm it introduced are covered in the sibling [DeepSeekMath data-pipeline and GRPO-origin post](/blog/machine-learning/large-language-model/deepseekmath-data-pipeline-and-grpo-origin). That matters because Prover does not start from a generic code model; it starts from a checkpoint already steeped in mathematical text and informal reasoning, which is what makes the *autoformalization* step — translating English math into Lean syntax — tractable at all.

## 1. Autoformalization: turning English problems into Lean statements

**Senior rule of thumb: an autoformalizer is a translator that you must assume is fluent but unreliable — design every downstream stage to catch its lies.**

The raw input to V1 is a corpus of 869,659 high-school and undergraduate-level competition problems scraped and cleaned from the web — the kind of material you find on olympiad archives and problem aggregators. Each one is a sentence or two of natural-language mathematics: "Prove that for all positive reals $a, b$ we have $a^2 + b^2 \geq 2ab$," or "Find all integers $n$ such that $n^2 + 1$ is divisible by 5." None of these is a Lean object. The first job of the engine is to prompt the model to emit a corresponding Lean 4 *theorem statement* — the signature, not the proof — of the form:

```lean
theorem amgm_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    a ^ 2 + b ^ 2 ≥ 2 * a * b := by
  sorry
```

The model produces the part before `:= by`; the `sorry` is a placeholder the proof search will later try to discharge. This is where the first wave of errors enters. Autoformalization is a hard task even for frontier models, and the failure modes are specific:

- **Syntactic noise**: the statement does not parse, references a lemma name that does not exist in Mathlib, or uses a coercion that Lean rejects. These are caught immediately — Lean refuses to even elaborate the statement, and it is dropped.
- **Semantic drift**: the statement parses and type-checks but means something subtly different from the English. "For all positive reals" becomes "for all reals," or a strict inequality becomes non-strict. The statement is *valid Lean* but *not the problem*.
- **Vacuity**: the hypotheses are mutually inconsistent, so the statement is vacuously true and a proof tells you nothing. This is the poisoned-positive case from the table above and it is the most dangerous, because it survives every syntactic check.

The engineering response is a two-stage filter — quality scoring followed by hypothesis rejection — and then, downstream, a search procedure (dual-path) that quietly discards the unprovable remainder. The point worth internalizing: V1 never trusts the autoformalizer. It treats every emitted statement as a *candidate* that must survive a gauntlet before it earns a place in the training corpus. The 869,659 inputs do not become 869,659 statements; they become far fewer, and only the survivors get proofs attached.

### The prompt design that makes autoformalization tractable

The autoformalizer is not a fine-tuned specialist in V1's first rounds; it is the base model prompted carefully, and the prompt is doing real work. Three design choices matter. First, the prompt provides the natural-language problem *and* asks for the Lean 4 theorem signature only, with the proof body left as `sorry` — separating statement synthesis from proof synthesis means a translation error and a proof error never get conflated into one un-debuggable failure. Second, the prompt pins down the convention for the variable domain: a competition problem that says "positive reals" must produce explicit `(ha : 0 < a)` hypotheses rather than leaving positivity implicit, because Lean has no notion of an implicit domain and a missing positivity hypothesis is the single most common way a true problem becomes a false formal statement. Third, the prompt encourages the use of Mathlib's standard names (`Real`, `Nat`, `Finset`, the standard inequality lemmas) so the emitted statements live in a namespace the prover actually knows how to manipulate. None of this guarantees correctness — that is what the downstream filters are for — but it raises the hit rate of the autoformalizer enough that the filters have a workable yield to cull from rather than a stream of pure noise.

A concrete sense of the failure distribution helps calibrate expectations. Of a batch of autoformalized statements, a large fraction fail to *elaborate* at all — Lean cannot even type-check the signature because of a bad lemma name, a coercion mismatch, or a syntax slip — and those are rejected for free before any scoring. Of the ones that elaborate, the model-based quality score then judges *faithfulness*, which is a strictly harder property: a statement can elaborate cleanly and still mean the wrong thing. The 869,659-to-712,073 cull is the faithfulness cull; the elaborate-or-die cull happens upstream of it and is essentially free because Lean does it for you. Stacking a free syntactic filter under a cheap semantic filter under an expensive logical filter is the ordering discipline that keeps the whole engine economical.

### Second-order optimization: scoring before you spend compute

The non-obvious cost here is search budget. Trying to prove a statement is expensive — it means sampling many candidate proofs and running each through the kernel. If 40% of your statements are garbage, you have burned 40% of an enormous compute bill producing proofs of nonsense (or, worse, *failing* to prove nonsense and learning nothing). So the filters are ordered cheapest-first: a fast model-based quality score culls the obvious dreck before the expensive prove-or-disprove search ever runs. That ordering is the difference between a pipeline that finishes and one that does not.

There is a deeper budget argument hiding in the ordering. The model-based quality score costs one cheap forward pass per statement. Hypothesis rejection costs a *full proof search* per statement. Dual-path proving costs *two* full proof searches. If you ran them in the reverse order — proving first, scoring last — you would pay the two-search cost on every piece of autoformalized garbage before the cheap scorer ever got a chance to throw it out. By running the $O(1)$-forward-pass scorer first and the $O(\text{search})$ logical filters last, you ensure the expensive operations only ever touch statements that have already cleared the cheap bar. The 18% the scorer culls is 18% of your search budget you never spend, and on a corpus the size of this one that is an enormous absolute saving in GPU-hours.

## 2. Two-stage filtering: quality scoring then hypothesis rejection

**Senior rule of thumb: filter for translation quality with a cheap heuristic, then filter for logical consistency with the kernel itself — never conflate the two.**

The first filter is **model-based quality scoring**. V1 prompts a scoring model to rate each autoformalized statement on a coarse five-point scale — *excellent*, *good*, *above-average*, *fair*, *poor* — judging whether the Lean statement is a faithful, non-degenerate rendering of the source problem. Statements rated *poor* (and, depending on the threshold, *fair*) are discarded. This is a pure-text judgment; it does not run Lean, so it is cheap and it runs first. After this pass, the 869,659 candidates are culled to **712,073** statements deemed worth the cost of a proof search. That is already a ~18% cull on translation quality alone, before any logical check.

The second filter is the clever one: **hypothesis rejection**. A statement can score *excellent* on faithfulness and still be worthless if its hypotheses are inconsistent, because from a contradiction you can prove anything — including the actual goal, and including `False`. A model trained on such "proofs" learns to exploit contradictions rather than reason. So V1 runs a logical consistency check that uses the proof assistant against itself.

![Hypothesis rejection: if Lean can prove the statement with its conclusion replaced by False, the hypotheses are inconsistent and the statement is discarded as worthless](/imgs/blogs/deepseek-prover-synthetic-data-rmaxts-2.webp)

The figure lays out the test as a keep-versus-drop decision. Take a statement $\Gamma \vdash P$, where $\Gamma$ is the set of hypotheses and $P$ is the conclusion. Replace the conclusion with `False` and ask the prover to discharge $\Gamma \vdash \texttt{False}$. If it succeeds, then $\Gamma$ alone — with no reference to $P$ — already entails a contradiction, which means $\Gamma$ is inconsistent and the original statement is vacuously true. Such a statement is discarded. If the prover *fails* to derive `False` from $\Gamma$ within the search budget, the hypotheses are (probably) consistent and the statement is retained as a genuine training target.

In Lean terms the check looks like this. Given the candidate, you synthesize a sibling goal whose conclusion is `False` and run the same proof search on it:

```python
def hypothesis_rejection(statement, prover, kernel, budget):
    """Drop statements whose hypotheses are inconsistent.

    statement.hypotheses : list of Lean hypothesis terms (the context Gamma)
    statement.conclusion : the goal P (unused here — we replace it with False)
    Returns True to KEEP the statement, False to DISCARD it.
    """
    falsified = make_goal(
        hypotheses=statement.hypotheses,
        conclusion="False",          # prove Gamma |- False
    )
    proof = prover.search(falsified, budget=budget)
    if proof is not None and kernel.verify(falsified, proof):
        # Gamma proves False on its own -> hypotheses are inconsistent.
        return False                 # DISCARD: vacuously true, teaches nothing
    return True                      # KEEP: hypotheses are (probably) consistent
```

Two engineering subtleties make this robust. First, the check is *one-sided and conservative in the right direction*: a failed search for `False` does not *prove* consistency — the search budget might simply have been too small — but a *successful* one *does* prove inconsistency, so every discard is justified and the only error mode is keeping a few bad statements, never wrongly throwing good ones away. Second, the same proof engine and the same kernel are reused, so there is no separate "consistency checker" to build or trust; the proof assistant's own soundness underwrites the filter. You are paying for a second proof search per statement, but you are buying a guarantee that survives adversarial autoformalizer outputs.

### Worked example: a statement that scores high and still gets dropped

Consider the autoformalized statement "for all reals $x$, if $x > 0$ and $x < 0$ then $x^2 = -1$." A faithfulness scorer might rate this *good* — it is grammatical Lean, the symbols are reasonable, and superficially it reads like a problem. But $x > 0 \land x < 0$ is `False`, so the hypotheses are inconsistent. Run hypothesis rejection: ask Lean to prove `(x > 0) → (x < 0) → False`. It succeeds trivially (`linarith` closes it in one tactic). The statement is discarded. Without this filter, the model would have happily produced a "proof" of $x^2 = -1$ from the contradiction and been *rewarded* for it, learning to treat impossible antecedents as license to assert anything. That is precisely the behavior you do not want a theorem prover to internalize, and it is invisible to any check that only looks at the proof and not at the statement.

### Second-order optimization: the filter doubles as a labeling signal

There is a quiet bonus here. The hypothesis-rejection search produces, as a side effect, a verified label for a chunk of your corpus: every statement on which `Γ ⊢ False` succeeded is *certified inconsistent*. You do not just drop these — you can mine them as negative examples for a statement-quality classifier, or as a sanity check on the autoformalizer's systematic biases (does it tend to flip inequality directions? to drop positivity constraints?). The cheapest information in a data pipeline is often the byproduct of a filter you were running anyway.

## 3. Dual-path search: proving and disproving at once

**Senior rule of thumb: when you do not know whether a statement is true, do not bet your whole search budget on one direction — race both directions and let the first proof decide.**

The quality scorer catches bad translations and hypothesis rejection catches inconsistent ones, but a third population survives both: statements that are perfectly well-formed, perfectly consistent, and simply *false*. "Find all $n$ such that..." mis-formalized into a universal claim that happens not to hold; an inequality that is true for the source problem's implicit domain but false for the domain Lean inferred. These will never be proved, and a naive pipeline that only ever tries to prove $P$ will spend its entire budget failing on them and emit no signal at all.

![Dual-path search forks one statement into concurrent proofs of P and of not-P; whichever returns first decides whether the statement is kept as a positive or discarded as false](/imgs/blogs/deepseek-prover-synthetic-data-rmaxts-3.webp)

The figure shows the fix as a fork. For each statement $\Gamma \vdash P$, V1 launches *two* proof searches concurrently: one for $\Gamma \vdash P$ and one for $\Gamma \vdash \lnot P$. The two searches draw from the same model and run in the same worker pool. Whichever returns a verified proof first wins, and the loser is cancelled:

- If the proof of $P$ returns first, the statement is *true* — keep it, and you now have a verified proof to add to the corpus.
- If the proof of $\lnot P$ returns first, the statement is *false* — discard it (and, optionally, keep the disproof of $\lnot P$ as a proof for the negated statement, which is itself a valid training example).
- If neither returns within the budget, the statement is *undecided* by the current model and is set aside, possibly to be retried by a stronger model in a later expert-iteration round.

The practical payoff is that the search budget is never wasted *blindly*. On a true statement, the $P$-search succeeds and the $\lnot P$-search would have failed — but you cancelled it the moment $P$ closed, so you only paid for the half that mattered. On a false statement, the $\lnot P$-search succeeds quickly (false statements often have short disproofs — a counterexample, a `decide`, an `omega` over the integers) and you have *labeled* the statement rather than merely failed on it. The dual-path scheme converts a silent failure into an informative one, and it does so for free relative to a single-direction search that would have spent the same budget producing nothing.

This is also where the data engine quietly becomes a *self-curating* loop. The autoformalizer's false-positive translations do not pollute the corpus, because the moment the disproof closes, the statement is removed. The corpus that reaches training is not just "statements the autoformalizer emitted" — it is "statements that survived quality scoring, are logically consistent, and have been *proven true* by a kernel." Each of those three adjectives is enforced by a different stage, and the dual-path search is what enforces the last one without leaving false statements in limbo.

It is worth distinguishing dual-path search from hypothesis rejection, because they look similar and solve different failures. Hypothesis rejection asks "are the *premises* self-contradictory" by proving $\Gamma \vdash \texttt{False}$ — it catches vacuous statements whose hypotheses can never all hold. Dual-path search asks "is the *conclusion* actually entailed" by proving $\Gamma \vdash \lnot P$ — it catches statements with perfectly consistent premises whose conclusion is simply wrong. A statement can pass hypothesis rejection (its premises are consistent) and still be false (its conclusion does not follow), so you need both checks; neither subsumes the other. Together they form a logical sieve: hypothesis rejection removes the vacuously-true, dual-path removes the plainly-false, and what survives is the set of statements that are consistent, non-vacuous, and — once the $P$-search closes — verifiably true. That is the only population worth attaching a training proof to, and getting there cheaply is the entire reason the engine is structured as a cascade of progressively more expensive filters rather than one monolithic check.

### Second-order optimization: budget asymmetry between the two paths

A subtlety the paper's framing invites: the two paths are not symmetric in expected cost. Disproofs of mis-formalized statements are usually *shorter* than proofs of genuine theorems, because a single counterexample or a decision procedure suffices. So a smart scheduler can bias slightly toward the $\lnot P$ path early — if a cheap disproof exists, find it fast and free the worker — while letting the $P$ path run longer before giving up. In practice the concurrency makes this mostly self-correcting (the cheap path returns first by construction), but it is worth knowing that "race both directions" has a favorable cost profile precisely because the failure-labeling direction tends to be the cheaper one.

## 4. Expert iteration: bootstrapping eight million proofs

**Senior rule of thumb: a model that can verify its own outputs can train on its own outputs — the only discipline you need is to keep nothing the verifier rejects.**

So far we have a model that emits statements and a gauntlet that keeps only the good, consistent, true ones with verified proofs. But where do the *proofs* come from at scale? The answer is **expert iteration**, the bootstrap loop that turns a mediocre initial prover into one strong enough to generate a corpus measured in millions.

![Expert iteration loops sampling, kernel verification, dataset growth, and retraining so the prover improves on its own verified outputs across rounds](/imgs/blogs/deepseek-prover-synthetic-data-rmaxts-4.webp)

The figure traces one round of the loop and how it opens the next. Start with prover $M_k$ (round 0 is DeepSeekMath-Base 7B lightly fine-tuned on whatever seed proofs exist). For each statement in the working set, sample $K$ candidate proofs. Run every candidate through the Lean 4 kernel. Discard the ones that error or time out; keep the ones that verify. Add the verified (statement, proof) pairs to a growing corpus. Fine-tune on the enlarged corpus to produce $M_{k+1}$, which is strictly better at proving — and crucially, can now close statements that $M_k$ could not. Feed $M_{k+1}$ back in and repeat.

The reason this converges to a strong model rather than collapsing is the kernel. Self-training loops normally rot because the model trains on its own *unverified* outputs and amplifies its own errors — the classic model-collapse failure. Here, *nothing enters the corpus that the kernel did not certify*. The proof is correct or it is gone. There is no noisy label, no reward-model approximation, no human annotator drift. The verifier is exact and adversarial-proof, so the only thing the loop can do is accumulate *more correct proofs* and *broaden coverage* to harder statements as the model strengthens. Across enough rounds — combined with the statement-generation engine producing fresh targets — the corpus grows to roughly **eight million** verified statement-and-proof pairs.

A subtle but important detail: expert iteration is not just about *more* proofs, it is about *harder* proofs. Round 0's model closes the easy statements. Those proofs train round 1, which can now close moderately harder statements, whose proofs train round 2, and so on. The difficulty frontier marches outward. This is the same dynamic that makes self-play work in games — the curriculum is generated by the agent's own improving competence — except that here the "win condition" is a kernel check, which cannot be gamed. You get the curriculum-generation benefit of self-play with none of the reward-hacking risk, because the reward is a theorem-prover, not a learned critic.

There is one more property of expert iteration worth naming, because it is the reason the loop is *stable* rather than merely convergent. In a self-training loop with a noisy label — a learned reward model, a heuristic, a majority vote — the model's errors compound: it trains on its own mistakes, drifts, and the drift accelerates because each round's training set is dirtier than the last. The kernel breaks that feedback entirely. The label is not an estimate of correctness; it *is* correctness, computed by a decision procedure with no learned parameters and no failure mode other than the proof genuinely being right or wrong. So the worst thing a bad round can do is fail to *add* proofs — it can never *poison* the corpus, because a wrong proof is rejected at the door rather than absorbed with a wrong label. That asymmetry — wrong outputs cost you opportunity but never correctness — is what lets you crank the loop for as many rounds as your compute allows without the usual collapse. The number eight million is not a delicate equilibrium; it is simply where the team chose to stop, with the difficulty frontier still advancing.

The same logic explains why the data engine and expert iteration are best run *interleaved* rather than sequentially. The statement-generation engine (autoformalize, filter, label) produces fresh targets; expert iteration produces proofs for the targets the current model can reach. Run them in lockstep and each strengthens the other: a stronger model from expert iteration can prove harder freshly-generated statements, and a broader pool of fresh statements keeps expert iteration from saturating on the targets it has already exhausted. The eight-million-proof corpus is the joint output of both loops turning together, not of a one-shot generation followed by a one-shot proving pass.

### Worked example: tracing a statement through three rounds

Take a moderately hard inequality, say a three-variable AM-GM bound. In round 0, the base model samples 64 candidate proofs; all 64 fail — the model does not yet know the right lemma chain. The statement is *undecided* and set aside. Round 0 does, however, close many easy two-variable bounds, whose proofs train round 1. In round 1, the strengthened model is sampled on the undecided pile again; now 2 of 64 candidates close the three-variable bound (it has learned the `nlinarith`-with-auxiliary-terms pattern from the round-0 proofs). Those 2 verified proofs enter the corpus. By round 2, with the three-variable proofs in its training set, the model closes the bound on roughly 15 of 64 samples and can attack four-variable generalizations. The statement migrated from undecided to easy across two retraining steps — no human wrote a single proof, and every proof in the corpus is kernel-certified.

### Second-order optimization: deduplication and the diversity tax

A failure mode lurks in naive expert iteration: the model converges on a *single* proof style and the corpus becomes redundant. If every kept proof uses the same `nlinarith` incantation, you have eight million near-duplicates and a model that knows exactly one tactic. The mitigation is to value *proof diversity* — keep multiple distinct verified proofs of the same statement when they take genuinely different routes, and weight sampling toward statements where the model's proofs are homogeneous. This diversity pressure is a recurring theme; it reappears, sharpened into an explicit exploration objective, in V1.5's RMaxTS. The lesson is that "keep everything the verifier accepts" is necessary but not sufficient — you also have to fight the entropy collapse that any self-training loop trends toward.

## 5. From V1 to V1.5: what the corpus unlocks

V1 ends with a 7B model and an eight-million-proof corpus, scoring **52.0% cumulatively** on miniF2F-test (and 46.3% with 64 samples) — more than double GPT-4's 23.0% on the same benchmark, and solving 5 of 148 problems on the FIMO olympiad benchmark where GPT-4 solved none. That is a strong result for a 7B model, and it is entirely attributable to the data engine. But V1's *inference* is comparatively blunt: sample many whole proofs, keep the first that verifies. V1.5 is the systematic upgrade of everything *after* the corpus exists — how you fine-tune on it, how you reinforce on the verifier signal, and how you search the proof space at test time.

The V1.5 pipeline has three stages, in order: continued pretraining on formal-language data, supervised fine-tuning, and reinforcement learning from proof-assistant feedback. The SFT stage trains on roughly **9,645k Lean 4 sequences** — a blend of Mathlib4's human-written proofs and the V1 synthetic corpus, augmented two ways that turn out to matter a great deal. First, natural-language **chain-of-thought comments** are interleaved into the proofs, so the model learns to articulate the informal reasoning that motivates each tactic before emitting it. Second, **tactic-state comments** are embedded inline — after each tactic, a comment records the resulting proof state (the remaining goals). This second augmentation is the seed of the truncate-and-resume mechanism we will reach shortly: the model is trained to read and write tactic states, which is exactly what it needs to do to resume a proof from the middle.

Both augmentations attack the same weakness in whole-proof generation: a raw Lean proof is a sequence of tactics with no visible intermediate reasoning, so a model trained on raw proofs is guessing tactic-by-tactic with no scratchpad. By interleaving the *why* (CoT) and the *what-now* (tactic state), the SFT data teaches the model to plan and to track state — the two things a long proof needs.

To make the augmentation concrete, a single training example after augmentation looks roughly like this — the natural-language motivation as a comment, then the tactic, then the resulting goal state as a comment, repeated down the proof:

```lean
theorem amgm_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    a ^ 2 + b ^ 2 ≥ 2 * a * b := by
  -- The gap a^2 + b^2 - 2ab is a perfect square (a - b)^2 ≥ 0,
  -- so nlinarith should close it once we hand it that square.
  nlinarith [sq_nonneg (a - b), sq_nonneg (a + b), mul_pos ha hb]
  -- goals now: (no goals)  -- proof complete
```

The model is trained to produce the `-- The gap ...` reasoning *before* committing to `nlinarith`, and to expect the `-- goals now:` annotation as a checkpoint. Two things fall out of this. The CoT comments make the model a better *planner* — it reasons about the shape of the proof in natural language, which is the modality DeepSeekMath-Base is strongest in, before translating that plan into tactics. The tactic-state comments make the model a fluent *reader and writer of proof states*, which is the precise skill the truncate-and-resume loop will demand at inference time, when it hands the model a half-finished proof and a `-- goals now:` annotation and asks it to continue. The SFT augmentation and the inference mechanism are co-designed; neither makes sense without the other.

## 6. RLPAF: reinforcement learning from proof-assistant feedback

**Senior rule of thumb: when your reward is a verifier, you do not need a reward model — you need an RL algorithm that tolerates a reward of exactly 0 or 1 and never needs a value network.**

The reinforcement-learning stage of V1.5 is named **RLPAF** — Reinforcement Learning from Proof Assistant Feedback. The name is the whole idea: the reward is the proof assistant. There is no learned reward model, no human preference data, no Bradley-Terry ranking. Lean either accepts the generated proof or it does not, and that binary outcome *is* the reward.

![RLPAF runs GRPO with a binary Lean reward over ~4.5k statements where the SFT model already scores moderate pass rates, then the updated policy resamples the next batch](/imgs/blogs/deepseek-prover-synthetic-data-rmaxts-5.webp)

The figure walks the loop. The SFT policy samples a *group* of candidate proofs for a statement. Each candidate is verified by Lean 4. The reward is binary: $r = 1$ if Lean accepts the proof, $r = 0$ otherwise. Because the reward is group-relative, the algorithm of choice is **GRPO** — Group Relative Policy Optimization, the value-network-free RL method that the [GRPO vs DPO vs PPO decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) covers in depth, and which originated in the DeepSeekMath work. GRPO computes the advantage of each sampled proof as its reward's z-score *within the group* — subtract the group mean, divide by the group standard deviation — and pushes the policy toward the above-average proofs and away from the below-average ones. No critic, no value function to fit, just the group statistics.

The binary reward and the group-relative advantage compose beautifully here. Write the group of $G$ sampled proofs for a statement as having rewards $r_1, \dots, r_G \in \{0, 1\}$. The advantage of proof $i$ is

$$A_i = \frac{r_i - \mu}{\sigma}, \quad \mu = \frac{1}{G}\sum_j r_j, \quad \sigma = \sqrt{\frac{1}{G}\sum_j (r_j - \mu)^2}.$$

When some proofs in the group verify and some do not — a *mixed* group — $\mu$ is strictly between 0 and 1, $\sigma > 0$, and the verified proofs get positive advantage while the failures get negative advantage. That is a clean, informative gradient. The selection of training statements is engineered to keep groups mixed: RLPAF runs on about **4,500 statements** chosen specifically because the SFT model already scores *moderate* pass rates on them — not 0% (where every proof fails, $\sigma = 0$, and the gradient vanishes) and not 100% (where every proof succeeds, $\sigma = 0$ again, and there is nothing to learn). The sweet spot is statements the model solves *sometimes*, because only there does the group carry signal.

```python
def grpo_advantages(rewards):
    """Group-relative advantages for a binary Lean reward.

    rewards : list of 0/1 verifier outcomes for one statement's group.
    Returns per-sample advantages; all-pass or all-fail groups give
    zero advantage (no learning signal) and are skipped upstream.
    """
    G = len(rewards)
    mu = sum(rewards) / G
    var = sum((r - mu) ** 2 for r in rewards) / G
    sigma = var ** 0.5
    if sigma == 0.0:
        return [0.0] * G          # degenerate group: no gradient
    return [(r - mu) / sigma for r in rewards]


def select_rl_statements(model, statements, n=4500, lo=0.1, hi=0.9):
    """Keep statements with moderate pass rates so groups stay mixed."""
    kept = []
    for s in statements:
        proofs = model.sample(s, k=16)
        pass_rate = sum(kernel_verify(s, p) for p in proofs) / len(proofs)
        if lo <= pass_rate <= hi:
            kept.append(s)
    return kept[:n]
```

The reason this works and PPO-style RLHF would be overkill is that the verifier removes every source of reward noise that makes a value network worth fitting. In RLHF the reward is a learned, noisy approximation of human preference, so you want a critic to reduce variance. Here the reward is *exact* — there is nothing to approximate and nothing to denoise — so the value network buys you nothing and costs you a second model's worth of memory and a second loss to balance. GRPO's group-baseline is exactly the right amount of machinery: enough to get a per-statement baseline for free, not so much that you are fitting a critic against a signal that is already perfect.

### Second-order optimization: the pass-rate window is a moving target

The 4,500-statement window is not static. As RLPAF trains, the model's pass rates on those statements climb — the moderate ones drift toward "always solved," at which point their groups stop being mixed and their gradient dies. So the curriculum has to *refresh*: periodically re-measure pass rates and swap in statements that have entered the moderate band from below (formerly-too-hard statements the strengthened model now solves *sometimes*). This is the same difficulty-frontier dynamic as expert iteration, applied to the RL curriculum: keep the training set parked at the model's current frontier of partial competence, because that is the only place the binary reward produces a usable advantage.

## 7. Whole-proof generation with truncate-and-resume

**Senior rule of thumb: a failed whole-proof attempt is not garbage — the prefix that verified before the error is exactly the scratch state a smarter resume should start from.**

There are two classical paradigms for neural proving. *Proof-step* generation (the tree-search style) predicts one tactic at a time, queries the kernel after each, and backtracks; it is precise but slow and search-heavy. *Whole-proof* generation emits the entire proof in one shot and checks it at the end; it is fast and lets the model plan globally, but a single error anywhere voids the whole attempt and throws away all the correct work that preceded it. V1.5 unifies the two with a mechanism called **truncate-and-resume**.

![Truncate-and-resume keeps the verified prefix and the tactic state when a whole-proof attempt errors, then resumes generation from that point instead of discarding the attempt](/imgs/blogs/deepseek-prover-synthetic-data-rmaxts-6.webp)

The figure contrasts the two regimes. On the left, a whole-proof attempt with no resume: the model emits a full proof block, Lean errors at tactic 7 of 12, and the entire attempt — including the six tactics that verified fine — is discarded. On the right, truncate-and-resume: the model emits the same full block, but when Lean reports the first error at tactic 7, the engine *truncates* the proof there, keeps the verified prefix (tactics 1 through 6), appends the *tactic state* at that point as a comment, and resumes generation from the prefix. The model now continues a proof that is already six tactics deep and correct, with the current goals written out for it, instead of starting from scratch.

```python
def truncate_and_resume(statement, model, kernel, max_rounds=8):
    """Whole-proof generation that recovers from the first error.

    Generate a full proof; if Lean reports an error, keep the verified
    prefix, annotate it with the tactic state, and resume from there.
    """
    prefix = ""                       # accumulated verified tactics
    for _ in range(max_rounds):
        proof = model.generate(statement, prefix=prefix)
        result = kernel.check(statement, prefix + proof)
        if result.ok:
            return prefix + proof     # fully verified

        # Truncate at the FIRST error; everything before it verified.
        good = proof[: result.first_error_offset]
        prefix = prefix + good

        # Append the tactic state at the truncation point as a comment,
        # so the resumed generation sees the remaining goals explicitly.
        state = kernel.tactic_state(statement, prefix)
        prefix = prefix + f"\n  -- goals now:\n  -- {state}\n"

    return None                       # gave up after max_rounds
```

This is where the SFT tactic-state augmentation pays off. The model was trained on proofs with inline tactic-state comments, so a resumed prompt that ends with `-- goals now: ...` is *in distribution* — the model has seen thousands of examples of "here is the state, here is the next tactic," and it knows how to continue. Without that augmentation, the resume prompt would be a strange artifact the model never saw in training and it would flail. The two pieces — the data augmentation and the inference mechanism — are designed as a unit.

Truncate-and-resume is the bridge between the two paradigms. Like whole-proof generation, each generation step plans a long stretch of proof globally rather than greedily picking one tactic. Like proof-step generation, it incorporates kernel feedback mid-proof and never discards verified progress. You get the planning horizon of the first and the error-recovery of the second, and the unification is what lets the same model serve as the rollout policy inside a tree search — which is the next and final piece.

### Worked example: a proof that needs two resumes

Take a number-theory goal where the model's first attempt sets up the right induction but botches the inductive step's arithmetic at tactic 9 of 14. Truncate at 9: tactics 1-8 (the induction setup and base case) are kept, the tactic state shows the remaining inductive-step goal, and the model resumes. Its second attempt closes the arithmetic but then over-reaches with a `simp` that fails at the new tactic 4. Truncate again: keep the corrected arithmetic, show the state, resume once more. The third attempt closes the goal. A whole-proof-only engine would have needed all 14 tactics correct in a single sample — exponentially unlikely — whereas truncate-and-resume converted one hard 14-tactic proof into three short, in-distribution continuations. This is why the mechanism raises pass rates so much on long proofs: it factorizes a low-probability joint event into a product of higher-probability conditional ones.

## 8. RMaxTS: exploration under a sparse proof reward

**Senior rule of thumb: when the reward you care about is almost always zero, stop optimizing it directly and start rewarding the discovery of states you have never seen — novelty is the only dense signal in a sparse-reward world.**

We arrive at the headline contribution. Test-time proof search is a tree: the root is the initial goal, each edge is a tactic, each node is the resulting proof state, and a leaf is either a closed proof (success) or a dead end (failure). Monte-Carlo tree search (MCTS) is the natural fit — but vanilla MCTS assumes a reward signal that, while perhaps sparse, eventually shows up often enough to guide the tree's growth. In proof search, the *only* extrinsic reward is "the proof is complete," and on a hard statement that event might occur in one rollout out of thousands, or never within budget. A search guided solely by that reward is a search guided by almost nothing — it has no gradient to climb until the very instant it stumbles onto a full proof, by which point the guidance is moot.

**RMaxTS** — the RMax-flavored tree search V1.5 introduces — solves this by replacing the absent extrinsic signal with a dense *intrinsic* one borrowed from the RMax family of exploration algorithms. The intrinsic reward is brutally simple: a rollout earns reward 1 if it adds *at least one new node* to the search tree (a tactic state never seen before), and 0 if every node it visited was already in the tree.

![RMaxTS rewards a rollout with intrinsic value one whenever it adds a newly discovered tactic-state node, driving diverse exploration when the proof-complete extrinsic reward is almost always zero](/imgs/blogs/deepseek-prover-synthetic-data-rmaxts-7.webp)

The figure shows a slice of the tree with this reward in action. The root and the already-explored states (`intro h`, `rw lemma`, `apply le`) carry $R_{\text{int}} = 0$ — they are seen, so revisiting them earns nothing. The newly discovered states (`nlinarith`, `simp`, `cases h`) carry $R_{\text{int}} = 1$ — they expand the frontier, so they are rewarded. And the one node that actually closes the proof (`QED`) carries the real extrinsic reward $R_{\text{ext}} = 1$. The intrinsic reward turns "explore broadly" into a quantity the search can optimize *before* any proof closes, which is exactly the period when vanilla MCTS is flying blind.

```python
def rmax_intrinsic_reward(rollout_states, tree):
    """RMax-style intrinsic reward for one MCTS rollout.

    Reward 1 if the rollout discovered at least one tactic state not
    already in the search tree; 0 if everything it touched was old.
    This is the ONLY dense signal until a proof actually completes.
    """
    discovered_new = False
    for state in rollout_states:
        key = state.canonical_hash()      # dedup equivalent tactic states
        if key not in tree.nodes:
            tree.add_node(state)
            discovered_new = True
    return 1.0 if discovered_new else 0.0
```

The selection rule that consumes this reward is **discounted UCB (DUCB)** with discount factor $\gamma = 0.99$. Standard UCB1 weights every past visit equally, which is wrong in a search where the policy itself is improving and early rollouts are stale. DUCB exponentially down-weights old visits, so a node's value estimate reflects *recent* exploration outcomes more than ancient ones. The selection score for a child node $c$ of parent $p$ is

$$\text{DUCB}(c) = \hat{Q}_\gamma(c) + C \sqrt{\frac{\log N_\gamma(p)}{N_\gamma(c)}},$$

where $\hat{Q}_\gamma$ and $N_\gamma$ are the discounted value estimate and discounted visit count (each past observation weighted by $\gamma^{\Delta t}$ for its age $\Delta t$), and $C$ is the exploration constant. The discounting and the intrinsic reward work in concert: the intrinsic reward makes *novelty* valuable, and the discounting makes *recent* novelty more valuable than stale novelty, so the search keeps pushing into genuinely unexplored regions rather than re-litigating branches it has already mined out.

### One full RMaxTS rollout, phase by phase

It helps to walk a single rollout end to end, because the four classical MCTS phases each take on a specific shape when the reward is intrinsic novelty rather than a game outcome.

**Selection.** Starting at the root, descend the tree by repeatedly picking the child with the highest DUCB score until you reach a node that has unexpanded actions (tactics not yet tried). The DUCB score blends the node's discounted value estimate — which, because the reward is novelty, is high for nodes whose subtrees have recently been productive sources of new states — with the exploration bonus that favors under-visited children. The discount $\gamma = 0.99$ means a child that was novel-rich a thousand rollouts ago but has since been mined out will have its value estimate decay, so selection naturally migrates away from exhausted regions.

**Expansion.** At the selected node, the policy (the V1.5 model, in whole-proof or truncate-and-resume mode) generates one or more candidate continuations. Each continuation is a sequence of tactics; running it forward through the kernel produces a sequence of new proof states. This is where novelty is measured: each resulting state is canonicalized and looked up in the tree's node set.

**Simulation / evaluation.** There is no random playout to a terminal state as in game MCTS — the "evaluation" of a rollout *is* the novelty computation. If the expansion produced at least one state not already in the tree, the rollout's intrinsic reward is 1; otherwise 0. If, additionally, one of the new states closes the proof, the extrinsic reward fires and the search can terminate with a verified proof.

**Backpropagation.** Propagate the rollout's reward up the path from the expanded node to the root, updating each ancestor's discounted value estimate and visit count. Because the reward is novelty, what gets reinforced up the tree is "this path leads to regions where new states are still being discovered" — exactly the credit signal you want when the real goal is coverage.

```python
def rmaxts_rollout(root, model, kernel, tree, gamma=0.99, C=1.0):
    """One RMaxTS rollout: select, expand, evaluate via novelty, backprop."""
    path, node = [], root
    # SELECTION: descend by discounted UCB until an expandable node.
    while node.fully_expanded() and not node.is_terminal():
        node = max(node.children, key=lambda c: ducb_score(c, gamma, C))
        path.append(node)

    # EXPANSION: sample a continuation; run it through the kernel.
    tactics = model.generate_continuation(node.state)
    new_states = kernel.apply(node.state, tactics)

    # EVALUATION: intrinsic reward = did we discover any new state?
    r_int = rmax_intrinsic_reward(new_states, tree)
    r_ext = 1.0 if any(s.is_closed for s in new_states) else 0.0

    # BACKPROPAGATION: discounted value + visit updates up the path.
    for anc in reversed([root] + path):
        anc.visits = gamma * anc.visits + 1
        anc.value = gamma * anc.value + (r_int + r_ext)
    return r_ext  # nonzero means a verified proof was found
```

The elegance is that the same machinery a game-playing MCTS uses for "did I win" is repurposed for "did I learn something," and that single substitution is what makes the search trainable in a regime where wins are vanishingly rare. The extrinsic reward is still there — it is what you ultimately want — but it is a bonus that occasionally fires on top of the intrinsic reward that does the steering ninety-nine point nine percent of the time.

### Why novelty is the right proxy

The deep reason RMaxTS works is that in formal proof search, *coverage of the tactic-state space is a near-perfect proxy for the probability of eventually closing the proof*. A proof exists somewhere in the tree; you do not know where; the more distinct states you visit, the higher your odds of hitting the one that admits a closing tactic. Rewarding novelty is rewarding the thing that mechanically increases your hit probability, even though you cannot see the target. Contrast this with a dense *heuristic* reward (say, "fewer remaining goals is better") — heuristics like that are seductive but they bias the search toward greedy goal-reduction and away from the setup tactics (introduce an auxiliary variable, generalize the goal) that *temporarily increase* complexity on the way to a proof. Novelty has no such bias: it rewards breadth without prejudging which states are "progress," which is exactly right when your model of progress is unreliable.

### The parallel architecture

RMaxTS is not a single tree ground out sequentially — it is a heavily parallel system, which is what makes it practical at the scale V1.5 reports. The deployment runs **256 MCTS runners**, each maintaining its own search tree for a statement, with **32 workers per tree** handling the two expensive operations: generating the next tactic (a model forward pass) and verifying it (a Lean kernel call). Both operations are run *asynchronously* — generation and verification are decoupled so that the GPU producing tactics and the CPU verifying them never block each other. A worker that is waiting on the kernel does not stall the generator; a generation that is mid-flight does not stall a verification. This async generate-and-verify pipeline is the difference between a tree search that spends most of its wall-clock idle (waiting on the kernel) and one that keeps both the accelerator and the verifier saturated.

The numbers that come out of this machine: V1.5 reaches **63.5%** on miniF2F-test with RMaxTS over a CoT/non-CoT mix (and **60.2%** single-pass), and **25.3%** on ProofNet-test. Those are large jumps over V1's 52.0% on miniF2F, and the bulk of the gain is search — the same trained model, given RMaxTS instead of naive whole-proof sampling, finds proofs it would otherwise have missed because the intrinsic-reward exploration drives it into branches that flat sampling never reaches.

![Across miniF2F-test, FIMO, and ProofNet-test the DeepSeek-Prover line decisively beats GPT-4 and prior whole-proof methods, with V1.5 plus RMaxTS setting the high mark](/imgs/blogs/deepseek-prover-synthetic-data-rmaxts-8.webp)

The results matrix above is the scoreboard for the whole effort. GPT-4 sits at 23.0% on miniF2F and 0 of 148 on FIMO — a capable general model with no formal-proving machinery. V1, the data engine alone with naive sampling, more than doubles that to 52.0% on miniF2F and 5 of 148 on FIMO. V1.5, adding RLPAF and RMaxTS on top of the same data foundation, reaches 63.5% on miniF2F and opens a strong 25.3% on ProofNet-test, a harder benchmark V1 did not report. Each column is a different facet — competition problems, olympiad problems, undergraduate-curriculum problems — and the Prover line leads on all of them. The progression reads cleanly: data engine (V1) doubles the baseline, then learning-plus-search (V1.5) adds another large increment on top.

## Case studies from production

The following eleven vignettes are the kind of thing you actually hit when you stand a Prover-style pipeline up. Each is the symptom, the wrong first hypothesis, the real root cause, the fix, and the lesson.

### 1. The autoformalizer that loved vacuous truths

**Symptom:** an early synthetic corpus produced a model with suspiciously high training pass rates that cratered on held-out miniF2F. The training proofs verified; the model still failed real problems. **Wrong first hypothesis:** overfitting — shrink the model or add dropout. **Root cause:** a large fraction of the autoformalized statements had inconsistent hypotheses. The model had learned to detect contradictions and exploit them, producing "proofs" that were valid Lean but pedagogically poison — they taught contradiction-exploitation, not reasoning. **Fix:** add hypothesis rejection (prove $\Gamma \vdash \texttt{False}$; discard on success) as a hard filter before any statement earns a proof search. Roughly 15-20% of statements were culled, training pass rates dropped to honest levels, and held-out performance climbed. **Lesson:** a high pass rate on a corpus you did not consistency-check is a measurement of the corpus's rot, not the model's skill.

### 2. The single-direction search that burned a weekend

**Symptom:** a prove-only pipeline ran for 60 GPU-hours over a batch of autoformalized statements and emitted far fewer proofs than the statement count justified, with no diagnostic for the gap. **Wrong first hypothesis:** the model is too weak; train longer. **Root cause:** a meaningful slice of the statements were *false* (mis-formalized universals, flipped inequalities). The prove-only search spent its full budget failing on them silently. **Fix:** dual-path search — race a proof of $\lnot P$ alongside the proof of $P$. The false statements were *disproved* quickly (counterexamples are short) and *labeled* rather than silently failed; the corpus stopped including limbo statements, and the wasted budget on false targets dropped to near zero. **Lesson:** in a pipeline that synthesizes its own targets, "failed to prove" is ambiguous between "hard" and "false," and you must disambiguate or you will throw compute at impossibilities.

### 3. The expert-iteration loop that flatlined

**Symptom:** after three expert-iteration rounds, the corpus stopped growing and benchmark numbers plateaued, even though plenty of statements remained undecided. **Wrong first hypothesis:** the model has saturated; it cannot get better. **Root cause:** sampling was too low-temperature and too few-shot per statement, so each round re-sampled essentially the same proof attempts and discovered nothing new on the hard pile. **Fix:** raise sampling temperature and $K$ on the undecided statements specifically, and add proof-diversity pressure so the corpus did not collapse onto one tactic style. The frontier started advancing again. **Lesson:** expert iteration only marches the difficulty frontier outward if each round actually *explores*; deterministic re-sampling of the same attempts is a loop that spins without moving.

### 4. The all-pass RL group with a dead gradient

**Symptom:** RLPAF training loss looked active but the policy was not improving on the held-out set; many statements contributed exactly zero gradient. **Wrong first hypothesis:** the learning rate is too low. **Root cause:** the statement set included many the SFT model already solved on *every* sample. Those groups had $\sigma = 0$, so every advantage was 0 — no signal, just wasted forward passes. **Fix:** the moderate-pass-rate selection window (keep statements at roughly 10-90% pass rate) and a periodic refresh that swaps saturated statements out. Mixed groups returned, gradients came back, the policy moved. **Lesson:** group-relative RL with a binary reward learns *only* from mixed groups; an all-pass or all-fail group is compute with no information, and your training set must be curated to stay mixed.

### 5. The whole-proof model that kept rediscovering the same prefix

**Symptom:** a whole-proof-only model achieved decent pass-1 but its pass-64 barely improved — 64 samples were not 64 *different* attempts. **Wrong first hypothesis:** the model is low-diversity; raise temperature. **Root cause:** raising temperature helped the early tokens but the model still re-derived the same correct prefix and re-failed at the same hard step, sample after sample, because each sample restarted from nothing. **Fix:** truncate-and-resume. Once the correct prefix was verified and pinned, with the tactic state shown, the 64 samples diverged at the *hard* step instead of wastefully re-deriving the easy prefix, and pass-64 jumped. **Lesson:** sampling diversity is only valuable where the proof is actually uncertain; pinning the verified prefix concentrates your sample budget on the step that matters.

### 6. The MCTS that re-litigated a solved subtree

**Symptom:** an early tree search spent enormous budget re-expanding branches it had already explored, and coverage of new tactic states grew sub-linearly with rollouts. **Wrong first hypothesis:** the UCB exploration constant is too low; raise $C$. **Root cause:** standard UCB1 weighted ancient visits equally with recent ones, so once a subtree's visit count was high it kept getting selected even though it had been mined out, while the improving policy's newer, more promising branches were starved. **Fix:** discounted UCB ($\gamma = 0.99$), which exponentially decays stale visit counts so the search follows the *current* policy's frontier. Coverage growth went back to roughly linear in rollouts. **Lesson:** when the rollout policy is itself improving during the search, equal-weight UCB is mis-specified; discount the past so selection tracks present competence.

### 7. The intrinsic reward that got hacked by trivial branching

**Symptom:** after adding a novelty reward, the search exploded the tree breadth-first into millions of shallow, useless states and never went deep enough to close proofs. **Wrong first hypothesis:** novelty is the wrong objective; revert it. **Root cause:** the novelty reward was being earned by trivial tactic variations that produced technically-distinct-but-equivalent states (reordering commutative steps, cosmetic rewrites). The search maximized novelty by generating cheap fake-new states. **Fix:** canonicalize tactic states before the novelty check (hash the normalized goal, not the raw syntax) so equivalent states collapse to one node and earn novelty only once. The breadth explosion stopped and the search balanced depth against breadth. **Lesson:** any novelty-based intrinsic reward is only as good as your notion of "same state"; without canonicalization, the agent will farm novelty from cosmetic differences.

### 8. The verification queue that starved the GPU

**Symptom:** a parallel proof-search deployment showed 30% GPU utilization despite a large batch — the accelerator was idle most of the time. **Wrong first hypothesis:** the batch size is too small; raise it. **Root cause:** generation and verification were synchronous — every generated tactic blocked on a Lean kernel call before the next generation could start, and the kernel (CPU-bound) was the bottleneck, leaving the GPU waiting. **Fix:** decouple them into an async pipeline — generate ahead into a queue while a pool of verification workers drains it — matching the 256-runner / 32-worker-per-tree structure. GPU utilization rose past 80%. **Lesson:** in a generate-then-verify loop, the kernel is a CPU bottleneck that will idle your accelerator unless you make the two stages asynchronous and independently scaled.

### 9. The benchmark number that did not reproduce

**Symptom:** a re-run of the miniF2F evaluation came back several points below the reported figure, prompting suspicion of a regression. **Wrong first hypothesis:** the checkpoint is wrong or corrupted. **Root cause:** the reported 63.5% was a *cumulative* number over a CoT/non-CoT mix and a specified search budget (RMaxTS rollouts), while the re-run used single-pass (60.2%-style) sampling with a smaller budget. The two numbers measure different protocols. **Fix:** pin the exact evaluation protocol — sampling mode, CoT/non-CoT mix, search budget, pass@k convention — alongside every reported number. The re-run matched once the protocol was aligned. **Lesson:** a theorem-proving score is meaningless without its protocol; "63.5% on miniF2F" and "60.2% on miniF2F" can be the same model differing only in how hard you let it search.

### 10. The CoT mix that helped one benchmark and hurt another

**Symptom:** turning on chain-of-thought generation lifted miniF2F but slightly depressed scores on a different problem distribution, and the team nearly disabled CoT globally. **Wrong first hypothesis:** CoT is net-negative; remove it. **Root cause:** CoT helps when the proof's hard part is *planning* (which lemma to invoke, how to set up an induction) and is neutral-to-harmful when the hard part is purely *mechanical* search that the planning prose only slows down with token overhead. The two benchmarks had different proof-shape distributions. **Fix:** run a *mix* of CoT and non-CoT rollouts and take the union of what each closes — exactly the CoT/non-CoT mix behind the reported 63.5%. The mix dominated either pure mode because the two modes close partially disjoint sets of problems. **Lesson:** CoT is not a global on/off switch; it is a rollout *flavor*, and the cumulative-over-flavors evaluation captures value that any single flavor leaves on the table.

### 11. The Mathlib version bump that broke ten thousand proofs

**Symptom:** after upgrading the Lean toolchain and Mathlib, a large slice of the previously-verified corpus stopped type-checking, and the next training run silently shrank. **Wrong first hypothesis:** the proofs were never valid and a bug was hiding them. **Root cause:** Mathlib is a moving target — lemma names get renamed, signatures change, tactics get deprecated — so a proof verified against Mathlib commit A can fail against commit B even though the mathematics is unchanged. The corpus was pinned to no specific Mathlib version. **Fix:** pin the exact Lean toolchain and Mathlib commit per corpus snapshot, re-verify on upgrade, and treat the verifier environment as a versioned dependency rather than ambient infrastructure. **Lesson:** "kernel-verified" is only meaningful relative to a *specific* kernel-plus-library version; a synthetic formal corpus is only as reproducible as the toolchain you pinned when you built it.

## A consolidated view: which mechanism solves which failure

It is worth collecting the mechanisms against the failures they address, because the value of V1 and V1.5 is precisely that each piece is a targeted answer to a specific way the naive approach breaks.

| Failure of the naive approach | V1 / V1.5 mechanism | What it guarantees |
| --- | --- | --- |
| Web problems are natural language, not Lean | Autoformalization from DeepSeekMath-Base | Candidate formal statements at web scale |
| Autoformalizer emits bad translations | Model-based quality scoring (excellent..poor) | 869,659 to 712,073 kept; obvious dreck culled cheaply |
| Statements with inconsistent hypotheses | Hypothesis rejection (prove $\Gamma \vdash \texttt{False}$) | Every kept statement is consistency-checked by the kernel |
| Well-formed but false statements | Dual-path search (race $P$ and $\lnot P$) | False statements are disproved and labeled, not silently failed |
| Where do millions of proofs come from | Expert iteration (sample, verify, retrain) | Corpus grows to ~8M with zero unverified entries |
| Whole-proof attempts discard partial progress | Truncate-and-resume | Verified prefixes are pinned; sampling concentrates on hard steps |
| Reward is sparse (proof-complete is rare) | RMaxTS intrinsic reward (novelty) | A dense exploration signal before any proof closes |
| Stale visits mislead an improving policy | Discounted UCB ($\gamma = 0.99$) | Selection tracks the current policy's frontier |
| RL reward is noisy and needs a critic | RLPAF / GRPO with binary Lean reward | Exact reward, no value network, group baseline only |
| Kernel CPU bottleneck idles the GPU | Async 256-runner / 32-worker parallelism | Generation and verification saturate independently |

Read top to bottom, the table is the architecture: a data engine (rows 1-5) that manufactures a verified corpus, and a learning-plus-search stack (rows 6-10) that extracts maximum proving power from it. Neither half works without the other — the search has nothing to search a good policy over without the corpus, and the corpus is just a static dataset without a search that can exploit the policy it trained.

## When to reach for this approach, and when not to

### Reach for the V1/V1.5 pattern when

- **Your task has an exact, cheap, automated verifier.** Formal proofs (Lean, Coq, Isabelle), but also code that must pass a test suite, SQL that must return a specified result, or any domain where correctness is a kernel call rather than a human judgment. The verifier is what makes self-training safe and makes the binary RL reward exact.
- **Your supply of human-labeled examples is small relative to your model's appetite.** If you have hundreds of thousands of examples and want millions, autoformalization plus expert iteration is the way to manufacture the rest without model collapse — because the verifier gates every synthetic example.
- **Your reward is sparse at test time.** If success is a rare event in a large search tree, an intrinsic novelty reward (RMaxTS-style) gives the search a dense signal to climb when the real reward is silent. This generalizes well beyond proofs to any sparse-reward search where state-coverage proxies for success probability.
- **You can afford a parallel generate-and-verify deployment.** The wins from tree search are real but they cost compute; the async runner/worker architecture is what makes that compute efficient rather than idle.

### Skip it when

- **Your verifier is approximate or learned.** If "correct" is a reward model's opinion rather than a kernel's verdict, the safety of the self-training loop evaporates — the model will reward-hack the approximation, and you are back to needing a critic, preference data, and the full RLHF apparatus. The entire elegance here depends on an *exact* verifier.
- **Your problems are dense-reward already.** If a usable reward shows up on most rollouts (numeric-answer math, most RLVR settings), vanilla GRPO or even simple best-of-$n$ sampling is enough; the RMaxTS intrinsic-reward machinery is solving a sparsity problem you do not have, and it adds complexity for no gain.
- **You cannot autoformalize your domain.** The whole data engine presupposes that natural-language problems can be translated into your formal language well enough that a filter can salvage the good translations. In domains with no formal target language, or where translation is hopeless, there is no engine to run.
- **You need the answer, not the derivation.** If the deliverable is a number and not a proof, you do not need any of this — outcome-checking is enough. This pattern earns its complexity specifically when the *derivation itself* is the product and must be certified step by step.

The throughline across all of it is a single discipline: never put anything into the loop that the verifier did not certify. That one rule is what lets V1 self-train to eight million proofs without rotting, lets V1.5's RL use a reward with no critic, and lets RMaxTS explore aggressively without fear of training on hallucinated success. The verifier is not a component of the system; it is the foundation the whole system is allowed to be aggressive on top of. Everything covered here — the filters, the dual-path race, the expert-iteration bootstrap, the RLPAF reward, the truncate-and-resume recovery, and the RMaxTS exploration — is downstream of that one decision to make a proof assistant, not a learned model, the arbiter of truth.

## Further reading

- **DeepSeek-Prover V1**, arXiv [2405.14333](https://arxiv.org/abs/2405.14333) — the data engine: autoformalization, two-stage filtering, dual-path search, expert iteration to ~8M proofs.
- **DeepSeek-Prover-V1.5**, arXiv [2408.08152](https://arxiv.org/abs/2408.08152) (ICLR 2025) — RLPAF, truncate-and-resume whole-proof generation, and RMaxTS.
- The [DeepSeek-Prover V2 follow-up](/blog/paper-reading/large-language-model/deepseek-prover-v2-advancing-formal-mathematical-reasoning-via-reinforcement-learning-for-subgoal-decomposition) — subgoal decomposition built on the V1/V1.5 substrate covered here.
- [GRPO vs DPO vs PPO: a decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) — why RLPAF picks GRPO and what a binary verifier reward buys you over a learned reward model.
- [DeepSeekMath data pipeline and GRPO origin](/blog/machine-learning/large-language-model/deepseekmath-data-pipeline-and-grpo-origin) — the base model and the algorithm Prover inherits.
- [Self-verifiable reasoning in DeepSeekMath-V2](/blog/machine-learning/large-language-model/self-verifiable-reasoning-deepseekmath-v2) — the complementary problem of grading *informal* proofs when there is no kernel to lean on.

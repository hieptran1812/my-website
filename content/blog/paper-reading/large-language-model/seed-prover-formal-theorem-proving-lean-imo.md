---
title: "Seed-Prover: deep-and-broad reasoning that formally solved 5 of 6 IMO 2025 problems"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A principal-engineer deep-dive into Seed-Prover's lemma-style whole-proof reasoning, its Lean 4 verification loop, the deep-vs-broad test-time strategies, Seed-Geometry, and the IMO 2025 case study where it produced machine-checked proofs for 5 of 6 problems."
tags: ["seed-prover", "formal-theorem-proving", "lean", "automated-reasoning", "reinforcement-learning", "imo-2025", "math-reasoning", "seed-geometry", "bytedance-seed", "lemma-decomposition"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 68
---

In July 2025, a system from ByteDance Seed did something that no large language model had been able to do before: it produced **Lean 4 proofs that the kernel actually accepted** for five of the six International Mathematical Olympiad 2025 problems. Not "the model wrote a convincing-looking argument and a human graded it generously." Not "the model matched a reference answer string." A 4000-line formal proof of Problem 4 that compiled, line by line, against the Lean type-checker. A 2000-line proof of Problem 3. And, almost insultingly, a geometry engine that disposed of Problem 2 in **about two seconds** once the figure was formalized.

If you have spent any time around LLMs doing math, your instinct should be deep suspicion. We have all watched a chatbot confidently "prove" a false statement, hallucinate a citation to a nonexistent theorem, or quietly assume the thing it was supposed to derive. The entire reason formal theorem proving matters is that it removes the human-trust bottleneck: a Lean proof is correct *because the kernel says so*, and the kernel does not care how persuasive the prose was. Seed-Prover is the first system to combine the fluency of a long-chain-of-thought reasoning model with the brutal honesty of a proof assistant — and to do it at a level where it is competitive with, and on several axes ahead of, human olympiad medalists.

This post is a deep-dive on three connected artifacts: **Seed-Prover** (arXiv:2507.23726), the lemma-style whole-proof reasoning model; **Seed-Geometry**, its companion symbolic engine for olympiad geometry; and **Seed-Prover 1.5** (arXiv:2512.17260), the agentic follow-up that pushed PutnamBench to ~88%. I want to be precise about what is genuinely new here, where the numbers come from, and — because I have shipped systems that lean on formal verification — where this approach helps and where it is still a research toy.

## Why formal proving is a different game

Let me start with the mismatch that makes this whole field interesting, because it is the thing most people get wrong.

The diagram below is the mental model for the rest of the article: a closed loop where a reasoning model proposes whole proofs, a formal verifier (the Lean kernel) returns *ground-truth* pass/fail signal, and the system either refines, decomposes into lemmas, or summarizes what it learned and tries again. Everything Seed-Prover does is an elaboration of this loop.

<!-- FIGSPEC 1
kind: graph
claim: Seed-Prover closes a loop where a reasoning model proposes whole proofs, the Lean 4 kernel returns ground-truth pass/fail, and the system refines, decomposes into lemmas, or self-summarizes before retrying.
caption: The deep-and-broad loop: conjecture in, machine-checked proof out, with Lean as the only source of truth.
nodes:
  - id: conj | label: "Conjecture / theorem (Lean 4 statement)" | color: gray
  - id: gen | label: "Whole-proof gen (lemma-style CoT)" | color: blue
  - id: lemA | label: "Lemma A draft" | color: blue
  - id: lemB | label: "Lemma B draft" | color: blue
  - id: lemC | label: "Main proof skeleton" | color: blue
  - id: lean | label: "Lean 4 kernel verify (pass/fail + errors)" | color: lavender
  - id: refine | label: "Refine on errors (medium loop)" | color: amber
  - id: summ | label: "Self-summarize proven lemmas" | color: green
  - id: done | label: "Verified proof (kernel-checked)" | color: green
edges:
  - conj -> gen
  - gen -> lemA
  - gen -> lemB
  - gen -> lemC
  - lemA -> lean
  - lemB -> lean
  - lemC -> lean
  - lean -> refine | label: "fail"
  - refine -> gen | label: "retry"
  - lean -> summ | label: "pass"
  - summ -> gen | label: "reuse lemma"
  - lean -> done | label: "all pass"
notes: vertical flow top-to-bottom; middle layer (lemA/lemB/lemC) is 3 wide so it is not wide-short; lavender = external Lean kernel, green = proven, amber = refine bottleneck
-->

![Seed-Prover deep-and-broad reasoning loop: a conjecture is turned into lemma drafts and a main skeleton, each verified by the Lean 4 kernel, with failures routed to a refinement loop and passes routed to a self-summarization step that feeds proven lemmas back into generation](/imgs/blogs/seed-prover-formal-theorem-proving-lean-imo-1.png)

Here is the assumption-vs-reality table that frames everything:

| Common assumption | What practitioners believe | The reality Seed-Prover exposes |
|---|---|---|
| "LLMs can already do competition math" | GPT-class models get IMO problems right | They produce *natural-language* arguments that look right; many have subtle gaps a grader catches. There is no machine check. |
| "Formal proving is just LLM + Lean glue" | Wrap a model in a Lean REPL and you are done | Naive whole-proof generation collapses on hard problems; you need lemma decomposition, refinement loops, and retrieval to get past MiniF2F-level. |
| "Stepwise/subgoal proving is strictly better" | Decompose into subgoals, prove each | Subgoal decomposition (DeepSeek-Prover-V2 style) recovers a proof tree but loses *global* reasoning; whole-proof reasoning keeps the model's long CoT intact. |
| "Test-time compute is one knob" | More samples = better | Seed-Prover separates *depth* (refine one proof many rounds) from *breadth* (explore many conjectures/lemmas) and spends them differently per problem. |
| "Geometry is the easy part" | Geometry is mechanical | Lean has essentially no usable synthetic-geometry library; Seed-Prover had to build a *separate* symbolic engine (Seed-Geometry) because formalizing geometry in Lean is impractical. |

The headline numbers, all from the Seed-Prover paper (arXiv:2507.23726) and its companion repo, set the stakes:

- **MiniF2F**: 100.0% on the validation split and **99.6% on test** under the medium setting — effectively *saturating* a benchmark where the previous bests were 90.6% (DeepSeek-Prover-V2) and 92.2% (Kimina-Prover).
- **Formalized past IMO problems**: **78.1%** (121 of 155 formalized tasks).
- **PutnamBench**: 201/657 under the light setting, climbing to **331/657 under medium** — versus 49/657 for DeepSeek-Prover-V2's 671B model.
- **CombiBench**: 30/100.
- **MiniCTX-v2**: 81.8%, versus an o4-mini baseline of 44.3% at Pass@8.
- **IMO 2025**: full, kernel-checked proofs for **5 of 6 problems**.

And the follow-up, **Seed-Prover 1.5** (arXiv:2512.17260): **88% of PutnamBench** (reported as 87.9%, ~580/660), **80% of Fate-H** (graduate-level), **33% of Fate-X** (PhD-level), and 11 of 12 problems from Putnam 2025 within nine hours.

Those are not incremental numbers. PutnamBench going from ~49 solved to ~580 solved in roughly five months is the kind of jump that only happens when something structural changed. Let me explain what.

## 1. Lemma-style whole-proof reasoning, and why it beats subgoal decomposition

> The senior rule of thumb: **let the model think in whole proofs, but force it to commit to named lemmas it can verify, reuse, and remember.**

There are two dominant paradigms in neural theorem proving, and Seed-Prover's central design choice is to reject the more popular one.

**Stepwise / subgoal proving** (the AlphaProof and DeepSeek-Prover-V2 lineage) treats a proof as a search over a tree of tactic states. You have a goal; you ask the model for one tactic (or one subgoal); the proof assistant applies it and returns a new goal; you recurse. DeepSeek-Prover-V2 formalized this beautifully — it uses a large model to do *natural-language* subgoal decomposition, then a smaller prover to close each subgoal, then RL to stitch them. (I wrote about that mechanism in detail in the [DeepSeek-Prover-V2 deep-dive](/blog/paper-reading/large-language-model/deepseek-prover-v2-advancing-formal-mathematical-reasoning-via-reinforcement-learning-for-subgoal-decomposition).) The appeal is obvious: every node is independently checkable, and the search is structured.

The cost is also real. When you decompose into subgoals and prove them one at a time, the model loses the *global narrative* of the proof. A reasoning model's superpower — the long chain of thought where it notices "this looks like a telescoping sum, so let me set up the partial fractions, which means I should prove the denominator never vanishes" — gets shredded into independent fragments. Each fragment is solved by a model that no longer sees the forest.

**Whole-proof reasoning** (the Kimina-Prover and Seed-Prover lineage) does the opposite: the model emits the *entire* Lean proof in one long generation, preceded by a long CoT. The CoT is where the math happens; the Lean code is the committed artifact. This keeps the global reasoning intact. The cost *here* is that one syntax error or one unfilled `sorry` anywhere in a 2000-line proof fails the whole thing, and you get a wall of Lean errors with no structured search to fall back on.

Seed-Prover's contribution is **lemma-style** whole-proof reasoning, which is the synthesis. The model still generates whole proofs with long CoT, but it is trained and prompted to **structure proofs around named, reusable lemmas**. Concretely, a Seed-Prover proof of a hard theorem looks like:

```lean
-- Seed-Prover-style lemma-structured proof (illustrative Lean 4)
-- The model commits to named lemmas it can verify and reuse.

theorem main_inequality (n : ℕ) (hn : 0 < n) (x : Fin n → ℝ)
    (hx : ∀ i, 0 < x i) : some_target_inequality n x := by
  -- Lemma 1: a structural fact the model proved separately and cached.
  have key_bound : ∀ i, x i ≤ partial_max x i := by
    exact partial_max_is_upper_bound x i
  -- Lemma 2: a normalization step, also proved + summarized earlier.
  have normalized : ∑ i, x i / total x = 1 := by
    exact sum_div_total_eq_one hx
  -- Main argument now reads like the natural-language proof.
  calc some_target_inequality n x
      = expanded_form n x          := by rw [expand_target]
    _ ≤ bound_using key_bound      := by gcongr <;> exact key_bound _
    _ = clean_form n x             := by rw [normalized]
    _ ≤ rhs n                      := by nlinarith [key_bound, normalized]
```

The two `have ... := by exact <lemma>` lines are the point. Each named lemma (`partial_max_is_upper_bound`, `sum_div_total_eq_one`) is a **separately verifiable unit**. The model can:

1. **Verify lemmas independently** — if `key_bound` fails to type-check, the error is localized to one `have`, not smeared across the whole proof.
2. **Reuse lemmas** — a proven lemma about partial maxima is exactly the kind of thing that recurs across an entire problem set. Seed-Prover maintains a library of proven lemmas and *retrieves* relevant ones during generation.
3. **Summarize lemmas** — once a lemma is proven, the model writes a natural-language summary of *what it says and when it is useful*, which becomes context for future attempts. This is the self-summarization mechanism, and it is the difference between a system that re-derives the same wheel every attempt and one that compounds.

Why does this beat subgoal decomposition? Because lemmas are chosen by the *mathematician's* logic, not the *proof tree's* topology. A subgoal is "whatever Lean's `cases` tactic spat out." A lemma is "the partial-max bound, which I, the reasoning model, decided is the crux." The first is mechanical; the second is mathematical. The empirical gap is the PutnamBench result: whole-proof lemma reasoning at 331/657 vs. subgoal decomposition at 49/657.

There is a deeper reason lemmas are the right unit, and it is worth dwelling on because it is the load-bearing idea of the whole system. A subgoal in a tactic-state search is *context-dependent*: it only makes sense relative to the exact hypotheses in scope at that node of the proof tree. You cannot lift "this particular `unsolved goal` from depth 7 of the search" out and reuse it elsewhere, because it is glued to a local context that will never recur verbatim. A lemma is the opposite — it is *context-free by construction*. `partial_max_is_upper_bound x i` states a self-contained mathematical fact with explicit hypotheses; it can be stated, proved, named, summarized, retrieved, and reused anywhere the hypotheses hold. The entire compounding machinery — the library, the retrieval, the self-summarization — is only possible because lemmas are portable and subgoals are not. Subgoal decomposition optimizes for *local checkability*; lemma decomposition optimizes for *global reusability*. On a single problem, both work. Across a problem set, only one of them compounds, and compounding is the difference between 49 and 331.

A second mechanism reinforces this. When the model emits a whole proof, the long CoT and the Lean code are produced in one autoregressive pass, so the code is *causally downstream of* the full plan. The model that writes `calc ... ≤ rhs n := by nlinarith [key_bound, normalized]` has already, in its CoT, decided that the inequality follows from those two facts by nonlinear arithmetic. In a subgoal system, the model that closes the final subgoal has *never seen* the plan that produced the earlier subgoals — it sees only the goal handed to it. The whole-proof model carries the intent into the code; the subgoal model receives the code without the intent. This is why whole-proof proofs read like the mathematics and subgoal proofs read like a search trace.

### Second-order consequence: the cold-start data problem inverts

There is a non-obvious downstream effect. In subgoal systems, your training data is *(tactic state → next tactic)* pairs, which are cheap to harvest from existing Lean libraries. In whole-proof lemma systems, your training data is *(theorem → entire lemma-structured proof + CoT)*, which barely exists in the wild. Almost no human writes Lean proofs with the explicit lemma-summarization structure Seed-Prover wants.

So Seed-Prover has to *manufacture* its training distribution through **expert iteration**: generate proofs with the current model, keep the ones Lean accepts, distill the lemma structure and summaries, and retrain. The Lean kernel is the labeling function — it is impossible to label a wrong proof as correct, which is the entire reason RL works here at all. We will get to the RL details in §3, but the structural point is that lemma-style proving *requires* the verification loop to bootstrap, whereas subgoal proving can limp along on scraped data.

## 2. Deep vs. broad: the whole-proof reasoning advantage made concrete

> The senior rule of thumb: **depth and breadth are different budgets. Spend depth on near-misses, breadth on dead ends.**

The before/after below makes the paradigm difference visceral. On the left, stepwise/subgoal proving: a tree of tactic states, each node a tiny local decision, the global plan invisible. On the right, Seed-Prover's whole-proof lemma reasoning: one long CoT carrying the global argument, materializing as a handful of named lemmas plus a main proof.

<!-- FIGSPEC 2
kind: before-after
claim: Subgoal/stepwise proving fragments a hard problem into many local tactic decisions and loses the global plan, whereas Seed-Prover keeps one long chain-of-thought and commits it to a few named, reusable lemmas.
caption: Why whole-proof lemma reasoning preserves the global argument that subgoal search throws away.
nodes:
  - id: b_title | label: "BEFORE: stepwise / subgoal (DeepSeek-Prover-V2)" | color: amber
  - id: b1 | label: "Goal -> tactic 1 (local)" | color: amber
  - id: b2 | label: "Subgoal A -> tactic" | color: amber
  - id: b3 | label: "Subgoal B -> tactic" | color: amber
  - id: b4 | label: "Global plan: invisible" | color: red
  - id: b5 | label: "PutnamBench: 49/657" | color: red
  - id: a_title | label: "AFTER: whole-proof lemma reasoning (Seed-Prover)" | color: green
  - id: a1 | label: "One long CoT (global argument)" | color: blue
  - id: a2 | label: "Lemma A (named, verified)" | color: blue
  - id: a3 | label: "Lemma B (named, reused)" | color: blue
  - id: a4 | label: "Main proof reads like the math" | color: green
  - id: a5 | label: "PutnamBench: 331/657" | color: green
edges:
  - b_title -> b1
  - b1 -> b2
  - b1 -> b3
  - b2 -> b4
  - b3 -> b4
  - b4 -> b5
  - a_title -> a1
  - a1 -> a2
  - a1 -> a3
  - a2 -> a4
  - a3 -> a4
  - a4 -> a5
notes: two vertical columns side by side; left column amber/red (loses plan), right column blue/green (keeps plan); numbers anchor the contrast
-->

![Before-after comparison: on the left, stepwise subgoal proving fragments into local tactic decisions with an invisible global plan and 49 of 657 PutnamBench solved; on the right, whole-proof lemma reasoning keeps one long chain-of-thought, named verified lemmas, and 331 of 657 solved](/imgs/blogs/seed-prover-formal-theorem-proving-lean-imo-2.png)

Once you accept whole-proof reasoning, the question becomes: at inference time, how do you spend compute? Seed-Prover's answer is the three-setting framework, and it is worth being precise about what each does because the names (light/medium/heavy) hide the actual mechanism.

The two axes are **depth** and **breadth**:

- **Depth** = how many *refinement rounds* you run on a *single* proof attempt. The model writes a proof, Lean rejects it with errors, the model reads the errors and rewrites, repeat. Depth is for problems where you are *close* — the structure is right but a `nlinarith` needs a hint or a coercion is missing.
- **Breadth** = how many *distinct* conjectures, lemma decompositions, and independent proof attempts you launch in parallel. Breadth is for problems where your first idea was simply *wrong* — you need to explore different mathematical approaches, not polish a broken one.

The three settings combine these as follows (per the paper's description and the GitHub repo's evaluation protocol):

| Setting | Depth (refine rounds) | Breadth (attempts / conjectures) | Lemma library use | Where it is used |
|---|---|---|---|---|
| **Light** | Few | Low — fast first attempts | Minimal retrieval | First pass on every problem (PutnamBench, CombiBench, MiniF2F) |
| **Medium** | Standard refinement loop | Moderate — multiple attempts + lemma decomposition | Retrieve + reuse proven lemmas | Applied to problems light failed to solve |
| **Heavy** | Maximum refinement | Extensive — broad conjecture exploration, deep lemma trees | Full library + self-summarization in the loop | Reserved for the hardest remaining problems (IMO, leftover MiniF2F) |

The evaluation protocol is a cascade, which is the practically important detail: **light first, then medium on whatever light missed, then heavy on whatever medium missed.** This is the only sane way to spend compute. You do not run heavy on MiniF2F problems that a 13-line proof solves; you run light, sweep up 95% of them, and reserve the expensive deep-and-broad search for the residual that actually needs it. On PutnamBench, the public numbers are light = 201/657, medium = 331/657; the heaviest setting pushes higher still on the hardest tail.

The deep-vs-broad framing maps directly onto how a human olympiad team works. A strong solver who is *almost* there refines (depth). A team facing a problem nobody can crack splits up and tries radically different attacks (breadth). Seed-Prover's contribution is making this a *tunable inference policy* rather than a fixed sampling budget — and crucially, the lemma library means breadth is not wasted: a lemma proven during a failed attempt on approach A is retrieved and reused during attempt B.

Let me make the tradeoff sharper, because "depth vs breadth" sounds like a slogan until you think about where compute actually goes. Consider a problem you fail on the first attempt. There are two distinct failure modes, and they want opposite responses. **Failure mode one: right idea, broken execution.** Your proof skeleton is correct — the lemmas you chose really do imply the theorem — but one `have` does not type-check because `nlinarith` needs a product term you forgot, or a `Nat`/`Int` coercion is implicit where Lean wants it explicit. Here, *breadth is pure waste*: re-sampling a fresh attempt from scratch throws away a structurally-correct proof to gamble on getting a *different* structurally-correct proof and then having to fix *its* execution bug. Depth is the answer — feed the error back, fix the one line, done. **Failure mode two: wrong idea.** Your skeleton is fundamentally misconceived — you tried induction where the problem wants a pigeonhole argument, and no amount of fixing individual `have` blocks will rescue it because the *plan* is wrong. Here, *depth is pure waste*: refining a doomed skeleton just produces a series of differently-broken doomed skeletons. Breadth is the answer — abandon it, sample a genuinely different approach.

The reason a *fixed* sampling budget (just take N i.i.d. samples and verify each) is suboptimal is that it blindly spends breadth on both failure modes. For failure mode one it is wildly inefficient — it would need to re-roll the entire proof just to fix a coercion. For failure mode two it is fine, but it cannot tell the two modes apart, so it over-spends on the easy near-misses and under-spends on the genuinely hard ones. Seed-Prover's cascade fixes the allocation: the light setting catches the easy problems where one shot suffices; the medium setting adds *depth* (refinement rounds) to rescue the near-misses for cheap; the heavy setting adds *breadth* (broad conjecture exploration plus deep lemma trees) for the problems that need a genuinely different idea *and* a long proof. The H20-day budgets in Seed-Prover 1.5 (~10 per problem on benchmarks, ~40 for Putnam 2025) only make sense under this allocation — you could never afford ~40 H20-days of *pure breadth* per problem, but you can afford it as "mostly depth, with breadth reserved for the residual that depth cannot fix."

There is one more subtlety the lemma library introduces, and it is what makes Seed-Prover's breadth fundamentally cheaper than naive resampling. In a stateless sampler, attempt B knows nothing about attempt A. In Seed-Prover, every lemma A proved on its way to failing is *banked*. So even a failed breadth attempt is not wasted compute — it deposits verified, summarized lemmas into the library that attempt B retrieves for free. Over a long heavy-setting run on a hard problem, the effective difficulty *decreases monotonically* as the library fills with the problem's recurring sub-facts. This is why the long IMO proofs (P3 at 2000 lines, P4 at 4000) were tractable at all: by the time the system assembled the final proof, most of the constituent lemmas were already in the bank from earlier exploration.

### Second-order consequence: refinement is only as good as Lean's error messages

A subtle thing worth flagging for anyone who wants to build this. The depth axis — refine on Lean feedback — is bottlenecked by the *quality* of Lean's error reporting. Lean 4 errors are precise about *where* the proof broke (unsolved goal, type mismatch, unknown identifier) but often unhelpful about *why* in a mathematical sense. A `nlinarith failed` tells you the nonlinear arithmetic tactic could not close the goal; it does not tell you which auxiliary term you forgot to feed it. Seed-Prover's refinement works because the *model* supplies the mathematical interpretation of the error, using the CoT. This is why a weaker base model would get much less out of the same refinement loop: refinement is reasoning, not pattern-matching on error strings.

## 3. The lemma decomposition tree: how a hard theorem actually gets proved

> The senior rule of thumb: **a hard proof is a tree of lemmas, and the model's job is to find the *small* set of lemmas that make the root trivial.**

Let me make the lemma structure concrete with a tree, because "lemma decomposition" is abstract until you see the shape. Take a hard inequality or number-theory problem. Seed-Prover does not attack the statement head-on. It conjectures intermediate lemmas — facts that, *if* proven, make the main theorem fall out by a short argument — and recurses, proving each lemma either directly or by further decomposition. Proven lemmas are cached and summarized; failed branches are abandoned and their *partial* lemmas may still be reused.

<!-- FIGSPEC 3
kind: tree
claim: Seed-Prover proves a hard theorem by conjecturing a small set of intermediate lemmas, recursively decomposing the hard ones, verifying each leaf in Lean, and caching every proven lemma for reuse.
caption: A lemma decomposition tree: the root theorem reduces to a handful of verified lemmas, some proved directly, some decomposed further.
nodes:
  - id: root | label: "Theorem T (IMO-hard)" | color: blue
  - id: l1 | label: "Lemma L1: structural bound" | color: blue
  - id: l2 | label: "Lemma L2: key identity" | color: blue
  - id: l3 | label: "Lemma L3: edge case n<=2" | color: green
  - id: l1a | label: "L1a: monotonicity (verified)" | color: green
  - id: l1b | label: "L1b: base case (verified)" | color: green
  - id: l2a | label: "L2a: telescoping sum" | color: green
  - id: l2b | label: "L2b: failed branch -> retry" | color: red
  - id: l2c | label: "L2c: partial fractions (verified)" | color: green
notes: root at top, three lemmas in middle layer (>=3 wide), leaves at bottom; green leaves are kernel-verified, red node is an abandoned branch whose partials are still cached
-->

![Lemma decomposition tree: the root IMO-hard theorem reduces to three lemmas (a structural bound, a key identity, an edge case), which further decompose into verified leaves like monotonicity, base case, telescoping sum, and partial fractions, with one failed branch marked for retry](/imgs/blogs/seed-prover-formal-theorem-proving-lean-imo-3.png)

The pseudocode for the proof-search-and-refine loop ties together everything in §1 and §2. This is the medium/heavy inference loop, written to show the *control flow* rather than any specific API:

```python
## Seed-Prover medium/heavy inference loop (illustrative pseudocode).
## Lean is the only source of truth; the model supplies all the math.

def prove(theorem, library, depth_rounds, breadth_attempts):
    proven_lemmas = library.retrieve_relevant(theorem)  # reuse past work

    for attempt in range(breadth_attempts):               # BREADTH axis
        # Long-CoT whole-proof generation, lemma-structured.
        cot, lemmas, main_skeleton = model.generate(
            theorem, context=summaries_of(proven_lemmas)
        )

        # Verify each named lemma independently in Lean 4.
        for lemma in lemmas:
            result = lean.verify(lemma.statement, lemma.proof)
            for r in range(depth_rounds):                 # DEPTH axis
                if result.ok:
                    break
                # Model reads Lean errors, supplies the math, rewrites.
                lemma.proof = model.refine(lemma, result.errors, cot)
                result = lean.verify(lemma.statement, lemma.proof)
            if result.ok:
                proven_lemmas.add(lemma)
                library.add(lemma, summary=model.summarize(lemma))  # compound
            # failed lemmas: their partial sub-lemmas are still cached

        # Now try to close the main theorem using proven lemmas.
        proof = model.assemble(main_skeleton, proven_lemmas)
        verdict = lean.verify(theorem, proof)
        if verdict.ok:
            return proof                                  # kernel-checked!

    return None  # escalate to a heavier setting
```

Three things in this loop are doing the heavy lifting, and each is a research contribution in its own right:

**`library.retrieve_relevant`** — the system carries a growing library of proven lemmas and retrieves the ones relevant to the current theorem. This is retrieval-augmented *proving*. The reason it matters: olympiad problems and benchmark suites share enormous structure. A lemma about "the maximum of a finite multiset is achieved" or "a strictly monotone sequence of naturals is unbounded" recurs across dozens of problems. Re-proving it every time is the dominant cost; retrieving it is nearly free.

**`model.summarize(lemma)`** — self-summarization. After a lemma is verified, the model writes a natural-language description of *what the lemma asserts and when to invoke it*. These summaries become the context for future generation. This is what turns the library from a pile of Lean blobs into something the model can *reason about* without re-reading the full proof. It is the formal-proving analogue of a senior engineer writing a one-line docstring instead of forcing the next person to read the implementation.

**`model.refine(lemma, result.errors, cot)`** — the depth loop. The model gets the Lean errors *and* its own original chain of thought, and rewrites. Carrying the CoT forward is essential: the model needs to remember *why* it wrote the broken line to fix it correctly, rather than guessing.

### A worked Lean feedback loop: one lemma, one error, one refinement

The refinement loop is abstract until you watch a single round of it. Let me walk through a small but representative lemma — the kind of structural fact that shows up inside a larger inequality or number-theory proof — and the exact verifier-feedback-and-rewrite cycle it goes through. Say the model, on its first attempt, emits this lemma as part of a larger proof:

```lean
-- Attempt 1 (what the model first generates).
-- Claim: for positive reals, the sum of squares dominates
-- the square of the sum divided by n (Cauchy-Schwarz / power-mean).
theorem sq_sum_ge (n : ℕ) (hn : 0 < n) (a : Fin n → ℝ) :
    (∑ i, a i) ^ 2 ≤ n * ∑ i, (a i) ^ 2 := by
  nlinarith [sq_nonneg (∑ i, a i)]
```

The model's CoT said, correctly, "this is Cauchy-Schwarz with the all-ones vector; `nlinarith` should close it with the right square hints." But `nlinarith` is a *bounded* nonlinear arithmetic tactic — it multiplies pairs of provided hypotheses and runs linear arithmetic on the products. It cannot see the cross-terms that make this inequality true, because the relevant nonnegative quantities are the *pairwise differences* $(a_i - a_j)^2$, summed — and `nlinarith` has no way to conjure a sum over pairs from a single `sq_nonneg` hint. So Lean returns:

```text
error: linarith failed to find a contradiction
  ⊢ (∑ i, a i) ^ 2 ≤ ↑n * ∑ i, a i ^ 2
  (nlinarith could not close the goal after trying the supplied terms)
```

This is the *typical* Lean error: precise about *where* (this exact goal, after the hint expansion) and silent about *why* in math terms. A naive fixer that pattern-matches on "nlinarith failed → add more `sq_nonneg`" will flail, because the missing ingredient is not another square of an existing term, it is the *structural* identity that rewrites the gap as a sum of squares of differences. This is where carrying the CoT forward earns its keep. The model re-reads its own plan ("Cauchy-Schwarz with all-ones"), recognizes that `nlinarith` cannot synthesize the pairwise structure, and *changes strategy* rather than adding hints — it reaches for the Mathlib lemma that already encodes exactly this fact:

```lean
-- Attempt 2 (after refinement on the Lean error + CoT).
-- The model retrieves the right Mathlib lemma instead of brute-forcing nlinarith.
theorem sq_sum_ge (n : ℕ) (hn : 0 < n) (a : Fin n → ℝ) :
    (∑ i, a i) ^ 2 ≤ n * ∑ i, (a i) ^ 2 := by
  have h := inner_mul_le_norm_mul_norm  -- Cauchy-Schwarz in Mathlib form
  -- specialize to the all-ones vector and Finset.card_fin
  simpa [Finset.card_fin, mul_comm] using
    Finset.inner_mul_le_norm_mul_norm (Finset.univ) (fun _ => (1 : ℝ)) a
```

The second attempt type-checks. Notice what actually happened: the model did not "debug" in the software sense of poking at syntax. It *re-derived the mathematical strategy* from the error signal — "the brute-force tactic cannot see the structure, so invoke the named theorem that *is* the structure" — and then expressed it via library retrieval. That is the depth loop in one round, and it is why I keep insisting refinement is reasoning. A model without the Cauchy-Schwarz fact in its competence, or without its original CoT in context, would loop forever adding `sq_nonneg` hints to a tactic that structurally cannot use them. In a 2000-line IMO proof, dozens of `have` blocks each go through exactly this micro-cycle — generate, get a localized error, re-reason, retrieve or restructure, re-verify — and the lemma structure is what keeps each cycle scoped to a few lines instead of the whole artifact.

### Second-order consequence: lemma libraries leak benchmark-specific shortcuts

A caveat I would raise in any design review. A retrieved-lemma library that grows during evaluation can, if you are not careful, blur the line between "general capability" and "memorized this benchmark's recurring lemmas." Seed-Prover's per-benchmark numbers are reported honestly, but if you build something like this, you must decide whether the library *persists across problems within a benchmark* (which is fair and is how a human team works) or *persists across the train/test boundary* (which would be leakage). The paper's framing — light/medium/heavy applied within a benchmark — is the defensible version, but it is a knob to watch.

## 4. Training: RL against the Lean verifier on a Seed1.5-Thinking base

> The senior rule of thumb: **when your reward function is a kernel that cannot be fooled, RL stops being a hack and becomes the main course.**

Most RL-for-reasoning struggles with reward hacking — the model finds a way to make the reward go up without actually getting better, because the reward is a learned or heuristic proxy. (This is a recurring theme in the broader reasoning-RL literature; see the [DeepSeek-R1 deep-dive](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) for how much engineering goes into not getting hacked.) Formal proving is the rare setting where the reward is *exactly* what you want: **a proof either type-checks or it does not.** There is no partial credit to game, no judge to flatter. This is why the Seed team has been explicit that formal math is a near-ideal RL environment.

The training stack, as best I can reconstruct from the paper and the Seed lineage:

- **Base model**: Seed-Prover is built on the Seed1.5-Thinking lineage — a Mixture-of-Experts reasoning model (reported as ~20B activated / ~200B total parameters) with strong STEM CoT. The reasoning ability of the base matters enormously, because (as noted in §2) refinement *is* reasoning. I covered the base model's RL recipe in the [Seed1.5-Thinking post](/blog/paper-reading/large-language-model/seed1-5-thinking-rl-reasoning-vapo-dapo); the relevant inheritance is the **VAPO** value-augmented PPO machinery.
- **RL algorithm**: multi-stage, multi-task RL based on **VAPO** (value-augmented proximal policy optimization). The "multi-task" part means the model is trained jointly on whole-proof generation, lemma proving, and refinement, so the same weights are good at all three roles in the inference loop.
- **Expert iteration**: the bootstrap. Generate proofs with the current policy, keep Lean-verified ones (the kernel is the filter), distill them into lemma-structured + summarized training data, retrain, repeat. Each round the policy gets stronger, which means it can verify harder lemmas, which means the next round's data is harder and richer. This is the flywheel.
- **Conjecture and lemma data**: a large amount of the value comes from *conjectured* lemmas — the model proposes intermediate statements, the kernel verifies them, and verified ones become both training data and library entries. This is how the system manufactures the lemma-structured distribution that does not exist in human-written Lean.

The flywheel is the thing to internalize. Because the kernel never mislabels, every expert-iteration round is *monotone* in proof quality — you can only add correct proofs to the training set. Contrast this with RLHF, where a noisy reward model can drag the policy sideways. Here the worst case is "you learn nothing this round," never "you learn something wrong." That monotonicity is why the numbers moved so fast.

## 5. The IMO 2025 case study, problem by problem

> The senior rule of thumb: **a benchmark percentage is a summary; a problem-by-problem breakdown is the truth.**

This is the section that made the field pay attention. At IMO 2025, Seed-Prover (the 1.0 system) produced kernel-checked Lean proofs. The honest, repo-stated framing is important: **four problems were solved during the competition window, P1 was solved post-competition, and P6 was not solved** — so the "5 of 6" headline is "fully proved given enough time," with four of those inside the contest timeframe. Here is the matrix.

<!-- FIGSPEC 4
kind: matrix
claim: Seed-Prover produced kernel-checked Lean proofs for 5 of 6 IMO 2025 problems, with proof sizes and times ranging from a 2-second Seed-Geometry solve of P2 to a 4000-line, 3-day proof of P4, while P6 remained unsolved.
caption: IMO 2025 problem-by-problem: method, proof size, time, and outcome.
notes: 6 rows (P1..P6) x columns [Topic | Method | Proof size/time | Outcome]; use a vertical matrix so it is tall not wide; green = solved, red = unsolved, amber = solved post-contest
nodes:
  - id: h | label: "P# | Topic | Method | Size / Time | Outcome" | color: gray
  - id: p1 | label: "P1 | Combinatorics | Seed-Prover | post-contest | Solved (after)" | color: amber
  - id: p2 | label: "P2 | Geometry | Seed-Geometry | ~2 seconds | Solved" | color: green
  - id: p3 | label: "P3 | Number theory | Seed-Prover | 2000 lines / 3 days | Solved" | color: green
  - id: p4 | label: "P4 | Number theory | Seed-Prover | 4000 lines / 3 days | Solved" | color: green
  - id: p5 | label: "P5 | Combi/algebra | Seed-Prover | ~1 day | Solved" | color: green
  - id: p6 | label: "P6 | Combinatorics | Seed-Prover | -- | Unsolved" | color: red
-->

![IMO 2025 problem-by-problem matrix: P1 combinatorics solved post-contest, P2 geometry solved in about 2 seconds by Seed-Geometry, P3 number theory 2000-line proof in 3 days, P4 number theory 4000-line proof in 3 days, P5 combinatorics-algebra in about 1 day, P6 combinatorics unsolved](/imgs/blogs/seed-prover-formal-theorem-proving-lean-imo-4.png)

Walking the problems in the order the contest presents them, because the *pattern* across the day-1/day-2 split and the three subjects is the real story:

**P1 (combinatorics) — solved post-competition.** Day-1 problem 1 is traditionally the easiest slot, but 2025's P1 was combinatorial, and combinatorics is exactly where formal provers struggle most. The honest record is that Seed-Prover did *not* close it inside the contest window — it found the proof afterward. This is the first signal of the subject gradient: a "P1" that an algebra or number-theory problem of the same difficulty would have fallen to in the light setting instead required extended search because the combinatorial structure (counting configurations, casework over arrangements) is punishing to formalize. The structure was findable; the formalization simply took longer than the clock allowed.

**P2 (geometry) — ~2 seconds, via Seed-Geometry.** This is the most lopsided result of the entire competition. Once a human formalized the figure into Seed-Geometry's predicate language, the symbolic engine closed it in roughly two seconds by computing the deductive closure of the geometric facts and observing that the goal predicate appeared in it. For comparison, AlphaGeometry2 reportedly solved the same problem from natural language in about 20 seconds — an order of magnitude slower, reflecting Seed-Geometry's engine-efficiency gains. The lesson is not "geometry is trivial." It is that *the right representation* converts a problem that would be hopeless in Lean (no synthetic-geometry library) into a deductive-closure computation that is essentially instant. The load-bearing caveat, which I will keep repeating because it changes how you should read the result: the human-supplied formalization of the figure is the input, and the two seconds is the time *after* that step. More on the engine in §6.

**P3 (number theory) — 2000-line Lean proof, ~3 days.** A genuinely hard day-1 problem that required the full deep-and-broad machinery. Number-theory problems at this level usually pivot on one or two sharp lemmas — a divisibility constraint, a bound on a valuation, a parity or size argument that collapses the search space — surrounded by a large amount of mechanical-but-tedious verification (case splits, arithmetic over `Nat` and `Int` with their coercion headaches). Seed-Prover's role split here is instructive: *breadth* found the right crux lemmas by exploring conjectures, and *depth* ground out the long tail of mechanical verification via refinement. Self-summarization was essential at this length — across 2000 lines, the model has to remember what each early lemma established to invoke it correctly two hundred lines later, and the natural-language summaries are how it keeps the lemma library coherent in-context. Two thousand kernel-accepted lines is an enormous artifact, and the only reason it did not collapse under its own weight is the lemma decomposition of §1.

**P4 (number theory) — 4000-line Lean proof, ~3 days.** The largest artifact of the competition, and among the largest LLM-produced formal proofs of a single olympiad problem at the time. This is heavy-setting territory: maximum refinement depth, broad conjecture exploration, and the lemma library doing serious work to avoid re-proving the number-theoretic sub-facts that recur across a proof this size. Twice the lines of P3 in the same wall-clock window tells you the per-lemma cost stayed roughly constant — which is exactly the property lemma decomposition is supposed to give you. If the proof had been a single monolithic argument, doubling its length would have far more than doubled the difficulty, because every error would smear across the whole thing. Because it is a *tree* of independently-verified `have` blocks, scaling from 2000 to 4000 lines is close to linear: more lemmas, each still locally checkable and locally refinable.

**P5 (algebra/combinatorics) — ~1 day, inside the window.** This is the quiet triumph. P5 was solved within the contest timeframe, and it sits on the boundary between algebra (where Seed-Prover is near-human) and combinatorics (where it is weak). That it fell in about a day, well inside the clock, suggests the problem's algebraic spine dominated its combinatorial flavor — the system could lean on its strong algebraic competence and reusable Mathlib algebra lemmas rather than getting bogged down in combinatorial casework. A clean day-2 solve on a problem with any combinatorial content is more impressive than the raw "solved" bit suggests.

**P6 (combinatorics) — not solved.** The one that got away, and the most informative single data point about the system's boundary. P6 is the traditional hardest slot of the entire olympiad, and 2025's was combinatorial — the worst possible intersection for a formal prover. Combinatorial P6s typically demand a *constructive insight* (an explicit configuration achieving a bound, or a clever invariant that no amount of mechanical deduction will surface) followed by formalization-hostile casework. Seed-Prover's failure here is not a tuning miss; it is the predictable consequence of the 7/14-on-IMO-combinatorics, 30/100-on-CombiBench weakness. Neither breadth (more conjectures) nor depth (more refinement) reliably manufactures the kind of creative combinatorial construction P6 needed, because the bottleneck is *idea generation in a regime with few reusable lemmas*, not proof execution. This is the honest ceiling, and it is the most interesting open problem in the whole line of work.

The category pattern is the real finding: **algebra and number theory are strong; combinatorics is weak.** Read the day-by-day, subject-by-subject record and it is unmistakable — geometry (its own engine) instant, number theory (P3/P4) solved with long proofs, algebra-leaning P5 solved inside the window, and the three combinatorial problems (P1, the combinatorial slice of P5, P6) ranging from "solved late" to "unsolved." This is not specific to IMO 2025; it falls straight out of the formalized-IMO breakdown in the next section, where combinatorics sits at roughly a coin flip while algebra is near 85%.

## 6. Seed-Geometry: why geometry needed its own engine

> The senior rule of thumb: **do not force a problem into Lean if Lean has no library for it — build the symbolic engine the problem actually wants.**

Here is a fact that surprises people: Lean's Mathlib has essentially no usable *synthetic* (olympiad-style) geometry. You can formalize geometry in coordinates, but olympiad geometry — "prove these three lines concur," "prove this point lies on that circle" — is brutal to express and prove that way. So Seed-Prover did the pragmatic thing and built a **separate symbolic engine**, Seed-Geometry, rather than torturing Lean.

Seed-Geometry sits in the AlphaGeometry tradition but is its own implementation. The architecture is a neuro-symbolic pair, and the grid below contrasts it with AlphaGeometry2 across the dimensions that matter.

<!-- FIGSPEC 5
kind: grid
claim: Seed-Geometry pairs a forward-chaining DDAR-style deduction engine with an LLM auxiliary-construction proposer, reaching deductive closure to solve 22 of 39 hard IMO geometry problems versus AlphaGeometry2's 19, and solving IMO 2025 P2 in about 2 seconds.
caption: Seed-Geometry vs AlphaGeometry2 across engine, search, auxiliary constructions, and results.
notes: 2 columns (Seed-Geometry | AlphaGeometry2) x 4 rows (Engine | Search | Aux constructions | Result); vertical grid so it renders tall; blue = Seed-Geometry, gray = AlphaGeometry2 context, green = win rows
nodes:
  - id: h | label: "Dimension | Seed-Geometry | AlphaGeometry2" | color: gray
  - id: r1 | label: "Engine: forward-chaining deductive closure (DDAR-style)" | color: blue
  - id: r2 | label: "Search: 8x more efficient; 230M+ unique synthetic problems / 7 days" | color: blue
  - id: r3 | label: "Aux: LLM proposes constructions to break stuck states" | color: blue
  - id: r4 | label: "Result: 22/39 hard IMO geometry (SOTA)" | color: green
  - id: r5 | label: "AlphaGeometry2: DDAR + LLM, 19/39, P2 in ~20s" | color: gray
-->

![Seed-Geometry vs AlphaGeometry2 grid: Seed-Geometry uses a forward-chaining DDAR-style deductive-closure engine with 8x more efficient search generating 230M-plus synthetic problems in 7 days, an LLM proposing auxiliary constructions, solving 22 of 39 hard IMO geometry problems versus AlphaGeometry2's 19](/imgs/blogs/seed-prover-formal-theorem-proving-lean-imo-5.png)

How it works, in the same neuro-symbolic spirit as AlphaGeometry:

1. **Symbolic deduction engine (DDAR-style, forward-chaining).** Given the figure's hypotheses, the engine computes the **deductive closure** — it repeatedly applies its fixed set of geometry deduction rules (angle chasing, similar triangles, power of a point, etc.) and adds every newly derivable fact until nothing more can be derived (closure). If the goal predicate appears in the closure, the problem is solved, and you can read off a proof. This is *forward chaining*: derive everything reachable, then check if you reached the goal. It is fast and complete *within its rule set* — which is why P2 closed in ~2 seconds.

2. **LLM auxiliary-construction proposer.** Pure deduction gets stuck when the proof needs a construction not in the figure — an extra point, a midpoint, an auxiliary circle. This is the part deduction cannot invent. So an LLM proposes auxiliary constructions; each proposal is added to the figure, the deduction engine reruns to closure, and if the goal now appears, done. This is exactly AlphaGeometry's "language model proposes, symbolic engine disposes" loop.

3. **Massive synthetic data generation.** Seed-Geometry generated an enormous corpus — the public framing is **over 230 million unique problems across seven days, with an ~8× improvement in search efficiency** over prior pipelines — to train the construction proposer. Synthetic data is how you teach the LLM *which* auxiliary construction tends to unstick *which* configuration.

The headline comparison: on the harder IMO shortlist geometry problems (2000–2022, 39 problems), **Seed-Geometry solved 22 versus AlphaGeometry2's 19** — a new state of the art. And the ~2s vs ~20s on IMO 2025 P2 reflects the engine-efficiency gains.

It is worth being precise about *where* Seed-Geometry pulls ahead of AlphaGeometry2, because "+3 problems and 10× faster" is a summary that hides the mechanism. Both systems share the same high-level architecture — DDAR-style forward-chaining deduction engine plus an LLM that proposes auxiliary constructions — so the gains are not architectural novelty; they are *engineering of the two halves*. On the **engine** side, the relevant lever is how fast you can recompute the deductive closure, because the LLM-proposes/engine-disposes loop reruns the closure after *every* proposed construction. If the engine is 8× faster, you can afford 8× more construction proposals in the same wall-clock budget, which is precisely what the reported ~8× search-efficiency improvement buys. The two-seconds-vs-twenty on P2 is the engine speed showing up end-to-end: same problem, same kind of solution, an engine that reaches closure roughly an order of magnitude faster. On the **proposer** side, the lever is the *quality* of the auxiliary-construction suggestions, and that is a data problem — which configurations call for a midpoint, which for a second intersection point, which for an auxiliary circle. The 230-million-problem synthetic corpus is the answer: more and better-targeted training data teaches the LLM to propose the *right* construction earlier, so fewer closure-recomputations are needed per solve. The +3 problems on the hard shortlist is mostly this — constructions AlphaGeometry2's proposer missed or reached too late, Seed-Geometry's proposer surfaces.

The structural similarity to AlphaGeometry is the point, not a knock: this is a domain where the *recipe* (symbolic deduction + neural construction proposal + massive synthetic data) is now well-understood, and the frontier moves by making each component faster and better-trained rather than by inventing a new paradigm. That is a healthy sign for a subfield — it means the geometry half of olympiad math is approaching the regime where it is an engineering exercise, not a research gamble. Contrast that with the *combinatorics* half, where no equivalent recipe exists, and you understand exactly why Seed-Prover treats geometry as a near-solved appendage (its own fast engine) and combinatorics as the open frontier.

### Second-order consequence: the formalization boundary is the human-in-the-loop

The thing to be clear-eyed about: Seed-Geometry solved P2 in two seconds *after a human formalized the figure into its predicate language*. The formalization step — turning "let ABC be a triangle with..." into the engine's hypotheses — was human. That is not cheating (AlphaGeometry has the same boundary), but it means the geometry result is "given a faithful formalization, the engine is near-instant," not "end-to-end from the natural-language problem." It is a different, smaller claim than the Lean results for P3/P4, which were closer to end-to-end given the formal statement.

## 7. Where Seed-Prover sits versus DeepSeek-Prover-V2, Kimina-Prover, and AlphaProof

> The senior rule of thumb: **compare on the benchmark the *other* system was designed to win, not the one you were.**

Let me lay out the head-to-head honestly. The before/after-style matrix below puts Seed-Prover against the prior SOTA on the benchmarks where direct comparison is possible.

<!-- FIGSPEC 6
kind: matrix
claim: Seed-Prover saturates MiniF2F at 99.6% test and reaches 331/657 on PutnamBench, beating DeepSeek-Prover-V2 (90.6% MiniF2F-test, 49/657 PutnamBench) and Kimina-Prover (92.2% MiniF2F-test) by large margins.
caption: Seed-Prover vs prior SOTA on the directly comparable formal-proving benchmarks.
notes: rows = systems (Seed-Prover, Kimina-Prover, DeepSeek-Prover-V2, AlphaProof); columns = MiniF2F-test | PutnamBench | Notes; vertical matrix, tall; green = Seed-Prover wins, blue = Seed-Prover, gray = baselines
nodes:
  - id: h | label: "System | MiniF2F-test | PutnamBench | Note" | color: gray
  - id: sp | label: "Seed-Prover | 99.6% | 331/657 | whole-proof + lemma" | color: green
  - id: kp | label: "Kimina-Prover | 92.2% | -- | whole-proof" | color: gray
  - id: dp | label: "DeepSeek-Prover-V2 671B | 90.6% | 49/657 | subgoal decomp" | color: gray
  - id: ap | label: "AlphaProof | IMO 2024 silver | -- | RL + AlphaZero-style" | color: gray
  - id: gap | label: "Gap: +7.4pts MiniF2F; ~6.7x PutnamBench" | color: blue
-->

![Comparison matrix of formal provers: Seed-Prover at 99.6% MiniF2F-test and 331 of 657 PutnamBench, Kimina-Prover at 92.2% MiniF2F-test, DeepSeek-Prover-V2 671B at 90.6% MiniF2F-test and 49 of 657 PutnamBench, AlphaProof at IMO 2024 silver-medal level, with Seed-Prover ahead by 7.4 points on MiniF2F and roughly 6.7x on PutnamBench](/imgs/blogs/seed-prover-formal-theorem-proving-lean-imo-6.png)

The detailed table, with sources and caveats:

| System | MiniF2F-test | PutnamBench | ProofNet-test | Paradigm | IMO |
|---|---|---|---|---|---|
| **Seed-Prover** | **99.6%** (medium; 100% valid) | **331/657** medium (201 light) | high (paper-reported) | whole-proof, lemma-style | 5/6 IMO 2025 (Lean) |
| Kimina-Prover | 92.2% | — | — | whole-proof CoT | — |
| DeepSeek-Prover-V2 671B | 90.6% | 49/657 | 37.1% (Pass@1024) | subgoal decomposition | — |
| AlphaProof (DeepMind) | — | — | — | RL, AlphaZero-style search | IMO 2024 silver (4/6, w/ AlphaGeometry2) |
| Seed-Prover 1.5 | — | **~88%** (580/660) | — | agentic RL + TTS | 5/6 IMO 2025 |

A few honest readings of this table:

**MiniF2F is saturated, and that is mostly a statement about MiniF2F.** When three systems are at 90–100%, the benchmark has stopped discriminating. The real frontier moved to PutnamBench (undergraduate competition), FATE (graduate/PhD), and CombiBench (combinatorics). Seed-Prover's MiniF2F number is impressive but the *interesting* numbers are PutnamBench and IMO. Why does Seed-Prover *lead* on MiniF2F at all, given that 90.6% and 92.2% were already very high? The honest answer is that the last ~8 points of MiniF2F are not "more high-school problems" — they are the residual problems that resist single-shot whole-proof generation and need either a sharper lemma decomposition or several refinement rounds. The medium setting's depth loop is exactly what closes that residual: most of the gap from 92% to 99.6% is problems where the first proof was *almost* right and one or two rounds of Lean-feedback refinement fixed it. So Seed-Prover's MiniF2F lead is less a statement about raw proving power and more a demonstration that the refinement loop reliably mops up near-misses. That is genuinely useful — but it is why I would not lead a 2026 capability claim with MiniF2F.

**PutnamBench is where the paradigm difference shows, and where Seed-Prover leads by the widest margin.** 331 vs 49 is not a tuning difference; it is whole-proof-lemma reasoning vs subgoal decomposition on hard, multi-step undergraduate problems. This is the single most persuasive number for the §1 thesis. The reason the gap is so large *specifically on Putnam* — rather than on MiniF2F, where it is a few points — is that Putnam problems have the structural property that punishes subgoal decomposition most: they require a *global plan* with several stages that only make sense together (pick the right generating function, *then* extract a coefficient, *then* bound it), and the right intermediate lemmas are *not* obvious from any local tactic state. A subgoal system that decomposes mechanically gets subgoals that are individually hard and collectively planless; a whole-proof system that reasons globally first chooses lemmas that make the stages click. MiniF2F problems are short enough that even a planless approach often stumbles onto the answer; Putnam problems are long enough that planlessness is fatal. The benchmark is, in effect, a direct measurement of how much the global plan matters — and 331-vs-49 is the field's clearest quantification of that.

**AlphaProof is the closest spiritual peer, and it is not directly comparable.** DeepMind's AlphaProof reached silver-medal level at IMO 2024 (solving 4/6, with AlphaGeometry2 handling the geometry), using an AlphaZero-style RL search over a formal language. Seed-Prover's IMO 2025 result (5/6 with Lean proofs) is one year later and on a different problem set, so "Seed-Prover beats AlphaProof" is not a claim the data cleanly supports — they are contemporaneous high points using different methods. What *is* fair: Seed-Prover's whole-proof reasoning is a different bet than AlphaProof's search-heavy approach, and it generalizes across the broad benchmark suite (MiniF2F/Putnam/Proof Net/CombiBench) in a way a pure search system does not obviously do.

**ProofNet**: DeepSeek-Prover-V2's 671B CoT model is at 37.1% Pass@1024 on ProofNet-test; Seed-Prover's paper-reported ProofNet number is materially higher, though I would point readers to the paper's exact table rather than trust a single number, because the ProofNet figure varied across the sources I cross-checked.

## 8. The full capability picture, and what the benchmarks reveal

> The senior rule of thumb: **read the benchmark *spread*, not the best number — the spread tells you what the system cannot do.**

The capability summary matrix below is the one to keep. It puts every benchmark next to the score and, crucially, next to *what the score implies about the system's competence boundary*.

<!-- FIGSPEC 7
kind: matrix
claim: Across benchmarks Seed-Prover is near-saturated on MiniF2F (99.6%) and strong on formalized IMO (78.1%) and MiniCTX-v2 (81.8%), but combinatorics is the clear weakness (CombiBench 30/100, 7/14 combinatorics IMO), and Seed-Prover 1.5 pushes Putnam to 88% while Fate-X stays at 33%.
caption: The capability spread: where Seed-Prover is saturated, where it is strong, and where combinatorics caps it.
notes: rows = benchmarks; columns = score | implication; vertical/tall matrix; green = saturated/strong, amber = mid, red = weak (combinatorics, Fate-X)
nodes:
  - id: h | label: "Benchmark | Score | Reading" | color: gray
  - id: m | label: "MiniF2F-test | 99.6% | saturated" | color: green
  - id: imo | label: "Formalized IMO | 78.1% (121/155) | strong" | color: green
  - id: ctx | label: "MiniCTX-v2 | 81.8% vs o4-mini 44.3% | generalizes" | color: green
  - id: put | label: "PutnamBench | 331/657 -> 1.5: 88% | strong" | color: blue
  - id: combi | label: "CombiBench | 30/100 | weak (combinatorics)" | color: red
  - id: imocombi | label: "IMO combinatorics | 7/14 | weak axis" | color: red
  - id: fate | label: "Fate-H 80% / Fate-X 33% (1.5) | research frontier" | color: amber
-->

![Capability summary matrix: MiniF2F-test 99.6% saturated, formalized IMO 78.1% (121 of 155) strong, MiniCTX-v2 81.8% versus o4-mini 44.3% generalizing, PutnamBench 331 of 657 rising to 88% in version 1.5, CombiBench 30 of 100 weak, IMO combinatorics 7 of 14 weak, Fate-H 80% and Fate-X 33% at the research frontier](/imgs/blogs/seed-prover-formal-theorem-proving-lean-imo-7.png)

The formalized-IMO breakdown (121/155 = 78.1%) is the most diagnostic single result, because the paper splits it by difficulty and by subject:

- **By difficulty**: 47/55 easy, 47/56 medium, **27/44 hard**. The system is excellent on easy/medium and degrades — but not catastrophically — on hard.
- **By subject**: 72/85 algebra, 42/55 number theory, **7/14 combinatorics**. Algebra ~85%, number theory ~76%, combinatorics ~50%.

That combinatorics number — 7 of 14, and CombiBench at 30/100 — is the honest weakness, and it explains the IMO 2025 pattern (P6 combinatorics unsolved, P1 combinatorics only solved post-contest). **Combinatorics resists formalization** for a structural reason: combinatorial arguments lean on "consider the following clever bijection" or "without loss of generality, order the elements," constructions that are easy to state informally but require enormous Lean scaffolding (finite sets, pigeonhole, careful index bookkeeping) to formalize. The reasoning is creative-construction-heavy, and there are fewer reusable Mathlib lemmas to retrieve. This is a known hard frontier, not a Seed-Prover-specific failure.

The MiniCTX-v2 result (81.8% vs o4-mini's 44.3% at Pass@8) is the generalization signal worth flagging — MiniCTX tests proving in *project contexts* the model has not seen, and the large margin suggests the lemma-style reasoning transfers beyond memorized benchmark structure.

The FATE family deserves its own paragraph, because it is the benchmark that maps the *ceiling* rather than the floor. FATE is graded by difficulty: Fate-H is graduate-level, Fate-X is PhD-level. Seed-Prover 1.5's numbers — **80% on Fate-H, 33% on Fate-X** — trace a steep cliff, and the shape of that cliff is the most honest thing in the whole evaluation. Eighty percent on graduate-level mathematics says the system has genuinely moved past competition math into the lower reaches of "real" mathematics: these are problems a strong graduate student would find non-trivial. But the drop to 33% at PhD level is where the method's nature shows through. PhD-level formal mathematics increasingly requires *synthesizing* machinery across multiple results — recognizing that a problem in one area is secretly an instance of a theorem in another, importing the right abstraction, building the bridge. That is a different cognitive act than "find the lemma decomposition that closes this self-contained statement," and it is precisely the act the authors acknowledge the system *cannot* yet do (see the limitation in §9). So Fate-H-to-Fate-X is not a smooth difficulty curve that more compute will straighten out; it is a *qualitative* boundary between "hard but self-contained" and "requires cross-domain synthesis." Reading 80%/33% as "almost at PhD level, just needs scaling" would be the wrong inference. The right inference is: the lemma-decomposition-plus-verification recipe saturates competition and graduate math, and hits a genuinely different problem at the research frontier. Where Seed-Prover *trails* is not on any benchmark of self-contained theorem proving — it is on the one axis (Fate-X) that starts to probe mathematical *invention*.

## 9. Seed-Prover 1.5: going agentic, and pushing PutnamBench to ~88%

> The senior rule of thumb: **when whole-proof generation tops out, stop regenerating the whole proof — let the model incrementally build and cache.**

Five months after the original, Seed-Prover 1.5 (arXiv:2512.17260) changed the architecture in a way that is instructive about where the field is heading. The timeline below traces the shift.

<!-- FIGSPEC 8
kind: timeline
claim: Seed-Prover evolved from whole-proof generation (1.0, IMO 2025, 5/6) to an agentic system (1.5) that incrementally invokes Lean, Mathlib search, and Python within a turn, caches verified lemmas, and reaches 88% PutnamBench at ~10 H20-days per problem.
caption: From whole-proof reasoning to agentic, tool-using, experience-accumulating proving.
notes: vertical timeline, 5 stops top-to-bottom; blue = method, green = result milestones, gray = base-model context, amber = compute cost
nodes:
  - id: t1 | label: "Seed-Prover 1.0: whole-proof + lemma, Seed1.5-Thinking base, VAPO RL" | color: blue
  - id: t2 | label: "IMO 2025: 5/6 formally proved (P2 in ~2s)" | color: green
  - id: t3 | label: "1.5: agentic RL on Doubao-Seed-1.6; multi-tool turns (Lean + Mathlib search + Python)" | color: blue
  - id: t4 | label: "Caches verified lemmas; incremental build vs regenerate" | color: blue
  - id: t5 | label: "PutnamBench 88%, Fate-H 80%, Fate-X 33%; Putnam 2025 11/12 in 9h; ~10 H20-days/problem" | color: green
  - id: t6 | label: "Limit acknowledged: cannot match human experts on frontier research" | color: amber
edges:
  - t1 -> t2
  - t2 -> t3
  - t3 -> t4
  - t4 -> t5
  - t5 -> t6
notes: single vertical chain, tall not wide
-->

![Timeline from Seed-Prover 1.0 to 1.5: 1.0 uses whole-proof lemma reasoning on a Seed1.5-Thinking base with VAPO RL and solves 5 of 6 IMO 2025 problems, 1.5 shifts to agentic RL on Doubao-Seed-1.6 with multi-tool turns and lemma caching, reaching 88% PutnamBench, 80% Fate-H, 33% Fate-X and Putnam 2025 11 of 12 in 9 hours, with the acknowledged limit of not matching human experts on frontier research](/imgs/blogs/seed-prover-formal-theorem-proving-lean-imo-8.png)

What changed, concretely:

**From whole-proof to agentic.** Instead of generating an entire proof and verifying it once, the 1.5 prover acts as an agent that **incrementally invokes multiple tools to construct the proof step by step**, caching verified lemmas as it goes. Within a single turn it can make multiple tool calls — Lean verification, Mathlib semantic search, and Python execution — in any order. This is a bet that *interleaving* generation and verification beats the generate-then-check cycle, because the model gets feedback before committing 4000 lines.

**A hierarchical three-agent test-time scaling workflow.** A natural-language prover generates an informal proof; a sketch model translates it into a Lean proof sketch (with holes); the agentic prover recursively verifies and fills individual lemmas. This explicitly bridges the natural-language↔formal gap by riding on advances in *natural-language* proving (the NL prover is initialized from Doubao-Seed-1.6) and only formalizing what the kernel must check.

**Learning from experience via large-scale agentic RL.** The model accumulates experience through extensive interactions with Lean and tools during RL — the framing the paper emphasizes is "scaling learning from experience, driven by high-quality formal feedback." This is the same kernel-as-reward flywheel from §4, now applied to a tool-using agent.

The results: **87.9% of PutnamBench (~580/660)**, **80% of Fate-H** (graduate-level), **33% of Fate-X** (PhD-level), and **11 of 12 Putnam 2025 problems within nine hours** — at a reported budget of ~10 H20-days per problem on the main benchmarks (and ~40 H20-days per problem for Putnam 2025), which the authors frame as a *smaller* compute budget than prior SOTA for comparable results. The Fate-X 33% is the most telling: it is the PhD-level frontier, and a third is both genuinely impressive and a clear marker of how far there still is to go.

## Case studies: specific theorems, what worked, what did not

### 1. IMO 2025 P2 (geometry) — the two-second solve

Seed-Geometry closed IMO 2025 P2 in roughly two seconds once the figure was formalized into its predicate language. The mechanism is worth restating precisely because it is so different from everything else in the system: forward-chaining deductive closure with the standard olympiad rule set (angle chasing, similar triangles, cyclic quadrilaterals, power of a point). The engine starts from the formalized hypotheses and repeatedly applies every rule whose preconditions are met, adding each newly-derived fact to its database, until no rule fires anything new — closure. Then it checks one bit: is the goal predicate (this point lies on that circle, these three lines concur) in the closure? For P2 the answer was yes, and the two seconds is the time to compute the closure and read off the supporting deduction chain. No sampling, no neural generation, no refinement — deterministic symbolic computation.

The lesson is the §6 thesis distilled into one data point: *representation dominates*. The identical problem took AlphaGeometry2 about 20 seconds, an order of magnitude slower, and a Lean-only approach would have been hopeless from the start because Mathlib has no usable synthetic-geometry primitives — you would be reduced to coordinate bashing, which is exactly the wrong substrate for "prove these three lines concur." By choosing a representation in which olympiad geometry *is* deductive closure, Seed-Geometry converts a class of problems that defeats general-purpose provers into a near-instant lookup.

The caveat, restated because it genuinely changes how you should read the headline: the human-supplied formalization of the figure is the load-bearing input, and the two seconds is the time *after* that translation. Turning "let $\omega$ be the circumcircle of triangle $ABC$, let $D$ be the foot of the altitude from $A$..." into the engine's predicate hypotheses is a human step, and getting it faithful is non-trivial. So the precise claim is "given a correct formalization, the geometry engine is essentially instant," not "end-to-end from the contest PDF." That is still a remarkable result — it is the cleanest example in the entire system of "build the engine the problem actually wants" decisively beating "force everything through one general substrate" — but it is a narrower claim than the P3/P4 Lean results, which were much closer to end-to-end from the formal statement.

### 2. IMO 2025 P4 (number theory) — the 4000-line proof

The largest artifact of the competition: a 4000-line Lean proof completed in about three days under the heavy setting, and among the largest LLM-produced formal proofs of a single olympiad problem at the time. What makes this a case study rather than just a number is *how* it stayed tractable, because a 4000-line proof should, naively, be impossible to produce reliably. A monolithic 4000-line argument would be unmaintainable even for an expert human, and — far worse for a machine — it would fail catastrophically on any single error: one unfilled `sorry`, one type mismatch, one `nlinarith` that does not close, and the entire artifact is rejected with no partial credit.

The lemma-style structure of §1 is precisely what converts that catastrophe into a manageable engineering problem. The proof is not 4000 lines of straight-line argument; it is a *tree* of named, independently-verified lemmas, each a self-contained `have` or top-level `theorem` that the kernel checks in isolation. When a lemma fails, the error is localized to its few lines, and the refinement loop fixes it there without touching the rest. When a sub-fact recurs — and in a 4000-line number-theory proof, divisibility and valuation facts recur constantly — the library retrieves the already-proven version instead of re-deriving it. The self-summarization keeps the whole thing navigable: the model reasons over one-line summaries of two hundred lemmas rather than re-reading two hundred proofs.

The deepest point this case makes is about *scaling*. P3 was 2000 lines; P4 was 4000, in the same wall-clock window. If proof difficulty grew super-linearly with length — as it would for a monolithic argument, where every new line can interact with every existing line — then doubling the length would have blown the time budget. That it did not is direct evidence that lemma decomposition makes proof length scale roughly *linearly*: 4000 lines is "twice as many lemmas," not "exponentially more entangled reasoning." This is the existence proof that whole-proof reasoning extends to artifacts far beyond what fits in a single coherent chain of thought — *provided* the lemma tree is the scaffolding holding it up.

### 3. IMO 2025 P3 (number theory) — 2000 lines, deep refinement

P3's 2000-line, three-day proof is the cleanest demonstration of the depth axis working at scale. Hard olympiad number theory has a characteristic shape: one or two genuinely sharp lemmas — a divisibility bound that pins down the candidates, a $p$-adic valuation argument, a size estimate that forces finiteness — surrounded by a large volume of mechanical-but-unforgiving verification. The mechanical part is where most of the 2000 lines live: case splits over residues, arithmetic over `Nat` and `Int` with their constant coercion friction, induction base cases, the bookkeeping that humans wave away with "similarly" and Lean refuses to.

Seed-Prover's two axes split this labor exactly along its natural seam. *Breadth* — exploring conjectured intermediate statements — is how the system found the *right* crux lemmas, the ones whose statements make the theorem fall out. You cannot grind your way to the right lemma by refinement; you have to *try different lemmas*, which is what breadth does. Then *depth* — the refinement loop — is how the system ground out the long mechanical tail once the crux statements were fixed: write the `have`, get a localized `unsolved goals` or `nlinarith failed`, re-reason with the CoT in context, retrieve the right Mathlib lemma or restructure the tactic, re-verify. Each of the dozens of mechanical lemmas went through several rounds of exactly the micro-cycle I walked through in §3.

Self-summarization was not optional at this length, and P3 is where you can see why. A 2000-line proof has a working memory problem: by line 1500, the model needs to correctly invoke a lemma it proved at line 200, with the right argument order and the right hypotheses discharged. Re-reading the line-200 proof every time would blow the context budget; instead, the model carries the one-line *summary* of what that lemma asserts and when to use it. The library plus summarization is the system's long-term memory, and a number-theory proof of this length is the regime where that memory stops being a nicety and becomes the thing that makes the proof possible at all.

### 4. IMO 2025 P6 (combinatorics) — the failure

The one Seed-Prover did not solve, and the most informative single negative in the whole record. P6 is the traditional hardest slot of the olympiad, and 2025's was combinatorial — the worst possible intersection for a formal prover, landing squarely on the system's documented weak axis (7/14 on formalized IMO combinatorics, 30/100 on CombiBench).

The root cause is structural, not a tuning failure, and it is worth being precise about because it tells you what would *actually* be needed to fix it. Combinatorial olympiad proofs typically hinge on a *constructive insight*: an explicit configuration that achieves a bound, a coloring or pairing that exposes an invariant, a clever way to count the same set two ways. These insights are easy to *state* informally — "two-color the grid like a checkerboard," "pair each element with its complement" — and brutal to *formalize*, because Lean wants the construction spelled out as an honest function with all its properties proved, and there are far fewer reusable Mathlib lemmas to lean on than in algebra or number theory. Worse, the bottleneck is *finding the insight*, and neither of Seed-Prover's axes reliably manufactures one. Breadth (more conjectured lemmas) helps only if the right lemma is in the neighborhood of what the model already considers; depth (more refinement) only polishes an idea you already have. P6 needed a creative combinatorial construction that the system's lemma-exploration did not surface, and you cannot refine your way to an idea you never had.

This is the honest ceiling of the current method, and it is the most interesting open problem in the line of work precisely *because* it is not a scale-only fix. Throwing ten times the compute at P6 would mostly produce ten times as many almost-right-but-planless attempts. What combinatorics seems to need is a different kind of search — over *constructions* and *invariants*, with a much richer reusable library of combinatorial primitives — and building that is a research program, not a knob.

### 5. PutnamBench — 49 to 331, the paradigm proof

PutnamBench is the single most persuasive case study for the §1 thesis, because it isolates the paradigm difference under near-laboratory conditions. DeepSeek-Prover-V2's 671B subgoal-decomposition model solves 49 of 657; Seed-Prover's whole-proof lemma reasoning solves 331 under the medium setting. Same problems, roughly contemporaneous systems, both built on strong reasoning bases with RL against Lean — so the ~6.7× gap is about as clean an A/B on "subgoal decomposition vs whole-proof lemma reasoning" as the field has produced.

Why *Putnam specifically* magnifies the gap is the lesson worth internalizing. Putnam problems are undergraduate-competition-hard and multi-stage: a typical one wants you to set up the right object (a generating function, a clever substitution, an auxiliary sequence), *then* manipulate it, *then* extract and bound the quantity you care about. The stages only make sense *together* — the substitution is "right" precisely because of what it lets you do three steps later. A subgoal decomposition that fragments this into locally-checkable pieces destroys the very thing that makes the stages coherent: the global plan that justifies each choice. The model closing a late subgoal never saw the plan that produced it, so it is solving a hard, planless fragment. The whole-proof model, by contrast, reasoned globally first and *then* committed the plan to named lemmas, so each lemma is chosen for its role in the overall argument.

The reason this gap is *small* on MiniF2F (a few points) but *enormous* on Putnam (6.7×) is itself diagnostic: MiniF2F problems are short enough that even a planless approach often stumbles into the answer, while Putnam problems are long enough that planlessness is fatal. PutnamBench is, in effect, a direct measurement of how much the global plan matters, and the answer is "a lot." The design-review lesson I take from this: when the reasoning is irreducibly global, keep the reasoning global, and use *lemmas* — chosen by the math — as your verifiable checkpoints, not *subgoals* — chosen by the proof tree's topology.

### 6. MiniF2F saturation — when a benchmark stops mattering

Seed-Prover hits 100.0% valid / 99.6% test on MiniF2F under the medium setting, versus 90.6% (DeepSeek-Prover-V2) and 92.2% (Kimina-Prover). The case-study value here is deliberately *negative*: this is a benchmark that has stopped discriminating, and recognizing when that has happened is a skill in itself.

MiniF2F is high-school-competition mathematics formalized in Lean, and it was the field's primary yardstick for years — the number everyone led with, the leaderboard everyone chased. When three independent systems all land between 90% and 100%, the benchmark has done its job and is now measuring noise: the differences between systems at that level are mostly about which handful of residual problems happen to need one extra refinement round. Seed-Prover's specific path to the last few points is instructive — most of the gap from ~92% to 99.6% is *not* new proving power but the medium-setting depth loop mopping up near-misses, problems where the first whole-proof attempt was almost right and one or two rounds of Lean-feedback refinement closed it. That is a real and useful capability, but it is a capability about *finishing*, not about *reach*.

The broader point is about benchmark hygiene. The field's center of gravity moved to PutnamBench (undergraduate), FATE (graduate/PhD), and CombiBench (combinatorics) precisely *because* of saturation results like this one — once MiniF2F could no longer separate the frontier, the frontier had to find harder yardsticks. If you are evaluating a theorem prover in 2026 and you lead with MiniF2F, you are reporting a 2023 metric and telling your reader nothing about where the system actually sits. The honest read of Seed-Prover's 99.6% is "MiniF2F is solved; look at PutnamBench and Fate-X to see what is not."

### 7. Formalized IMO 78.1% — the difficulty/subject gradient

121 of 155 formalized historical IMO problems, broken down two ways: by difficulty as 47/55 easy, 47/56 medium, 27/44 hard; and by subject as 72/85 algebra, 42/55 number theory, 7/14 combinatorics. The aggregate 78.1% is the headline, and the case-study lesson is that the headline is the *least* useful number in the set. The breakdown is where the actual capability profile lives.

Read the difficulty axis first. The system is at roughly 85% on easy, 84% on medium, and 61% on hard. That is a remarkably *flat* curve from easy to medium — the system does not fall off a cliff as problems get harder within a subject; it degrades gracefully. Even on *hard* historical IMO problems it clears 60%, which a few years ago would have been science fiction. The graceful degradation is itself evidence for the deep-and-broad machinery: harder problems just consume more depth (refinement rounds) and breadth (conjecture exploration), and the cascade keeps paying off rather than hitting a wall.

Now read the subject axis, which is where the story turns. Algebra at 72/85 (~85%) is near-human. Number theory at 42/55 (~76%) is strong. Combinatorics at 7/14 (~50%) is a coin flip. This is the same gradient that produced the IMO 2025 pattern (geometry instant, number theory solved with long proofs, combinatorics late or unsolved), and it is not a coincidence — it is the system's fundamental shape. The deployment lesson is concrete and I would put it in any decision memo: *make the call on the cell that matches your problem distribution, not on the 78.1%.* If your problems are algebraic, you are buying an ~85% system; if they are combinatorial, you are buying a ~50% system; the aggregate is a weighted average that describes neither. Benchmarks that report only the headline are hiding exactly the information a deployer needs.

### 8. MiniCTX-v2 generalization — 81.8% vs 44.3%

On MiniCTX-v2, which tests proving inside *unseen project contexts*, Seed-Prover reaches 81.8% against an o4-mini baseline of 44.3% at Pass@8. This is the case study that answers the most natural skeptical objection to every benchmark number above: "it just memorized the benchmarks." MiniCTX is specifically constructed to defeat memorization — it asks the system to prove theorems inside Lean *projects it has not seen*, which means using definitions, notation, and local lemmas that exist only in that project and could not have been in any training set.

The reason Seed-Prover's machinery transfers to this setting is mechanistic, not lucky. Proving in a novel project is *exactly* the task the lemma-retrieval and self-summarization apparatus was built for: you cannot rely on memorized proofs, so you must *read the local context*, identify the project-specific lemmas relevant to your goal, summarize what they give you, and assemble a proof out of them. That is the same retrieve-summarize-assemble loop the system runs over its own lemma library, pointed at an external project instead. A system that had merely memorized MiniF2F-and-Putnam-shaped proofs would have nothing to retrieve here and would collapse to roughly the base model's competence — which is presumably near the 44.3% the o4-mini baseline shows.

The ~37-point margin is, to my eye, the single strongest piece of evidence in the entire evaluation that the *method generalizes* rather than the weights memorizing. Saturating MiniF2F could be memorization; leading Putnam could be a stronger base model; but tripling the proof rate on *unseen project contexts* can only come from a procedure that reads and uses context it has never been trained on. That is the capability you actually want from a theorem prover you intend to point at a real, unfamiliar formalization project.

### 9. Putnam 2025, 11 of 12 (Seed-Prover 1.5) — the agentic payoff

Seed-Prover 1.5 solved 11 of 12 Putnam 2025 competition problems within nine hours, at a budget of ~40 H20-days per problem. Two things make this a load-bearing case study rather than just another number. First, *recency*: Putnam 2025 is a fresh competition, post-dating the training cutoff, so 11/12 cannot be explained by memorization — these are genuinely novel problems solved cold. This is the cleanest "not memorization" result in the whole line of work, even stronger than MiniCTX, because it is a real competition with no possibility of the problems having leaked into pretraining.

Second, it is the clearest demonstration of *why the 1.5 architecture shift paid off*. Recall the change (§9): 1.0 generated whole proofs and verified them once; 1.5 acts as an *agent* that interleaves generation with tool calls — Lean verification, Mathlib semantic search, Python — within a single turn, caching verified lemmas as it goes. On a multi-stage Putnam problem, the agentic loop has a structural advantage over whole-proof generation: the agent gets Lean feedback *before* committing to a full proof, so it discovers that stage-two does not type-check while it is still on stage two, instead of after emitting all four stages. And the lemma caching means that when it backtracks and tries a different stage-three, the verified stage-one and stage-two are still in the bank. Whole-proof generation, by contrast, would re-emit and re-verify the entire proof on every attempt, churning compute on the parts that were already correct. Eleven of twelve in nine hours is what that efficiency buys: the agent spends its compute on the *frontier* of each proof rather than re-litigating settled ground.

### 10. Fate-X 33% (Seed-Prover 1.5) — the PhD frontier

Fate-X is PhD-level formal mathematics; Seed-Prover 1.5 solves 33%. This case study is entirely about *calibration*, and it is the most important number in the post for resisting hype. A third of PhD-level problems is simultaneously a genuinely remarkable capability — these are problems most working mathematicians outside the relevant subfield could not do — and an unambiguous ceiling that should discipline any "AI has solved mathematics" narrative.

The shape of the cliff is what makes it a ceiling rather than a way-station. Seed-Prover 1.5 is at 80% on Fate-H (graduate-level) and 33% on Fate-X (PhD-level): the drop is not a gentle difficulty slope but a phase change. The authors are explicit about why, and it is the right diagnosis — the system "cannot make significant mathematical contributions comparable to human experts" because PhD-level mathematics increasingly requires *synthesizing insights across research papers*, recognizing that a problem in one area is an instance of a structure from another and importing the bridge. That is a different cognitive act from "find the lemma decomposition that closes this self-contained statement," and it is the act Seed-Prover cannot yet do.

The discipline this imposes: do not extrapolate the Fate-H→Fate-X line and conclude that one more model generation reaches research-level. The 80%-to-33% drop is the method hitting a *qualitatively* different problem, not just a harder instance of the same one. Competition and graduate math are, in a real sense, exercises in *applying* known machinery cleverly; research math is exercises in *building new* machinery, and the lemma-decomposition-plus-verification recipe is an application engine, not an invention engine. 33% is exactly where that distinction becomes visible, and it is the honest boundary of what kernel-checked LLM proving can do as of late 2025.

### 11. The 230-million-problem geometry data engine

Seed-Geometry's training data came from a synthetic generator that produced over 230 million unique problems in seven days, with an ~8× search-efficiency gain over prior pipelines. The case study is about *where the work actually is* in neuro-symbolic geometry, and the answer is counterintuitive: almost none of it is in the deduction engine.

The DDAR-style forward-chaining engine is, conceptually, the *easy* half — it is a fixed set of geometry rules applied to closure, deterministic and fast. The hard half is the LLM that proposes *auxiliary constructions*, because that is the only part that requires judgment: pure deduction gets stuck exactly when the proof needs a point, line, or circle that is not in the original figure, and deciding *which* construction unsticks *which* configuration is a learned skill with no closed-form answer. Teaching that skill is a data problem, and 230 million synthetic problems is the brute-force answer — generate vast numbers of (figure, construction, solution) triples so the proposer learns the empirical regularities ("configurations with two circles tangent at a point want the radical axis," "this incenter setup wants the contact triangle").

This reframes the whole AlphaGeometry-vs-Seed-Geometry comparison. Seed-Geometry's edge (22/39 vs 19/39 on the hard shortlist) is not a smarter engine or a new paradigm — it is *better construction proposals from more and better-targeted synthetic data*, plus the engine speed that lets it test more proposals per unit time. The lesson generalizes beyond geometry: in a neuro-symbolic system, the symbolic half is usually the part you can *engineer* and the neural half is the part you have to *feed*, and the frontier moves by industrializing the data pipeline for the neural component. 230M problems in seven days is what industrializing it looks like.

### 12. The 2000-line refinement loop in practice

Across the long IMO proofs — P3 at 2000 lines, P4 at 4000 — the depth loop is the unsung workhorse: generate a `have`, get a localized Lean error, re-reason with the CoT in context, retrieve or restructure, re-verify, repeat. The case-study insight, which I want to state as bluntly as possible because it is the most transferable lesson in the post, is that *refinement quality is bounded by reasoning quality, not by Lean error verbosity.*

Lean's error messages are precise about *where* a proof broke — `unsolved goals at line 1842`, `type mismatch`, `nlinarith failed` — and almost silent about *why* in any mathematical sense. The worked example in §3 is the canonical case: `nlinarith failed` does not tell you "you needed Cauchy-Schwarz, not more square hints"; it just reports that the tactic could not close the goal. The bridge from "the tactic failed" to "I need a structurally different argument" is *reasoning*, and it has to come from the model. This is why the same refinement loop wrapped around a weaker base model would refine far worse — it would pattern-match on the error string ("failed → add hints") and loop, instead of re-deriving the mathematical strategy.

This is also the strongest argument for *why the base model choice was load-bearing rather than incidental*. Seed-Prover 1.0 is built on the Seed1.5-Thinking reasoning lineage; Seed-Prover 1.5's natural-language prover is initialized from Doubao-Seed-1.6. Both choices put a strong long-CoT reasoner at the center of the refinement loop on purpose. A 2000-line proof is not 2000 lines of lucky generation; it is dozens of `have` blocks, each surviving several rounds of the micro-cycle, and each round's success depends on the model's ability to *read a terse Lean error and recover the mathematical intent*. The proof length the system can sustain is, in the end, a function of how reliably its base model reasons under refinement — which is why "build the prover on the strongest available reasoning base" is not a detail but the whole game.

## When formal proving helps, and when it does not

### Reach for a Seed-Prover-style system when:

- **You need a guarantee, not a vibe.** If the cost of a wrong "proof" is high — verifying a protocol, a safety property, a critical algorithm invariant — the Lean kernel's pass/fail is worth more than any natural-language argument, no matter how fluent. This is the whole reason the field exists.
- **The domain is algebra, analysis, or number theory.** Seed-Prover is near-human on formalized-IMO algebra (~85%) and strong on number theory (~76%). If your problems live there, the system is genuinely useful today.
- **You can supply a faithful formal statement.** The system proves *formal* theorems. If you (or a formalization pipeline) can write the Lean statement correctly, Seed-Prover can often prove it. The formalization is the human-in-the-loop boundary; budget for it.
- **You have a benchmark that recurs.** The lemma library and retrieval compound over a problem set with shared structure. A one-off proof gets less benefit than a campaign over a coherent body of theorems.
- **The problem is geometry with a clean figure.** Seed-Geometry will likely dispose of it in seconds *once formalized* — far faster and more reliably than forcing it through Lean.

### Skip it (for now) when:

- **The problem is heavily combinatorial.** 7/14 on IMO combinatorics and 30/100 on CombiBench is the honest number. If your problem hinges on a clever bijection or a slick double-counting argument, expect a coin flip at best. This is a research frontier, not a solved capability.
- **You need frontier research mathematics.** The 1.5 authors explicitly acknowledge the system cannot make contributions comparable to human experts, because it cannot synthesize insights across research papers. Fate-X at 33% is the marker. Competition math ≠ research math.
- **You cannot afford the compute on hard problems.** ~10 H20-days per problem (and ~40 for the hardest) is real money. For easy problems the light setting is cheap; for IMO-hard problems the heavy setting is a server-farm-week. Match the setting to the difficulty, and do not run heavy by default.
- **You only need a plausible argument for a human reader.** If a human will read and trust the proof anyway, a strong natural-language reasoning model is faster and cheaper. The formal overhead pays off precisely when you *cannot* trust a human grader — otherwise it is ceremony.
- **The statement is hard to formalize faithfully.** If translating your problem into Lean is itself error-prone or ambiguous, a kernel-checked proof of the *wrong* statement is worse than useless — it launders a formalization bug into false confidence. Get the statement right first.

The throughline: Seed-Prover is the first system where "the model proved it" and "the kernel verified it" are the same sentence for olympiad-level mathematics. That is a real milestone. It is also still bounded — by combinatorics, by the formalization boundary, by compute, and by the gap between contest math and research math. Hold both truths at once.

## Further reading

- Seed-Prover paper: arXiv:2507.23726, "Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving."
- Seed-Prover 1.5 paper: arXiv:2512.17260, "Mastering Undergraduate-Level Theorem Proving via Learning from Experience."
- ByteDance Seed blog and the [Seed-Prover GitHub repo](https://github.com/ByteDance-Seed/Seed-Prover).
- [DeepSeek-Prover-V2 deep-dive](/blog/paper-reading/large-language-model/deepseek-prover-v2-advancing-formal-mathematical-reasoning-via-reinforcement-learning-for-subgoal-decomposition) — the subgoal-decomposition paradigm Seed-Prover argues against.
- [DeepSeek-R1: RL for reasoning](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) — the broader reasoning-RL context.
- [Seed1.5-Thinking and VAPO](/blog/paper-reading/large-language-model/seed1-5-thinking-rl-reasoning-vapo-dapo) — the base model and RL machinery Seed-Prover inherits.
- [The ByteDance Seed model universe](/blog/machine-learning/large-language-model/bytedance-seed-model-universe-by-use-case) — where Seed-Prover sits in the broader family.

---
title: "Kimina-Prover: Large Formal Reasoning Models with RL"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "paper-reading"
subcategory: "Reasoning"
tags:
  - formal-theorem-proving
  - reinforcement-learning
  - lean4
  - reasoning-models
  - autoformalization
  - minif2f
  - kimi-k1-5
description: "How Kimina-Prover replaces tree search with a single RL-trained reasoning model, sets a new miniF2F record at 80.7%, and shows the first clean size-scaling curve for formal provers."
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/kimina-prover-1.png"
readTime: 30
---

For most of the last decade, the recipe for a strong neural theorem prover looked nothing like the recipe for a strong language model. You took a modest model that could score a single tactic, and you wrapped it in an enormous search algorithm — best-first search, or Monte Carlo tree search — that expanded thousands of partial proofs, guided by a value function that scored how promising each branch looked. The model was the cheap part. The search was the expensive part, and it was where the intelligence lived. Worse, when you made the model bigger, the proofs did not get noticeably better. The field had no scaling law: a 7B prover and a 1.5B prover landed in roughly the same place once you matched their search budgets.

That is a strange state of affairs in 2025, when every other corner of machine learning has a scaling law and the whole story of reasoning models is that you let a single model think for longer and it gets smarter. Kimina-Prover is the paper that asks the obvious question: what if we throw away the search, throw away the value function, throw away the process reward model, and just train one large language model to reason its way to a Lean 4 proof the way a human mathematician reasons in a proof assistant — sketching informally, dropping in tactic blocks, checking the structure in their head — and then we scale it the way we scale any reasoning model, with reinforcement learning and a longer thinking budget?

The answer, on the standard miniF2F-test benchmark, is a new state of the art: **80.7% at pass@8192**, up from a prior search-prover best of 70.8%, with no external search, no value function, no process reward model, and no Lean compiler feedback during training or testing. And for the first time in this subfield, the same recipe produces a clean scaling curve across 1.5B, 7B, and 72B parameters. The diagram above is the mental model: a single RL-trained model takes a formalized Lean statement and emits the entire proof end to end, with the old machinery of search and learned critics crossed out and replaced by the model's own reasoning.

![One RL-trained model writes the whole Lean proof with no external search](/imgs/blogs/kimina-prover-1.png)

> [!tldr] TL;DR
> - **What it claims:** A single LLM (RL-trained from Qwen2.5-72B using the Kimi k1.5 pipeline) can write complete Lean 4 proofs end to end via a "formal reasoning pattern," reaching **80.74% on miniF2F-test at pass@8192** — a new SOTA — with no MCTS/BFS, no value function, no process reward model, and no compiler feedback at train or test time.
> - **Why it matters:** It is the first formal prover to show a consistent size-scaling curve (1.5B → 7B → 72B), turning theorem proving from a search-engineering problem into a model-scaling problem, and it is dramatically more sample-efficient than search baselines (pass@1 = 52.94% beats a baseline that spends 102,400 samples).
> - **Most surprising finding:** It proves `imo_1968_p5_1`, an IMO problem no previously released public model had ever solved, using only learned reasoning and a binary reward.
> - **Where it fails:** It is a "Preview"; it excludes geometry and combinatorics from training (poor Lean tooling), it over-uses high-level automation tactics, and it does not yet do iterative compiler-feedback refinement. Early RL is also fragile — it suffers "format collapse" that needs a deliberate stabilization trick to survive.

## Context: what came before

To see why Kimina-Prover is a departure, we have to be precise about the lineage it is breaking from. Formal theorem proving in a system like Lean 4 is the task of producing a sequence of *tactics* — proof commands — that the Lean kernel accepts as a complete, machine-checked derivation of a stated theorem. Unlike informal mathematics, there is no partial credit and no benefit of the doubt: either the proof type-checks or it does not. This is both the appeal (the reward signal is perfectly clean and ungameable) and the difficulty (one wrong tactic and the whole proof is worthless).

The dominant paradigm before Kimina-Prover was **search over a tactic model**. You train a model to predict, given a proof state, the next tactic to try. Then at inference time you run a search algorithm that maintains a frontier of partial proofs and expands the most promising ones. DeepSeek-Prover-V1.5-RL paired with RMaxTS (a Monte Carlo variant) hit 63.5% on miniF2F-test with a budget the paper records as `32×16×400`. InternLM2.5-StepProver with best-first search and a critic got 65.9% at `256×32×600`. HunyuanProver reached 68.4%. BFS-Prover, the prior search SOTA-class system, reached 70.8% at a budget of `2048×2×600`. Those multiplicative budget notations are the giveaway: these systems spend their compute on *breadth of search*, expanding huge trees of partial proofs.

There were also "whole-proof" generators that skip the step-by-step search and instead sample entire candidate proofs and check them. DeepSeek-Prover-V1.5-RL in whole-proof mode reached 60.2% — but at **102,400 samples per problem**. Goedel-Prover-SFT reached 64.7% at 25,600 samples. Leanabell-Prover reached 61.1% at 128. The pattern is the same: throw enormous sample budgets at the problem because each individual sample is weak.

Two structural problems hang over all of this. The dossier states them plainly. First, these search-based provers "introduce substantial computational overhead" — the search is the cost center, not the model. Second, and more damning, "previous neural theorem provers tailored for formal mathematics have generally not demonstrated clear improvements in performance corresponding to increases in model size." There was no scaling law. Making the model bigger did not reliably help, because the model was just a tactic-scorer feeding a search algorithm, and a bigger scorer does not fundamentally change what the search can find.

It is worth dwelling on *why* search-based provers refused to scale, because the explanation is what motivates the entire redesign. In a search-over-tactics system, the model's job is local: given a single proof state, predict a good next tactic. The global structure of the proof — the high-level strategy, the choice of induction variable, the decision to split into cases — emerges from the search algorithm exploring combinations of local tactic predictions, not from any single forward pass of the model. When you scale the tactic model from 1.5B to 7B, you make each local prediction a little better, but the proof you eventually find is still bottlenecked by the *breadth* of the search, which is set by your compute budget, not your model size. A bigger flashlight does not let you see further if the room is the same size and you are still feeling your way wall by wall. The search, not the model, owns the strategy, so scaling the model leaves the strategy untouched. That is the structural reason there was no scaling law, and it is exactly the reason a reasoning model — which produces the *whole* proof in one trace and therefore owns the global strategy itself — can inherit the reasoning-model scaling law.

Meanwhile, on the other side of the fence, general-purpose reasoning models like OpenAI's o3-mini and Gemini 2.5 Pro had learned to reason about mathematics *informally* with remarkable skill. But they "struggle to formalize" — they can sketch a correct informal argument and then fail to translate it into a verifier-checkable Lean proof. The failure is specific and instructive: informal mathematics tolerates leaps ("clearly," "without loss of generality," "by symmetry") that the Lean kernel rejects outright, so a model trained on informal proofs learns to *gloss* exactly the steps a formal proof must spell out in full. The gap Kimina-Prover sets out to fill is exactly here: bridge **formal verification** and **informal mathematical intuition** inside a single model, and do it in a way whose performance scales with model size and with sample budget, the way a reasoning model is supposed to.

## Contributions

The paper's contributions are best read as a set of deliberate removals paired with a few deliberate additions. I will tighten them from the dossier into five claims:

1. **A reasoning-first prover that needs no external search.** Kimina-Prover replaces BFS/MCTS, value functions, and process reward models with a single model that reasons end to end. All proofs are generated "without any prover feedback during training and test." This is the central architectural bet.
2. **A formal reasoning pattern that binds chain-of-thought to the actual proof.** The model thinks inside `<think>…</think>` tokens, interleaving informal reasoning with relevant Lean 4 code snippets ("informal-formal alignment"), under a hard constraint that tactic blocks collectively cover **at least 60%** of the final proof's Lean code. The thinking is not free-floating prose; it is structurally tied to the output.
3. **Large-scale RL from the Kimi k1.5 pipeline with a binary reward.** A policy-gradient objective with a KL constraint to a reference policy, rewarding a fully compiling proof with 1 and everything else with 0 — plus a specific stabilization trick (negative-gradient discarding at probability $\omega = 0.5$) to survive early-RL format collapse.
4. **An autoformalization data engine.** A separate 7B model (Kimina-Autoformalizer) converts informal Numina problems into Lean 4 statements at **90% one-shot compilation and 66% accuracy**, trained by expert iteration with a QwQ-32B judge plus Lean compiler verification, enabling a 200K-problem RL set.
5. **The first size-scaling curve for formal provers, plus open distilled models.** The same recipe produces consistent gains across 1.5B / 7B / 72B, and the 72B is distilled into open 1.5B and 7B variants — the first time a clean model-size scaling law has been shown for theorem proving.

The single most important sentence in the whole paper is the one about scaling. Search-based provers had no scaling law; reasoning models do. By reframing proving as a reasoning task, Kimina-Prover inherits the reasoning-model scaling law. That reframing is the contribution; everything else is the machinery that makes it work.

## Method

The method has four moving parts that fit together: the **data construction** that produces formalized problems, the **formal reasoning pattern** that shapes the model's output, the **large-scale RL** that trains the policy, and the **stabilization trick** that keeps early RL from collapsing. The full pipeline runs left to right from raw Numina problems through autoformalization and a small SFT warm-up into the RL stage, and finally a distillation step.

![From Numina problems to a scaling RL prover in four stages](/imgs/blogs/kimina-prover-2.png)

The diagram above is the mental model for the training recipe: data curation feeds a 20K-example SFT warm-up, which hands off a competent-but-not-great policy to the Kimi-k1.5 RL stage running on 200K Lean statements, and the resulting 72B model is distilled down into the open 7B and 1.5B variants. Let us walk each stage.

### Data construction

The source corpus is **Numina 1.5**, a large dataset of olympiad-style mathematics problems. The authors first filter it to exclude **geometry and combinatorics**, because Lean 4 tooling in those domains is weak and the autoformalizer cannot reliably translate them. The remaining problems are difficulty-rated and difficulty-balanced using **QwQ-32B** as a rating model, so the RL problem set is not dominated by trivially easy or impossibly hard problems — a balanced curriculum matters a lot when your reward is binary and your only learning signal is whether a proof compiles.

The autoformalizer is its own small system worth pausing on. It is a 7B model (**Kimina-Autoformalizer**, based on Qwen2.5-Coder-7B-Instruct) trained by **expert iteration**: generate candidate Lean 4 statements for informal problems, judge their faithfulness with a **QwQ-32B judge**, verify that they compile with the **Lean 4 compiler**, keep the good ones, and retrain. After iteration it reaches **90% one-shot compilation rate and 66% accuracy** (the gap between the two is the share of statements that compile but do not faithfully capture the informal problem). That 90% one-shot compile rate is what makes a 200K-problem RL set economically feasible — you are not throwing away nine out of ten formalization attempts.

The expert-iteration loop deserves a closer look because it is the quiet workhorse of the whole data engine, and the two filters it stacks are doing different jobs. The Lean 4 compiler is a *syntactic and type* gate: it answers "is this even a well-formed Lean statement?" but it says nothing about whether the statement means what the informal problem meant — you can perfectly compile a statement that is true but trivial, or that quietly weakens the hypotheses. That semantic gap is what the QwQ-32B judge closes: it reads the informal problem and the candidate Lean statement and rules on faithfulness. Stacking the two — compiler for form, model-judge for meaning — is what lets the 90% compile rate translate into a usable 66% faithful-accuracy rate, and it is the same two-filter discipline (one mechanical, one learned) that the RL stage will reuse with the Lean compiler and the format filters. Each round of the loop adds the surviving (compiling, faithful) statements back into the training set, so the formalizer gets monotonically better at producing statements that pass both gates, which is precisely the behavior you want from expert iteration on a verifiable target.

The final RL problem set is **200K total problems**, built at a deliberate **1:1 ratio** between autoformalized statements and human-annotated statements. There were only about **10K** human-annotated statements available, so they were resampled to match the size of the autoformalized subset. The intuition behind the 1:1 mix is straightforward: the human-annotated statements are higher quality and keep the model anchored to clean, idiomatic Lean, while the autoformalized statements provide the volume and diversity that RL needs to explore.

Separately, a small **mini-SFT dataset** of about **20K examples** warms up the model before RL. These are combined informal+formal proofs synthesized using **Claude 3.7 Sonnet**, mixed with informal math data from Kimi k1.5. The SFT stage is intentionally small — the whole bet of the paper is that RL, not supervised imitation, does the heavy lifting. But you cannot start RL from a model that has never seen the `<think>` format and the informal-formal interleaving, so the 20K warm-up teaches the *shape* of a good answer, and RL teaches the *content*.

### The formal reasoning pattern

This is the conceptual core, and it is worth being concrete about what it actually constrains. The model's output is wrapped in `<think>…</think>` tokens. Inside the thinking block, the model is trained to interleave two things: *informal* reasoning in natural language (the kind of intuition a human writes on scratch paper), and *relevant Lean 4 code snippets* — actual tactic blocks that will, for the most part, end up in the final proof. The paper calls this **informal-formal alignment**.

![Anatomy of the formal reasoning pattern](/imgs/blogs/kimina-prover-5.png)

The diagram above lays out the anatomy of one output: the Lean statement at the top, then the bounded `<think>` region, which itself contains the informal sketch and the inline Lean snippets, then the coverage filter, then the final compiling proof at the bottom. The design constraint that makes this work — and the part I find genuinely clever — is the **coverage requirement**: "the majority of the Lean 4 code snippets appear in the final proof," operationalized as the rule that **tactic blocks must collectively cover at least 60% of the Lean code in the final proof**. Combined with a filter that each kept sample must contain **at least one tactic block**, this prevents the failure mode where the model writes a beautiful informal essay and then a disconnected proof that has nothing to do with it.

Why does that matter? Because the most natural way for a chain-of-thought model to "cheat" at formal proving is to let the reasoning trace and the final answer drift apart. The reasoning becomes performative — it sounds like thinking but does not actually constrain the output. By demanding that 60% of the final proof's code already appeared (in tactic blocks) inside the reasoning, the training signal forces the reasoning to *be* the proof-construction process, not a narration bolted on afterward. The model is rewarded for thinking in a way that produces the code it ends up using.

Here is what one of these outputs looks like in shape — a worked example of the pattern for proving that the sum of the first $n$ odd numbers is $n^2$:

```lean
<think>
We want to show the sum of the first n odd numbers equals n squared.
The clean way in Lean 4 is induction on n. Base case n = 0 is trivial.
For the step, we assume the sum up to k is k^2 and push through k+1.
Let me draft the skeleton as actual tactics so I can check it compiles
in my head before committing:

    induction n with
    | zero => simp
    | succ k ih =>
      rw [Finset.sum_range_succ, ih]
      ring

The `rw` unfolds the sum at k+1 using the inductive hypothesis ih,
then `ring` closes the arithmetic obligation (k^2 + (2k+1) = (k+1)^2).
That should be the whole proof; nothing exotic is needed here.
</think>

theorem sum_odd (n : Nat) :
    (Finset.range n).sum (fun i => 2 * i + 1) = n ^ 2 := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [Finset.sum_range_succ, ih]
    ring
```

Notice that the tactic block inside `<think>` — the `induction … | succ … rw … ring` skeleton — is almost verbatim the body of the final proof. That is the 60% coverage constraint doing its job. The informal sentences explain *why* each tactic is there, but they are scaffolding around code that actually ships. This is the difference between a model that reasons about Lean and a model that reasons *in* Lean.

### Large-scale RL from Kimi k1.5

The training engine is inherited from **Kimi k1.5**, the large-scale RL pipeline behind Moonshot's reasoning models. (If you want the full machinery of that pipeline — the synchronous master, the rollout/trainer split, the partial-rollout replay buffer — it is worth reading the [Kimi k1.5](/blog/paper-reading/reinforcement-learning/kimi-k1-5) writeup; Kimina-Prover reuses that infrastructure and swaps in a theorem-proving reward.) The reward here is as simple as a reward can be: **binary outcome reward**, 1 for a fully correct proof that compiles in Lean 4, and 0 for everything else. There is no partial credit for "almost compiling," no shaped reward for proof length, no learned reward model. The Lean kernel is the judge, and it is incorruptible.

The objective is a **policy gradient with a KL-divergence constraint** to a reference policy (the paper's Eq. 1). Concretely, let $\pi_\theta$ be the current policy and $\pi_{\text{ref}}$ the reference. For a problem $x$ with sampled proof $y$ and binary reward $r(x, y) \in \{0, 1\}$, the update pushes up the log-probability of rewarded proofs while a term proportional to the KL coefficient $\tau$ penalizes drifting too far from $\pi_{\text{ref}}$:

$$
\mathcal{L}(\theta) = -\,\mathbb{E}_{x,\,y \sim \pi_\theta}\big[\, r(x, y) \,\big] + \tau \cdot \mathrm{KL}\!\left(\pi_\theta \,\|\, \pi_{\text{ref}}\right)
$$

The KL term is the regularizer that keeps the model from collapsing into a narrow, degenerate distribution — a constant risk when your reward is sparse and binary. The hyperparameters are spelled out in the dossier and they are unusually conservative:

| Hyperparameter | Symbol | Value | Why it matters |
|---|---|---|---|
| Learning rate | — | $2 \times 10^{-6}$ (constant) | Tiny and fixed — RL on a 72B model is fragile; no schedule, just a small steady step. |
| KL coefficient | $\tau$ | $0.4$ | Keeps the policy anchored to the reference; sparse binary reward invites collapse without it. |
| Negative-gradient discard prob. | $\omega$ | $0.5$ | Drops half of the failed-proof samples to stop early-RL format collapse (see below). |
| Rollout problems / iteration | $N$ | $1000$ | Breadth of the exploration frontier per update. |
| Candidate solutions / problem | $k$ | $8$ | Multiple attempts per problem give the gradient something to compare. |
| Final-proof coverage filter | — | $\geq 60\%$ | Each kept sample must reuse $\geq 60\%$ of its tactic code in the final proof. |

Each RL iteration samples $N = 1000$ problems, draws $k = 8$ candidate proofs per problem, compiles each one in Lean to get the binary reward, applies the format filters, and takes a single small gradient step. The whole loop is shown below.

![The RL loop and the trick that stops format collapse](/imgs/blogs/kimina-prover-6.png)

The diagram above is the mental model for one RL iteration: a problem batch fans out into $k=8$ rollouts, each is compiled to a binary reward, the samples split by reward sign, and — critically — the negative-gradient (failed) samples are filtered before merging back into a single policy update. The split-by-sign step is where the next piece comes in.

### Stabilizing RL: the format-collapse trick

Here is the part that does not show up in the headline numbers but is, I suspect, the difference between this recipe working and not working. The dossier states it directly: "limited SFT data and formal structure cause early RL format collapse from negative gradients."

The mechanism is this. Early in RL, the model is bad at proving, so most of its $k = 8$ samples per problem fail and get reward 0. A policy gradient on a failed sample is a *negative* gradient — it pushes down the probability of everything in that sample, including the *format* (the `<think>` structure, the tactic blocks, the informal-formal interleaving) that the SFT stage worked hard to install. Because the SFT data is small (20K) and the formal structure is rigid, the model has not deeply internalized the format, so these early negative gradients erode it faster than the rare positive gradients can reinforce it. The model "forgets how to write a well-formed answer" before it ever learns to write a *correct* one. That is format collapse.

The fix is deliberately blunt: **randomly discard samples with negative gradients with probability $\omega = 0.5$**. Half the failed samples never contribute to the update. This halves the downward pressure on the format while leaving the positive signal untouched, buying the model enough stability to keep emitting well-formed answers until it starts producing correct ones often enough for RL to take off. On top of the discard, the format filters (at least one tactic block, $\geq 60\%$ coverage) act as a quality gate on what is allowed into the gradient at all.

A clean with/without quantitative ablation table for this trick was not surfaced in the read, so I will be honest: the paper motivates $\omega = 0.5$ as a stabilization mechanism rather than proving its magnitude with a controlled sweep. We have the *what* and the *why*; we do not have a published number for "accuracy with $\omega = 0$ vs $\omega = 0.5$." That is a gap worth flagging.

In rough pseudocode, the kept-sample selection looks like this:

```python
def select_training_samples(rollouts, omega=0.5, coverage_min=0.60):
    """Filter one iteration's rollouts before the policy update.

    rollouts: list of (problem, proof_text, reward in {0, 1}).
    Keeps every rewarded sample; keeps a rewarded-zero sample only if it
    survives the random discard AND passes the format filters.
    """
    kept = []
    for problem, proof, reward in rollouts:
        if not has_tactic_block(proof):
            continue                      # format filter: >= 1 tactic block
        if final_proof_coverage(proof) < coverage_min:
            continue                      # format filter: >= 60% reuse
        if reward == 1:
            kept.append((problem, proof, reward))     # positive: always keep
        else:
            if random.random() >= omega:              # negative: drop w.p. omega
                kept.append((problem, proof, reward))
    return kept
```

The `final_proof_coverage` function is the operationalization of informal-formal alignment: it measures the fraction of the final proof's Lean code that already appeared as tactic blocks in the reasoning, and rejects anything under 60%.

### Architecture and context length

The base model is **Qwen2.5-72B** (dense, no MoE), with distilled **1.5B** and **7B** variants. The autoformalizer is a separate **Qwen2.5-Coder-7B-Instruct**. The target language is Lean 4. The exact layer count, attention configuration, and head layout are not stated beyond the Qwen2.5 lineage; the dossier marks these as not reported, and I will not invent them.

One architectural number does stand out: the **context length is 32K tokens**, described as the longest context used in neural theorem proving to date. This is not incidental. The whole point of the formal reasoning pattern is that the model writes a long interleaved trace — informal reasoning plus tactic blocks — before committing to the final proof. The output-length scaling result (proofs grow from ~2,500 to over 10,000 tokens as training proceeds) means the model genuinely uses that context budget. A short context would cap the reasoning and, by the paper's own scaling evidence, cap the accuracy.

Here is the architecture comparison that captures the shift from the search paradigm to the reasoning paradigm:

| Dimension | Search-based provers | Kimina-Prover |
|---|---|---|
| Inference-time engine | BFS / MCTS over partial proofs | Single forward generation of full proof |
| Learned critics | Value function + process reward model | None (binary outcome reward only) |
| Compiler in the loop | Often used to expand/score nodes | Not used at train or test time |
| Sample/search budget | e.g. $2048\times2\times600$ (BFS-Prover) | pass@k whole proofs (e.g. pass@32) |
| Context length | Short (per-step states) | 32K tokens (full reasoning trace) |
| Size scaling | Flat — no clear law | 1.5B → 7B → 72B monotone improvement |

The contrast in the table is the contrast the whole paper is built around, and it is worth visualizing as a before/after as well.

![Replacing tree search with learned reasoning](/imgs/blogs/kimina-prover-3.png)

The diagram above is the before/after: on the left, the old stack of an explicit search algorithm, a value function plus PRM to guide it, and a model that did not scale with size; on the right, a single policy that reasons without MCTS or BFS, a binary reward with no learned critics, and a model that scales monotonically from 1.5B to 72B (61.9% → 77.87% at pass@1024).

## Experiments

The headline experiment is miniF2F-test, the standard benchmark of formalized competition mathematics. The result matrix below is the one to anchor on.

![miniF2F-test: Kimina sets a new SOTA at a fraction of the budget](/imgs/blogs/kimina-prover-4.png)

The diagram above is the results matrix: each row is a system, the columns are size, sample budget, and miniF2F score, and the green bottom row is Kimina-Prover-Preview at 72B reaching 80.74% at pass@8192. Here is the full Table 1 reproduced with exact numbers from the dossier — note that the "Sample Budget" notation is reproduced verbatim from the source, including the multiplicative search budgets:

| Model | Size | Sample budget | miniF2F-test |
|---|---|---|---|
| DeepSeek-Prover-V1.5-RL + RMaxTS | 7B | $32\times16\times400$ | 63.5% |
| InternLM2.5-StepProver-BF+CG | 7B | $256\times32\times600$ | 65.9% |
| HunyuanProver v16+BFS+DC | 7B | $600\times8\times400$ | 68.4% |
| BFS-Prover | 7B | $2048\times2\times600$ | 70.8% |
| DeepSeek-Prover-V1.5-RL (whole-proof) | 7B | $102400$ | 60.2% |
| Goedel-Prover-SFT | 7B | $25600$ | 64.7% |
| Leanabell-Prover | 7B | $128$ | 61.1% |
| **Kimina-Prover-Preview-Distill-1.5B** | 1.5B | pass@1 / 32 / 1024 | 42.6% / 56.2% / 61.9% |
| **Kimina-Prover-Preview-Distill-7B** | 7B | pass@1 / 32 / 1024 | 52.5% / 63.1% / 70.8% |
| **Kimina-Prover-Preview** | 72B | pass@1 / 8 / 32 / 1024 / 8192 | 52.94% / 65.16% / 68.85% / 77.87% / **80.74%** |

Three things in this table are load-bearing. First, the **80.74%** at pass@8192 is the new SOTA, beating the prior search-prover best of 70.8%. (The abstract rounds it to 80.7%; the table reads 80.74%.) Second, the **sample efficiency** is the quietly more impressive result: at **pass@1**, the 72B model already scores **52.94%**, and at **pass@32** it scores **68.85%** — that pass@32 number beats DeepSeek-Prover-V1.5-RL's whole-proof 60.2%, which spent **102,400 samples per problem**. Kimina-Prover is competitive at three orders of magnitude lower sample budget. Third, the **7B distill ties BFS-Prover** (70.8%) at pass@1024, meaning a 7B reasoning model with no search matches the prior search SOTA.

### The scaling curve

The scaling story is the one the field had been missing, so it deserves its own look. At pass@1024, the three sizes land at:

| Model size | miniF2F-test (pass@1024) |
|---|---|
| 1.5B (distill) | 61.9% |
| 7B (distill) | 70.8% |
| 72B | 77.87% |

That is a clean, monotone curve — the first one reported for formal provers. And the dossier notes that the **72B advantage over the 7B distill widens with sample budget**: roughly +0.44%, +5.75%, +7.87% as the sampling budget increases. The interpretation is that the bigger model's per-sample distribution is genuinely better, so as you draw more samples the bigger model's lead compounds rather than washing out. This is exactly the behavior you want from a reasoning model and exactly the behavior search-based provers never showed.

It is worth being concrete about why a widening gap is the *good* kind of scaling result and not an artifact. If two models had the same per-sample success probability $p$ on a given problem, then drawing more samples would push both of their pass@k rates toward 1 at the same rate, and the gap between them would *shrink* toward zero as $k$ grows — the weaker model would simply catch up given enough tries. The fact that the 72B's lead instead *grows* from +0.44% at low budget to +7.87% at high budget means the two models are not solving the same set of problems with different luck; the 72B is solving problems the 7B essentially never solves no matter how many samples it draws. Those are problems where the 7B's per-sample probability is effectively zero and the 72B's is small-but-positive, so only the bigger model's pass@k curve ever lifts off the floor for them. That is a qualitative capability difference, not a sampling-variance difference, and it is the strongest single piece of evidence that scaling the model — not just the budget — is buying real proving ability.

### General-purpose reasoning models

The second experiment (Table 2) compares against general-purpose reasoning models at **pass@32**, across miniF2F and the harder IMO and AIME subsets:

| Model (pass@32) | miniF2F | IMO subset | AIME subset |
|---|---|---|---|
| OpenAI o3-mini | 24.59% | 0% | 6.67% |
| Gemini-2.5-pro-preview | 37.70% | 5% | 13.33% |
| **Kimina-Prover-Preview** | **68.85%** | **20.00%** | **46.67%** |

The gap here is stark and it is the empirical justification for the whole premise. o3-mini and Gemini 2.5 Pro are strong informal reasoners, but on *formal* proving they score 24.59% and 37.70% on miniF2F respectively, and on the IMO subset o3-mini scores **0%**. Kimina-Prover, a specialized formal reasoner, scores 68.85% / 20.00% / 46.67%. The general models can reason; they cannot reliably *formalize*. That is precisely the gap the paper set out to close, and it closes it by a wide margin.

The most quotable single result: Kimina-Prover proves **`imo_1968_p5_1`**, "an IMO problem that no previous publicly released model has ever solved," and also solves all five IMO problems that prior methods had solved. A binary reward and learned reasoning, with no search, cracked a problem that the search-prover lineage never had.

### What is load-bearing, and what might not transfer

I want to be careful about what these numbers do and do not establish. The benchmark is miniF2F-test, which is competition mathematics — algebra, number theory, inequalities — and the training data **explicitly excludes geometry and combinatorics** because Lean tooling there is weak. So the 80.7% is a result about the domains where Lean tooling is good and the autoformalizer works. It does not tell us how the recipe does on geometry, on combinatorics, or on research-level mathematics with sparse library support. The strong IMO and AIME subset numbers are encouraging, but those are still within the algebraic/number-theoretic comfort zone.

The other load-bearing factor is the **autoformalizer quality**. The 200K RL set exists because a 7B model can produce a compiling, faithful Lean statement 90% / 66% of the time. In a domain where autoformalization is harder — where the 90% compile rate drops to 40% — the entire data engine becomes uneconomical, and with it the RL. The recipe is not domain-agnostic; it is as good as your formalizer. That is the part I would watch most closely when anyone claims this transfers to a new area of mathematics.

## Critique

Let me start with what is genuinely strong, because there is a lot of it.

**What's strong.** The core scientific claim — that reframing proving as reasoning gives you a scaling law the field never had — is supported by the cleanest evidence available: three model sizes, one recipe, monotone improvement, and a gap that widens with sample budget. That is hard to argue with. The sample efficiency is the unsung win: matching or beating baselines at three orders of magnitude fewer samples is a real, practical result, not a benchmark artifact. And the binary-reward-only, no-search, no-critic design is a strong falsifiable bet that paid off — the simplicity is a feature, because every removed component is one fewer thing that can be over-tuned or that can break in deployment. The `imo_1968_p5_1` solve is the kind of concrete, hard-to-game evidence that a result is more than a number.

**What's weak or unfalsifiable.** The format-collapse trick ($\omega = 0.5$) is described as load-bearing but not quantified — there is no published with/without table, so we cannot tell whether $\omega = 0.5$ is the sweet spot or just *a* value that worked. The same goes for the 60% coverage threshold: why 60% and not 50% or 70%? These are presented as design choices with motivations, not as swept hyperparameters with curves, and a skeptic is right to ask whether the recipe is robust to them or sensitive in ways that would not transfer. The paper is also a "Preview," which is an honest disclaimer but also means the full ablation rigor is not here yet.

**What ablation is missing.** Three things I would want and do not have. (1) A controlled ablation of the negative-gradient discard: accuracy as a function of $\omega \in \{0, 0.25, 0.5, 0.75\}$, to confirm it is the mechanism and not a coincidence. (2) An ablation of the SFT-warm-up size: does RL recover from 5K SFT examples, or does the whole thing depend on the 20K warm-up to install the format? (3) A pass@k cost analysis against the search baselines in *wall-clock* or *FLOPs*, not just sample count — pass@8192 of a 72B model is not free, and the fair comparison to a search prover is total compute, which the sample-budget columns obscure.

**What would change my mind.** If a careful reimplementation showed that you can hit comparable miniF2F-test numbers *without* the formal reasoning pattern — i.e., that a plain whole-proof RL model with the same Kimi-k1.5 pipeline and binary reward gets to ~78% — then the "formal reasoning pattern" would be revealed as decoration rather than the mechanism, and the paper's central conceptual claim would weaken to "big RL on Lean works." Conversely, if removing the 60% coverage constraint causes the informal-formal alignment to collapse and accuracy to crater, that would confirm the pattern is doing real work. I would also change my mind about transferability if someone showed the recipe working on geometry or combinatorics with a comparably cheap formalizer — that would prove the domain exclusions are an artifact of current tooling, not a fundamental limit.

## What I'd build with this

Concrete extensions I would actually pursue if I had this model and recipe in hand:

1. **Iterative compiler-feedback refinement.** The paper explicitly flags this as future work: the preview does *not* use multi-turn error-fixing with the Lean compiler. I would build a refinement loop where a failed proof's compiler error is fed back into the model's context for a second attempt. The binary reward stays clean; you just give the model the kernel's complaint and let it reason about the fix. This is the single highest-leverage extension because it adds a cheap, ungameable feedback signal the current system throws away.
2. **A high-level-tactic penalty.** The authors note the model over-uses high-level automation tactics (things like `nlinarith` or `decide` that hide the reasoning). I would add a lightweight reward shaping or output filter that penalizes proofs which lean on automation, to push the model toward proofs whose structure is legible and whose `<think>` trace genuinely tracks the proof. This trades a little pass-rate for a lot of proof quality and pedagogical value.
3. **Autoformalizer co-training for new domains.** Since the whole data engine is bottlenecked by the 7B autoformalizer's 90%/66% rates, I would invest in a stronger formalizer specifically for geometry/combinatorics — even a domain-specific one — to unlock the excluded domains. The expert-iteration recipe (QwQ-32B judge + Lean compiler) is reusable; it just needs domain-appropriate seed data and Lean tooling.
4. **Distilling the reasoning trace into a tactic-completion tool.** The `<think>` traces are a goldmine of informal-formal alignment data. I would distill them into a smaller, faster model that does inline tactic suggestion inside an editor — bringing the reasoning model's intuition to interactive theorem proving at IDE latencies, where pass@8192 is not an option but a single strong suggestion is gold.
5. **Cross-pollination with the broader Kimi reasoning stack.** The RL engine is Kimi k1.5; the natural next step is to push the same formal-reasoning reward into the larger [Kimi K2 Thinking](/blog/paper-reading/large-language-model/kimi-k2-thinking) line and see whether a much bigger general reasoner, given the formal reasoning pattern, extends the scaling curve past 72B and past 80.7%.

## When to reach for a reasoning-first prover (and when not to)

**Reach for this approach when** your domain has good Lean tooling (algebra, number theory, inequalities), when you can build or already have a cheap, reliable autoformalizer, and when you care about *per-sample* quality and proof legibility rather than just raw pass-rate at any cost. The sample efficiency is the practical reason to prefer it: if you are paying for compute, a model that hits 68.85% at pass@32 is enormously cheaper to run than a search prover that needs 100K samples, and the proofs it produces are human-readable reasoning traces, not opaque search outputs. It is also the right choice if you believe in scaling — this is the only formal-proving recipe with a demonstrated size-scaling law, so it is the one that will keep improving as you scale the base model.

**Do not reach for it when** you are in a domain Lean handles poorly (geometry, combinatorics, anything where autoformalization falls below ~80% compile rate) — there the data engine collapses and you would be better served by domain-specific tooling or a search prover with hand-built tactics. Do not reach for it if you need a guarantee on a *single* attempt at very low budget and cannot afford even pass@8 — pass@1 is 52.94%, which is strong for the field but still a coin flip on individual hard problems. And do not assume the 80.7% transfers to your benchmark: it is a miniF2F-test result in well-supported domains, and the honest reading is that this is a powerful, sample-efficient, scaling-friendly prover *for the mathematics Lean is good at* — which is a large and important slice, but not all of mathematics. For the rest, the compiler-feedback and autoformalizer extensions above are the bridge, and they are not built yet.

## References

- **Paper (arXiv abstract):** [Kimina-Prover Preview: Towards Large Formal Reasoning Models with Reinforcement Learning](https://arxiv.org/abs/2504.11354) (arXiv:2504.11354, April 15, 2025)
- **Code & models (GitHub):** [MoonshotAI/Kimina-Prover-Preview](https://github.com/MoonshotAI/Kimina-Prover-Preview)
- Related reading on the RL engine this work inherits: [Kimi k1.5: Scaling Reinforcement Learning with LLMs](/blog/paper-reading/reinforcement-learning/kimi-k1-5)
- The reasoning-model line this prover descends from: [Kimi K2 Thinking: An Open-Source Reasoning Model Built on K2](/blog/paper-reading/large-language-model/kimi-k2-thinking)
- A complementary formal-math benchmark on combinatorics (a domain Kimina excludes): [CombiBench: Benchmarking LLMs on Combinatorial Mathematics](/blog/paper-reading/reasoning/combibench)
- The agentic-coding sibling in the Moonshot lineage: [Kimi-Dev: Agentless Training as Skill Prior for SWE Agents](/blog/paper-reading/large-language-model/kimi-dev)

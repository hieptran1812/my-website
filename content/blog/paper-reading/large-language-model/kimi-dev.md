---
title: "Kimi-Dev: Agentless Training as Skill Prior for SWE Agents"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "paper-reading"
subcategory: "Large Language Model"
tags:
  - kimi-dev
  - swe-bench
  - agentless
  - swe-agent
  - reinforcement-learning
  - skill-prior
  - code-generation
  - moonshot-ai
  - paper-reading
description: "A close read of Kimi-Dev, the 72B open coding model that uses reasoning-intensive Agentless training to hit 60.4% on SWE-bench Verified and then transfers those skill priors into a competitive SWE-Agent with ~64x less SFT."
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/kimi-dev-1.png"
readTime: 30
---

There is a quiet schism running through every team that tries to make a language model fix real software bugs. On one side sit the **SWE-Agent** frameworks: you hand the model a shell, a file editor, and a directive — *resolve this GitHub issue* — and let it loose for fifty or a hundred turns of free interaction. The ceiling is high because the model can do whatever a developer would do: grep the codebase, run the test suite, read a stack trace, edit, re-run. The problem is that the reward arrives only at the very end, after a long horizon of decisions, and is a single bit: did the final patch pass the hidden tests or not. Try to reinforcement-learn that signal from scratch and you are asking the optimizer to assign credit across a hundred turns from one noisy outcome. It mostly does not work; the policy collapses.

On the other side sit the **Agentless** workflows. Instead of free interaction, you decompose issue resolution into a fixed, verifiable pipeline — localize the buggy files, then edit them — and run each stage exactly once. Every stage is short, every stage is checkable, and the whole thing is friendly to reinforcement learning because the credit-assignment horizon is tiny. The cost is rigidity: the pipeline is a human's guess at how to fix a bug, and it cannot improvise when the bug does not fit the template. The conventional wisdom has been that you pick a side. Agentless gives you trainability and a respectable score; SWE-Agent gives you headroom and a frontier score, if you can afford the proprietary model behind it.

Kimi-Dev, from the Moonshot AI / Kimi team (arXiv [2509.23045](https://arxiv.org/abs/2509.23045), submitted to ICLR), argues the schism is false. Its thesis is that reasoning-intensive Agentless training does not just produce a good Agentless model — it *induces skill priors*, the atomic capabilities of bug localization, code editing, and self-reflection, and those priors are exactly what a SWE-Agent needs to be trainable. So you do not choose. You spend your training budget building an Agentless model first, and that model becomes a skill-rich initialization that a tiny supervised pass turns into an efficient agent. The Agentless branch reaches **60.4% on SWE-bench Verified** — state of the art among workflow methods and open-source models. The agent branch, after a light SFT on 5,016 trajectories, reaches **48.6% pass@1**, on par with Claude 3.5 Sonnet, using about 64x fewer SFT tokens than a from-base initialization would need.

![The Kimi-Dev training pipeline: a base model is mid-trained, cold-started, RL-tuned, and then branches into an Agentless model and a SWE-Agent](/imgs/blogs/kimi-dev-1.png)

The diagram above is the mental model: a single base model, Qwen2.5-72B-Base, flows left-to-right through mid-training, cold-start SFT, and outcome-based RL, and then the *same* checkpoint forks. One fork adds test-time self-play and ships as the Agentless Kimi-Dev that scores 60.4%. The other fork takes a small detour through SWE-Agent SFT and emerges as a 48.6% agent. The entire rest of this article is an elaboration of that one picture: what each training stage is actually doing to the model, why the order matters, how the skill priors get measured, and how well the whole thing holds up under scrutiny.

> [!tldr] TL;DR
> - **What it claims:** Reasoning-intensive Agentless training (mid-training, cold-start SFT, outcome RL on the edit stage, and execution-based test-time self-play) induces transferable *skill priors* — localization, code edit, self-reflection — that make a 72B open model both a state-of-the-art workflow solver (**60.4%** SWE-bench Verified) and an efficient initialization for a SWE-Agent (**48.6%** pass@1).
> - **Why it matters:** It dissolves the Agentless-versus-Agent tradeoff. The same checkpoint serves both paradigms, and the agent fork reaches the from-base initialization's best pass@1 with roughly **64x fewer** SFT tokens, turning a months-long agent-RL problem into a 5,016-trajectory SFT.
> - **Most surprising finding:** A mid-trained-only prior, given to SWE-Agent RL from scratch, **collapses below 2% pass@1 in ten RL steps**, while the RL-trained Agentless prior keeps improving *past 70 agent turns*. The skill prior is not a nice-to-have; it is the difference between training and not training at all.
> - **Where it fails:** The TestWriter half of the system has **false positives** from thin reproduction coverage, so test-time self-play stays below the pass@N you would get with ground-truth tests. The reflection-skill measurement is coarse, GPU-hours and optimizer details are not reported, and the transformer internals are simply inherited from Qwen2.5-72B and never enumerated.

## Context: what came before

SWE-bench Verified is the benchmark that organizes this whole field. It is a curated set of real GitHub issues from real Python repositories, each paired with the maintainer's actual fix and a hidden set of unit tests; a model "resolves" an instance when its generated patch makes the failing tests pass without breaking the passing ones. It is hard in the way real software is hard: the model has to find the right files in a repository it has never seen, understand the issue's intent, and produce a syntactically and semantically correct diff. The leaderboard has become the de facto measure of whether a model can do software engineering rather than just autocomplete.

Two solution families grew up around it. The **Agentless** family — the lineage runs through systems that decompose resolution into localize-then-repair-then-validate stages — treats issue resolution as a structured workflow. Its great virtue is that each stage is a short, single-turn, independently verifiable task, which makes it tractable to train and cheap to run. Prior Agentless systems often leaned on a single root-level test as the validation signal, which throws away repository context. Its vice is rigidity: the workflow is fixed, so the model cannot recover from a mislocalization by going back and looking somewhere else.

The **SWE-Agent** family takes the opposite bet. Give the model a real environment — a terminal, a file viewer, the ability to run tests — and let it interact freely over many turns. This is how the frontier proprietary results are produced; Claude 4.0 Sonnet in a SWE-Agent harness reaches 72.7%, GPT-5 reaches 74.9%. The ceiling is clearly higher because the model is not boxed into one developer's idea of the workflow. But the training story is grim. The outcome reward is sparse — one bit after a long, branching trajectory — so naive outcome RL from a non-specialized base struggles with long-horizon credit assignment and tends to collapse. Most strong open SWE-Agents are therefore *distilled*: they SFT on large piles of trajectories collected from a stronger proprietary teacher, which is expensive and caps the student at the teacher's behavior.

The gap Kimi-Dev fills is the bridge between these two. The paper's claim is that the skills you build while training a good Agentless model — knowing how to find the buggy file, how to write a minimal correct diff, when to stop and reconsider — are *the same skills* a SWE-Agent needs, and that an Agentless RL run is a far more sample-efficient way to instill them than agent RL from scratch. Instead of training the agent's skills inside the agent's hard optimization problem, you train them inside the Agentless easy one, then transfer. That reframing — Agentless training as a *skill prior* rather than as an end in itself — is the contribution.

## Contributions

The paper makes five contributions that I would rank in this order.

1. **The skill-prior framing and its empirical payoff.** The central, falsifiable claim is that reasoning-intensive Agentless training produces a prior that transfers to SWE-Agents far more efficiently than alternatives. Concretely, the RL-trained Agentless prior reaches the from-base prior's best agent pass@1 with about **64x fewer SFT tokens** ($2^{23}$ versus $1.5 \times 2^{28}$), and it keeps improving where the others plateau.

2. **A reasoning-intensive Agentless recipe that is itself SoTA.** Even ignoring transfer, the Agentless Kimi-Dev hits **60.4%** on SWE-bench Verified, the best among workflow methods and open-source models, ahead of SWE-SWISS 32B (58.2%), DeepSeek-R1-0528 671B (57.6%), and MiniMax-M1 456B (56.0%).

3. **The BugFixer / TestWriter duo with execution-based self-play.** Framing resolution as two cooperating roles — one that writes the fix, one that writes the test — and then using a Docker-execution scoring function to select the final patch from a 40x40 grid of candidates. This selection beats BugFixer majority voting consistently and lifts the resolve rate from 48.0% to 60.4% as the grid scales.

4. **A two-stage minimal Agentless workflow** — localize, then edit — for *both* roles, which captures repository context and mirrors real developer workflows better than prior single-test Agentless systems.

5. **A concrete, reproducible-in-spirit training pipeline** with public 72B weights, including the design choices that matter: RL on the edit stage only, no KL or entropy regularization, adaptive prompt curriculum, and positive-example reinforcement late in training.

## Method

The method has two layers. The outer layer is the *training pipeline* — the sequence of stages a base model passes through. The inner layer is the *task formulation* — how an issue becomes a learnable, rewardable problem. I will start with the task formulation, because the pipeline only makes sense once you see what it is optimizing.

### The BugFixer / TestWriter duo

Kimi-Dev frames every issue as a job for two roles. The **BugFixer** reads the issue and the repository and produces a patch — a unified diff — that it believes fixes the bug. The **TestWriter** reads the same inputs and produces a *reproduction test*: a unit test that should **fail** on the unfixed code and **pass** once the bug is fixed. The two roles are symmetric in machinery but asymmetric in purpose. The BugFixer is the thing you ship; the TestWriter is the thing that lets you *select* among many BugFixer attempts without peeking at the ground-truth tests.

![The BugFixer and TestWriter roles branch from the issue, share a localize-then-edit workflow, and merge at a Docker execution check](/imgs/blogs/kimi-dev-2.png)

The figure above shows why this is a graph and not a pipeline: the GitHub issue branches into the two roles, both roles flow through the *same* two-stage skeleton, and their outputs merge back at the Docker execution check that produces the binary reward. A resolution is successful when the BugFixer's patch passes the issue's tests. Both roles use exactly two core skills, and naming them is the whole point of the paper:

- **File localization** — given a repository snapshot and an issue, identify which files (and ultimately which lines) are relevant. The dataset curation constrains this to be tractable: repositories with at least five GitHub stars, Python-focused (`.py`, `.md`, `.rst`), and pull requests touching at most five Python files.
- **Code edit** — given the localized context, emit the actual change. For the BugFixer this is the fix diff; for the TestWriter this is the new test file.

That is the entire task surface. There is no tree search, no separate "repair planner," no learned tool-use policy. The bet is that if you can do localization and editing well, and you can reflect on your own attempts, you have the atoms from which both a workflow solver and an agent can be assembled.

### The two-stage Agentless workflow

Both roles run the same minimal workflow: **(1) File Localization, then (2) Code Edits.** Two stages, run once each. This is deliberately more than the single-step "edit the file the test points at" that earlier Agentless systems used, and deliberately less than the open-ended exploration of a SWE-Agent. The two-stage design captures repository context — the model sees the structure of the repo during localization and carries that into editing — while staying short enough that each stage's output is a single, gradeable artifact.

Why does staging matter for trainability? Because it lets you put the reinforcement-learning pressure exactly where it is needed. After mid-training and cold-start, localization is already strong: the model reliably finds the right files. The hard, high-variance part is the edit. So the RL stage trains the *edit* and leaves localization frozen — a much smaller, lower-variance optimization problem than trying to RL the whole trajectory at once. I will come back to this; it is one of the most important design decisions in the paper.

### Mid-training: injecting skill priors

The pipeline begins with Qwen2.5-72B-Base — a dense 72B model, not a mixture-of-experts — and runs roughly **150B tokens** of mid-training. The goal is not to teach facts; it is to teach the model *how human developers reason over GitHub issues, implement fixes, and write tests*. The data, drawn from millions of GitHub issues and PR commits, breaks down as:

- **~50B tokens** in Agentless form, derived from the natural diff patch — the localize-then-edit structure, reconstructed from real PRs.
- **~20B tokens** of curated PR commit packs — coherent sequences of commits that show a fix evolving.
- **~20B tokens** of synthetic data with reasoning and agentic interaction patterns, **upsampled 4x** during training so it punches above its raw weight.

The three named components sum to about 90B; the report states the total is ~150B, and the remainder beyond these three is implied by the upsampling and is not fully enumerated in the main text. (I flag this because it is the kind of gap a reader should not paper over: the exact composition of roughly 60B of mid-training tokens is not stated.) Crucially, the data is **strictly decontaminated** to exclude any repository that appears in the SWE-bench Verified test set, which is the difference between a real result and a leaked one. Mid-training hyperparameters that are stated: global batch size 256, maximum sequence length 32K tokens, run until ~150B tokens are processed.

The ablation on this stage is the cleanest evidence that mid-training is doing real work. Taking the mid-trained Qwen2.5-72B at 50B, 100B, and ~150B tokens, and lightly activating each with the *same* 2,000 BugFixer pairs, the BugFixer pass@1 (temperature 1) improves **monotonically** with more mid-training tokens. The per-budget numbers live only in the figure, but the trend is unambiguous: more skill-prior injection, higher downstream pass rate. The skill prior is not a metaphor; it is a measurable, dose-dependent quantity.

### Reasoning cold-start SFT

Mid-training teaches the model the *shape* of developer reasoning; cold-start SFT teaches it to *produce long, deliberate reasoning trajectories* on demand. The data is small and pointed: **2,000 BugFixer input-output pairs** with long chain-of-thought trajectories, built on SWE-Gym and SWE-bench-extra, and generated by **DeepSeek-R1 (the 20250120 version)** acting as both BugFixer and TestWriter. The trajectories are chosen to activate four behaviors that recur throughout the rest of training: problem analysis, method sketching, self-refinement, and alternative-solution exploration.

Two thousand pairs is a tiny SFT set by any standard, and that is the point. It is a *cold start* — just enough to switch the model from terse completion mode into deliberate reasoning mode, so that the subsequent RL has something to optimize. The heavy lifting on capability was already done in mid-training; the cold start is the activation key.

### Outcome-based RL on the edit stage

Now the model can localize and can reason. RL sharpens the edit. The reward is the rawest signal available — the **Docker execution outcome, 0 or 1** — and the algorithm is the Kimi k1.5 policy optimization: a simplified policy gradient based on REINFORCE (Williams 1992), using the average reward over multiple rollouts as a baseline to normalize returns, which is the GRPO-style normalization without a separate critic. The runs use **no KL and no entropy regularization** and a batch size of **256**, at a fixed **64K-token** training context.

The reward definitions are worth stating precisely, because the asymmetry between the two roles is where the subtle bugs live:

- **BugFixer reward** is +1 if and only if the generated patch passes **all** ground-truth unit tests.
- **TestWriter reward** is +1 if and only if (i) the predicted test raises a failure **without** the ground-truth fix applied, **and** (ii) that failure is resolved **once** the fix is applied.

The TestWriter reward is a fail-then-pass condition: a good reproduction test must catch the bug *and* not be a flaky always-failing test. Here is the edit-stage RL loop in pseudocode, to make the moving parts concrete:

```python
def rl_step(policy, prompts, run_tests_in_docker, n_rollouts=16):
    """One outcome-RL step on the code-edit stage only.

    Localization is frozen after mid-training + cold-start; we only
    reinforce the edit. Reward is the raw 0/1 Docker outcome. We use a
    multi-rollout mean-reward baseline (GRPO-like), no KL, no entropy.
    """
    batch_loss = 0.0
    for prompt in prompts:
        # Sample n candidate edits given the (already localized) context.
        rollouts = [policy.sample(prompt) for _ in range(n_rollouts)]
        rewards = []
        for edit in rollouts:
            patch = apply_edit(prompt.repo_snapshot, edit)
            rewards.append(1.0 if run_tests_in_docker(patch).all_pass else 0.0)

        baseline = sum(rewards) / len(rewards)        # mean-reward baseline
        if baseline in (0.0, 1.0):
            continue                                  # no signal: skip prompt

        for edit, r in zip(rollouts, rewards):
            advantage = r - baseline                  # normalized return
            batch_loss += -advantage * policy.logprob(prompt, edit)

    return batch_loss / max(len(prompts), 1)
```

Two design choices in that loop deserve emphasis. First, **RL is on the edit only**. Because localization is already strong, the optimizer is not wasting variance on a solved sub-problem; every gradient is about producing a better diff. Second, prompts where every rollout gets the same reward (all-pass or all-fail) contribute no advantage and are skipped — which is the seed of the curriculum described next.

### Adaptive prompt selection and curriculum

A flat RL run on a fixed prompt set wastes most of its compute on problems that are either trivially solved or hopelessly hard. Kimi-Dev manages the difficulty distribution actively:

- **Filter by solvability.** Prompts with **pass@16 = 0** — the model cannot solve them even once in 16 tries — are discarded, because they contribute nothing to the batch loss. This yields an initial working set of **1,200 problems**.
- **Reintroduce as the model improves.** Every **100 RL steps**, **500 previously-excluded** prompts (problems the model could not touch before but might now) are reintroduced, raising the difficulty floor in step with the policy's growing competence.
- **Positive-example reinforcement.** When gains plateau late in training, recent successful samples are re-injected into the current batch to reinforce the patterns that worked and accelerate convergence.

This is a curriculum in the literal sense: the problem set tracks the student. It is also a pragmatic answer to the sparse-reward problem — by keeping the working set in the band where pass@16 is neither 0 nor 16, every step has gradient signal.

### Execution-based test-time self-play

This is the cleverest part of the Agentless branch, and it is where the TestWriter earns its keep. At inference, for each instance, the system generates **40 BugFixer patches** and **40 TestWriter tests**. The first rollout of each is greedy (temperature 0); the remaining 39 are sampled at temperature 1.0. Then an execution-based scoring function picks the single patch to ship — without ever looking at the ground-truth tests.

The scoring works by running each candidate test file twice against each candidate patch: once on the unpatched code, once on the patched code. From those runs, for patch $b_i$ and test $t_j$, define $F(j)$ and $P(j)$ as the failed and passed test counts on the unpatched baseline, and $FP(i,j)$ and $PP(i,j)$ as the fail-to-pass and pass-to-pass counts after applying $b_i$. The score is:

$$
S_i = \frac{\sum_j FP(i,j)}{\sum_j F(j)} + \frac{\sum_j PP(i,j)}{\sum_j P(j)}
$$

The first term is a **reproduction term** — of the tests that were failing, how many did this patch flip to passing — and the second is a **regression term** — of the tests that were already passing, how many did this patch keep passing. A patch that fixes the bug *and* breaks nothing maximizes both. You pick the $b_i$ with the highest $S_i$.

```python
def select_patch(patches, valid_tests, run_test):
    """Execution-based self-play selection (Eq. 1).

    For each (patch, test) pair, run the test on the unpatched and
    patched repo. Score = reproduction term (failing -> passing)
    + regression term (passing -> still passing). Pick the argmax.
    """
    F = sum(count_failing(run_test(t, patched=False)) for t in valid_tests)
    P = sum(count_passing(run_test(t, patched=False)) for t in valid_tests)

    best_patch, best_score = None, float("-inf")
    for b in patches:                      # 40 BugFixer patches
        fp = pp = 0
        for t in valid_tests:              # 40 TestWriter tests, validity-filtered
            base = run_test(t, patched=False)
            fixed = run_test(t, patch=b)
            fp += fail_to_pass(base, fixed)   # reproduction signal
            pp += pass_to_pass(base, fixed)   # regression signal
        score = (fp / F if F else 0) + (pp / P if P else 0)
        if score > best_score:
            best_patch, best_score = b, score
    return best_patch
```

The payoff is large and it scales. As the patch-test grid grows from 1x1 to 40x40, the resolve rate climbs from **48.0% to 60.4%**, and execution-based self-play **consistently beats BugFixer majority voting**. The most striking efficiency result: self-play with just **3 patches x 3 tests** already beats a 40-patch majority vote. You get more out of nine well-scored candidates than out of forty unscored ones, because the TestWriter is supplying a real, execution-grounded signal that majority voting cannot.

![Self-play resolve rate climbs from 48.0% at one patch and one test to 60.4% at the full 40-by-40 grid](/imgs/blogs/kimi-dev-6.png)

The timeline above traces that scaling. The honest caveat, visible at the right end, is that even at 40x40 the resolve rate stays **below the pass@N you would get with the ground-truth test patch** — which means the ceiling here is set by the TestWriter's reproduction coverage, not by the BugFixer. That gap is the paper's clearest piece of remaining headroom, and I will return to it in the critique.

### From Agentless to SWE-Agent: the transfer

Everything so far produces the Agentless model. The central claim of the paper is what happens next. Take the RL-trained Agentless checkpoint and run a **light SFT on 5,016 SWE-Agent trajectories** — public trajectories released by SWE-smith, collected with Claude 3.7 Sonnet in synthetic environments. The result is a SWE-Agent that, during inference, runs up to **128K tokens and 100 turns** (64K during agent SFT training), and scores **48.6% pass@1**.

The transfer is efficient because the Agentless RL already installed the skills the agent needs. The paper sweeps the SWE-Agent SFT over four candidate priors — Base, mid-trained (MT), SFT, and RL — across a wide range of token budgets, and the RL prior dominates.

![Four priors feed the SWE-Agent SFT; only the RL prior transfers efficiently without mode collapse](/imgs/blogs/kimi-dev-7.png)

The tree above is the headline transfer result. The **RL prior matches the Base prior's best agent pass@1 using only $2^{23}$ SFT tokens, versus the Base prior's $1.5 \times 2^{28}$** — roughly 64x more efficient. It is the best prior in nearly all settings. The **SFT prior collapses at 200 trajectories ($2^{24}$ tokens)** due to mode collapse, while the RL prior generalizes; the **MT prior, given to agent RL from scratch, collapses below 2% pass@1 after just 10 RL steps**. Out of four candidate initializations, only the one that went through outcome RL on the Agentless edit stage produces an agent that trains stably and keeps getting better.

### Why from-scratch agent training fails, and skill priors do not

It is worth pausing on the contrast, because it is the empirical heart of the paper.

![From-scratch SWE-Agent RL collapses under sparse reward, while the skill-prior initialization keeps improving past 70 turns](/imgs/blogs/kimi-dev-3.png)

The before-after above sets the two regimes side by side. On the left, training a SWE-Agent from a mid-trained-only prior: the outcome reward is sparse over a long horizon, a pass@8 filter keeps only **260 of 6,202** prompts as usable, and within 10 RL steps the policy collapses below 2% pass@1. On the right, the skill-prior-first route: the atomic skills (localize, edit, reflect) are already seeded, the same pass@8 filter keeps **2,062 of 6,202** prompts, and after adaptation the RL prior keeps improving **beyond 70 agent turns** where the SFT, MT, and Base priors plateau around 70, 60, and 50 turns respectively. The skill prior is not a marginal accelerant. It is the difference between an optimization problem that has signal and one that does not.

### Context and compute budgets

Because the model touches four different context regimes across its lifecycle, it is worth laying them out as a single stack rather than leaving them scattered through the text.

![Context budgets widen at every stage, from 32K mid-training tokens to 128K-token, 100-turn agent inference](/imgs/blogs/kimi-dev-5.png)

Reading the stack from the bottom up: mid-training runs at **32K** max sequence length; Agentless RL runs at **64K** with the 0/1 reward; SWE-Agent RL training also runs at **64K** with batch size 256; and SWE-Agent inference extends to **128K tokens and 100 turns** (the deployment config exposes 131,072). The widening window is not incidental — each stage asks the model to reason over more of the repository and more of its own trajectory than the last, and the inference-time budget is what lets the agent explore for a hundred turns before committing a patch.

One thing the stack does *not* show, because the report does not state it, is the model's internal geometry. Layer count, attention type, and head configuration are inherited from Qwen2.5-72B and never enumerated in the report. Total GPU-hours, cluster size, wall-clock, the optimizer name, and the learning-rate schedule are likewise not stated; the only optimization details given are "REINFORCE-style policy gradient, batch 256, no KL, no entropy." For a reproduction effort this is the binding constraint.

### A comparison of the two paradigms

To make the design space concrete, here is how the two paradigms — and Kimi-Dev's position between them — line up:

| Axis | Agentless workflow | SWE-Agent | Kimi-Dev's move |
|---|---|---|---|
| Interaction | Single-turn, fixed stages | Multi-turn free interaction | Agentless first, then transfer to agent |
| Reward horizon | Short, per-stage verifiable | Long, sparse outcome only | RL on the edit stage only |
| RL trainability | High (small credit-assignment) | Low (collapses from scratch) | Build skills in the easy regime |
| Ceiling | Lower, rigid | Higher, flexible | Inherit the agent ceiling via transfer |
| Context at inference | Repo + 2 stages | Up to 128K / 100 turns | 128K / 100 turns after SFT |
| Selection signal | Single root test (prior work) | Trajectory-level | 40x40 execution self-play |
| Best result here | 60.4% resolve | 48.6% pass@1 | Both, from one checkpoint |

The table makes the thesis legible: Kimi-Dev does not try to make Agentless competitive with agents on the agent's terms, nor does it try to RL an agent from scratch. It builds the skills where they are cheap to build, then pays a small SFT to cash them out in the agent regime.

## Experiments

The experiments answer three questions in order: how good is the Agentless model, how good is the transferred agent, and how do we know the skill prior is the cause rather than a correlate.

### Agentless results: SWE-bench Verified, workflow methods

On SWE-bench Verified evaluated as an Agentless workflow, Kimi-Dev is state of the art among workflow and open-source methods.

| System | Params | SWE-bench Verified resolve rate |
|---|---|---|
| **Kimi-Dev 72B** | **72B** | **60.4%** |
| SWE-SWISS 32B | 32B | 58.2% |
| DeepSeek-R1-0528 671B | 671B | 57.6% |
| MiniMax-M1 456B | 456B | 56.0% |
| Claude 3.5 Sonnet (241022) | proprietary | 50.8% |
| DeepSeek-R1-0120 671B | 671B | 49.2% |

The headline is not just the 60.4%; it is the parameter efficiency of it. Kimi-Dev at 72B beats DeepSeek-R1-0528 at 671B and MiniMax-M1 at 456B by margins of a few points each. A dense 72B model is outscoring models with five to nine times its parameter count, which is exactly what you would expect if the gain comes from a well-designed training recipe rather than from raw scale.

### Agent results: SWE-bench Verified, end-to-end agentic

On the agentic side, the SFTed Kimi-Dev reaches 48.6% pass@1 (single attempt, as of 2025.09).

![SWE-bench Verified pass@1 across agentic frameworks, with Kimi-Dev at 48.6% versus same-data baselines and frontier models](/imgs/blogs/kimi-dev-4.png)

The matrix above places Kimi-Dev in the full agentic field. The numbers that matter most are not the frontier ones — GPT-5 at 74.9%, Claude 4.0 Sonnet at 72.7%, Kimi-K2-0905 1T at 69.2% — but the *like-for-like* ones.

| System | Params | Pass@1 |
|---|---|---|
| GPT-5 (proprietary) | — | 74.9% |
| Claude 4.0 Sonnet (SWE-Agent) | proprietary | 72.7% |
| Qwen3-Coder 480B (OpenHands) | 480B | 69.6% |
| Kimi-K2-0905 1T (SWE-Agent) | 1T | 69.2% |
| Claude 3.7 Sonnet (SWE-Agent) | proprietary | 62.3% |
| Gemini 2.5 Pro | proprietary | 60.3% |
| Devstral-Small-2507 24B (OpenHands) | 24B | 53.6% |
| Claude 3.5 Sonnet (241022, SWE-Agent) | proprietary | 49.0% |
| **Kimi-Dev (SFTed) 72B** | **72B** | **48.6%** |
| DeepSWE 32B (OpenHands) | 32B | 42.2% |
| SWE-agent-LM 32B (SWE-Agent) | 32B | 40.2% |
| Skywork-SWE 32B (OpenHands) | 32B | 38.0% |
| Openhands-LM 32B (OpenHands) | 32B | 37.2% |

Two comparisons carry the argument. First, **Kimi-Dev 48.6% is on par with Claude 3.5 Sonnet at 49.0%** — an open 72B model matching a proprietary frontier model of its generation. Second, and more important for the thesis, **Kimi-Dev uses the same SFT data as SWE-agent-LM yet outscores it 48.6% to 40.2%.** Same trajectories, +8.4 points. The only difference is the prior the SFT was applied to: SWE-agent-LM starts from a generic base, Kimi-Dev starts from the RL-trained Agentless checkpoint. That controlled comparison is the cleanest evidence in the paper that the skill prior, not the SFT data, is doing the work.

### The cross-over: agent pass@10 beats Agentless pass@30

There is one more result that crystallizes why the transfer matters. After adaptation, the SWE-Agent reaches **74.0% pass@10**, which **surpasses the Agentless Kimi-Dev's 73.8% pass@30**. The agent, given ten attempts, beats the workflow given thirty. The flexibility of the agent regime — the ability to explore, backtrack, and re-run tests — pays off once the model has the skills to use it, and the skills came from the Agentless training. This is the loop closing: Agentless training built the skills, and the agent regime cashed them out at a higher ceiling than Agentless itself could reach.

### What is load-bearing and what might not transfer

The load-bearing pieces, in my reading, are three. **The decontamination** — excluding every SWE-bench Verified repository from mid-training — is what makes the 60.4% a real generalization number rather than memorization; if that decontamination is incomplete, the headline collapses. **The edit-only RL** is what makes the optimization tractable; it works *because* localization is already solved by mid-training, so the recipe would not transplant cleanly to a domain where localization is the hard part. **The TestWriter's reproduction quality** is what makes self-play work; the entire 48.0% to 60.4% lift depends on the test signal being trustworthy, and the paper itself flags that it is not yet trustworthy enough.

What might not transfer is the **Python-centric, five-files-or-fewer** data regime. The curation explicitly targets Python repos with small PRs, and the localization-is-easy assumption is downstream of that. The paper does test generalization to **SWE-bench Multilingual** (300 tasks, 42 repos, non-Python) and **SWE-bench-live** (real-world issues with higher distribution shift), and reports that the RL prior generalizes *better* than the alternatives there, especially in data-scarce settings, with the gap narrowing as trajectories increase. That is encouraging, but "generalizes better than the other priors" is a relative claim; it does not tell us the absolute resolve rate on a large, sprawling, multi-language monorepo, which is where production software engineering actually lives.

## Critique

**What is strong.** The controlled comparison against SWE-agent-LM is the kind of experiment I wish more papers ran: hold the SFT data fixed, vary only the prior, measure the gap. +8.4 points from the prior alone is a clean, hard-to-explain-away result. The mid-training dose-response ablation (50B / 100B / 150B tokens, monotonic improvement) is similarly clean. And the test-time self-play formula is genuinely elegant — a reproduction term plus a regression term, both computed by execution, no learned reward model to drift. The fact that 3x3 self-play beats a 40-patch majority vote is a result I would put on a slide.

**What is weak or unfalsifiable.** The "skill prior" abstraction is doing a lot of rhetorical work, and the paper's attempt to *measure* the skills is the softest part. The "skill counting" — BugFixer skill rising from 484 (Base) to 605 (RL), reflection skill from +94 to +113 — is operationalized as "Stage-3-cutoff resolutions across 3 runs" and "gain from Stage-3 to end," which the authors themselves admit conflates true reflection and redo with intermediate test-writing. So the most evocative claim, that RL specifically grows the *reflection* skill, rests on a coarse proxy. I believe the transfer-efficiency result; I am less sure the mechanism is "reflection" as opposed to some less interpretable improvement in edit quality.

**The missing ablation.** The paper convincingly shows the RL prior beats the SFT, MT, and Base priors. What it does not isolate is *which component of the Agentless recipe* contributes most to transferability. We get a mid-training dose-response and a prior comparison, but not a clean "RL with the curriculum vs RL without it" or "with positive-example reinforcement vs without" on the *transfer* metric. Several of the recipe's most interesting choices — the adaptive curriculum, the positive-example reinforcement, the edit-only RL — are justified by intuition and by the final number, not by an ablation showing each one's marginal contribution to the agent's pass@1. The single most consequential missing experiment is the TestWriter ceiling study: how much of the 60.4-to-pass@N gap would close if you swapped in a stronger TestWriter? The paper points at this as future work but does not bound it.

**What would change my mind.** I would update strongly against the skill-prior thesis if someone showed that a *generic* reasoning prior — say, a 72B model mid-trained on math and general long-CoT but *not* on the Agentless localize-edit-test structure — transferred to a SWE-Agent just as efficiently as Kimi-Dev's RL prior. That would mean the magic is "reasoning RL" in general, not "Agentless software-engineering skill priors" specifically, and the paper's framing would be a special case of something broader. Conversely, the result that would most cement the thesis is a demonstration that the transfer efficiency holds when the *target* agent task is in a different language or a much larger repository than the Agentless training distribution — i.e., that the prior is about software-engineering skill and not about Python-shaped pattern matching.

## What I'd build with this

1. **A stronger TestWriter as a first-class model.** The paper's own ceiling analysis says the binding constraint on self-play is reproduction coverage. I would train a dedicated TestWriter with its own RL loop whose reward explicitly rewards *edge-case* coverage — not just fail-then-pass, but fail-then-pass across a diversity of inputs — echoing the edge-case-checking phase the authors note from Anthropic's work. If that closes even half the gap to ground-truth pass@N, it is several free points of resolve rate.

2. **Skill-prior transfer to a non-software agent domain.** The abstraction — build atomic skills in a verifiable single-turn regime, then transfer to a sparse-reward multi-turn agent — is not specific to code. I would test it on, say, a tool-using research agent: build "skill priors" (query formulation, evidence localization, claim verification) in a verifiable Agentless-style workflow, then SFT-transfer to the open-ended agent. If the 64x efficiency holds, it is a general recipe for bootstrapping agents.

3. **A curriculum that is automatic rather than scheduled.** The current curriculum reintroduces 500 excluded prompts every 100 steps — a fixed schedule. I would replace it with a bandit that tracks each prompt's recent pass@k and keeps the working set in the high-gradient band automatically, which should remove the two hand-tuned constants and adapt the difficulty floor to the policy's actual learning rate.

4. **Edit-only RL as a reusable pattern for staged tasks.** The insight that you should put RL pressure only on the stage that is still high-variance (the edit), having solved the rest with cheaper SFT (localization), generalizes to any staged pipeline. I would build a small library that, given a multi-stage task, measures per-stage pass@k after SFT and automatically selects the stage to RL.

5. **A distillation-free agent at smaller scale.** Kimi-Dev's agent still SFTs on Claude-3.7-collected trajectories. The more interesting open question is whether the skill prior is strong enough that you could RL the agent directly (the paper shows from-scratch RL collapses, but the *RL prior* keeps improving past 70 turns). I would push on agent RL *from the RL prior*, with no trajectory SFT at all, to see whether the prior alone can carry an agent past the SFT-bootstrapped 48.6%.

## When to reach for Agentless skill priors (and when not to)

Reach for this recipe when three conditions hold. **First, your task decomposes into verifiable stages** where at least one stage (here, localization) can be made reliable with cheap supervised data, so you can concentrate RL on the genuinely hard stage. **Second, you have an execution oracle** — a Docker sandbox, a test suite, anything that turns "did it work" into a 0/1 you can compute thousands of times in parallel. Kimi-Dev's infrastructure supports more than 10,000 concurrent Docker instances and more than 25,000 images precisely because the whole method is execution-bound. **Third, your real deployment target is a sparse-reward agent** that you cannot train from scratch. If all three hold, building the Agentless skill prior first is, on this evidence, the most sample-efficient path to a trainable agent — 64x more efficient than the from-base alternative.

Do not reach for it when localization is the hard part of your task. The recipe's tractability rests on localization being solved by mid-training so that RL can focus on the edit; in a domain where finding the relevant context is itself the open problem, the edit-only RL loses its footing and you are back to the long-horizon credit-assignment problem the method was designed to avoid. Be cautious, too, if your domain is far from the Python, small-PR distribution the data was curated for: the multilingual and live-issue generalization is reported as *relatively* better than the other priors, not as a strong absolute number, and you should expect to re-curate mid-training data for your own language and repository scale before trusting the transfer. And if you only ever need a workflow solver and never an agent, you can stop after the Agentless branch — the 60.4% is already SoTA among workflow methods, and you would be paying for transfer machinery you never use.

The deeper lesson outlasts the specific numbers. The Agentless-versus-Agent debate was framed as a choice between trainability and ceiling, and Kimi-Dev's contribution is to show that you can have the trainability of the workflow regime *and* the ceiling of the agent regime, because the skills are shared and the cheap regime is where you should build them. That is a reusable idea, and it is the part of this paper I expect to still matter when the leaderboard has moved on.

## References

- Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents — arXiv abstract: [https://arxiv.org/abs/2509.23045](https://arxiv.org/abs/2509.23045)
- Kimi-Dev GitHub: [https://github.com/MoonshotAI/Kimi-Dev](https://github.com/MoonshotAI/Kimi-Dev)
- Project page: [https://moonshotai.github.io/Kimi-Dev/](https://moonshotai.github.io/Kimi-Dev/) — Model weights: [https://huggingface.co/moonshotai/Kimi-Dev-72B](https://huggingface.co/moonshotai/Kimi-Dev-72B)
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](/blog/paper-reading/reinforcement-learning/kimi-k1-5) — the policy-optimization recipe Kimi-Dev's RL is built on.
- [Kimi K2: Open Agentic Intelligence](/blog/paper-reading/large-language-model/kimi-k2) — the larger agentic sibling whose 1T-parameter agent appears in the comparison table.
- [Kimi-Researcher: End-to-End RL for Autonomous Research](/blog/paper-reading/ai-agent/kimi-researcher) — a related take on training agents end-to-end with RL.
- [Muon is Scalable for LLM Training: Inside Moonlight](/blog/paper-reading/large-language-model/muon-moonlight) — Moonshot's optimizer work that underpins their training stack.

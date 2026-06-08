---
title: "Kimi-Researcher: End-to-End RL for Autonomous Research"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "paper-reading"
subcategory: "AI Agent"
tags:
  - reinforcement-learning
  - agentic-rl
  - autonomous-research
  - deep-research-agent
  - tool-use
  - reinforce
  - long-horizon
  - moonshot-ai
description: "How Moonshot AI trained a single model to plan, search, browse, run code, and verify entirely through end-to-end agentic reinforcement learning, lifting Humanity's Last Exam Pass@1 from 8.6% to 26.9%."
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/kimi-researcher-1.png"
readTime: 31
---

The hard part of building a research agent is not the search box. It is what happens after the third search comes back with two sources that disagree, the fourth tool call times out, and the model has already burned half its context window on documents it will never cite. Most agent stacks handle that situation with a human-authored control flow: a planner module hands off to a search module, a verifier module checks the answer, and a brittle pile of prompts holds the whole thing together. Every one of those seams is a place where the system can drift away from the underlying model's actual strengths, and every model upgrade forces you to re-tune the seams by hand.

Kimi-Researcher is Moonshot AI's bet that you should delete the seams. Instead of orchestrating a multi-agent workflow or imitating human demonstrations, they trained one model to do the entire job — planning, multi-turn search, cross-source verification, and tool use — through end-to-end agentic reinforcement learning. The headline result is striking: on Humanity's Last Exam (HLE), Pass@1 climbed from a starting point of **8.6%** to **26.9%**, a gain the blog attributes "almost entirely" to the RL training itself rather than to a bigger base model or a cleverer prompt scaffold. That is the kind of number that makes you re-examine your priors about where agent capability actually comes from.

![One model runs the whole research loop end to end](/imgs/blogs/kimi-researcher-1.png)

The diagram above is the mental model: a single policy enters a plan-search-browse-verify-answer loop, and the loop is not a fixed pipeline of separate modules but one model deciding, at every step, whether to issue another query, open another URL, run more code, or commit to an answer. In a typical task this loop runs for an average of **23 reasoning steps**, explores **200+ URLs**, and issues **70+ search queries** per trajectory, with single rollouts stretching past **50 iterations**. Everything that used to be a hand-coded handoff is now a learned decision inside the policy. One important caveat before we go further: there is no formal arXiv technical report for this work. Kimi-Researcher was released as an official Moonshot AI blog and project page in June 2025, so the level of detail here is recipe-and-results, not a reproducible spec. Where the source does not state a number — parameter count, exact context length, GPU budget, dataset size — we will say so plainly rather than invent it.

> [!tldr] TL;DR
> - **What it claims:** A single Kimi k-series model, trained end-to-end with agentic RL (REINFORCE), learns planning, search, verification, and tool use jointly — no multi-agent workflow, no SFT-on-demonstrations dependency.
> - **Why it matters:** HLE Pass@1 went from **8.6% to 26.9%** (state-of-the-art at the time) and **40.17% Pass@4**, with xbench-DeepSearch at **69% Pass@1** beating "o3 with search tools" — gains the blog credits almost entirely to RL, not architecture.
> - **Most surprising finding:** A simple **gamma-decay reward** ($r \times \gamma^{T-i}$) plus **strategic discarding of negative samples** and a **context-management mechanism** are enough to keep on-policy RL stable across 50+ iteration rollouts; the context mechanism alone buys **+30% more iterations**.
> - **Where it fails:** Trajectories are expensive (50+ iterations, hundreds of thousands of tokens, 200+ URLs each), generalization beyond search/reasoning benchmarks is uncharacterized, and the report discloses no params, no compute budget, no hyperparameters, and no exact scores for three of its five benchmarks.

## Context: what came before

To understand why end-to-end RL is the interesting choice, you have to look at the two paradigms it is reacting against, because Kimi-Researcher is defined as much by what it refuses to do as by what it does.

The first paradigm is **workflow-based or multi-agent systems built on prompt engineering**. This is the dominant pattern in production today: you decompose "do research" into a planner agent, a search agent, a synthesis agent, and a critic agent, then wire them together with prompts and a state machine. It works, and it ships, but the blog names its weaknesses precisely. These systems are tied to a specific underlying model — the prompts are tuned for one model's quirks, so when the model changes, the prompts rot. They require frequent manual updates as models and environments evolve. And they are brittle: a single tool returning an unexpected format can cascade into a failed run because no part of the system was *trained* to recover; it was only *prompted* to follow a script.

The second paradigm is **imitation learning**, specifically supervised fine-tuning on human demonstrations of multi-step tasks. The appeal is obvious — show the model good trajectories and have it copy them. The problem is twofold. First, labeling long-horizon trajectories is brutally hard and expensive: a single deep-research task can involve dozens of steps, and producing a clean, correct demonstration for each is labor-intensive in a way that does not scale. Second, demonstrations are tied to fixed tool versions. The moment a search API changes its response schema or a browser tool updates, the demonstrated behavior breaks, and the model has memorized the wrong thing. You have taught it to press buttons that no longer exist.

Kimi-Researcher's thesis is that **end-to-end agentic RL** dissolves both problems at once. If you let one model learn the whole loop — plan, search, verify, use tools — by trial and reward rather than by imitation or orchestration, then the model is not locked to a script or a tool version. It learns *behaviors* that achieve outcomes, and those behaviors adapt as the environment shifts. The cost is that you now have to solve a harder set of problems: dynamic environments that change under the agent, long-horizon reasoning that makes credit assignment painful, training-data scarcity for tasks worth learning, and rollout efficiency when each trajectory is enormous. The rest of this article is about how they tackled those four.

![Why end-to-end RL beats hand-built agent recipes](/imgs/blogs/kimi-researcher-2.png)

The before-after split makes the lineage concrete: on the left, workflow prompting and SFT-on-demonstrations both share a fatal coupling — to a specific model or a specific tool version — that forces manual maintenance. On the right, end-to-end RL co-optimizes planning, search, and verification inside one policy, so the same training process that produces good plans also produces good recovery behavior when a source conflicts or a tool misfires. This is the same philosophical move that drove [Kimi k1.5's](/blog/paper-reading/reinforcement-learning/kimi-k1-5) reasoning RL and the broader [Kimi K2](/blog/paper-reading/large-language-model/kimi-k2) agentic line: push capability into the weights through outcome-driven RL rather than into the scaffold through prompting.

## Contributions

The blog's contributions, tightened and de-duplicated from the dossier, are five:

1. **An end-to-end, single-model agentic RL recipe** that trains one policy to jointly learn planning, multi-turn search, verification, and tool use — explicitly rejecting both orchestrated multi-agent workflows and SFT-on-demonstrations as the primary training signal. The capability gains are claimed to come "almost entirely through end-to-end RL."

2. **A reward and stabilization scheme tuned for long-horizon rollouts:** an outcome-based reward (format + correctness) shaped by a gamma-decay factor $r \times \gamma^{T-i}$ that rewards shorter correct trajectories, combined with strategic discarding of negative samples to prevent entropy collapse, all trained strictly on-policy with format enforcers disabled.

3. **A context-management mechanism** that lets the model retain important information while discarding unnecessary documents, extending a single rollout from a naive ceiling of roughly 10 iterations to over 50 — an ablation shows it yields **30% more iterations** and correspondingly higher performance.

4. **A systems layer for efficient agentic RL:** a fully asynchronous, Gym-like rollout server orchestrating actors, environments, and rewards in parallel; a **Turn-level Partial Rollout** mechanism that parks over-budget trajectories in a replay buffer and resumes them with updated weights for a claimed **≥1.5× rollout speedup**; and a unified Kubernetes hybrid-cloud sandbox with three tools connected via the Model Context Protocol (MCP).

5. **An automated data-synthesis and validation pipeline** that generates and validates QA pairs across tool-centric and reasoning-intensive task families, using ground-truth extraction and Pass@N filtering to keep only non-trivial questions, with minimal manual labeling.

The result is state-of-the-art HLE performance and strong showings across xbench-DeepSearch, FRAMES, Seal-0, and SimpleQA — cited as "compelling evidence that end-to-end agentic RL can significantly advance agent intelligence."

## Method

The method has four moving parts that have to work together: the agent loop and its tools, the data that defines what "good research" even means, the reward that scores a trajectory, and the RL recipe that turns rewards into weight updates without the training collapsing. We will take them in that order, defining symbols as we go.

### The three tools and MCP

The agent acts on the world through exactly three tools, and the choice of three is deliberate — each covers a distinct failure mode of pure language-model reasoning.

- An **internal, parallel, real-time search tool.** "Parallel" is the load-bearing word: the agent can fan out multiple queries at once rather than serializing them, which is what makes 70+ queries per trajectory tractable within a time budget. "Internal" means it is Moonshot's own search backend, not a third-party API the agent has to babysit.
- A **text-based browser tool** for interactive web tasks. Search returns links; the browser is what lets the agent actually read a page, follow a citation, or navigate a multi-step site. This is the tool that drives the 200+ URLs explored per task.
- A **coding tool** for automated code execution. When a question reduces to arithmetic, data parsing, or a small simulation, the agent writes and runs code rather than reasoning about numbers in its head — the same failure mode that pure-LLM math hits.

All three communicate with the agent through the **Model Context Protocol (MCP)**, a standardized tool-invocation interface. Using MCP rather than bespoke function-calling glue is part of the same anti-brittleness philosophy: the tool contract is explicit and uniform, so the policy learns one calling convention rather than three.

The crucial property is that none of these tools is wrapped in a hand-coded controller. The policy decides when to search, when to browse, when to run code, and when to stop — and it learns those decisions from reward, not from a flowchart.

### Data synthesis: defining what is worth learning

RL is only as good as the tasks you train on. If your tasks are trivially solvable, the policy learns nothing; if they are unsolvable, you get no reward signal. The blog describes a fully automated pipeline that generates and validates many question-answer pairs with minimal manual intervention, organized into two task families.

![Automated data synthesis spans two task families](/imgs/blogs/kimi-researcher-4.png)

The taxonomy above splits the training data cleanly. **Tool-centric tasks** are constructed so that solving them *requires* invoking specific tools — the tool dependency is embedded in the problem itself. A question whose answer lives behind a search query or requires running code cannot be answered by reasoning alone, which forces the policy to actually learn the tool-use behavior rather than memorizing shortcuts. **Reasoning-intensive tasks** split further into (i) math and code reasoning, where the challenge is multi-step logical derivation, and (ii) hard search tasks that require iterative synthesis across many sources — the contradictory-evidence case that motivates the whole project.

The quality control is where this becomes RL-grade. Two filters apply. First, **ground-truth extraction**: every synthesized question must have a robustly extractable correct answer, because without a ground truth there is no correctness reward. Second, **Pass@N filtering**: the pipeline runs candidate questions through the model $N$ times and discards questions that are too easy — if the model already solves a question reliably, it carries no learning signal and only inflates reward variance. What survives is a set of non-trivial questions that sit in the productive band where the policy is wrong often enough to learn but right often enough to get gradient.

Here is the shape of that filtering loop in pseudocode. Note the comment style — everything is indented inside the function so no line begins with a hash at column zero:

```python
def build_training_set(candidates, policy, N=8, low=0.1, high=0.9):
    """Keep only non-trivial, ground-truthable QA pairs.

    candidates: synthesized (question, answer) pairs
    policy:     the current model used for Pass@N estimation
    N:          rollouts per question for difficulty estimation
    low/high:   keep questions whose empirical pass-rate is in (low, high)
    """
    kept = []
    for q, gold in candidates:
        # drop anything we cannot grade against a ground truth
        if gold is None or not is_extractable(gold):
            continue

        # Pass@N: estimate difficulty by sampling the policy N times
        successes = 0
        for _ in range(N):
            traj = policy.rollout(q)                  # full agentic trajectory
            if correctness_reward(traj.answer, gold) == 1.0:
                successes += 1
        pass_rate = successes / N

        # discard trivial (too easy) and likely-broken (never solvable) items
        if low < pass_rate < high:
            kept.append((q, gold))
    return kept
```

The exact thresholds and value of $N$ are not stated in the source — the dossier only specifies that Pass@N "ensures only non-trivial questions are retained." The pseudocode above shows the *mechanism* with placeholder bounds; treat `low`, `high`, and `N` as illustrative, not reported.

### Strict on-policy training

There is a subtle data-distribution trap in agentic RL that the blog calls out explicitly. To make tool calls parse cleanly, it is tempting to wrap the model's output in a **format enforcer** — a constrained-decoding layer or a regex-repair step that guarantees valid JSON tool calls. The problem is that an enforcer changes the distribution of what the model actually emits. If you train on enforcer-corrected outputs, you are training on data the policy did not really produce, which makes the data **off-policy**: the gradient points toward a distribution the model cannot reach on its own.

Kimi-Researcher's stance is to disable format enforcers during training and generate **strict on-policy data** — the policy's true, unedited output distribution is what gets sampled and scored. If the model emits a malformed tool call, that malformed call is part of the trajectory and gets penalized by the format reward (below). The model learns to format correctly *because* it is rewarded for doing so, not because an external crutch fixed its output after the fact. This is more painful early in training, when the model produces a lot of garbage, but it keeps the data and the gradient honest, and it is why the format reward exists as a first-class term.

### The reward: only outcome-based

The reward has exactly two components, and both are outcome-based — there is no process reward, no learned reward model scoring intermediate steps, no human preference signal. For a completed trajectory:

- **Format reward** penalizes invalid tool calls and format violations. This is the signal that, combined with strict on-policy training, teaches the model to emit well-formed MCP calls without an enforcer.
- **Correctness reward** compares the final answer to the extracted ground truth. This is binary in the spirit of the Pass@N grading above: the answer is right or it is not.

The combined outcome reward is then shaped by a gamma-decay factor that is the one explicit formula in the entire report. Let $T$ be the number of steps in a trajectory and $i$ the index of a step. The reward attributed to step $i$ is:

$$
r_i = r \times \gamma^{\,T - i}, \qquad 0 < \gamma < 1
$$

where $r$ is the outcome reward for the whole trajectory. Read the exponent carefully: $T - i$ is large for early steps and zero for the final step. So with $0 < \gamma < 1$, **earlier steps in a correct trajectory get discounted less the closer they are to the end** — and crucially, a shorter trajectory (small $T$) keeps more of the reward at every step than a long one. The design intent stated in the blog is to make "earlier-but-correct, shorter trajectories more valuable, encouraging shorter, more efficient exploration paths." In an agent that can wander across 200+ URLs, a reward that quietly pays a premium for getting there in fewer steps is a cheap, effective pressure toward efficiency without needing a separate length penalty term.

Here is the reward in code, again indented so no comment line starts at column zero:

```python
import math

def trajectory_reward(traj, gold, gamma=0.9):
    """Outcome reward = format + correctness, shaped by gamma decay.

    Returns a per-step reward vector r_i = r * gamma**(T - i).
    """
    # format term: penalize any invalid tool call across the trajectory
    format_ok = all(is_valid_tool_call(step.action) for step in traj.steps)
    format_term = 0.0 if format_ok else -1.0

    # correctness term: compare final answer to extracted ground truth
    correctness_term = 1.0 if grade(traj.answer, gold) else 0.0

    # outcome reward r for the whole trajectory
    r = correctness_term + format_term

    # gamma-decay shaping: later steps keep more reward, short paths win
    T = len(traj.steps)
    per_step = []
    for i, _step in enumerate(traj.steps, start=1):
        per_step.append(r * (gamma ** (T - i)))
    return per_step
```

The exact value of $\gamma$, and the precise format-penalty magnitude, are not reported — the blog gives only the functional form $r \times \gamma^{T-i}$ and the constraint $0 < \gamma < 1$. The $\gamma = 0.9$ and $-1.0$ penalty in the snippet are illustrative defaults, not quoted numbers.

### Training: the REINFORCE recipe

The RL algorithm is **REINFORCE** — the classic policy-gradient estimator, not PPO, not GRPO, not an actor-critic with a value network. This is a notable choice, and the report is candid that it does not describe an explicit baseline, advantage estimator, or variance-reduction trick beyond three mechanisms: strict on-policy data, the gamma-decay reward, and negative-sample discarding. Plain REINFORCE has notoriously high gradient variance, so the interesting engineering question is what keeps it stable across 50-iteration rollouts. The answer is the third mechanism.

**Negative-sample control to prevent entropy collapse.** The failure mode is specific. In policy-gradient training, negative samples — trajectories that earned low reward — push down the probabilities of the tokens they contain. If too many negatives flow through, token probabilities collapse toward a peaky, low-entropy distribution: the policy becomes overconfident and stops exploring, which in an agent means it stops trying new search strategies and gets stuck. The blog's exact framing: "Negative samples lead to a decrease in token probabilities, which increases the risk of entropy collapse." The mitigation is blunt and effective: "we discard some negative samples strategically." By not letting every failed trajectory hammer the probabilities down, the policy retains enough entropy to keep exploring. The exact discard ratio and selection criterion are not stated.

![The RL stabilization stack that makes long rollouts trainable](/imgs/blogs/kimi-researcher-6.png)

The stack above is the right way to hold these mechanisms in your head: strict on-policy data is the foundation, the outcome reward sits on top of it, gamma decay shapes that reward toward efficiency, negative-sample discarding keeps entropy from collapsing under REINFORCE's variance, and context management at the apex is what physically allows the long rollouts to exist at all. Pull any layer out and the one above it wobbles — without on-policy data the reward is measuring the wrong distribution; without negative-sample control the policy collapses; without context management the rollouts hit a wall around 10 iterations and never reach the regime where the rest of the recipe pays off.

A minimal sketch of the update loop ties the reward and the discarding together:

```python
def reinforce_step(policy, batch, optimizer, gamma=0.9, keep_neg=0.5):
    """One on-policy REINFORCE update with negative-sample discarding."""
    trajectories = [policy.rollout(q) for (q, _gold) in batch]

    losses = []
    for traj, (q, gold) in zip(trajectories, batch):
        rewards = trajectory_reward(traj, gold, gamma=gamma)
        traj_return = sum(rewards)

        # strategically drop a fraction of negative trajectories so that
        # they do not collapse token probabilities (entropy preservation)
        if traj_return <= 0.0 and random_uniform() > keep_neg:
            continue

        # standard REINFORCE: weight log-prob of each emitted token by reward
        for step, r_i in zip(traj.steps, rewards):
            logp = policy.log_prob(step.action, context=step.state)
            losses.append(-(r_i * logp))

    loss = mean(losses)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

The `keep_neg` fraction here stands in for "discard some negative samples strategically" — the real selection policy is unspecified. There is no critic, no clipping, and no KL-to-reference term in the reported recipe, which is what makes the stability of the result genuinely surprising and what makes the supporting mechanisms (on-policy data, gamma decay, negative-sample control, context management) load-bearing rather than decorative.

### Context management: the unlock for 50+ iterations

A 50-iteration trajectory that explores 200+ URLs would, naively, accumulate hundreds of thousands of tokens of retrieved documents in context — and then run out of room long before it finishes. The blog describes a **context-management mechanism that allows the model to retain important information while discarding unnecessary documents.** Rather than carrying every fetched page forever, the agent learns to summarize or drop documents it no longer needs, freeing context for new exploration. This is the difference between an agent that hits a wall at roughly 10 iterations and one that runs past 50. We will return to the ablation that quantifies this in the experiments, but the architectural point is that context management is not a nice-to-have — it is the precondition that makes long-horizon agentic RL physically possible within a finite context window described only as "hundreds of thousands of tokens" (no exact length is given).

### Systems: async rollout and partial rollout

The last method component is pure systems engineering, and it is where the efficiency challenge from the motivation gets answered. Agentic RL has a nasty property: a single batch of trajectories has a heavy long tail. Most rollouts finish quickly, but a few wander into 50-iteration marathons, and synchronous training stalls waiting for the slowest one. Two mechanisms attack this.

![Asynchronous rollout system feeds RL with partial rollouts](/imgs/blogs/kimi-researcher-3.png)

The dataflow above shows the asynchronous rollout system. An actor policy samples on-policy (with no format enforcer), and a **fully asynchronous, Gym-like rollout server** orchestrates actor rollouts, environmental interactions, and reward calculation in parallel. The server fans each trajectory one of two ways: trajectories within the time budget run their tool calls in the Kubernetes sandbox (search, browser, code, all via MCP) and flow to reward; trajectories that blow the time budget get parked in a **replay buffer** via **Turn-level Partial Rollout**. The blog's description: "tasks that exceed a time budget would be saved to a replay buffer. In subsequent iterations, the remaining turns would be executed with updated model weights." Both paths merge at the reward calculation, which feeds the REINFORCE update that pushes new weights forward to the next iteration.

The payoff of partial rollout is a claimed **≥1.5× rollout acceleration**, because the slow long-tail trajectories no longer block the batch — they are resumed later with fresh weights instead of holding up everyone else. The sandbox itself is a unified architecture deployed on a Kubernetes hybrid cloud, which is what makes running three real tools across many parallel rollouts operationally feasible. This systems pattern — disaggregate the slow path, keep the fast path moving — echoes the serving philosophy behind [Mooncake](/blog/paper-reading/large-language-model/mooncake), where the long-tail problem lives in KV-cache movement rather than rollout turns.

The one architectural caveat worth flagging: resuming a partial rollout "with updated model weights" means the front half of a trajectory was produced by an older policy than the back half. That is a mild violation of the strict-on-policy principle the recipe otherwise insists on, and the blog does not discuss how the two halves are reconciled in the gradient. It is a reasonable trade — a small staleness in exchange for a 1.5× throughput win — but it is a seam, and it is the kind of detail a formal paper would have to address.

### A comparison table for the recipe

It helps to put Kimi-Researcher's choices side by side with the paradigms it rejects and with a conventional RLHF-style setup, so the design space is legible:

| Design axis | Workflow / multi-agent | SFT on demonstrations | Conventional RLHF | Kimi-Researcher |
|---|---|---|---|---|
| Primary training signal | None (prompt-engineered) | Imitation of human demos | Preference / reward model | Outcome reward (format + correctness) |
| Who controls the loop | Hand-coded orchestrator | Memorized from demos | Single-turn policy | Learned multi-turn policy |
| RL algorithm | — | — | PPO / GRPO (clipped, critic) | REINFORCE (no critic, no clip) |
| On-policy discipline | n/a | Off-policy by construction | Approximately on-policy | Strict on-policy, enforcers off |
| Long-horizon support | Brittle, manual | Poor (label cost) | Limited | Context mgmt → 50+ iterations |
| Adapts to tool/model change | Poor (manual re-tune) | Poor (version-locked) | Partial | Learned, robust to drift |
| Stability trick | n/a | n/a | KL + clipping | Gamma decay + discard negatives |

The table makes the bet explicit. Kimi-Researcher trades the variance-control machinery of PPO/GRPO (clipping, a critic, a KL leash) for a much simpler REINFORCE core, and pays for that simplicity with three targeted mechanisms — gamma-decay shaping, negative-sample discarding, and strict on-policy data — plus the systems layer that makes long rollouts affordable. Whether that trade is robust or fragile is exactly the question the critique section returns to.

## Experiments

The evaluation spans five benchmarks, all tested in mid-June 2025: Humanity's Last Exam (HLE, tested June 17), and xbench-DeepSearch, FRAMES, Seal-0, and SimpleQA (tested June 18). The headline is HLE, and it is worth stating the result with full precision.

![Benchmark results across search and reasoning suites](/imgs/blogs/kimi-researcher-5.png)

The matrix above lays out every reported number. On HLE, Kimi-Researcher reaches **26.9% Pass@1**, up from an initial **8.6%** — a state-of-the-art result at the time, with the gain attributed almost entirely to end-to-end RL. The Pass@4 figure is **40.17%**, also reported as state-of-the-art. On xbench-DeepSearch, it scores **69% Pass@1** averaged over four runs, which the blog states outperforms "o3 with search tools." On FRAMES, Seal-0, and SimpleQA, the blog reports only qualitative "strong performance" — and here we have to be honest about the disclosure gap: **the exact scores for those three benchmarks are not stated in the source.** We are not going to fill them in.

Here is the full results table with everything the source actually quotes:

| Benchmark | Metric | Score | Baseline / note |
|---|---|---|---|
| Humanity's Last Exam (HLE) | Pass@1 | **26.9%** | from initial **8.6%**; SOTA; gain almost entirely from RL |
| Humanity's Last Exam (HLE) | Pass@4 | **40.17%** | reported as SOTA |
| xbench-DeepSearch | Pass@1 | **69%** | averaged over 4 runs; beats "o3 with search tools" |
| FRAMES | multi-turn search reasoning | "strong performance" | exact score not stated |
| Seal-0 | multi-turn search reasoning | "strong performance" | exact score not stated |
| SimpleQA | factual QA | "strong performance" | exact score not stated |

The single explicit ablation is the one that matters most for the central claim, and it is about context management. A model trained **with** the context-management mechanism uses **30% more iterations** than one without, and those extra iterations let it acquire more information and reach higher performance. Read in reverse, this tells you that without context management the agent is effectively capped near the naive ~10-iteration ceiling, and that the long, >50-iteration trajectories — the ones that produce the 200+ URL exploration and the 70+ queries — only exist because of this mechanism. The ablation is the evidence that the headline behavior is *caused* by the recipe and not incidental to a strong base model.

It is worth working a small number to feel how the gamma-decay reward interacts with these long trajectories. Take two correct trajectories that both earn outcome reward $r = 1$: a tight one with $T = 10$ steps and a sprawling one with $T = 50$ steps, and set $\gamma = 0.95$ for illustration. The first step of the short trajectory keeps $\gamma^{T-1} = 0.95^{9} \approx 0.63$ of the reward, while the first step of the long trajectory keeps only $0.95^{49} \approx 0.08$. The total discounted return of the short path, $\sum_{i=1}^{10} 0.95^{10-i} \approx 8.0$, comfortably exceeds the long path's per-step values at matched positions. The policy gradient therefore pays visibly more, step for step, for reaching the same answer in fewer steps — which is precisely the "shorter, more efficient exploration paths" pressure the blog describes, achieved without a separate length-penalty term that would need its own coefficient to tune. Pair that pressure with context management's ability to *afford* long trajectories when they are genuinely needed, and you get an agent that goes long only when the task forces it to.

The blog also reports two qualitative emergent behaviors that are worth taking seriously even though they are not benchmarked. First, **conflicting-information resolution**: the agent performs iterative hypothesis refinement and self-correction across multiple sources to reconcile contradictory evidence — exactly the situation we opened this article with. Second, **caution and rigor**: even for questions that look straightforward, the agent deliberately performs additional searches and cross-validates before answering, which the blog illustrates with worked reasoning on complex cultural and geographical questions. These are presented as behaviors that *emerged from RL*, not behaviors that were prompted in, which is the qualitative complement to the quantitative HLE jump.

### What is load-bearing, and what might not transfer

The number that carries the entire thesis is the HLE 8.6% → 26.9% jump under RL. If you are deciding whether to adopt this recipe, that is the claim to interrogate, and a few things about it are load-bearing.

It is load-bearing that the gain is attributed to RL "almost entirely" — if a large fraction actually came from the internal base model being stronger than the 8.6% starting checkpoint suggests, the recipe is less impressive than advertised. The blog asserts the RL attribution but, lacking a controlled "same base model, no RL" comparison at full scale beyond the training-curve figures, we are taking that attribution somewhat on faith. It is load-bearing that the benchmarks are search-and-reasoning heavy: HLE, xbench-DeepSearch, FRAMES, Seal-0, and SimpleQA all reward exactly the multi-turn-search-and-verify loop the agent was trained on. That is a fair test of the claim "this trains good research agents," but it is not evidence that the same recipe produces good agents for, say, long-horizon software engineering or GUI control — domains where [Kimi-Dev](/blog/paper-reading/large-language-model/kimi-dev) and [Kimi K2.5](/blog/paper-reading/large-language-model/kimi-k2-5) operate, and where the reward structure and tool set differ substantially.

What might not transfer is the stability of plain REINFORCE. The recipe works here with these three tools, this reward, and this data distribution. There is no guarantee that REINFORCE-without-a-critic stays stable when you change the reward to something denser, add tools with longer latency, or train on a task family where negative samples dominate the early distribution. The negative-sample-discarding and gamma-decay mechanisms are tuned to *this* setting, and the report gives no sweep showing how sensitive the result is to $\gamma$, to the discard ratio, or to the Pass@N thresholds. That sensitivity is the open empirical question.

## Critique

**What is strong.** The central result is genuinely impressive and the framing is honest about its own method's lineage. Tripling HLE Pass@1 mostly through RL, with a recipe this simple at its core (REINFORCE + outcome reward), is a real data point in favor of the "capability lives in the weights, not the scaffold" hypothesis. The context-management ablation is the right ablation — it directly supports the claim that long rollouts cause the performance, and the +30%-iterations number is concrete. The systems work (async rollouts, Turn-level Partial Rollout, ≥1.5× speedup) is the kind of unglamorous engineering that actually makes agentic RL trainable at scale, and the blog deserves credit for treating it as a first-class contribution rather than an afterthought. The decision to disable format enforcers and train strictly on-policy is principled and easy to get wrong; calling it out as critical is a sign the authors understand where off-policy bugs creep in.

**What is weak or unfalsifiable.** This is a blog, not a paper, and the disclosure gaps make several claims hard to falsify or reproduce. There is no parameter count, no exact context length (only "hundreds of thousands of tokens"), no GPU type or count, no GPU-hours, no dataset size, no optimizer or learning rate, no batch size, and no gamma value. "Strong performance" on FRAMES, Seal-0, and SimpleQA with no numbers is not a result a reader can check or compare. The "almost entirely through RL" attribution is asserted without a clean controlled comparison at full scale. And the REINFORCE recipe is described at a level that omits the baseline/variance-reduction details that would let someone reproduce the stability — the difference between "we used REINFORCE" and a working REINFORCE-for-agents implementation is enormous, and that difference is exactly what is not disclosed.

**What ablation is missing.** The one ablation present (context management) is good, but the recipe makes at least four other choices that beg for sweeps and get none. There is no gamma-value sweep showing how reward shaping affects trajectory length and final score. There is no negative-sample-discard on/off comparison with numbers, despite entropy collapse being named as the risk it prevents — that is the single most important stability claim and it is unquantified. There is no REINFORCE-vs-PPO/GRPO comparison, so we cannot tell whether the simple algorithm is a feature or a constraint they worked around. And there is no Pass@N threshold sensitivity, so the data-difficulty band is a black box. Each of these is the kind of plot that would turn an assertion into evidence.

**What would change my mind.** I would update strongly toward "this recipe is robust and general" if I saw two things: first, a controlled curve showing the *same* base checkpoint with and without the full RL recipe, on the *same* HLE split, isolating the RL contribution from base-model strength; and second, a negative-sample-discard ablation with entropy and reward curves showing the policy collapsing when the mechanism is off and recovering when it is on. Conversely, I would update toward "this is a tuned-to-the-benchmark result" if the open-sourced model, once released, failed to reproduce the 26.9% HLE Pass@1 under independent evaluation, or if the recipe proved unstable when ported to a task family with a denser reward or a different tool set. As of the blog's publication the models were not yet released — Moonshot stated plans to open-source the base pretrained model and the RL-trained model "in the following months" — so independent reproduction is the test that will settle it.

## What I'd build with this

Five concrete extensions I would pursue if I had this recipe in hand:

1. **A controlled RL-attribution harness.** Before anything else, build the missing experiment: freeze the base checkpoint, run the full agentic-RL recipe, and report HLE Pass@1 at fixed RL-step intervals against a no-RL control on an identical split. This is the experiment that turns "almost entirely through RL" from a claim into a measurement, and it is cheap relative to the training itself.

2. **A dense-reward variant for partial credit.** The outcome reward is binary-ish (correct or not) plus a format term. For research tasks with structured answers — a table, a multi-part question, a citation list — I would add a partial-correctness reward and re-run the gamma-decay shaping on top of it, then measure whether REINFORCE stays stable when negatives no longer dominate the early distribution. This directly probes the transfer limit flagged in the critique.

3. **Tool-set transfer to software engineering.** Swap the three tools (search, browser, code) for an SWE tool set (repo navigation, test runner, patch apply) and keep the rest of the recipe, then evaluate against the agentless-prior approach in [Kimi-Dev](/blog/paper-reading/large-language-model/kimi-dev). If the recipe transfers, it is general; if it collapses, the stability was domain-specific. Either answer is valuable.

4. **A context-management policy you can inspect.** The mechanism that retains key info and discards documents is the highest-leverage component and the least described. I would instrument it to log *what* it keeps and drops per iteration, then train a small auxiliary classifier on those decisions to understand whether the policy learned a sensible relevance signal or a brittle heuristic. The +30%-iterations win deserves an interpretability pass.

5. **Partial-rollout staleness accounting.** Since resumed trajectories mix old and new weights, I would build an importance-correction or staleness-bucketing scheme for the gradient and measure whether it improves stability or final score versus the as-described "just resume" approach. This closes the one seam the recipe leaves open, and it is a self-contained systems-plus-algorithm project.

## When to reach for end-to-end agentic RL (and when not to)

Reach for this recipe when three conditions hold together. First, the task is **genuinely long-horizon and tool-dependent** — multi-turn search, verification across sources, iterative synthesis — so that the loop itself is the thing worth learning and a single-turn policy or a static prompt chain leaves capability on the table. Second, you can **define a clean outcome reward**: a robustly extractable ground truth and an automated grader, because without that the whole REINFORCE machinery has nothing to optimize. Third, you have the **systems budget** to run expensive rollouts — the async server, the sandbox, the partial-rollout buffer — because each trajectory here spans 50+ iterations, hundreds of thousands of tokens, and 200+ URLs, and that cost is real even if the blog does not quantify it. When those three line up, the payoff is an agent that adapts to model and tool drift instead of rotting, and that learns recovery behavior you could never have prompted in.

Do not reach for it when any of those legs is missing. If your task is **short-horizon or single-tool**, a well-prompted workflow or a fine-tuned single-turn model will get you there faster and cheaper, and the RL variance is not worth fighting. If you **cannot define an automated outcome reward** — the answer is subjective, or grading requires a human every time — then outcome-based RL has no signal and you are better off with preference data or imitation. And if you are **compute-constrained**, the rollout cost will dominate your budget before you see the gains; the recipe's whole systems layer exists precisely because these rollouts are expensive, and that expense does not disappear at small scale. The honest summary is that Kimi-Researcher shows end-to-end agentic RL works beautifully for autonomous *research* — a domain with long horizons, real tools, and gradable answers — and leaves open, by its own disclosure gaps, exactly how far that recipe travels beyond the benchmarks it was tuned on.

## References

- **Kimi-Researcher official blog / project page** — "Kimi-Researcher: End-to-End RL Training for Emerging Agentic Capabilities", Moonshot AI, June 2025: [moonshotai.github.io/Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/). (Primary source. There is **no formal arXiv technical report** for this work; all method and result claims above are at blog-level granularity.)
- **Kimi-Researcher GitHub (project-page mirror)** — [github.com/MoonshotAI/Kimi-Researcher](https://github.com/MoonshotAI/Kimi-Researcher). (Hosts the project page HTML; no code or technical README as of this writing.)
- **Kimi k1.5 technical report** (the public k-series base-model family the internal base derives from) — [arxiv.org/abs/2501.12599](https://arxiv.org/abs/2501.12599).
- **Humanity's Last Exam (HLE)** benchmark — [agi.safe.ai](https://agi.safe.ai/).
- **FRAMES** benchmark — [arxiv.org/abs/2409.12941](https://arxiv.org/abs/2409.12941).
- **SimpleQA** benchmark — [arxiv.org/abs/2411.04368](https://arxiv.org/abs/2411.04368).

Related reading on this blog:

- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](/blog/paper-reading/reinforcement-learning/kimi-k1-5) — the reasoning-RL recipe and base-model lineage Kimi-Researcher builds on.
- [Kimi K2: Open Agentic Intelligence](/blog/paper-reading/large-language-model/kimi-k2) — the agentic-model line that motivates pushing capability into weights, not scaffolds.
- [Kimi-Dev: Agentless Training as Skill Prior for SWE Agents](/blog/paper-reading/large-language-model/kimi-dev) — a contrasting agent recipe in the software-engineering domain.
- [Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving](/blog/paper-reading/large-language-model/mooncake) — the disaggregate-the-slow-path systems philosophy that the async rollout layer echoes.

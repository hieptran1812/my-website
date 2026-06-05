---
title: "GRPO vs DPO vs PPO: A Decision Guide for Post-Training LLMs"
date: "2026-06-04"
publishDate: "2026-06-04"
description: "A practitioner's decision guide to the three dominant post-training algorithms — when PPO's critic earns its memory, when DPO's offline simplicity wins, and when GRPO's group baseline is the only sane choice."
tags: ["llm", "grpo", "dpo", "ppo", "rlhf", "reinforcement-learning", "preference-optimization", "post-training", "reasoning", "alignment", "trl", "deep-learning"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 50
---

Every few months someone asks me, in some form, the same question: "We have a base model and we want it to be better — should we use DPO, PPO, or GRPO?" And almost every time, the question is malformed. It treats the three as interchangeable knobs on the same machine, as if you could swap `PPO` for `GRPO` in a config and the only thing that changes is a leaderboard number. That is not how it works. These three algorithms consume different data, hold different things in GPU memory, fail in different ways, and answer different questions. Choosing between them is less like picking a hyperparameter and more like choosing whether to build a railway, a highway, or a canal — the right answer is dictated by the terrain long before you start laying track.

This guide is the consolidated decision framework I wish I'd had when I started doing post-training seriously. It assumes you already know roughly what each method is — if you want the from-scratch derivations, I've written dedicated deep dives on [fine-tuning with GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) and [fine-tuning with DPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo). What I want to do here is line them up against each other on the axes that actually determine the choice in a real project: reward signal, data, compute, stability, and policy regime. By the end you should be able to look at a task, name the one or two methods that fit, and — just as important — name the ones that don't and say why.

![The post-training method space: reward-signal type and on-policy rollout budget together pin you to a method before any tuning begins](/imgs/blogs/grpo-vs-dpo-vs-ppo-decision-guide-1.png)

The diagram above is the mental model for this entire article. The two questions that fence in your options are written on the axes: *what kind of reward signal can you build* (a learned reward model, a deterministic verifier, or raw preference pairs), and *can you afford to generate fresh rollouts every training step* (on-policy RL) or not (offline). Pin those two down and most of the decision is already made — PPO lives in the on-policy-with-a-reward-model cell, GRPO in the on-policy-with-a-verifier cell, DPO in the offline-with-preference-pairs cell. The empty and awkward cells matter too: offline RL with a learned reward model needs rollouts you don't have, and an offline verifiable task collapses into plain rejection-sampling SFT. The rest of this article is a tour of that grid, one axis at a time, ending with a decision tree you can actually run in your head.

## They Are Not Three Flavors of the Same Thing

The single most common mistake I see — in blog posts, in team discussions, in interview answers — is treating PPO, DPO, and GRPO as points on a single "complexity" slider, where DPO is the easy one, PPO is the hard one, and GRPO is the new one. That framing predicts the wrong choices, because it hides the thing that actually varies: the *signal* each one consumes.

| Assumption | The naive view | The reality |
|---|---|---|
| "They all align a model to preferences" | Pick whichever is easiest to run | PPO and GRPO optimize a *reward*; DPO optimizes a *preference contrast*. The objective is different, not just the implementation. |
| "GRPO is just a cheaper PPO" | Always use GRPO to save memory | GRPO drops the value network, which only works when your reward is dense enough that a group mean is a good baseline — i.e., verifiable tasks. On sparse learned rewards, the critic earns its keep. |
| "DPO is RLHF without the RL" | DPO replaces PPO everywhere | DPO is *offline*. The moment your policy drifts away from the data that generated the preferences, the gradient is describing a model that no longer exists. |
| "More compute = better choice" | Pick the heaviest method you can afford | The heaviest method (PPO) is frequently the wrong one. If you have a verifier, GRPO matches or beats PPO at half the memory. |
| "The choice is about the algorithm" | Compare the loss functions | The choice is about the *signal you can build*. The algorithm is downstream of the data and reward you have access to. |

The throughline: **the algorithm is the last thing you choose, not the first.** You choose your reward signal, you choose whether you can pay for on-policy rollouts, and those two decisions narrow you to one or two algorithms. Picking the algorithm first and then trying to manufacture a reward signal to fit it is how teams end up six weeks into a PPO run with a reward model that's been hacked into uselessness, or two epochs into a DPO run on preference data that no longer resembles anything the model generates.

> If you cannot describe, in one sentence, where your reward signal comes from, you are not ready to choose a post-training algorithm.

I'll keep coming back to that sentence, because almost every bad post-training decision I've watched unfold traces to skipping it. The team that "wanted to try RL" without a verifier. The team that ran DPO on preferences harvested from a model three generations stronger than theirs. The team that built a reward model, watched it get gamed, and concluded "RL doesn't work" when the real conclusion was "our reward signal was weak." The algorithm was never the problem.

## First Principles: What Each Method Actually Solves

Before the axes, let's nail down what problem each of the three was designed to solve, because the design intent explains every downstream tradeoff.

**PPO (Proximal Policy Optimization)** is a general-purpose policy-gradient RL algorithm, borrowed wholesale from the RL literature and adapted for language models in the InstructGPT/RLHF pipeline. Its job is to maximize an arbitrary scalar reward while not straying too far from a reference policy. It makes no assumption about where the reward comes from — it could be a learned reward model, a human, or a rule. To do this stably it learns a *value function* (a critic) that estimates the expected reward from any partial generation, so it can compute a low-variance advantage for each token. PPO is the most general and the most expensive: it is the right tool when your reward is a noisy, learned scalar and you need every bit of variance reduction you can get. Its lineage is robotics and game-playing, where the reward is genuinely a black box and the policy must explore a huge action space — that generality is both why it works on open-ended language alignment and why it carries so much machinery.

**DPO (Direct Preference Optimization)** solves a narrower problem: given a static dataset of preference pairs (response A is better than response B for this prompt), produce a policy that prefers the better responses. Its key insight is mathematical — the optimal policy under a KL-constrained reward objective has a closed form, and you can rearrange that closed form to express the reward implicitly in terms of the policy itself. That lets you skip the reward model and the RL loop entirely and train with a single supervised-style classification loss on the pairs. DPO is the right tool when you already have preference data and you want alignment without standing up an RL system. It was, in a real sense, a *systems* insight as much as a statistical one: it collapsed a four-model online RL pipeline into a two-model offline supervised one, and that collapse is what made preference alignment accessible to anyone with a GPU.

**GRPO (Group Relative Policy Optimization)** solves PPO's expense problem for a specific and increasingly important case: tasks with a *verifiable* reward. DeepSeek's insight was that if you sample a whole group of responses to the same prompt and score each one, the *mean reward of the group* is a perfectly good baseline — you don't need a learned critic to tell you whether a response is above or below average, you can just measure it. Drop the value network, normalize each response's reward against its group, and you get most of PPO's variance reduction at half the memory. GRPO is the right tool when you can verify correctness — math answers, unit tests, format compliance — and want to train reasoning at scale. The deeper point is that GRPO is *not* a general PPO replacement; it's a PPO specialization that trades the critic's generality for a measured baseline, and that trade is only sound when the reward is dense and cheap enough that a group of 8–16 samples actually spans the quality range.

These three intents — *maximize an arbitrary learned reward*, *match a static preference set*, *climb a verifiable reward cheaply* — are genuinely different problems. The whole guide flows from there.

| Dimension | PPO | DPO | GRPO |
|---|---|---|---|
| Origin | RL literature → InstructGPT | preference learning, 2023 | DeepSeek, 2024–25 |
| Optimizes | arbitrary scalar reward | preference contrast | verifiable group-relative reward |
| Baseline | learned critic (GAE) | none (contrastive) | group mean |
| Regime | on-policy | offline | on-policy |
| Best at | open-ended alignment | preference alignment | verifiable reasoning |

## The Objectives Side by Side

Let's make the abstraction concrete by looking at what each method does on a single training step. The picture below is three parallel assembly lines: same raw material at the top (a prompt or a preference pair), three different ways of turning it into a gradient at the bottom.

![Three objectives, three update signals: PPO scores rollouts with a learned critic, GRPO replaces the critic with a group mean, DPO skips rollouts entirely](/imgs/blogs/grpo-vs-dpo-vs-ppo-decision-guide-2.png)

Read each column top to bottom. PPO rolls out the policy, scores the rollout with a reward model, computes a token-level advantage using a learned critic and GAE, and applies a clipped policy-gradient update. GRPO rolls out the policy $G$ times, scores each with a verifier, normalizes the rewards within the group to get an advantage, and applies a clipped policy-gradient update with a KL penalty. DPO never rolls anything out — it takes a fixed pair, computes the log-probability ratio of each response under the policy versus a frozen reference, and pushes the chosen response's implicit reward above the rejected one's through a logistic loss.

Now the math, with every symbol defined.

**PPO** maximizes the clipped surrogate objective. For a token $t$ with policy ratio $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ and advantage estimate $\hat{A}_t$:

$$\mathcal{L}_{\text{PPO}}(\theta) = -\mathbb{E}_t\left[\min\left(r_t(\theta)\,\hat{A}_t,\ \text{clip}\big(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\big)\,\hat{A}_t\right)\right]$$

Here $\pi_\theta$ is the policy being trained, $\pi_{\theta_{\text{old}}}$ is the policy that generated the rollout, $\epsilon$ is the clip range (typically 0.2), and $\hat{A}_t$ is the advantage, usually computed via Generalized Advantage Estimation from a learned value function $V_\psi$. The clip is the "proximal" part: it prevents any single update from moving the policy ratio outside $[1-\epsilon, 1+\epsilon]$, which is what keeps online RL from diverging. That value function is a second full-sized network — the critic — and it's the reason PPO is expensive.

**GRPO** keeps the same clipped surrogate but replaces the GAE advantage with a group-relative one. Sample $G$ responses $\{o_1, \dots, o_G\}$ for a prompt, score each with reward $R_i$, and set:

$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_1, \dots, R_G\})}{\text{std}(\{R_1, \dots, R_G\})}$$

Every token in response $o_i$ gets the same advantage $\hat{A}_i$. The objective then adds an explicit KL penalty against a reference policy $\pi_{\text{ref}}$:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\big(r_i\hat{A}_i,\ \text{clip}(r_i, 1-\epsilon, 1+\epsilon)\hat{A}_i\big)\right] + \beta\, D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})$$

No $V_\psi$ anywhere. The group mean *is* the baseline. $\beta$ controls how hard the KL leash pulls. Note the two normalizations baked into $\hat{A}_i$ — subtracting the mean and dividing by the std — because both turn out to be biased, and removing them is precisely what the [2025 GRPO variants](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo) are about.

**DPO** throws out the RL machinery entirely. Given a preference pair $(y_w, y_l)$ — winner and loser — for prompt $x$, and a frozen reference policy $\pi_{\text{ref}}$:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\log\sigma\left(\beta\log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta\log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$$

where $\sigma$ is the logistic function and $\beta$ controls how much the policy is allowed to deviate from the reference. The quantity $\beta\log\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ is the *implicit reward* — DPO's central trick is that this expression is exactly the reward whose optimal KL-constrained policy is $\pi_\theta$, so optimizing the contrast of implicit rewards is equivalent to RLHF on the corresponding explicit reward, without ever materializing it.

Here is a useful way to feel the difference. Take a prompt and a single response that the verifier marks correct. Under **GRPO**, that response's contribution depends entirely on its *group*: if the other seven samples were also correct, its advantage is zero and it teaches nothing; if the other seven were wrong, its advantage is large and positive. The exact same response, same tokens, can produce a strong gradient or no gradient at all, depending only on its peers. Under **PPO**, that response's gradient depends on the critic's estimate — a learned number that the same response produces regardless of what else was sampled. Under **DPO**, the response only enters training if it's half of a pair, and its gradient depends on the *other* response in the pair and on how wrong the model currently is about which is better. Three different notions of "how much should this response move the weights," and they disagree constantly.

The three losses share a family resemblance — there's a $\beta$, there's a reference policy, there's a ratio — but the inputs are different (rollouts vs. pairs), the baseline is different (critic vs. group mean vs. nothing), and the optimization regime is different (online vs. offline). Those differences are the whole story.

## What the Gradient Actually Moves

It helps to ask a blunt question: when you take a gradient step, what gets reinforced and what gets suppressed? The answer is different enough across the three that it's worth a figure of its own.

![What the gradient actually moves: DPO contrasts two fixed responses; PPO and GRPO reweight tokens by an advantage recomputed every step](/imgs/blogs/grpo-vs-dpo-vs-ppo-decision-guide-3.png)

PPO reinforces tokens with high advantage and suppresses tokens with low advantage, where "advantage" is the critic's verdict on whether that token did better than expected. DPO reinforces the entire chosen response and suppresses the entire rejected response, weighted by how badly the model is currently getting the preference wrong — the gradient is largest exactly when the implicit reward gap points the wrong way. GRPO reinforces responses that scored above their group's mean and suppresses those below it.

The practically important difference hides in the phrase "recomputed every step." PPO and GRPO recompute the advantage on freshly generated rollouts at every iteration — the signal always reflects the current policy. DPO's pair is fixed forever; the only thing that changes across steps is the policy's log-probabilities on that frozen pair. This is the seed of DPO's signature failure mode, *likelihood displacement*, which we'll get to. For now, hold onto the asymmetry: **two of these methods score a moving target, one scores a photograph.**

A second subtlety: DPO's gradient operates at the sequence level — chosen up, rejected down, full stop. PPO operates at the token level via the critic. GRPO sits in between: the advantage is per-response (sequence level) but it's applied to every token's policy-gradient term, and the loss aggregation across tokens turns out to matter enormously — that aggregation choice is the entire subject of the [follow-up post on GRPO variants](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo). For this guide, the takeaway is that the *granularity* of credit assignment goes PPO (per token, learned) → GRPO (per response, measured) → DPO (per response, contrastive). The finer the credit assignment, the more machinery you pay for: PPO's per-token credit needs a critic; GRPO's per-response credit needs only a group; DPO's per-response contrast needs only a pair.

## The Data Axis: What Each Method Eats

Algorithms are downstream of data. The fastest way to eliminate two of the three options is to look honestly at what data you have and where your reward comes from.

![Where the training signal comes from: PPO buys a learned reward model, GRPO rents a rule verifier, DPO ships preference pairs with no model at all](/imgs/blogs/grpo-vs-dpo-vs-ppo-decision-guide-4.png)

PPO needs a **reward model** — a separate network, usually trained on the same kind of preference pairs DPO would consume directly, that maps any (prompt, response) to a scalar. Building this is a whole sub-project: you collect preferences, you train the reward model, you validate that it isn't trivially hackable, and only then do you start PPO. The reward model is both PPO's superpower (it generalizes to responses never seen in the preference data) and its Achilles' heel (the policy will find and exploit every flaw in it). A reward model is itself a research artifact — [evaluating reward models](/blog/paper-reading/ai-interpretability/rethinking-reward-model-evaluation-are-we-barking-up-the-wrong-tree) is a live subfield precisely because a reward model that looks great on held-out pairs can still be catastrophically gameable under optimization pressure.

DPO needs **preference pairs** and nothing else. No reward model, no verifier, no rollout infrastructure. If you have a dataset of (prompt, chosen, rejected) triples — from human annotators, from a stronger model's judgments, from thumbs-up/thumbs-down production logs — you can run DPO this afternoon. This is why DPO ate the open-source alignment world: the barrier to entry is a dataset and a single GPU. A DPO row is about as simple as data gets:

```json
{
  "prompt": "Explain why the sky is blue to a 10-year-old.",
  "chosen": "Sunlight is made of all the colors mixed together...",
  "rejected": "The sky is blue due to Rayleigh scattering of electromagnetic radiation..."
}
```

GRPO needs **prompts with verifiable answers**. A math problem with a known solution, a coding task with unit tests, a structured-output task with a schema to validate against. The "reward model" is a deterministic function — `extract_answer(response) == gold_answer` — which is free to evaluate, impossible to hack in the reward-model sense, and gives a clean binary (or graded) signal. A GRPO row carries the gold answer instead of a paired response:

```json
{
  "prompt": "What is the remainder when 7^100 is divided by 13?",
  "answer": "9"
}
```

The constraint is real, though: if you can't write a verifier, GRPO has nothing to optimize. And "can you write a verifier" is a deceptively deep question — checking a final numeric answer is easy, but checking a proof, an essay, or a design document is not, and faking it with a weak heuristic verifier is how you train a model to satisfy the heuristic instead of solving the problem.

| Method | Training input | Reward source | Needs gold answer? | Data collection |
|---|---|---|---|---|
| PPO | prompts | learned reward model | no | online rollouts + RM training |
| DPO | (chosen, rejected) pairs | human / model A-B labels | no | static, pre-collected |
| GRPO | prompts + gold answers | rule verifier (0/1 or graded) | yes | online rollouts |

There's a clarifying way to see the relationship between DPO and PPO here: **they often consume the same raw preference data, but at different points in the pipeline.** PPO spends that data to train a reward model and then optimizes against the model. DPO optimizes against the data directly. PPO's indirection buys generalization (the RM scores novel responses) at the cost of a hackable intermediary; DPO's directness buys simplicity at the cost of being frozen to the responses in the dataset. If you find yourself debating DPO vs PPO on preference data, you are really debating whether the reward model's generalization is worth its hackability and its cost — that is the entire question, and the next few axes will help you answer it.

## The Compute and Memory Axis

Now the axis that decides whether your run fits on the hardware you have. Post-training is memory-bound long before it's compute-bound, and the methods differ by roughly a factor of two in how many full-sized model copies they keep resident.

![What sits in GPU memory: PPO holds four model-sized tensors at once; DPO and GRPO hold two, but GRPO pays again in G parallel rollouts](/imgs/blogs/grpo-vs-dpo-vs-ppo-decision-guide-5.png)

Count the model-sized objects each method holds in memory during a step:

- **PPO** holds four: the **policy** (trained), a **frozen reference** (for the KL term), the **reward model** (resident, scoring rollouts), and the **critic / value network** (trained, the same size as the policy). Two of those four are being trained, which means two sets of optimizer states and gradients. This is why a 7B PPO run can feel like training a 14B model.
- **DPO** holds two: the **policy** (trained) and the **frozen reference**. No reward model, no critic. And because DPO is offline, there's no rollout buffer either — you're just doing forward/backward passes on pre-tokenized pairs. DPO is by far the lightest of the three. In practice you can even precompute the reference log-probabilities once and drop the reference model entirely, leaving a single resident network.
- **GRPO** holds two model-sized networks — **policy** and **frozen reference** — with no critic and no reward model (the verifier is a CPU function). But it pays a different tax: it generates $G$ rollouts per prompt (typically 8–16), so the activation memory and the generation time per step scale with $G$. GRPO trades the critic's *parameter* memory for the group's *activation and throughput* cost.

| Method | Resident networks | Trained networks | Per-step rollouts | Rough relative cost |
|---|---|---|---|---|
| PPO | 4 (policy, ref, RM, critic) | 2 | 1 | highest — ~2× the model footprint |
| DPO | 2 (policy, ref) → can be 1 | 1 | 0 (offline) | lowest |
| GRPO | 2 (policy, ref) | 1 | G = 8–16 | medium — light on params, heavy on generation |

A quick back-of-the-envelope makes the factor-of-two concrete. Take a 7B model in bf16 with the Adam optimizer:

```python
def trained_network_gb(params_billions, bytes_per_param=2, adam_state_bytes=8):
    params_gb = params_billions * bytes_per_param          # bf16 weights
    grads_gb = params_billions * bytes_per_param           # bf16 gradients
    adam_gb = params_billions * adam_state_bytes           # fp32 m + v
    return params_gb + grads_gb + adam_gb

per_trained = trained_network_gb(7)                        # about 84 GB
frozen_gb = 7 * 2                                          # a frozen network is just weights, 14 GB

ppo_gb = 2 * per_trained + 2 * frozen_gb                   # policy + critic trained, ref + RM frozen
dpo_gb = 1 * per_trained + 1 * frozen_gb                   # policy trained, ref frozen
grpo_gb = 1 * per_trained + 1 * frozen_gb                  # same residents as DPO; rollouts cost activations

print(ppo_gb, dpo_gb, grpo_gb)                             # ~196, ~98, ~98 (before activations)
```

Those numbers are rough — they ignore activation memory, the rollout buffer, and the fact that GRPO's $G$ generations inflate activations — but the shape is right: PPO's optimizer footprint is roughly double DPO's and GRPO's because it trains two networks instead of one and keeps two more resident. In practice that's the difference between fitting on a single 80 GB card with offloading (DPO/GRPO with LoRA) and needing a multi-GPU node with sharding (full PPO). The practical consequence: if you are memory-starved and have a verifier, GRPO is the obvious win — you get on-policy RL without the critic. If you are memory-starved and only have preferences, DPO is the only one that fits comfortably. PPO is the method you choose when you have both the hardware *and* a specific reason the critic is worth it.

One nuance that bites teams: GRPO's generation cost is *latency*, not just memory, and it's on the critical path. Every training step waits for $G$ full generations to complete before it can compute advantages. If your generation backend is slow, GRPO steps are slow regardless of how much compute the backward pass uses — which is exactly why the serious GRPO stacks couple a fast inference engine (vLLM, SGLang) to the trainer and overlap generation with optimization. DPO has no generation in the loop at all, so its step time is simply forward+backward on a batch of pairs, the same as SFT.

## The Stability Axis: How Each One Fails

You will not get a clean run on the first try. Knowing each method's signature pathology — and its standard mitigation — is what separates a debuggable training run from a mysterious one.

![How each method fails: PPO hacks reward, DPO displaces likelihood, GRPO collapses entropy, and each has a known fix](/imgs/blogs/grpo-vs-dpo-vs-ppo-decision-guide-6.png)

**PPO's pathology is reward hacking and KL blow-up.** Because PPO optimizes a *learned* reward model as hard as it can, it finds the model's blind spots. The classic symptom: reward climbs beautifully while sample quality cratered ten thousand steps ago, because the policy discovered that the reward model loves responses that end with a polite sign-off, or that are exactly 312 tokens long, or that repeat the question back. The other PPO failure is the KL term going unstable — the policy lunges away from the reference, the ratio explodes past the clip range, and the run diverges. Mitigations: a well-tuned KL coefficient (often adaptive), a conservative clip range, reward-model ensembles or regularization, and early stopping on a held-out *real* metric rather than the reward. The cruel part of reward hacking is that your dashboard looks *better* as the model gets worse — the reward curve is a liar, and the only defense is evaluating against something the policy can't optimize against directly.

**DPO's pathology is likelihood displacement.** This one is subtle and counterintuitive. DPO's loss only cares about the *gap* between chosen and rejected log-probabilities. It can reduce that gap by pushing the rejected response down — which is what you want — or by an unintended route where *both* the chosen and rejected probabilities fall, just the rejected one faster. The model learns to dislike the rejected response by becoming generally less confident, and the absolute probability of your preferred responses can actually decrease over training. The symptom: DPO "works" by the loss and by pairwise win-rate, but the model becomes oddly hedgy or starts producing degenerate completions outside the preference distribution. Mitigations: mix in an SFT loss on the chosen responses (the widely used trick, exposed as `rpo_alpha` in TRL), use a variant like DPOP or a length-normalized loss, and watch the *absolute* chosen log-prob, not just the margin. There is a whole literature of DPO variants — IPO, KTO, SimPO, and the [temporal-decay](/blog/paper-reading/ai-interpretability/earlier-tokens-contribute-more-learning-direct-preference-optimization-from-temporal-decay-perspective) and [token-level importance](/blog/paper-reading/ai-interpretability/tis-dpo-token-level-importance-sampling-for-direct-preference-optimization-with-estimated-weights) refinements — and most of them exist to patch some corner of this exact failure.

**GRPO's pathology is entropy collapse and the zero-gradient trap.** Because GRPO's advantage is a *group-relative* signal, two degenerate cases bite. First, if every response in a group gets the same reward — all correct on an easy prompt, or all wrong on a hard one — the standard deviation is zero, the advantage is zero (or numerically unstable), and that prompt contributes no gradient. Spend a batch on prompts that are all-easy or all-hard and you've wasted it. Second, GRPO can collapse entropy: the policy becomes overconfident, sampling diversity within the group drops, the group stops exploring, and learning stalls. Mitigations: dynamic sampling (filter out prompts whose group is all-correct or all-wrong — the DAPO trick), an entropy bonus or KL leash, and curating prompt difficulty so groups land in the productive middle. The zero-gradient trap is the single most common reason a first GRPO run "doesn't learn" — the loss isn't broken, it's that half your batch had no spread and silently contributed nothing.

| Method | Signature failure | What you see | Standard mitigation |
|---|---|---|---|
| PPO | reward hacking, KL blow-up | reward up, quality down; divergence | KL tuning, clip, RM regularization, early stop on real metric |
| DPO | likelihood displacement | win-rate up, chosen prob down; hedging | SFT mix, length normalization, watch absolute log-probs |
| GRPO | entropy collapse, zero-gradient groups | stalled learning, wasted batches | dynamic sampling, entropy/KL control, difficulty curation |

The meta-lesson: **each method's strength is also the source of its failure.** PPO's reward model gives it generalization and gives it something to hack. DPO's offline simplicity gives it cheapness and freezes it to a static target. GRPO's group baseline gives it critic-free efficiency and makes it fragile when the group is uniform. There is no free lunch — there is only choosing which failure mode you'd rather debug.

## On-Policy vs Off-Policy: The Distribution-Shift Problem

The single deepest difference between DPO on one side and PPO/GRPO on the other is *on-policy versus off-policy*, and it deserves its own treatment because it's the axis people most often underestimate.

![Off-policy DPO vs on-policy RL: DPO learns from a frozen snapshot of responses while on-policy RL always scores the current model](/imgs/blogs/grpo-vs-dpo-vs-ppo-decision-guide-7.png)

PPO and GRPO are **on-policy**: every gradient step is computed on responses the *current* policy just generated. The reward and advantage always describe the model as it is right now. When the policy improves, the next batch of rollouts reflects that improvement, and the training signal moves with it. This is expensive — you pay for generation every step — but it means the gradient is never stale.

DPO is **off-policy**: the preference pairs were generated by some other model (or an earlier version of yours, or a mix), frozen into a dataset, and replayed over and over. At the start of training, when the policy is close to whatever produced the data, the gradient is meaningful. But as the policy moves, it diverges from the data distribution, and DPO keeps optimizing a contrast on responses the current model would never produce. You're sharpening a preference between two restaurant meals the chef stopped cooking months ago.

This is why **on-policy DPO** and **iterative DPO** exist — you periodically regenerate the preference data with the current policy (and re-label it with a judge or reward model), pulling DPO partway toward the on-policy regime. Several strong open recipes (the iterative rounds in Llama-3's post-training, for instance) do exactly this. But vanilla DPO, run once on a static dataset, is fully off-policy, and the more steps you take, the more the staleness bites. If you ever see DPO's metrics look great for the first epoch and then plateau or degrade, distribution shift is your prime suspect. There's a subtle theoretical bridge here too: the [connection between imitation learning and RLHF](/blog/paper-reading/ai-interpretability/on-a-connection-between-imitation-learning-and-rlhf) shows that off-policy preference learning is, in a precise sense, imitating a fixed distribution — and imitation degrades exactly when the learner outgrows the demonstrator.

The flip side: off-policy is a *feature* when you can't afford rollouts. If generating samples is prohibitively expensive — huge models, slow verifiers, tight budgets — DPO's ability to learn from a fixed dataset is exactly what makes it viable. On-policy purity is worth paying for only when the staleness actually hurts your task, and for short alignment runs on well-matched data, it often doesn't. The decision isn't "on-policy good, off-policy bad" — it's "how far will my policy move during training, and does my data move with it?"

## Reward Availability: The Three Regimes

Stepping back, almost the entire decision reduces to a single question: *what reward signal can you actually build?* The three regimes map cleanly onto the three methods.

![Reward regimes map to methods: verifiable rewards unlock critic-free GRPO; preferences feed DPO directly or train a reward model for PPO](/imgs/blogs/grpo-vs-dpo-vs-ppo-decision-guide-9.png)

Trace the graph. A **verifiable** reward — a function that says correct or incorrect with certainty — flows straight to GRPO, because a clean dense-enough signal is exactly what makes the group baseline work and the critic unnecessary. A **preference** signal — humans or models saying A is better than B — has two destinations: feed it *directly* to DPO, or spend it to *train a reward model* and then run PPO against that model. That fork is the crux of the DPO-vs-PPO debate for preference data.

When should preference data go to DPO versus through a reward model to PPO? The reward-model route (PPO) wins when:

- You expect to run for a long time and the reward model's ability to score *novel* responses keeps the signal fresh as the policy explores.
- You can invest in making the reward model robust (ensembles, regularization) and you have the infrastructure for online RL.
- You need fine-grained control — shaping the reward, combining multiple reward sources, gating on safety classifiers.

The direct route (DPO) wins when:

- Your preference data already covers the response distribution you care about, and you're doing a focused alignment pass rather than open-ended exploration.
- You don't want to build, validate, and babysit a reward model and an RL loop.
- You value reproducibility and a stable, supervised-style training curve.

And when you have *neither* a verifier nor preferences — only demonstrations — you're not in post-training-with-rewards territory at all; you want SFT or rejection-sampling fine-tuning first, and you come back to these three once you can generate a preference or verification signal. A surprising amount of "we need RL" turns out, on inspection, to be "we need better SFT data and then maybe a preference pass" — the regime question saves you from reaching for heavy machinery you don't yet have the signal to drive.

## The Decision Tree

Here is the whole guide compressed into something you can run in your head before you write a single line of training code.

![Which post-training method to reach for: a verifier sends you to GRPO, raw preference pairs to DPO, a learned reward model with budget to PPO](/imgs/blogs/grpo-vs-dpo-vs-ppo-decision-guide-8.png)

Start at the top — *what signal do you have?* — and follow the branch:

1. **Do you have a verifiable answer?** Math with known solutions, code with tests, structured output with a schema, anything where a function can score correctness. If yes → **GRPO**. You get on-policy RL, no critic, no reward model, and a signal that can't be hacked the way a learned reward model can. This is the default for reasoning training in 2025–2026.

2. **Do you have only preference pairs?** (chosen, rejected) data from humans or a judge model, and you want alignment without standing up an RL system. If yes → **DPO**. Cheapest to run, most reproducible, fits on the least hardware. Mix in SFT to dodge likelihood displacement.

3. **Do you have a learned reward model and the GPU budget — and a reason you need it?** A robust reward model, online-RL infrastructure, and a task where the reward model's generalization to novel responses genuinely matters. If yes → **PPO**. Maximum control and the strongest signal when the reward is a noisy learned scalar, at the highest cost.

The tree is deliberately ordered by *how often the answer is yes for a well-scoped project*. Most reasoning teams in this era have a verifier and reach for GRPO. Most alignment teams have preferences and reach for DPO. PPO is the specialist's choice — when you've decided, deliberately, that the critic and the reward model earn their substantial cost.

There's a subtlety the tree flattens: these are not mutually exclusive across a *pipeline*. The strongest open recipes chain them — SFT, then DPO for broad alignment, then GRPO (or RLVR more generally) for verifiable reasoning. The tree answers "which method for *this stage*," not "which method forever." Keep that in mind as we look at the code and then the production case studies, almost all of which use more than one of the three.

## Putting It Together: Code

Let's make the three concrete with the [TRL](/blog/machine-learning/open-source-library/trl-lib) library, which implements all three behind a near-uniform trainer API. Install once:

```bash
pip install "trl>=0.15" "transformers>=4.48" "peft>=0.14" accelerate
```

Start with PPO, and notice how much it has to wire up. PPO's constructor takes *four* models — that verbosity is the whole point, it's the memory cost made literal in code:

```python
from trl import PPOConfig, PPOTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

model_id = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)

policy = AutoModelForCausalLM.from_pretrained(model_id)            # trained
ref_policy = AutoModelForCausalLM.from_pretrained(model_id)        # frozen, for KL
reward_model = AutoModelForSequenceClassification.from_pretrained( # learned RM
    "trl-internal-testing/reward-model-7b", num_labels=1)
value_model = AutoModelForSequenceClassification.from_pretrained(  # critic, same size
    model_id, num_labels=1)

ppo_cfg = PPOConfig(
    learning_rate=1e-6,
    batch_size=64,
    mini_batch_size=8,
    num_ppo_epochs=4,
    cliprange=0.2,          # epsilon in the clipped surrogate
    kl_coef=0.05,           # leash to the reference policy
    gamma=1.0, lam=0.95,    # GAE discount + lambda for the critic
)
ppo_trainer = PPOTrainer(
    ppo_cfg, tok, policy, ref_policy, reward_model, value_model,
    train_dataset=ppo_data,
)
```

DPO, by contrast, needs a model, a reference (which TRL can create implicitly), and a dataset of pairs. The whole RL apparatus is gone:

```python
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM

dpo_cfg = DPOConfig(
    learning_rate=5e-7,
    beta=0.1,               # KL strength: lower = freer to move from the reference
    loss_type="sigmoid",    # vanilla DPO; "ipo", "kto_pair", "sppo_hard" are variants
    max_length=2048,
    max_prompt_length=1024,
    rpo_alpha=1.0,          # auxiliary SFT loss on y_w — the anti-displacement trick
)
dpo_trainer = DPOTrainer(
    model=AutoModelForCausalLM.from_pretrained(model_id),
    args=dpo_cfg,
    train_dataset=dpo_pairs,   # columns: prompt, chosen, rejected
    processing_class=tok,
)
```

GRPO needs a model, a dataset of *prompts*, and one or more reward functions — no reward model, no critic. The reward functions are pure Python, and that's the point: they're the verifier:

```python
import re
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM

def correctness_reward(completions, answer, **kwargs):
    """+1 if the boxed answer matches the gold solution, else 0."""
    rewards = []
    for completion, gold in zip(completions, answer):
        match = re.search(r"\\boxed\{(.+?)\}", completion)
        pred = match.group(1).strip() if match else None
        rewards.append(1.0 if pred == gold.strip() else 0.0)
    return rewards

def format_reward(completions, **kwargs):
    """Small shaping reward for emitting the think/answer structure."""
    pat = re.compile(r"<think>.*?</think>.*?\\boxed\{.+?\}", re.DOTALL)
    return [0.2 if pat.search(c) else 0.0 for c in completions]

grpo_cfg = GRPOConfig(
    learning_rate=1e-6,
    num_generations=8,          # G — the group size; the group mean is the baseline
    max_prompt_length=512,
    max_completion_length=2048,
    beta=0.04,                  # KL penalty against the reference
    epsilon=0.2,                # clip range
)
grpo_trainer = GRPOTrainer(
    model=AutoModelForCausalLM.from_pretrained(model_id),
    reward_funcs=[correctness_reward, format_reward],
    args=grpo_cfg,
    train_dataset=math_prompts,   # columns: prompt, answer
)
```

The asymmetry in the code is the asymmetry in the methods. PPO's constructor takes four models; DPO's takes a dataset of pairs and a $\beta$; GRPO's takes prompts and reward functions. If your data doesn't look like the third argument of one of these trainers, that trainer isn't your method.

Finally, the decision tree as code you could actually drop into a project's planning notebook:

```python
def choose_post_training_method(signal: dict) -> str:
    """
    signal keys:
      has_verifier        can a function score correctness?
      has_preferences     do we have (chosen, rejected) pairs?
      has_reward_model    a trained, validated scalar RM?
      can_afford_rollouts GPU budget for online generation?
      long_horizon        many steps of open-ended exploration?
    """
    if signal["has_verifier"] and signal["can_afford_rollouts"]:
        return "GRPO"   # on-policy RL, no critic, unhackable signal

    if signal["has_reward_model"] and signal["can_afford_rollouts"] and signal["long_horizon"]:
        return "PPO"    # the RM's generalization to novel responses earns the critic

    if signal["has_preferences"] and signal["can_afford_rollouts"]:
        return "online-DPO / iterative-DPO"   # regenerate pairs to fight staleness

    if signal["has_preferences"]:
        return "DPO"    # offline, cheap, reproducible; add SFT mix

    return "collect a signal first (SFT / rejection sampling), then revisit"
```

It's a toy, but the ordering encodes the real priority: a verifier beats everything (GRPO), a robust reward model with budget and a long horizon justifies PPO, and preferences fall back to DPO. The last line is the one teams skip and regret — if you can't fill in the signal dict, no algorithm will save you.

## Cross-Cutting Concerns: Cost, Labeling, and Infra

Three things cut across the choice and deserve a moment.

**Labeling cost.** DPO and PPO both ultimately rest on preference labels (PPO via the reward model). Those labels are expensive — human annotation is slow and noisy, and LLM-as-judge labels carry their own biases (the [one-token-to-fool](/blog/paper-reading/ai-interpretability/one-token-to-fool-llm-as-a-judge) failure mode is a cautionary tale about trusting judges blindly). GRPO's verifier sidesteps labeling entirely *when a verifier exists* — that's a large part of why verifiable-reward RL took over reasoning. If your task can be made verifiable, you trade a labeling budget for an engineering effort, and that's usually a great trade. The catch is that a lot of valuable behavior — helpfulness, tone, safety, judgment — is genuinely not verifiable, and for those, preference labels are not a cost you can engineer away.

**Infrastructure.** DPO runs on a standard supervised-training stack — it's barely more complex than SFT. PPO and GRPO need a *generation* loop tightly coupled to a *training* loop, which is a genuinely harder system: you need fast inference (vLLM/SGLang) feeding fresh rollouts to the trainer without starving the GPUs. Libraries like [verl](/blog/machine-learning/open-source-library/verl-rlhf-library-deep-dive) exist precisely to manage that rollout-train interleaving at scale, including the weight-resharding dance between the training engine and the inference engine. If you don't have that infrastructure and can't adopt it, DPO's offline simplicity isn't just convenient — it's the only thing that ships this quarter.

**Total cost of ownership.** The honest cost comparison isn't GPU-hours per step; it's the whole pipeline. PPO's TCO includes building and validating a reward model, debugging reward hacking, and running an RL system. DPO's TCO is collecting pairs and watching for displacement. GRPO's TCO is writing verifiers and curating prompt difficulty. Different teams have different cheap resources — a team with strong data engineers and verifiable tasks should reach for GRPO; a team with a preference dataset and a single node should reach for DPO; a team with deep RL infrastructure and a hard alignment target should reach for PPO. The mistake is comparing the three on the narrow axis of "training cost per step" when the real cost lives in the data, the reward, and the operational burden.

## Hyperparameters That Matter Most

Each method has one or two hyperparameters that dominate the others — get these wrong and nothing else you tune will save the run. Here's the short list per method, with the values I reach for as a starting point and what each one trades off.

**For DPO, $\beta$ is the master dial.** It controls how far the policy is allowed to move from the reference. A high $\beta$ (0.5) keeps the policy close to the reference — safe, but the alignment barely takes. A low $\beta$ (0.01) lets the policy move freely — strong alignment, but you risk drifting off-distribution and amplifying likelihood displacement. The sweet spot for most chat alignment is $\beta \in [0.1, 0.3]$. The second dial is the learning rate, which for DPO is brutally small: $5\times10^{-7}$ to $1\times10^{-6}$ is typical, an order of magnitude below SFT, because DPO's gradient is sharp and a too-large step blows past the preference and into degeneration.

**For GRPO, the group size $G$ and the KL coefficient $\beta$ matter most.** $G$ controls the quality of the group baseline: too small (2–4) and the mean is a noisy baseline; too large (32+) and you pay generation cost for diminishing variance reduction. $G \in [8, 16]$ is the standard range. The KL coefficient $\beta$ leashes the policy to the reference — many strong recipes set it very low (0.04) or even to zero once they trust the verifier, because a verifiable reward is less prone to the runaway exploitation that the KL term exists to prevent. The clip range $\epsilon$ (0.2) and the learning rate ($1\times10^{-6}$) round out the set.

**For PPO, the KL coefficient and the clip range govern stability.** PPO's KL coefficient is often *adaptive* — it grows when the measured KL exceeds a target and shrinks when it's below, holding the policy at a roughly constant distance from the reference. The clip range $\epsilon$ (0.2) bounds per-step movement. The GAE parameters $\gamma$ (1.0 for full-sequence credit) and $\lambda$ (0.95) shape how the critic assigns credit across tokens. PPO has the most knobs and the least forgiving stability profile, which is a real part of its cost.

| Hyperparameter | PPO | DPO | GRPO |
|---|---|---|---|
| Learning rate | $1\times10^{-6}$ | $5\times10^{-7}$ | $1\times10^{-6}$ |
| KL / $\beta$ | adaptive KL, target ~6 | $\beta = 0.1$–$0.3$ | $\beta = 0.0$–$0.04$ |
| Clip $\epsilon$ | 0.2 | n/a | 0.2 |
| Group / batch | rollout batch 64 | pairs per batch | $G = 8$–$16$ |
| Most sensitive | KL coefficient | $\beta$ and LR | $G$ and difficulty mix |

A pattern worth internalizing: PPO and GRPO use a KL term as a *safety leash* against an optimizer that's actively trying to exploit a reward, while DPO uses $\beta$ as the *strength* of the alignment itself. Same Greek letter, opposite spirit — in DPO you lower $\beta$ to align harder, in GRPO you lower $\beta$ to let the verifier pull harder. Confusing the two is a common source of "I copied the hyperparameters and it didn't work."

## What to Monitor During Training

Each method has a small set of curves that tell you, at a glance, whether the run is healthy or quietly failing. Watching the loss alone is not enough — for two of the three, the loss can look perfect while the model degrades.

**PPO dashboard.** Watch the **reward**, the **KL divergence** from the reference, and — critically — a **held-out real metric** that the policy cannot optimize against. The failure signature is reward climbing while the real metric falls: that's reward hacking, and the reward curve is lying to you. Also watch the KL: if it spikes, the policy is lunging away from the reference and you're heading for divergence; if it's pinned at zero, your KL coefficient is too high and nothing is learning. A healthy PPO run shows reward and the real metric rising together while KL holds near its target.

**DPO dashboard.** Watch the **margin** (chosen minus rejected implicit reward), the **pairwise win-rate**, and — the one people skip — the **absolute chosen log-probability**. The trap is that the margin and win-rate can both improve while the absolute chosen log-prob falls, which is likelihood displacement in the act. If the chosen log-prob trends down, the model is learning the preference by becoming less confident overall, and you'll see the damage off-distribution in production. A healthy DPO run shows the margin widening *and* the chosen log-prob flat or rising.

**GRPO dashboard.** Watch the **mean reward**, the **fraction of groups with zero advantage** (all-correct or all-wrong), and the **policy entropy**. The zero-advantage fraction is the most actionable GRPO metric there is: if a large share of your groups have no spread, that share of your batch is teaching nothing, and you should either curate harder prompts or turn on dynamic sampling. Falling entropy warns of collapse — the policy is becoming overconfident and the groups are losing diversity. A healthy GRPO run shows mean reward rising, the zero-advantage fraction staying modest, and entropy declining gently rather than crashing.

| Metric to watch | PPO | DPO | GRPO |
|---|---|---|---|
| Primary health signal | reward + held-out real metric | margin + absolute chosen log-prob | mean reward + entropy |
| The lying metric | reward (reward hacking) | margin/win-rate (displacement) | mean reward (if entropy crashing) |
| Early-warning metric | KL spike or pin | chosen log-prob falling | zero-advantage group fraction |
| "Run is healthy" looks like | reward and real metric rise together | margin widens, chosen prob steady | reward up, entropy gently down |

The unifying lesson across all three dashboards: **never trust the metric the optimizer is allowed to game.** PPO games the reward, DPO games the margin, and GRPO — if entropy is collapsing — games the mean reward by narrowing its distribution. For each method there's a second metric (the real held-out eval, the absolute log-prob, the entropy and zero-advantage fraction) that the optimizer can't directly inflate, and that second metric is the one that tells you the truth.

## A Worked Example: One Team, Three Decisions

To see how the framework runs in practice, walk through a single hypothetical team building a coding assistant, making the choice three times for three sub-problems.

**Sub-problem 1: the model should pass more unit tests.** This is verifiable — run the generated code against the test suite, reward = fraction of tests passed. There's a clean, dense, unhackable signal and the team has GPUs for rollouts. The framework says **GRPO**, and the team writes a verifier that executes code in a sandbox and returns the pass rate. The main operational concern becomes prompt difficulty: too-easy tasks where every sample passes contribute no gradient, so they curate a difficulty distribution.

**Sub-problem 2: the model should write clearer explanations of its code.** "Clearer" is not verifiable — there's no function that returns clarity. But the team can collect preferences: show annotators two explanations and ask which is clearer. They now have (prompt, chosen, rejected) triples and want a focused alignment pass. The framework says **DPO**. They run it with an SFT mix to avoid displacement, on a single node, in an afternoon, and watch absolute chosen log-prob alongside the win-rate.

**Sub-problem 3: the model should follow a complex, evolving style guide with many soft tradeoffs.** Here the team wants fine-grained control — multiple reward components (style, correctness, safety), the ability to reshape the reward as the guide evolves, and a long training horizon as the model explores. They have RL infrastructure and are willing to build and harden a reward model. The framework says **PPO**, and they accept its cost because the reward model's generalization to novel code styles and the reward-shaping flexibility are exactly what the problem needs.

Same team, same week, three different answers — because the *signal* differed three times. That is the entire thesis of this guide in one story: you don't have a favorite algorithm, you have a reward signal, and the reward signal chooses the algorithm.

## Hybrid Pipelines: Why It's Rarely Just One Method

The case studies below will hammer this home, but it's worth stating plainly: at the frontier, post-training is a *pipeline*, not a single algorithm. The canonical 2025–2026 shape is roughly:

1. **SFT** — teach the base model the format and basic behavior from demonstrations.
2. **Preference alignment (DPO)** — broad alignment to human preferences on the not-easily-verifiable stuff: tone, helpfulness, safety, formatting.
3. **Verifiable-reward RL (GRPO / RLVR)** — sharpen the capabilities you *can* verify: math, code, precise instruction following.
4. **(Optional) a final RL polish (PPO or GRPO)** with a learned reward model for the last mile of preference alignment.

Each stage uses the method whose signal matches that stage's goal. DPO carries the preference-aligned breadth; GRPO carries the verifiable depth; PPO appears when a robust reward model is worth its cost. The teams that do this well don't argue about "DPO vs GRPO" as if they had to pick one forever — they sequence them, each one doing the job it's best at. When you read "X model used DPO" or "Y model used GRPO," it almost always means "at this stage, for this capability," not "instead of the others."

## Case Studies from Production

Theory is cheap. Here's how the choice actually played out across ten well-documented systems.

### 1. InstructGPT — PPO because there was nothing else

The original RLHF pipeline (OpenAI, 2022) is the canonical PPO case. They had human preference data, no verifier (the tasks were open-ended instruction following), and they needed a reward that generalized to novel responses. They trained a reward model on the preferences and ran PPO against it, complete with a KL penalty to a frozen SFT reference. It worked, and it defined the template. But it also surfaced every PPO pathology: reward hacking, the need for careful KL tuning, the operational weight of a four-model RL loop. InstructGPT is why we know PPO works for open-ended alignment — and why everyone has been trying to avoid its cost ever since. Read through the modern lens, InstructGPT made the only choice available: no verifier ruled out GRPO (which didn't exist yet), and the offline directness of DPO (which also didn't exist yet) would have frozen the model to its SFT distribution. The reward-model route was the right call for its regime.

### 2. Zephyr-7B — DPO proves you can skip the RL loop

Zephyr (HuggingFace H4, late 2023) was the proof point that DPO could match RLHF-style alignment without the RL machinery. They took Mistral-7B, did distilled SFT on UltraChat, then ran DPO on UltraFeedback preference pairs — entirely offline, on modest hardware. The result was competitive with much heavier RLHF models on chat benchmarks. The lesson the whole open-source world took from Zephyr: if you have good preference data, DPO is the pragmatic default, and the RL loop is optional complexity. Nearly every open chat model for the next year followed this SFT-then-DPO template. In framework terms, Zephyr's signal was preferences and its goal was a focused alignment pass on a distribution its SFT model already covered — squarely DPO's home turf, and the off-policy staleness never had time to bite in a short single-pass run.

### 3. DeepSeek-R1-Zero — GRPO and the verifiable-reward revolution

DeepSeek-R1-Zero (early 2025) is the case study that made GRPO famous. They ran GRPO on a base model with *purely* verifiable rewards — math answers checked against gold, code checked against tests, plus a format reward — and no SFT warm start at all. The reasoning behavior (long chains of thought, self-verification, the "aha moment") *emerged* from RL on the verifiable signal. No reward model, no critic, no human preference labels in the loop. R1-Zero is the existence proof that for verifiable domains, GRPO's group baseline plus a clean reward is enough to bootstrap reasoning from scratch. It reset the field's defaults: after R1, "we have a verifier, so we'll use GRPO" became the obvious first move for any reasoning task, and the entire variants literature in [the follow-up post](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo) is downstream of this single run.

### 4. Llama-3.1 — iterative DPO with rejection sampling, not PPO

Meta's Llama-3 post-training (2024) is instructive because they *chose DPO over PPO* at scale and said why. Their pipeline was rounds of rejection-sampling SFT followed by DPO, iterated — regenerate responses with the current model, score them, build fresh preference pairs, DPO, repeat. This is on-policy-flavored DPO: by regenerating data each round, they fought the staleness that plagues one-shot DPO, while keeping DPO's training stability and avoiding PPO's reward-hacking headaches. The takeaway: at the frontier, *iterative* DPO was judged a better cost/quality/stability trade than PPO for general alignment. It's the clearest large-scale vote for "DPO, made partly on-policy through iteration" over "full PPO" — a deliberate engineering choice, not a capability gap.

### 5. Tulu-3 — DPO for alignment, RLVR for verifiable skills

AI2's Tulu-3 (late 2024) made the *pipeline* point explicit: it isn't one method, it's a sequence. Tulu-3 did SFT, then DPO on a large curated preference mix for broad alignment, then a final RL stage with verifiable rewards (RLVR — the same family as GRPO) specifically to sharpen math and precise-instruction-following. They used DPO where they had preferences and verifiable RL where they had verifiers, in the same model. This is the template the field converged on: don't pick one algorithm for the whole model, pick the right one per capability. Tulu-3 is also valuable because it's documented in the open — you can read exactly which capabilities each stage was responsible for, and the division maps perfectly onto the reward-regime axis.

### 6. Qwen2.5-Math — GRPO/PPO for the last mile of math

The Qwen math models leaned on RL with verifiable rewards to push competition-math accuracy. The verifier is the gold answer; the reward is whether the model's boxed answer matches. GRPO-style training on hard math prompts, with careful attention to the zero-gradient trap (filtering prompts that are all-correct or all-wrong for the current model), drove large gains on AIME-style benchmarks. The case study underlines a GRPO operational truth: *prompt difficulty curation is a first-class concern*, because a group with no spread teaches nothing. Qwen's experience here directly motivated the dynamic-sampling fix that DAPO would later formalize — when a meaningful fraction of your batch contributes zero gradient, "just sample more prompts until the group has spread" stops being a hack and becomes part of the algorithm.

### 7. OLMo-3 — Delta Learning DPO plus an RL stage, fully open

AI2's [OLMo-3](/blog/paper-reading/large-language-model/olmo-3-training-finetuning-techniques) (2026) is a fully-open window into a modern post-training stack. Their preference stage used a Delta-Learning flavor of DPO (constructing preference pairs from a strong-vs-weak model delta), followed by their own RL stage (OlmoRL) with verifiable rewards. Because everything is open, you can see the deliberate division of labor: DPO carries broad preference alignment; the RL stage carries verifiable reasoning. It's the Tulu lineage, documented end to end, and it reinforces that the DPO-then-RLVR sequence is now standard practice, not an experiment. OLMo-3's Delta-Learning twist is also a reminder that "where do the preference pairs come from" is itself a design space — you don't always need human labels; a capability gap between two models can manufacture them.

### 8. A likelihood-displacement near-miss — when DPO quietly degraded

I'll close with a composite of a failure I've seen more than once. A team runs DPO on a clean preference set, the loss drops, pairwise win-rate against the SFT baseline climbs, everyone's happy — and then production users report the model has gotten *more* evasive and occasionally degenerate on prompts outside the preference data. The cause was likelihood displacement: the DPO gradient had been lowering the rejected probability mostly by lowering *all* probabilities, and the absolute log-prob of the chosen responses had been quietly falling the whole run. The fix was exactly the textbook one — add the SFT auxiliary loss on the chosen responses (the `rpo_alpha` term), and monitor absolute chosen log-prob, not just the margin. The deeper lesson: DPO's offline metrics can look great while the model degrades off-distribution, because the loss only measures a *contrast*, not absolute quality. If your only dashboard is the margin and the win-rate, you are flying blind to the exact way DPO most often fails.

### 9. Sparrow — PPO with rule rewards before verifiers were a thing

DeepMind's Sparrow (2022) is a useful counterpoint because it sits at the boundary between learned-reward PPO and verifiable-reward RL. Sparrow used PPO against a *combination* of a learned preference reward model and a set of *rule-based* reward signals — programmatic checks for specific harmful or rule-violating behaviors. In hindsight, the rule-based components were a primitive verifier, and the preference reward model was the classic PPO ingredient; Sparrow blended both into one scalar and optimized it with PPO. The lesson for the framework: reward signals aren't always purely one regime. When part of your objective is verifiable and part is preference-based, you can either combine them into a single reward for PPO/GRPO, or stage them — and Sparrow's choice to fold rules into the PPO reward presaged today's practice of mixing verifiable and learned rewards in a single RL stage. It's a reminder that the three regimes are corners of a space, and real systems sometimes live on an edge between them.

### 10. A GRPO-on-MoE instability — the case that motivated GSPO

The last case is the one that bridges directly to [the variants post](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo). A team training a large Mixture-of-Experts model with vanilla GRPO watched their runs go unstable in a way dense-model GRPO never did — loss spikes, sudden divergence, runs that were reproducible only in their irreproducibility. The root cause turned out to be specific to MoE: between gradient updates, the set of experts the router activates for the same input can shift by roughly ten percent, which makes GRPO's token-level importance ratios fluctuate wildly and pollutes the gradient. The fix wasn't a hyperparameter — it was a different ratio granularity, computing importance at the *sequence* level instead of the token level, which is exactly what Qwen's GSPO does. The case study makes a framework point: GRPO is not a single fixed algorithm but a family with known failure modes on specific architectures, and choosing "GRPO" in 2026 increasingly means choosing *which* GRPO variant. That choice is its own decision guide, which is the subject of the companion article.

## When to Reach for Each — and When Not To

**Reach for GRPO when:**

- You have a verifiable reward — math, code, tests, schema validation, anything a function can score.
- You're training reasoning and want emergent chain-of-thought to develop under RL.
- You're memory-constrained and want on-policy RL without paying for a critic.
- You can curate prompt difficulty so groups land in the productive middle.

**Skip GRPO when:** you have no verifier and can't build one; your task is open-ended generation with no notion of "correct"; your prompts are uniformly trivial or uniformly impossible for the current model (every group has zero spread and you'll waste the run); or your verifier is a weak heuristic the policy will learn to satisfy instead of actually solving the task.

**Reach for DPO when:**

- You have preference pairs and want alignment without an RL system.
- You're memory- or infra-constrained — DPO runs on a near-SFT stack.
- You value a stable, reproducible, supervised-style training curve.
- You're doing a focused alignment pass on data that matches your target distribution.

**Skip DPO when:** you need long-horizon open-ended exploration (the off-policy staleness will bite); you can't tolerate likelihood displacement and won't add the SFT mix; or your preference data is far from the responses your current model produces (regenerate it, or you're optimizing a contrast on ghosts).

**Reach for PPO when:**

- You have a robust, validated reward model and the infrastructure for online RL.
- You need fine-grained reward control — shaping, multiple reward sources, safety gates.
- The reward model's ability to score *novel* responses genuinely matters for a long run.
- You've consciously decided the critic and reward model are worth ~2× the memory.

**Skip PPO when:** you have a verifier (use GRPO — cheaper, unhackable); you only need a focused alignment pass (use DPO); you don't have the infrastructure to keep generation and training interleaved at scale; or you can't invest in making the reward model robust (an un-robust RM will be hacked, and you'll ship a worse model that scores higher).

The honest summary: in 2026, **GRPO is the default for verifiable reasoning, DPO is the default for preference alignment, and PPO is the specialist's tool** for when you've earned the right to a reward model and a critic. Most projects never need PPO. Many projects need both DPO and GRPO, at different stages, for different capabilities. The skill isn't memorizing the loss functions — it's reading your reward signal and your budget honestly enough to let the choice make itself. Get the signal question right and the algorithm question answers itself; get it wrong and no amount of hyperparameter tuning will save the run.

## Frequently Asked Edge Cases

A few questions come up so often that they deserve direct answers.

**"I have a verifier *and* preference data — which wins?"** Use both, in sequence. Run GRPO on the verifiable part to sharpen the capability that can be checked, and DPO (or a preference-RM PPO stage) on the part that can't. They're not competitors here; they're two stages targeting two different slices of behavior. If you must pick one because of budget, pick the one that targets your bottleneck capability — if your model is bad at math, GRPO on the verifier moves the needle far more than DPO on chat preferences.

**"Can I just use GRPO for everything by writing a reward function for chat quality?"** You can write the function, but a hand-rolled "chat quality" reward is a weak heuristic verifier, and GRPO will optimize the heuristic, not the quality. The moment your reward function is itself an approximation of human judgment, you've reinvented the reward model — and now you have all of PPO's reward-hacking exposure with none of PPO's variance reduction. For genuinely subjective quality, preferences (DPO) or a real reward model (PPO) are honest; a toy verifier is a trap.

**"My DPO run looks great offline but the deployed model is worse — what happened?"** Almost always one of two things: likelihood displacement (check the absolute chosen log-prob, not just the margin) or distribution shift (your preference data no longer resembles what the model generates). The fix for the first is the SFT mix; the fix for the second is to regenerate preferences with the current model and iterate. Both are covered above, and both are invisible if your only dashboard is the loss and the win-rate.

**"Is PPO ever the right first choice anymore?"** Rarely as a *first* choice, but yes as a *considered* one: when you have a robust reward model, the infrastructure to run online RL, a long exploration horizon, and a need for reward shaping that DPO can't express and a verifier can't capture. That's a real combination of conditions — frontier labs hit it — but if you're asking the question without already having the reward model and the infrastructure, the answer is almost certainly "start with DPO or GRPO."

**"Should I tune $\beta$ first or the learning rate first?"** For DPO, fix a small learning rate ($5\times10^{-7}$) and sweep $\beta$ — it's the parameter that changes the *behavior*, not just the speed. For GRPO, fix $\beta$ low and get the prompt-difficulty distribution right first, because no $\beta$ saves a batch where every group has zero spread. For PPO, get the KL target stable before touching anything else, because an unstable KL makes every other measurement meaningless.

## Further Reading

- [Fine-Tuning LLMs with GRPO: From Theory to Implementation](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) — the full GRPO derivation and training recipe.
- [Fine-Tuning LLMs with DPO: A Practical Guide](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) — DPO math, frameworks, and production lessons.
- [Beyond GRPO: DAPO, Dr. GRPO, GSPO, and the Loss-Aggregation Fixes of 2025](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo) — what to do once vanilla GRPO isn't enough.
- [Training LLMs for Math](/blog/machine-learning/large-language-model/training-llm-for-math) — the verifiable-reward pipeline end to end.
- [TRL: Transformer Reinforcement Learning Library](/blog/machine-learning/open-source-library/trl-lib) — the library that implements all three.
- [DeepSeek-R1: Incentivizing Reasoning via Reinforcement Learning](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) — the paper that made GRPO famous.

---
title: "GRPO: Group Relative Policy Optimization for Reasoning Models"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A from-first-principles guide to GRPO — why DeepSeek deleted the critic, how the group-relative advantage replaces a value network, the full objective with KL, a from-scratch PyTorch implementation, the DeepSeek-R1 recipe, and the DAPO and Dr. GRPO fixes."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "grpo",
    "rlhf",
    "llm-alignment",
    "policy-gradient",
    "machine-learning",
    "pytorch",
    "reasoning",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/grpo-group-relative-policy-optimization-1.png"
---

The first time you try to train a reasoning model with PPO, the thing that breaks your spirit is not the policy. It is the critic. You have a 7-billion-parameter language model that you want to teach to solve competition math, and the textbook tells you that to estimate advantages you need a value function — a second network, usually about the same size, that reads a half-finished chain of thought and predicts how much reward the model will eventually collect. So you spin up the second network. It doubles your memory. It needs its own optimizer state, its own forward and backward pass, its own learning rate that you have to tune separately. And then, when you finally get it running, you discover the cruel part: for a reasoning trace that is two thousand tokens long and gets a reward of exactly 1.0 if the final boxed answer is right and 0.0 if it is wrong, the value function has almost nothing useful to say at token 300. The reward is a single bit that arrives at the very end. The critic is trying to predict a coin flip from the middle of the chain, and its estimates are noise. You are spending half your compute training a network whose output is barely better than a constant.

This is the exact situation the DeepSeek team faced while building DeepSeek-Math and, later, DeepSeek-R1. Their response was not to build a better critic. It was to ask whether they needed one at all. Group Relative Policy Optimization — GRPO, introduced by Shao et al. in the DeepSeekMath paper in early 2024 — is the answer: you delete the value network entirely and replace its job, providing a baseline for the advantage, with a trick that is almost embarrassingly simple. For each question, sample a whole group of answers. Score them all. The average score of the group is your baseline. An answer that beat the group average gets a positive advantage; one that lost to the group average gets a negative advantage. No critic, no value head, no second network. The figure below shows what that buys you — the same RLHF loop with two fewer models in memory and the noisiest part of the loss deleted.

![Side by side comparison of PPO-RLHF needing a learned critic and four models versus GRPO using a group baseline with only two models in memory](/imgs/blogs/grpo-group-relative-policy-optimization-1.png)

This post is a complete working understanding of GRPO, built the way the series builds everything: the theory of *why* a group-relative baseline is a valid, low-variance advantage estimator, the algorithm in real PyTorch you can copy and run, and concrete measured results from the DeepSeek-Math and DeepSeek-R1 papers. We will derive the group-relative advantage from the policy gradient theorem and see exactly which term the value function was playing, work through the full GRPO objective term by term including the per-token KL penalty, write a complete GRPO update from scratch, walk the DeepSeek-R1 training recipe stage by stage including the famous "aha moment," and finish with the second-generation fixes — DAPO and Dr. GRPO — that patch the places where the vanilla formulation quietly biases your gradients. The whole series rests on one frame: the RL loop is an agent interacting with an environment, collecting rewards, and updating a policy, and every algorithm is a different answer to *which objective to optimize* and *how to estimate the gradient*. GRPO's answer is unusually clean, and by the end you should be able to implement it, debug it, and explain to a skeptical colleague why a group mean is a perfectly good substitute for a half-billion-dollar critic network.

## The setup: an LLM is a policy, a reasoning trace is a trajectory

Before we can delete the critic we need to be precise about what the objects are, because the mapping from "language model" to "reinforcement learning policy" is where most of the confusion lives.

A language model $\pi_\theta$ is a stochastic policy. The state is the prompt-so-far: the question $q$ concatenated with whatever tokens the model has already generated. The action at each step is the next token. The model emits a probability distribution over the vocabulary, you sample a token, append it, and repeat. A complete generated answer $o = (o_1, o_2, \dots, o_T)$ — a full chain of thought ending in a boxed answer — is a *trajectory*. Its probability under the policy factorizes the way autoregressive models always do:

$$\pi_\theta(o \mid q) = \prod_{t=1}^{T} \pi_\theta(o_t \mid q, o_{<t})$$

The environment is almost trivial. There are no dynamics to learn: appending a token deterministically produces the next state. The only interesting thing the environment does is hand out a reward, and in the reasoning setting it hands out *exactly one* reward, at the very end, after the model has emitted its final answer. For a math problem the reward is whether the boxed answer matches the ground truth: reward 1.0 if correct, 0.0 if wrong. This is what people mean when they call it an *outcome reward* — the signal is the outcome, not the process. There is no per-token reward, no shaping, no dense feedback. One question, one full answer, one bit of reward.

That single structural fact — sparse terminal reward over a long trajectory — is the entire reason reasoning RL is hard, and it is the reason the critic is such a bad deal here. In a control task like CartPole, reward arrives every step, and a value function genuinely helps because it can learn the rich gradient of "how good is this state." In reasoning, the reward is a coin flip revealed only at the end. A value function asked to predict that coin flip from the middle of a chain is being asked to do something close to impossible, and it will return noise that you then propagate into your advantage estimates. We can do better by refusing to estimate it token-by-token at all.

#### Worked example: the trajectory and its reward

Take the question "What is $7 \times 8 + 6$?". The model generates: "Let me compute step by step. First $7 \times 8 = 56$. Then $56 + 6 = 62$. The answer is $\boxed{62}$." That is a trajectory $o$ of, say, 31 tokens. The ground-truth answer is 62, the boxed answer is 62, so the reward $r(o) = 1.0$. A second sampled answer might say "$7 \times 8 = 54$, so $54 + 6 = 60$, answer $\boxed{60}$" — wrong, reward $0.0$. The policy assigns each token in each trajectory a probability; the product of those is $\pi_\theta(o \mid q)$. Everything GRPO does is decide how much to push those per-token probabilities up or down based on the single terminal reward, and crucially, *relative to the other answers sampled for the same question*. Hold onto that last phrase — it is the whole idea.

## Why the policy gradient needs a baseline at all

To see what GRPO replaces, we have to recall what the baseline is *for*. Start from the policy gradient theorem, which the [policy gradient theorem post](/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem) derives in full. We want to maximize expected reward $J(\theta) = \mathbb{E}_{o \sim \pi_\theta}[r(o)]$, and the gradient is

$$\nabla_\theta J(\theta) = \mathbb{E}_{o \sim \pi_\theta}\big[ r(o)\, \nabla_\theta \log \pi_\theta(o \mid q) \big]$$

This is the score-function estimator: weight the gradient of each trajectory's log-probability by the reward it earned. Push up the log-probability of high-reward answers, push down low-reward ones. It is unbiased, and it is also catastrophically high-variance, and the reason is worth stating precisely because it is the reason baselines exist.

The variance comes from the fact that $r(o)$ is a raw, un-centered number. Suppose every answer to a question gets a reward somewhere between 0 and 1, and the typical answer gets 0.6. The estimator multiplies *every* trajectory's gradient by a positive number, so it pushes *up* the probability of nearly every answer the model produces, including mediocre ones — it just pushes the good ones up a bit harder. The signal you actually care about, "this answer was better than typical," is buried inside a large common-mode term "this answer got positive reward." That common-mode term contributes nothing to the expected gradient (we will prove this in a moment) but it contributes enormously to the variance of any finite-sample estimate.

The fix is a baseline. We subtract a quantity $b$ that does not depend on the action from the reward:

$$\nabla_\theta J(\theta) = \mathbb{E}_{o \sim \pi_\theta}\big[ (r(o) - b)\, \nabla_\theta \log \pi_\theta(o \mid q) \big]$$

The remarkable fact, and the one that makes everything else legal, is that subtracting *any* baseline that does not depend on the sampled action leaves the gradient unbiased. The proof is short. The extra term we introduced is

$$\mathbb{E}_{o \sim \pi_\theta}\big[ b \, \nabla_\theta \log \pi_\theta(o \mid q) \big] = b \sum_o \pi_\theta(o \mid q)\, \nabla_\theta \log \pi_\theta(o \mid q) = b \sum_o \nabla_\theta \pi_\theta(o \mid q) = b\, \nabla_\theta \sum_o \pi_\theta(o \mid q) = b\, \nabla_\theta 1 = 0$$

The sum of probabilities over all trajectories is 1, a constant, whose gradient is zero. So the baseline shifts nothing in expectation. But it changes the variance dramatically, and the variance-minimizing baseline is, roughly, the expected reward itself — the average reward you would get from this state. That is precisely the quantity a value function $V(s)$ is trained to estimate. So now we can name the critic's real job: **the value function exists to be a baseline.** It is the thing we subtract to center the reward. The advantage $A = r - V$ is just "reward minus baseline." Once you see that the critic is, fundamentally, a learned estimate of the per-question average reward, GRPO's move becomes obvious — why learn that average with a half-billion-parameter network when you can just *measure* it by sampling a few answers and taking their mean?

It is worth dwelling on exactly *how much* variance the common-mode term contributes, because the size of the effect is the whole reason anyone bothers with a baseline. Variance of a sum of a centered signal and a common-mode offset is dominated by whichever term is larger, and the common-mode reward is large precisely when the policy is doing reasonably well — when most answers earn similar, mostly-positive rewards. So the regime where the unbaselined estimator is *worst* is the regime you spend most of training in: a policy that is already decent and is trying to get a little better. In that regime nearly every trajectory has reward around 0.6, the gradient gets multiplied by 0.6 across the board, and the tiny differences that carry the actual learning signal — the 0.1 by which one answer beat another — are swamped by the shared 0.6. Subtract the mean and the 0.6 vanishes; what remains is the $\pm 0.1$ that you actually wanted to learn from. This is not a marginal tweak. On hard benchmarks where pass rates are 40–60%, the unbaselined gradient can have an order of magnitude more variance than the baselined one, which translates directly into needing an order of magnitude more samples to get the same gradient signal-to-noise ratio. For LLM RL, where each sample is a thousand-token generation, that factor is the difference between a feasible run and an infeasible one.

There is one more property of the optimal baseline worth stating, because it explains why GRPO normalizes by the standard deviation and not just by subtracting the mean. The variance-minimizing baseline is not exactly the mean reward; it is a *gradient-magnitude-weighted* mean. But the plain mean is extremely close to optimal in practice, and it has the enormous practical advantage of being trivially estimable from samples — you do not need the per-token gradient norms to compute it. GRPO takes the plain mean as its baseline and then adds the standard-deviation division as a *scale* normalization, which is a separate concern: it does not reduce variance of the estimator so much as it makes the advantage scale comparable across questions of different difficulty, so a single learning rate works across an entire heterogeneous batch. We will see the cost of that division later when we discuss Dr. GRPO, but the intuition for why it is there is just: questions vary wildly in how spread-out their group rewards are, and dividing by the spread puts them on a common footing.

## The GRPO insight: the group is the baseline

Here is the whole idea in one sentence. For each question $q$, sample a group of $G$ answers $\{o_1, \dots, o_G\}$ from the current policy, score them all, and use the group's mean reward as the baseline. The advantage of answer $o_i$ is its reward minus the group mean, divided by the group standard deviation:

$$A_i = \frac{r_i - \operatorname{mean}(r_1, \dots, r_G)}{\operatorname{std}(r_1, \dots, r_G)}$$

That is it. The mean is the variance-minimizing baseline we just argued for, computed by Monte Carlo instead of by a learned network. The division by the standard deviation is a normalization step that puts advantages on a consistent scale across questions of wildly different difficulty — a question where every answer is correct (mean 1.0) and a question where every answer is wrong (mean 0.0) both produce a degenerate group, but a question where 3 of 8 answers are right produces a clean spread of positive and negative advantages. The figure below shows the data flow: one question fans out into a group, the group's reward statistics fan back in as a shared baseline, and the normalized advantages flow into the loss.

![Diagram of GRPO group sampling where one question fans out to several responses whose rewards merge into a group mean and standard deviation that feed normalized advantages into the clipped policy loss](/imgs/blogs/grpo-group-relative-policy-optimization-2.png)

There is a subtlety worth making explicit. In standard advantage-actor-critic, the advantage is per-state: $A(s_t, a_t)$ depends on which token you are at. In vanilla GRPO the advantage is *per-trajectory* — the same scalar $A_i$ is assigned to every token in answer $o_i$. This makes sense given the reward structure: the reward is a single terminal bit, so there is no principled way to say "token 300 of this answer was more responsible for the correct answer than token 301." The whole answer succeeded or failed together, so the whole answer gets the same advantage. This is sometimes called the outcome-supervision variant, and it is what DeepSeek-R1 used. There is also a process-supervision variant where a reward model scores intermediate steps and the advantage at each token is the sum of normalized step rewards from that token onward, but the outcome variant is simpler, avoids training a process reward model, and is what made R1 work.

It is worth pausing on *why* assigning the same advantage to every token is not as crude as it sounds, because the first instinct of anyone who has done credit assignment is to recoil at it. In a long reasoning chain, surely some tokens mattered more than others? Yes — but the policy gradient does not need you to know *which* tokens mattered to learn the right thing. Here is the mechanism. When an answer is correct and gets a positive advantage, GRPO pushes up the probability of *every* token in it, including the filler and the dead ends. When an answer is wrong and gets a negative advantage, GRPO pushes down the probability of every token in it. The tokens that are *causally responsible* for correctness — the right intermediate computation, the correct setup of the equation — appear disproportionately in correct answers and rarely in wrong ones, so over many groups and many questions they accumulate net-positive gradient. The tokens that lead to errors appear disproportionately in wrong answers and accumulate net-negative gradient. The filler tokens that appear equally in correct and wrong answers get pushed up and down about equally and net out to roughly zero. The averaging over the group and over the dataset *does the credit assignment for you*, statistically, without ever needing a per-token value estimate. This is the same reason REINFORCE works at all despite multiplying the whole trajectory's gradient by the whole trajectory's return — credit assignment emerges from the statistics, not from a per-step estimator. GRPO simply makes that emergence cleaner by using a good baseline.

This is also the deepest reason GRPO suits reasoning specifically. A learned per-token critic, in this setting, is *pretending* to a precision it does not have — it claims to know that token 300 has value 0.6 and token 301 has value 0.62, when in truth the reward is a coin flip the critic cannot predict. The per-trajectory advantage makes no such false claim. It says, honestly, "this entire answer was above or below the group average," and lets the dataset statistics sort out which tokens deserve the credit. Refusing to estimate something you cannot estimate well is, more often than people admit, the right move in machine learning, and GRPO's flat per-token advantage is a clean instance of it.

If you do want a peek at the process-supervision variant — where a step-level reward model gives you per-step rewards — the advantage computation changes only slightly: instead of one scalar per response, you normalize the step rewards across the group and assign each token the cumulative normalized reward from its step onward. In code that is a small change to the advantage function:

```python
def process_advantages(step_rewards, step_index):
    # step_rewards: (G, num_steps) -- a reward-model score per reasoning step per response.
    # step_index:   (G, L) -- which step each token belongs to.
    flat = step_rewards.reshape(-1)
    mu, sigma = flat.mean(), flat.std(unbiased=False)
    norm = (step_rewards - mu) / (sigma + 1e-6)        # normalize all steps in the group
    cum = norm.flip(-1).cumsum(-1).flip(-1)            # cumulative reward from each step onward
    # gather the cumulative value for each token by its step index
    return torch.gather(cum, 1, step_index)            # (G, L) per-token advantage
```

But DeepSeek's own ablations found the extra machinery of a process reward model was not worth it for math and code: the outcome variant matched or beat it while avoiding the training and the reward-hacking surface of a step-level model. So the rest of this post stays with the outcome variant.

#### Worked example: computing group-relative advantages

Take a math question and sample $G = 8$ answers. Score them with a verifier: the rewards come back as $r = [1, 0, 1, 0, 1, 1, 0, 1]$ — five correct, three wrong. Compute the group statistics:

$$\mu = \frac{1 + 0 + 1 + 0 + 1 + 1 + 0 + 1}{8} = \frac{5}{8} = 0.625$$

$$\sigma = \sqrt{\frac{1}{8}\sum_i (r_i - \mu)^2} = \sqrt{\frac{5(1 - 0.625)^2 + 3(0 - 0.625)^2}{8}} = \sqrt{\frac{5(0.1406) + 3(0.3906)}{8}} = \sqrt{0.2344} \approx 0.484$$

Now the advantages. A correct answer gets

$$A_{\text{correct}} = \frac{1 - 0.625}{0.484} \approx +0.775$$

and a wrong answer gets

$$A_{\text{wrong}} = \frac{0 - 0.625}{0.484} \approx -1.291$$

Notice what the normalization did. There are more correct answers than wrong ones, so each wrong answer is "more surprising" and gets a larger-magnitude (negative) advantage than each correct answer gets (positive). The gradient will push down the rare failures harder than it pushes up the common successes — exactly the behavior you want when most answers are already right and you are hunting the remaining mistakes. If instead only 1 of 8 were correct, that single success would get a large positive advantage and the seven failures would get small negative ones, focusing the gradient on reinforcing the one path that worked. The group baseline adapts to the difficulty of each question automatically, with no learning and no tuning. The stack figure below lays out these five steps in order.

![Layered diagram of the group relative advantage computation showing sampling G responses, computing each reward, the group mean, the group standard deviation, and the normalized advantage](/imgs/blogs/grpo-group-relative-policy-optimization-3.png)

## The full GRPO objective, term by term

The advantage is the heart of GRPO, but the full objective borrows the rest of its machinery from PPO, which the [PPO post](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) covers in depth. We keep PPO's clipped importance ratio (so we can reuse each batch of samples for several gradient steps without the policy running away) and we add an explicit KL penalty against a frozen reference policy (so the model stays in-distribution and does not reward-hack). Here is the objective in full:

$$J_{\text{GRPO}}(\theta) = \mathbb{E}_{q, \{o_i\}}\left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\Big( \rho_{i,t} A_i,\; \operatorname{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon) A_i \Big) - \beta\, D_{\text{KL}}\big[\pi_\theta \,\|\, \pi_{\text{ref}}\big] \right]$$

Let us name every piece.

The **importance ratio** $\rho_{i,t}$ is the per-token ratio of the current policy to the policy that generated the samples:

$$\rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$$

When you have just collected the samples and haven't updated yet, $\theta = \theta_{\text{old}}$ and every ratio is exactly 1. As you take gradient steps on the same batch, the ratios drift. The ratio is what lets GRPO, like PPO, squeeze multiple gradient updates out of one expensive round of generation — the most expensive thing in the whole loop.

The **clip** is the trust region. If the advantage $A_i$ is positive (a good answer) the unclipped term $\rho_{i,t} A_i$ wants to push the ratio up without bound; the clip caps it at $1 + \varepsilon$, so once you have increased that token's probability by a factor of $1+\varepsilon$ the objective stops rewarding you for pushing further. If $A_i$ is negative the clip caps the downward push at $1 - \varepsilon$. The $\min$ of the clipped and unclipped terms ensures the clip only ever *removes* incentive, never adds it — it is a one-sided brake. The typical value is $\varepsilon = 0.2$.

The **two normalizations** are easy to skip past but they matter. The outer $\frac{1}{G}$ averages over the group — every question contributes equally regardless of how its answers turned out. The inner $\frac{1}{|o_i|}$ averages over the *tokens* of each answer. That inner one is the per-response length normalization, and it exists to stop long answers from dominating the gradient simply because they have more tokens. We will see later that this exact term is also where one of GRPO's subtle biases hides, and where Dr. GRPO intervenes.

The **KL penalty** $\beta\, D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]$ keeps the policy anchored to a frozen reference (usually the SFT model you started from). Without it, the policy is free to drift toward degenerate text that happens to fool the reward function — answers that are technically correct but unreadable, or that exploit quirks in the verifier. The KL term is the leash. DeepSeek estimates the KL with the unbiased low-variance estimator $D_{\text{KL}} \approx \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$, which is always positive and has lower variance than the naive log-ratio. A typical coefficient is $\beta = 0.04$, though as we'll see, some later recipes set it to zero entirely.

Stack these against PPO and the contrast is stark. PPO's advantage comes from GAE over a learned value function; GRPO's comes from the group mean. PPO applies the KL implicitly through the clip and an optional adaptive penalty; GRPO adds an explicit per-token KL to the reference. PPO holds four models in memory (policy, value, reward, reference); GRPO holds two (policy, reference) plus whatever computes the reward, which for verifiable tasks is a rule, not a network. The graph figure earlier in this post is exactly this objective drawn as a computation: groups fan out, statistics fan in, advantages and ratios combine into the clipped loss minus KL.

A side-by-side of the two objectives makes the inheritance precise — GRPO keeps PPO's outer machinery and swaps only the advantage source and the KL treatment:

| Component | PPO | GRPO |
| --- | --- | --- |
| Advantage source | GAE over learned $V(s)$ | group-relative $(r_i - \mu)/\sigma$ |
| Baseline | learned value network | group mean (Monte Carlo) |
| Per-token vs per-traj advantage | per-token (GAE) | per-trajectory (broadcast) |
| Importance ratio | per-token $\rho_t$ | per-token $\rho_{i,t}$ |
| Clip | symmetric $[1-\varepsilon, 1+\varepsilon]$ | symmetric $[1-\varepsilon, 1+\varepsilon]$ |
| KL to reference | optional, adaptive | explicit per-token, coeff $\beta$ |
| Trainable networks | policy + value | policy only |
| Frozen networks | reference + reward model | reference (+ rule reward) |
| Rollouts per question | 1 | $G$ (group) |

Read that table top to bottom and you can see GRPO is not a new algorithm so much as a *surgical edit* of PPO: every row that mentions the value network changes, and nothing else does. That is by design. DeepSeek wanted the smallest possible change to a battle-tested algorithm that would let them delete the critic, and the group-relative advantage is exactly that minimal edit. The reuse of PPO's clip and importance ratio is what lets GRPO inherit PPO's stability and its ability to take several gradient steps per batch — properties that were hard-won over years of trust-region research and that you do not want to re-litigate.

## Why rule-based rewards beat reward models for reasoning

A piece of GRPO that is easy to underweight is the *reward function*, and DeepSeek made a deliberate, somewhat contrarian choice here that turned out to be load-bearing. For verifiable domains — math with known answers, code with unit tests — they used **rule-based rewards**, not a learned reward model.

The reasoning is about reward hacking. A learned reward model is a neural network trained to imitate human preference judgments, and like any learned function it has exploitable seams. Optimize against it hard enough and the policy finds inputs where the reward model is confidently wrong — answers that score high but are actually bad. This is the central failure mode of RLHF, and it is why the [reward modeling literature](/blog/machine-learning/training-techniques/deepseekmath-data-pipeline-and-grpo-origin) spends so much effort on keeping the policy near the reward model's training distribution. For a domain where correctness is *checkable*, you can sidestep the whole problem. A math answer either matches the ground truth or it does not. A program either passes its tests or it does not. There is no seam to exploit because the reward is not a learned approximation of correctness — it *is* correctness.

DeepSeek-R1 used two kinds of rule-based reward:

- **Accuracy reward**: for math, parse the model's final answer (the contents of `\boxed{}`) and compare it to the ground truth, with some normalization for equivalent forms. Reward 1 if equal, 0 otherwise. For code, run the submission against test cases and reward the fraction that pass (or a binary all-pass).
- **Format reward**: a small reward for putting the reasoning inside `<think>...</think>` tags and the final answer after them. This shapes the *structure* of the output without trying to judge its *content*, which keeps it un-hackable — you cannot game a format check by being wrong-but-confident.

The deeper point is philosophical. RLHF with a learned reward model is trying to capture fuzzy human preference, and fuzzy targets get gamed. Reasoning RL with a rule-based reward is optimizing against ground truth, and ground truth does not get gamed — the worst the model can do is find a *correct* answer you didn't expect, which is a feature. This is why GRPO took off in math and code first: those are exactly the domains where a cheap, un-hackable, verifiable reward exists. The matrix figure below places GRPO against the other methods on the axes that actually decide which one you reach for.

![Comparison matrix of GRPO, PPO, REINFORCE, DPO, and rejection sampling across whether a critic is needed, memory cost, gradient variance, suitability for math, and where each was deployed](/imgs/blogs/grpo-group-relative-policy-optimization-5.png)

## A from-scratch GRPO implementation in PyTorch

Theory is cheap; let us write the thing. We will build the GRPO update in stages — group sampling, advantage computation, the per-token log-probabilities, the clipped loss with KL — and assemble them into a training step. The code uses plain PyTorch and Hugging Face `transformers` so it runs against any causal LM. I have kept it readable rather than maximally optimized; a production loop would add vLLM for generation and FSDP for sharding, but the logic is identical.

First, group sampling. For each prompt in the batch we generate $G$ completions. The key flag is sampling with a temperature high enough to get genuine diversity in the group — a group of eight identical answers carries no signal.

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B", torch_dtype=torch.bfloat16, device_map="cuda")
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B", torch_dtype=torch.bfloat16, device_map="cuda")
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

GROUP_SIZE = 8
MAX_NEW = 1024

def sample_group(prompt_text):
    # Expand one prompt into GROUP_SIZE identical copies, then sample G diverse completions.
    msgs = [{"role": "user", "content": prompt_text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok([prompt] * GROUP_SIZE, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW,
            do_sample=True,
            temperature=0.9,        # high enough for a diverse group
            top_p=1.0,
            pad_token_id=tok.pad_token_id,
        )
    gen = out[:, enc.input_ids.shape[1]:]   # strip the prompt, keep completions
    return enc.input_ids, gen
```

Next, the reward. For a math task this is a verifier, not a network. We parse the boxed answer and compare to ground truth, plus a small format bonus for using the think tags. This function is the entire "reward model" for verifiable reasoning.

```python
import re

def extract_boxed(text):
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    return m.group(1).strip() if m else None

def reward_fn(completion_text, ground_truth):
    # Accuracy reward: 1.0 if the boxed answer matches ground truth, else 0.0.
    pred = extract_boxed(completion_text)
    accuracy = 1.0 if (pred is not None and pred == ground_truth) else 0.0
    # Format reward: small bonus for well-formed reasoning structure.
    has_think = bool(re.search(r"<think>.*</think>", completion_text, re.DOTALL))
    fmt = 0.1 if has_think else 0.0
    return accuracy + fmt
```

Now the advantage. This is the GRPO heart — take the $G$ rewards, center by the mean, scale by the std, broadcast the per-trajectory advantage to every token.

```python
def group_advantages(rewards):
    # rewards: tensor of shape (G,) -- one scalar reward per completion in the group.
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    mu = rewards.mean()
    sigma = rewards.std(unbiased=False)
    advantages = (rewards - mu) / (sigma + 1e-6)   # eps guards a degenerate group
    return advantages   # shape (G,), one scalar per response
```

To form the clipped objective we need per-token log-probabilities under both the current policy and the old policy (the one that generated the samples). A helper turns a model and a token sequence into the log-prob of each generated token.

```python
def token_logprobs(a_model, prompt_ids, gen_ids):
    # Concatenate prompt + generation, run a forward pass, gather the log-prob of each generated token.
    full = torch.cat([prompt_ids, gen_ids], dim=1)
    logits = a_model(full).logits[:, :-1, :]          # predict token t from positions < t
    logits = logits[:, prompt_ids.shape[1] - 1:, :]   # keep only positions that predict generated tokens
    logp = F.log_softmax(logits.float(), dim=-1)
    token_lp = torch.gather(logp, 2, gen_ids.unsqueeze(-1)).squeeze(-1)
    return token_lp   # shape (G, gen_len)
```

Finally, the GRPO loss. We combine the per-token ratio, the broadcast advantage, the clip, the length normalization, and the KL penalty into one scalar to minimize.

```python
EPS = 0.2
BETA = 0.04

def grpo_loss(prompt_ids, gen_ids, advantages, old_logp, mask):
    # advantages: (G,)  -- broadcast to every token of each response.
    # old_logp:   (G, L) -- log-probs under the sampling (old) policy, no grad.
    # mask:       (G, L) -- 1 for real generated tokens, 0 for padding.
    new_logp = token_logprobs(model, prompt_ids, gen_ids)        # (G, L), requires grad
    with torch.no_grad():
        ref_logp = token_logprobs(ref_model, prompt_ids, gen_ids)  # (G, L), frozen reference

    ratio = torch.exp(new_logp - old_logp)                        # rho_{i,t}
    adv = advantages.unsqueeze(1)                                 # (G, 1) -> broadcasts over tokens

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv
    surrogate = torch.min(unclipped, clipped)                     # one-sided brake

    # Unbiased low-variance KL estimator: exp(d) - d - 1, with d = ref_logp - new_logp.
    d = ref_logp - new_logp
    kl = torch.exp(d) - d - 1.0

    per_token = surrogate - BETA * kl
    # Per-response length normalization, then average over the group.
    per_response = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    loss = -per_response.mean()                                   # maximize objective -> minimize negative
    return loss
```

Tie it together into a training step. The structure mirrors the pipeline figure: sample groups, score them, normalize advantages, then do a few inner epochs of the clipped update on the same batch before throwing it away and generating fresh.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
INNER_EPOCHS = 2

def train_step(prompt_text, ground_truth):
    prompt_ids, gen_ids = sample_group(prompt_text)
    mask = (gen_ids != tok.pad_token_id).float()

    texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
    rewards = [reward_fn(t, ground_truth) for t in texts]
    advantages = group_advantages(rewards).to("cuda")

    with torch.no_grad():
        old_logp = token_logprobs(model, prompt_ids, gen_ids)   # snapshot the sampling policy

    for _ in range(INNER_EPOCHS):
        loss = grpo_loss(prompt_ids, gen_ids, advantages, old_logp, mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return float(loss), sum(rewards) / len(rewards)
```

That is a complete, runnable GRPO loop in well under a hundred lines, and there is no value network anywhere in it. The pipeline figure below is this exact step drawn as a flow — questions in, a policy update out, no critic in the loop.

![Pipeline diagram of one GRPO training step flowing from sampling a batch of questions to generating G responses each, computing rule rewards, normalizing advantages, computing the clipped loss, and updating the policy](/imgs/blogs/grpo-group-relative-policy-optimization-4.png)

In practice you would not write `model.generate` in the training loop — generation is by far the most expensive operation, and you would run it with vLLM or SGLang on separate inference workers, then ship the completions and their token log-probs back to the trainer. TRL's `GRPOTrainer` does exactly this and is the path most teams take in production; the snippet below is the same algorithm at the framework level.

```python
from trl import GRPOConfig, GRPOTrainer

def math_reward(completions, ground_truth, **kwargs):
    out = []
    for c, gt in zip(completions, ground_truth):
        pred = extract_boxed(c)
        out.append(1.0 if pred == gt else 0.0)
    return out

config = GRPOConfig(
    output_dir="grpo-math",
    learning_rate=1e-6,
    num_generations=8,          # the group size G
    per_device_train_batch_size=8,
    max_completion_length=1024,
    temperature=0.9,
    beta=0.04,                  # KL coefficient
    epsilon=0.2,                # clip range
    use_vllm=True,              # offload generation to vLLM
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-Math-1.5B",
    reward_funcs=math_reward,
    args=config,
    train_dataset=math_dataset,
)
trainer.train()
```

## Why the group baseline actually lowers variance

We argued that the group mean is a valid baseline. But is it a *good* one — does it actually reduce variance compared to the alternatives? This is where GRPO sits in an interesting middle ground, and being precise about it tells you when GRPO will shine and when it won't.

Consider three baselines for the same policy gradient. **No baseline** (plain REINFORCE): you multiply each trajectory's gradient by its raw reward. Highest variance, because the common-mode reward term is fully present. **A single-sample baseline**: for each question you sample one extra answer and use its reward as the baseline for the others. This centers the reward but with a single noisy sample — the baseline itself is high-variance, so you have traded one noise source for another. **A learned value function** (PPO): a network that has averaged over many questions to produce a smooth, low-variance estimate of expected reward. Lowest variance per sample, but you paid for it with a whole second network and the noise of training it on a hard prediction problem.

GRPO's group mean sits between the single-sample baseline and the learned value function. It is a Monte Carlo estimate of the expected reward, like the single-sample baseline, but averaged over $G$ samples instead of one, so its variance is roughly $1/G$ of the single-sample case. With $G = 8$ the baseline is eight times less noisy than a single-sample baseline. It does not match the learned value function's variance — a well-trained critic has effectively averaged over thousands of questions — but it never has to be *trained*, so it never injects the critic's own estimation noise, and for sparse terminal rewards where the critic is nearly useless anyway, the group mean is often the better-conditioned estimate in practice. The graph figure below contrasts the noisy single-sample path with the averaged group path.

![Diagram contrasting a single sample baseline producing a high variance gradient against a group of responses whose shared mean produces a low variance stable gradient](/imgs/blogs/grpo-group-relative-policy-optimization-7.png)

This is also why the group size matters and why there is a floor. With $G = 2$ the standard deviation in the denominator is estimated from two points and is itself wildly noisy; the normalization can blow up. With $G = 4$ it becomes usable, and $G = 8$ to $16$ is the sweet spot most recipes land on — enough samples to get a stable mean and std without spending too much generation compute per gradient step. Pushing $G$ higher keeps helping the baseline but with diminishing returns, while the generation cost grows linearly, so somewhere around 8–16 the marginal sample stops being worth its compute.

#### Worked example: GRPO versus PPO memory and compute

Let us make the trade-off concrete with a 7B model. Under PPO-RLHF you hold four 7B networks: the **policy** being trained (with optimizer state, so roughly 4 bytes for the bf16 weights plus 12 bytes per parameter for AdamW moments and master copy — about 112 GB for 7B), the **value network** (another full network with its own optimizer state, another ~112 GB if same size, though often a smaller head), the **reward model** (inference only, ~14 GB in bf16), and the frozen **reference** (inference only, ~14 GB). Even being generous about sharing, the value network roughly doubles your *trainable* footprint, which is the expensive part because of optimizer state.

Under GRPO you hold the **policy** (~112 GB with optimizer state), the frozen **reference** (~14 GB), and *no value network and no learned reward model at all* — the reward is a Python function that runs a verifier. You have deleted the single most expensive auxiliary object, the trainable value network with its optimizer state. On a node where PPO would force you to shard aggressively or drop batch size, GRPO fits with room to spare.

The compute trade runs the other way, and it is the honest cost. GRPO generates $G$ completions per question instead of one, so each gradient step costs roughly $G\times$ the generation compute. Generation dominates the wall-clock in LLM RL, so this is not a rounding error — a GRPO step with $G = 8$ does about eight times the rollout work of a single-rollout step. The bet GRPO makes is that this extra generation compute, which parallelizes trivially across inference workers, is cheaper and simpler to scale than maintaining and training a value network that was producing noise anyway. For reasoning, where generation is already the bottleneck and the critic was nearly worthless, that bet has paid off repeatedly.

## The DeepSeek-R1 recipe: from cold start to the aha moment

GRPO is the optimizer, but DeepSeek-R1 is a *recipe* — a sequence of stages where GRPO is the engine. Understanding the recipe matters because the most striking result, R1-Zero learning to reason with no supervised warm-up at all, came from GRPO applied directly to a base model, and the production R1 wrapped that in stages to fix the rough edges. Here is the pipeline as the DeepSeek-R1 paper describes it.

**R1-Zero: pure RL from the base model.** The headline experiment applied GRPO directly to the DeepSeek-V3 base model with only rule-based rewards — no SFT, no human-written reasoning traces, nothing but "here is a math problem, you get reward 1 if your boxed answer is right." And it worked. Over thousands of GRPO steps, the model's accuracy on the AIME 2024 competition math benchmark climbed from around 15.6% to 71.0% pass@1, and to 86.7% with majority voting over samples — comparable to OpenAI's o1 on the same benchmark. More striking than the number was *how* it improved: the model spontaneously learned to generate longer chains of thought, to backtrack, to check its own work. None of that was in a training label. It emerged from the pressure of the reward.

**The aha moment.** The DeepSeek paper documents a specific, almost eerie phenomenon: at some point in training, R1-Zero began producing text like "Wait, wait. Wait. That's an aha moment I can flag here. Let me reconsider..." in the middle of a solution, and then *re-deriving* a step it had gotten wrong. The model was not told to second-guess itself. The behavior emerged because, statistically, answers that included a self-correction step were more often correct, so the group-relative advantage kept pushing the probability of that behavior up. This is the cleanest demonstration in the literature that reasoning strategies are *learnable from outcome reward alone* — you do not need to demonstrate the strategy, you only need to reward the outcome and let GRPO find the strategy that produces it. It is, in a real sense, the policy gradient theorem doing exactly what it promises: reinforcing whatever precedes high reward.

**Why R1 added a cold start.** R1-Zero had two warts: its chains of thought were sometimes unreadable (mixing languages, rambling) and it could be unstable early in training. So the production R1 recipe added stages around the GRPO core. First a **cold-start SFT** on a few thousand curated, well-formatted chain-of-thought examples, to give the model a clean starting style before RL. Then a **reasoning-oriented GRPO** stage — the main event — that drives the math and code accuracy up. Then **rejection sampling**: use the GRPO-improved model to generate many answers to a broad prompt set, keep only the correct ones (filtered by the verifier and by the model's own judgment for non-verifiable tasks), and do another round of SFT on this self-generated high-quality data, which broadens the model beyond just math and code. Finally a **second RL stage** that includes both rule-based rewards for reasoning and a helpfulness/harmlessness reward for general alignment, so the final model is both a strong reasoner and a usable assistant.

**Distillation.** The last move is the one that put R1-level reasoning in everyone's hands. DeepSeek took the data generated by the full R1 model and used it to do *plain SFT* on smaller dense models — Qwen and Llama checkpoints from 1.5B up to 70B. No RL on the small models at all: just supervised fine-tuning on R1's outputs. The distilled 7B and 32B models posted reasoning numbers that beat much larger models trained without this data, and the paper made the point explicitly that distilling from a strong RL-trained teacher is more effective than running RL on the small model directly. The timeline figure below places GRPO on the longer arc from REINFORCE to reasoning models.

![Timeline of reasoning reinforcement learning from REINFORCE in 1992 through PPO, AlphaCode, DeepSeek-Math GRPO in 2024, DeepSeek-R1 in 2025, and the DAPO and Dr GRPO fixes](/imgs/blogs/grpo-group-relative-policy-optimization-6.png)

## Temperature, length, and the practical knobs

Three practical details separate a GRPO run that converges from one that collapses, and all three trace back to the group structure.

**Temperature controls group diversity, which controls signal.** The group baseline only works if the group has variance — if all $G$ answers are identical (all correct or all wrong), the standard deviation is near zero, the advantages are undefined (or saturate after the epsilon guard), and that question contributes no gradient. So you sample at a temperature high enough to get genuine diversity, typically $T$ between 0.6 and 1.0. Too low and groups collapse to identical answers, killing the signal. Too high and the model produces incoherent text whose rewards are uninformative noise. The right temperature keeps groups diverse enough that, on the questions the model is currently learning, some answers land right and some land wrong — that mix is exactly where the gradient is most informative.

**Length normalization prevents a degenerate optimum, but introduces a bias.** Recall the inner $\frac{1}{|o_i|}$ in the objective. Without it, a long answer contributes more total gradient than a short one simply by having more tokens, which would push the model toward verbosity regardless of correctness. The normalization equalizes them. But — and this is the subtle part that Dr. GRPO later attacked — dividing by length means that for a *correct* (positive advantage) answer, making it longer dilutes the per-token gradient, while for an *incorrect* (negative advantage) answer, making it longer dilutes the penalty. The optimizer can exploit this: it learns that wrong answers should be long (to dilute the penalty per token) and the result is the well-documented "length bias" where GRPO-trained models pad their failures. More on the fix shortly.

**The minimum group size and degenerate groups.** Questions where the whole group is correct or the whole group is wrong produce zero advantage and waste their generation compute. Some recipes filter these out (DAPO's dynamic sampling does exactly this) so that every group in a batch carries a real gradient. At minimum, the epsilon in the standard-deviation denominator (`sigma + 1e-6` in our code) keeps a degenerate group from producing infinities, but those groups still contribute nothing useful.

For a starting point you can actually plug into a run, here is the hyperparameter set most GRPO recipes converge on, what each knob does, and the failure you see when it is wrong:

| Hyperparameter | Typical value | Effect | Symptom when wrong |
| --- | --- | --- | --- |
| Group size $G$ | 8–16 | samples per question; sets baseline quality | $G \le 2$: noisy std, advantages blow up |
| Temperature $T$ | 0.6–1.0 | group diversity | too low: collapsed groups, no signal |
| Clip $\varepsilon$ | 0.2 | trust region on the ratio | too high: cliff-style instability |
| KL coeff $\beta$ | 0.0–0.04 | leash to reference | too high: learning stalls; 0: drift risk |
| Learning rate | 1e-6 to 2e-6 | step size | too high: entropy collapse, reward spikes then craters |
| Inner epochs | 1–4 | gradient steps per batch | too many: ratios drift, clip saturates |
| Max completion length | 1k–8k+ | room for reasoning | too short: truncated chains score 0 |
| Batch (questions) | 64–1024 | gradient stability | too small: noisy advantage averaging |

Two of these interact in a way that bites people. The learning rate for LLM GRPO is *tiny* — 1e-6 is a thousand times smaller than what you might use for supervised fine-tuning — because the policy is already a competent model and you are only nudging it. Crank it up to speed things along and you get the classic entropy-collapse signature: reward climbs fast for a few hundred steps, the policy becomes overconfident, groups stop being diverse, and the whole thing flatlines or craters. The fix is almost always "lower the learning rate," not "tune something clever."

#### Worked example: reading a healthy versus collapsing GRPO run

Suppose you launch a GRPO run on GSM8K with a 1.5B model, $G = 8$, and you log three quantities every 50 steps: mean group reward, mean group standard deviation (your diversity proxy), and mean completion length. A *healthy* run looks like this. Mean reward starts around 0.45 and climbs steadily to 0.78 over a few thousand steps. Group std starts around 0.49 (close to the maximum for binary rewards, which is 0.5 at a 50/50 split) and *stays high* — drifting down only slowly to maybe 0.38 as the model gets good — because the model keeps encountering questions at the edge of its ability where answers genuinely vary. Completion length creeps up, say from 180 tokens to 320, as the model learns to write out more steps. That length growth is the good kind: the chains get longer because longer reasoning earns more reward.

A *collapsing* run looks different in a way you can catch early. Mean reward spikes from 0.45 to 0.70 in the first 200 steps — suspiciously fast — and then plateaus or slides back to 0.60. Group std drops off a cliff, from 0.49 to 0.15 within a few hundred steps: the model has become overconfident, every group now contains eight near-identical answers, and most groups are either all-right or all-wrong, contributing zero gradient. Completion length either explodes (the model padding wrong answers to dilute the penalty, the length-bias pathology) or collapses (the model emitting terse, low-entropy answers). The diagnostic that separates the two cases is the group std: as long as it stays healthy, the baseline is informative and learning continues; once it collapses, you are burning generation compute on degenerate groups and no amount of patience will recover it. Catch it by logging group std, and respond by lowering the learning rate, raising the temperature, or — the DAPO move — raising the upper clip bound to let exploration back in.

## Extending GRPO: DAPO and Dr. GRPO

Vanilla GRPO works, but a wave of 2025 follow-ups identified specific biases in the standard formulation and patched them. Two are worth knowing because they are now common in serious recipes.

**DAPO** (Decoupled Clip and Dynamic Sampling Policy Optimization, from a ByteDance/Tsinghua collaboration) targets **entropy collapse** — the failure mode where, as training proceeds, the policy becomes increasingly confident and deterministic, the groups stop being diverse, and learning stalls. DAPO's main interventions: **Clip-Higher** decouples the upper and lower clip bounds, raising the upper bound (e.g. $\varepsilon_{\text{high}} = 0.28$ vs $\varepsilon_{\text{low}} = 0.2$) so that low-probability-but-promising tokens can still gain probability — the symmetric clip was quietly suppressing exploration by capping the upside of rare good tokens. **Dynamic Sampling** filters out the degenerate all-correct and all-wrong groups and keeps sampling until the batch is full of groups that actually carry gradient, so no compute is wasted on zero-advantage groups. **Token-level loss** changes the normalization so that longer sequences are not down-weighted relative to short ones at the batch level. And DAPO often runs with $\beta = 0$ — no KL penalty at all — on the argument that for verifiable rewards the leash is unnecessary and the KL term mostly slows learning. DAPO reported reaching 50 points on AIME 2024 with the Qwen2.5-32B base, outperforming a comparable vanilla-GRPO setup with fewer training steps.

**Dr. GRPO** ("GRPO Done Right," from Sea AI Lab and collaborators) is a sharper, more surgical critique. It identifies two optimization biases baked into the standard GRPO objective. The first is the **length bias** from the $\frac{1}{|o_i|}$ per-response normalization we discussed — Dr. GRPO removes the length division, summing the per-token losses instead of averaging them, which kills the incentive to pad wrong answers. The second is a **difficulty bias** from dividing by the group standard deviation: questions with very low variance (almost all correct or almost all wrong) get their advantages scaled *up* by the small denominator, over-weighting easy and impossible questions relative to the medium-difficulty ones where learning actually happens. Dr. GRPO removes the standard-deviation normalization too, using just $A_i = r_i - \mu$ — the raw centered reward, no division. The claim, backed by their experiments, is that these two removals make the optimization unbiased and improve token efficiency: the model reaches the same accuracy with shorter, less-padded chains of thought.

Let us trace the length bias precisely, because it is the most instructive of the two and the easiest to reproduce in your own runs. The standard GRPO loss for a response divides the summed per-token loss by the response length $|o_i|$. Now consider a *wrong* answer with advantage $A_i < 0$. Its contribution to the objective is the negative advantage summed over its tokens and divided by its length — which is just the advantage itself, length-independent at first glance. But the gradient that the model receives *per token* is the advantage divided by length. So a longer wrong answer delivers a *smaller* downward gradient to each of its tokens. The optimizer, which sees gradients per parameter and per token, discovers that it can soften the penalty on any individual token of a wrong answer by making the answer longer. Over training, wrong answers grow. You can watch this happen: plot mean completion length split by correct versus incorrect, and in a vanilla-GRPO run the incorrect curve climbs faster. Dr. GRPO's fix — sum the per-token losses without dividing by length — removes the per-token dilution and the incentive to pad disappears. The difficulty bias is analogous but on the question axis: dividing by a small group std inflates the advantages of near-degenerate (very easy or very hard) questions, so the gradient spends disproportionate effort on questions where it can learn the least. Removing the std division weights every question by its raw reward spread, which is what you want.

The meta-lesson across both is that GRPO's elegance came at the cost of a few quietly biased terms, and the second generation of methods is about getting the estimator *honest* — removing the normalizations that, while well-intentioned, were nudging the gradient in directions you did not ask for. If you are starting a GRPO project today, the practical default is closer to Dr. GRPO's objective (no length division, no std division, careful clip) than to the literal DeepSeekMath formula. None of this changes the central idea — the group is still the baseline — it just cleans up the arithmetic around it. And that is a healthy sign for a young method: the core insight survived contact with reality, and the patches are all in the second-order details.

A practical note on the KL term ties the two methods together. DeepSeekMath's original formula kept $\beta = 0.04$; DAPO drops it to zero. Which is right depends on your reward. With a truly un-hackable verifier — exact-match on a math answer — the reference leash buys you little, because the model cannot drift toward high-reward gibberish when the only way to score is to be *correct*. There, $\beta = 0$ removes a term that mostly slows learning. But the moment your reward has *any* exploitable seam — a format bonus the model can spam, a verifier that accepts a degenerate edge case, or a learned reward model — the KL leash earns its keep by keeping the policy near the well-behaved reference. So the rule of thumb is: drop the KL only when you are confident the reward is genuinely un-gameable, and keep it otherwise.

## Case studies

**Earlier RL-on-reasoning: AlphaCode and the score-function lineage.** GRPO did not arrive in a vacuum. DeepMind's AlphaCode (2022) had already shown that RL-style filtering on a verifiable reward — generate many programs, keep the ones that pass tests — could lift code-generation to competitive-programming median, and the underlying score-function estimator dates all the way back to Williams' REINFORCE in 1992. What was missing in that whole lineage was a *cheap, low-variance baseline that scaled to long language generations* — REINFORCE was too noisy, and PPO's learned critic was too expensive and too noisy in exactly the sparse-terminal-reward setting reasoning lives in. GRPO's contribution is best understood as filling that specific gap: the group mean is the baseline that the reasoning setting had been missing, simple enough to scale and accurate enough to learn from. Seeing it in this lineage is what the history timeline figure above is for — GRPO is one well-placed step, not a break from the past.

**DeepSeek-Math (Shao et al., 2024).** The paper that introduced GRPO applied it to DeepSeekMath-Instruct 7B after a chain-of-thought SFT stage. GRPO with rule-based rewards on the GSM8K and MATH training sets lifted GSM8K accuracy from 82.9% to 88.2% and MATH from 46.8% to 51.7% for the 7B model — a several-point gain on hard benchmarks from RL alone, using no value network. The paper's contribution was framed precisely as removing the critic to cut the memory and compute of PPO-style RL for LLMs while matching or beating its results, and it laid out the group-relative advantage as the mechanism. The honest read of the DeepSeek-Math numbers is that GRPO did not produce a *dramatically* better model than a well-tuned PPO would have — the gains over PPO were in the low single digits — but it produced a comparable model at roughly half the trainable-memory footprint and with a far simpler implementation. That efficiency-and-simplicity story, not a raw-accuracy story, is what made the method spread.

**DeepSeek-R1 and R1-Zero (2025).** The R1 work scaled GRPO to a frontier base model and demonstrated the most consequential result: R1-Zero, with *pure* GRPO and no SFT warm-up, reached 71.0% pass@1 on AIME 2024 (86.7% with majority voting) up from a base around 15.6%, and matched OpenAI o1-level reasoning on several benchmarks. The full R1 (with the cold-start and multi-stage recipe) posted 79.8% pass@1 on AIME 2024 and strong results on MATH-500, Codeforces, and GPQA. The "aha moment" — emergent self-correction with no supervision for it — became the most-cited single observation, because it showed reasoning behavior arising purely from outcome-reward pressure.

It is worth being explicit about *why* the multi-stage recipe is shaped the way it is, because the ordering is not arbitrary — each stage exists to fix a specific failure of the previous one. The cold-start SFT exists because pure RL from a base model wanders early: with no sense of what a clean answer looks like, the first thousand GRPO steps are spent learning basic formatting that a few hundred SFT examples teach for free, and the rambling, language-mixed chains of R1-Zero were the visible symptom. The reasoning GRPO stage exists because that is where the actual capability gain lives — it is the only stage that *teaches the model to reason better* rather than to imitate. The rejection-sampling stage exists because GRPO on a narrow math-and-code prompt set produces a narrow model; by generating from the improved model across a broad prompt distribution and keeping only the good outputs, you manufacture a broad, high-quality SFT set that generalizes the reasoning skill outward without needing reward signals for every domain. And the final mixed-reward RL stage exists because a model that is a brilliant reasoner but an unpleasant assistant is not shippable; folding helpfulness and harmlessness rewards back in at the end re-aligns the now-strong reasoner to be usable. Read as a whole, the recipe is a careful interleaving of imitation (cheap, broad, stabilizing) and reinforcement (expensive, narrow, capability-building), each covering the other's weakness.

**Distilled R1 models.** Plain SFT of smaller Qwen and Llama checkpoints on R1-generated data produced DeepSeek-R1-Distill-Qwen-7B and -32B, which beat much larger non-reasoning models on math benchmarks — the 32B distill, for instance, posted AIME and MATH numbers competitive with models several times its size. The case study within the case study is that *RL produced the teacher, distillation spread the capability*, and that for small models distillation from a GRPO-trained teacher beat running GRPO on the small model directly. The reason is sample efficiency: a small model running its own GRPO has to *discover* good reasoning traces through expensive exploration, succeeding only on the small fraction of problems already within its reach, while distillation hands it tens of thousands of already-correct, already-well-structured traces from a stronger model to imitate directly. Discovery is hard; imitation of a good demonstrator is easy. So if a strong RL-trained teacher exists, the small-model path is SFT on its outputs, full stop.

**The open-source wave.** Within months of R1, the open-source community reproduced the recipe at small scale — projects training 1.5B and 3B models with GRPO on math datasets and observing the same length-growth and self-correction dynamics on a single node. TRL shipped `GRPOTrainer`, making the algorithm a few lines of config, and the open-r1 effort set out to reproduce the full pipeline openly. The speed of reproduction is itself evidence of GRPO's main selling point: it is *simple* — no critic to tune, a rule for a reward — so a small team can run it.

## When to use GRPO (and when not to)

GRPO is not a universal upgrade over PPO; it is the right tool for a specific shape of problem, and being honest about the boundary saves you from misapplying it.

**Reach for GRPO when you have a verifiable reward.** The single strongest predictor of GRPO success is whether correctness is checkable by a rule — math with known answers, code with tests, structured tasks with a programmatic grader. There the rule-based reward is un-hackable, the group-relative advantage is well-behaved, and you get all of GRPO's simplicity and memory savings with none of the reward-model fragility. This is the domain it was built for and where every headline result lives.

**Reach for GRPO when critic budget is the binding constraint.** If you are memory-bound on a large model and the value network is what is pushing you over the edge — forcing aggressive sharding or a tiny batch — GRPO's deletion of the trainable critic is a direct, large win. You trade it for more generation compute, which scales out across cheap inference workers more gracefully than optimizer state scales.

**Prefer PPO with a learned critic when reward is dense and per-step.** For classic control and robotics — CartPole, MuJoCo locomotion, anything with reward at every timestep — a value function genuinely earns its keep because it can learn the rich per-state value landscape, and the [PPO post](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) covers why GAE over that critic is so effective there. GRPO's group baseline is a per-trajectory scalar; it throws away the per-step structure that PPO's critic exploits. For dense-reward control, PPO wins.

**Prefer DPO or reward-model PPO when the target is fuzzy human preference.** If there is no programmatic check for "good" — open-ended helpfulness, tone, style, safety — you are back in the world of learned preference, and either Direct Preference Optimization on pairwise data or [PPO](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) against a reward model is the appropriate tool. GRPO needs a reward it can compute on a group of fresh samples; if scoring requires a human or a fragile reward model, the un-hackability advantage evaporates. That said, the line is blurring: practitioners increasingly run GRPO even on preference tasks by turning a reward model's scalar score into the group reward, accepting some reward-hacking risk in exchange for GRPO's simplicity — a sign of how far the critic-free idea has spread beyond its math-and-code origin.

**Prefer distillation when you have a strong teacher and a small student.** If a GRPO-trained model already exists at large scale, do not run GRPO on your small model — distill. DeepSeek's own ablations show SFT on teacher outputs beats RL-from-scratch on the small model, at a fraction of the cost.

The decision tree figure below compresses this into the order you actually ask the questions: is the reward verifiable, is critic budget tight, is the target human preference, do you have a teacher to distill from.

![Decision tree for choosing a reasoning RL method starting from whether the reward is verifiable, then critic budget, leading to GRPO, PPO, DPO, or distillation](/imgs/blogs/grpo-group-relative-policy-optimization-8.png)

## Key takeaways

- **The critic's real job is to be a baseline.** A value function in policy-gradient RL exists to center the reward so the gradient sees "better than average" instead of raw reward. GRPO computes that baseline by Monte Carlo — the group mean — instead of learning it with a network.
- **The group-relative advantage is $A_i = (r_i - \mu)/\sigma$ over $G$ sampled answers to the same question.** It is unbiased (any action-independent baseline is) and lower-variance than a single-sample baseline by roughly a factor of $G$.
- **The same advantage is broadcast to every token of a response** in the outcome-supervision variant, because the reward is a single terminal bit — there is no principled per-token credit to assign.
- **Rule-based rewards beat learned reward models for verifiable tasks.** Ground truth cannot be gamed; a learned reward model can. This is why GRPO took off in math and code first.
- **GRPO trades a trainable critic for more generation compute.** It removes the most memory-expensive auxiliary network and pays with $G\times$ the rollouts per step — a good trade when generation parallelizes and the critic was producing noise anyway.
- **Group diversity is the signal.** Sample at temperature 0.6–1.0 and use $G \ge 4$ (8–16 is the sweet spot); a degenerate all-correct or all-wrong group carries zero gradient.
- **Reasoning strategies emerge from outcome reward alone.** R1-Zero learned to backtrack and self-correct with no supervision for those behaviors — the "aha moment" is the policy gradient reinforcing whatever precedes high reward.
- **The DeepSeek-R1 recipe wraps GRPO in stages**: cold-start SFT, reasoning GRPO, rejection sampling, a final mixed-reward RL stage, then distillation to smaller models.
- **The vanilla GRPO objective has subtle biases.** The per-response length division pads wrong answers; the std normalization over-weights easy and impossible questions. DAPO and Dr. GRPO fix these; modern recipes lean toward the corrected objectives.

## Further reading

- Shao, Wang, et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (2024) — the paper that introduced GRPO and the group-relative advantage.
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025) — the R1 and R1-Zero recipe, the aha moment, and the distillation results.
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017) — the clipped surrogate objective GRPO inherits; see also Schulman's note on KL estimators for the low-variance KL term.
- Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (2025) — Clip-Higher, dynamic sampling, and the entropy-collapse fixes.
- Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective" / Dr. GRPO (2025) — the length-bias and difficulty-bias critique and the unbiased objective.
- Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed.) — chapter 13 on policy gradients and baselines, the foundation under all of this.
- Within the series: the [policy gradient theorem](/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem) for the derivation GRPO builds on, [Proximal Policy Optimization](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) for the clipped objective and the RLHF connection, [Trust Region Policy Optimization](/blog/machine-learning/reinforcement-learning/trust-region-policy-optimization-trpo) for the trust-region origin of the clip, and the [DeepSeekMath data pipeline and GRPO origin](/blog/machine-learning/training-techniques/deepseekmath-data-pipeline-and-grpo-origin) for the data side of the story. The unified map and the playbook capstone tie GRPO into the full taxonomy of RL methods once they land.

---
title: "MiniMax-M1: CISPO and the Economics of Test-Time Compute"
publishDate: "2026-06-10"
date: "2026-06-10"
category: "paper-reading"
subcategory: "Large Language Model"
tags:
  - minimax
  - cispo
  - reinforcement-learning
  - reasoning-models
  - test-time-compute
  - grpo
  - lightning-attention
  - reward-modeling
  - rlhf
  - long-context
description: "A deep read of MiniMax-M1: how a lightning-attention hybrid makes long chains of thought cheap, and how CISPO fixes the gradient bug that quietly mutes reasoning tokens in GRPO and PPO."
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/minimax-m1-cispo-test-time-compute-1.png"
readTime: 30
---

There is a specific moment in a reasoning model's chain of thought where the magic happens: the model writes "Wait," or "However," or "Let me recheck that," and reverses a wrong line of reasoning. Those reflective tokens are rare, they have low probability under the model's own distribution, and — this is the uncomfortable part — the most popular reinforcement-learning algorithms for training reasoning models *systematically silence them*. MiniMax-M1's central contribution, an RL objective called CISPO, is a fix for exactly that bug, and understanding why it works requires understanding why GRPO and PPO break in the first place.

MiniMax-M1 (arXiv [2506.13585](https://arxiv.org/abs/2506.13585), June 2025) is a reasoning model built on the [MiniMax-01 hybrid backbone](/blog/paper-reading/large-language-model/minimax-01-lightning-attention-hybrid-moe) — the same 456B/45.9B mixture-of-experts with lightning linear attention interleaved 7:1 with softmax. Two things make M1 worth a dedicated read. First, the architecture has a second superpower beyond long *input*: it makes long *output* cheap, which is the entire economic argument for scaling test-time compute. Second, the RL recipe that turns the base model into a reasoner contains CISPO plus a fistful of engineering fixes — an FP32 output head, no KL penalty, a repetition halt, a staged length curriculum — that are individually small and collectively the difference between an RL run that converges and one that does not. If you want the whole MiniMax lineage in one place, the [combined overview](/blog/paper-reading/large-language-model/minimax-papers-lightning-attention-cispo) is the hub; this post goes deep on the reasoning model.

![The MiniMax-M1 post-training pipeline from the Text-01 backbone through SFT to a two-stage RL curriculum](/imgs/blogs/minimax-m1-cispo-test-time-compute-1.png)

The diagram above is the mental model: M1 is not trained from scratch. It starts from the Text-01 backbone, gets a short supervised fine-tuning "cold start" to teach it the shape of a chain of thought, and then goes through a two-stage reinforcement-learning curriculum — reasoning tasks with verifiable rewards first, general tasks with a learned reward model second — that culminates in two released models, M1-40K and M1-80K, distinguished by how many tokens of "thinking" they are allowed. Everything below unpacks that pipeline.

> [!tldr] TL;DR
> - **What it claims:** A lightning-attention hybrid makes long reasoning chains cheap (≈25% of DeepSeek-R1's attention FLOPs at 100K generation), and a new RL objective, CISPO, trains the model ≈2× faster than DAPO.
> - **Why it matters:** CISPO clips a *stop-gradient importance weight* instead of masking tokens, so rare reflective tokens keep contributing gradient — a directly portable fix for a real GRPO/PPO failure mode. The full RL run cost a disclosed **$534,700** on 512 H800s in three weeks.
> - **Most surprising finding:** A bf16 LM output head silently desynced the training and inference engines' log-probabilities; computing the head in **FP32** moved their correlation from ~0.90 to ~0.99 and unblocked reward growth.
> - **Where it's soft:** The exact clip bound $\varepsilon_{\text{high}}$ is undisclosed, the rollout engine and global batch size are not stated, and M1-40K beats M1-80K on several long-context tasks — a non-monotonicity the report does not fully explain.

## Context: reasoning models and the cost of thinking

The reasoning-model era reframed where capability comes from. Instead of squeezing more knowledge into the weights, you let the model spend more *compute at inference* — a longer chain of thought, more self-checking, more exploration before committing to an answer. OpenAI's o1 and DeepSeek-R1 made this concrete: scale the test-time token budget and accuracy on hard math and code climbs. The scaling is real enough that "test-time compute" became its own axis alongside parameters and pretraining data — a third dimension you can buy capability along, by letting a fixed model think longer rather than training a bigger one. But this only works if long generations are affordable, and with standard attention they are not. Every token a reasoning model emits must attend to the entire chain of thought generated so far, so a 100,000-token reasoning trace pays a quadratic price in attention as it grows. The longer the model thinks, the more each additional thought costs — exactly the wrong cost curve for a paradigm built on thinking longer.

This is the bind M1 is designed around. If test-time compute is the new scaling axis, then the cost of a unit of test-time compute is the price of admission, and full attention makes that price climb with every token already generated. A lab that wants to compete on reasoning either eats that cost — capping how long its models can think — or changes the architecture so that thinking longer stays cheap. MiniMax chose the second path, and M1 is the result: the same hybrid backbone that gave MiniMax-01 cheap long *input* turns out to give cheap long *output* too, which is the property a reasoning model most needs. The two halves of this post — the architecture that makes thinking cheap and the RL objective that makes thinking *good* — are really one argument: that you should co-design the model and the training recipe around the cost curve of the paradigm you are betting on.

The reinforcement-learning side has its own lineage, and it is worth a paragraph because CISPO is a reaction to a specific point in it. PPO (proximal policy optimization) was the workhorse of RLHF. Its defining feature is a *clipped surrogate objective* — the trust region that stops a single update from moving the policy too far — and its defining cost is a separate *value network*, a second model the size of the policy that estimates expected return to compute the advantage. At a few billion parameters that value network is an annoyance; at hundreds of billions it is a second training job you cannot afford. GRPO (group relative policy optimization), introduced with DeepSeek's math models, deleted the value network with one idea: instead of learning a baseline, sample a *group* of completions for each prompt and normalize each one's reward against the group's mean and standard deviation. The group itself becomes the baseline. That made reasoning-RL cheap enough to run at scale, and it is the basis for most open reasoning-RL today. DAPO, Dr. GRPO, GSPO, and a wave of 2025 variants then refined GRPO's clipping, normalization, and sampling. The blog's [survey of GRPO, DAPO, Dr. GRPO, and GSPO](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo) maps that space.

CISPO is the MiniMax entry in this lineage, and it keeps GRPO's group-normalized advantage — it does not bring back the value network — but it disagrees sharply with the entire family on *one* point: where to put the clip relative to the gradient. Every member of the GRPO/PPO family clips the importance ratio *inside* the objective the gradient differentiates, which means clipping can zero a token's gradient. CISPO argues that is a bug, not a feature, and moves the clip out of the gradient path. That single relocation is the whole idea, and the rest of this post is about why it matters so much for reasoning specifically.

The gap M1 fills is the intersection of these two stories: an architecture cheap enough to make long thinking economical, and an RL objective robust enough to train that thinking without silencing the tokens that make reasoning work. The rest of this post is those two halves.

## Contributions

Tightened from the report, the contributions are four:

1. **CISPO** (Clipped IS-weight Policy Optimization), an RL objective that clips a stop-gradient importance-sampling weight rather than masking tokens, demonstrated ≈2× more sample-efficient than DAPO in a controlled study.
2. **A lightning-attention reasoning model** that exploits the hybrid backbone's linear-cost generation to make large test-time-compute budgets (40K and 80K thinking tokens) economically viable.
3. **A reproducible RL engineering recipe** — FP32 output head, no KL penalty, token-level loss normalization, a repetition halt, a staged length curriculum, and an online length-bias monitor — that stabilizes large-scale reasoning RL.
4. **A disclosed cost**: full RL training on 512 H800 GPUs in three weeks for $534,700, a rare concrete number in a field that usually hides compute.

## Why the hybrid makes long thinking cheap

The architecture is inherited from MiniMax-01, so the mechanics of lightning attention live in [that post](/blog/paper-reading/large-language-model/minimax-01-lightning-attention-hybrid-moe). What matters here is the consequence for *generation*.

![Attention FLOPs by generation length: M1 spends a fraction of DeepSeek-R1's at long output](/imgs/blogs/minimax-m1-cispo-test-time-compute-2.png)

Think of it this way. In a full-attention model, the cost of generating token number $t$ grows with $t$, because that token attends to all $t-1$ tokens before it — so the total cost of a length-$L$ generation scales as $L^2$. In the 7:1 hybrid, seven of every eight layers carry a fixed $d \times d$ state whose cost per token does not grow with $L$ at all; only the one-in-eight softmax layers pay the growing cost. The matrix above quantifies the result against DeepSeek-R1: at a 64K-token generation length, M1 spends **less than 50%** of R1's attention FLOPs, and at 100K tokens it spends **about 25%**. That is the whole economic case for the architecture in the reasoning era — a large thinking budget is affordable precisely because the marginal cost of each additional thought stays nearly flat instead of climbing.

This is why the two released models are defined by their *thinking budget* — 40K and 80K tokens — rather than by parameter count. The budget is a knob the hybrid lets you turn cheaply. A full-attention model with an 80K thinking budget would pay dearly for the tail of every long reasoning trace; M1 pays a near-constant per-token rate for seven-eighths of its layers, so it can afford to let the model ramble toward a correct answer.

It is worth making the RL consequence concrete, because it is not just an inference saving. Reinforcement learning for reasoning is dominated by *rollout* cost — you generate many long completions per prompt (a group, in GRPO terms) just to compute one gradient step. If each completion is 80K tokens and you sample, say, 16 of them per prompt, you are generating well over a million tokens to produce a single training signal, and you do this for hundreds of thousands of steps. Halving the per-token generation cost does not halve the training bill; it changes which experiments are *possible*. A full-attention model might force you to cap completions at 16K to keep rollouts affordable, which caps how much the model can learn to think. The hybrid's flat per-token cost is what lets MiniMax run rollouts at 40K and 80K without the rollout phase eating the entire compute budget — and that, more than the inference saving, is why the architecture and the reasoning recipe belong in the same paper.

## CISPO: clip the weight, not the token

Now the centerpiece. To see why CISPO exists, you have to look at what GRPO and PPO do to one specific, important kind of token.

![CISPO versus PPO and GRPO clipping: the gradient is kept rather than zeroed on reflection tokens](/imgs/blogs/minimax-m1-cispo-test-time-compute-3.png)

Both PPO and GRPO are built around a *token-level importance-sampling ratio*, $r_{i,t}(\theta) = \pi_\theta(o_{i,t}) / \pi_{\theta_{\text{old}}}(o_{i,t})$ — the probability the current policy assigns a sampled token divided by the probability the old (rollout) policy assigned it. The ratio corrects for the fact that the data was generated by a slightly stale policy. The standard objective then *clips* this ratio into a band $[1-\varepsilon, 1+\varepsilon]$ and takes a minimum, which has the effect that when the ratio leaves the band, the gradient for that token is set to zero. The clip is a trust region: it stops any single token from moving the policy too far in one update.

The before-and-after split shows the casualty. Reflective tokens — the report names "However," "Recheck," "Wait," "Aha" — are exactly the tokens whose probability is *low* under the old policy, because the base model rarely second-guesses itself. The moment RL starts rewarding correct reasoning, these tokens' current probability shoots up relative to the old policy, so their ratio $r_{i,t}$ jumps well above $1+\varepsilon$, leaves the clip band, and gets its gradient zeroed. The report's words: they "were clipped out after the first on-policy update, preventing them from contributing to subsequent off-policy gradient updates." You are silencing the precise tokens that drive the reasoning behavior you are trying to reward. The model wants to learn to say "Wait, let me recheck" more often, and the clipping mechanism keeps slamming the door.

### The objective, factor by factor

CISPO's fix is to never mask a token. Instead of clipping the ratio inside a min that can zero the gradient, it clips the importance weight, applies a *stop-gradient* so the clipped weight becomes a pure scalar coefficient, and multiplies that coefficient against the log-probability of every token.

![The CISPO objective as a product of a stop-gradient clipped weight, the advantage, and the log-probability](/imgs/blogs/minimax-m1-cispo-test-time-compute-4.png)

The objective is:

$$
J_{\text{CISPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{\sum_i |o_i|} \sum_i \sum_t \text{sg}\big(\hat{r}_{i,t}(\theta)\big)\, \hat{A}_{i,t}\, \log \pi_\theta(o_{i,t} \mid q, o_{i,\lt t})\right]
$$

Reading the three factors in the figure left to right: $\text{sg}(\hat{r}_{i,t})$ is the **stop-gradient clipped importance weight**, where $\hat{r}_{i,t} = \text{clip}(r_{i,t}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})$; $\hat{A}_{i,t}$ is the **group-normalized advantage** (GRPO-style, the reward minus the group mean over the group standard deviation); and $\log \pi_\theta(o_{i,t})$ is the **log-probability that carries the gradient**. Because the weight is wrapped in a stop-gradient, the only term the gradient flows through is the log-probability — and it flows for *every* token, regardless of how far its importance ratio strayed. Nothing is masked. In clean PyTorch:

```python
import torch

def cispo_loss(logp_new, logp_old, advantages, eps_high=0.2, eps_low=1e9):
    # logp_new, logp_old: [B, T] log-probs of sampled tokens under pi_theta and pi_old
    # advantages: [B, T] GRPO group-normalized advantage, broadcast over tokens
    ratio = torch.exp(logp_new - logp_old)                 # IS ratio r_{i,t}
    # clip the IS weight, then DETACH -> it becomes a scalar coefficient, not a gradient path.
    # eps_low is set huge, so there is no effective lower bound (the paper tunes only eps_high).
    is_weight = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high).detach()
    per_token = is_weight * advantages * logp_new          # gradient flows through logp_new for EVERY token
    return -per_token.sum() / logp_new.numel()             # token-level normalization; no KL term
```

For contrast, the GRPO objective that does the silencing:

```python
def grpo_clip_loss(logp_new, logp_old, advantages, eps=0.2):
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
    # the min() is the trust region: when ratio leaves the band on the wrong side,
    # the clipped branch is flat in `ratio`, so this token's gradient is ZERO.
    return -torch.min(unclipped, clipped).mean()
```

The structural difference is exactly where the clip lives relative to the gradient. In GRPO the clip is *inside* the term the gradient differentiates, so clipping flattens the gradient. In CISPO the clip is *outside* the gradient path (stop-gradient), so clipping only rescales the step size — it never zeros it.

The advantage $\hat{A}_{i,t}$ deserves a closer look, because it is the part CISPO keeps from GRPO. For a prompt $q$, you sample a group of $G$ completions, score each with the reward function to get rewards $R_1, \dots, R_G$, and set each completion's advantage to $\hat{A}_i = (R_i - \text{mean}(R)) / \text{std}(R)$, broadcast to every token in that completion. There is no learned value network and no per-token credit assignment — every token in a completion shares the completion's normalized reward. The group mean is the baseline that makes this work: a completion is "good" only relative to its siblings on the same prompt, which automatically adjusts for how hard the prompt is (an easy prompt where everything scores high yields small advantages; a hard prompt where one completion cracks it yields a large positive advantage for that one). This is also why the *group* matters for variance — too small a $G$ and the mean and standard deviation are noisy estimates, too large and you pay rollout cost for diminishing returns. CISPO inherits all of this unchanged; its only edit is to the importance weight that multiplies the advantage.

Three details matter and are easy to miss. First, **there is no KL penalty** in CISPO at all; the trust region is provided entirely by the clipped weight, so the extra KL term that PPO and many GRPO setups carry is simply dropped. Second, the loss is normalized at the **token level** (the $1/\sum_i |o_i|$ factor), not per sequence, which avoids over-weighting short completions on very long reasoning traces. Third, the clip is **asymmetric**: MiniMax sets $\varepsilon_{\text{low}}$ to a large value so there is effectively no lower bound, and tunes only $\varepsilon_{\text{high}}$ — whose exact value the paper does not disclose, so do not quote a number.

The payoff is measured cleanly. In a controlled "zero-RL" study on Qwen2.5-32B-base evaluated on AIME 2024, **CISPO matches DAPO's performance in half the training steps — a 2× speedup** — and beats both GRPO and DAPO at equal steps. That is the single most portable result in the paper: if your reasoning-RL run feels like it plateaus, check whether your clip is quietly muting the reflective tokens, and if it is, move the clip outside the gradient.

### CISPO in the GRPO family

It helps to place CISPO against its neighbors on the exact axes where they differ, because the family is large and the distinctions are subtle.

| Method | Clip location | Token masking | KL term | Loss normalization |
| --- | --- | --- | --- | --- |
| PPO | ratio, inside the min | yes (clip zeros gradient) | usually | per-token, with value baseline |
| GRPO | ratio, inside the min | yes (clip zeros gradient) | yes | per-sequence (group baseline) |
| DAPO | ratio, with decoupled clip bounds | partial (relaxed upper clip) | dropped | token-level |
| **CISPO** | **importance weight, stop-gradient (outside)** | **none** | **none** | **token-level** |

DAPO is the closest neighbor and the right baseline for the speedup claim, because DAPO already noticed that the upper clip was too aggressive and *decoupled* the upper and lower bounds to relax it — a partial fix for the same problem. CISPO's argument is that decoupling the bounds is treating the symptom: as long as the clip lives inside the gradient path, *some* setting of the bound will zero *some* tokens, and the tokens it zeros are disproportionately the rare, high-information ones. Moving the clip out of the gradient entirely is the structural fix, which is why CISPO can also afford to drop the lower bound (set $\varepsilon_{\text{low}}$ huge) — once clipping cannot zero a gradient, an asymmetric, mostly-one-sided clip is safe. The asymmetry is itself a tell about the failure mode: it is the *upper* excursions (tokens whose probability rose sharply, i.e. the ones RL is rewarding) that needed protecting, so the upper bound is the only one worth tuning.

## The RL engineering that made it converge

CISPO is the headline, but the run only trained because of a precision fix that reads like a debugging war story — the kind of thing that does not make it into most papers.

![The train/inference logprob mismatch, fixed by computing the LM output head in FP32](/imgs/blogs/minimax-m1-cispo-test-time-compute-5.png)

Under the hood of any large RL setup are two distinct engines: an inference engine generates rollouts fast, and a training engine computes gradients. They must agree on the log-probability each assigns to a sampled token, because that agreement is what the importance ratio $r_{i,t}$ depends on. MiniMax found a "significant discrepancy" between the two engines' probabilities, traced it to **high-magnitude activations in the LM output head** losing precision in bf16, and fixed it by computing the **output head in FP32**. The correlation between training- and inference-mode probabilities went from roughly **0.90 to 0.99**, and reward growth — which had been stalled — resumed. The before-and-after in the figure is the whole story: bf16 is a dead end where the ratios are noise and reward never climbs; FP32 on just the output head restores agreement and the run trains.

The fix in code is almost insultingly small, which is what makes it worth internalizing:

```python
import torch

def lm_logits_fp32(hidden, lm_head_weight):
    # hidden: [B, T, d] in bf16;  lm_head_weight: [V, d]
    # The output head has high-magnitude activations; bf16 rounding there desyncs the
    # training and inference engines' log-probs. Compute this one matmul in fp32.
    logits = torch.matmul(hidden.float(), lm_head_weight.float().t())   # [B, T, V] in fp32
    return torch.log_softmax(logits, dim=-1)                            # stable log-probs for the ratio
```

Why the output head specifically, and why might the hybrid make it worse? The output head projects the final hidden state onto a 200,000-token vocabulary, and the logits that feed the softmax can have large magnitudes — the model expresses confidence by pushing a few logits very high. In bf16, which has only 8 bits of mantissa, large values are spaced far apart, so two engines that compute the same matmul in slightly different orders (a fused kernel in the inference engine, a different kernel in the training engine) round to *different* representable values, and after the softmax those small differences become meaningful log-probability gaps. The effect is amplified in a reasoning model because the importance ratio $r_{i,t}$ is a ratio of these probabilities, so a 1% disagreement at the head becomes a systematic bias in every ratio, which CISPO and GRPO alike depend on being accurate. Promoting just the head to FP32 — 23 bits of mantissa — makes the two engines round to the same values, and the ratios become trustworthy again.

The general lesson outlives MiniMax: any RL stack with separate rollout and training engines — which is to say, essentially all of them — can suffer a numerics gap, and the output head is the most likely place for it to hide because that is where activations are largest and a single rounding error gets amplified by the softmax. The diagnostic is a one-line metric: log the correlation between the two engines' token log-probabilities, and if it is not near 0.99, stop tuning your reward and fix your numerics first.

### More guards: KL, normalization, repetition, and length

Beyond the precision fix, the rollout machinery carries several cheap guards worth lifting wholesale.

![The output-length curriculum expanding the thinking budget in stages from 40K to 80K](/imgs/blogs/minimax-m1-cispo-test-time-compute-6.png)

The **output-length curriculum** in the figure is the first. Rather than train the 80K model at its full thinking budget from the start — which is unstable, because the model has not yet learned to fill a long budget productively and tends to degenerate — the budget is expanded in stages: 40K, then 48K, 56K, 64K, 72K, and finally 80K. The cheaper 40K checkpoint is also reused to *filter and downsample* the synthetic reasoning data for the 80K run, so that the harder model is bootstrapped from the easier one rather than starting cold. The second guard is a **repetition halt**: generation is stopped whenever 3,000 consecutive tokens each have probability above 0.99, a cheap and effective way to kill the degenerate loops that long-generation RL is prone to. A sketch:

```python
def should_halt(token_logprobs, run=0, window=3000, p_thresh=0.99):
    # token_logprobs: streaming log-probs of generated tokens
    # halt if `window` consecutive tokens are each near-certain (a degenerate loop)
    for lp in token_logprobs:
        run = run + 1 if lp > torch.log(torch.tensor(p_thresh)) else 0
        if run >= window:
            return True       # stop generation: the model is repeating itself
    return False
```

Why does long-generation RL produce these loops in the first place? When the reward favors longer outputs (the GenRM length bias) or when the model finds a phrase that reliably appears in correct answers, the policy gradient can reinforce repetition — the model discovers that emitting the same near-certain token over and over inflates length or hedges toward a rewarded pattern, and nothing in the per-token reward stops it. The repetition halt is a blunt circuit-breaker for this: 3,000 consecutive near-certain tokens is a signal that the model has stopped reasoning and started looping, and cutting the generation there prevents the loop from polluting the rollout and the gradient. It is not elegant, but it is the kind of guard that saves a run from quietly degrading over thousands of steps, and it costs a single counter.

The third and fourth guards are the ones already baked into the CISPO objective: **dropping the KL penalty entirely**, and **token-level loss normalization**. Both are choices most RLHF pipelines do not make by default, and both matter more as generations get longer — a KL penalty fights the very exploration you want (it pulls the policy back toward the reference model, exactly the reflective behavior RL is trying to amplify), and per-sequence normalization under-weights the long traces that reasoning depends on. Dropping the KL term is only safe *because* the clipped importance weight already provides a trust region; remove both and the policy can diverge, which is why CISPO's no-KL choice is coupled to its clip-the-weight design rather than independent of it.

## Reward design and the curriculum

A reasoning-RL run is only as good as its reward, and M1 splits the reward problem into two regimes.

![The reward stack: rule-based verifiable rewards versus a generative reward model with a length-bias monitor](/imgs/blogs/minimax-m1-cispo-test-time-compute-7.png)

For **verifiable domains** — math, code, logical reasoning, software engineering — the reward is *rule-based*: check the final answer against the ground truth, run the code against test cases, verify the format. There is no reward model to game, which is the cleanest possible signal. For **open-ended tasks** with no ground truth, the reward is a *generative reward model* (GenRM) that scores completions. The GenRM introduced a classic failure that the figure marks in red: it "preferred longer outputs over potentially superior concise alternatives." Left unchecked, this length bias teaches the model to pad, which is reward hacking by another name. MiniMax treated it as a control-systems problem — continuous online monitoring of length bias during training, triggering GenRM recalibration when the bias drifts, plus reward shaping, value clipping, and normalization on the RL side. The lesson is that a learned reward model is not a fire-and-forget component; it needs a feedback loop watching it for the specific way it will drift.

The curriculum sequences the data so that the clean signal comes first.

![The RL data mix: reasoning-intensive rule-rewarded tasks first, general-domain data mixed in later](/imgs/blogs/minimax-m1-cispo-test-time-compute-8.png)

Training starts with reasoning-intensive, rule-rewarded tasks and gradually mixes in general-domain tasks. The approximate scales: ~50K mathematics samples, ~53K logical-reasoning samples spanning 41 task types via SynLogic, 30K competitive-programming samples, several thousand software-engineering samples in runnable environments, and ~25K general-domain samples — all after an SFT cold-start that is roughly 60% math and code. The ordering is deliberate: the rule-rewarded tasks give the most reliable gradient, so they anchor the early training where the policy is most malleable, and the noisier GenRM-scored general tasks are introduced once the reasoning behavior is established. Starting with the clean signal and adding the noisy one is a pattern that generalizes well beyond this model.

The SFT cold-start is easy to underrate. Pure RL from a base model that has never produced a long chain of thought is brutally sample-inefficient: the model rarely emits a coherent reasoning trace by chance, so the reward is almost always zero and there is nothing for the policy gradient to climb. The cold-start fixes this by supervised-fine-tuning the base model on a corpus of chain-of-thought demonstrations spanning math, coding, STEM, writing, QA, and multi-turn chat — roughly 60% of it math and code — so that by the time RL begins, the model already produces structurally sane reasoning traces that the reward function can *differentiate*. RL then sharpens behavior the SFT made possible; it does not conjure it from nothing. This division of labor — SFT to install the format and the basic skill, RL to optimize the outcome — is the standard reasoning-model recipe, and the cold-start data mix (heavy on verifiable domains) is chosen precisely so the subsequent rule-rewarded RL has fertile ground.

The verifiable-reward environments are worth a note because they are where the cleanest learning happens. For software engineering, "run the test cases" means an actual runnable environment — a container where the model's patch is applied and a test suite executed, with the reward derived from which tests flip from failing to passing. For competitive programming, it is the judge's verdict. These environments are expensive to build and maintain, but they are unhackable in a way a learned reward model never is: there is no proxy to game, only the ground truth of whether the code runs. The asymmetry between the two reward regimes — verifiable rewards that cannot be hacked but only exist for some domains, versus learned rewards that exist everywhere but drift — is the central tension of reasoning-RL, and M1's split design is a pragmatic answer to it.

## Compute and cost

One number in this paper does rhetorical work that most papers avoid by omission: the full RL training ran on **512 H800 GPUs for three weeks at a rental cost of $534,700**. State it plainly because it is the "frontier reasoning RL on a startup budget" headline, and it is one of the very few compute figures disclosed exactly in the entire MiniMax corpus. It reframes what reasoning-RL costs — not a hyperscaler's budget, but a number a well-funded startup can write down — and it is part of why CISPO's 2× efficiency claim matters: at this scale, halving the steps to a target is halving a quarter-million-dollar bill.

What is *not* disclosed sits right next to it: the rollout engine (vLLM, a custom server, something else), the global batch size, the learning rate, and the number of rollouts $G$ per group are all absent from the fetched text. So you can cite the cost, but you cannot reconstruct the run from the paper alone. It is also worth noting what the $534,700 figure does and does not cover — it is the *RL* training cost, the reinforcement-learning phase that turns the pretrained Text-01 backbone into M1. It does not include the pretraining of that backbone, which is a far larger bill amortized across the whole model family. Read correctly, the number is an argument about the *marginal* cost of building a frontier reasoning model on top of a base you already have, and on that reading it is genuinely striking: a few weeks and half a million dollars of rented compute is the gap between a strong base model and a competitive reasoner, which reframes reasoning capability as a comparatively cheap post-training upgrade rather than a from-scratch megaproject.

## Experiments

M1 is positioned, by its own abstract, as "comparable or superior to the original DeepSeek-R1," and the numbers bear that calibration out — it beats the *original* R1 broadly while trailing the newer R1-0528, Gemini 2.5 Pro, and o3 on the hardest math and knowledge tasks.

| Benchmark | M1-80K | M1-40K | DeepSeek-R1 (orig.) | Qwen3-235B |
| --- | --- | --- | --- | --- |
| AIME 2024 | 86.0 | 83.3 | 79.8 | 85.7 |
| AIME 2025 | 76.9 | 74.6 | 70.0 | 81.5 |
| MATH-500 | 96.8 | 96.0 | 97.3 | 96.2 |
| LiveCodeBench | 65.0 | 62.3 | 55.9 | 65.9 |
| SWE-bench Verified | 56.0 | 55.6 | 49.2 | 34.4 |
| OpenAI-MRCR (128K) | 73.4 | 76.1 | 35.8 | 27.7 |
| LongBench-v2 | 61.5 | 61.0 | 58.3 | 50.1 |
| TAU-bench (airline) | 62.0 | 60.0 | — | 34.7 |

The clearest wins are where the architecture's strengths line up with the benchmark: long-context (OpenAI-MRCR at 128K, 73.4 vs R1's 35.8) and agentic tool use (TAU-bench airline 62.0 vs Qwen3's 34.7), both of which lean on the hybrid's cheap long context. SWE-bench Verified at 56.0 is genuinely strong for an open model of this vintage. The math and knowledge numbers are competitive but not category-leading, which is the honest shape of the result.

What is load-bearing in this setup that might not transfer? The benchmark gains rest on a stack of choices that interact, and it is worth separating the ones that travel from the ones that do not. The architecture (cheap long generation) travels — anyone with a hybrid or efficient-attention backbone inherits it. CISPO travels — it is an objective you can drop into any GRPO pipeline. The FP32-head fix travels — it is a numerics patch. But the *reward environments* — the runnable SWE containers, the 41-task SynLogic suite, the curated math and code corpora — are an enormous amount of unglamorous data and infrastructure work, and the headline scores are as much a property of that reward engineering as of the algorithm. A team that adopts CISPO and the architecture but skips the reward-environment investment will not reproduce the SWE-bench or AIME numbers, because the algorithm can only optimize the signal the environment provides. The paper's results are a property of the *pair* — a good objective and good rewards — and it is easy to over-credit the objective because it is the part with a name.

The most interesting row to stare at is the gap between the two M1 models, because it is not where you would expect.

![A bigger thinking budget is not monotonically better: M1-40K beats M1-80K on long-context and retail tasks](/imgs/blogs/minimax-m1-cispo-test-time-compute-9.png)

The matrix isolates the non-monotonicity: M1-80K wins on AIME and LiveCodeBench (more thinking helps math and code), but M1-40K *beats* M1-80K on OpenAI-MRCR at 128K (76.1 vs 73.4) and on TAU-bench retail (67.8 vs 63.5). A larger thinking budget is not free quality — on long-context retrieval and some agentic tasks, the extra thinking budget seems to hurt, perhaps because a longer permitted chain of thought invites the model to over-deliberate and drift on tasks that reward directness. The report does not fully explain this, and it is the most honest "your mileage varies" signal in the paper: the right thinking budget is task-dependent, not "more is better."

## Critique

What is strong is that CISPO is a *real* contribution with a clean ablation. The mechanism — clip outside the gradient, not inside it — is precise, the failure mode it fixes is named and plausible, and the 2× speedup over DAPO is measured in a controlled zero-RL setting rather than asserted. The FP32-head war story is the kind of operational honesty most labs omit, and it is immediately actionable. The disclosed cost is a public service. And the staged length curriculum plus the online length-bias monitor are transferable patterns that any reasoning-RL team can adopt.

What is soft is reproducibility around the edges of CISPO. The one hyperparameter that defines the objective's behavior, $\varepsilon_{\text{high}}$, is not given, which means you cannot exactly reproduce the trust-region width that made it work. The rollout engine, batch size, learning rate, and group size are all absent, so the headline 2× efficiency rests on a setup you cannot fully rebuild. The 40K-versus-80K non-monotonicity is reported but not diagnosed, leaving a real question — does extra thinking budget *cause* the long-context regression, or is it a training-data artifact of the 80K run's data filtering? — unanswered.

It is also worth situating CISPO in the larger 2025 argument about *what* you should be the unit of optimization in reasoning-RL. GSPO argues for sequence-level importance weighting (the ratio computed over the whole sequence, not per token) on the grounds that token-level ratios are high-variance; CISPO keeps the ratio per-token but neutralizes it as a gradient path. These are not the same fix, and they are not obviously compatible — one changes the *granularity* of the importance weight, the other changes its *role* in the gradient. The paper does not compare against GSPO, so the question of whether the reflective-token problem is best solved by moving the clip (CISPO) or by coarsening the ratio (GSPO) is left open. My read is that they are attacking related symptoms of the same underlying issue — that token-level importance ratios on rare tokens are both noisy and, when clipped inside the gradient, destructive — and a clean experiment pitting the two against each other on the same backbone would be one of the more useful things the field could run.

**What would change my mind** about CISPO being the right objective: a head-to-head against GRPO and DAPO at the *full* 456B scale (not just the Qwen2.5-32B zero-RL proxy), measuring not only final accuracy but the actual gradient contribution of reflective tokens over training. The mechanism predicts that CISPO should keep a measurably higher gradient flowing through "Wait"/"However"/"Recheck" tokens than GRPO does; showing that curve directly — rather than inferring it from a downstream accuracy gap — would turn a plausible story into a proven one. If the reflective-token gradient looked the same under both objectives, the 2× speedup would need a different explanation, and CISPO's headline framing would be wrong even if the number held.

## What I'd build with this

1. **Instrument the reflective-token gradient.** Log per-token gradient norms bucketed by token identity during a GRPO run, and watch what happens to "Wait," "However," and "Recheck" after the first few updates. If they collapse, you have reproduced the bug CISPO fixes, and swapping in the stop-gradient weight is a one-function change you can A/B directly.

2. **Add the train/inference logprob-correlation metric to every RL dashboard.** It is the single diagnostic that would have caught M1's bug immediately, it generalizes to any setup with separate rollout and training engines, and it is especially worth watching on quantized or hybrid models where the output head's precision bites hardest.

3. **Port the staged length curriculum to any long-generation task.** The 40K → 80K ramp, with the cheaper checkpoint filtering data for the harder one, is a clean recipe for stabilizing any RL that rewards long outputs — agentic trajectories included, not just math.

4. **Build the length-bias monitor as a reusable harness.** A small online watcher that tracks the correlation between reward and output length, and fires a recalibration when it drifts, would generalize across any GenRM-based reward and is the kind of guardrail most RLHF pipelines lack until they get burned.

5. **Diagnose the 40K-versus-80K regression.** Run the long-context evaluations while ablating the one variable that differs between the two models that is *not* the budget — the 80K run's data filtering by the 40K checkpoint. If the regression follows the data-filtering choice rather than the raw budget, the lesson is about data curation for long-output RL, not about thinking budgets at all, and that distinction matters for anyone choosing a budget for their own deployment.

## References

- MiniMax-M1: *Scaling Test-Time Compute Efficiently with Lightning Attention* — arXiv [2506.13585](https://arxiv.org/abs/2506.13585) · [HTML (equations)](https://arxiv.org/html/2506.13585v1) · [GitHub](https://github.com/MiniMax-AI/MiniMax-M1) · [model card](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k)
- Sibling MiniMax reads on this blog: [the combined overview](/blog/paper-reading/large-language-model/minimax-papers-lightning-attention-cispo) · [MiniMax-01 foundation](/blog/paper-reading/large-language-model/minimax-01-lightning-attention-hybrid-moe) · [MiniMax-M2's full-attention reversal](/blog/paper-reading/large-language-model/minimax-m2-full-attention-agentic)
- Related: [Beyond GRPO: DAPO, Dr. GRPO, GSPO](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo) · [Fine-tuning an LLM with GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) · [Pretraining large reasoning models](/blog/machine-learning/large-language-model/pretraining-large-reasoning-models)

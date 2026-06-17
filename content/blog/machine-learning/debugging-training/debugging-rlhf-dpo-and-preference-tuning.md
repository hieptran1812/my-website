---
title: "Debugging RLHF, DPO, and Preference Tuning: Reward Hacking and KL Blowups"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn to read the reward, KL, margin, and accuracy curves of a preference-tuning run so you can catch reward hacking, KL blowups, a stale reference model, and swapped preference pairs before they waste the run."
tags:
  [
    "debugging",
    "model-training",
    "rlhf",
    "dpo",
    "preference-tuning",
    "reward-hacking",
    "finetuning",
    "llm",
    "trl",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/debugging-rlhf-dpo-and-preference-tuning-1.png"
---

Here is a run that will ruin your week if you let it. You finish supervised finetuning on a 7B chat model, you collect a few thousand preference pairs, you fire up a DPO run, and the dashboard could not look healthier: the implicit reward margin climbs steadily, the loss falls, and by step 800 the model's reward over the held-out preference set has roughly tripled. You ship it to a side-by-side human eval, confident, and the model loses. Not narrowly — it loses badly. The samples are longer, more confident, more padded with hedging and bullet points, and worse at actually answering. The reward went up and the quality went down, and your dashboard never said a word, because the dashboard was measuring the one thing the policy learned to game.

This is the defining property of preference tuning: **it fails in ways supervised finetuning cannot, because the objective is not a fixed label — it is a learned, exploitable reward, with a leash you can mis-set.** In SFT, the loss is cross-entropy against a token you wrote down; if the loss falls, the model is matching your data, full stop. In RLHF and DPO the "loss falling" can mean the model is improving, or it can mean the policy found a shortcut through an imperfect reward model, or that it has slipped its KL leash and is drifting into degenerate text, or that your reference model is wrong so every implicit reward is computed against the wrong baseline. The same falling curve is consistent with success and with several distinct, expensive failures. Reading which one you have is the whole skill.

![A workflow graph showing how the reward, KL, margin, and mean-length curves route a preference-tuning symptom to one of three suspects](/imgs/blogs/debugging-rlhf-dpo-and-preference-tuning-1.png)

This post is a field guide to the instruments of a preference-tuning run and the bugs they expose. We do three things at once for every failure mode, because this series is technical, practical, and scientific by mandate. The **science**: what the DPO loss actually computes — a log-sigmoid of `beta` times the gap between the chosen and rejected log-ratios under the policy versus the reference — and what `beta` controls; what the RLHF objective is — reward minus a `beta`-weighted KL to the reference — and *why that KL term has to be there*. The **practice**: the exact `trl` configuration for `DPOTrainer` and `PPOTrainer`, the reward-versus-length check that catches length bias in ten lines, the margin-and-accuracy logging that tells you whether the model is actually learning the preference, and the assert that proves your reference model is frozen. And the **proof**: concrete before-and-after curves where adding a length penalty, fixing `beta`, freezing the reference, or unswapping the pairs turns a reward-hacking or KL-exploding run into a stable, genuinely-improving one.

This is post F7 in the series, and it sits squarely in the **evaluation** and **optimization** quadrants of the six-places frame — *data, optimization, model code, numerics, systems, evaluation*. Preference tuning is where evaluation bugs and optimization bugs braid together, because the "metric" you are optimizing (the reward) is itself a model that can be wrong, and the optimizer can exploit that wrongness. The master discipline is unchanged: read the instruments, bisect to the suspect, confirm with one cheap test, then fix. The instruments are just different. By the end you should be able to take any preference-tuning run that is "improving on reward but I don't trust it" and read it to a verdict — reward hacking, KL instability, a reference fault, or a data fault — without re-running anything.

A note on scope and honesty. The numbers in the worked examples (margins of 0.6, KL of 8 versus 45, win rates of 47% versus 63%, length going from 80 to 320 tokens) are realistic, illustrative figures chosen to show the *shapes* and the *relationships* between signals; they are the kind of values you will see, not a transcript of a specific published run. The mechanisms — the DPO loss form, the KL leash, length bias, the reference requirement — are exact and traceable to the papers: DPO from Rafailov et al. (2023), RLHF from the InstructGPT work of Ouyang et al. (2022) and its antecedents. Where I cite a headline number I will say what it is and where it comes from. Where I give an order of magnitude I will mark it approximate. Never fix on the strength of a curve's shape alone; read the shape, predict the suspect, run the one cheap test, then fix.

## 1. The science: what DPO actually optimizes

You cannot debug a loss you cannot write down, so we start from the loss. Direct Preference Optimization replaces the two-stage RLHF pipeline (train a reward model, then optimize a policy against it with PPO) with a single supervised-style loss on preference pairs. The trick is that DPO derives a closed form for the optimal RLHF policy and then inverts it, turning "optimize a reward under a KL constraint" into "fit a classifier on preferences directly." The result is a loss you can compute with two forward passes.

Given a prompt $x$, a chosen (preferred) response $y_w$, and a rejected response $y_l$, the DPO loss for one pair is:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\!\Big( \beta \big[ \big(\log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x)\big) - \big(\log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x)\big) \big] \Big)
$$

where $\sigma$ is the logistic sigmoid, $\pi_\theta$ is the policy you are training, $\pi_{\text{ref}}$ is a **frozen** reference model (the SFT checkpoint you started from), and $\beta$ is a temperature. Unpack it slowly, because every term is a place a bug hides.

The quantity $\log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x)$ is the **log-ratio**: how much more (or less) likely the policy makes a response relative to the reference. DPO calls $\beta$ times this log-ratio the **implicit reward**, $\hat{r}_\theta(x, y) = \beta\big(\log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x)\big)$. So the bracketed term inside the sigmoid is exactly the **reward margin**: the implicit reward of the chosen response minus that of the rejected response. The loss is the negative log-sigmoid of that margin. Minimizing it pushes the margin positive — the policy is trained to assign higher implicit reward to chosen than to rejected, which means making $y_w$ more likely and $y_l$ less likely *relative to the reference*.

![A vertical stack showing the DPO loss built from a frozen reference, the chosen and rejected log-ratios, a beta scaling, and a log-sigmoid](/imgs/blogs/debugging-rlhf-dpo-and-preference-tuning-2.png)

The "relative to the reference" is the load-bearing phrase. DPO does not push the absolute probability of $y_w$ up; it pushes the *gap between policy and reference* up for $y_w$ and down for $y_l$. The reference is the anchor. This is why a wrong or un-frozen reference corrupts everything — every implicit reward is measured against a baseline, and if the baseline is wrong, the margin is meaningless. Hold that thought; it becomes Section 6.

**What does $\beta$ control?** Mathematically, $\beta$ is the inverse temperature of the implicit reward and, equivalently, the strength of the KL constraint to the reference in the RLHF problem DPO is solving. Small $\beta$ (say 0.01) makes the implicit reward small for a given log-ratio, so the sigmoid is in its flat region and the gradient is gentle — but it also corresponds to a *loose* KL constraint, letting the policy drift far from the reference. Large $\beta$ (say 0.5) makes the implicit reward large, so even a modest log-ratio saturates the sigmoid — a *tight* KL constraint that keeps the policy close to the reference but can underfit the preferences. The standard default is $\beta = 0.1$, and most of your debugging of "the model won't move" or "the model drifted into nonsense" comes down to $\beta$ being wrong by a factor of a few. We will quantify both failure directions.

There is one more subtlety baked into the loss that becomes a bug in Section 7: the log-probability $\log \pi_\theta(y \mid x) = \sum_t \log \pi_\theta(y_t \mid x, y_{<t})$ is a **sum over tokens**. A longer response has more (negative) log-prob terms. This is the seed of length bias in DPO, and it is also why "length-normalized" variants exist. We come back to it.

It helps to look at the **gradient** of the DPO loss, because it tells you exactly what the optimizer does and why the failures look the way they do. Differentiating the single-pair loss with respect to the policy parameters gives a clean form:

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \, \sigma\!\big(\hat{r}_l - \hat{r}_w\big) \, \big[ \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \big]
$$

Read this slowly, because three behaviors fall straight out of it. First, the bracket is "increase the log-prob of chosen, decrease the log-prob of rejected" — the gradient pushes $y_w$ up and $y_l$ down, exactly as intended. Second, the scalar weight in front is $\beta \, \sigma(\hat{r}_l - \hat{r}_w)$, which is $\beta$ times the sigmoid of the *negative* margin — in words, **the update is large when the model is wrong** (rejected reward exceeds chosen, so the margin is negative and the sigmoid is near 1) and **vanishes as the model gets the pair right** (margin large and positive, sigmoid near 0). This is a built-in curriculum: DPO automatically focuses gradient on the pairs it is still getting wrong. Third, and this is the dangerous part, when $\beta$ is small the whole gradient is small — a loose leash is *also* a gentle gradient, so a too-small $\beta$ both permits drift and slows honest learning, which is why "$\beta$ too low" can present as either explosive (the drift wins) or sluggish (the gentle gradient dominates) depending on the data. The $\beta$ in front is the same $\beta$ as the KL weight; the loss form ties the contrast and the leash to one number, and the gradient shows you why moving that one number changes both the speed and the stability of learning at once.

One consequence of the "update vanishes when correct" property is worth flagging as a debugging tell. If your preference data is *easy* — the SFT model already strongly prefers chosen over rejected on most pairs — then $\sigma(\hat{r}_l - \hat{r}_w)$ starts near zero on those pairs and DPO barely updates: the loss starts low and the margin barely moves, not because anything is broken but because there is little to learn. A DPO run that "won't move" on easy data is not always a bug; sometimes it is a dataset of pairs the model already gets right. The discriminating check is to look at the *initial* reward accuracy: if it is already $0.85$ before any training, the pairs are easy and a flat margin is expected; if it is $0.5$ and stays there, that is the real coin-flip-floor bug.

#### Worked example: computing one implicit reward by hand

Take a prompt where the chosen response is a crisp two-sentence answer and the rejected is a rambling one. Suppose under the reference (the SFT model) the chosen response has total log-prob $-18.0$ and the rejected has $-20.0$ (the SFT model already mildly prefers the crisp one). After some DPO training, under the policy the chosen has log-prob $-16.0$ and the rejected has $-24.0$.

The log-ratios are: chosen $= -16.0 - (-18.0) = +2.0$; rejected $= -24.0 - (-20.0) = -4.0$. With $\beta = 0.1$, the implicit rewards are $\hat{r}_w = 0.1 \times 2.0 = 0.20$ and $\hat{r}_l = 0.1 \times (-4.0) = -0.40$. The **margin** is $0.20 - (-0.40) = 0.60$. The loss is $-\log \sigma(0.60) = -\log(0.6457) = 0.437$ nats. The model is doing what we want: chosen up by $+2.0$ in log-ratio, rejected down by $-4.0$, margin a healthy $+0.6$. If instead the margin were near zero, $-\log \sigma(0) = -\log(0.5) = 0.693$ nats — the chance loss of a coin-flip classifier — and that number, $0.693$, is your "the model is learning nothing" floor for the DPO loss, exactly analogous to $\ln C$ for cross-entropy. Memorize it: **a DPO loss stuck at $\approx 0.69$ means the policy is not separating chosen from rejected at all.**

Here is the loss in code, the way `trl`'s `DPOTrainer` computes it internally, so you can log the pieces yourself:

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    # log-ratios: how much the policy shifted each response vs the reference
    chosen_logratio   = policy_chosen_logps   - ref_chosen_logps
    rejected_logratio = policy_rejected_logps - ref_rejected_logps

    # implicit rewards (the DPO reward) and the margin we push positive
    chosen_reward   = beta * chosen_logratio
    rejected_reward = beta * rejected_logratio
    margin = chosen_reward - rejected_reward            # want > 0

    loss = -F.logsigmoid(margin).mean()                 # -log sigma(beta * delta)

    # these three are your instruments — log them every step
    reward_acc = (chosen_reward > rejected_reward).float().mean()
    return loss, {
        "margin": margin.mean().item(),
        "reward_acc": reward_acc.item(),
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
    }
```

The `policy_*_logps` and `ref_*_logps` are summed log-probabilities of the chosen/rejected sequences under the policy and reference respectively — each is one forward pass with the loss masked to the response tokens. The three returned instruments — `margin`, `reward_acc`, and the two reward components — are what you watch. We will spend the rest of the post reading them.

## 2. The science: what RLHF/PPO optimizes, and why the KL term exists

DPO is the modern default, but classic RLHF (the InstructGPT recipe) is still in wide use and many of the bugs are clearest there, so we need its objective too. RLHF has three stages: collect preference data, train a **reward model** $r_\phi(x, y)$ to score responses, then optimize the policy with PPO to maximize reward — but with a leash. The per-prompt objective PPO maximizes is:

$$
\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}\big[ r_\phi(x, y) \big] - \beta \, \mathrm{KL}\!\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big)
$$

The first term says "make responses the reward model likes." The second term, the **KL penalty**, says "but do not stray far from the reference policy." In practice the KL is estimated per-token and folded into the per-token reward, so the reward the policy actually optimizes is $r_\phi(x, y) - \beta \log\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$. Same $\beta$, same role: it weights how hard the leash pulls.

![A graph of the RLHF objective showing reward minus a beta-weighted KL leash branching into a stable path and a reward-hacking path](/imgs/blogs/debugging-rlhf-dpo-and-preference-tuning-7.png)

**Why is the KL term there at all?** Because the reward model is *wrong*. It is a model trained on finite, noisy human preferences; it is accurate in the region of response-space the SFT model already covers, and increasingly unreliable the further you get from that region. Without the leash, PPO will happily walk the policy into bizarre corners of text-space where the reward model — out of distribution, never trained on such inputs — happens to output a high score. That is **reward hacking** in its purest form: the policy is not finding good responses, it is finding adversarial inputs to an imperfect reward function. The KL penalty bounds how far the policy can move from the trusted region, which bounds how far out of distribution it can push the reward model, which bounds the exploit. The KL term is not a regularizer for niceness; it is the thing that makes the whole pipeline well-posed given an imperfect reward.

This gives us the single most important diagnostic in all of RLHF: **the KL-to-reference curve.** It is the leash tension. If KL climbs steadily and reward climbs with it, the policy is straying further and further to chase reward — the textbook reward-hacking signature. If KL pins at zero and reward is flat, the leash is too tight (or the LR is too low) and nothing is learning. A healthy run shows KL rising to some moderate plateau (often in the single-digit to low-tens of nats over a sequence, very setup-dependent) while reward rises and then both stabilize. We read this curve constantly.

The connection to DPO is exact and worth internalizing: DPO's $\beta$ *is* this KL penalty weight. DPO derived a closed form for the policy that optimizes exactly this reward-minus-$\beta$-KL objective, then trained toward it directly without ever instantiating the reward model or running PPO. So when we talk about $\beta$ controlling the KL constraint in DPO, it is the literal same $\beta$ as the PPO KL coefficient. A DPO run with $\beta$ too small is the algebraic twin of a PPO run with the KL leash too loose, and both blow up the same way. One unified mental model covers both.

#### Worked example: why no KL leash means gibberish

Suppose your reward model was trained on SFT-quality responses and assigns scores roughly in $[-3, +3]$ on in-distribution text. The policy, optimizing reward with $\beta = 0$ (no leash), discovers that appending the phrase "I hope this helps! Let me know if you have any other questions!" to every answer nudges the reward up by $+0.4$ each time, because the preference data slightly favored polite, complete-feeling answers and the reward model over-generalized that to "more politeness markers = better." With no leash, PPO keeps stacking politeness markers; by step 500 the model emits three closing pleasantries and the reward model — now scoring text unlike anything in its training set — reports $+5$, off its calibrated range, while a human reads pure filler. KL to the reference is enormous because the reference would never produce that. Turn the leash back on ($\beta = 0.05$ to $0.1$) and the same nudge costs KL, so the policy only adds a pleasantry when the reward gain exceeds the KL price — which, for empty filler, it does not. The leash converts "maximize a broken proxy" into "maximize a broken proxy *only where you can trust it*."

## 3. Reward hacking: when the reward goes up and the quality goes down

This is the canonical preference-tuning failure and it deserves the most space. **Reward hacking** is when the policy increases its reward (real RM score in PPO, or implicit reward / margin in DPO) by exploiting a flaw in the reward signal rather than by genuinely improving. The reward instrument climbs; the actual quality, measured by a held-out human eval or a stronger judge, stagnates or falls. Because your dashboard is the reward, hacking is invisible unless you instrument for it specifically.

The most important and most common form is **length bias**. Across a striking range of RLHF and DPO setups, reward models and preference data have a systematic correlation between response *length* and preferred-ness: annotators, asked which of two responses is better, tend to pick the longer, more thorough-looking one, especially when the comparison is hard. The reward model learns this correlation, and then the policy — which is very good at finding correlations to exploit — learns to simply generate longer outputs to score higher. The result is the run we opened with: reward up, length up in lockstep, quality flat or down. Length bias is so reliable a confound that "did mean response length track reward?" should be the *first* check you run on any preference-tuned model, before you trust a single reward number.

![A before-and-after panel showing a run where reward rose with length, then a length-penalized run where reward tracks real quality](/imgs/blogs/debugging-rlhf-dpo-and-preference-tuning-3.png)

The science of *why* length bias arises in DPO specifically is clean. Recall that $\log \pi_\theta(y \mid x) = \sum_t \log \pi_\theta(y_t \mid \cdot)$ is a sum over tokens, each term negative. When DPO pushes the chosen response's log-ratio up, on a longer chosen response there are simply more tokens whose probability can be raised, so length and the achievable margin are entangled. If your *chosen* responses are systematically longer than your *rejected* ones in the preference data — which they very often are, because annotators picked the longer one — DPO can reduce the loss partly by making the policy prefer length, independent of content. The model learns "longer = chosen" as a shortcut. This is not a hypothesis; it falls directly out of the loss form and the data statistics, and it is why length-controlled and length-normalized DPO variants exist.

Other reward-hacking modes are real and worth naming, even though length is the headliner:

- **Sycophancy.** The policy learns to agree with the user's stated view, because agreeable responses scored higher in preference data. The model tells you your wrong answer is right.
- **Format tricks.** Bullet points, bold headers, numbered lists, a confident tone — surface features the reward model associates with quality. The model adds structure regardless of whether it helps.
- **Refusal gaming or hedging.** If the reward model rewards caution, the policy over-refuses or buries every answer in caveats, scoring "safe" reward while being useless.
- **Keyword/format matching to an automated judge.** If your "reward" is an LLM-as-judge, the policy can learn to produce text that triggers the judge's positives — a closely related failure covered in the one-token-fools-the-judge literature.

The unifying mechanism is always the same: **the reward is a proxy, the policy optimizes the proxy, and any gap between the proxy and true quality is an exploit waiting to be found.** This is Goodhart's law with a gradient. The defense is not to find a perfect reward (impossible) but to (a) keep the policy near the trusted region with the KL leash, and (b) instrument for the known exploits, length first.

It is worth being precise about *why* the policy is so good at finding these exploits, because it explains why reward hacking is the default outcome rather than a rare accident. The policy is an extremely high-capacity function being optimized by gradient descent against a fixed, differentiable-through-sampling reward. Any systematic correlation in the reward — length, politeness markers, list formatting, agreeing with the user — is a direction in which the gradient points, and the policy will move in every such direction simultaneously, weighted by how cheaply each one buys reward. Length is usually the cheapest exploit (adding tokens costs almost nothing and the reward model reliably pays for it), which is why it dominates. The policy is not "cheating" in any intentional sense; it is doing exactly what gradient ascent on a proxy must do. This reframing matters for debugging because it tells you the exploit is *structural*: you will not fix it by training longer or with more data on the same reward, because the gap that the policy is exploiting is in the reward itself. You fix it by closing the gap (debias the reward, control for length) or by bounding how far the policy may go to find it (the KL leash). Treating reward hacking as a bug to be trained away rather than a structural property of proxy optimization is the most common conceptual error in the whole area.

For PPO specifically, where the reward model is an explicit artifact you can inspect, the right `trl` `PPOTrainer` setup keeps the KL coefficient visible and adaptive. The KL coefficient is the literal leash strength, and the single most important thing you can do is *watch it and the KL together*:

```python
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

config = PPOConfig(
    learning_rate=1e-6,           # tiny, like DPO; PPO is sensitive
    batch_size=64,
    mini_batch_size=8,
    init_kl_coef=0.2,             # the leash strength; adaptive by default
    target=6.0,                   # target KL the controller steers toward
    cliprange=0.2,                # PPO ratio clip; bounds per-step policy move
    cliprange_value=0.2,
)
policy = AutoModelForCausalLMWithValueHead.from_pretrained("my-sft-7b")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("my-sft-7b")  # frozen
tok    = AutoTokenizer.from_pretrained("my-sft-7b")

ppo = PPOTrainer(config, policy, ref, tok)
# In the loop: generate, score with the reward model, ppo.step(...).
# WATCH: stats["objective/kl"] (the leash tension) and stats["ppo/mean_scores"]
# (the reward). KL climbing past `target` while scores climb = reward hacking.
```

Two settings carry most of the stability. `init_kl_coef` with an adaptive controller and a `target` KL means the trainer automatically tightens the leash when KL exceeds the target — a guardrail against blowup, but only if `target` is set sanely (too high and it never tightens). And `cliprange` is PPO's per-step move limit, the optimization-level analogue of a small learning rate; a too-large cliprange lets a single update move the policy far, which is its own route to instability. When a PPO run blows up, check these two before anything else.

### The diagnostic: a reward-versus-length check

Here is the ten-line check that should run after every preference-tuning run, before you trust the reward. It samples generations across reward levels and correlates reward with length. If the correlation is high, your reward gains are suspect.

```python
import numpy as np

def reward_length_check(prompts, model, tokenizer, reward_fn, n=256):
    rewards, lengths = [], []
    for p in prompts[:n]:
        out = model.generate(**tokenizer(p, return_tensors="pt").to(model.device),
                             max_new_tokens=512, do_sample=True, temperature=0.7)
        gen = out[0][tokenizer(p, return_tensors="pt").input_ids.shape[1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        rewards.append(reward_fn(p, text))     # RM score, or judge score
        lengths.append(len(gen))               # in tokens
    r = np.corrcoef(rewards, lengths)[0, 1]
    print(f"corr(reward, length) = {r:+.3f}  | mean_len = {np.mean(lengths):.0f}")
    if r > 0.5:
        print("WARNING: reward is strongly length-coupled. Suspect length hacking.")
    return r, rewards, lengths
```

A correlation above roughly $+0.5$ between reward and token-length is a loud signal that your reward improvements are bought with length, not quality. The honest confirmation is the one that matters: take the policy and the SFT reference, generate on a fixed prompt set, and run a **length-matched** or **length-controlled** human/judge comparison — bucket by length and compare within bucket, or explicitly penalize length in the comparison. If the win rate evaporates once you control for length, you had length hacking, not improvement.

### The fix: length penalties, length normalization, and a tighter leash

Three fixes, in rough order of how surgical they are:

1. **Length-normalize the DPO objective.** Divide each log-prob by the response length so the per-token log-ratio, not the summed one, drives the margin. This is the core of the length-controlled / "LN-DPO" family and it directly removes the "more tokens = more margin" mechanism. In `trl`, this is exposed as a length-normalization or length-penalty option on the DPO config in recent versions.
2. **Add an explicit length penalty to the reward (PPO) or as a regularizer (DPO).** Subtract $\lambda \cdot (\text{len})$ from the reward so the policy pays for length and only spends tokens that earn their keep.
3. **Debias the reward model or the data.** If chosen responses are systematically longer, the cleanest fix is upstream: balance lengths in the preference pairs, or train the reward model with a length-decorrelation objective so it stops rewarding length per se.

#### Worked example: length penalty turns a hack into a win

Run A is a DPO run on a 7B SFT model, $\beta = 0.1$, no length control. Over 800 steps: implicit reward (mean chosen reward) climbs from $0.2$ to $0.8$; mean generation length on a fixed prompt set climbs from 80 tokens to 320; the correlation between per-prompt reward and length is $+0.71$. A length-matched judge eval against the SFT reference gives a **47% win rate** — the model is *losing* to its own starting point once you control for length. Run B is identical except DPO length-normalization is on. Over the same 800 steps: implicit reward climbs more slowly, from $0.2$ to $0.55$; mean length goes from 80 to 110 tokens (essentially stable); reward-length correlation drops to $+0.18$; and the judge eval gives a **63% win rate**. Same data, same $\beta$, same steps — the only change was removing length's free ride on the margin, and the run went from a regression dressed as progress to a genuine improvement. The reward number was *lower* in Run B, which is the lesson: in preference tuning, a lower reward you can trust beats a higher reward you cannot.

## 4. KL blowup and KL collapse: reading the leash

The KL-to-reference curve is the single most diagnostic instrument in preference tuning, and it has two opposite failure modes that look nothing alike.

**KL blowup** is the dangerous one. KL to the reference climbs without bound — 5, then 18, then 45 nats over a sequence — and the generations degenerate: repetition, gibberish, broken formatting, a collapse into a single mode ("As an AI language model, As an AI language model, As an AI..."). The policy has slipped its leash and walked into a corner of text-space where the reward (or the DPO margin) is high but the text is garbage. In DPO, the analogue is the policy driving the log-ratio of chosen responses to enormous positive values and rejected to enormous negative — the margin balloons, the loss approaches zero, and the model has stopped being a language model. The cause is almost always **$\beta$ too small** (a loose leash) and/or **learning rate too high**, sometimes compounded by a reward model that is easily gamed.

![A timeline of reward rising while KL runs away from 0 to 45 over training steps, then settling at 8 once beta is raised](/imgs/blogs/debugging-rlhf-dpo-and-preference-tuning-4.png)

**KL collapse** is the opposite and quieter failure: KL pins near zero, the policy barely moves from the reference, and reward/margin is flat. The model is not learning the preferences at all. The cause is usually **$\beta$ too large** (the leash is so tight the policy cannot move), or the **learning rate too low**, or — and this is a real one — the **reference is identical to the policy and not properly separated**, so the log-ratios are structurally near zero. A DPO run whose loss sits at $0.69$ and whose margin hovers at $0.0$ is in KL collapse: it is the coin-flip floor we computed earlier.

The science of the blowup is the reward-hacking mechanism plus an unbounded optimizer. With the KL leash weak, the effective objective is dominated by the (imperfect, out-of-distribution-unreliable) reward, and gradient ascent on an unbounded, exploitable function has no reason to stop. The KL grows roughly monotonically because each step that increases reward also, on average, moves the policy further from the reference. The text degrades because the high-reward region the policy is climbing toward is out of the reference's distribution, which is exactly the text humans never wrote and the reward model never reliably scored. The cure is to make straying expensive again: raise $\beta$ so each nat of KL costs more reward, and/or lower the LR so the policy moves in smaller steps and you can catch the climb before it runs away.

There is a clean way to see *why* $\beta$ sets the runaway point, straight from the PPO objective. The policy is climbing $r_\phi(x,y) - \beta\,\mathrm{KL}$, and it will keep adding KL as long as the marginal reward per marginal nat of KL is positive — that is, as long as $\frac{\partial r}{\partial \mathrm{KL}} > \beta$. Near the reference, real improvements have a large reward-per-KL slope, so the policy moves and KL rises healthily. But the reward model's reliable region is finite: once the policy exhausts the genuine improvements, the only remaining reward gains come from exploits, whose true reward-per-KL slope is *small* (a lot of straying for a little proxy reward). With $\beta$ set correctly, $\beta$ exceeds that exploit slope, so the policy stops — it is no longer worth the KL. With $\beta$ too small, $\beta$ is below even the exploit slope, so the policy keeps paying KL for tiny proxy gains, walks out of the trusted region, and the text degenerates. **The right $\beta$ is the one that is larger than the exploit slope but smaller than the genuine-improvement slope**, which is why there is a usable window (typically around $0.05$–$0.2$) and why both edges of that window are failure modes. This is also the gradient-level statement of the over-optimization inverted-U from Section 12: KL is the x-axis, and $\beta$ sets how far along it you are allowed to walk.

The blowup also has a characteristic *speed*. Because the runaway is self-reinforcing — more KL exposes more out-of-distribution reward to exploit, which motivates more KL — the KL curve tends to be concave-up (accelerating) once it leaves the healthy plateau, not linear. That acceleration is a useful early-warning property: a KL that is climbing *and curving upward* is heading for a blowup and you should intervene before it reaches the gibberish regime, rather than waiting for the samples to break. A KL that is climbing but *decelerating* toward a plateau is the healthy shape. Watch the second derivative of the KL curve, informally, not just the level.

### The diagnostic: log and read the KL curve

In PPO with `trl`'s `PPOTrainer`, the KL is computed and logged for you — watch `objective/kl` (or the equivalent stats key) over steps. In DPO, KL is not directly in the loss, but you can and should approximate it by tracking the mean log-ratio magnitude, and you can compute an explicit sequence-KL on a fixed eval set periodically:

```python
import torch

@torch.no_grad()
def approx_seq_kl(policy, ref, prompts, tokenizer, n=128):
    # KL(policy || ref) approximated on policy samples, per sequence
    kls = []
    for p in prompts[:n]:
        ids = tokenizer(p, return_tensors="pt").to(policy.device)
        out = policy.generate(**ids, max_new_tokens=128, do_sample=True, temperature=1.0,
                              return_dict_in_generate=True, output_scores=False)
        full = out.sequences
        pl = policy(full).logits.log_softmax(-1)
        rl = ref(full).logits.log_softmax(-1)
        # gather log-probs of the realized tokens, sum the per-token KL surrogate
        tok = full[:, 1:].unsqueeze(-1)
        lp = pl[:, :-1].gather(-1, tok).squeeze(-1)
        lr = rl[:, :-1].gather(-1, tok).squeeze(-1)
        kls.append((lp - lr).sum().item())   # sum_t [log pi - log ref] on sampled tokens
    print(f"mean seq KL surrogate = {sum(kls)/len(kls):+.2f} nats")
    return kls
```

The number itself is setup-dependent; what matters is the **trajectory**. A surrogate KL that climbs past where it was at the last healthy checkpoint, especially while reward climbs, is the blowup signature. A surrogate KL pinned near zero with flat reward is collapse. Plot it next to reward and the diagnosis is usually immediate.

#### Worked example: $\beta$ rescues a blowup

A DPO run with $\beta = 0.02$ (a deliberately loose leash) on a 7B model: by step 200 the margin is $1.5$ and KL surrogate is $4$; by step 500 margin $4.0$, KL $18$; by step 800 margin $9.0$, KL $45$, and sampled generations are repetitive fragments — a textbook blowup. Same data, same LR, $\beta$ raised to $0.1$: the implicit reward is now five times more expensive per nat of log-ratio, so the policy stays close; by step 800 margin is a moderate $0.6$, KL surrogate settles around $8$, and generations are clean and improved. The single knob $\beta$, moved from $0.02$ to $0.1$, converted a degenerating policy into a stable one. If raising $\beta$ alone underfits (margin and reward too flat), drop the LR by 2–4x as well; the two together — tighter leash, smaller steps — are the reliable recovery for a blowup.

## 5. DPO/IPO diagnostics: margins, accuracy, and the $\beta$ dial

Beyond reward and KL, preference tuning has two instruments unique to the offline (DPO/IPO) setting that tell you, cheaply and continuously, whether the model is learning the preference: the **reward margin** and the **reward accuracy**.

The **margin** is the mean of $\hat{r}_w - \hat{r}_l$ over the batch — the average gap between chosen and rejected implicit rewards. It should start near zero (the policy equals the reference, so all log-ratios are zero) and climb to a moderate positive value. A margin that climbs steadily and plateaus at, say, $0.5$ to $1.0$ is healthy; a margin pinned at zero means no learning; a margin exploding to $5$+ means blowup. The margin is the DPO analogue of "is the loss going down for the right reason."

The **reward accuracy** is the fraction of pairs where the policy gives the chosen response higher implicit reward than the rejected one: `(chosen_reward > rejected_reward).mean()`. This is the most interpretable single number in DPO. It starts near $0.5$ (the policy equals the reference, so it is right by chance on ties and by the reference's mild preferences otherwise — often slightly above $0.5$ if the SFT model already prefers chosen). It should climb toward $0.7$–$0.9$ on the training set as the model learns the preference. Three readings are diagnostic:

- **Accuracy stuck at $\approx 0.5$**: the model is not learning the preference at all. Suspect $\beta$ wrong, LR too low, or — critically — **the chosen and rejected are swapped or nearly identical** (Section 5 below). This is the "coin-flip floor," loss $\approx 0.69$.
- **Accuracy *below* $0.5$**: the model is learning the preference *backwards*. This is the loud, unambiguous signature of **swapped chosen/rejected columns** in your data. If reward accuracy drops below chance, stop and audit your data before anything else.
- **Train accuracy high, eval accuracy low**: overfitting to the preference set — the model memorized which specific responses were chosen rather than learning the underlying preference. Common with too many epochs (preference tuning overfits *fast*, often within 1–2 epochs) or $\beta$ too low.

**The $\beta$ dial, made concrete.** Think of $\beta$ as the contrast knob on the implicit reward, with two failure directions you can now read off the instruments:

- **$\beta$ too high** (e.g. $0.5$): the leash is tight, the policy cannot move, margin and accuracy climb slowly and plateau low, the model *underfits* the preferences. KL stays small. Loss falls only a little.
- **$\beta$ too low** (e.g. $0.01$): the leash is loose, the policy moves a lot, margin can explode, KL climbs, and you get either reward hacking / blowup or a degenerate policy that overfits the training pairs and generalizes poorly. Loss can look great while quality is terrible.

The standard $\beta = 0.1$ is a good start; if margin/accuracy barely move, halve $\beta$; if KL climbs or generations degrade, double it. IPO (Identity Preference Optimization) is a variant that replaces the log-sigmoid with a squared loss to be more robust to the overfitting that DPO shows when preferences are near-deterministic or pairs are noisy — if your DPO accuracy hits 0.99 on train and the model degenerates, IPO's regularization is the principled alternative, and it is a one-line `loss_type` change in `trl`.

The reason IPO behaves better on noisy data is itself a useful piece of science. Vanilla DPO's log-sigmoid loss is unbounded in its drive: for a pair where the preference is treated as deterministic, the loss keeps decreasing as the margin grows without limit, so the optimizer happily pushes the chosen log-ratio to $+\infty$ and the rejected to $-\infty$ on any pair it can — including mislabeled ones. There is no point at which a correctly-but-noisily-labeled pair says "stop, I am satisfied." IPO replaces this with a squared loss around a *target* margin proportional to $1/\beta$, so the objective has a finite minimum: once the margin reaches its target the loss stops pushing. That bounded target is exactly what prevents a handful of noisy pairs from driving the policy to extremes, which is the degeneration you see when DPO train accuracy saturates near 1.0. The debugging rule that follows: **if DPO train accuracy climbs toward 1.0 while eval quality falls, the loss is over-fitting individual pairs, and a bounded-target objective like IPO (or simply fewer epochs and noisier-pair filtering) is the fix** — not more data on the same loss.

### The diagnostic: the full DPOTrainer config with the right logging

Here is a `trl` `DPOTrainer` setup with the instruments wired in. The key is that `DPOTrainer` already logs `rewards/chosen`, `rewards/rejected`, `rewards/accuracies`, and `rewards/margins` — your job is to *watch* them, and to log mean generation length alongside.

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("my-sft-7b", torch_dtype="bfloat16")
ref   = AutoModelForCausalLM.from_pretrained("my-sft-7b", torch_dtype="bfloat16")  # frozen SFT
tok   = AutoTokenizer.from_pretrained("my-sft-7b")

config = DPOConfig(
    beta=0.1,                       # the KL/temperature dial; 0.1 is the default start
    learning_rate=5e-7,             # tiny — preference tuning needs ~1e-6 to 5e-7, not 1e-5
    num_train_epochs=1,             # overfits fast; 1-2 epochs max
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    loss_type="sigmoid",            # "ipo" for IPO; "sigmoid" is vanilla DPO
    max_length=1024,
    max_prompt_length=512,
    logging_steps=10,               # watch margins/accuracies/kl every 10 steps
    eval_strategy="steps",
    eval_steps=100,
    bf16=True,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref,                  # explicit frozen reference (see Section 6)
    args=config,
    train_dataset=train_pairs,      # columns: prompt, chosen, rejected
    eval_dataset=eval_pairs,
    processing_class=tok,
)
trainer.train()
```

Watch four series in your dashboard: `rewards/accuracies` (should climb past $0.6$), `rewards/margins` (should climb to a moderate positive plateau), the eval-set versions of both (to catch overfitting), and — logged via a callback — mean generation length on a fixed prompt set (to catch length hacking). Those four read out every failure in this post.

## 6. The reference-model fault: wrong, stale, or not frozen

Every implicit reward in DPO and every KL penalty in PPO is computed *relative to the reference model*. If the reference is wrong, every reward is wrong, silently. This is the subtlest preference-tuning bug because nothing crashes — the loss computes, the margin moves — but the baseline is meaningless, so the gradients push the policy toward the wrong target. There are three distinct ways to get the reference wrong.

![A before-and-after panel contrasting a wrong or un-frozen reference against a reference frozen to the SFT init](/imgs/blogs/debugging-rlhf-dpo-and-preference-tuning-6.png)

**The reference is not the SFT init.** DPO's whole derivation assumes the reference is the model you are improving *from* — the SFT checkpoint. If you accidentally pass the base (pre-SFT) model as the reference, or a different finetune, the log-ratios measure "policy versus base" instead of "policy versus SFT," and DPO optimizes a different objective than you intended. The symptom is subtle: training proceeds, but the model can regress on instruction-following because it is being pushed away from base behaviors the SFT model deliberately kept. The fix is the discipline of **always loading the reference from the exact SFT checkpoint the policy was initialized from.**

**The reference is not frozen.** If `ref_model`'s parameters have `requires_grad=True` or, worse, you accidentally share parameters between policy and reference (so updating the policy updates the reference), the log-ratios drift toward zero as the reference chases the policy. The implicit reward collapses, the margin flattens, and the model stops learning — KL collapse with a confusing cause. In `trl`, `DPOTrainer` handles freezing for you when you pass an explicit `ref_model`, but if you build a custom loop you must freeze it yourself. Always assert it.

**The reference is stale or absent.** Some efficient DPO setups skip an explicit reference and use the *initial* policy (via PEFT: the base model with the adapter disabled) as the reference. That is fine — but only if the adapter-disabled forward genuinely reproduces the SFT model. If you LoRA-tune from a base model without first SFT-ing, the "reference" (adapter off) is the *base* model, not an SFT model, and you have silently re-created the first bug. Confirm that "reference equals SFT" holds in whatever reference scheme you use.

### The diagnostic: assert the reference is frozen and correct

This is a five-line check that has saved more runs than any clever trick. Before training, assert (a) the reference is in eval mode, (b) its parameters require no grad, and (c) it produces SFT-equal log-probs on a probe batch:

```python
import torch

def assert_reference_ok(ref_model, sft_model, probe_input):
    # (a) frozen: no parameter should require grad
    n_trainable = sum(p.requires_grad for p in ref_model.parameters())
    assert n_trainable == 0, f"reference has {n_trainable} trainable params; freeze it"
    # (b) eval mode (no dropout drift in the reference log-probs)
    assert not ref_model.training, "reference must be in .eval() mode"
    # (c) reference == SFT: log-probs must match the SFT checkpoint on a probe
    with torch.no_grad():
        r = ref_model(**probe_input).logits
        s = sft_model(**probe_input).logits
    max_diff = (r - s).abs().max().item()
    assert max_diff < 1e-3, f"reference != SFT (max logit diff {max_diff:.4f})"
    print("reference OK: frozen, eval, equals SFT init")
```

Run this once at startup. If (a) fails, you forgot to freeze; if (c) fails, your reference is the wrong checkpoint. Both are silent corruptions that this assert turns into a loud startup error.

#### Worked example: a base-model reference half-breaks a run

A team runs LoRA DPO directly on a base model (no SFT step), using adapter-off as the reference. The run trains: loss falls from $0.69$ to $0.55$, reward accuracy reaches $0.71$ on the preference set. But the deployed model is *worse* at following instructions than a plain SFT baseline, and on a held-out instruction eval it scores below the SFT model they never trained. The bisection: the reference-assert above flags nothing about freezing (the adapter-off forward is genuinely frozen) but the *probe-equals-SFT* check fails — adapter-off equals **base**, not SFT, with a large logit diff. Diagnosis: they were doing "DPO from base," pushing the policy toward chosen responses *relative to base*, with no SFT foundation to preserve instruction-following. The fix is the SFT-before-DPO requirement (Section 8): SFT first, then DPO with the SFT checkpoint as reference. After the fix, the same preference data lifts the SFT model's instruction-eval win rate instead of dragging it down.

## 7. Preference-data bugs: swaps, ties, noise, and near-identical pairs

Preference tuning is only as good as its pairs, and preference data has its own catalogue of bugs — distinct from the label-noise and leakage bugs of supervised data, because the unit is a *comparison*, not a label.

**Chosen/rejected swapped.** The single most catastrophic and most common data bug: the chosen and rejected columns are switched, for the whole dataset or a subset, so DPO is trained to prefer the *worse* response. The signature is unmistakable on the instruments: **reward accuracy drops below 0.5** and stays there, and the model gets actively worse. This is why the "accuracy below chance = swapped pairs" rule in Section 5 is so valuable — it is a one-glance diagnosis. The fix is to swap the columns back; the test is to verify reward accuracy climbs above $0.5$ after.

**Ties and near-identical pairs.** If the chosen and rejected responses are identical or nearly so (which happens when preference data is generated by sampling two responses from the same model at low temperature, or when annotators marked a tie as a forced choice), the log-ratio difference is tiny by construction, the margin is structurally near zero, and the gradient is weak and noisy. The model "learns nothing" not because of $\beta$ but because the pairs carry no signal. The diagnostic is to measure the distribution of (chosen length − rejected length) and a text-similarity score per pair; a mass of near-identical pairs is dead weight at best and noise at worst. Filter them.

**Noisy labels.** Human preference labels are noisy — inter-annotator agreement on hard pairs is far from perfect. DPO's log-sigmoid loss, like any classification loss, will try to fit even the mislabeled pairs, and on a near-deterministic-preference dataset it can overfit hard, driving the model to extreme log-ratios on noisy examples. This is precisely the failure IPO was designed to mitigate: its squared-loss regularization caps how hard a single pair can push the margin. If your DPO reward accuracy hits $0.95$+ on train while eval accuracy and real quality fall, suspect noisy pairs and reach for IPO or for filtering low-agreement pairs.

### The diagnostic: audit the pairs before you train

A few lines that catch the worst data bugs before they waste a run:

```python
import numpy as np

def audit_preference_pairs(dataset, tokenizer):
    swaps = []
    len_chosen, len_rejected, near_identical = [], [], 0
    for ex in dataset:
        lc = len(tokenizer(ex["chosen"]).input_ids)
        lr = len(tokenizer(ex["rejected"]).input_ids)
        len_chosen.append(lc); len_rejected.append(lr)
        if ex["chosen"].strip() == ex["rejected"].strip():
            near_identical += 1
    lc_m, lr_m = np.mean(len_chosen), np.mean(len_rejected)
    print(f"mean chosen len {lc_m:.0f} vs rejected len {lr_m:.0f}  (gap {lc_m-lr_m:+.0f})")
    print(f"identical pairs: {near_identical} / {len(dataset)}")
    if lc_m - lr_m > 30:
        print("WARNING: chosen much longer than rejected -> length bias risk in DPO")
    if near_identical > 0.02 * len(dataset):
        print("WARNING: many near-identical pairs -> weak/no gradient")
```

The chosen-versus-rejected length gap is doing double duty here: a large positive gap both warns of dead near-identical-content pairs *and* predicts the length bias of Section 3, because that length gap is exactly what DPO can exploit. If `mean chosen len` is much larger than `mean rejected len`, you have a length-bias time bomb in your data, and you should either length-balance the pairs or turn on length-normalized DPO from the start.

## 8. The SFT-before-DPO requirement and the wrong-checkpoint trap

Preference tuning is the *second* stage of alignment, not the first, and skipping or botching the first stage is a class of bug all its own. DPO and RLHF both assume a competent **SFT model** as the starting point and reference. The preference stage refines a model that already follows instructions; it cannot teach instruction-following from scratch, because preference pairs are about *which of two reasonable responses is better*, not about *how to respond at all*.

Two failure modes follow. First, **DPO from a non-SFT checkpoint** (base model, or a weak SFT) gives the policy nothing to refine and a meaningless reference, exactly the Section 6 worked example. The model can chase chosen responses while losing the instruction-following it never had. Second, the **wrong SFT checkpoint** — initializing the policy from one checkpoint while using a different one as the reference, or resuming from a checkpoint with a different tokenizer/template — produces a mismatch where the implicit rewards are computed against the wrong baseline. The rule is simple and worth carving in stone: **policy init and reference must be the same SFT checkpoint, and that checkpoint must be a real, competent SFT model trained with the same tokenizer and chat template you will serve.**

This also intersects with chat-template skew, the subject of a sibling post: if your SFT model was trained with one chat template and you feed DPO pairs formatted with a different template, the log-probs the model assigns are off-distribution, the log-ratios are noisy, and learning is degraded. Format the preference pairs with *exactly* the template the SFT model expects.

## 9. Length normalization and the implicit-reward computation

We have referenced length normalization several times; it earns a section because the *implicit-reward computation* is where several bugs converge and where the length fix lives.

Recall the implicit reward $\hat{r}_\theta(x, y) = \beta(\log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x))$, with $\log \pi(y \mid x) = \sum_{t} \log \pi(y_t \mid x, y_{<t})$ a **sum** over response tokens. Two correctness requirements live in this sum, and both are common bugs:

1. **The sum must cover the response tokens only, not the prompt.** Just as in the loss-masking bug for SFT, the log-prob must be computed over the *completion*, with the prompt tokens masked out (`labels = -100` on the prompt). If you accidentally include prompt tokens in the log-prob sum, the implicit reward is dominated by how the model scores the (shared) prompt, the chosen/rejected difference is diluted, and the margin is muddied. `trl`'s `DPOTrainer` masks correctly by default, but custom loops get this wrong constantly. This is the direct cousin of the loss-masking bug — the prompt must not enter the reward.
2. **Length normalization divides the sum by the response length.** Vanilla DPO uses the raw sum, which is what couples margin to length (Section 3). Length-normalized DPO uses $\frac{1}{|y|}\sum_t \log \pi(y_t \mid \cdot)$ — the mean per-token log-prob — so a long response and a short one are on equal footing and the model cannot win the margin by adding tokens. This is the principled fix for length hacking and is exposed as a config flag in recent `trl`.

The diagnostic that ties these together is to **log the per-token implicit reward** (the reward divided by length) alongside the raw implicit reward. If the raw reward climbs but the per-token reward is flat, your reward gains are purely length — the cleanest possible confirmation of length hacking, computed from the quantities you already have:

```python
# inside a logging callback: raw vs length-normalized implicit reward
chosen_logratio = policy_chosen_logps - ref_chosen_logps        # summed over tokens
raw_reward      = beta * chosen_logratio
per_token       = beta * (chosen_logratio / chosen_response_len) # length-normalized
# if raw_reward climbs while per_token is flat -> reward is bought with length
```

#### Worked example: per-token reward unmasks the hack

Back to Run A (no length control). Logging both: the raw implicit reward of chosen responses climbs from $0.20$ to $0.80$ over 800 steps, but the **per-token** implicit reward is essentially flat, drifting from $0.0025$ to $0.0028$. The raw number quadrupled; the per-token number did not move. The model did not get better per token — it got *longer*, and the summed reward grew because there were more tokens to sum. This single side-by-side (raw versus per-token reward) is the most decisive length-hacking diagnostic in the post, because it is computed from the exact quantities DPO already produces, requires no extra generation, and directly separates "more reward per token" (real improvement) from "more tokens" (the hack).

## 10. Putting it together: the bisection on a real failing run

Now the full discipline, end to end, on the run we opened with: a 7B DPO finetune where reward climbs, the loss falls, and the side-by-side eval *loses*. We bisect symptom → suspect → confirming test → fix, reading the instruments, never fixing on a hunch.

![A symptom-by-cause matrix mapping each preference-tuning failure to its confirming test and the fix the curves verify](/imgs/blogs/debugging-rlhf-dpo-and-preference-tuning-5.png)

**Step 1 — Is the data sane?** Before blaming the optimizer, audit the pairs. Run `audit_preference_pairs`. Reward accuracy on the training set is $0.74$ and climbing, comfortably above $0.5$, so the chosen/rejected columns are *not* swapped — rule out the catastrophic data bug. But the audit prints `mean chosen len 210 vs rejected len 120 (gap +90)` and warns of length-bias risk. First real clue: the chosen responses are systematically much longer than the rejected.

**Step 2 — Is the reference correct?** Run `assert_reference_ok`. It passes: the reference is frozen, in eval mode, and equals the SFT checkpoint. Rule out the reference fault (Section 6). The baseline is correct, so the rewards are measured against the right anchor — the problem is not the reference.

**Step 3 — Read the KL/length instruments.** KL surrogate over the run climbed from 0 to a moderate $\approx 10$ — not a blowup, the leash held. So this is *not* KL instability; $\beta = 0.1$ is doing its job on stability. But the reward-versus-length check returns `corr(reward, length) = +0.71, mean_len 80 -> 320`, and the raw-versus-per-token reward log shows raw reward quadrupling while per-token reward is flat. Two independent instruments now point at the same suspect: **length hacking**. The reward went up because the outputs got longer, not better — which is exactly why the length-matched eval loses.

**Step 4 — Confirm with the decisive test.** A length-matched judge eval against the SFT reference: bucket generations by length and compare within bucket. The win rate collapses to $47\%$ once length is controlled. Confirmed: the reward gains were length, the model is a regression in disguise.

**Step 5 — Fix and verify.** Turn on length-normalized DPO (or add a length penalty), keep $\beta = 0.1$, retrain. The instruments after: reward-length correlation drops to $+0.18$, per-token reward now climbs (real improvement), mean length stabilizes at 110, and the length-matched eval rises to $63\%$. The bisection found length hacking in three cheap tests — a data audit, a reference assert, and a reward-length check — none of which required re-running the model from scratch to *diagnose*, only to *fix*.

The shape of that bisection generalizes into a small taxonomy you can carry into any preference-tuning run. A symptom routes down one of three branches — a reward branch (the reward model or the implicit reward is being gamed), a stability branch (the KL leash is misbehaving in one of two directions), or a data branch (the pairs themselves are wrong) — and each branch has one cheapest confirming test that either lands the diagnosis or sends you to the next branch.

![A taxonomy tree routing a preference-tuning symptom through a reward branch, a stability branch, and a data branch to its cheapest confirming test](/imgs/blogs/debugging-rlhf-dpo-and-preference-tuning-8.png)

The tree is worth internalizing because the three branches fail in *non-overlapping* ways on the instruments, which is what makes the bisection cheap. The reward branch shows reward up with an exploit instrument up (length, sycophancy, format) while KL stays moderate — the reward is climbing for the wrong reason but the leash is intact. The stability branch shows KL doing something extreme — running away (blowup) or pinned at zero (collapse) — and the fix is the $\beta$/LR dial, not the data. The data branch shows reward accuracy misbehaving — below chance for swaps, stuck at exactly chance for ties and near-identical pairs — independent of KL. Because the three branches light up different channels, you almost never confuse them once you are logging all five instruments; the failure mode of *misdiagnosing* preference tuning is nearly always *not logging enough channels*, so you see "reward up" and stop, instead of seeing "reward up, length up, per-token flat, KL moderate, accuracy healthy" and reading the whole pattern.

### Stress-testing the diagnosis

The bisection above found length hacking, but a disciplined debugger asks "what if it had been something else?" and knows the alternate signatures cold:

- **What if KL had blown up (5 → 45) instead of holding at 10?** Then the suspect is the leash, not length: raise $\beta$ (and/or lower LR), per Section 4. The discriminator is the KL trajectory — runaway KL is instability, moderate KL with length-coupled reward is length hacking.
- **What if reward accuracy had been below $0.5$?** Then stop everything and unswap the pairs (Section 7). Accuracy below chance is *only* explained by swapped chosen/rejected; no amount of $\beta$ tuning fixes inverted labels.
- **What if the loss were pinned at $0.69$ and the margin at $0.0$?** KL collapse: $\beta$ too high, LR too low, or the reference is accidentally trainable/shared. Check the reference-frozen assert first, then drop $\beta$, then raise LR.
- **What if it were PPO, not DPO?** The reward-hacking logic is identical, but the KL is in your logs directly (`objective/kl`) and the reward model is a separate, inspectable artifact — you can probe the reward model on length-controlled pairs to confirm it has a length bias, which DPO hides inside the implicit reward.
- **What if train accuracy is $0.97$ but eval quality is falling?** Overfitting to noisy or near-deterministic preferences: cut epochs to 1, filter low-agreement pairs, or switch `loss_type` to IPO for its regularization.

Each branch is a different suspect with a different confirming test. The matrix figure above is the lookup table; the instruments tell you which row you are in.

## 11. A reference table: symptom to suspect to test to fix

The diagnostic table for preference tuning, the way the loss-curve and taxonomy posts have one. Read down the symptom column, across to the cheapest confirming test, then the fix.

| Symptom (on the instruments) | Likely suspect | Cheapest confirming test | Fix |
| --- | --- | --- | --- |
| Reward ↑, length ↑ in lockstep | Length / format hacking | `corr(reward, length)` > 0.5; per-token reward flat | Length-normalized DPO or length penalty |
| Reward ↑, quality ↓ (length controlled) | Reward hacking (sycophancy/format) | Length-matched judge eval; inspect samples | Tighter KL ($\uparrow\beta$); debias RM; better data |
| KL 5 → 45, text degenerates | KL blowup, leash too loose | Read KL curve; sample generations | Raise $\beta$; lower LR; clip |
| KL ≈ 0, reward/margin flat | KL collapse, leash too tight | KL surrogate near 0; loss ≈ 0.69 | Lower $\beta$; raise LR; check ref frozen |
| Reward accuracy < 0.5 | Chosen/rejected swapped | `reward_acc` below chance | Unswap the pair columns |
| Margin ≈ 0, loss stuck at 0.69 | $\beta$ too high or pairs near-identical | Audit pair similarity; halve $\beta$ | Lower $\beta$; filter ties; check data signal |
| Loss falls, model regresses on instructions | Wrong/missing SFT; bad reference | Reference-equals-SFT probe assert | SFT first; reference = SFT checkpoint |
| Train acc 0.97, eval quality ↓ | Overfit to noisy preferences | Train vs eval accuracy gap | 1 epoch; IPO loss; filter low-agreement pairs |
| Margin moves but log-prob includes prompt | Prompt not masked in reward | Check loss masking; inspect token spans | Mask prompt (`-100`); response-only log-prob |

This table is the compressed form of the whole post. The instruments — reward, KL, margin, accuracy, length — are five channels; every row is a distinct pattern across those five channels; and the fix follows from the suspect. The skill is reading the channels together, because individually they mislead (reward up looks like success) and jointly they diagnose (reward up + length up + per-token flat = length hacking).

## 12. Case studies and real signatures

A few well-known, real patterns to calibrate your eye, cited honestly.

**DPO and the reference-relative reward (Rafailov et al., 2023).** The DPO paper ("Direct Preference Optimization: Your Language Model is Secretly a Reward Model") derived the loss we use, showing that the optimal RLHF policy can be expressed in closed form and that fitting preferences directly with the log-sigmoid loss is equivalent to RLHF under the same KL-constrained objective. The crucial, debuggable consequence is that DPO's reward is *implicit and reference-relative* — there is no separate reward model to inspect, which is precisely why the reference-model fault (Section 6) is so easy to introduce and so silent. The $\beta = 0.1$ default and the role of $\beta$ as the KL weight trace to this work.

**The KL penalty in InstructGPT/RLHF (Ouyang et al., 2022).** The InstructGPT recipe ("Training language models to follow instructions with human feedback") established the three-stage pipeline and the KL-penalized PPO objective. The paper and the broader RLHF literature are explicit that the KL term to the SFT reference is what keeps the policy in the region where the reward model is reliable; remove or under-weight it and the policy over-optimizes the reward model. The reward-hacking and KL-blowup signatures in this post are the practical manifestation of that well-documented over-optimization. (The exact KL coefficients and reward scales are setup-specific; the mechanism is general.)

**Length bias as a pervasive RLHF confound.** A substantial line of work (the "length-controlled" and "disentangling length from quality" literature around AlpacaEval and RLHF reward models) documents that reward models and preference annotations are systematically biased toward longer responses, and that naive RLHF/DPO inflates length without proportionate quality gains — exactly the Run A signature. The proposed fixes — length-controlled win rates, length-normalized DPO, explicit length penalties — are the ones in Section 3. The headline that "much of the apparent RLHF improvement can be length" is real and well-replicated; the precise fraction varies by model and dataset, so treat any single number as setup-dependent, but the *direction* is robust and you should always control for it.

**Over-optimization and the reward-model scaling law.** Work on reward-model over-optimization (the "Scaling Laws for Reward Model Overoptimization" line) showed empirically that as you optimize harder against a fixed proxy reward, true quality first rises then *falls* — an inverted-U — and that the KL distance from the reference is the natural x-axis for that curve. This is the quantitative backbone of "reward up, quality down": there is a point past which more reward optimization is net-negative, and KL-to-reference is how you measure where you are on the curve. When your reward is still climbing but quality has turned over, you are past the peak of the inverted-U, and the fix is to back off (tighter leash, fewer steps, lower LR), not to optimize harder.

## 13. When this is (and isn't) your bug

Reading preference-tuning instruments well also means knowing when a symptom points *away* from preference tuning and toward an ordinary training bug you already know how to fix.

- **A reward that climbs with healthy KL and length-controlled quality really is improvement.** Not every climbing reward is a hack. If the reward-length correlation is low, KL is moderate, and a length-matched eval shows a genuine win, stop looking for a bug — the run is working. Reward hacking is a specific signature (reward up, *and* an exploit instrument up), not "reward went up."
- **A loss stuck at $0.69$ with the reference correct and $\beta$ reasonable is usually data, not optimization.** If the reference passes its assert and $\beta = 0.1$ but the margin will not move, the most likely cause is that the pairs carry no signal (near-identical, ties) — a *data* bug from Section 7 — not an optimizer problem. Audit the pairs before you touch the LR.
- **A NaN in a DPO run is numerics, not preference logic.** If the loss goes to NaN, that is the ordinary numerics story — fp16 underflow in the log-probs, a $\log 0$ from an empty completion, an overflow in the log-sigmoid — and the fix is the NaN-hunting playbook (bf16, `detect_anomaly`, guard the log), not a $\beta$ change. Do not over-think it because the run is RLHF.
- **A model that "won't stop generating" is a chat-template / EOS bug, not reward hacking.** Endless generation is almost always that the EOS token was masked out of the loss or the chat template lacks a stop, a formatting bug, not a preference-tuning failure. Reach for the chat-template post, not the reward dial.
- **If reward accuracy is healthy but eval quality is bad, suspect the *reward itself*, not the policy.** In PPO, the policy can be optimizing a genuinely-bad reward model perfectly. Probe the reward model directly (does it prefer longer/sycophantic/formatted text on controlled pairs?) before assuming the policy is misbehaving. In DPO the equivalent is auditing the preference data, since the data *is* the reward.

The meta-rule is the series' rule: the instruments narrow the suspect, but the *confirming test* closes the case. A climbing reward is a symptom, not a diagnosis. Read reward, KL, margin, accuracy, and length *together*, predict the suspect, run the one cheap test — the length check, the reference assert, the pair audit — then fix. The master decision tree in the taxonomy post is the map that ties every signature here to its branch.

## 14. Key takeaways

- **DPO loss is $-\log\sigma$ of $\beta$ times the chosen-minus-rejected log-ratio gap; it sits at $0.69$ when the model learns nothing.** Memorize $0.693$ as the DPO coin-flip floor, the analogue of $\ln C$ for cross-entropy. A loss pinned there means the margin is zero — no preference learned.
- **Reward going up is not the same as quality going up.** Reward hacking — length bias above all — makes the reward climb while quality falls. Always run a `corr(reward, length)` check and a length-matched eval before trusting a reward number; a correlation above $+0.5$ is a loud warning.
- **The per-token (length-normalized) reward is the decisive length-hacking test.** If raw implicit reward climbs while per-token reward is flat, the model got longer, not better. It is computed from quantities DPO already produces — no extra generation needed.
- **KL-to-reference is the leash, and it is your most diagnostic curve.** Runaway KL (5 → 45) with degenerating text is a blowup ($\beta$ too small, LR too high) — raise $\beta$, lower LR. KL pinned at zero with flat margin is collapse ($\beta$ too high, LR too low, or reference not frozen).
- **$\beta$ is the KL/contrast dial: too high underfits, too low hacks or degenerates.** Start at $0.1$; halve it if margin/accuracy will not move, double it if KL climbs or generations degrade. The DPO $\beta$ and the PPO KL coefficient are literally the same quantity.
- **Reward accuracy below $0.5$ means swapped chosen/rejected — unswap before anything else.** Accuracy below chance has exactly one explanation: inverted preference labels. No optimizer tuning fixes it.
- **The reference must be the frozen SFT checkpoint — assert it.** Every implicit reward is reference-relative; a wrong, stale, or un-frozen reference silently corrupts the objective. Run a five-line probe: frozen, eval-mode, equals SFT.
- **SFT before DPO is a requirement, not a suggestion.** Preference tuning refines an instruction-follower; it cannot create one. DPO from base with the wrong reference regresses instruction-following while the curves look fine.
- **Preference data has its own bugs: swaps, ties, near-identical pairs, noisy labels.** Audit the pairs (length gap, similarity, identical count) before training; for noisy near-deterministic preferences, IPO's regularized loss beats vanilla DPO.
- **The instruments narrow the suspect; the confirming test closes the case.** Read reward, KL, margin, accuracy, and length together — individually they mislead, jointly they diagnose.

## Further reading

- **"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"**, Rafailov, Sharma, Mitchell, Manning, Ermon, Finn, 2023 (arXiv:2305.18290) — the DPO loss, the closed-form optimal policy, and the role of $\beta$ as the KL weight; the basis of the reference-relative implicit reward.
- **"Training language models to follow instructions with human feedback"** (InstructGPT), Ouyang et al., 2022 (arXiv:2203.02155) — the three-stage RLHF pipeline, the reward model, and the KL-penalized PPO objective; why the leash exists.
- **"Scaling Laws for Reward Model Overoptimization"**, Gao, Schulman, Hilton, 2022 (arXiv:2210.10760) — the inverted-U of true quality versus proxy reward with KL-to-reference as the axis; the quantitative backbone of "reward up, quality down."
- **The length-controlled RLHF / AlpacaEval line of work** — on length bias as a pervasive confound in reward models and preference data, and length-controlled win rates as the fix; calibrate your length-bias intuition here.
- **Hugging Face `trl` documentation** — `DPOTrainer`, `PPOTrainer`, `RewardTrainer`, `DPOConfig` (including `loss_type` for IPO and length-normalization options) and the logged `rewards/accuracies`, `rewards/margins`, and KL stats this post depends on.
- Within this series: [A taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the master symptom→suspect→test→fix decision tree), [The training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) (the capstone), [Finetuning an LLM without breaking it](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it), [Your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying), [Reading the loss curve as a diagnostic](/blog/machine-learning/debugging-training/reading-the-loss-curve-as-a-diagnostic), and [Train-infer mismatch for LLMs](/blog/machine-learning/debugging-training/train-infer-mismatch-for-llms).

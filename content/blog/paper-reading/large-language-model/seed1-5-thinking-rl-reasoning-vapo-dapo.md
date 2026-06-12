---
title: "Seed1.5-Thinking: How ByteDance stabilized long-CoT RL with VAPO and DAPO"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A deep-dive into how ByteDance Seed made long chain-of-thought RL stable enough to ship a 20B-active MoE near the frontier — the four DAPO tricks, the eight VAPO levers, the two-tier reasoning verifier, and the streaming rollout infra behind it."
tags: ["seed1.5-thinking", "reinforcement-learning", "reasoning-models", "dapo", "vapo", "ppo", "grpo", "moe", "reward-modeling", "bytedance-seed"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 58
---

There is a quiet, ugly fact behind every reasoning-model release of the last eighteen months: long chain-of-thought reinforcement learning is unstable, and most of the engineering is just keeping it from falling over. The headline numbers — AIME this, Codeforces that — are real, but they are the part of the iceberg above the waterline. Below it sits a much larger mass of stabilizer tricks, reward-hacking defenses, and infrastructure surgery that nobody puts in the abstract because none of it is glamorous. When [DeepSeek-R1](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) showed that pure RL could teach a base model to reason, it also quietly demonstrated how narrow the path is: GRPO works, but it works the way a unicycle works — fine until it isn't, and then you are on the ground.

ByteDance Seed's **Seed1.5-Thinking** report (arXiv:2504.13914) is interesting precisely because it refuses to pretend the path is wide. It is one of the few frontier-adjacent reports that treats RL stability as the central engineering problem rather than a footnote, and it ships **two** RL algorithms instead of one — VAPO (arXiv:2504.05118), a value-based actor-critic method, and DAPO (arXiv:2503.14476), a value-free policy-gradient method — because the team concluded that no single algorithm is the right tool for every data regime. The model itself is a 20B-active / 200B-total Mixture-of-Experts that lands near the frontier on hard reasoning (AIME'24 86.7, GPQA Diamond 77.3, Codeforces 55.0 pass@8) while also *leading* every listed peer on abstract reasoning (ARC-AGI 39.9) and generalizing to non-reasoning chat (+8% win rate over DeepSeek-R1 on human evals). It does all this with **a third of the active parameters** of DeepSeek-R1 (20B vs 37B) and roughly **a sixth of the total** (200B vs 671B).

<!-- FIGSPEC 1
kind: pipeline
claim: Seed1.5-Thinking post-training flows from a 400k SFT cold-start through three RL data buckets into evaluation.
caption: The full post-training pipeline: a small verifiable-heavy cold-start, then RL over three data regimes, each routed to the algorithm that fits it.
nodes:
  - id: a | label: "SFT cold-start\n400k instances" | color: blue
  - id: b | label: "300k verifiable\nSTEM/code/logic" | color: blue
  - id: c | label: "100k non-verifiable\ncreative/chat" | color: gray
  - id: d | label: "RL: verifiable\n~100k STEM" | color: green
  - id: e | label: "RL: non-verifiable\ngenerative RM" | color: amber
  - id: f | label: "RL: hybrid\n~10k logic" | color: green
  - id: g | label: "Eval: AIME 86.7\nARC-AGI 39.9" | color: blue
edges:
  - a -> b
  - a -> c
  - b -> d | label: "verifier"
  - c -> e | label: "pref RM"
  - b -> f | label: "puzzles"
  - d -> g
  - e -> g
  - f -> g
notes: left=SFT (blue/gray), middle=three RL buckets (green=verifiable wins, amber=non-verifiable cost), right=eval blue
-->

![Seed1.5-Thinking post-training pipeline from a 400k SFT cold-start through three RL data buckets into evaluation](/imgs/blogs/seed1-5-thinking-rl-reasoning-vapo-dapo-1.png)

The diagram above is the mental model for the whole post. Everything downstream — the four DAPO tricks, the eight VAPO levers, the two-tier verifier, the streaming rollout system — exists to make one of those three RL buckets actually train without collapsing. This is a paper-reading deep-dive, so we will go through the mechanics with the rigor they deserve: not "DAPO adds dynamic sampling" but *why* a group with accuracy 1.0 produces exactly zero gradient, and what that does to your effective batch size. If you have read our broader [RL-for-LLM-reasoning deep-dive](/blog/paper-reading/large-language-model/part-i-tricks-or-traps-a-deep-dive-into-rl-for-llm-reasoning), treat this as the case study where every one of those tricks shows up in a single shipped system.

## Why GRPO alone is not enough

Let me state the senior-engineer rule of thumb up front, because the rest of this section is its justification:

> GRPO is a beautiful simplification that pays for itself in the easy regime and silently bankrupts you in the hard one. The hard regime is exactly where frontier reasoning lives.

GRPO — Group Relative Policy Optimization — is the algorithm DeepSeek-R1 used, and its appeal is obvious. You delete the critic. Instead of training a separate value network $V(s)$ to estimate "how good is this state," you sample a *group* of $G$ completions for the same prompt, compute their rewards, and define each completion's advantage as its reward normalized within the group:

$$
A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}
$$

That is it. No value network to train, no value bias to fight, half the memory. For a verifiable-reward task where the reward is a clean binary "is the final answer correct," group normalization gives you a perfectly reasonable signal: completions that beat the group average get pushed up, ones below get pushed down. DeepSeek-R1-Zero-Qwen-32B hit AIME'24 47 with this. It works.

The problem is what GRPO throws away to get there, and the bill comes due in three places that all matter more as chains of thought get longer.

**First, credit assignment is trajectory-final and therefore coarse.** GRPO assigns the *same* normalized advantage to every token in a completion. A 12,000-token chain of thought that arrives at the right answer gets a uniform positive signal across all 12,000 tokens, including the 3,000 tokens of a dead-end branch the model explored and abandoned. There is no per-token notion of "this step was the good one." For short completions this barely matters. For long-CoT — where the entire point is that the model explores, backtracks, and self-corrects — you are smearing a single scalar across a trajectory whose internal structure is the whole game. A value-based method can in principle say "the advantage at token 4,000 is high because the model just had the key insight"; GRPO structurally cannot.

**Second, the group-normalization denominator collapses on easy and hard prompts alike.** If all $G$ completions for a prompt are correct, the rewards are all $1.0$, the mean is $1.0$, the standard deviation is $0$, and every advantage is $0/0$ — in practice clamped to zero. Same for all-wrong groups. That prompt contributed a full forward-and-backward pass of compute and produced *exactly zero gradient*. As the policy improves, more and more of your sampled prompts become all-correct, and your effective batch shrinks under you without warning. This is the single most under-appreciated failure mode in naive GRPO, and DAPO's dynamic sampling exists entirely to fix it.

**Third, the loss is sample-level, which systematically under-weights long sequences.** GRPO computes a per-sample mean over tokens, then averages across samples. That means a 500-token completion and a 10,000-token completion contribute equally to the loss, so each token in the long completion gets $1/20$ the gradient weight of each token in the short one. In a setting where the *correct* behavior is to produce long, careful reasoning, this is exactly backwards: you are down-weighting the tokens you most want to learn from. DAPO's token-level loss fixes this.

Here is the same comparison as a table, because the differences are easiest to hold in your head side by side. The figure under this heading shows the same axes laid out as a matrix.

<!-- FIGSPEC 2
kind: matrix
claim: GRPO, DAPO, and VAPO differ on critic, advantage, clipping, sampling, AIME score, and stability.
caption: Three algorithms across six axes — DAPO and VAPO keep GRPO's group trick but fix its three structural leaks, and VAPO adds a critic back.
nodes:
  - id: r1 | label: "Critic" | color: gray
  - id: r2 | label: "Advantage" | color: gray
  - id: r3 | label: "Clipping" | color: gray
  - id: r4 | label: "Sampling" | color: gray
  - id: r5 | label: "AIME'24" | color: gray
  - id: r6 | label: "Stability" | color: gray
  - id: c1 | label: "GRPO\nno critic" | color: amber
  - id: c2 | label: "DAPO\nvalue-free" | color: blue
  - id: c3 | label: "VAPO\nvalue-based" | color: green
edges:
  - r5 -> c1 | label: "30"
  - r5 -> c2 | label: "50"
  - r5 -> c3 | label: "60.4"
  - r6 -> c1 | label: "fragile"
  - r6 -> c3 | label: "0 crashes"
notes: matrix rows=six axes (critic/advantage/clipping/sampling/AIME/stability), cols=GRPO amber / DAPO blue / VAPO green; fill cells with text from the prose table
-->

![Matrix comparing GRPO, DAPO, and VAPO across critic, advantage, clipping, sampling, AIME score, and stability](/imgs/blogs/seed1-5-thinking-rl-reasoning-vapo-dapo-2.png)

| Axis | GRPO (R1) | DAPO (value-free) | VAPO (value-based) |
| --- | --- | --- | --- |
| Critic / value net | None | None | Yes, pretrained |
| Advantage | Group-normalized, trajectory-final | Group-normalized, trajectory-final | Per-token GAE |
| Clipping | Symmetric $\varepsilon$ | Decoupled 0.2 / 0.28 | Decoupled 0.2 / 0.28 |
| Sampling | Fixed group | Dynamic (drop 0% / 100% groups) | Group 512×16 |
| Loss granularity | Sample-level | Token-level | Token-level |
| AIME'24 (Qwen2.5-32B) | 30 | 50 | 60.4 ± 0.4 |
| Stability | Fragile | Good | Zero crashes |

Notice what *both* DAPO and VAPO inherit from GRPO and what they fix. They both keep the group-sampling trick — it is genuinely good and cheap. They both add decoupled clipping and token-level loss. The fork is the critic: DAPO stays value-free and leans on the group baseline; VAPO brings a value network back and spends its entire technique budget on making that value network not poison the policy. Seed's default for reasoning is VAPO. DAPO is the fallback for regimes where a good value function is hard to learn. That paired-algorithm decision is, to me, the most genuinely novel thing in the whole report, and it only makes sense once you understand both sides deeply. So let us.

Before we dive in, one shared-machinery note that the report is explicit about and that is easy to lose in the per-algorithm detail. Both VAPO and DAPO run on a *common* PPO-style spine with the same set of shared ingredients: decoupled clipping ($\varepsilon_{\text{low}} = 0.2$, $\varepsilon_{\text{high}} = 0.28$), token-level policy-gradient loss, length-adaptive GAE, dynamic sampling, a positive-example NLL term folded in as $\mathcal{L} = \mathcal{L}_{\text{PPO}} + 0.1 \cdot \mathcal{L}_{\text{NLL}}$, and value-pretraining. On top of that spine sits **Online Data Distribution Adaptation**, which we will reach later — it dynamically rebalances the multi-domain RL data to cut cross-domain interference. The reason this matters for reading the two papers correctly: VAPO and DAPO are *not* two unrelated algorithms with overlapping names. They are two instantiations of one carefully-built spine, differing primarily in whether they spend the extra cost of a value network. Understanding the spine once means you understand 80% of both; the remaining 20% is the critic question, which is the whole point of the fork.

## DAPO, technique by technique

DAPO — Decoupled Clip and Dynamic Sampling Policy Optimization — is the value-free side. It keeps GRPO's critic-free structure and patches the three leaks above plus a fourth (length hacking) with four named techniques. I will go through all four with the actual mechanics, because the ablation ladder shows that they are not interchangeable: the order and magnitude of each contribution is itself the lesson.

<!-- FIGSPEC 3
kind: graph
claim: DAPO's four techniques each feed a distinct correction into the same policy-gradient update.
caption: Four independent patches — clip-higher, dynamic sampling, token-level loss, overlong shaping — converge on one policy update, each fixing a different GRPO leak.
nodes:
  - id: a | label: "Clip-Higher\nε 0.2/0.28" | color: green
  - id: b | label: "Dynamic Sampling\ndrop 0%/100%" | color: green
  - id: c | label: "Token-Level Loss\nbatch-mean" | color: green
  - id: d | label: "Overlong Shaping\nL_cache 4096" | color: amber
  - id: e | label: "Policy Update\nAIME 30→50" | color: blue
edges:
  - a -> e | label: "exploration"
  - b -> e | label: "nonzero grad"
  - c -> e | label: "long-seq weight"
  - d -> e | label: "no len-hack"
notes: four green/amber technique nodes on left fan into one blue policy-update node on right; d is amber (cost/tradeoff)
-->

![Graph of DAPO's four techniques feeding into a single policy update](/imgs/blogs/seed1-5-thinking-rl-reasoning-vapo-dapo-3.png)

### Clip-Higher: decouple the PPO clip bounds

Start with the most subtle one. Standard PPO clips the importance-sampling ratio symmetrically: $\text{clip}(\rho, 1-\varepsilon, 1+\varepsilon)$ with a single $\varepsilon$, say $0.2$. The intent is to stop any single update from moving the policy too far. The unintended consequence — and this is the part most people miss — is that a *symmetric* clip caps how fast a *low-probability* token can grow, and that asymmetry kills exploration.

Work the arithmetic. Suppose a token currently has probability $0.01$ under the policy. The clip ceiling $1+\varepsilon = 1.2$ limits the ratio $\rho = \pi_{\text{new}}/\pi_{\text{old}}$ to at most $1.2$, so the new probability can rise to at most $0.01 \times 1.2 = 0.012$. A token that the model is currently unsure about — exactly the kind of token that represents a novel reasoning move the model should be *learning* to favor — can only inch upward by 20% per update. Meanwhile a high-probability token at $0.9$ has plenty of room. The net effect is that the clip systematically suppresses the growth of rare-but-good tokens, the policy's entropy collapses toward its already-confident tokens, and exploration dies. You see this in training curves as entropy falling off a cliff and the policy fixating on a narrow band of completions.

The fix is embarrassingly simple once you see the problem: decouple the bounds. Use a lower clip $\varepsilon_{\text{low}} = 0.2$ to keep the brakes on shrinking good tokens, but raise the *upper* clip to $\varepsilon_{\text{high}} = 0.28$ so rare tokens have more room to grow. The asymmetry is the point — you want to be generous when pushing a low-probability token up and conservative when pulling one down.

```python
import torch

def clip_higher_pg_loss(logp_new, logp_old, advantages,
                        eps_low=0.2, eps_high=0.28):
    """Decoupled-clip policy-gradient loss (DAPO / VAPO 'clip-higher').

    logp_new, logp_old, advantages: shape [num_tokens] (already flattened
    across the batch — see token-level loss below).
    """
    # importance ratio per token
    ratio = torch.exp(logp_new - logp_old)

    # the asymmetry: lower bound 1-eps_low, UPPER bound 1+eps_high
    clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)

    # PPO-style pessimistic min of unclipped vs clipped surrogate
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)

    # token-level mean (NOT sample-then-average) — see next subsection
    return -surrogate.mean()
```

On the DAPO ablation ladder, clip-higher alone moves AIME'24 from 36 to 38 — modest in isolation, but it is the technique that keeps entropy alive long enough for the other three to matter. In the VAPO ablation, removing clip-higher drops the score from 60 to 46, a 14-point hit, because without exploration the value-based method never finds the high-reward trajectories it needs to bootstrap from. The lesson: clip-higher is cheap insurance whose value compounds with the rest of the stack.

The asymmetry in the two ablation numbers — +2 in DAPO, +14 in VAPO — is itself instructive and not a contradiction. In DAPO, the group baseline already provides some exploration pressure because the advantage is computed relative to a *group* of samples, so a few of the techniques partially cover for each other; clip-higher's marginal contribution is small because dynamic sampling and the group structure are doing related work. In VAPO, the critic is a single learned function, and if exploration dies the critic only ever sees a narrow band of trajectories, so its value estimates outside that band are garbage and the policy has no way to escape — clip-higher is load-bearing in a way it is not for the value-free method. This is a recurring pattern in both papers worth internalizing: the *marginal* value of any one technique depends on what else is in the stack, so reading a single ablation number out of context is misleading. The DAPO ladder and the VAPO ladder are not directly comparable rung-for-rung, because they are ablations *off different bases*. What is comparable is the shape: in both, the techniques that protect the model's distribution (exploration, sampling hygiene, value warm-up) dominate the ones that tune the gradient math.

### Dynamic Sampling: drop the zero-gradient groups

This is the leak we flagged in the GRPO section, and DAPO's answer is brutally direct. After sampling a group of $G$ completions for a prompt, look at the group accuracy. If it is $0$ (all wrong) or $1$ (all correct), the group-normalized advantage is zero for every completion, the gradient contribution is zero, and the only thing you have spent is compute. So **drop those prompts and resample** until the batch is full of prompts with $0 < \text{correct} < G$.

```python
def dynamic_sampling(prompts, policy, reward_fn, G=16, target_batch=512):
    """Over-sample, keep only prompts with a non-degenerate group."""
    kept = []
    while len(kept) < target_batch:
        batch = sample_more_prompts(prompts)
        for p in batch:
            comps = policy.generate(p, n=G)
            rewards = [reward_fn(p, c) for c in comps]
            n_correct = sum(r > 0 for r in rewards)
            # keep iff the group has BOTH a winner and a loser
            if 0 < n_correct < G:
                kept.append((p, comps, rewards))
            # else: all-0 or all-G -> zero advantage -> discard
    return kept[:target_batch]
```

The cost is real: you over-sample, so you generate more completions than you keep. But the report is explicit that the *effective* batch — the prompts that actually produce gradient — is what determines training efficiency, and a batch that is 40% zero-gradient prompts is a batch running at 60% efficiency without you knowing it. Dynamic sampling is the single largest jump on the DAPO ablation ladder: it takes the score from 42 (everything else on) to 50 (full DAPO). That 8-point jump is the difference between "competitive" and "beats R1-Zero-Qwen-32B," and it comes entirely from not wasting forward passes on degenerate groups.

There is a second-order effect worth naming. As training progresses and the policy gets stronger, the *fraction* of prompts that come back all-correct rises. Dynamic sampling automatically holds the effective batch constant by sampling more, which means it also acts as an implicit curriculum: the prompts that survive the filter are always the ones at the current frontier of the model's ability. You get curriculum learning for free, as a side effect of a gradient-efficiency fix.

### Token-Level Policy-Gradient Loss

The third technique fixes the long-sequence under-weighting. GRPO's loss is, schematically, `mean over samples of (mean over tokens of per-token loss)`. The inner mean normalizes by sequence length, so every *sample* contributes equally regardless of length, and therefore every *token* in a long sample contributes less. DAPO replaces this with a single mean over all tokens in the batch:

$$
\mathcal{L}_{\text{token}} = \frac{1}{\sum_i |o_i|} \sum_{i} \sum_{t=1}^{|o_i|} \ell_{i,t}
$$

where $|o_i|$ is the length of completion $i$ and $\ell_{i,t}$ is the per-token surrogate loss. Now every token carries the same weight, so a 10,000-token completion contributes 20× the gradient of a 500-token one — proportional to how much reasoning it actually contains. In code this is exactly the `.mean()` over the flattened `[num_tokens]` tensor in the `clip_higher_pg_loss` above; the subtlety is entirely in *not* taking a per-sample mean first.

This one is worth 1 point on the DAPO ladder (41 → 42) but it interacts strongly with overlong reward shaping, because once you weight long sequences correctly, you also have to make sure the model is not rewarded for being long *for its own sake*. Which brings us to the fourth technique.

### Overlong Reward Shaping

If you reward correct answers and you have just told the optimizer to value long sequences, you have built a machine that will happily learn to ramble — pad the chain of thought, hedge, restate — because longer-and-correct scores the same as shorter-and-correct but is easier to stumble into once the model is verbose. Worse, completions that hit the generation length cap get truncated mid-thought and scored as wrong even when the reasoning was fine, injecting noise. DAPO addresses both with a soft length penalty.

Set a hard max length $L_{\max} = 20480$ and a cache buffer $L_{\text{cache}} = 4096$, so the "soft zone" starts at $L_{\max} - L_{\text{cache}} = 16384$. Completions shorter than 16,384 tokens get no penalty. From 16,384 to 20,480 the penalty ramps linearly from $0$ to $-1$. Beyond 20,480 the penalty is a flat $-1$.

```python
def overlong_reward_shaping(length, base_reward,
                            L_max=20480, L_cache=4096):
    """Soft length penalty added to the base (correctness) reward.

    < 16384 tokens: no penalty.
    16384..20480:   linear ramp 0 -> -1.
    > 20480:        flat -1.
    """
    soft_start = L_max - L_cache          # 16384
    if length <= soft_start:
        penalty = 0.0
    elif length <= L_max:
        penalty = -(length - soft_start) / L_cache   # 0 -> -1
    else:
        penalty = -1.0
    return base_reward + penalty
```

The "soft" part matters. A *hard* cutoff (reward 0 below the cap, -1 above) makes the gradient at the boundary discontinuous and teaches the model nothing about the cost of approaching the limit; the linear ramp gives a smooth signal that says "you are getting too long, and it is getting worse the longer you go." On the ladder, the soft overlong penalty is worth 3 points (38 → 41) and the cruder "overlong filter" (just masking truncated samples out of the loss) is worth 6 points off the GRPO baseline (30 → 36). Length hygiene is, somewhat surprisingly, one of the bigger levers in the whole DAPO stack.

### The DAPO ablation ladder, in full

Here is the ladder as a table. Read it top to bottom as "what each technique buys you, in order."

| Configuration | AIME'24 |
| --- | --- |
| GRPO baseline | 30 |
| + overlong filter | 36 |
| + clip-higher | 38 |
| + soft overlong shaping | 41 |
| + token-level loss | 42 |
| + dynamic sampling (full DAPO) | 50 |

Full DAPO on Qwen2.5-32B reaches AIME'24 **50**, beating DeepSeek-R1-Zero-Qwen-32B's **47**, and it does so in roughly **half** the training steps. The two biggest single contributions are length hygiene (the overlong filter, +6) and dynamic sampling (+8) — neither of which is about the policy-gradient math at all. They are both about not feeding the optimizer garbage. That is the DAPO thesis in one sentence: most of long-CoT RL instability is data and length hygiene wearing an algorithm costume.

## VAPO: bringing the critic back without the bias

If DAPO is the value-free side, VAPO — Value-based Augmented Proximal Policy Optimization — is the value-based comeback, and it is Seed's default for reasoning. The premise is contrarian. The whole field moved *away* from critics (that is what GRPO's popularity is) because value networks are notoriously hard to train in the long-CoT regime: rewards are sparse and trajectory-final, sequences are enormous, and a badly-fit value function injects bias that quietly poisons every advantage estimate. VAPO's bet is that a *correctly* trained critic gives you per-token, lower-variance advantages that are worth the trouble — *if* you neutralize the bias with the right scaffolding.

The payoff is stark. On Qwen2.5-32B, VAPO reaches AIME'24 **60.4 ± 0.4** in fewer than 5,000 steps, beating DAPO's 50 and R1-Zero-Qwen-32B's 47 by more than 10 points, with **zero training crashes** across runs. The vanilla PPO baseline — same critic idea, none of the scaffolding — scores **5**. That gap from 5 to 60 is the entire VAPO paper: eight techniques that turn a critic from a liability into the best advantage estimator in the comparison.

<!-- FIGSPEC 4
kind: before-after
claim: Symmetric PPO clipping collapses entropy while decoupled clip-higher preserves exploration.
caption: Left, a symmetric clip caps low-probability token growth and entropy falls off a cliff; right, decoupled bounds let rare tokens grow and exploration survives.
nodes:
  - id: a | label: "Symmetric clip\nε 0.2 both" | color: red
  - id: b | label: "0.01 token\n→ max 0.012" | color: red
  - id: c | label: "Entropy\ncollapse" | color: red
  - id: d | label: "Decoupled clip\n0.2 / 0.28" | color: green
  - id: e | label: "0.01 token\nmore headroom" | color: green
  - id: f | label: "Exploration\npreserved" | color: green
edges:
  - a -> b
  - b -> c
  - d -> e
  - e -> f
notes: left column red (symmetric clip -> entropy collapse), right column green (decoupled -> exploration preserved); before-after split down the middle
-->

![Before-after comparison of symmetric PPO clipping causing entropy collapse versus decoupled clip-higher preserving exploration](/imgs/blogs/seed1-5-thinking-rl-reasoning-vapo-dapo-4.png)

The before-after figure above isolates the clip-higher idea, because it is the technique VAPO and DAPO share and the cleanest illustration of *why* a one-line change to a clip bound is worth 14 points. But VAPO has seven more techniques, and the order of their ablation impact tells you exactly where the critic's risk lives.

<!-- FIGSPEC 5
kind: layered-stack
claim: VAPO's eight techniques stack on a value-pretraining foundation, each rung adding less than the one below.
caption: Read bottom-up: value-pretraining is the foundation that takes AIME from 11 to 60, and every layer above it adds progressively smaller gains.
nodes:
  - id: a | label: "Value-Pretraining\nMC returns (→11)" | color: green
  - id: b | label: "Decoupled-GAE\nvalue λ=1.0 (→33)" | color: blue
  - id: c | label: "Length-Adapt GAE\nα=0.05 (→45)" | color: blue
  - id: d | label: "Clip-Higher\n0.2/0.28 (→46)" | color: blue
  - id: e | label: "Token-Level loss\n(→53)" | color: blue
  - id: f | label: "Positive-NLL μ=0.1\n+ group (→54/55)" | color: blue
  - id: g | label: "VAPO total\nAIME 60.4" | color: green
edges:
  - a -> b
  - b -> c
  - c -> d
  - d -> e
  - e -> f
  - f -> g
notes: vertical layered stack, bottom=value-pretraining (green, widest/foundational), top=VAPO total (green); each rung label shows the AIME score WHEN THAT LAYER IS REMOVED, so smaller drop = higher up
-->

![Layered stack of VAPO's eight value-based techniques on a value-pretraining foundation, each rung adding diminishing gains](/imgs/blogs/seed1-5-thinking-rl-reasoning-vapo-dapo-5.png)

### Value-pretraining: the biggest lever by a mile

The single most important technique in VAPO is the one that addresses the critic's cold-start problem directly. At the start of RL, your value network is random. If you let it learn $V(s)$ from scratch via TD bootstrapping while the policy is also moving, you have two moving targets chasing each other, and early in training the value estimates are noise. Noise in $V$ becomes noise in the advantage $A = R - V$, which becomes a noisy gradient, which destabilizes the policy, which makes the value target worse. This positive feedback loop is why vanilla PPO scores 5.

Value-pretraining breaks the loop before RL starts. Freeze the SFT policy. Roll out a large set of trajectories with it. Compute the **Monte-Carlo returns** — the actual discounted rewards-to-go, no bootstrapping, no bias — for every state. Train the value network to regress those MC returns until it has a warm, low-bias estimate of "how good is each state under the current policy." *Then* start RL with this warmed-up critic.

The ablation is the most dramatic number in either paper: remove value-pretraining and VAPO drops from **60 to 11**. Not 55, not 45 — eleven. A value network that starts cold simply never recovers in the long-CoT regime; the policy has already been knocked off the manifold by noisy early advantages before the critic can catch up. If you take one thing from VAPO, take this: **a value-based method's success is almost entirely determined by how the value network is initialized.** Everything else is refinement around this one decision.

### Decoupled GAE: unbiased value, lower-variance policy

The Generalized Advantage Estimator has a knob $\lambda \in [0,1]$ that trades bias for variance. $\lambda = 1$ is the unbiased Monte-Carlo estimate (high variance); $\lambda \to 0$ is the heavily-bootstrapped, low-variance, high-bias estimate. The standard practice is to use one $\lambda$ everywhere. VAPO's insight is that the value network and the policy want *different* $\lambda$s.

The value network is being trained to predict returns, and for *learning the value function* you want the unbiased target, so use $\lambda_{\text{value}} = 1.0$ — the value loss regresses toward the true MC return. The policy, on the other hand, wants a lower-variance advantage signal, so it uses a *smaller* $\lambda_{\text{policy}}$. Decoupling these — different $\lambda$ for the value update and the advantage computation — is worth a huge amount: removing decoupled-GAE drops VAPO from 60 to **33**. The reason it matters so much is that coupling them forces a bad compromise: either your value network is biased (small shared $\lambda$) or your policy advantages are high-variance (large shared $\lambda$), and in long-CoT both failure modes are fatal.

### Length-adaptive GAE: scale λ with sequence length

Here is the most elegant idea in VAPO. The bias-variance tradeoff in GAE depends on sequence length, because $\lambda$ controls an *effective horizon*: roughly, the advantage at a token is a $\lambda$-weighted sum over future TD errors, and the number of future tokens that meaningfully contribute scales like $1/(1-\lambda)$. For a short completion, a moderate $\lambda$ already reaches the end of the sequence; for a 10,000-token completion, the *same* $\lambda$ only "sees" a small window and you lose long-range credit. So VAPO makes $\lambda$ a function of the policy completion length $l$:

$$
\lambda_{\text{policy}} = 1 - \frac{1}{\alpha \cdot l}, \qquad \alpha = 0.05
$$

```python
def length_adaptive_lambda(seq_len, alpha=0.05):
    """VAPO length-adaptive GAE lambda for the POLICY advantage.

    Longer completions -> lambda closer to 1 -> longer effective
    horizon, so long-range credit is not truncated.
    """
    lam = 1.0 - 1.0 / (alpha * seq_len)
    return max(0.0, lam)   # clamp for very short sequences
```

Plug in numbers. For a 1,000-token completion, $\lambda = 1 - 1/(0.05 \times 1000) = 1 - 1/50 = 0.98$. For a 200-token completion, $\lambda = 1 - 1/10 = 0.9$. For a 20,000-token completion, $\lambda = 1 - 1/1000 = 0.999$. The effective horizon stretches with the sequence so that long chains of thought keep meaningful credit flowing from the final reward all the way back to the early reasoning steps. Remove length-adaptive GAE and VAPO drops from 60 to **45** — a 15-point hit, the third-largest in the ablation, which tells you how badly fixed-$\lambda$ GAE handles the heavy-tailed length distribution of real reasoning traces.

It is worth grounding this in the actual GAE recurrence, because the "effective horizon" hand-wave hides the precise reason length matters. GAE defines the advantage as a discounted, $\lambda$-weighted sum of one-step temporal-difference errors:

$$
A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{k=0}^{T-t-1} (\gamma \lambda)^k \, \delta_{t+k}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

In the long-CoT setting the per-token reward $r_t$ is zero everywhere except the final token, where the verifier's verdict lands. So $A_t$ at an early token is essentially $(\gamma\lambda)^{T-t}$ times the final reward signal, propagated back through the value baseline. The factor $(\gamma\lambda)^{T-t}$ is the thing that decays: if $\gamma\lambda$ is even slightly below 1, then for a token 15,000 positions from the end the weight $(\gamma\lambda)^{15000}$ is numerically zero, and that early reasoning step receives *no* gradient from whether the final answer was right. The model literally cannot learn that its opening move mattered. Length-adaptive GAE pushes $\lambda$ toward 1 exactly as $T$ grows, holding $(\gamma\lambda)^{T-t}$ in a workable range so the terminal reward still reaches the start of a 20,000-token trace. This is why it is a 15-point lever and not a rounding error: on the longest, hardest traces — the ones where the model explores most and the credit-assignment problem is worst — a fixed $\lambda$ silently severs the reward signal from the reasoning that earned it. The decoupling from the value $\lambda = 1.0$ is what makes this safe: the value net still regresses against the unbiased MC return, so pushing the *policy's* $\lambda$ around does not corrupt the baseline the advantage subtracts.

### The remaining VAPO techniques

The other four techniques are smaller but each removes a real failure mode:

- **Token-level policy-gradient loss** — the same fix as DAPO, weighting every token equally. Removing it: 60 → 53.
- **Positive-example LM loss** — VAPO adds an NLL term on correct completions to the objective, $\mathcal{L} = \mathcal{L}_{\text{PPO}} + \mu \cdot \mathcal{L}_{\text{NLL}}$ with $\mu = 0.1$. This is a small supervised pull toward the trajectories the model got right, which stabilizes learning by anchoring the policy to known-good behavior. Removing it: 60 → 54.
- **Group sampling** — VAPO keeps GRPO's grouping (512 prompts × 16 samples), which both supplies a group baseline as a sanity check on the critic and gives dynamic-sampling-style batch hygiene. Removing it: 60 → 55.
- **Self-imitation learning** — replay the best trajectories found so far and imitate them, a form of experience reuse that reinforces the policy's own discovered successes. This is a meaningful piece of why VAPO converges in under 5,000 steps.

The VAPO ablation ladder, sorted by impact:

| Technique removed | AIME'24 drops to |
| --- | --- |
| Value-pretraining | 11 |
| Decoupled-GAE | 33 |
| Length-adaptive GAE | 45 |
| Clip-higher | 46 |
| Token-level loss | 53 |
| Positive-example LM loss | 54 |
| Group sampling | 55 |
| (full VAPO) | 60.4 |
| Vanilla PPO (none of the above) | 5 |

The shape of this ladder is the argument. The top three removals — all about the value function (its initialization, its GAE bias, its length-handling) — account for the lion's share of the gap. The bottom four are refinements. **A value-based method lives or dies by how well it controls value bias and variance**, and VAPO's eight techniques are, with the exception of clip-higher and token-level loss, all in service of exactly that.

### Value-based vs value-free: when to reach for which

Now we can answer the question the paired-algorithm design poses. Value-free methods (GRPO, DAPO) drop the critic and lean on the trajectory-final group baseline. They are cheaper (no second network, half the memory for the value model), simpler, and robust — but their credit assignment is coarse, because the advantage is uniform across the trajectory. Value-based VAPO keeps a critic for per-token, lower-variance advantages and *finer* credit assignment — it can in principle reward the specific reasoning step that mattered — but it pays in value bias, which it neutralizes with value-pretraining plus decoupled and length-adaptive GAE.

The practical decision rule that falls out: reach for VAPO when you can afford to train a good value function and the task rewards fine-grained credit assignment (hard multi-step math, where *which* step was the insight genuinely matters). Reach for DAPO when a good value function is hard to learn — noisier reward, more heterogeneous data, or simply a compute budget that does not stretch to a well-pretrained critic. Seed runs both and routes data to whichever fits, which is the kind of pragmatism you only get from a team that has been burned by trying to make one algorithm do everything. If you want the opposite design philosophy — one dead-simple recipe scaled hard — contrast this with [JustRL](/blog/paper-reading/large-language-model/justrl-scaling-a-1-5b-llm-with-a-simple-rl-recipe), which deliberately strips the stabilizer stack and bets on scale instead.

## Reward modeling: the two-tier verifier and the generative RM

You can have the best policy-optimization algorithm on Earth and it will faithfully optimize whatever your reward says, including the ways your reward is wrong. In long-CoT RL the reward function is not a detail — it is the thing the model spends millions of gradient steps trying to game. Seed splits the reward problem into two regimes, verifiable and non-verifiable, and treats them completely differently.

<!-- FIGSPEC 6
kind: graph
claim: Verifiable rewards use a two-tier verifier and non-verifiable rewards use a generative preference model.
caption: A fast rule-based verifier (82.7% test) is backstopped by a reasoning verifier (99.3% test) that resists reward hacking, while non-verifiable tasks fall back to a pairwise generative RM.
nodes:
  - id: a | label: "Model output" | color: gray
  - id: b | label: "Seed-Verifier\nrule, 82.7% test" | color: amber
  - id: c | label: "Seed-Thinking-Verifier\nreasoning, 99.3%" | color: green
  - id: d | label: "Generative RM\nYES/NO prob" | color: lavender
  - id: e | label: "Verifiable\nreward" | color: blue
  - id: f | label: "Non-verifiable\nreward" | color: blue
edges:
  - a -> b | label: "verifiable"
  - b -> c | label: "hard cases"
  - c -> e
  - a -> d | label: "creative/chat"
  - d -> f
notes: top path two-tier verifier (amber rule -> green reasoning), bottom path lavender generative RM; both produce blue reward nodes
-->

![Tree of the two reward regimes: a two-tier verifiable verifier and a generative reward model for non-verifiable tasks](/imgs/blogs/seed1-5-thinking-rl-reasoning-vapo-dapo-6.png)

### Verifiable: why a reasoning verifier beats rule matching

For STEM, code, and logic, the answer is checkable, so the obvious move is a rule-based check: does the model's final answer match the gold answer? Seed's first tier, **Seed-Verifier**, is exactly this — a rule-based LLM judge that does string/equivalence matching. It hits ~98% accuracy on the training distribution but only **82.7% on the test set**. That 15-point train-test gap is the tell: rule matching is brittle. The model writes the right answer in a form the rule does not recognize ($\frac{1}{2}$ vs $0.5$ vs `1/2`), or it writes a *wrong* answer that happens to surface-match the gold, and your reward signal lies. Worse, a brittle verifier is *hackable*: the policy will learn to produce outputs that trip the verifier into a false positive, which is reward hacking in its purest form.

The second tier, **Seed-Thinking-Verifier**, is the genuinely interesting design. Instead of pattern-matching, it *emits a reasoning trace before its verdict* — it thinks through whether the model's answer is actually equivalent to the gold answer, the way a careful TA grades, and only then outputs YES/NO. This pushes test-set accuracy to **99.3%**, a 16.6-point jump over the rule-based tier, and it closes the train-test gap almost entirely (>99% train, 99.3% test). Crucially, it **resists reward hacking**: because the verdict comes from reasoning about equivalence rather than surface matching, the policy cannot find a cheap surface trick that fools it. Seed-Thinking-Verifier is the one they actually rely on for the reward signal; Seed-Verifier is the fast first pass.

This is a real conceptual shift and it is one of the report's genuine contributions. The reward model for a *reasoning* RL run is itself a reasoning model. You are using chain-of-thought to verify chain-of-thought. It is more expensive per judgment, but in a regime where a hackable reward silently corrupts every subsequent gradient, the robustness is worth far more than the latency.

It is worth dwelling on *why* reasoning before verdict buys robustness, because the mechanism is not obvious. A rule-based matcher makes its decision from a fixed, shallow comparison — does the extracted answer string equal the gold string under some normalization? That decision boundary is exposed and therefore exploitable: the policy, under enough gradient pressure, will find the inputs that sit on the favorable side of the boundary without actually being correct, because the boundary is a simple function the optimizer can reverse-engineer. A reasoning verifier moves the decision boundary into a *deep* computation — it reads the model's answer, reads the gold answer, and reasons about whether they are mathematically equivalent (is $\frac{\sqrt 2}{2}$ the same as $\frac{1}{\sqrt 2}$? is this matrix the transpose the problem asked for?). That computation is far harder for the policy to game, because gaming it would require producing an answer that is genuinely equivalent — which is just being correct — or fooling a reasoner, which is much harder than fooling a regex. The 16.6-point test-accuracy jump (82.7 → 99.3) is the measurable shadow of moving the boundary from shallow to deep. The cost is a full verifier forward pass per judgment instead of a string compare, but in RL you are spending orders of magnitude more compute on the *policy* rollouts anyway, so the verifier overhead is in the noise relative to the catastrophe of a hacked reward. This is the trade the report makes deliberately and the one most worth copying.

| Verifier | Mechanism | Train acc | Test acc | Reward-hacking resistance |
| --- | --- | --- | --- | --- |
| Seed-Verifier | Rule-based LLM judge | ~98% | 82.7% | Low (surface-matchable) |
| Seed-Thinking-Verifier | Emits reasoning, then verdict | >99% | 99.3% | High (reasons about equivalence) |

### Non-verifiable: the pairwise generative reward model

For creative writing, open-ended chat, and the 100k non-verifiable instances, there is no gold answer to check against. Here Seed uses a **pairwise generative reward model** trained on human preferences. Given two responses, it predicts which a human would prefer, surfacing the signal as a YES/NO probability. This is what lets Seed1.5-Thinking generalize *beyond* reasoning — the +8% human-eval win rate over DeepSeek-R1 on non-reasoning tasks comes from RL'ing against this preference model, not just from the math RL.

But the report is honest that this is the soft spot. General reward modeling for non-verifiable domains is an **open problem**, and it is the thing that caps how far the generalization goes. A preference model is a learned approximation of human taste, and it is far more hackable than a reasoning verifier: there is no ground truth to anchor it. This is exactly why the model's factuality is weak (SimpleQA 12.9, which we will get to) — a preference model rewards *plausible-sounding* answers, and plausible is not the same as true. The two-tier verifier solved reward hacking for verifiable tasks; nobody has solved it for non-verifiable ones, and Seed does not pretend otherwise.

## Online Data Distribution Adaptation

Here is a problem you only hit when you run RL over multiple domains at once, and it is the kind of thing that does not show up in single-domain papers. You are training on STEM, code, and logic-puzzle data simultaneously. These domains have different reward scales, different difficulty curves, and different rates of improvement. If you hold the data mix fixed, you get **cross-domain interference**: the model starts over-fitting whichever domain is currently giving the cleanest gradient, the others stall, and the easy domains saturate (all-correct groups, zero gradient) while the hard ones are still struggling and dominating the loss in a destabilizing way.

Seed's answer is **Online Data Distribution Adaptation** — dynamically rebalancing the multi-domain RL data mix *during* training to cut interference. As a domain saturates (its prompts increasingly come back all-correct), its weight in the mix is reduced; as a domain shows headroom, its weight goes up. The effect is to keep every domain in the productive middle band — the same $0 < \text{correct} < G$ sweet spot that dynamic sampling targets within a domain — but now at the level of *which domains* feed the batch. It is dynamic sampling's logic lifted one level up, from prompts to data sources.

This pairs naturally with the streaming infra. Because you are constantly measuring per-domain accuracy to drive the rebalancing, you need fresh rollouts and a system that can act on the measurements without stalling the trainer. It is also one of the report's quieter novelties: most RL recipes treat the data mixture as a hyperparameter you set once. Seed treats it as a control variable adjusted online, which is the natural thing to do once you accept that the *point* of RL is to keep the model at the edge of its competence in every domain at once. The connection to broader curriculum-and-distribution ideas is worth chasing if you have read our piece on [bridging offline and online RL for LLMs](/blog/paper-reading/large-language-model/bridging-offline-and-online-reinforcement-learning-for-llms).

There is a subtlety in *how* you drive the rebalancing that the report handles carefully and that naive implementations get wrong. The obvious signal is per-domain reward, but reward is the wrong control variable: a domain can have high average reward simply because it is easy, which is exactly the saturated case you want to *down*-weight, not up-weight. The right signal is closer to per-domain *learnability* — the rate at which the domain still produces non-degenerate groups ($0 < \text{correct} < G$) and the rate at which its accuracy is still moving. A domain that is 95% correct and flat is saturated; a domain at 40% and climbing is where the gradient should go. So Online Data Distribution Adaptation is really tracking the gradient of accuracy, not the level, and rebalancing toward domains with the steepest remaining slope. This is also why it cannot be a fixed schedule set in advance: the slope of each domain depends on the current policy, which is changing, so the optimal mix at step 1,000 is not the optimal mix at step 4,000. Treating the mix as static is equivalent to assuming all domains saturate at the same rate, which they emphatically do not — code, competition math, and logic puzzles have completely different difficulty curves under RL. The mechanism is the multi-domain analogue of dynamic sampling, and the two together form a tidy two-level hygiene system: dynamic sampling keeps the *prompts within a batch* productive, Online Data Distribution Adaptation keeps the *domains feeding the batch* productive. Both are answers to the same underlying disease — gradient wasted on prompts and domains the model has already mastered or cannot yet touch.

## The Streaming Rollout System and the rest of the infra

Algorithms get the abstracts; infrastructure gets the model shipped. The dirty secret of long-CoT RL is that the rollout phase — generating completions — is the bottleneck, and it is a bottleneck with a particularly nasty shape: **completion lengths are wildly variable**. Some prompts produce a 500-token answer; some produce a 20,000-token chain of thought. In a synchronous RL loop, the trainer waits for the *entire* batch of rollouts to finish before it can do a gradient step, which means it waits for the single longest completion in the batch. This is the GPU straggler problem, and with long-CoT it is brutal: a handful of 20k-token generations hold the whole cluster hostage while most GPUs sit idle.

<!-- FIGSPEC 7
kind: timeline
claim: Decoupling rollout from model-version evolution with streaming units yields about a 3x speedup.
caption: Synchronous RL stalls on the longest completion in each batch; the streaming system decouples generation from training and mitigates stragglers for roughly a 3x faster iteration.
nodes:
  - id: a | label: "Sync RL\nwait longest CoT" | color: red
  - id: b | label: "GPU idle\nstraggler stall" | color: red
  - id: c | label: "Streaming units\ndecouple rollout" | color: green
  - id: d | label: "α on/off-policy\nratio ∈[0,1]" | color: blue
  - id: e | label: "~3× faster\niteration" | color: green
edges:
  - a -> b | label: "blocks"
  - b -> c | label: "fix"
  - c -> d
  - d -> e | label: "3×"
notes: timeline left-to-right; red sync/straggler phase, then green streaming phase; α node blue in the middle
-->

![Before-after of synchronous RL versus a streaming rollout system that decouples generation from training for roughly a 3x speedup](/imgs/blogs/seed1-5-thinking-rl-reasoning-vapo-dapo-7.png)

### Decoupling model-version evolution from rollout

The **Streaming Rollout System (SRS)** decouples model-version evolution from rollout. Instead of the trainer waiting for a synchronized batch, generation runs continuously on standalone streaming-compute units, and completions stream back as they finish. A straggler 20k-token generation no longer blocks the cluster — the trainer keeps consuming the steady stream of completed rollouts and the slow one simply arrives later. This is the core of the ~**3× faster iteration** versus a synchronous loop.

Decoupling generation from training raises an immediate question: the completions arriving now were generated by a slightly *older* policy version than the one you are about to update, so they are partially off-policy. Seed exposes this as an explicit, tunable parameter $\alpha \in [0,1]$ that sets the **ratio of on-policy to off-policy samples** in each update. At $\alpha = 1$ you insist on fully on-policy rollouts (slower, theoretically cleaner); lower $\alpha$ admits more stale, off-policy rollouts (faster, but you rely on the clip-higher importance-sampling machinery to correct for the policy mismatch). Making this an explicit knob — rather than an accident of system timing — is a genuine contribution, because it turns the on/off-policy tradeoff into something you can dial per run instead of something the scheduler decides for you. The connection to importance sampling under policy lag is exactly the territory covered by [GSPO](/blog/paper-reading/reinforcement-learning/group-sequence-policy-optimization), and SRS is essentially the infra that makes a controlled amount of off-policyness safe.

The reason $\alpha$ is safe to push below 1 is the same importance ratio $\rho = \pi_{\text{new}}/\pi_{\text{old}}$ that PPO already computes. A completion generated by an old policy version is corrected for by that ratio inside the surrogate loss — that is literally what PPO's importance weighting is *for*. The danger is that as the gap between the generating policy and the current policy widens, the ratio's variance explodes, and a handful of completions with extreme $\rho$ can dominate and destabilize the update. The clip is what bounds this: any completion whose ratio drifts outside $[1 - \varepsilon_{\text{low}}, 1 + \varepsilon_{\text{high}}]$ is clamped, so a very stale completion contributes a clipped, bounded gradient rather than an explosive one. This is the deep reason the clip-higher machinery and the streaming system are co-designed: the clip is not only an exploration tool, it is the safety rail that makes controlled off-policyness — and therefore the 3× speedup — possible at all. Set $\alpha$ too low and even the clip cannot rescue you, because too many completions are too stale and the *clipped* gradient is uniformly attenuated, so you train slowly on a heavily-clipped batch. Set $\alpha = 1$ and you pay the full straggler tax. The right operating point is a per-run tuning decision, which is precisely why exposing it as a parameter — rather than burying it in scheduler timing — is the contribution. It turns a system accident into a hyperparameter you can sweep.

### The rest of the stack

The supporting cast is a who's-who of large-scale training infra, and it is worth listing because each piece addresses a specific failure mode of long-CoT RL at scale:

- **Hybrid parallelism engine.** TP (tensor parallel) and CP (context parallel) for the attention/dense path, EP (expert parallel) for the MoE experts, FSDP for sharding — assembled with **HybridFlow on Ray**. Different parts of the model want different parallelism, and forcing one scheme everywhere wastes either memory or bandwidth.
- **Dynamic FP8 rollout quantization.** Rollout (generation) is throughput-bound, not precision-critical, so quantizing the rollout model to FP8 dynamically buys speed where accuracy matters least. Training stays in higher precision.
- **KARP sequence-length balancing.** With completion lengths spanning 40× ranges, naive batching wastes compute on padding. KARP balances sequence lengths across the batch so GPUs are not padding-bound.
- **ByteCheckpoint** for fast, reliable checkpointing — non-negotiable when a run takes thousands of steps and a crash without a recent checkpoint costs days.
- **AutoTuner** for automatically searching the parallelism/throughput config space rather than hand-tuning it.

The blog reports the result of all this as roughly **95% training stability** and a claimed **50% lower unit inference cost** versus DeepSeek-R1 — the latter being the direct dividend of the 20B-active MoE design. Inference cost scales with *active* parameters, and 20B active vs 37B active is most of where that 50% comes from. This is the business case for the whole architecture choice: a smaller active footprint that still reaches the frontier is worth more than a bigger model that ties on benchmarks, because you pay for the active parameters on every single token you ever serve.

The architecture details, for completeness: MoE with 20B active / 200B total, TP for the dense layers, EP for the MoE layers, SP/CP for long context. SFT truncates at ~32k tokens (the maximum context is at least 32k; the report does not cleanly state a larger figure). The SFT cold-start uses a peak learning rate of 2e-5 decaying to 2e-6 over the 400k-instance run.

## The architecture and SFT cold-start, examined

It is tempting to skip past "20B-active MoE" as a spec-sheet line, but the architecture choice is load-bearing for the whole story, and it interacts with the RL stack in ways worth pulling apart.

**Why 20B active is the headline, not 200B total.** A dense 200B model and a 200B-total MoE that activates 20B per token have radically different cost profiles. Training cost is dominated by total parameters and the optimizer state you carry for all of them — that is where the 200B figure lives. But *inference* cost, the number you pay forever on every served token, scales with the *active* parameters that actually do the matmul. DeepSeek-R1 activates 37B per token from a 671B total; Seed1.5-Thinking activates 20B from 200B. So Seed carries less than a third of R1's total weights and runs roughly half the active FLOPs per token, which is exactly the lever behind the claimed 50% lower unit inference cost. The strategic bet is sharp: a *sparser* model that reaches the same reasoning frontier is strictly more valuable than a denser one that ties, because the gap compounds across every inference request for the model's entire deployment life. The whole point of the RL stabilizer stack is to make that smaller active model *reach* the frontier despite having less per-token compute to spend — the algorithm work is buying back the capacity the sparsity gave away.

**Why MoE makes RL harder, and why the parallelism is heterogeneous.** MoE is not free in RL. Routing is discrete, so the active expert set shifts as the policy updates, and a value network has to estimate $V(s)$ for states whose underlying compute graph is changing token to token — one more reason the critic needs the value-pretraining warm-up to stay grounded. The parallelism layout reflects the model's heterogeneity: tensor parallel (TP) and context parallel (CP) for the dense attention path, where the bottleneck is the long context window; expert parallel (EP) for the MoE layers, where the bottleneck is fitting 200B of experts across devices and routing tokens to the right shard; FSDP to shard whatever remains. Forcing one scheme everywhere would either blow memory (TP everything) or waste interconnect bandwidth (EP everything), which is why HybridFlow on Ray exists: it lets different sub-graphs of the same model run under different parallelism simultaneously. This is genuinely hard systems work, and it is the unglamorous reason a 200B-total MoE can be RL'd at all.

**The SFT cold-start is not a throwaway.** The 400k-instance cold-start — 300k verifiable STEM/code/logic plus 100k non-verifiable creative/chat — is doing two jobs at once. The verifiable majority gives RL a policy that already produces parseable, checkable reasoning so the verifier reward is meaningful from step one; without it, early RL would spend its budget teaching basic format compliance instead of reasoning. The non-verifiable quarter is what seeds the generalization: it ensures the cold-start policy has not collapsed onto a narrow math-only distribution before the preference-model RL even starts, which protects the entropy the later non-reasoning RL needs. The learning-rate schedule (peak 2e-5 decaying to 2e-6) is deliberately gentle — a high terminal LR would leave the policy too plastic and prone to the kind of distribution collapse the RFT-as-init finding warns about. The cold-start is, in effect, the floor that the entire stabilizer stack is built to raise without cracking.

## Results, and what generalization beyond math actually buys

Now the scoreboard. Math benchmarks are averaged over 32 responses, which matters — a single-sample AIME number is noise, and averaging over 32 is the difference between a real measurement and a lucky run.

<!-- FIGSPEC 8
kind: matrix
claim: Seed1.5-Thinking lands near the frontier on reasoning and leads all peers on ARC-AGI.
caption: Across six benchmarks Seed1.5-Thinking trails the very top on hard math but leads every listed peer on abstract reasoning while running 20B active.
nodes:
  - id: c1 | label: "Seed1.5\n20B active" | color: green
  - id: c2 | label: "o3-mini-high" | color: lavender
  - id: c3 | label: "DeepSeek-R1\n37B active" | color: lavender
  - id: c4 | label: "Gemini 2.5 Pro" | color: lavender
  - id: r1 | label: "AIME'24" | color: gray
  - id: r2 | label: "GPQA" | color: gray
  - id: r3 | label: "ARC-AGI" | color: gray
edges:
  - r1 -> c1 | label: "86.7"
  - r1 -> c4 | label: "92.0"
  - r2 -> c1 | label: "77.3"
  - r3 -> c1 | label: "39.9 lead"
  - r3 -> c4 | label: "27.6"
notes: matrix rows=benchmarks (AIME/GPQA/ARC-AGI), cols=Seed green / three lavender peers; highlight Seed leading ARC-AGI in green
-->

![Matrix benchmark table comparing Seed1.5-Thinking against o3-mini-high, DeepSeek-R1, and Gemini 2.5 Pro](/imgs/blogs/seed1-5-thinking-rl-reasoning-vapo-dapo-8.png)

| Benchmark | Seed1.5 | o3-mini-high | DeepSeek-R1 | Gemini 2.5 Pro | Grok 3 |
| --- | --- | --- | --- | --- | --- |
| AIME'24 | 86.7 | 87.3 | 79.8 | 92.0 | 83.9 |
| AIME'25 | 74.0 | 86.5 | 65.0 | 86.7 | 77.3 |
| GPQA Diamond | 77.3 | 79.7 | 71.5 | 84.0 | 80.2 |
| Codeforces (pass@8) | 55.0 | 67.5 | 45.0 | 56.3 | — |
| LiveCodeBench v5 | 64.9 | 74.1 | 64.3 | 70.4 | 70.6 |
| ARC-AGI | 39.9 | 25.8 | 18.3 | 27.6 | 31.9 |
| SimpleQA | 12.9 | 13.8 | 30.1 | 52.9 | 43.6 |

Read this honestly. On AIME'24, Seed1.5-Thinking's 86.7 is within a point of o3-mini-high (87.3), well ahead of DeepSeek-R1 (79.8), and behind Gemini 2.5 Pro (92.0). On AIME'25 — a harder, fresher set — the gap to the top widens to 12 points, which is the report's own admission that the hardest math is where it still trails. GPQA Diamond 77.3 is competitive but not leading. On code, Codeforces 55.0 pass@8 beats R1 (45.0) and edges Gemini (56.3) but trails o3-mini-high (67.5).

Then look at **ARC-AGI: 39.9, leading every listed peer** — o3-mini-high 25.8, R1 18.3, Gemini 27.6, Grok 31.9. ARC-AGI is the abstract-reasoning benchmark designed specifically to resist memorization and reward genuine generalization. Seed leading it by 8 points over the next-best is the most interesting single number in the report, because it suggests the RL stack taught *transferable* reasoning rather than benchmark-shaped pattern-matching. Combined with the +8% human-eval win rate over R1 on non-reasoning tasks and IFEval 87.4 on instruction-following, the story is "this model reasons well and the reasoning generalizes" — which is exactly what you would hope a well-stabilized RL run would produce, as opposed to a run that overfit the math reward.

This is the part worth sitting with, because it is the most defensible *interpretation* the report supports and the one easiest to get wrong. There are two ways a model can score 86.7 on AIME'24. One is to learn the *shape* of competition-math problems so well that it pattern-matches solutions — which produces a model that is great at AIME and worthless at anything that does not look like AIME. The other is to learn *reasoning* — search, backtracking, self-correction, decomposition — which happens to be very useful for AIME but also for ARC-AGI puzzles it has never seen and for open-ended chat. The discriminator between these two stories is exactly a held-out abstract-reasoning benchmark like ARC-AGI, and Seed leading it while *trailing* on the most AIME-like tasks (AIME'25) is the signature of the second story, not the first. A model that had overfit the math reward would do the reverse: dominate the in-distribution math and collapse on ARC-AGI. The +8% non-reasoning win rate is the second piece of corroborating evidence — overfit reasoning does not transfer to creative writing and chat, but *learned* reasoning, married to the non-verifiable preference RL, does. The honest caveat is that benchmarks are imperfect and ARC-AGI is one data point; but as evidence that the stabilizer stack bought transferable capability rather than a leaderboard number, it is the strongest single signal in the report. For practitioners the implication is concrete: if your reasoning-RL run improves your target benchmark but *degrades* held-out generalization, you have overfit the reward, and the fix is upstream — more entropy protection, a less hackable verifier, broader data — not more steps.

The weak spot is unmissable: **SimpleQA 12.9**, against R1's 30.1, Grok's 43.6, and Gemini's 52.9. Factuality is bad. This is the direct fingerprint of the non-verifiable reward problem. A preference-based generative RM rewards plausibility, and a reasoning model with weak factual grounding will confidently reason its way to a wrong fact. The two-tier reasoning verifier fixed reward hacking for *checkable* answers; there is no equivalent ground-truth anchor for "is this fact true," and SimpleQA is where that gap shows. Seed also released a new harder math benchmark, **BeyondAIME**, precisely because AIME is saturating and they wanted headroom to measure against.

## The RFT-as-init finding, and why it matters

One short but important finding deserves its own section because it is counter-intuitive and practitioners keep getting it wrong. A natural idea is to warm up RL with **Rejection-sampling Fine-Tuning (RFT)** — generate completions, keep the correct ones, SFT on them, and *then* start RL from that stronger checkpoint. It feels like it should help: you are starting RL from a better policy.

Seed's finding is that **RFT-as-RL-init saturates faster but ends LOWER.** The RFT warm-up gives you a head start in the first few hundred steps, but the resulting policy has lower entropy and a narrower distribution — RFT, by construction, distills the model onto its own already-confident correct answers, which shrinks exploration. RL then runs out of room: there is less for it to discover because the policy has already collapsed onto a narrow band. The team's conclusion is to *avoid* RFT-as-init and start RL from the broader SFT cold-start instead, accepting slower early progress for a higher ceiling.

This rhymes with the clip-higher finding and the whole entropy theme of the report. The recurring lesson across DAPO, VAPO, and this RFT result is the same: **anything that prematurely narrows the policy's distribution costs you the ceiling, even when it helps the floor.** Exploration is the scarce resource in long-CoT RL, and almost every stabilizer trick in the paper is, at bottom, a way to spend the compute budget without spending the entropy budget. Hold that thought — it unifies more of the report than any single equation does.

## Case studies: the failure modes these tricks defend against

The fastest way to internalize a stabilizer stack is to walk through the specific disaster each piece prevents. These are framed as the failure modes the report's mechanisms exist to defend against — the war stories implied by the ablation numbers.

### 1. The all-correct batch that trains on nothing

You launch a GRPO run on a math set your model is already decent at. Loss looks fine, then plateaus early. The wrong hypothesis is "the model has converged." The actual root cause: 45% of your sampled prompts come back all-correct, contribute zero advantage, and your effective batch is 55% of nominal — and shrinking every step as the model improves. The fix is dynamic sampling: over-sample, keep only $0 < \text{correct} < G$ groups, hold the effective batch constant. On the DAPO ladder this is the +8 jump from 42 to 50. The lesson: in RL, "loss plateaued" and "the model converged" are different claims, and the gap between them is often a batch quietly going degenerate.

### 2. Entropy collapse under a symmetric clip

A PPO run looks healthy for 800 steps, then entropy nosedives and the policy starts emitting the same handful of completions. The wrong hypothesis is "learning rate too high." The actual root cause is the symmetric clip: a $0.01$-probability token can only grow to $0.012$ per update, so rare-but-promising tokens never get a chance to rise, and the policy collapses onto its already-confident tokens. The fix is clip-higher — raise the upper bound to 0.28 while keeping the lower at 0.2. In VAPO's ablation this one technique is worth 14 points (60 → 46). The lesson: an entropy cliff is usually a clipping-geometry problem, not a learning-rate problem, and the two have opposite fixes.

### 3. The cold critic that poisons the policy

You bring a critic back for finer credit assignment, initialize it randomly, and start PPO. The score lands at 5 and never moves. The wrong hypothesis is "value-based methods just do not work for long-CoT." The actual root cause: a random critic produces noisy advantages, the noisy gradient knocks the policy off the manifold in the first few hundred steps, and the value target gets worse from there — a death spiral. The fix is value-pretraining: regress the critic on Monte-Carlo returns from the frozen SFT policy *before* RL starts. This is the 11 → 60 difference, the largest single lever in either paper. The lesson: a value-based method's outcome is set at initialization; a cold critic is not a slow start, it is a different (failed) basin.

### 4. The rambling model that games the length reward

You add token-level loss to weight long sequences properly, and within a few hundred steps your average completion length doubles with no accuracy gain. The wrong hypothesis is "the model is reasoning more carefully." The actual root cause: longer-and-correct scores identically to shorter-and-correct, so the optimizer drifts toward verbosity because it is an easier basin to stumble into once the model is already wordy. The fix is overlong reward shaping — a soft linear penalty from 16,384 to 20,480 tokens. On the DAPO ladder, length hygiene (overlong filter + soft shaping) is worth 9 points combined. The lesson: any time you up-weight a behavior, check that you have not made a degenerate version of it free.

### 5. The verifier that rewards the wrong answer

Your rule-based verifier reports 98% training accuracy, but the RL'd model's *real* accuracy on held-out problems is far lower, and worse, the model has learned to format answers in ways that trip false positives. The wrong hypothesis is "the model is overfitting the training problems." The actual root cause: the rule-based verifier is only 82.7% accurate on test, it is surface-matchable, and the policy found the surface tricks — textbook reward hacking. The fix is Seed-Thinking-Verifier, which reasons about answer equivalence before emitting a verdict (99.3% test) and cannot be fooled by formatting. The lesson: a hackable reward does not produce obviously-broken loss curves; it produces a model that scores great against its own broken judge and fails everywhere else.

### 6. The straggler completion holding the cluster hostage

Your synchronous RL loop's throughput is a third of what the GPU-hours predict. The wrong hypothesis is "the model is too big, we are memory-bound." The actual root cause: completion lengths span 40× (500 to 20,000 tokens), and the synchronous trainer waits for the longest completion in every batch, so most GPUs idle while a few stragglers finish their 20k-token chains. The fix is the Streaming Rollout System: decouple generation from training, stream completions as they finish, mitigate stragglers with standalone streaming units — ~3× faster. The lesson: in long-CoT RL the cost distribution is heavy-tailed, and any synchronous barrier pays the tail on every step.

### 7. The fixed data mix that lets easy domains starve hard ones

You train RL over STEM, code, and logic with a fixed 50/30/20 mix. STEM saturates first (all-correct groups), but it still occupies 50% of every batch contributing nothing, while the logic puzzles — still hard, still learnable — get only 20% of the gradient. The wrong hypothesis is "logic is just intrinsically hard for this model." The actual root cause: the fixed mix lets a saturated domain crowd out a learnable one. The fix is Online Data Distribution Adaptation: rebalance the mix online, downweighting saturated domains and upweighting ones with headroom. The lesson: multi-domain RL has a batch-allocation problem on top of the per-prompt one, and a static mix solves neither.

### 8. The RFT warm-up that caps the ceiling

You warm up RL with rejection-sampling fine-tuning to get a stronger starting policy. The first few hundred steps look great — faster than starting from raw SFT. Then it plateaus below where the from-SFT run eventually lands. The wrong hypothesis is "RFT init is strictly better, we just need more steps." The actual root cause: RFT distilled the model onto its own confident correct answers, collapsing entropy, so RL has less to explore and a lower ceiling. The fix is to skip RFT-as-init and start from the broader SFT cold-start. The lesson — the same one that runs through the whole report: in long-CoT RL, the floor and the ceiling are governed by different things, and most shortcuts that raise the floor lower the ceiling by spending entropy you cannot get back.

### 9. The sample-level loss that under-trains the careful answers

Your math RL run improves, but inspection shows the model gets *short* problems right and keeps fumbling the ones that need long, multi-step derivations — even though those long derivations are exactly the behavior you are trying to instill. The wrong hypothesis is "the hard problems are just harder, give it time." The actual root cause is the sample-level loss: GRPO averages per-token loss *within* each sample before averaging across samples, so a 10,000-token correct derivation contributes the same total gradient as a 500-token one, meaning each token of the long answer gets one-twentieth the learning signal. You are systematically under-training the tokens you most want to reinforce. The fix is token-level loss — one mean over all tokens in the batch — which restores proportional weight to long sequences. On the DAPO ladder it is "only" +1 (41 → 42), but that number understates it: token-level loss is a *precondition* for overlong reward shaping to work, because once long sequences carry their fair gradient weight you suddenly *need* the length penalty to stop the model from gaming length. The lesson: a loss-normalization choice that looks like a numerical detail can quietly decide which capabilities your model is allowed to learn.

### 10. The model that reasons its way to a confident falsehood

Your reasoning model crushes math and code but bombs SimpleQA at 12.9, confidently asserting wrong facts with fluent supporting reasoning. The wrong hypothesis is "it needs more knowledge, scale the pretraining data." The actual root cause is the reward signal for non-verifiable tasks: the generative preference RM rewards responses humans *prefer*, and humans prefer confident, well-structured, plausible answers — which is orthogonal to whether the answer is true. There is no ground-truth anchor the way there is for a checkable math answer, so the model learns the surface features of trustworthiness without the substance. The "fix" is honest acknowledgment rather than a clean solution: general reward modeling for non-verifiable domains is an open problem, and until it is solved, factuality will lag in any model RL'd primarily against preference. The lesson: reward hacking is not always a flagrant exploit; sometimes it is the model faithfully optimizing a reward that is a sincere but imperfect proxy for what you actually wanted.

## Limitations the authors actually acknowledge

A good paper-reading post does not just relay the wins; it relays the honest self-assessment, and Seed's is unusually candid:

- **Hard-math gap.** On AIME'25 (74.0) and BeyondAIME, Seed1.5-Thinking trails o3-mini-high and Gemini 2.5 Pro by 12+ points. The frontier on the *hardest* math is not yet reached; the model is near-frontier, not frontier, on this axis.
- **Non-verifiable reward modeling is unsolved.** The generative preference RM is the soft spot, and the authors say so. It caps how far the non-reasoning generalization can go and is far more hackable than the reasoning verifier.
- **Weak factuality.** SimpleQA 12.9 is the visible symptom of the reward-modeling gap. A preference model rewards plausible, not true.
- **The RFT-as-init tradeoff** is a real constraint: the obvious efficiency shortcut hurts the ceiling, so they leave performance on the table early to protect the endgame.
- **Long-CoT RL is unstable without the full stabilizer stack.** Vanilla PPO scores 5. Naive GRPO scores 30. The entire 30→60 climb is stabilizer engineering, which means *reproducing* these results requires reproducing the whole stack — there is no single trick to copy.

## Takeaways for practitioners

If you are building a reasoning-RL system, here is what Seed1.5-Thinking actually teaches, stripped to the load-bearing claims:

> Long-CoT RL stability is not one problem with one fix. It is four problems — exploration, credit assignment, length hygiene, and reward integrity — and you need a defense for each.

**Match the algorithm to the data regime.** The paired VAPO/DAPO design is the report's most transferable idea. Value-based VAPO when you can train a good critic and the task rewards fine credit assignment; value-free DAPO when a value function is hard to learn. Do not force one algorithm to cover both.

**Protect entropy above all else.** Clip-higher, no RFT-as-init, dynamic sampling — three of the biggest levers are all, fundamentally, about not collapsing the policy's distribution prematurely. Exploration is the scarce resource. Spend compute, not entropy.

**For value-based methods, initialization is destiny.** Value-pretraining on MC returns is the 11→60 lever. A cold critic does not recover. If you bring a critic back, warm it on the frozen SFT policy first or do not bother.

**Make your reward a reasoner.** Rule-based verifiers are 82.7% on test and hackable; a reasoning verifier that thinks before it judges hits 99.3% and resists hacking. For *verifiable* tasks this is a solved problem and you should use the solution. For non-verifiable tasks, accept that you are operating on unsolved ground and that your factuality will pay for it.

**Decouple rollout from training.** The straggler problem is intrinsic to heavy-tailed completion lengths. A streaming system with an explicit on/off-policy $\alpha$ knob turns the 3× speedup from an infra heroics into a dialable parameter — and pairs naturally with the importance-sampling corrections you already have from clip-higher.

The deepest lesson is the one the report never states outright but every ablation confirms: in long-CoT RL, almost every failure mode is a *distribution-collapse* failure mode, and almost every fix is a way to keep the policy exploring while still converging. Seed1.5-Thinking reaches the frontier at 20B active not because of one brilliant algorithm but because it has a named, ablated defense against each of the four ways the distribution can collapse. That is what "stabilized long-CoT RL" actually means, and it is why a report that looks like a benchmark flex is really a manual for not falling off the unicycle.

## Where this sits in the literature

It helps to place Seed1.5-Thinking against its neighbors, because the design choices are sharper when you see what they are reacting to. The lineage starts with [DeepSeek-R1](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning), which proved pure RL could incentivize reasoning and made GRPO the default. Seed's whole report can be read as the answer to one question R1 left open: *what do you add to GRPO when the easy regime ends?* The answer is the four DAPO tricks and the eight VAPO levers — and the meta-answer is "stop trying to make one algorithm do everything."

Read alongside the broader [tricks-or-traps deep-dive on RL for LLM reasoning](/blog/paper-reading/large-language-model/part-i-tricks-or-traps-a-deep-dive-into-rl-for-llm-reasoning), Seed1.5-Thinking is the case where nearly every individual trick that survey catalogs — clip asymmetry, dynamic sampling, length penalties, value warm-up — shows up together in a shipped system with ablation numbers attached, which is rare. Most papers introduce one trick; this report introduces a *stack* and tells you the marginal value of each rung.

The contrast with [JustRL](/blog/paper-reading/large-language-model/justrl-scaling-a-1-5b-llm-with-a-simple-rl-recipe) is the most instructive. JustRL bets that a dead-simple recipe scaled hard beats a complicated one — the opposite philosophy. Both can be right depending on scale and regime: JustRL's simplicity wins when the model is small and the task is narrow enough that the distribution does not collapse; Seed's stack is the cost of admission once you are training a 200B-total MoE on heterogeneous multi-domain data where collapse is the default outcome. The disagreement is really about *when* the stabilizer stack starts paying for its complexity, and the honest answer is "later than you think for small models, exactly here for frontier ones."

On the policy-optimization machinery itself, [GSPO](/blog/paper-reading/reinforcement-learning/group-sequence-policy-optimization) is the natural companion read — it formalizes sequence-level importance weighting under policy lag, which is precisely the regime the Streaming Rollout System's $\alpha$ knob operates in. And for the on/off-policy spectrum that SRS turns into a parameter, our piece on [bridging offline and online RL for LLMs](/blog/paper-reading/large-language-model/bridging-offline-and-online-reinforcement-learning-for-llms) is the conceptual backdrop: $\alpha$ is, in effect, a continuous dial between the online and offline ends of that spectrum, chosen for throughput rather than forced by data availability.

If you only chase three primary sources after this post, make them the Seed1.5-Thinking report (arXiv:2504.13914) for the system and reward design, the DAPO paper (arXiv:2503.14476) for the four value-free tricks and their ablation ladder, and the VAPO paper (arXiv:2504.05118) for the eight value-based levers and the 5→60 climb that is the strongest single argument in the trio for ever bringing a critic back.

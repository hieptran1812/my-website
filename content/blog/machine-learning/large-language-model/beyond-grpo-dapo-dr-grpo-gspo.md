---
title: "Beyond GRPO: DAPO, Dr. GRPO, GSPO, and the Loss-Aggregation Fixes of 2025"
date: "2026-06-04"
publishDate: "2026-06-04"
description: "GRPO works, but it is biased — by response length, by question difficulty, and by token-level importance ratios. A deep dive into the 2025 variants (Dr. GRPO, DAPO, GSPO, CISPO, VAPO) that fix each leak, with the math, the TRL knobs, and the benchmark numbers."
tags: ["llm", "grpo", "dapo", "dr-grpo", "gspo", "cispo", "vapo", "reinforcement-learning", "rlvr", "reasoning", "trl", "moe", "deep-learning"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 50
---

When DeepSeek-R1 landed in early 2025, GRPO went from a footnote in a technical report to the default reinforcement-learning algorithm for reasoning models almost overnight. Drop the value network, normalize each response against its group's mean, optimize a verifiable reward — it was simple, it was cheap, and it worked well enough to bootstrap chain-of-thought reasoning from a base model. If you want that story from scratch, I've written it up in [Fine-Tuning LLMs with GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo).

But "it works" and "it's correct" are different claims, and within a few months of R1, several teams independently noticed the same thing: **vanilla GRPO is biased.** Not buggy — biased, in the precise statistical sense. Its gradient systematically prefers longer responses over shorter ones regardless of correctness. It systematically over-weights the easiest and hardest prompts and under-weights the ones in the productive middle. And on Mixture-of-Experts models, its token-level importance ratios become so noisy that training simply falls over. None of these are exotic edge cases; they are structural consequences of how the objective is written, and they showed up in nearly every serious GRPO run.

The result was a wave of variants in 2025 — Dr. GRPO, DAPO, GSPO, CISPO, VAPO, GMPO — each one a targeted fix to a specific leak in the GRPO objective. This post is a guided tour of that wave: what each bias is, which variant fixes it, the math behind the fix, the knob in [TRL](/blog/machine-learning/open-source-library/trl-lib) that turns it on, and the benchmark number that tells you whether it mattered. If the [decision guide on GRPO vs DPO vs PPO](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) told you *when* to reach for GRPO, this post tells you *which* GRPO to reach for.

What makes this worth a full deep dive rather than a changelog is that the variants are not a grab-bag of unrelated tricks — they are a coherent set of corrections to a single objective, and once you see the objective the way the people who fixed it did, you can predict where the next correction will come from. Almost none of these papers changed the *shape* of GRPO. They didn't add new networks (except VAPO, which deliberately added one back), didn't change the group-baseline idea, didn't touch the verifiable-reward setup. They changed a normalization here, a clip bound there, a ratio granularity in one case — small surgical edits to terms most practitioners had copied from the R1 paper without reading closely. That's the pattern to internalize: progress in this corner of the field has come from *reading the objective carefully*, not from inventing new ones. By the end of this post you should be able to look at any new GRPO variant, find the one or two terms it touches, and slot it into the same five-lever map without needing anyone to explain it.

![The GRPO objective has five leverage points; every 2025 variant tweaks one or two of these knobs without rewriting the objective](/imgs/blogs/beyond-grpo-dapo-dr-grpo-gspo-1.png)

The table above is the map for the entire post. The GRPO objective has five places you can intervene — how you normalize the advantage, how you clip, how you aggregate the loss across tokens, how you compute the importance ratio, and how you sample prompts. Every variant in this post turns one or two of those knobs and leaves the rest of GRPO intact. Dr. GRPO fixes the advantage normalization. DAPO fixes the clip range, the aggregation, and the sampling. GSPO fixes the importance ratio. CISPO fixes the ratio a different way. Read the variants as a set of patches to a shared objective, not as competing algorithms, and the whole landscape clicks into place.

## GRPO Works — But It's Biased

Let me start by puncturing the assumption that the GRPO in the R1 paper is the GRPO you should run today. The original formulation has known biases, and shipping it unmodified leaves measurable performance and token-efficiency on the table.

| Assumption | The naive view | The reality |
|---|---|---|
| "GRPO is a solved algorithm" | Use the R1 formulation as-is | The R1 formulation has length and difficulty biases that inflate response length and waste gradient. |
| "Longer responses mean more reasoning" | Length growth is the model thinking harder | Much of the length growth is the length-normalization bias rewarding long *wrong* answers, not better reasoning. |
| "GRPO is GRPO" | All implementations are equivalent | The loss-aggregation choice alone (token-mean vs sequence-mean) changes which responses dominate the gradient. |
| "It works on dense models, so it works on MoE" | Scale it to your MoE and go | MoE expert routing shifts ~10% per update, making token-level ratios unreliable and training unstable. |
| "The value network is dead" | GRPO killed the critic for good | VAPO brought a well-tuned critic back and beat DAPO by 10 AIME points — the critic isn't dead, it was under-engineered. |

The throughline: **GRPO is a family, not a fixed algorithm, and in 2026 "we used GRPO" is underspecified.** The interesting question is which biases you patched. This post walks through them in roughly the order they bite a real run: length normalization, difficulty normalization, the clipping and sampling fixes DAPO bundles, the loss-aggregation debate, and finally the importance-ratio fix that GSPO needed to make MoE RL stable.

> The danger with a method that "just works" is that you stop reading the objective. GRPO's biases live in terms most people skim past — the `1/|o_i|`, the `/std`, the sum over tokens.

## A 60-Second GRPO Recap

So the post stands on its own, here's GRPO compressed to one pass. The figure traces it; the math pins down exactly where the biases hide.

![GRPO in one pass: sample a group, score each with a verifier, normalize against the group mean, clip the update](/imgs/blogs/beyond-grpo-dapo-dr-grpo-gspo-2.png)

For a prompt $q$, GRPO samples a group of $G$ responses $\{o_1, \dots, o_G\}$ from the current policy. A verifier scores each one, giving rewards $\{R_1, \dots, R_G\}$. The advantage of response $i$ is its reward standardized against the group:

$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_1, \dots, R_G\})}{\text{std}(\{R_1, \dots, R_G\})}$$

Then the policy is updated with a clipped, length-normalized policy-gradient objective plus a KL penalty:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\Big(r_{i,t}\hat{A}_i,\ \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i\Big) + \beta\, D_{\text{KL}}$$

where $r_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,\lt t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,\lt t})}$ is the per-token importance ratio, $|o_i|$ is the length of response $i$ in tokens, $\epsilon$ is the clip range, and $\beta$ scales the KL term against a reference policy.

Look at where the biases live. The $\frac{1}{|o_i|}$ is **length normalization** — it divides each response's loss by its token count. The $\text{std}(\cdot)$ in the denominator of $\hat{A}_i$ is **difficulty normalization**. The $\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t$ structure is the **loss aggregation** — a mean over responses of a mean over tokens. The $r_{i,t}$ is the **token-level importance ratio**. And the implicit "sum over all $G$ responses, for every prompt" is the **sampling** policy. Four of those five terms turned out to be subtly wrong, and the rest of this post is about fixing them one at a time.

## Bias #1: Length Normalization

The first bias hides in the most innocent-looking term: the $\frac{1}{|o_i|}$ that normalizes each response's loss by its length. It seems obviously correct — of course you average the per-token loss over the response — but it is exactly what teaches GRPO models to ramble.

![Length normalization is a bias, not a feature: dividing each response's loss by its token count makes long wrong answers cheap, so the model learns to ramble](/imgs/blogs/beyond-grpo-dapo-dr-grpo-gspo-3.png)

Here's the mechanism. When a response is *wrong*, its advantage $\hat{A}_i$ is negative — the gradient wants to push its tokens' probabilities down. But that negative signal gets divided by $|o_i|$. A long wrong answer therefore receives a *smaller per-token penalty* than a short wrong answer, because the same negative advantage is spread across more tokens. The optimizer notices: if being wrong is going to cost you, it's cheaper to be wrong *at length*. Symmetrically, for correct responses, the length normalization means short correct answers get a larger per-token reward, but the asymmetry on the negative side dominates because that's where length acts as a shield. The net effect, observed across many R1-style runs, is that response length climbs steadily during training — and a large fraction of that growth is the bias, not improved reasoning.

The [Sea AI Lab paper "Understanding R1-Zero-Like Training"](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) diagnosed this precisely and proposed **Dr. GRPO** (GRPO Done Right) as the fix: drop the $\frac{1}{|o_i|}$ term entirely. Instead of normalizing each response by its own length, Dr. GRPO uses a single global constant, so every token contributes to the loss with equal weight regardless of which response it belongs to. The bias vanishes, response length stops inflating, and token efficiency improves — the model reaches the same accuracy with shorter, denser chains of thought.

The numbers back it up: Qwen2.5-Math-7B trained with Dr. GRPO reached **43.3% on AIME 2024**, substantially ahead of contemporaneous open recipes on the same base (SimpleRL-Zero at 36.0%, Prime-Zero at 27.6%, OpenReasoner-Zero at 16.7%), and critically, it did so *without* the runaway length inflation. The headline for practitioners: if your GRPO model's responses are getting longer without getting better, the length-normalization bias is the first thing to remove.

## Bias #2: Std / Difficulty Normalization

The second bias is in the advantage itself — specifically, the division by the group's standard deviation. Standardizing the advantage looks like textbook variance reduction, but it silently re-weights prompts by their difficulty.

![Std normalization distorts by difficulty: dividing the advantage by the group's std over-weights low-variance prompts, the easy and the impossible ones](/imgs/blogs/beyond-grpo-dapo-dr-grpo-gspo-4.png)

Walk through the matrix. For an *easy* prompt where most of the $G$ samples are correct, the rewards are nearly uniform (mostly 1s), so the group's standard deviation is *low*. Dividing the advantage by that small std *inflates* it — the prompt gets a large gradient magnitude. The same thing happens for a *very hard* prompt where most samples are wrong (mostly 0s, low std again): inflated. But for a prompt of *average* difficulty, where the group splits roughly half-correct and half-wrong, the std is *high*, so dividing by it *shrinks* the advantage — the prompt gets a small gradient. The result is backwards from what you want: GRPO over-weights the prompts at the extremes of difficulty (where there's least to learn) and under-weights the prompts in the productive middle (where the gradient is most informative).

Dr. GRPO fixes this too, by the same logic as the length fix: remove the normalization term. Drop the $\frac{1}{\text{std}}$ from the advantage, leaving the unnormalized mean-subtracted advantage:

$$\hat{A}_i^{\text{Dr. GRPO}} = R_i - \text{mean}(\{R_1, \dots, R_G\})$$

Now every prompt contributes a gradient proportional to how far each response landed from its group mean, with no difficulty-dependent rescaling. In TRL this is a one-line change — set `scale_rewards=False` in the `GRPOConfig`, and the trainer skips the std division. Combined with the length fix (exposed via `loss_type="dr_grpo"`), you get the full unbiased Dr. GRPO objective. These two removals are the cheapest performance win in the whole GRPO-variant landscape: you are deleting terms, not adding machinery, and the model gets both more accurate and more token-efficient.

## A Worked Example: The Length Bias in Numbers

It's worth grinding through the arithmetic once, because the length bias is easy to nod along to and hard to actually feel until you see it in numbers. Take a single prompt and two responses sampled from the policy, both wrong (so both have negative advantage), but one short and one long.

Say response A is 20 tokens and response B is 200 tokens, and both earned reward 0 in a group whose mean reward was 0.4, giving each a raw advantage of $0 - 0.4 = -0.4$. Under vanilla GRPO with sequence-mean aggregation, response A's contribution to the loss is its mean per-token term — the $-0.4$ advantage spread across 20 tokens — and response B's contribution is the *same* $-0.4$ spread across 200 tokens. Per token, response B's penalty is one-tenth of response A's. The optimizer, seeing that a wrong answer is penalized ten times more gently when it's long, has a clear incentive: pad. Every extra token in a wrong answer dilutes that answer's per-token penalty, so the policy drifts toward verbosity exactly on the responses you most want it to *not* repeat.

Now switch to the unbiased token-level view that Dr. GRPO and DAPO both implement. Pool all 220 tokens and weight each one equally. Response B, with 200 tokens, now carries $\frac{200}{220}$ of the aggregate gradient and response A carries $\frac{20}{220}$ — the long wrong answer gets *more* total penalty, not less, because it spent more tokens being wrong. The incentive to pad evaporates. This is the same arithmetic the loss-aggregation figure earlier showed with 10- and 100-token responses; the point of repeating it with wrong answers is to make the *direction* of the bias visceral. Length normalization doesn't just fail to penalize length — it actively subsidizes it, and the subsidy is largest precisely on the responses (the wrong, rambling ones) you'd most like to discourage.

The difficulty bias compounds the picture. Suppose your training batch is 40% easy prompts (group mostly correct, low std), 40% hard prompts (group mostly wrong, low std), and 20% medium prompts (group split, high std). Under std normalization, the 80% of low-std prompts at the extremes get their advantages inflated and dominate the gradient, while the 20% of high-std medium prompts — the ones where the model is right on the boundary and has the most to learn — get their advantages shrunk and barely register. You are spending most of your gradient on prompts the model has already mastered or cannot yet touch, and almost none on the prompts at the frontier of its ability. Removing the std normalization rebalances the batch toward that frontier without any change to the data. Two deleted terms, two biases gone, and the run both shortens its outputs and concentrates its learning where it counts.

## DAPO: Four Engineering Fixes

Where Dr. GRPO is a minimalist's correction — remove two biased terms — **DAPO** (Decoupled Clip and Dynamic sAmpling Policy Optimization, from ByteDance Seed and Tsinghua AIR) is an engineer's bundle. It stacks four independent fixes onto GRPO and reports each one's contribution in an ablation.

![DAPO is four engineering fixes to GRPO; together they took Qwen2.5-32B to AIME-24 = 50 with half the training steps](/imgs/blogs/beyond-grpo-dapo-dr-grpo-gspo-5.png)

**Clip-Higher.** The standard PPO/GRPO clip is symmetric: $[1-\epsilon, 1+\epsilon]$ with $\epsilon = 0.2$. DAPO observed that the symmetric upper bound throttles exploration — when the model has a low-probability token that turns out to be good, the clip caps how much its probability can rise in one step, so promising-but-rare tokens are suppressed and the policy's entropy collapses. The fix is to *decouple* the bounds and raise the upper one: lower bound stays at 0.2, upper bound relaxes to **0.28**. That asymmetry lets the policy move further toward good rare tokens while still being conservative about moving away from current behavior, which keeps exploration alive deeper into training.

**Dynamic Sampling.** This is the formal fix for GRPO's zero-gradient trap. When every response in a group is correct (accuracy 1) or every one is wrong (accuracy 0), the advantage is zero for the whole group and it contributes no gradient — that prompt's slot in the batch is wasted. DAPO filters these out: it keeps sampling and discards groups with accuracy exactly 0 or 1, so every group in the batch has reward variance and therefore a real gradient. It costs extra generation but keeps the effective batch full of learning signal.

**Token-Level Policy Gradient Loss.** This is DAPO's answer to the length-aggregation problem — the same bias Dr. GRPO attacks, fixed a different way. Instead of averaging per-response and then per-token (which gives each *response* equal weight regardless of length), DAPO averages over *all tokens in the group at once*, so each *token* gets equal weight. Long responses, which have more tokens, therefore contribute proportionally more total gradient — which is what you want, because the bias being removed is the artificial down-weighting of long correct reasoning. We'll see the exact arithmetic in the next section.

**Overlong Reward Shaping.** Responses that hit the generation length limit and get truncated are a noise source: a truncated response might have been on its way to a correct answer, but it scores as wrong, injecting a wrong label into the reward. DAPO adds a soft, length-aware penalty for overlong responses (and filters the worst cases) so truncation doesn't pollute the reward signal.

What makes DAPO especially useful as a reference is that the paper reports each fix's *marginal* contribution rather than only the final number, so you can see which levers matter most. The progression in their ablation is instructive: starting from a naive GRPO baseline, adding the overlong filtering and Clip-Higher together lifts the AIME score substantially, token-level loss adds a further chunk while also controlling length, and dynamic sampling contributes both accuracy and faster convergence by keeping the batch full of gradient. No single fix is the whole story — they compound — but the two that move the needle most for typical dense-model runs are the token-level loss (which removes the length bias) and dynamic sampling (which removes the zero-gradient waste). If you only have budget to adopt two of the four, those are the two.

Stacked together, these four took **Qwen2.5-32B to 50 points on AIME 2024**, beating the previous open state-of-the-art (DeepSeek-R1-Zero-Qwen-32B at 47) while using only **50% of the training steps**. And because ByteDance open-sourced the full system — code, data, and recipe — DAPO became the reference implementation that a lot of teams actually run. In TRL terms, three of the four map to config flags: `epsilon_high=0.28` for Clip-Higher, the dynamic-sampling logic in the trainer, `loss_type="dapo"` for the token-level aggregation, and `mask_truncated_completions=True` for overlong handling.

## The Loss-Aggregation Debate

Two of the fixes above — Dr. GRPO's length-norm removal and DAPO's token-level loss — are both attacking the same thing: how you aggregate the per-token losses into one number. This deserves its own treatment because the choice is more consequential than it looks, and the variants genuinely disagree about it.

![Loss aggregation: token-mean gives a 100-token response ten times the weight of a 10-token one, which is the length bias in one line](/imgs/blogs/beyond-grpo-dapo-dr-grpo-gspo-7.png)

Take two responses in a group: a short one of 10 tokens and a long one of 100 tokens. There are two ways to turn their token losses into the group loss.

**Sequence-mean** (the R1/GRPO default) computes each response's mean token loss, then averages across responses. Each response gets weight $\frac{1}{2}$ in the group, regardless of length. This is the $\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t$ structure — and it's exactly where the length bias enters, because dividing by $|o_i|$ means each token in the long response only counts $\frac{1}{100}$ as much as each token in the short response toward its response's loss.

**Token-mean** (DAPO's choice) pools all tokens in the group and takes one mean: $\frac{1}{\sum_i |o_i|}\sum_i \sum_t$. Now the short response's tokens carry weight $\frac{10}{110}$ of the total and the long response's carry $\frac{100}{110}$. Every *token* is weighted equally; longer responses, having more tokens, contribute more total gradient.

The matrix makes the asymmetry concrete: under token-mean, the 100-token response has ten times the aggregate weight of the 10-token response. Whether that's right depends on what you believe a token "is." DAPO's argument is that in long chain-of-thought, every token is a reasoning step and deserves equal weight, so token-mean is correct and sequence-mean artificially flattens long reasoning. Dr. GRPO's argument is subtly different — it removes the per-response length normalization to make the *gradient* unbiased with respect to length, which lands close to token-level weighting in practice. The two arrive at compatible places from different directions. The practical upshot, and the reason TRL exposes `loss_type` as an explicit choice with `grpo`, `dapo`, `dr_grpo`, and `bnpo` options, is that **this is a real decision with measurable consequences, not an implementation detail** — and the original `grpo` sequence-mean is the one you most often want to change.

| `loss_type` | Aggregation | Length behavior |
|---|---|---|
| `grpo` | mean over tokens per response, then over responses | length-biased (the original) |
| `dapo` | mean over all active tokens in the global batch | length-unbiased, equal per token |
| `dr_grpo` | sum over tokens, divide by a global constant | length-unbiased (Dr. GRPO) |
| `bnpo` | mean over active tokens in the local batch | length-unbiased, batch-dependent |

## GSPO: Sequence-Level Importance Ratios

The fixes so far — advantage normalization, clipping, aggregation, sampling — all leave one term untouched: the per-token importance ratio $r_{i,t}$. On dense models that term is fine. On Mixture-of-Experts models, it is the thing that makes training fall over, and fixing it is what **GSPO** (Group Sequence Policy Optimization, from the Qwen team) is for.

![Token-level vs sequence-level importance ratio: GSPO moves the ratio from per-token to per-sequence, which is what finally stabilized MoE RL](/imgs/blogs/beyond-grpo-dapo-dr-grpo-gspo-6.png)

The problem is specific and mechanical. GRPO's importance ratio is computed per token, $r_{i,t} = \pi_\theta / \pi_{\theta_{\text{old}}}$ at each position. The variance of this ratio accumulates over the length of the sequence — a long response is a long product of noisy per-token ratios. On a dense model that's tolerable. But on an MoE model, the router selects which experts process each token, and between gradient updates the set of activated experts for the *same* input can shift by around **10%** (measured on a 48-layer Qwen3-30B-A3B model). When 10% of the experts processing a token are different from the ones that processed it during rollout, the token-level ratio $\pi_\theta / \pi_{\theta_{\text{old}}}$ becomes meaningless noise — the two policies are routing through different sub-networks. Token-level GRPO on MoE therefore sees wildly fluctuating ratios, the clipping fights the noise, and training destabilizes or collapses.

GSPO's fix is to define the importance ratio at the **sequence** level instead. Rather than a ratio per token, GSPO computes one ratio per response, based on the sequence likelihood with length normalization:

$$s_i(\theta) = \left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}\right)^{1/|o_i|}$$

and performs clipping, rewarding, and optimization at the sequence level. The length-normalization exponent $\frac{1}{|o_i|}$ keeps the ratio in a sane range regardless of response length, collapsing the accumulated per-token variance into a single stable per-sequence quantity. Because the ratio now describes the whole sequence's likelihood under each policy — not the routing of individual tokens — it's robust to the expert-shuffling that wrecks token-level ratios. The payoff: GSPO **stabilizes MoE RL training** where GRPO diverges, and it's a direct contributor to the latest Qwen3 models. The Qwen team's framing is worth remembering: GSPO helps most on MoE and large models, and may make little difference on small dense models — so it's a fix you reach for when the architecture demands it, not a universal upgrade.

It's worth dwelling on *why* the length-normalization exponent in $s_i(\theta)$ is doing such heavy lifting, because it's the crux of the whole fix. Without the $\frac{1}{|o_i|}$ exponent, the raw sequence likelihood ratio $\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}$ would be a product of $|o_i|$ per-token ratios, and its magnitude would swing exponentially with response length — a 2000-token response's ratio could be astronomically large or small, swamping any sane clip range. The geometric-mean form (raising to the $\frac{1}{|o_i|}$ power is exactly taking the geometric mean of the per-token ratios) collapses that product back to a per-token *scale*, so a long sequence and a short sequence produce ratios in the same range. That's what makes a single sequence-level clip bound sensible across responses of wildly different lengths. The connection to GMPO is not a coincidence — both fixes recognize that a *product* of per-token quantities is the wrong thing to clip or average, and that a length-normalized geometric form is the stable alternative.

The clipping then happens at the sequence level too: instead of clipping each token's ratio independently (which lets some tokens in a response be clipped while others aren't, fragmenting the update), GSPO clips the whole sequence's ratio as one unit, so a response is either fully in-range or fully clipped. That coherence is part of why it's stable — the update for a response is internally consistent rather than a patchwork of clipped and unclipped tokens. The cost is granularity: GSPO can't give different credit to different tokens within a response, because the ratio is now per-sequence. On verifiable tasks where the reward is per-response anyway, that's no loss; on tasks where you genuinely want per-token credit, it's a real tradeoff, and it's one of the reasons the value-based branch (with its per-token critic) still has a place.

## The Variant Zoo, Organized

By now the pattern should be clear, but the field has produced enough variants that a map is worth having. Here's the zoo organized by which of GRPO's five levers each one turns.

![The variant zoo, organized by lever: each column shows exactly which knob a variant turns; most touch one or two and leave the rest of GRPO intact](/imgs/blogs/beyond-grpo-dapo-dr-grpo-gspo-8.png)

Read the table column by column. **Dr. GRPO** touches only the advantage and aggregation — remove the std normalization, remove the length normalization. Surgical, subtractive, cheap. **DAPO** touches the clip (Clip-Higher), the aggregation (token-level), and the sampling (dynamic) — an engineering bundle that leaves the advantage and ratio alone. **GSPO** touches the ratio (sequence-level) and, consequently, the clip and aggregation, which move to sequence granularity with it. **CISPO** (MiniMax's variant) touches the ratio and clip in yet another way: instead of clipping the policy update like PPO/GRPO, it clips the *importance-sampling weight* and keeps every token's contribution — no token is zeroed out by clipping, which preserves gradient signal from tokens that would otherwise be discarded. MiniMax reported CISPO converging about **twice as fast** as DAPO on AIME, matching DAPO's performance with half the steps in the MiniMax-M1 work.

The one variant that breaks the "patch a lever" pattern is **VAPO** (Value-based Augmented PPO, also ByteDance), and it's the most interesting precisely because it argues the opposite thesis. Where GRPO's whole premise is "kill the critic, use the group mean," VAPO brings the critic *back* — but engineered properly this time, with a length-adaptive GAE that adjusts its bias-variance tradeoff based on response length, plus six other modifications. On Qwen-32B it reached **60.4 on AIME 2024**, beating DAPO and R1-Zero-Qwen-32B by more than 10 points. VAPO is the field's reminder that the value network wasn't fundamentally wrong — it was under-tuned for long-CoT RL — and that "critic-free" is a convenience, not a law. The honest reading of the zoo is that there are two branches: the critic-free branch (GRPO → Dr. GRPO → DAPO → GSPO → CISPO) that fixes biases while keeping the group baseline, and the value-based branch (PPO → VAPO) that reinvests in a better critic. Both work; they make different bets about where the next point of accuracy comes from. A third, orthogonal line of work attacks *outlier sensitivity*: **GMPO** (Geometric-Mean Policy Optimization) replaces GRPO's arithmetic mean over token rewards with a geometric mean, which is far less sensitive to a single outlier token blowing up the importance ratio. Because the geometric mean constrains how much any one term can move the objective, GMPO holds the importance ratio in a tighter range and reports a smaller KL divergence and higher token entropy than GRPO through training — it's a stability fix that sits alongside, not instead of, the bias fixes.

## Hyperparameters Across the Variants

Knowing which lever a variant turns is half the battle; knowing how to set the knob it exposes is the other half. Here's the practical tuning guidance for the parameters these variants add, with the values I start from.

**The clip bounds (`epsilon`, `epsilon_high`).** Vanilla GRPO uses a symmetric $\epsilon = 0.2$. DAPO's Clip-Higher decouples them — keep the lower bound at 0.2 and raise the upper to **0.28**. The upper bound governs how fast a low-probability good token can rise; too low and exploration dies, too high and the policy lunges and destabilizes. If you see entropy collapsing early (the policy getting overconfident and stopping exploration), raising `epsilon_high` is the first thing to try. If you see instability and exploding gradients, it's too high. The asymmetry is the point: you want to be permissive about moving *toward* good rare tokens and conservative about moving *away* from current behavior.

**The KL coefficient (`beta`).** This is the most situational knob in the modern recipes. The original GRPO and PPO use a meaningful KL penalty (0.04 or an adaptive target) because they're guarding against a learned reward model being exploited. But with a *verifiable* reward — a reward that cannot be hacked the way a learned model can — many 2025 recipes set `beta=0` entirely, removing the reference model from the loss and saving both memory and a forward pass. The logic: the KL leash exists to prevent reward exploitation, and a verifier isn't exploitable in that way, so the leash mostly just slows learning. Start at a small value (0.04) if you're unsure, and try zero once you trust your verifier and your other stability fixes are in place.

**The group size (`num_generations`, $G$).** This sets the quality of the group baseline. Too small (2–4) and the group mean is a noisy estimate of expected reward, so the advantage is noisy. Too large (32+) and you pay generation cost for diminishing returns. The standard range is **8–16**, and DAPO-style dynamic sampling interacts with it: if you're filtering out zero-variance groups, you need to over-generate to keep the effective batch full, so budget for generating more groups than you keep.

**The aggregation (`loss_type`).** As covered, this is `grpo` (biased), `dapo`, `dr_grpo`, or `bnpo`. Default to `dapo` or `dr_grpo`; reach for `grpo` only as a baseline. `bnpo` (batch-normalized) is a reasonable middle ground that normalizes by active tokens in the local batch.

| Parameter | Vanilla GRPO | Modern default | What it controls |
|---|---|---|---|
| `epsilon` / `epsilon_high` | 0.2 / 0.2 | 0.2 / 0.28 | exploration vs stability |
| `beta` (KL) | 0.04 | 0.0 with a trusted verifier | reward-exploitation leash |
| `num_generations` (G) | 8 | 8–16 (over-generate if filtering) | baseline quality |
| `loss_type` | `grpo` | `dapo` / `dr_grpo` | length-bias handling |
| `scale_rewards` | true | false (Dr. GRPO) | difficulty-bias handling |

A pattern worth naming: the modern recipe is mostly about *removing* things — remove the std normalization, remove the length normalization, remove the KL leash, remove zero-variance groups. The R1 formulation added safety machinery that, once the biases were understood, turned out to cost more than it bought on verifiable tasks. The variants are, to a striking degree, a story of careful subtraction.

## Detecting Each Bias in Your Own Run

You don't have to trust the papers — each bias leaves a fingerprint in your training metrics, and learning to read those fingerprints lets you diagnose which fix you need. Here's the dashboard.

**Length bias shows up as response length climbing faster than accuracy.** Plot mean response length and mean reward on the same time axis. In a healthy run, length grows modestly and tracks accuracy gains. When the length-normalization bias is active, length climbs steeply and keeps climbing even as accuracy plateaus — the model is padding, not reasoning. The cleanest confirmation is to break length out by correctness: if your *wrong* responses are growing longer over training, that's the length subsidy in action, because there's no reasoning reason for wrong answers to lengthen. Switching to `loss_type="dr_grpo"` or `"dapo"` flattens that curve.

**Difficulty bias shows up as a skewed advantage distribution.** Log a histogram of per-prompt advantage magnitudes. Under std normalization, you'll see a bimodal-ish distribution where the extreme-difficulty prompts carry outsized magnitudes and the medium prompts cluster near zero. Setting `scale_rewards=False` evens this out — the advantage magnitude then reflects how far responses landed from the mean, not how low the group's variance was. If your model is acing easy prompts and stuck on the medium ones, the difficulty bias is plausibly starving the medium prompts of gradient.

**Zero-gradient waste shows up as the fraction of groups with no reward variance.** This is the single most actionable GRPO metric. Log, every step, the fraction of groups where every response scored identically (all-correct or all-wrong). If that fraction is large — say above a third — that fraction of your batch is contributing nothing, and you're paying full generation cost for partial learning signal. Dynamic sampling drives this toward zero by construction; without it, you should at least curate prompt difficulty so groups have spread.

**MoE ratio instability shows up as gradient-norm spikes that correlate with nothing in your data.** On a dense model, gradient spikes usually trace to a bad batch. On MoE with token-level ratios, they trace to the router reshuffling experts between rollout and update, which makes the importance ratio meaningless for a subset of tokens. The tell is that the spikes are irreproducible — the same data in a different order spikes at different steps — and that lowering the learning rate buys stability without fixing the root cause. Switching to GSPO's sequence-level ratio removes the spikes because the ratio no longer depends on per-token routing.

| Bias | Metric that reveals it | Healthy | Biased | Fix |
|---|---|---|---|---|
| Length norm | length vs accuracy, length-by-correctness | length tracks accuracy | wrong answers lengthen | `loss_type=dr_grpo`/`dapo` |
| Difficulty norm | advantage-magnitude histogram | even across difficulty | extremes dominate | `scale_rewards=False` |
| Zero-grad groups | fraction of no-variance groups | low (< 1/3) | high | dynamic sampling |
| MoE ratio | gradient-norm spikes | stable, data-correlated | irreproducible spikes | GSPO sequence ratio |

The unifying habit: **instrument the objective's known failure points, not just the loss.** The loss will look fine while length inflates, while the batch wastes gradient, while the MoE ratios drift. Each bias has a second metric that exposes it, and watching those second metrics is the difference between debugging a GRPO run and guessing at it.

## What Actually Moves the Needle

It's worth stepping back from the mechanics to ask which of these fixes actually changes your benchmark number, and by how much. The timeline below sequences the 2025 wave with each variant's headline contribution.

![The GRPO variant wave of 2025: within five months four labs published targeted fixes, each addressing a distinct GRPO weakness](/imgs/blogs/beyond-grpo-dapo-dr-grpo-gspo-9.png)

A crucial caveat before reading any of these numbers: **the headline scores use different base models, datasets, and budgets, so they are not directly comparable.** Dr. GRPO's 43.3 is a 7B model; DAPO's 50 and VAPO's 60.4 are 32B; GSPO's contribution is stability on MoE, which doesn't reduce to a single AIME point. Treat the timeline as "what each fix contributed," not "a leaderboard." With that said, the practitioner-level lessons are clear:

- **Removing the biases (Dr. GRPO) is the cheapest win.** You delete two terms and get better accuracy plus shorter responses. There is almost no reason to run the length-and-std-biased original.
- **The DAPO bundle is the best-engineered default for dense models.** Clip-Higher plus dynamic sampling plus token-level loss is a robust, open, reproducible recipe, and dynamic sampling in particular fixes the single most common "GRPO won't learn" failure.
- **GSPO is non-optional for MoE.** If you're doing RL on a Mixture-of-Experts model, token-level ratios will eventually destabilize you; sequence-level ratios are the fix, and there isn't really a competing approach with the same track record.
- **The critic isn't dead (VAPO).** If you have the engineering budget for a well-tuned value network with length-adaptive GAE, the value-based branch can beat the critic-free one outright. It's more work, but the ceiling is higher.

The meta-lesson: the gains came not from a grand redesign but from carefully reading the objective and fixing one biased term at a time. That's a good model for how this field actually progresses — not with new algorithms, but with people noticing that a `1/|o_i|` they'd skimmed past was quietly steering the whole run.

## The Deeper Pattern: Why These Objectives Leak

Step back from the individual fixes and a pattern emerges in *why* GRPO had so many biases waiting to be found. Every one of them came from a normalization or aggregation choice that was correct in some other setting and wrong here.

The length normalization is the clearest case. Dividing per-token loss by sequence length is the right thing to do in ordinary language modeling, where you want each *sequence* to count equally regardless of how long it is. But in policy-gradient RL on reasoning traces, the response length is not a nuisance to normalize away — it's part of what the policy controls, and normalizing by it hands the policy a lever to game. A choice that's neutral in supervised learning becomes a bias in RL because the quantity being normalized is now an action the agent takes.

The std normalization has the same shape. Standardizing a signal by its variance is textbook variance reduction in statistics and works beautifully when the variance is a nuisance parameter. But GRPO's per-group std isn't a nuisance — it's correlated with difficulty, and difficulty is exactly the thing you want to weight by. Dividing it out throws away the signal you care about along with the noise you don't.

The token-level importance ratio is the third instance. Per-token importance sampling is standard and correct when each token's probability is computed by a stable function of the parameters. On a dense model it is. On an MoE model, the per-token probability depends on a routing decision that shifts between rollout and update, so the "function" computing it is not stable, and the per-token ratio measures partly a real policy change and partly a routing reshuffle. The standard tool breaks because an assumption it quietly relies on — stable per-token likelihood — no longer holds.

The unifying lesson is methodological: **a normalization is a statement that the thing you're dividing by is a nuisance you want to remove.** GRPO inherited three normalizations from settings where their denominators genuinely were nuisances, and applied them in a setting where those denominators carried signal — length, difficulty, routing — that the policy either controls or that correlates with what you want to learn. The fixes are all the same move in different clothes: stop normalizing by a quantity that isn't actually noise. Once you see that, the next variant becomes predictable. Find a term in the objective that divides by something the policy influences, ask whether that something is really a nuisance, and if it isn't, you've found the next bias. The variant wave isn't going to stop, because the objective still has terms nobody has interrogated yet.

## Implementing in TRL and verl

The good news for practitioners is that almost all of these variants are config flags now, not forks. Here's the GRPO-variant landscape as TRL knobs. Install first:

```bash
pip install "trl>=0.18" "transformers>=4.51" accelerate
```

The single config carries the whole variant story. Each commented knob maps to one of the fixes above:

```python
from trl import GRPOConfig

cfg = GRPOConfig(
    num_generations=16,             # G — the group size for the baseline
    learning_rate=1e-6,

    loss_type="dapo",               # aggregation: "grpo" (biased), "dr_grpo", "dapo", "bnpo"
    scale_rewards=False,            # Dr. GRPO: drop the /std difficulty normalization

    epsilon=0.2,                    # clip lower bound
    epsilon_high=0.28,              # DAPO Clip-Higher: decoupled, higher upper bound

    mask_truncated_completions=True, # DAPO overlong handling: don't reward truncated noise
    beta=0.0,                       # KL coefficient; many recipes set 0 with a trusted verifier

    max_completion_length=8192,
)
```

The combination above is roughly "DAPO with the Dr. GRPO std fix" — token-level aggregation, no std normalization, asymmetric clipping, overlong masking, and a verifier trusted enough to drop the KL term. Flip `loss_type` back to `"grpo"` and `scale_rewards` to `True` and you have the original biased R1 formulation, which is a useful A/B to run once so you can see the length inflation for yourself.

To make the loss-aggregation difference concrete, here's the token-level versus sequence-level aggregation written out from scratch. The two functions differ only in how they pool the per-token losses:

```python
import torch

def sequence_mean_loss(per_token_loss, mask):
    """GRPO default: mean over tokens per response, then mean over responses.
    Each response gets equal weight regardless of length (length-biased)."""
    per_response = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return per_response.mean()

def token_mean_loss(per_token_loss, mask):
    """DAPO: pool all tokens, one mean. Each token gets equal weight, so longer
    responses contribute proportionally more gradient (length-unbiased)."""
    return (per_token_loss * mask).sum() / mask.sum().clamp(min=1)

per_token_loss = torch.randn(2, 100)            # 2 responses, padded to 100 tokens
mask = torch.zeros(2, 100)
mask[0, :10] = 1                                # response 0 is 10 tokens
mask[1, :100] = 1                               # response 1 is 100 tokens
print(sequence_mean_loss(per_token_loss, mask)) # 10-tok response weighted 1/2
print(token_mean_loss(per_token_loss, mask))    # 10-tok response weighted 10/110
```

Dynamic sampling — DAPO's fix for zero-gradient groups — is just a filter on groups whose rewards have no spread:

```python
def keep_group(rewards, eps=1e-8):
    """Drop a group if every response scored the same (all-correct or all-wrong):
    its advantage is zero and it contributes no gradient."""
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    return rewards.std() > eps          # keep only groups with reward variance

groups = [[1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0]]
kept = [g for g in groups if keep_group(g)]     # only the mixed group survives
print(kept)                                     # [[1, 0, 1, 0]]
```

And the GSPO sequence-level ratio, the length-normalized sequence likelihood ratio that replaces the per-token one:

```python
def gspo_sequence_ratio(logp_new, logp_old, mask):
    """One importance ratio per sequence: exp of the length-normalized
    log-likelihood difference. Robust to MoE per-token routing shifts."""
    seq_logp_new = (logp_new * mask).sum(dim=1)
    seq_logp_old = (logp_old * mask).sum(dim=1)
    length = mask.sum(dim=1).clamp(min=1)
    return torch.exp((seq_logp_new - seq_logp_old) / length)   # length-normalized
```

For larger-scale runs, [verl](/blog/machine-learning/open-source-library/verl-rlhf-library-deep-dive) implements DAPO and GSPO recipes directly, with the rollout/train resharding handled for you — which matters more here than in DPO, because every one of these variants is on-policy and pays the generation cost each step.

## A Migration Path: From Vanilla GRPO to a Modern Recipe

If you have a working vanilla-GRPO setup and want to modernize it without breaking everything at once, here's the order I'd change things, easiest and highest-confidence first. Each step is independent, so you can stop when your metrics stop improving.

**Step 1: remove the biases (free, do it first).** Set `loss_type="dr_grpo"` (or `"dapo"`) and `scale_rewards=False`. You're deleting two terms; there's almost no risk and the payoff is shorter responses and a rebalanced batch. Run it against your old config and watch the length-by-correctness curves diverge — your wrong answers should stop lengthening. This single change is the one I'd never skip.

**Step 2: add Clip-Higher.** Set `epsilon_high=0.28`, keep `epsilon=0.2`. Watch token entropy: it should stop collapsing as early. If you see instability, back the upper bound down toward 0.24. This keeps exploration alive deeper into training and is most valuable on longer runs where vanilla GRPO's entropy would otherwise crash.

**Step 3: turn on dynamic sampling.** Filter zero-variance groups and over-generate to keep the batch full. The cost is extra generation; the benefit is that every group in your batch carries a gradient. Measure the zero-variance fraction before and after — if it was high, this is a large effective-batch-size win disguised as a sampling tweak.

**Step 4: handle overlong responses.** Set `mask_truncated_completions=True` (or add a soft length penalty). This cleans truncation noise out of your reward, which matters most when your `max_completion_length` is tight enough that a non-trivial fraction of responses hit it. If almost nothing truncates, this step does little.

**Step 5: consider the KL term.** If your reward is genuinely verifiable and steps 1–4 are stable, try `beta=0`. Dropping the reference model saves memory and a forward pass per step. If you see the policy drifting into degeneracy, put a small KL back. This is the one step that's task-dependent enough that I'd A/B it rather than assume.

**Step 6 (MoE only): switch to GSPO.** If you're on a Mixture-of-Experts model, move the importance ratio to sequence level before you do anything else fancy — ideally before your first long run, because token-level instability on MoE is the kind of failure that wastes a lot of compute before you diagnose it.

| Step | Change | Risk | Skip if |
|---|---|---|---|
| 1 | `dr_grpo`/`dapo` + `scale_rewards=False` | none | never skip |
| 2 | `epsilon_high=0.28` | low | run is very short |
| 3 | dynamic sampling | generation cost | zero-variance fraction already low |
| 4 | `mask_truncated_completions` | none | nothing truncates |
| 5 | `beta=0` | task-dependent | reward not verifiable |
| 6 | GSPO sequence ratio | low | dense model |

The ordering reflects confidence and cost: the free bias removals first, the cheap engineering fixes next, the task-dependent KL decision later, and the architecture-specific GSPO change gated on whether you're on MoE at all. Most dense-model teams land on "Dr. GRPO bias fixes + DAPO's four techniques" and stop there, which is a strong, well-tested place to stop.

## Case Studies from the 2025 Wave

The variants are easier to remember as the runs that produced them. Here are ten.

### 1. DeepSeek-R1-Zero — the length explosion that started it

R1-Zero is the origin point for the whole conversation. Trained with pure GRPO on verifiable rewards, it produced the famous emergent reasoning — and the famous monotonic response-length growth. The original interpretation was entirely positive: the model is learning to think longer, so length growth is reasoning growth. The variant wave's contribution was to complicate that story — a meaningful share of the length growth was the length-normalization bias rewarding long responses, not pure reasoning gain. R1-Zero is both the proof that GRPO works and the case that motivated every bias fix that followed; you can't understand Dr. GRPO or DAPO's token-level loss without it as the baseline they're correcting. The broader significance is that R1-Zero made verifiable-reward RL the center of gravity for reasoning research overnight — before it, RL on language models mostly meant RLHF with a learned reward model; after it, "RL" defaulted to GRPO on a verifier. Every variant in this post exists because R1-Zero made that setup the thing everyone was running, which meant its biases became everyone's biases, which meant fixing them was suddenly worth a paper each. A single influential run reshaped what the whole field spent 2025 optimizing.

### 2. Dr. GRPO — Sea AI Lab reads the objective carefully

The "Understanding R1-Zero-Like Training" work is the case study in *reading the math*. The team traced the length inflation to the $\frac{1}{|o_i|}$ term and the difficulty distortion to the $\frac{1}{\text{std}}$ term, removed both, and showed Qwen2.5-Math-7B hitting 43.3% on AIME 2024 — ahead of several contemporaneous open recipes on the same base — with better token efficiency. They also made a second, under-appreciated point: a lot of R1-Zero's apparent reasoning ability was already present in the Qwen base model before any RL, which reframes what the RL is actually contributing. Dr. GRPO is the minimalist's triumph: the biggest single improvement came from *deleting* terms, not adding them. The base-model observation deserves its own emphasis, because it changed how careful people interpret RL results. If a Qwen base model already exhibits self-reflection and strong math ability before any RL, then a chunk of what looks like "RL taught the model to reason" is really "RL surfaced reasoning the pretraining already installed." That doesn't make the RL worthless — it still elicits and sharpens the behavior — but it means you should be skeptical of any claim that attributes an emergent capability purely to the RL stage without controlling for what the base model could already do. Dr. GRPO's careful baselines are a model for how to report RL results honestly: separate what the RL added from what it merely revealed, and don't let a dramatic training curve fool you into over-crediting the algorithm.

### 3. DAPO — ByteDance ships the open reference system

DAPO's case study is as much about open-sourcing as about the algorithm. ByteDance and Tsinghua released the full system — the four techniques, the training code, and a curated dataset — and reported AIME 2024 = 50 on Qwen2.5-32B with half the steps of the prior best. Because it was fully open and reproducible, DAPO became the recipe a lot of teams actually ran in production, and its four fixes (Clip-Higher, dynamic sampling, token-level loss, overlong shaping) became the de facto checklist for "things to fix in vanilla GRPO." The lesson: in this field, a reproducible open recipe with an ablation table moves the community more than a higher number behind a closed door.

### 4. VAPO — the critic strikes back

VAPO is the contrarian case. While everyone was busy removing machinery from GRPO, ByteDance's VAPO team went the other way and reinvested in the value network, with a length-adaptive GAE and six other careful modifications. The result — 60.4 on AIME 2024 with Qwen-32B, beating DAPO by over 10 points — is a strong argument that the critic-free orthodoxy overcorrected. VAPO's lesson is methodological humility: "GRPO killed the critic" was a useful simplification, but the critic's poor showing in early RLHF was an engineering failure, not a fundamental one. When you have the budget to tune a value network properly, the value-based branch can win. The length-adaptive GAE at the heart of VAPO is itself a small lesson worth carrying: the standard GAE $\lambda$ that trades bias for variance in advantage estimation was tuned for short episodes, and on long reasoning traces a fixed $\lambda$ is wrong at one end or the other. VAPO adapts it to the response length, which is the same kind of "a constant that was fine in the original setting is wrong here" insight that drives the critic-free fixes — just applied to the critic's machinery instead of the group baseline's. Both branches, in the end, are doing the same intellectual work: finding the inherited constants and assumptions that don't survive the move to long-form, verifiable-reward RL, and replacing them with something that fits the actual setting.

### 5. GSPO — Qwen3 makes MoE RL stable

GSPO's case is the clearest example of an architecture forcing an algorithm change. The Qwen team was training MoE models and watching GRPO destabilize in a way it never did on dense models. The diagnosis — token-level importance ratios become unreliable when ~10% of activated experts shift per update — was specific and measurable, and the fix (sequence-level ratios) was specific and measurable. GSPO is now a contributor to the production Qwen3 models. The takeaway for anyone training MoE with RL: this is not an optional refinement, it's the difference between a run that converges and one that doesn't, and you should reach for sequence-level ratios before your first MoE RL run, not after it diverges.

### 6. CISPO — MiniMax keeps every token

MiniMax's CISPO, introduced with MiniMax-M1, attacks the ratio from a third angle. The observation is that PPO/GRPO-style clipping *zeroes out* the gradient from tokens whose ratio falls outside the clip range — and on long reasoning traces, those clipped tokens can carry important signal. CISPO instead clips the importance-sampling *weight* while keeping every token's contribution to the gradient, in a REINFORCE-with-clipped-IS style. MiniMax reported roughly 2× faster convergence than DAPO on AIME, matching its performance with half the steps. CISPO's lesson: clipping is a blunt instrument, and *what* you clip — the update versus the IS weight — changes how much signal you throw away. The insight generalizes beyond CISPO itself. Standard PPO-style clipping was designed to prevent destructive updates by zeroing the gradient on tokens whose ratio strayed too far, but on long reasoning traces those out-of-range tokens are often the most informative ones — the surprising, low-probability tokens where the model made a leap. Throwing their gradient away to stay safe is exactly backwards when the goal is to reinforce good leaps. CISPO keeps every token in the gradient while bounding the importance weight, which preserves that signal without letting any single token dominate. It's another instance of the post's recurring theme: a mechanism inherited from a different setting (conservative clipping for robotics-style RL) turned out to discard signal that matters in the LLM-reasoning setting, and the fix was to keep the signal while controlling the variance a gentler way.

### 7. A MoE collapse, before and after GSPO

A composite from several teams' experiences: a group trains a large MoE with the DAPO recipe, which works beautifully on their dense baseline, and watches it diverge on MoE — loss spikes, gradient norms exploding, runs that fail at different steps each time. Every dense-model fix they try (lower LR, tighter clip, more KL) buys a little stability but doesn't solve it, because the problem isn't the optimizer's aggression, it's that the token-level ratio is measuring two different routings. Switching the importance ratio to sequence-level (GSPO) resolves it directly. The debugging lesson: when an RL recipe that works on dense models falls over specifically on MoE, suspect the importance ratio before you suspect your hyperparameters.

### 8. The loss_type ablation everyone should run once

The last case is a practice, not a paper. Several teams, after reading the Dr. GRPO and DAPO work, ran the simple A/B: same data, same model, same everything, only `loss_type` changed between `grpo` and `dapo`/`dr_grpo`. The consistent finding was that the original sequence-mean `grpo` loss produced longer responses and slightly worse accuracy than the length-unbiased options — visible within a few hundred steps. It's a cheap experiment with a clear result, and running it once converts "I read that GRPO is length-biased" into "I watched my own run inflate." If you take one operational habit from this post, make it this: A/B the aggregation on your own task before trusting the default.

### 9. GMPO — suppressing outlier tokens with a geometric mean

GMPO comes at stability from an angle none of the others take: the *outlier sensitivity* of the arithmetic mean. GRPO and PPO both optimize an arithmetic mean over token-level terms, and an arithmetic mean is dominated by its largest element — a single token whose importance ratio spikes can swing the whole objective and the resulting gradient. GMPO's fix is to maximize the *geometric* mean of token-level rewards instead, which by construction is far less moved by any one outlier term and keeps the importance ratio in a tighter range. The reported result on a 7B model was a 4.1% average Pass@1 gain over GRPO across five math-reasoning benchmarks, with a multimodal gain on Geometry3K as well — but the more telling metrics were the *stability* ones: through training, GMPO held a smaller KL divergence from the base model and higher token entropy than GRPO, both signs of steadier, more exploratory optimization. GMPO's lesson is that "which mean" is itself a design choice, and the arithmetic mean's outlier sensitivity is one more quiet assumption in the GRPO objective worth interrogating.

### 10. The KL-free trend — removing the reference model entirely

The last case is a trend rather than a single paper, and it's one of the clearest signs of how the field's understanding matured. Vanilla GRPO and PPO carry a KL penalty against a frozen reference model, inherited from RLHF where the reference is the leash that stops the policy from exploiting a learned reward model into nonsense. Through 2025, recipe after recipe reported that with a *verifiable* reward, you can set the KL coefficient to zero — drop the reference model from the loss entirely — and not only does training stay stable, it often learns faster, because the leash was holding the policy back from a reward it couldn't game anyway. The practical payoff is real: removing the reference model saves a full model's worth of memory and a forward pass per step. The conceptual payoff is bigger — it's the field collectively realizing that a piece of machinery everyone copied from RLHF was solving a problem (reward hacking) that verifiable rewards don't have. The KL-free trend is the purest example of this post's recurring theme: the biggest wins came from deleting inherited assumptions, not adding new ones.

## When to Reach for Each Variant — and When Vanilla GRPO Is Fine

**Always apply the Dr. GRPO bias fixes** (`loss_type="dr_grpo"` or `"dapo"`, `scale_rewards=False`) unless you have a specific reason not to. They cost nothing — you're removing terms — and they improve both accuracy and token efficiency. The biased original `grpo` formulation is mostly useful as a baseline to A/B against.

**Reach for the full DAPO bundle when** you're training a dense model and want a robust, reproducible default. Clip-Higher keeps exploration alive, dynamic sampling fixes the zero-gradient trap, token-level loss removes the length bias, and overlong shaping cleans the reward. It's the best-tested open recipe for dense long-CoT RL.

**Reach for GSPO when** you're training a Mixture-of-Experts model, full stop. Token-level ratios and MoE routing don't mix, and sequence-level ratios are the fix with the track record. Reach for it preemptively, not reactively.

**Reach for CISPO when** you want faster convergence and you're worried about clipping discarding signal on long traces — particularly if you've seen entropy collapse or stalled learning that the DAPO fixes didn't resolve.

**Reach for VAPO (the value-based branch) when** you have the engineering budget to tune a value network properly and you want the highest ceiling. It's more work than any critic-free variant, but it can beat them, and the critic gives you per-token credit assignment that the group mean can't.

**Vanilla GRPO is fine when** you're prototyping, the run is short, the responses aren't long enough for length bias to compound, and you just want a quick signal that RL is moving your metric at all. For anything you intend to ship, apply at least the Dr. GRPO bias removals — they're free.

The honest summary for 2026: **"use GRPO" should now mean "use GRPO with the bias fixes," and on MoE it should mean GSPO.** The vanilla R1 formulation was a brilliant starting point and a biased production algorithm. The variants didn't replace it — they finished it, one carefully-read term at a time. If you internalize the five-lever map from the top of this post, every future variant will slot into it, because the objective isn't changing; people are just getting better at reading it. And the practical bottom line is simple enough to act on today: apply the Dr. GRPO bias removals to every run, adopt DAPO's token-level loss and dynamic sampling on dense models, switch to GSPO on Mixture-of-Experts, and keep one eye on the value-based branch for when you need a higher ceiling than the group baseline can reach.

## Common Pitfalls When Adopting These Variants

The fixes are mostly config flags, which makes them easy to apply and easy to misapply. A few traps recur often enough to call out.

**Stacking fixes blindly and changing five things at once.** The temptation, after reading a post like this, is to flip every flag at once and run. Then something regresses and you have no idea which change caused it. Change one lever at a time, in the migration order above, and keep the baseline run alongside. Each fix targets a specific bias with a specific metric signature — if you change them one at a time, you can watch each signature respond and confirm the fix did what you expected. If you change them all at once, you've traded a biased-but-understood run for an unbiased-but-mysterious one.

**Setting `beta=0` before the other fixes are stable.** Dropping the KL term is a real win on verifiable rewards, but it removes a safety leash, and if your run isn't otherwise stable, the leash was masking the instability. Get steps 1–4 of the migration solid first, *then* try removing KL. If you drop KL early and the run degrades, you'll wrongly conclude that KL-free doesn't work, when the real problem was an unaddressed bias elsewhere.

**Forgetting that dynamic sampling changes your effective batch size.** When you filter out zero-variance groups, you keep fewer groups than you generate, so your effective batch shrinks unless you over-generate. Teams that turn on dynamic sampling without increasing generation see their batch quietly shrink and their gradient noise rise, then blame the wrong thing. Budget extra generation when you enable filtering.

**Applying GSPO to a dense model and expecting a jump.** GSPO's sequence-level ratio is a fix for MoE instability; on a dense model, where token-level ratios are already stable, it may do little or nothing. The Qwen team said as much. If you switch a dense model to GSPO and see no change, that's expected — it's not a universal upgrade, it's an MoE-specific fix.

**Trusting a weak verifier.** Every critic-free variant assumes the reward is verifiable and trustworthy. If your "verifier" is actually a brittle regex that misses correct answers formatted differently, or a heuristic that can be satisfied without solving the task, all the bias fixes in the world won't save you — the policy will optimize the verifier's flaws. Audit your verifier's false-negative and false-positive rates before you blame the algorithm. A clean objective on a dirty reward still gives you a dirty model.

**Comparing numbers across base models.** The benchmark scores in this post — 43.3, 50, 60.4 — come from different base models and setups and are not directly comparable. It's tempting to rank the variants by their headline AIME number, but VAPO's 60.4 isn't "better than" DAPO's 50 in a controlled sense; they used different bases and different budgets. Rank variants by *which bias they fix*, then pick the ones whose biases are hurting your run.

## Frequently Asked Questions

A few questions come up every time I walk a team through this material.

**"Do I need all of DAPO's four fixes, or can I cherry-pick?"** Cherry-pick freely — they're independent. The token-level loss and dynamic sampling give the largest gains for most teams; Clip-Higher matters most on long runs where entropy would otherwise collapse; overlong shaping matters most when your length limit is tight. There's no requirement to adopt all four, and the migration order above is exactly a cherry-picking guide.

**"Is Dr. GRPO or DAPO the right way to fix the length bias?"** They're compatible and they land in similar places. Dr. GRPO removes the per-response length normalization (and the std normalization); DAPO's token-level loss re-weights tokens equally across the group. In TRL both are `loss_type` options (`dr_grpo` and `dapo`), and either fixes the length bias. Pick `dapo` if you also want the rest of the DAPO bundle's framing; pick `dr_grpo` if you want the minimal subtractive change. Don't agonize — A/B them on your task if it matters.

**"Why would I ever go back to a value network (VAPO) after GRPO removed it?"** Because the group mean is a *cheaper* baseline, not a *better* one. A well-tuned critic gives per-token credit assignment that the group mean can't — it can tell you which specific tokens in a correct response were the good decisions, where the group mean assigns the whole response one advantage. If you have the engineering budget to tune a value network with length-adaptive GAE, that finer credit assignment can raise your ceiling, which is how VAPO beat DAPO. GRPO traded that ceiling for simplicity; VAPO buys it back.

**"My GRPO run isn't learning at all — where do I look first?"** The zero-variance group fraction. The single most common cause of "GRPO won't learn" is that a large share of your groups are all-correct or all-wrong, contributing no gradient, so the effective signal is tiny. Log that fraction; if it's high, turn on dynamic sampling and curate prompt difficulty. The second place to look is your verifier — confirm it's actually scoring correctly, because a verifier that marks everything wrong gives every group zero variance too.

**"Are these variants only for math and code?"** They're for any *verifiable* reward, which is broader than math and code but narrower than "everything." Structured output you can validate against a schema, tasks with checkable constraints, tool-use traces you can verify executed correctly — all fit. The common thread is a reward function that returns a trustworthy score without a learned model. For genuinely subjective quality, you're back in preference-learning territory (DPO/PPO), which the [decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) covers.

**"How stable is this landscape — will these variants be obsolete in a year?"** The specific names may shift, but the five-lever map won't. Every variant so far has been a patch to one of advantage normalization, clipping, aggregation, importance ratio, or sampling. Future variants will almost certainly be more patches to those same levers (or a newly-interrogated term in the objective). Learn the map and the levers, and new variants become easy to place rather than a new thing to memorize.

## Further Reading

- [Fine-Tuning LLMs with GRPO: From Theory to Implementation](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) — the GRPO foundation these variants build on.
- [GRPO vs DPO vs PPO: A Decision Guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) — when to reach for GRPO at all, versus DPO or PPO.
- [Training LLMs for Math](/blog/machine-learning/large-language-model/training-llm-for-math) — the verifiable-reward pipeline these variants optimize.
- [Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning](/blog/paper-reading/large-language-model/part-i-tricks-or-traps-a-deep-dive-into-rl-for-llm-reasoning) — an empirical look at which RL tricks survive scrutiny.
- [DeepSeek-R1: Incentivizing Reasoning via Reinforcement Learning](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) — the run that started the wave.
- [verl: An Engineer's Deep Dive into HybridFlow RL](/blog/machine-learning/open-source-library/verl-rlhf-library-deep-dive) — the infrastructure for running these at scale.

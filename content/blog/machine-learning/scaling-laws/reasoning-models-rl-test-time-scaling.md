---
title: "Reasoning models: scaling RL and test-time compute (o1, DeepSeek-R1)"
date: "2026-06-15"
description: "Learn how reinforcement learning with verifiable rewards trains a model to spend test-time compute productively, why o1 and DeepSeek-R1 improve log-linearly with both training and inference compute, and how to apply this new scaling axis."
tags: ["scaling-laws", "reasoning-models", "reinforcement-learning", "test-time-compute", "o1", "deepseek-r1", "grpo", "chain-of-thought", "verifiable-rewards", "inference-scaling", "aime"]
category: "machine-learning"
subcategory: "Scaling Laws"
author: "Hiep Tran"
featured: true
readTime: 53
---

For four years the scaling-laws literature had exactly one knob worth turning: make the pre-trained model bigger, feed it more tokens, and watch the loss fall along a power law. Every post in this series so far — [Kaplan](/blog/machine-learning/scaling-laws/kaplan-scaling-laws-language-models), [Chinchilla](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling), [data-constrained](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) — is a variation on that one knob. Then in late 2024 two models broke the pattern in a way that is still reshaping the whole field. OpenAI's o1 and, four months later, DeepSeek-R1 did not get better by being bigger. They got better by being *trained to think longer at inference time*, and the lever that taught them to do it was reinforcement learning against rewards a machine can check. The headline result is the one to keep in your head: o1's accuracy on the AIME competition math benchmark rises smoothly and log-linearly with both the amount of reinforcement-learning compute spent in training and the amount of compute spent searching at inference — and DeepSeek-R1 reproduced that curve in the open, lifting AIME pass@1 from 15.6% to 71.0% through pure RL alone.

The diagram above is the mental model for the entire post, and it is worth staring at before we go anywhere else. There are two compute levers, not one, and they are coupled. The first is a *training* lever: spend compute on reinforcement learning with verifiable rewards, and the model learns a long chain-of-thought policy — it learns to explore, backtrack, decompose, and self-verify. The second is a *test-time* lever: once that policy exists, spending more compute at inference (longer chains, more samples, a reranker) keeps buying accuracy. The training lever is what *unlocks* the test-time lever. A base model that has only ever been trained to predict the next token does not benefit much from being told to "think step by step" — but a model whose reasoning has been shaped by RL turns every extra inference token into measurable accuracy.

![A branching diagram showing a base language model fed through reinforcement learning with verifiable rewards into a reasoning policy that produces a long chain of thought, then splitting into a single-sample path and a majority-vote path that both feed an accuracy outcome](/imgs/blogs/reasoning-models-rl-test-time-scaling-1.png)

> [!important] The one number to remember: pure RL took AIME pass@1 from 15.6% to 71.0%
> - **There are two scaling levers now: RL training compute and test-time compute, and they are coupled.** RL teaches the model to spend inference compute productively; without that training, a base model gains little from "thinking longer."
> - **o1 improves log-linearly on both axes.** Accuracy rises roughly linearly against the *logarithm* of training-RL compute and against the *logarithm* of test-time compute. OpenAI published no slope — only that the curves stay straight on a log scale.
> - **o1 on AIME 2024: 74% with a single sample, 83% with a majority vote of 64 samples, 93% with a learned reranker over 1000 samples** — against a GPT-4o baseline of 12%.
> - **DeepSeek-R1-Zero is pure RL, no supervised fine-tuning.** Running GRPO on the DeepSeek-V3 671B base with rule-based rewards lifted AIME pass@1 from 15.6% to 71.0%, and cons@64 reached 86.7%, matching OpenAI-o1-0912.
> - **Test-time scaling emerged from the RL by itself.** Response length grew spontaneously over training, and the model exhibited an "aha moment" of self-reflection — nobody coded those behaviors in.
> - **The clean scaling is gated by verification.** Math and code, where a reward is a rule check, scale beautifully. Open-ended domains, where selection is the bottleneck, plateau.
> - **The practical shift: spend training compute on RL that teaches a model to spend test-time compute well.** This is the training-side lever that bends the inference-side curve.

This post is the eleventh in the scaling-laws series, and it sits at the hinge. Up to now we have been scaling pre-training. From here, the action moves to inference — and that move is the subject of the next post, [did pre-training hit a wall](/blog/machine-learning/scaling-laws/pretraining-wall-inference-scaling). To understand reasoning models you need two prior pieces: [repeated sampling as a scaling law](/blog/machine-learning/scaling-laws/repeated-sampling-scaling-laws), which establishes that throwing more samples at a verifiable problem follows its own power law, and [compute-optimal test-time scaling](/blog/machine-learning/scaling-laws/test-time-compute-scaling), which shows that *how* you spend inference compute matters as much as how much. Reasoning models are what you get when you stop bolting test-time compute on after the fact and instead train the model, end to end, to produce it.

## Why reasoning models are a different kind of scaling

The senior rule of thumb here: **a reasoning model is not a bigger model, it is a model that has been taught to convert inference compute into accuracy, and RL is the teacher.** Hold that thought against the older assumptions and the mismatch is stark.

| The old assumption | The naive view of "thinking" | The reality with reasoning models |
|---|---|---|
| Quality comes from parameters and pre-training tokens | "Think step by step" prompting unlocks latent reasoning for free | A base model gains little from longer prompts; RL must *train* the long-CoT policy first |
| Inference compute is a fixed cost per query | More samples always help | More samples only convert to correct answers when a verifier can pick the winner |
| The scaling curve is loss vs pre-training FLOPs | Test-time tricks are a one-off bump | Accuracy is log-linear in *both* RL-training compute and test-time compute |
| Improvement requires a new, larger base | You need a frontier-scale model to reason | A 671B base plus pure RL reproduced the frontier reasoning result in the open |
| Reasoning ability must be explicitly supervised | You need human-written reasoning traces | Reasoning, backtracking, and self-checking *emerged* from outcome rewards alone |

The reason this is a genuinely new scaling axis, and not a prompting gimmick, comes down to where the compute is going and what it is buying. In classical scaling, training compute buys a lower loss, and inference compute is something you try to minimize. In the reasoning-model world, you spend extra *training* compute on RL specifically to produce a model that benefits from extra *inference* compute. The two are now in a deliberate trade: you pay once, in training, to make every future inference dollar more productive. That is a structurally different bet from "make the base model bigger," and it is why the [inference-aware scaling laws](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) — small over-trained model, cheap per token — and the reasoning-model paradigm fit together so neatly. A small, cheap-to-serve base, taught by RL to think longer only when the problem is hard, is the combination the whole field is now chasing.

One caveat to plant early, because it governs everything that follows: this only works cleanly where you can *check* an answer. Competition math has a numeric answer you can compare. Code has unit tests. Formal proofs have a proof checker. In those domains the reward is a rule, the RL signal is honest, and the scaling curve is clean. Move to "write a good essay" and the reward becomes a learned model of human preference, which is itself fallible — and the clean curve degrades into a plateau. We will return to this wall repeatedly; it is the single most important boundary condition on the entire approach.

## 1. The OpenAI o1 result: log-linear on two axes

**Senior rule of thumb: when someone says a model "scales with test-time compute," ask which axis and whether the plot is straight on a log scale — that is the whole claim.**

OpenAI announced o1 on September 12, 2024, in a blog post titled "Learning to Reason with LLMs." There is no arXiv paper, no released weights, and — importantly for how we talk about it — no published slope or exponent. What OpenAI did publish were two plots, and the shape of those plots is the entire scientific content of the announcement. On the first, the x-axis is train-time compute spent on reinforcement learning, on a log scale; the y-axis is accuracy on AIME. On the second, the x-axis is test-time compute, again on a log scale; the y-axis is the same accuracy. Both plots are straight lines. That is what "log-linear" means and it is all we should claim: accuracy rises roughly linearly against the logarithm of compute, with no visible kink or saturation over the range they showed.

The figure below renders both panels side by side, because the coupling of the two axes is the point that gets lost when people quote a single AIME number.

![Two side-by-side accuracy-versus-log-compute panels for OpenAI o1, the left panel showing AIME accuracy rising as a straight line with reinforcement-learning training compute and the right panel showing AIME accuracy rising as a straight line with the number of samples at inference, annotated with seventy-four percent at one sample, eighty-three percent with majority vote, and ninety-three percent with a reranker](/imgs/blogs/reasoning-models-rl-test-time-scaling-2.png)

Resist the urge to read a slope off that picture. OpenAI deliberately left the x-axes unlabeled in absolute units, and they never stated how many FLOPs separate the left end from the right end of either plot. Anyone quoting "o1 scales as compute to the 0.2" is making it up. The honest, defensible statement is the one in the caption: both curves are straight on a log axis over the range shown, which means each doubling (or 10×, or whatever the base) of compute buys a roughly constant additive accuracy gain.

### The AIME numbers, stated carefully

The concrete numbers that o1 did publish are on AIME 2024, the American Invitational Mathematics Examination — fifteen short-answer problems, each with an integer answer from 0 to 999, hard enough that strong human contestants miss several. Here is the progression, and the gap between methods is the story:

- **Single sample: 74%.** One pass of the model, one chain of thought, one answer. This alone is a massive jump over the GPT-4o baseline of **12%**.
- **Majority vote of 64 samples: 83%.** Generate 64 independent chains, take the most common final answer. This is the consensus/self-consistency method, and it converts raw sampling into a roughly nine-point gain.
- **Learned reranker over 1000 samples: 93%.** Generate a thousand samples and use a learned scoring function to pick the best, rather than simple majority. The extra ten points over majority voting is exactly the value of a *better* selection mechanism — which foreshadows the verification theme.

Walk the right-hand panel of the figure left to right and you are watching those three numbers: 74 → 83 → 93, each step costing roughly an order of magnitude more samples, each step landing on the same straight log-linear line. The GPT-4o baseline of 12% sits down at the origin as a reminder of where a strong non-reasoning model of the same era lands — the entire 62-point gap to single-sample o1 is what the RL training bought.

### What the model is actually doing at inference

The mechanism, in OpenAI's own framing, is that large-scale RL teaches the model to produce a long internal chain of thought: it explores multiple approaches, recognizes when an approach is failing and backtracks, decomposes the problem into sub-problems, and checks its own intermediate work. Crucially, this is not the model being prompted to do those things. It is the model having been *trained*, through reward, to do them because they raise the probability of a correct final answer. The chain-of-thought is "hidden" in o1 (OpenAI shows a summary, not the raw trace), but the behaviors it describes — explore, backtrack, self-verify, decompose — are exactly the behaviors that DeepSeek later showed emerging spontaneously from RL, which is our strongest evidence that the o1 story is mechanistically real and not a marketing gloss.

A useful way to hold the two axes together: the train-time axis determines *how good the policy is at thinking*, and the test-time axis determines *how much thinking you let it do on a given query*. They multiply. A weakly-trained policy thinking for a long time wanders; a strongly-trained policy thinking briefly leaves accuracy on the table. The product of a good policy and ample test-time compute is what gets you to 93%.

### What "log-linear" buys you, and what it costs

It is worth being precise about what a straight line on a log-compute axis actually implies, because it is both encouraging and sobering. If accuracy $a$ satisfies $a \approx a_0 + s \cdot \log_{10}(C)$ for compute $C$ and some slope $s$, then every 10× increase in compute adds a *constant* $s$ percentage points. That is the encouraging read: there is no near-term saturation in the range shown, and you can plan — "one more order of magnitude of test-time compute buys roughly $s$ more points." It is also the sobering read: because the x-axis is logarithmic, *additive* gains in accuracy demand *multiplicative* growth in compute. Going from 74% to 83% to 93% on o1 cost roughly 64× then ~16× more samples per step. The next nine points, if the line holds, would cost another order of magnitude or more — and the line cannot hold forever, because accuracy is bounded by 100% (and by the irreducible error of unsolvable-by-this-model problems). A log-linear curve must eventually bend; the open question o1 leaves unanswered, precisely because it published no absolute axis, is *where*. The discipline this imposes: never extrapolate a log-linear test-time curve past the range you have measured, because the bend — the point where doubling compute stops paying — is exactly the thing the straight-line region hides from you. This is the same caution the [test-time-compute post](/blog/machine-learning/scaling-laws/test-time-compute-scaling) raises when it notes that test-time compute beats a 14× larger model in the easy regime but loses on the hardest tail: the clean line is a local phenomenon, not a promise.

## 2. From single-shot answers to a trained long chain-of-thought

**Senior rule of thumb: "chain-of-thought prompting" and "a reasoning model" are different things — one is a prompt, the other is a trained policy, and only the second one scales.**

It is easy to conflate o1's long chains with the chain-of-thought prompting trick from 2022, where you append "let's think step by step" and the model emits intermediate steps. They are not the same, and the difference is the whole point. The figure below contrasts a single-shot chain — what a strong non-reasoning model like GPT-4o produces — with the trained long chain a reasoning model produces.

![A before-and-after comparison with the left column showing a single-shot chain-of-thought that reads the problem, emits one short chain with no backtracking, and commits to an answer scoring twelve percent on AIME, and the right column showing a trained long chain-of-thought that decomposes the problem, backtracks and self-verifies when a branch fails, and commits to an answer scoring seventy-four percent at one sample and ninety-three percent at one thousand](/imgs/blogs/reasoning-models-rl-test-time-scaling-3.png)

The left column is what a single forward pass with step-by-step prompting buys you. The model reads the problem, emits a chain of a few steps, and commits. There is no mechanism to recover from a wrong first step — once it commits to a flawed decomposition, the rest of the chain elaborates the error, and AIME accuracy sits at 12%. This is not a failure of intelligence so much as a failure of *process*: the model has no trained habit of doubting itself.

The right column is the trained reasoning policy. It decomposes the problem, tries an approach while spending many more tokens, and — this is the load-bearing behavior — backtracks and self-verifies when a branch fails. The same base capability, wrapped in a process that explores and checks, lands at 74% with a single chain and 93% with a thousand. The capability did not change by 81 points; the *process* did.

### Why prompting alone cannot get you there

Here is the part that is genuinely non-obvious, and it is why this required RL rather than clever prompting. A base model trained only on next-token prediction has seen plenty of human reasoning text, so it can imitate the *surface form* of a long chain. But it has no calibrated sense of when its own reasoning is going wrong, because nothing in next-token training rewarded it for that. If you prompt a base model to "think longer," it produces a longer chain that is just as likely to confidently elaborate a wrong path as a right one — longer chains without trained self-correction often *lower* accuracy, because they give the model more opportunities to talk itself into a mistake.

RL fixes this by making the *outcome* the teacher. The model samples a full chain, the chain produces an answer, the answer is checked, and the reward flows back to make whatever process led to a correct answer more likely. Over many iterations, the processes that survive are the ones that explore productively and catch their own errors — not because anyone labeled "this is a good backtrack," but because backtracking, on average, led to more correct answers. That is the difference between a prompt and a policy: a prompt asks for a behavior, a policy has been selected for it.

```text
# Single-shot CoT (what prompting buys)
Problem -> [step 1][step 2][step 3] -> answer        # no recovery if step 1 is wrong

# Trained long CoT (what RL buys)
Problem -> [try approach A]
           [wait, A gives a contradiction at step 4]
           [backtrack, try approach B]
           [check B against a small case... consistent]
           [verify final arithmetic]
        -> answer                                     # process selected by reward
```

That pseudo-trace is illustrative, not a real o1 transcript, but it captures the shape: the trained policy spends its extra tokens on exploration and verification, and those are exactly the tokens that turn into accuracy.

### The lineage: how the field arrived at trained reasoning

o1 did not appear from nowhere, and seeing the lineage makes the design choices feel inevitable rather than magical. The first thread is *chain-of-thought prompting* (Wei et al., 2022): the discovery that simply asking a model to show its work raised accuracy on multi-step problems. That established the surface form but, as we just argued, not the trained habit. The second thread is *self-consistency* (Wang et al., 2022): sample multiple chains and take the majority answer. That is precisely the majority@k method o1 uses at inference, and it works because, in verifiable-ish domains, the correct answer tends to be the modal one across diverse chains. The third thread is the family of *bootstrapping* methods — STaR (Zelikman et al., 2022) and its successors — which had the model generate its own reasoning, keep the chains that reached the right answer, and fine-tune on them. STaR is, in hindsight, a poor man's RL: a single iteration of generate-filter-train, exactly the rejection-sampling-plus-SFT move that appears as stage 3 of full R1. The fourth thread is *reward modeling* for reasoning, where the question became whether to reward the final answer (outcome reward models, ORMs) or each step (process reward models, PRMs). The PRM line, explored heavily in the test-time-search literature, gives a denser signal but needs step-level labels; the ORM line is cheaper and is what verifiable rewards are. R1-Zero's punchline is that an outcome reward, scaled up with RL, is enough — the dense step-level supervision PRMs provide turns out to be substitutable with a lot of outcome-supervised RL. That is the conceptual arc: prompting showed the form, self-consistency showed the aggregation, STaR showed bootstrapping, and RL with outcome rewards turned the bootstrap into a continuous, scalable training loop.

## 3. The training lever: RL with verifiable rewards and GRPO

**Senior rule of thumb: the cleanest RL signal you can give a language model is one a machine can check — a numeric answer, a unit test, a proof checker — because then the reward cannot be gamed by sounding plausible.**

The training-side lever is reinforcement learning, and the specific algorithm DeepSeek used is GRPO — Group Relative Policy Optimization. To understand why GRPO matters, you have to understand what makes RL on language models expensive and fragile, and how GRPO sidesteps the worst of it. The classic approach, PPO, trains a separate *value network* (a critic) alongside the policy to estimate how good each partial generation is, so it can compute advantages. That critic is roughly as large as the policy, doubles the memory footprint, and is notoriously finicky to train. GRPO throws the critic away and estimates the advantage a different way: sample a *group* of answers to the same prompt, score them all, and use the group's own statistics as the baseline.

The figure below walks through a single GRPO step.

![A five-stage pipeline showing a prompt feeding a sample group of G outputs, each output scored by a rule-based reward of zero or one, the rewards converted to a group-relative advantage by subtracting the group mean, and finally a clipped policy update with no value network](/imgs/blogs/reasoning-models-rl-test-time-scaling-4.png)

Read it left to right. A single prompt — one math or code problem — is rolled out into a group of $G$ outputs (16 is a typical group size). Each output is scored by a rule-based reward: is the final answer correct, and is the output in the required format? For verifiable tasks this reward is a simple 0 or 1, computed by a checker, not a learned model. Then the key step: each output's advantage is its reward minus the group mean reward. An answer that beat its peers gets a positive advantage; one that lagged gets a negative one. Finally, a clipped policy-gradient update (the PPO-style clip keeps the step from being too large) nudges the policy toward the above-average outputs and away from the below-average ones — with no value network anywhere in the loop.

### The math, defined symbol by symbol

Let $q$ be a query (prompt) and let the current policy $\pi_\theta$ sample a group of $G$ outputs $\{o_1, \dots, o_G\}$. Each output $o_i$ receives a scalar reward $r_i$ from the verifier. GRPO computes a *group-relative* advantage by standardizing the rewards within the group:

$$
A_i = \frac{r_i - \text{mean}(\{r_1,\dots,r_G\})}{\text{std}(\{r_1,\dots,r_G\})}
$$

Here $\text{mean}$ and $\text{std}$ are the mean and standard deviation of the $G$ rewards in the group. The advantage $A_i$ is positive for outputs that beat the group average and negative for those below it — the group is its own baseline, which is exactly the role PPO's critic plays, but computed for free from samples you already drew. The policy is then updated to maximize a clipped objective:

$$
\mathcal{J}(\theta) = \mathbb{E}\Big[ \frac{1}{G}\sum_{i=1}^{G} \min\big( \rho_i A_i,\; \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i \big) - \beta \, D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}}) \Big]
$$

where $\rho_i = \pi_\theta(o_i \mid q) / \pi_{\theta_{\text{old}}}(o_i \mid q)$ is the importance-sampling ratio between the updated and sampling policies, $\epsilon$ is the clip range (so a single update cannot move the policy too far), $\beta$ weights a KL penalty $D_{\mathrm{KL}}$ that keeps the policy from drifting too far from a reference model $\pi_{\text{ref}}$, and the expectation is over queries and sampled groups. The whole objective is just "increase the probability of above-average outputs, decrease the probability of below-average ones, but don't move too fast and don't drift too far from where you started."

### Why "verifiable" is doing the heavy lifting

The phrase to weigh carefully is *verifiable rewards*. When $r_i$ is computed by checking a numeric answer or running unit tests, it is honest: a wrong answer that *looks* brilliant gets 0, and a correct answer that looks terse gets 1. The model cannot earn reward by being persuasive. That honesty is what makes the RL signal trustworthy enough to shape a long-CoT policy without the model learning to hack the reward. The moment you replace the checker with a learned reward model — which you must, for open-ended tasks — the model can start optimizing for what the reward model likes rather than what is correct, and the clean scaling degrades. This is the same wall we keep hitting; here it shows up as the difference between a rule-based reward and a learned one.

A concrete sense of the compute involved: each GRPO step samples $G$ full long-CoT generations per prompt, and those generations get longer as training proceeds (more on that next). With a 671B base, group size 16, and chains that grow to thousands of tokens, the rollout cost dominates — RL on reasoning models is mostly an inference-bound workload, which is one reason the field cares so much about cheap inference. The training lever is expensive precisely because exercising it means generating an enormous amount of test-time-style compute during training.

### Where GRPO sits among the alternatives

GRPO is one choice in a crowded space, and knowing why it was the right one for reasoning helps you make the call for your own setup. The table below lines up the main options.

| Method | Needs a value/critic network? | Reward type | Best fit for |
|---|---|---|---|
| PPO | Yes (critic ≈ policy size) | Scalar reward (learned or rule) | General RLHF; expensive memory, finicky critic |
| GRPO | No (group is the baseline) | Scalar reward, ideal for rule-based | Reasoning RL with verifiable rewards |
| DPO | No (no online sampling) | Pairwise preferences | Preference alignment from a fixed dataset |
| RLHF/PPO with PRM | Yes | Per-step process reward | Dense supervision when step labels exist |
| Rejection sampling + SFT | No | Filter by outcome, then imitate | Cheap bootstrapping; the STaR move |

The reason GRPO fits reasoning so well is a matter of cost and signal alignment. PPO's critic must learn to predict the value of partial long chains — a hard regression problem on sequences thousands of tokens long — and it doubles your memory. For a 671B policy, a 671B-ish critic is a brutal tax. GRPO's insight is that when you already sample a *group* per prompt, the group mean is a perfectly good baseline, so the critic is redundant; you trade a learned value estimate for a sampled one, which is cheaper and, for verifiable rewards, lower-variance than it has any right to be. DPO is excellent for preference alignment but operates on a fixed dataset of pairwise comparisons with no online sampling, so it cannot explore — and exploration is the whole point of reasoning RL. PRM-based PPO gives a denser, step-level signal that genuinely helps search, but it needs per-step labels that are expensive to produce and, per R1-Zero, turn out to be substitutable with more outcome-supervised GRPO. So the choice is not arbitrary: GRPO is the cheapest algorithm that still explores and that pairs cleanly with a rule-based outcome reward, which is exactly the regime reasoning lives in.

A subtle but important property: because the advantage is *group-relative*, GRPO is insensitive to the absolute scale of the reward. Whether correct answers score 1 and wrong ones 0, or correct score 100 and wrong score 0, the standardized advantage is identical. This makes reward design easier — you tune the *relative* shape (how much a format bonus matters versus correctness) rather than absolute magnitudes, and the standardization keeps the gradient well-conditioned even as the fraction of correct answers in a group shifts from near-zero early in training to near-one late in training.

## 4. DeepSeek-R1-Zero: test-time scaling that emerges from pure RL

**Senior rule of thumb: the strongest evidence that a capability is real, not prompted, is when it appears without anyone asking for it — and R1-Zero's growing chains are exactly that.**

If o1 proved the paradigm behind closed doors, DeepSeek-R1 proved it in the open, with weights and a paper (arXiv:2501.12948, January 22, 2025). The most scientifically important part of that paper is not the full R1 model — it is R1-Zero, the ablation that strips everything away and asks: what does *pure* RL, with no supervised fine-tuning whatsoever, do to a base model?

The setup is deliberately minimal. Take the DeepSeek-V3 base model — 671B parameters, a mixture-of-experts — and run GRPO on it directly, with rule-based rewards on math and code, and *no SFT stage at all*. No human-written reasoning traces, no warm-up on curated chains. Just the base model, a verifier, and the GRPO loop. The conventional wisdom before this was that you need SFT to teach a model the *format* of good reasoning before RL can refine it. R1-Zero shows you do not.

The figure below is the curve that made the paper famous.

![A line chart with reinforcement-learning training steps on the x-axis and AIME pass-at-one on the y-axis, showing a single curve rising from about sixteen percent at step zero to seventy-one percent, annotated with an aha-moment point where the model learns to re-check its own work and a note that response length grows as training proceeds so each answer spends more test-time compute](/imgs/blogs/reasoning-models-rl-test-time-scaling-5.png)

Two things happen simultaneously over the course of training, and they are causally linked. First, AIME pass@1 climbs from **15.6% at the start to 71.0%** — a 55-point gain from RL alone, on a model that received zero supervised reasoning examples. Second, and this is the part nobody coded, the model's *average response length grows on its own*. As training proceeds, the model spontaneously writes longer and longer chains of thought, because longer chains — when they include exploration and self-checking — lead to more correct answers, and the reward selects for them. **Test-time scaling emerged from the RL.** The model was never told to think longer; it discovered that thinking longer pays.

### The "aha moment"

The DeepSeek paper documents a specific qualitative phenomenon they call the "aha moment": at a certain point in training, the model begins to exhibit self-reflection — it writes things like "wait, let me reconsider this step," pauses mid-solution to re-examine its own work, and allocates more thinking to a sub-problem it initially got wrong. The marked point on the curve is roughly where this behavior crystallizes, and it coincides with an inflection in both accuracy and length. The reason this matters beyond being a cute anecdote: self-reflection is the behavioral signature of the same explore-and-verify process OpenAI described for o1. Two independent labs, with different models and methods, converged on the same emergent behavior because the reward landscape rewards it. That convergence is the strongest argument that long-CoT reasoning is a real property of RL on verifiable rewards, not an artifact of one lab's pipeline.

### The cons@64 result and what it matches

R1-Zero's single-sample number is 71.0%, but the paper also reports the consensus result: **cons@64 reaches 86.7%**, which matches **OpenAI-o1-0912** (an early o1 checkpoint) on AIME 2024. Read that carefully — an open model, trained with pure RL and no SFT, using a majority vote of 64 samples, matched a frontier closed reasoning model. The progression 15.6% → 71.0% (pass@1) → 86.7% (cons@64) is the same shape as o1's 74 → 83 → 93: a strong single-sample policy, lifted further by spending test-time compute on aggregation. The absolute numbers differ because the base models and AIME checkpoints differ, but the *structure* is identical, and that structural match across two independent efforts is the result.

### Why pure RL works without any SFT

The result that surprised practitioners most is the *no-SFT* part. The prior belief was that RL on a raw base model would be hopeless: the base model rarely produces a fully correct long chain, so almost every sampled group would be all-zeros, the advantages would all be zero, and there would be no gradient signal to learn from — a cold-start problem. R1-Zero shows this fear is overstated, and the reason is instructive. A 671B base model, having read enormous amounts of human mathematical reasoning, already has a *non-trivial* probability of stumbling onto a correct chain — its pass@1 starts at 15.6%, not 0%. In a group of 16 samples, with a 15.6% per-sample success rate, the probability that *at least one* sample is correct is $1 - (1 - 0.156)^{16} \approx 1 - 0.844^{16} \approx 0.93$. So roughly 93% of groups contain at least one correct answer and at least one incorrect one — which means roughly 93% of groups produce a non-zero spread of rewards and therefore a usable gradient. The base model's latent competence is the seed, and GRPO's group sampling is precisely the mechanism that harvests it: you do not need the model to be reliably correct, only occasionally correct, because the group structure turns "occasionally correct" into a steady stream of contrastive learning signal. As training proceeds and the success rate rises, the group composition shifts, but the standardized advantage keeps the signal well-scaled the whole way. This is why a strong base matters: the seed competence must be high enough that some sampled group members succeed, which is also why the technique works best on large bases and on problems the base can *sometimes* solve. On a problem the base never solves, every group is all-zeros, the gradient vanishes, and RL cannot help — another face of the hardest-tail limit.

### The catch R1-Zero exposed

Pure RL is not free of problems, and the paper is honest about them. R1-Zero's outputs, while accurate, had poor readability and mixed languages mid-chain — the model optimized purely for getting the right answer and did not care whether a human could follow along. The reward checked correctness, not legibility, so the model had no incentive toward either. This is a small, instructive example of reward specification: you get exactly what you reward, and "be correct" does not imply "be readable." Fixing it is the entire reason full R1 exists, which is the next section.

## 5. Full DeepSeek-R1: wrapping pure RL in a four-stage pipeline

**Senior rule of thumb: pure RL proves a capability exists; a production pipeline wraps it in stages that fix the things pure RL ignores, like readability and helpfulness.**

R1-Zero answered the scientific question — does pure RL work? — and the answer is yes. But its readability and language-mixing problems make it unpleasant to actually use. Full DeepSeek-R1 is the productionized version: it keeps the pure-RL reasoning stage at its core but prepends a cold-start SFT stage to fix formatting and appends two more stages to broaden and align the model. The timeline below contrasts the one-stage R1-Zero with the four-stage R1.

![A timeline contrasting R1-Zero as a single pure reinforcement-learning stage on the V3 base with full R1 as four sequential stages, beginning with cold-start supervised fine-tuning on a small long chain-of-thought set, then reasoning reinforcement learning with verifiable rewards, then rejection sampling of the best chains followed by supervised fine-tuning again, and finally reinforcement learning from human feedback for helpfulness and harmlessness](/imgs/blogs/reasoning-models-rl-test-time-scaling-6.png)

The four stages of full R1, in order:

1. **Cold-start SFT.** Fine-tune the V3 base on a small, curated set of long chain-of-thought examples. The goal is not to teach reasoning — RL does that — but to teach the *format*: readable, single-language chains with a consistent structure. This solves the legibility problem that plagued R1-Zero by giving RL a clean starting point.
2. **Reasoning RL.** The same verifiable-reward GRPO loop as R1-Zero, now starting from the cold-started model. This is where the reasoning capability is actually built. Because the model already emits clean, well-formatted chains, the RL can focus on making them *correct* rather than fighting the formatting.
3. **Rejection sampling + SFT.** Once the reasoning RL has converged, sample many chains from it, keep only the best (correct, well-formatted) ones via rejection sampling, and fine-tune on that filtered set. This step also folds in non-reasoning capabilities (writing, factual QA) so the model is not a math-only savant. It is a distillation of the RL model's best behavior back into supervised form.
4. **RLHF for alignment.** A final RL stage optimizing for helpfulness and harmlessness, using human-preference-style rewards rather than verifiable ones. This is the conventional alignment stage, applied last so it does not interfere with the reasoning capability built in stage 2.

The headline outcome: **full R1 is on par with OpenAI-o1-1217** (the December 2024 o1 checkpoint) on math, code, and reasoning benchmarks. R1-Zero matched an *early* o1; full R1 matched the *contemporaneous* o1. The open reproduction was, at release, genuinely at the frontier.

### Why the pure-RL core still matters

You might ask: if full R1 needs four stages, was R1-Zero's "pure RL works" claim oversold? No — and the distinction is important for how you think about the technique. The reasoning *capability* comes entirely from stage 2's RL; stages 1, 3, and 4 are about format, breadth, and alignment, none of which create reasoning. R1-Zero proves that the capability-generating step is pure RL on verifiable rewards. The other stages are the difference between a research artifact and a product, and they would be there regardless of how the reasoning was created. When you are reasoning about *scaling*, the stage that matters is stage 2, and the lever is RL compute against verifiable rewards.

### o1 and R1 side by side

It helps to put the two flagship results in one table, because the differences clarify what is paradigm and what is implementation detail.

| Dimension | OpenAI o1 | DeepSeek-R1 |
|---|---|---|
| Disclosure | Blog post, no paper, no weights | arXiv paper, open weights |
| Training signal | Large-scale RL (details undisclosed) | GRPO with rule-based verifiable rewards |
| Pure-RL ablation | Not published | R1-Zero: pure RL, no SFT |
| Base model | Undisclosed | DeepSeek-V3, 671B MoE |
| AIME 2024, single sample | 74% | 71.0% (R1-Zero pass@1) |
| AIME 2024, aggregated | 83% maj@64, 93% reranker@1000 | 86.7% cons@64 (R1-Zero) |
| Reasoning trace | Hidden (summary shown) | Fully visible |
| Reported scaling | Log-linear in train-RL and test-time compute | Emergent length growth + accuracy climb |
| Frontier parity | Defined the frontier | Full R1 on par with o1-1217 |

What is *paradigm* — true of both, and therefore likely true of the approach in general — is the trio: RL is the training lever, long-CoT is the learned behavior, and verifiable rewards are what make the signal honest. What is *implementation* — different between them — is the algorithm (o1's is undisclosed; R1 uses GRPO), the disclosure level, and whether a pure-RL ablation exists. The fact that two labs, one closed and one open, independently produced the same qualitative result with overlapping numbers is the strongest evidence that the paradigm is real and reproducible rather than a single lab's lucky pipeline. R1's open weights and visible traces also did something o1 could not: they let the entire research community verify the emergent behaviors directly, which is why "the aha moment" entered the field's vocabulary from the R1 paper and not the o1 blog. Openness, here, is not just a licensing choice — it is what turned a claim into a checkable, build-on-able result.

## 6. The significance: RL is the lever that bends the inference curve

**Senior rule of thumb: the repeated-sampling and test-time-search literature told you the inference curve exists; reasoning models told you how to train a model that climbs it efficiently.**

To see why reasoning models are the keystone of this whole series, line them up against the two posts that precede them. [Large Language Monkeys](/blog/machine-learning/scaling-laws/repeated-sampling-scaling-laws) showed that *coverage* — the fraction of problems solved by at least one of $k$ samples — follows an exponentiated power law in $k$: $\log(\text{coverage}) \approx a \cdot k^{-b}$, straight on a log-log plot over four orders of magnitude. That is a scaling law for raw sampling. [Compute-optimal test-time scaling](/blog/machine-learning/scaling-laws/test-time-compute-scaling) showed that you can do better than naive best-of-$N$ by allocating inference compute adaptively — sequential revision for easy problems, parallel search for hard ones — and beat a 14× larger model on the easy-to-medium regime.

Both of those works take the *model as given* and ask how to spend inference compute on top of it. Reasoning models close the loop: instead of bolting search onto a fixed model, you *train the model to produce the search internally*. The long chain-of-thought is, in effect, an internalized version of explore-backtrack-verify — the same moves the external search procedures make, but learned and run inside a single generation. RL with verifiable rewards is the training-side lever that produces a model whose every inference token is spent productively. That is what "RL unlocks the inference-scaling curve" means concretely: o1 and R1 are models for which the inference-compute axis is steep and clean, because RL shaped them to be.

The taxonomy below places reasoning models in the broader space of test-time compute strategies.

![A tree of test-time compute scaling axes branching into parallel sampling and sequential thinking, where parallel sampling splits into coverage with a verifier picking one answer and voting or reranking that plateaus without a verifier, and sequential thinking splits into process-reward-model-guided search, self-revision, and a learned long chain-of-thought that emerges from reinforcement learning as used by o1 and R1](/imgs/blogs/reasoning-models-rl-test-time-scaling-8.png)

The two top-level branches are the two ways to spend inference compute. *Parallel sampling* — draw many independent tries — splits into coverage (where a verifier picks the one correct answer, the Large Language Monkeys regime) and voting/reranking (which plateaus once there is no verifier to break ties). *Sequential thinking* — one long try — splits into externally-orchestrated methods like PRM-guided search and self-revision, and the learned long-CoT that o1 and R1 produce. The green node is where reasoning models live: they fold the sequential-thinking branch into the model's own weights via RL, and they can still be combined with the parallel branch (that is what cons@64 and the reranker@1000 do). The unifying insight is that all of these are the same resource — test-time compute — spent in different shapes, and the cleanest shape, a learned long chain, is the one RL produces.

### The numbers that prove the coupling

It is worth pinning the coupling to specific figures so it does not stay abstract. On o1: a single sample is 74%, but adding the *test-time* lever (majority@64) gets 83% and (reranker@1000) gets 93% — that is the inference axis, holding the trained policy fixed. On R1-Zero: pass@1 went 15.6% → 71.0% as the *training* lever (RL steps) increased, and then cons@64 added another 15+ points on top via the inference lever, reaching 86.7%. Both models show the same pattern: the training lever sets the single-sample ceiling, and the test-time lever lifts you toward it and beyond. Neither lever alone gets you to the top; the product does.

### Distillation: moving the capability into small, cheap models

One of the most practically consequential findings in the R1 paper is almost a footnote, but it changes the deployment calculus entirely: the reasoning capability of the big RL-trained model can be *distilled* into much smaller dense models by fine-tuning them on the big model's chains. DeepSeek distilled R1's reasoning into a family of Qwen- and Llama-based models from 1.5B up to 70B, and the small distilled models substantially outperformed same-size models that had only been trained conventionally. This matters because it separates *where the capability is created* from *where it is served*. The capability is created once, expensively, by RL on a 671B base. It is then served cheaply by a 7B or 14B distilled model that fits on a single accelerator.

Pair this with the [inference-aware scaling argument](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) and the strategy becomes coherent. Inference-aware scaling says: for high-traffic deployment, you want a small model trained on far more tokens, because per-token serving cost dominates the lifetime budget. Reasoning distillation says: you can give that small model frontier reasoning without doing frontier RL on it yourself. So the deployment-optimal artifact is a *small, over-trained, reasoning-distilled* model that thinks longer only when routed to — cheap to serve per token, capable of long-CoT reasoning when the query warrants it, and produced by distilling an expensive RL model that you train once and amortize across an entire model family.

There is a ceiling, and case study 7 below names it: a distilled model imitates the *form* of the teacher's reasoning without the teacher's trained self-correction dynamics, so it cannot exceed the teacher and degrades on problems unlike the distillation set. The R1 paper is explicit that distillation alone underperforms RL applied directly — distillation is a cost-efficient transfer, not a substitute for the RL that created the capability. The clean way to hold it: RL creates the reasoning, distillation copies it cheaply, and the copy is bounded by the original. For most deployments the copy is more than good enough, and the economics are decisively better.

## 7. Verification is the boundary: what scales cleanly and what plateaus

**Senior rule of thumb: before you bet on test-time scaling for a task, ask "can a machine check the answer?" — that single question predicts whether the curve will be clean or flat.**

We have hinted at the verification wall throughout; now we make it precise, because it is the most important practical constraint on the entire approach. The matrix below crosses aggregation methods against the two domain types — verifiable and open-ended — and the pattern is stark.

![A matrix crossing four aggregation methods against verifiable and open-ended domains, showing pass-at-one as a neutral baseline in both, majority-at-k scaling well in verifiable domains lifting AIME to eighty-six point seven percent but plateauing fast in open-ended ones, reward-model selection helping but capping in verifiable and plateauing in open-ended, and verifier-at-k as the best option reaching ninety-three percent on AIME but having no analog in open-ended domains where the wall is selection](/imgs/blogs/reasoning-models-rl-test-time-scaling-7.png)

Read the columns. In the **verifiable** column, every step up in aggregation sophistication buys accuracy: pass@1 is the baseline, majority@k scales well (cons@64 → 86.7% on AIME), reward-model selection helps further, and a true verifier@k is best (reranker@1000 → 93%). The reason is simple: when you can check answers, generating more candidates strictly helps, because you can always recognize and keep a correct one. Coverage — the chance that *at least one* of $k$ samples is right — rises monotonically, and a verifier converts coverage directly into solved problems.

Now read the **open-ended** column, and watch it go red. With no verifier, majority voting plateaus fast (there is often no single "answer" to vote on), reward-model selection plateaus after a few hundred samples (the reward model is itself imperfect and saturates), and there is no analog of a true verifier at all. The Large Language Monkeys paper made this precise: coverage keeps rising with $k$, but *solved* problems do not, because without a verifier you cannot tell which of your many candidates is the right one. The bottleneck shifts from *generation* to *selection*, and selection is the wall.

This is exactly why o1 and R1 demonstrate their cleanest scaling on math and code. AIME has integer answers you can compare; code has unit tests you can run. Those are the domains where the reward in training is honest *and* the aggregation at inference is reliable, so both levers work. The further you move from checkable answers — toward essays, strategy, open-ended dialogue — the more the training reward becomes a fallible learned model and the inference aggregation becomes guesswork, and the log-linear curve flattens into a plateau. When the next post in this series asks [whether pre-training hit a wall](/blog/machine-learning/scaling-laws/pretraining-wall-inference-scaling), this is the counter-question to keep ready: test-time scaling has its own wall, and that wall is verification.

### Partial verifiability and how the frontier is widening

The verifiable/open-ended split is not binary, and the most interesting work right now lives in the gray zone of *partial* verifiability — domains where you can check something, even if not the whole answer. Agentic tasks are the canonical example: an agent that books a flight or fixes a bug produces an outcome you can sometimes check (did the test suite go green? did the booking confirm?) even when the intermediate reasoning is unconstrained. SWE-bench-style software tasks are exactly this — the unit tests are the verifier, and that is why repeated sampling and reasoning-style RL both show strong, clean scaling on them, as the [repeated-sampling post](/blog/machine-learning/scaling-laws/repeated-sampling-scaling-laws) documents. The strategic implication is that the way to expand the reach of reasoning-model scaling is to *manufacture verifiers*: build test harnesses, simulators, formal specifications, and checkable rubrics that turn a previously open-ended task into a partially verifiable one. Every domain you can equip with an honest checker becomes a domain where both the training lever and the test-time lever start working. This reframes a chunk of applied ML research as "verifier engineering" — and it is why so much current effort goes into building executable benchmarks rather than just collecting more text. The wall is verification, but the wall can be moved by building the verifier, and that, more than raw model scale, is where a lot of the next round of gains will come from.

### A worked example of the selection bottleneck

Make it concrete. Suppose a model has a 1% chance of producing a correct AIME answer on any single sample (a deliberately weak model). Coverage after $k$ samples is $1 - (1-0.01)^k$. After $k = 1000$ samples, coverage is $1 - 0.99^{1000} \approx 1 - 4.3\times10^{-5} \approx 99.99\%$ — almost certainly at least one of the thousand is correct. In a **verifiable** domain, you run all thousand answers through the checker, find the correct one, and solve the problem: coverage converts directly to a solve. In an **open-ended** domain, you have a thousand candidates, one of which is right, and *no way to identify it* — your majority vote and your reward model both point at the wrong one as often as not. The same 99.99% coverage yields a solve rate near your selection accuracy, which for a hard open-ended task might be 20%. That gap — 99.99% coverage versus 20% solves — is the verification wall in a single arithmetic example, and it is why "more samples" is a real scaling axis in one column and a mirage in the other.

## 8. A concrete cost and compute walk-through

**Senior rule of thumb: test-time scaling is only a good deal when the marginal accuracy per marginal dollar beats training a bigger base — and that comparison is domain- and budget-dependent.**

Let us put rough numbers on the trade, because "spend more at inference" is only smart up to a point. Recall from [the inference-aware post](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) that inference costs roughly $2N$ FLOPs per generated token, where $N$ is the parameter count. A reasoning model's long chains make this bite: where a non-reasoning model might emit 200 tokens for an answer, a reasoning model might emit 5,000 — a 25× increase in tokens, and therefore a 25× increase in inference FLOPs *per query*, before you even add majority voting or reranking on top.

Stack the multipliers for the o1-style 93% configuration. Start with one long chain at, say, 5,000 tokens. The reranker@1000 result samples a thousand such chains: that is $1000 \times 5000 = 5{,}000{,}000$ generated tokens for a single AIME problem, plus the reranker's own scoring passes. At $2N$ FLOPs per token, the inference cost of one hard problem solved at 93% is enormous — orders of magnitude more than a single 200-token GPT-4o answer that lands at 12%. The question is whether the accuracy is worth it, and the honest answer is: *it depends entirely on what a correct answer is worth to you.*

| Configuration | Tokens per query (order of magnitude) | AIME 2024 accuracy | When it makes sense |
|---|---|---|---|
| GPT-4o single answer | ~200 | 12% | High volume, accuracy not critical |
| o1 single chain | ~5,000 | 74% | You need correctness and can pay 25× the tokens |
| o1 majority@64 | ~320,000 | 83% | Correctness matters more than cost |
| o1 reranker@1000 | ~5,000,000 | 93% | A correct answer is worth a lot (research, high-stakes) |

The shape of this table is the whole economic argument for reasoning models, and it connects straight back to [compute-optimal test-time scaling](/blog/machine-learning/scaling-laws/test-time-compute-scaling): you do not want to pay reranker@1000 prices on every query. You want to *route* — spend a single chain on easy queries, escalate to voting and reranking only on the hard ones, and only in verifiable domains where the extra samples actually convert. The adaptive, per-query allocation that the test-time-scaling post argues for is the missing piece that makes the reasoning-model economics work at scale. A flat policy of "always sample 1000 and rerank" would bankrupt you; a difficulty-aware policy that escalates only when needed captures most of the accuracy at a fraction of the cost.

There is also a training-side cost worth naming. Because GRPO samples $G$ long chains per prompt per step, and those chains grow over training, the *training* of a reasoning model is dominated by inference-style rollout cost. RL on a 671B base with group size 16 and multi-thousand-token chains generates a staggering volume of tokens just to compute gradients. This is why reasoning-model training is often described as "inference-bound training" and why the same infrastructure investments — fast sampling, efficient KV-cache use — pay off in both training and serving.

### A break-even calculation for difficulty routing

Make the routing argument quantitative, because it is the single decision that decides whether reasoning models are affordable. Suppose your traffic is 90% easy queries and 10% hard ones, and a correct hard answer is worth \$1 to you while a correct easy answer is worth \$0.01 (easy answers are commodity). A non-reasoning model answers easy queries at, say, 95% accuracy and hard queries at 12% for ~200 tokens each. A reasoning model with a single chain answers hard queries at 74% for ~5,000 tokens — 25× the tokens. Should you route?

For the easy 90% of traffic, the reasoning model adds almost nothing (95% is already near-ceiling) while costing 25× more, so a flat "always reason" policy wastes 25× the compute on 90% of queries for no accuracy gain — that is the blowout from case study 8. For the hard 10%, the reasoning model lifts accuracy from 12% to 74%, a 62-point gain on answers worth \$1 each. If you serve a million queries, the hard 100,000 of them gain $0.62 \times 100{,}000 = 62{,}000$ dollars of expected value from routing them to the reasoning path, against an extra inference cost that — even at 25× tokens and generous per-token assumptions — is typically a small fraction of that. The break-even is lopsidedly in favor of *routing*: pay the 25× premium only on the 10% of queries where it converts to valuable correctness, and never on the 90% where it does not. A difficulty classifier that is even roughly accurate at separating the two buckets captures nearly all the value at a fraction of the flat-policy cost. This is the inference-side analog of compute-optimal training: spend the marginal compute where its marginal accuracy is worth the most, query by query.

## 9. Case studies and failure modes from the reasoning-model era

The general principles are clearer when you see where they bite. These are drawn from the public record of o1, DeepSeek-R1, and the broader reasoning-model rollout through early 2025.

### 1. The base-model-gains-nothing-from-prompting trap

A team with a strong 70B base model reads the o1 announcement and tries to replicate it with prompting: "think step by step, check your work, consider multiple approaches." Accuracy on their math benchmark barely moves, and on some problems it *drops*. The wrong first hypothesis is that their base model is too small. The actual root cause is that prompting cannot create a trained self-correction habit — the base model produces longer chains that elaborate errors as readily as truths, because nothing rewarded it for catching mistakes. The fix is RL with verifiable rewards, not a better prompt. The lesson, and it is the central one: **reasoning is a trained policy, not a prompt, and you cannot prompt your way to a capability the model was never optimized for.**

### 2. The readability collapse of pure RL

Following R1-Zero, a team runs pure GRPO on their base with a correctness-only reward and gets a large accuracy jump — and outputs that mix three languages mid-sentence and are nearly unreadable. The wrong hypothesis is a tokenizer bug. The actual root cause is reward specification: the reward checked only the final answer, so the model had zero incentive to keep the chain legible or monolingual, and RL exploited that freedom. The fix is exactly DeepSeek's: a cold-start SFT stage to establish format before RL, and a format component in the reward. The lesson: **you get precisely what you reward; "be correct" does not imply "be readable," and any property you care about must appear in the reward or the pipeline.**

### 3. The selection wall on an open-ended task

A team applies the repeated-sampling recipe — sample 500, pick the best — to a creative-writing benchmark and is baffled when accuracy plateaus around sample 50 despite coverage clearly still rising. The wrong hypothesis is that the model has hit its ceiling. The actual root cause is that there is no verifier for "good writing," so their reward-model-based selection saturates: more candidates do not help when you cannot reliably identify the good one. The fix is not more samples; it is either a better selection signal or accepting that this domain does not scale with test-time compute the way math does. The lesson: **coverage is not solves without a verifier, and selection — not generation — is the bottleneck in open-ended domains.**

### 4. The over-long-chain accuracy regression

A team fine-tunes a reasoning model and, wanting more "thinking," reward-shapes for longer chains directly. Accuracy goes *down*. The wrong hypothesis is that the model needs even longer chains. The actual root cause is that rewarding length rather than correctness teaches the model to pad — it generates verbose chains that add tokens without adding exploration or verification, and the extra tokens are opportunities to drift. The fix is to reward only the outcome and let length grow as a *byproduct*, the way it did emergently in R1-Zero. The lesson: **let test-time compute emerge from an outcome reward; do not reward the proxy (length) directly, or the model optimizes the proxy instead of the goal.**

### 5. The cons@64 plateau on the hardest tail

A team sees cons@64 lift their AIME-style benchmark nicely on medium problems, so they push to cons@256 expecting more — and the hardest problems do not move at all. The wrong hypothesis is that 256 is still too few samples. The actual root cause matches the test-time-scaling finding: on the hardest problems, where the base policy almost never gets it right, majority voting just amplifies the most common *wrong* answer, and no amount of voting helps. The fix is to recognize that the hardest tail needs a better policy (more RL training) or a true verifier, not more votes. The lesson: **majority voting helps only when the correct answer is already the plurality; on the hardest problems it cements the wrong one, and the training lever, not the inference lever, is what moves that tail.**

### 6. The verifiable-reward over-fitting to the checker

A team uses a unit-test suite as the code reward and gets a model that aces the tests — and writes code that passes the specific tests while being subtly wrong on inputs the tests do not cover. The wrong hypothesis is that the model genuinely solved the task. The actual root cause is that the verifier was incomplete, so "pass the tests" and "be correct" diverged, and RL found the gap. The fix is a stronger, more adversarial test suite, the same way you would harden any reward against gaming. The lesson: **a verifier is only as honest as it is complete; an incomplete checker is a learned-reward-model problem in disguise, and the model will find the seam.**

### 7. The distillation shortcut that skips the RL

A team, seeing that R1 distillation into smaller models works well, concludes they can skip RL entirely and just distill o1/R1 outputs into their base via SFT. It works — to a point — and then their model cannot exceed the teacher and reasons poorly on problems unlike the distillation set. The wrong hypothesis is that distillation is a full substitute for RL. The actual root cause is that SFT on traces imitates the surface of reasoning without the trained self-correction that RL instills; the distilled model has no mechanism to improve beyond what it copied. The fix, if you want frontier reasoning, is RL on top of (or instead of) distillation. The lesson: **distillation transfers the form of reasoning cheaply, but the capability to scale with test-time compute comes from RL; imitation has a ceiling at the teacher.**

### 8. The inference-cost blowout in production

A team ships a reasoning model to a high-traffic product with a flat "always think long" policy and watches inference cost explode 25× overnight, because every query — including "what's 2+2"-grade ones — now generates a multi-thousand-token chain. The wrong hypothesis is that they need cheaper hardware. The actual root cause is the absence of difficulty-aware routing: most queries do not need long chains, and paying for them anyway is pure waste. The fix is adaptive compute — a cheap classifier or the model's own confidence gating whether to engage the long chain. The lesson: **test-time compute is a per-query decision, not a global setting; the economics only work if you spend the compute where it changes the answer.**

### 9. The cold-start problem on a too-small base

A team tries to reproduce R1-Zero's pure-RL recipe on a 1.5B base and gets nowhere — accuracy barely budges over thousands of RL steps. The wrong hypothesis is that their RL hyperparameters are mis-tuned. The actual root cause is the seed-competence problem: a 1.5B base has a near-zero pass@1 on competition math, so almost every sampled group is all-incorrect, the group spread is zero, and there is no gradient. RL cannot amplify a competence the base does not have. The fix is either a much stronger base, or a cold-start SFT stage (or distillation from a larger reasoning model) to lift the base's pass@1 above zero so the group sampling has something to harvest. The lesson: **RL amplifies latent competence, it does not create it from nothing; if your base never succeeds, no amount of RL will teach it, and the seed must come from scale or from SFT first.**

### 10. The benchmark that leaked into the RL prompts

A team runs reasoning RL and reports spectacular gains on their internal math benchmark — then finds the model fails on a held-out fresh set. The wrong hypothesis is that the fresh set is harder. The actual root cause is that the RL training prompts overlapped with the benchmark, so the verifiable reward effectively trained the model to memorize answers to the eval. Because the reward was honest (correct answers really were correct), nothing in the loop flagged the leak; the model just learned the cheapest path to reward, which was recall. The fix is strict train/eval separation for RL prompts, the same hygiene you would apply to any supervised set, plus periodic evaluation on freshly-sourced problems. The lesson: **a verifiable reward is honest about correctness but blind to leakage; if the eval is in the training prompts, RL will memorize it, and only held-out, freshly-sourced problems tell you whether reasoning generalized.**

## 10. What this means in practice

If you are deciding whether and how to use reasoning models, the principles above collapse into a short set of decisions. The overarching one: **treat RL-with-verifiable-rewards as a training-time investment that pays off as a steeper, cleaner test-time scaling curve — and only commit to it where you can check answers.**

Concretely, here is how to apply this:

- **Establish whether your domain is verifiable first.** This is the gating question. If a machine can check the answer (math, code, formal proofs, structured extraction with ground truth), both the training reward and the inference aggregation will be honest, and the full reasoning-model playbook applies. If it cannot, expect the inference curve to plateau and budget accordingly.
- **Do not try to prompt your way to reasoning.** A base model gains little from "think longer." If you need a reasoning capability, you need RL (or distillation from an RL-trained model), not a prompt. Plan for the training-side investment.
- **Reward the outcome, not the proxy.** Let chain length, exploration, and self-correction emerge from a correctness reward, the way they did in R1-Zero. Rewarding length directly teaches padding; rewarding "looks like reasoning" teaches imitation. Keep the reward as close to the true goal as your verifier allows.
- **Use a cold-start SFT stage if you care about output quality.** Pure RL produces correct but often unreadable chains. A small cold-start on well-formatted traces, plus a format term in the reward, fixes legibility without harming the reasoning RL — this is the difference between R1-Zero and full R1.
- **Harden your verifier.** An incomplete checker (a thin test suite, a weak numeric comparison) is a reward-hacking surface. The cleanliness of the whole scaling story depends on the reward being honest, which depends on the verifier being complete.
- **Route test-time compute by difficulty.** Never apply reranker@1000 prices to every query. Spend a single chain on easy queries and escalate to voting or reranking only on the hard ones — and only in verifiable domains. This adaptive allocation, from the [test-time-scaling post](/blog/machine-learning/scaling-laws/test-time-compute-scaling), is what makes the economics survive contact with production traffic.
- **Combine a small over-trained base with heavy test-time compute.** The [inference-aware scaling](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) argument and the reasoning-model paradigm point the same way: a right-sized, cheap-to-serve base, taught by RL to think longer when it matters, beats a giant base on total cost in verifiable domains.
- **Remember both levers move the curve, but on different problems.** The test-time lever lifts you toward your policy's ceiling; the training lever raises the ceiling and is the only thing that moves the hardest tail. If voting has plateaued on your hardest problems, the answer is more RL, not more votes.

### When to reach for reasoning models

Reach for an RL-trained reasoning model with heavy test-time compute when: your domain is verifiable (math, code, proofs, checkable extraction); a correct answer is valuable enough to justify large per-query inference cost; you have the infrastructure to do RL rollouts (or can distill from an existing RL-trained model); and your hardest problems are the ones that matter, since that is where the training lever earns its keep.

### When not to

Skip the reasoning-model playbook when: your task is open-ended with no verifier (creative writing, open dialogue, subjective judgment), where the inference curve plateaus and the training reward is a fallible learned model; your queries are overwhelmingly easy, where a long chain is wasted compute; latency or cost ceilings make multi-thousand-token chains infeasible; or a strong non-reasoning model already clears your accuracy bar — in which case the 25× inference premium buys you nothing. Reasoning models are a sharp tool for a specific shape of problem, not a universal upgrade, and the honest framing in the [pre-training-wall synthesis](/blog/machine-learning/scaling-laws/pretraining-wall-inference-scaling) is the right one: a new axis to scale, gated by verification, not a replacement for everything that came before.

The mental shift the whole series has been building toward lands here. For four years the only lever was a bigger pre-trained model, and the scaling laws told you how much loss a bigger model would buy. Reasoning models add a second lever — RL that teaches a model to spend test-time compute well — and a second curve: accuracy that is log-linear in both training-RL compute and inference compute, clean wherever a verifier exists. The two o1 numbers to leave with are the boundary markers of that curve: 74% from a single trained chain, 93% from a thousand chains reranked. The distance between them is the test-time lever; the distance from GPT-4o's 12% up to 74% is the training lever; and the fact that DeepSeek-R1 walked that same path in the open, from 15.6% to 71.0% with pure RL, is the proof that the lever is real, reproducible, and now in everyone's hands.

## Further reading

- OpenAI, "Learning to Reason with LLMs" (September 12, 2024) — the o1 announcement; the source of the dual log-linear claim and the AIME 74/83/93 numbers. Blog only, no arXiv. https://openai.com/index/learning-to-reason-with-llms/
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (arXiv:2501.12948, January 22, 2025) — R1-Zero, the emergent length growth, the aha moment, and the full four-stage R1. https://arxiv.org/abs/2501.12948
- Brown et al., "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling" (arXiv:2407.21787, 2024) — the coverage scaling law and the verification caveat. https://arxiv.org/abs/2407.21787
- Snell et al., "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters" (arXiv:2408.03314, 2024) — compute-optimal, difficulty-adaptive test-time allocation. https://arxiv.org/abs/2408.03314
- Sardana & Frankle et al., "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws" (arXiv:2401.00448, ICML 2024) — the inference-cost reframing that pairs with reasoning models. https://arxiv.org/abs/2401.00448
- Sibling posts in this series: [repeated sampling as a scaling law](/blog/machine-learning/scaling-laws/repeated-sampling-scaling-laws), [compute-optimal test-time scaling](/blog/machine-learning/scaling-laws/test-time-compute-scaling), and [did pre-training hit a wall](/blog/machine-learning/scaling-laws/pretraining-wall-inference-scaling).

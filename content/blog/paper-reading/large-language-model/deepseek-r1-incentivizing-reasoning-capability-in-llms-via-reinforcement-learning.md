---
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
date: "2026-06-05"
publishDate: "2026-06-05"
description: "A close read of the DeepSeek-R1 paper — how pure reinforcement learning on a verifiable reward grew reasoning from scratch in R1-Zero, how the four-stage R1 pipeline made it usable, and why rule-based rewards and distillation reset the field's defaults."
tags: ["deepseek-r1", "reasoning", "reinforcement-learning", "grpo", "rlvr", "chain-of-thought", "distillation", "llm", "o1", "paper-reading"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 30
---

Some papers are interesting; a few reset everyone's defaults. DeepSeek-R1, released in January 2025, is firmly in the second category. Before it, the standard recipe for a capable model was supervised fine-tuning followed by RLHF with a learned reward model, and "reasoning" was something you coaxed out with prompting and clever data. After it, the default first move for any verifiable task became "run reinforcement learning on a rule-based reward and let the reasoning emerge." The paper didn't just report a strong model — it demonstrated that long chain-of-thought reasoning, self-verification, and the now-famous "aha moment" could be *grown* by RL on a base model with no reasoning supervision at all. That single result, R1-Zero, is one of the most important empirical findings in recent LLM history, and the engineering around it in the full R1 model is a master class in how to turn a research curiosity into a usable, o1-class system.

This is a close read of the paper. I want to walk through what R1-Zero actually did, why its reward design was the crucial choice, what the four-stage R1 pipeline added on top, how it reached OpenAI o1 parity as an open model, and why the distillation results may matter even more than the flagship model. If you want the algorithmic foundation in depth, I've written a separate deep dive on [fine-tuning with GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo), and the [decision guide on GRPO vs DPO vs PPO](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) places R1's method in the broader post-training landscape.

![The four-stage DeepSeek-R1 pipeline: alternating supervised fine-tuning and reinforcement learning turns a base model into a reasoning model](/imgs/blogs/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning-1.png)

The diagram above is the mental model for the whole paper. There are really two models here: **R1-Zero**, the pure-RL experiment that proves reasoning can emerge from a verifiable reward alone, and **R1**, the production model that wraps that core in a four-stage pipeline — a cold-start supervised warm-up, a reasoning-focused RL stage, a rejection-sampling-plus-SFT stage, and a final RL stage covering all use cases. The paper's narrative runs through both: R1-Zero is the scientific result, R1 is the engineering that made it ship. We'll take them in that order.

## The Setup: What Problem R1-Zero Tackled

The question the DeepSeek team posed is deceptively simple: *can reasoning ability be incentivized by reinforcement learning alone, without any supervised reasoning data?* The prevailing assumption in early 2025 was no — that you needed large amounts of high-quality chain-of-thought demonstrations to teach a model to reason, and that RL could only polish a capability the model already had from supervised data. OpenAI's o1 had shown that inference-time reasoning scaled impressively, but the recipe was closed, and the open community largely assumed it rested on a mountain of curated reasoning traces.

R1-Zero attacks this assumption head-on. The team took DeepSeek-V3-Base — a strong but non-reasoning base model — and ran reinforcement learning directly on it, with no supervised fine-tuning step in between. No reasoning demonstrations, no chain-of-thought examples, nothing to imitate. The only signal was a reward that checked whether the model's final answer was correct. The hypothesis was that if you reward correctness and let the model explore how to get there, the reasoning would develop on its own as an instrumental strategy for earning reward.

This is a genuinely different bet from RLHF. In RLHF, the reward model encodes human preferences over style and helpfulness, and the RL stage aligns the model to those preferences — it's polishing behavior the SFT stage already installed. R1-Zero's bet is that RL can *create* a capability, not just align an existing one. If it worked, it would mean the bottleneck for reasoning wasn't demonstration data; it was just giving the model a verifiable goal and enough RL to find its way there.

## R1-Zero: Reasoning from Pure RL

It worked, and dramatically. The picture below is the R1-Zero loop and its outcome.

![DeepSeek-R1-Zero: with no supervised warm-up, GRPO on a rule reward alone grew AIME accuracy from 15.6 to 71.0 and produced emergent chains of thought](/imgs/blogs/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning-2.png)

The mechanism is GRPO — Group Relative Policy Optimization — running on DeepSeek-V3-Base. For each problem, the policy samples a group of candidate solutions, each is scored by the rule-based reward, and the policy is updated to favor the higher-scoring members of the group relative to the group's mean. There's no value network (GRPO's defining simplification) and no learned reward model. The model generates, gets told which generations were correct, and adjusts.

The headline result: on AIME 2024, a hard competition-math benchmark, R1-Zero's pass@1 accuracy climbed from **15.6% to 71.0%** over the course of RL training — and with majority voting over 64 samples (cons@64), it reached **86.7%**, matching OpenAI o1's level. From a base model that could barely touch these problems, pure RL on a correctness signal produced competition-level mathematical reasoning. No reasoning demonstrations were ever shown to the model.

What makes this more than a benchmark number is *how* the model got there. Over training, R1-Zero spontaneously learned to allocate more thinking to harder problems — its responses grew longer, not because anyone rewarded length, but because longer chains of reasoning earned more correct answers. It developed self-verification, re-deriving results to check them. It learned to backtrack when a line of reasoning wasn't working. None of these behaviors were supervised; all of them emerged as instrumental strategies for getting the answer right. The capability the field assumed required demonstration data turned out to be reachable through reward and exploration.

It's worth being precise about the training dynamics, because they're the most striking part of the result. The paper plots two curves over RL steps that move together: average response length and benchmark accuracy. Both rise steadily and roughly in tandem. The model is not being told "write more"; it's being told "be correct," and it discovers that spending more tokens on intermediate reasoning is the route to correctness. This is the clearest possible demonstration that inference-time compute — thinking longer at test time — is something a model can learn to *want* to do, given the right training signal. The length growth is a side effect of the model finding that deliberation pays, and the fact that it emerges from a pure correctness reward, with no length term anywhere, is exactly what makes it remarkable.

There's also a subtle but important point about exploration. For the group-relative advantage to teach anything, the group of samples for a prompt has to contain both better and worse solutions — there has to be variance to learn from. Early in training, on a problem the model can't yet solve, all samples are wrong and the group is uniform, so there's nothing to learn from that prompt. As the model improves, more prompts land in the productive zone where some samples are right and some are wrong, and those are the prompts that drive learning. The whole process is a slow expansion of the frontier of problems the model can partially solve, with each newly-reachable problem becoming a source of gradient as soon as the model can get it right sometimes.

## The Reward: Why Rules, Not a Neural Model

The single most important design decision in R1-Zero is the reward, and it's the part most worth dwelling on because it's where the paper quietly breaks with RLHF orthodoxy.

![Why rule-based rewards, not a neural reward model: a learned reward model gets hacked under RL pressure, so R1 uses a deterministic verifier](/imgs/blogs/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning-3.png)

R1 uses a **rule-based reward** with two components. The first is an **accuracy reward**: for math, the model is asked to put its final answer in a specified format (a box) so it can be extracted and checked against the known solution; for code, the answer is run against test cases. This is a deterministic verifier — it returns correct or incorrect with certainty, and it cannot be fooled by anything other than actually being right. The second is a **format reward**: the model must wrap its reasoning in `<think>` and `</think>` tags, separating the chain of thought from the final answer, and it earns a small reward for complying.

The team explicitly chose *not* to use a neural reward model, and they say why: under the optimization pressure of large-scale RL, a learned reward model gets hacked. The policy is an adversary that will find and exploit every blind spot in the reward model — producing outputs the reward model scores highly but that aren't actually good. A learned reward model also adds the overhead of training and maintaining a second large network, and it introduces a moving target. A deterministic verifier sidesteps all of this: there's nothing to hack, because correctness is checked, not estimated. The reward is cheap (running a verifier costs nothing compared to a forward pass through a reward model) and it's incorruptible.

This choice is the seed of what the field would later call RLVR — reinforcement learning from verifiable rewards — and it's a large part of why R1's approach scaled where reward-model RL had struggled. The price is a constraint: you can only use this on tasks where a verifier exists. Math, code, and structured output fit; open-ended helpfulness does not, which is exactly why the full R1 pipeline needs additional stages to cover non-verifiable behavior. But for the reasoning core, the rule-based reward is the choice that makes the whole thing work.

In rough pseudocode, the reward is as simple as it sounds:

```python
import re

def r1_reward(response: str, gold_answer: str) -> float:
    """R1's rule-based reward: accuracy (verifier) plus a small format bonus."""
    reward = 0.0

    # Format reward: did the model wrap its reasoning in think tags?
    if re.search(r"<think>.*?</think>", response, re.DOTALL):
        reward += 0.1

    # Accuracy reward: extract the boxed answer and check against the gold solution.
    match = re.search(r"\\boxed\{(.+?)\}", response)
    if match and match.group(1).strip() == gold_answer.strip():
        reward += 1.0

    return reward
```

No learned parameters, no training, no reward-model checkpoint to babysit. The entire reasoning signal for R1-Zero is a function you could write in an afternoon.

The format reward is worth a second look, because the `<think>...</think>` template it enforces turned out to be more consequential than it first appears. By requiring the model to wrap its reasoning in explicit tags and then give a final answer, the template cleanly separates the *process* (the chain of thought) from the *product* (the answer). That separation does three useful things at once. It makes the accuracy reward easy to compute — you extract the answer from outside the think block and check it, without the reasoning getting in the way. It gives the model a designated space to deliberate without that deliberation being mistaken for the final response. And it makes the model's reasoning legible to a reader or a downstream system that wants to inspect or hide the chain of thought. This template — visible thinking in a delimited block, then a clean answer — became a de facto standard across reasoning models in 2025, and you can trace its popularity directly to R1's format reward. A design choice made to enable a verifier ended up shaping how the whole field presents reasoning.

## The Aha Moment: Emergent Behaviors

The paper's most quoted passage is about what the authors called the "aha moment," and it's worth understanding precisely because it's easy to over-romanticize.

![What emerges over RL training: without being told to, the policy learns to write longer, to re-check its own work, and to pause before correcting course](/imgs/blogs/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning-4.png)

During training, the researchers observed R1-Zero, mid-solution, generate text along the lines of "wait, wait, let me reconsider" — pausing, re-evaluating its approach, and correcting course. The model had learned, purely from the reward signal, that stopping to reconsider a flawed line of reasoning led to more correct answers. The authors framed this as an "aha moment" not because the model is conscious of anything, but because a recognizable problem-solving behavior — the productive self-interruption a human solver does — emerged without ever being demonstrated.

Alongside the aha moment, several behaviors developed together, all visible in the before-and-after of training. Early in RL, responses were short and the model rarely checked its work; late in training, responses were long, structured chains of thought with explicit self-verification, and accuracy had climbed accordingly. The response length grew steadily, and the team was careful to frame this as the model learning to *think longer on harder problems* — allocating more inference-time computation where it helps. (As I discuss in the [follow-up on GRPO variants](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo), later work showed that some of this length growth was also an artifact of a length-normalization bias in vanilla GRPO — a useful corrective, but it doesn't erase that genuine reasoning depth was developing too.)

The deeper significance is what the emergence implies. If these behaviors can be incentivized rather than demonstrated, then the path to better reasoning isn't necessarily more curated chain-of-thought data — it's better verifiable rewards and more RL. That reframing is a large part of why the paper landed the way it did.

## R1-Zero's Problems — and Why R1 Exists

R1-Zero is a beautiful scientific result and an unpleasant model to actually use. Because it was trained with no supervised data and only a correctness-and-format reward, nothing in its training rewarded readability. Its chains of thought mix languages mid-solution (switching between English and Chinese), are often poorly formatted, and read as a stream of consciousness optimized for getting the answer rather than for a human to follow. The reasoning is correct; the presentation is rough.

This is the gap the full DeepSeek-R1 pipeline exists to close. The team's goal for R1 was to keep R1-Zero's reasoning power while producing output that's readable, consistent in language, and aligned with human preferences across all kinds of tasks — not just verifiable ones. They couldn't get there with pure RL from base, so they built a four-stage pipeline that combines the strengths of supervised fine-tuning (readability, alignment, coverage of non-verifiable tasks) with the strengths of RL (emergent reasoning depth).

## DeepSeek-R1: The Four-Stage Pipeline

Return to the first figure — the four stages — and now we can walk each one, because each was added to fix a specific shortcoming.

**Stage 1: Cold-start SFT.** Before any RL, the team fine-tunes DeepSeek-V3-Base on a few thousand high-quality long chain-of-thought examples. This "cold start" gives the model a readable reasoning format to begin from, rather than the rough output R1-Zero produced. The cold-start data is carefully curated for readability — proper formatting, a consistent language, a summary at the end — so the model starts the RL stage already producing human-friendly reasoning. This is the fix for R1-Zero's readability problem: seed the format with a little supervision, then let RL take over. The team gathered this data in a few ways — using R1-Zero itself to generate long-CoT examples and then cleaning them up, prompting with few-shot reasoning examples, and post-processing with human annotators for readability. The crucial design choice is that the cold start is *small*: a few thousand examples, not hundreds of thousands. It's just enough to establish a good format and stable starting point, deliberately not enough to teach the reasoning itself — that's still RL's job. Over-investing in cold-start SFT would risk the model imitating the demonstrations' ceiling instead of discovering better strategies through RL, so the team kept it minimal on purpose.

**Stage 2: Reasoning-oriented RL.** Now the model goes through large-scale RL much like R1-Zero — GRPO on the rule-based accuracy-and-format reward — to develop deep reasoning. The key addition here is a **language-consistency reward** that penalizes mixing languages within a response, directly addressing R1-Zero's language-mixing problem. The team notes this slightly reduces benchmark accuracy (forcing a single language is a mild constraint on the optimizer) but produces far more usable output — a deliberate trade of a fraction of a point for readability.

**Stage 3: Rejection sampling and SFT.** Once the reasoning RL converges, the team uses the resulting model to generate a large dataset and fine-tune on it. They sample many responses, keep the good ones (rejection sampling — filter by the verifier for reasoning tasks, and by a generative reward model or human-aligned criteria for others), and assemble roughly **800,000 training examples**: about **600,000 reasoning** samples and **200,000 non-reasoning** samples covering writing, factual QA, self-cognition, and general tasks. Fine-tuning DeepSeek-V3-Base on this mix broadens the model beyond pure reasoning, restoring general capabilities that a reasoning-only RL stage would neglect. This is the stage that turns a reasoning specialist into a generalist.

**Stage 4: RL for all scenarios.** Finally, a second RL stage aligns the model across all use cases — not just verifiable reasoning but helpfulness and harmlessness too. For reasoning tasks it keeps using rule-based rewards; for general tasks where there's no verifier, it brings in reward signals for helpfulness and safety. This final pass is what makes R1 a well-rounded assistant rather than a math-and-code engine, covering the non-verifiable behavior the rule-based reward can't reach.

| Stage | Method | What it adds | Fixes |
|---|---|---|---|
| 1. Cold-start SFT | SFT on long-CoT seeds | readable reasoning format | R1-Zero's rough output |
| 2. Reasoning RL | GRPO + rules + language reward | deep reasoning, one language | language mixing |
| 3. Rejection sample + SFT | 800k samples (600k reason + 200k general) | general capability | reasoning-only narrowness |
| 4. Final RL | rules + helpfulness/safety | alignment across all tasks | non-verifiable coverage |

The elegance of this pipeline is that it alternates the two paradigms to play each to its strength: SFT for format, readability, and broad coverage; RL for emergent reasoning depth and final alignment. Neither alone gets you R1; the interleaving does.

## The GRPO Algorithm

Since GRPO is the engine of both RL stages, it's worth stating precisely. For a question $q$, GRPO samples a group of $G$ outputs $\{o_1, \dots, o_G\}$ from the old policy $\pi_{\theta_{\text{old}}}$, scores each with the reward to get $\{R_1, \dots, R_G\}$, and computes each output's advantage relative to the group:

$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_1, \dots, R_G\})}{\text{std}(\{R_1, \dots, R_G\})}$$

The policy is then updated to maximize a clipped objective with a KL penalty against a reference policy:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\min\!\Big(r_i(\theta)\hat{A}_i,\ \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i\Big) - \beta\, D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})\right]$$

where $r_i(\theta) = \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}$ is the importance ratio, $\epsilon$ is the clip range, and $\beta$ scales the KL term. The crucial simplification versus PPO is the advantage: PPO learns a value network (a critic, the same size as the policy) to estimate the baseline, while GRPO just uses the group's mean reward. For a verifiable task where you can cheaply sample a whole group and score each member, the group mean is a perfectly good baseline — so GRPO drops the critic entirely, roughly halving the memory footprint of the RL stage. That efficiency is a big part of why training reasoning at this scale was feasible. The mechanics, hyperparameters, and failure modes of GRPO are covered in depth in the [dedicated GRPO guide](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo); here the point is just that R1's reasoning rests on this critic-free, group-baselined policy gradient.

## Results: Matching o1 as an Open Model

The benchmark story is the one that made R1 front-page news.

![DeepSeek-R1 matches o1-1217 on reasoning: R1 reaches o1-class accuracy as an open model, while R1-Zero shows how far pure RL gets before the cold-start polish](/imgs/blogs/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning-6.png)

On AIME 2024, DeepSeek-R1 scored **79.8% pass@1**, slightly ahead of OpenAI o1-1217's 79.2%. On MATH-500 it reached **97.3%**, again matching o1. On competitive programming it landed around the **96th percentile** on Codeforces, comparable to o1. Across the reasoning benchmarks that defined the frontier at the time, R1 was at o1's level — and it was an open model with published weights and a published method, where o1 was a closed system. The cost framing made the splash even bigger: R1 reached this level at a small fraction of the API cost of the closed alternatives.

The matrix also shows R1-Zero's numbers next to R1's, which is instructive. R1-Zero, with pure RL and no cold start, already reached 71.0% on AIME — most of the way to the final model. The four-stage pipeline closed the remaining gap to 79.8% *and*, more importantly, fixed the readability and generality that the raw benchmark number doesn't capture. The lesson: the reasoning came overwhelmingly from RL; the pipeline's extra stages were about making that reasoning usable and well-rounded, not about the bulk of the capability.

| Benchmark | R1-Zero | DeepSeek-R1 | OpenAI o1-1217 |
|---|---|---|---|
| AIME 2024 (pass@1) | 71.0% | 79.8% | 79.2% |
| MATH-500 | 95.9% | 97.3% | 96.4% |
| Codeforces (percentile) | — | ~96.3 | ~96.6 |

The story isn't only math and code. Because the third and fourth stages of the pipeline deliberately broaden the model beyond verifiable reasoning, R1 is also strong on general knowledge and language benchmarks — it posts competitive numbers on MMLU and GPQA Diamond (graduate-level science questions), and on instruction-following and writing evaluations it holds up as a general assistant, not just a reasoning engine. This breadth is the payoff of the 200k non-reasoning samples in the rejection-sampling stage and the helpfulness-and-safety rewards in the final RL stage. A model trained with reasoning-only RL would ace AIME and fumble a request to write a polite email; R1 does both, and that roundedness is what made it usable as a general product rather than a benchmark specialist. It's a useful reminder that the headline reasoning numbers are necessary but not sufficient — the engineering that made R1 *deployable* is in the stages that don't show up on the math leaderboard.

A note on reading these numbers responsibly: benchmark scores at this level are close enough that small differences are within noise and depend on evaluation details (sampling temperature, number of samples, prompt format). The right reading of the table is not "R1 beats o1 by 0.6 points on AIME" but "R1 reached o1's tier" — the achievement is parity from an open model, not a decimal-point lead. Reasoning models also tend to show large gaps between pass@1 and majority-voting scores, which is why R1-Zero's 71.0% pass@1 becomes 86.7% with cons@64: the model often *can* solve a problem but doesn't always do so on the first sample, and aggregating samples recovers much of that gap.

## Distillation: Reasoning for Smaller Models, Almost for Free

The result that may matter most in the long run is in the back half of the paper: distillation. The team took the 800k curated samples R1 produced and used them as plain supervised fine-tuning data to train a family of much smaller dense models.

![Distilling R1 reasoning into smaller models: R1's reasoning traces, used as SFT data, transfer its reasoning to small models better than running RL on them directly](/imgs/blogs/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning-5.png)

They distilled into Qwen and Llama backbones across a range of sizes — 1.5B, 7B, 8B, 14B, 32B, and 70B — by simply fine-tuning each on R1's reasoning traces. No RL on the small models, just SFT on the teacher's outputs. The distilled models inherited a remarkable amount of R1's reasoning ability: the distilled 32B and 70B models posted reasoning benchmark scores that beat much larger non-reasoning models and even strong reasoning baselines like QwQ-32B-preview, and the smaller distilled models punched far above their weight.

The paper's most pointed finding is the comparison the team ran deliberately: distilling R1 into a small model *outperforms running RL directly on that small model*. They took a small base model, ran the same large-scale RL R1-Zero used, and found it underperformed simply distilling from R1 — while costing far more compute. The interpretation is that the reasoning patterns a large, capable model discovers through RL are hard for a small model to discover on its own through RL, but easy for it to absorb through imitation once the large model has found them. RL on a strong model is how you *discover* reasoning; distillation is how you *propagate* it cheaply to smaller models. For practitioners, this is enormous: you don't need to run expensive RL for every model size — you run it once on a capable model and distill down. It's the mechanism by which R1's breakthrough became a fleet of small, open, reasoning-capable models almost immediately.

The specific numbers drive the point home. The distilled DeepSeek-R1-Distill-Qwen-32B reached competition-math and coding scores that beat OpenAI's o1-mini on several reasoning benchmarks — a 32B dense model, trained with nothing but SFT on R1's traces, competing with a frontier reasoning product. The distilled 7B and 14B models posted scores that would have been state-of-the-art for their size only months earlier. And because the recipe is just "fine-tune on R1's outputs," anyone with the released traces and a few GPUs could reproduce it. This is the part of the paper that changed what a small team could build: reasoning ability stopped being something only a frontier lab with a massive RL budget could produce, and became something you could distill into a 7B model over a weekend. The distilled checkpoints DeepSeek released — across Qwen and Llama backbones — seeded an entire ecosystem of open reasoning models almost overnight.

There's a deeper implication worth stating. Distillation working this well suggests that the hard part of reasoning is *discovery* — finding the strategies, the self-verification habits, the productive deliberation — and that once discovered, those patterns are compressible into a form a much smaller model can imitate. The capability isn't fundamentally tied to scale; the *discovery* of it benefits from scale, but the result transfers down. That reframes the scaling story: you scale to discover, then distill to deploy.

## What Didn't Work: PRM and MCTS

A paper-reading should cover the negative results, because they're as informative as the wins, and R1's are unusually candid. The team reports two approaches they tried and abandoned.

The first is **process reward models (PRM)** — rewarding the model for each intermediate reasoning step being correct, rather than only the final answer. In principle this gives denser, more informative feedback. In practice the team found it hard to make work: defining what counts as a correct intermediate step is genuinely difficult at scale, training a reliable step-level reward model is hard, and — the recurring theme — a learned PRM is susceptible to reward hacking, with the policy learning to produce steps that score well without leading anywhere. The dense signal wasn't worth the fragility.

The second is **Monte Carlo Tree Search (MCTS)** — the search technique behind AlphaGo, adapted to guide token generation by exploring a tree of reasoning paths. The team found it didn't scale to language: the search space of tokens is vastly larger than the legal moves in a board game, the value model needed to guide the search was hard to train well enough, and the whole approach didn't yield the gains that the simpler GRPO-on-verifiable-reward approach did. The honest reporting here matters — it tells you the team tried the sophisticated, theoretically-appealing approaches and found that the simple, scalable one (a clean reward and a lot of RL) won. That's a recurring lesson in this corner of the field: scalable and simple beats clever and fragile.

It's worth dwelling on why the simple approach won, because the reason generalizes. Both PRM and MCTS try to inject *more structure* into the learning signal — step-level rewards in one case, explicit search over reasoning paths in the other. That extra structure is exactly what makes them fragile at scale: a step-level reward needs a reliable notion of a "correct step," and a search needs a reliable value estimate, and both of those are learned components that break down or get exploited when pushed hard. The outcome-only verifiable reward sidesteps the whole problem by asking the model to figure out the structure of good reasoning *itself*, through exploration, rather than having that structure imposed by a fragile auxiliary model. The bet R1 makes — and wins — is that a large model given a trustworthy outcome signal and enough RL will discover better intermediate structure than you could hand-design with a process reward or a search procedure. That's a humbling result for anyone who likes elegant algorithms, and a recurring pattern in deep learning: the method that imposes the least structure, given enough scale and a clean signal, tends to win.

## R1 in Context: o1, Reproductions, and the Variant Wave

It helps to situate R1 against what came before and after, because its importance is partly about timing and openness.

Before R1, OpenAI's o1 had established that inference-time reasoning — letting a model think at length before answering — was a major new axis of capability. But o1 was a closed system: no weights, no method, no clarity on whether it rested on enormous amounts of curated reasoning data or some proprietary technique. The open community could see the destination but not the road. R1's contribution was to publish a road that reached the same destination, and to show that the road was simpler than many assumed — a clean verifiable reward and a lot of GRPO, not a mountain of secret reasoning traces. That R1 matched o1-1217 on the headline reasoning benchmarks, as an open model, is what made it a genuine inflection point rather than just a strong release.

After R1, two things happened fast. First, a wave of open reproductions — Open-R1 from Hugging Face, and many independent efforts — set out to replicate the recipe, and largely succeeded, confirming that the result was real and the method was robust. Second, the research community began stress-testing and refining GRPO itself. Within months, papers identified biases in the vanilla GRPO objective that R1 used — a length-normalization bias, a difficulty-normalization bias, instability on Mixture-of-Experts models — and proposed fixes (Dr. GRPO, DAPO, GSPO, and others). I cover that whole wave in [Beyond GRPO](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo), but the point here is that R1 didn't just produce a model; it produced a research agenda. By making verifiable-reward RL the default and publishing the exact objective, it gave the field a concrete, shared target to improve, and the improvements came quickly.

The lineage matters for anyone reading R1 today: the paper is the foundation, but the GRPO it describes is the *un-patched* version. If you're building on R1's method now, you'd apply the bias fixes from the variant wave — which is a sign of how generative the original was. A paper whose method gets refined by a dozen follow-ups within a year is a paper that opened a door.

## Practical Takeaways for Building Reasoning Models

For someone actually building a reasoning model, R1 distills into a handful of concrete lessons that have held up well.

**If your task is verifiable, start with RL on the verifier.** R1's central practical message is that you don't need a pile of chain-of-thought demonstrations to get reasoning — you need a reward that checks correctness and enough RL to let the model find its way there. For math, code, and any task with a checkable answer, a rule-based reward plus GRPO is a strong default, and it sidesteps the cost and fragility of a learned reward model entirely.

**Don't use a neural reward model where a verifier will do.** The reward-hacking failure mode is real and it gets worse under more optimization pressure. A deterministic verifier can't be gamed, costs almost nothing to evaluate, and never drifts. Reserve learned reward models for the genuinely non-verifiable parts of behavior — helpfulness, tone, safety — where there's no rule to check against.

**Use a small SFT cold start to fix readability, not to teach reasoning.** R1-Zero showed reasoning can come from RL alone, but its output was rough. A few thousand well-formatted long-CoT examples before RL buy you readable, language-consistent output without throwing away the emergent-reasoning benefit. The cold start is about format and presentation; the RL is about capability. Keep those roles distinct.

**Alternate SFT and RL rather than picking one.** R1's four-stage pipeline works because it plays each paradigm to its strength: SFT for format, breadth, and general capability; RL for emergent reasoning depth and final alignment. A reasoning-only RL model is narrow; a SFT-only model lacks the depth. The interleaving is the recipe.

**Discover with scale, deploy with distillation.** If you need reasoning at multiple model sizes, run the expensive RL once on a capable model and distill its traces into the smaller ones. Distillation outperforms running RL on the small models directly, at a fraction of the cost. This is the single biggest cost lever the paper hands you.

**Curate prompt difficulty.** Because the group-relative advantage needs variance within a group to produce a gradient, prompts that the model always gets right or always gets wrong contribute nothing. Keeping the training set populated with prompts in the productive middle — solvable sometimes but not always — is what keeps the RL learning efficiently. This operational detail, implicit in R1, became explicit in the dynamic-sampling fixes of the later variants, and it's the single most common reason a first RL run appears to stall when the algorithm is actually fine.

## Why This Paper Mattered

Stepping back, R1's significance is larger than its benchmark numbers, and it lands on several axes at once.

**It proved reasoning can be incentivized, not just demonstrated.** R1-Zero is an existence proof that a verifiable reward plus RL can grow reasoning from a non-reasoning base, with no chain-of-thought supervision. That reframed the bottleneck for reasoning from "we need more demonstration data" to "we need better verifiable rewards and more RL," and a great deal of 2025 research followed that reframing.

**It made rule-based, verifiable rewards the default for reasoning.** The deliberate rejection of neural reward models in favor of deterministic verifiers — to dodge reward hacking and the cost of a second network — became RLVR, the dominant paradigm for training reasoning models. Every GRPO variant that followed, the whole [2025 wave of DAPO, Dr. GRPO, and GSPO](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo), assumes this verifiable-reward setup as its starting point.

**It was open.** The weights, the method, and the distilled model family were all published. This collapsed the gap between closed frontier reasoning models and the open community almost overnight, and it gave researchers a concrete, reproducible recipe to build on rather than a black box to guess at. The open reproductions (Open-R1 and many others) that appeared within weeks are a direct consequence.

**It made distillation a first-class strategy.** By showing that R1's reasoning could be distilled into small models more effectively than running RL on them, the paper handed the community a cheap path to reasoning-capable small models. The practical effect was a sudden proliferation of small open reasoning models, all descended from one expensive RL run.

For the broader context of how R1's choices fit into the post-training toolkit — when a verifiable reward (and thus R1's approach) is the right call versus when you want preference-based DPO or a learned-reward PPO — the [GRPO vs DPO vs PPO decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) lays out the full decision, and [Training LLMs for Math](/blog/machine-learning/large-language-model/training-llm-for-math) walks the verifiable-reward pipeline R1 popularized end to end.

## Limitations and Open Questions

The paper is refreshingly honest about what R1 doesn't solve. The reasoning gains are concentrated in domains with verifiable rewards — math, code, logic — and the approach has less to offer where correctness can't be checked. R1 can still be sensitive to prompting, and like other reasoning models it can over-think simple problems, spending a long chain of thought where a short answer would do. The language-mixing problem is mitigated but not entirely gone.

The deepest open question is whether RL is truly *creating* new reasoning ability or *eliciting* latent ability already present in the base model from pretraining. This is not a pedantic distinction. If RL creates the capability, then the path to better reasoning is more and better RL. If RL merely surfaces what pretraining already installed, then the base model's pretraining mix is doing more of the work than the dramatic training curves suggest, and the ceiling is set earlier than it looks. The paper raises this but doesn't settle it, and the question turned out to be genuinely subtle. Later critical re-examinations of R1-Zero-like training found that strong base models often already exhibit reasoning-like behaviors before any RL — self-reflection, multi-step solutions — which means a meaningful fraction of what looks like "RL taught the model to reason" may be "RL elicited and sharpened reasoning the base model already had." That doesn't diminish R1's result, but it does complicate the cleanest reading of it, and it's a healthy reminder to control for base-model capability before crediting an RL algorithm with an emergent ability.

There are also practical caveats the paper is candid about. The pipeline is expensive — the RL stages on a model of V3's scale are a serious compute investment, which is precisely why the distillation result matters so much as the cheaper path for everyone downstream. And the approach inherits the limits of its verifiers: a verifier that's subtly wrong, or that can be satisfied without genuinely solving the task, will train a model to exploit it. The clean reward is only as trustworthy as the verifier behind it, and building good verifiers for harder domains than competition math remains an open engineering problem.

None of these caveats diminish the core contribution. R1 showed, openly and reproducibly, that reasoning can be grown by reinforcement learning on a verifiable reward, that a clean rule-based reward beats a hackable neural one, that a thoughtful SFT-RL pipeline turns a research result into a usable model, and that distillation spreads the gains cheaply. Those four findings reset the field's defaults, and nearly everything in reasoning-model training since has built on them. If you read only one reasoning-model paper to understand how the modern recipe came to be, this is the one — not because it's the final word, but because it's the one that opened the conversation everyone has been having ever since.

## Further Reading

- [Fine-Tuning LLMs with GRPO: From Theory to Implementation](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) — the algorithm at R1's core, in depth.
- [GRPO vs DPO vs PPO: A Decision Guide for Post-Training LLMs](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) — where R1's verifiable-reward RL fits among the alternatives.
- [Beyond GRPO: DAPO, Dr. GRPO, GSPO, and the Loss-Aggregation Fixes of 2025](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo) — how the field refined GRPO after R1.
- [Training LLMs for Math](/blog/machine-learning/large-language-model/training-llm-for-math) — the verifiable-reward reasoning pipeline end to end.
- [DeepSeek-R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — the original paper.

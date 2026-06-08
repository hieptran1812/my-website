---
title: "Kimi k1.5: Scaling Reinforcement Learning with LLMs"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "paper-reading"
subcategory: "Reinforcement Learning"
tags:
  - kimi-k1-5
  - reinforcement-learning
  - long-context
  - chain-of-thought
  - reasoning
  - mirror-descent
  - long2short
  - multimodal
  - paper-reading
description: "A principal-engineer read of Kimi k1.5: how a deliberately simple RL recipe — long-context chain-of-thought, online mirror descent, outcome rewards, and long2short distillation — reaches o1-level reasoning without MCTS, value functions, or process reward models."
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/kimi-k1-5-1.png"
readTime: 33
---

There is a ceiling built into next-token prediction, and everyone training large models has been quietly walking toward it. The recipe that produced GPT-style models is beautifully simple — predict the next token over an enormous corpus, scale the parameters and the data, and capabilities emerge. But the data is finite. You can scale compute all you want; once you have ingested the readable internet, the supply of high-quality human text that teaches a model *new* things runs out. The scaling law does not break, but it stops being cheap. The question the field has been circling is: what is the *next* axis you scale once you have run out of text to imitate?

Kimi k1.5, from Moonshot AI's Kimi Team ([arXiv:2501.12599](https://arxiv.org/abs/2501.12599)), answers with reinforcement learning — but not the kind of RL that the post-ChatGPT world made famous. RLHF tunes a pretrained model toward human preferences; it polishes behavior the model already has. The RL in k1.5 is doing something more ambitious: it lets the model *generate its own training signal* by exploring a reasoning space and getting rewarded for reaching correct answers. The model writes a long chain of thought, checks whether it landed on the right answer, and the reward pulls the whole trajectory up or down. Crucially, the model can scale the *length* of that reasoning — and the paper's central empirical finding is that as you give RL a longer context to reason in, accuracy keeps climbing. The new axis is not more data. It is more *thinking per problem*, learned end to end.

What makes the paper worth a close read is not that it reaches o1-level numbers — 77.5 on AIME 2024, 96.2 on MATH-500, the 94th percentile on Codeforces — but *how plainly* it gets there. The contemporaneous orthodoxy for "reasoning RL" assumed you needed Monte Carlo tree search at inference, a value network to assign credit to intermediate steps, and a process reward model to score the reasoning rather than just the answer. Kimi k1.5 throws all three away. It uses a single scalar outcome reward, a policy-gradient update derived from online mirror descent with a mean-reward baseline, and a long autoregressive chain of thought that the authors argue *is* the search — an implicit one, learned rather than hand-built. That is a strong, falsifiable claim about what actually matters in reasoning RL, and the rest of this post is about pressure-testing it.

![The k1.5 RL system: rollout fans out to rewards, then merges to one update](/imgs/blogs/kimi-k1-5-1.png)

The diagram above is the mental model: a central master drives a synchronous loop where rollout workers generate chains of thought, those rollouts fan out to specialized reward models (code with a real execution service, a chain-of-thought math reward model at 98.5% spot-check accuracy, and vision/K-12 reward models), a replay buffer holds the unfinished tail of any over-long trajectory, and every reward merges back into a single gradient step that produces the next reference policy. No tree. No critic. No per-step labels. One loop, repeated, scaling context as it goes. The whole paper is an argument that this is enough.

> [!tldr] TL;DR
> - **What it claims.** A simplistic RL recipe — long-context chain-of-thought up to 128k tokens, online mirror descent with a mean-reward baseline, and outcome-only rewards — reaches o1-level reasoning (77.5 AIME, 96.2 MATH-500, 94th-percentile Codeforces, 74.9 MathVista) without MCTS, value functions, or process reward models.
> - **Why it matters.** It reframes the scaling question: instead of scaling pretraining data, you scale the *length of learned reasoning*. Response length and accuracy grow together across RL iterations, and the paper treats reasoning tokens as the compute budget of an implicit search.
> - **Most surprising finding.** A *short*-CoT model distilled from the long one (via "long2short") beats GPT-4o and Claude 3.5 Sonnet by up to ~+550% on AIME (60.8 vs GPT-4o's 9.3) while using only ~3,272 tokens on average — the long-CoT prior transfers into a token-efficient model.
> - **Where it fails / is thin.** No parameter count, GPU-hours, or token budgets are disclosed; credit assignment stays coarse (outcome reward only); "overthinking" (uncontrolled length growth) is an inherent failure mode the length penalty only partially fixes; and o1 still leads on LiveCodeBench (67.2 vs 62.5) and MMMU (77.3 vs 70.0).

## Context: what came before

To see why k1.5 is interesting, you have to see the three things it is reacting against.

The **first** is the data ceiling of pretraining itself. Next-token prediction is a supervised imitation objective: the loss measures how well the model reproduces human-written continuations. That works astonishingly well, and it scales with compute — but the paper states the constraint bluntly: it is *bounded by the amount of available training data*. RL is attractive precisely because it sidesteps this. The model is not imitating a fixed corpus; it is generating trajectories, scoring them against a reward, and learning from its own exploration. In the authors' framing, RL "unlocks a new axis" because the model can scale its own training data by learning to explore with rewards. You are no longer rate-limited by how much text humans wrote.

The **second** is the disappointing track record of RL on LLMs. The paper is candid that "prior published work has not produced competitive results" — RL-on-LLMs had a reputation for being finicky, sample-inefficient, and prone to collapse. Plenty of papers reported small gains on narrow benchmarks; very few reported a *recipe that scales* to frontier reasoning. So the contribution here is partly sociological: this is a report of an RL pipeline that actually worked at scale, written so others can reproduce the ingredients.

The **third**, and the one that shapes the method most, is the assumption that frontier reasoning requires *explicit search*. The mental model coming out of AlphaGo and into the o1 era was that strong reasoning looks like planning: you build a tree of partial solutions, use a value function to estimate how promising each node is, score reasoning steps with a process reward model, and run Monte Carlo tree search to navigate. That machinery is powerful but heavy — it is expensive at inference, it introduces a value function that can systematically penalize exploratory-but-correct detours, and a process reward model is itself a hard thing to train and a rich surface for reward hacking.

Kimi k1.5's core thesis is that **scaling context length** does the same job more cheaply. A long autoregressive chain of thought can backtrack, second-guess, try an approach, abandon it, and try another — all within a single generated sequence. The authors argue this is an *implicit search* over reasoning space: the number of reasoning tokens plays the role of the compute budget of a planning algorithm, and you train the model to *approximate* search rather than build an explicit tree at inference time. That is the gap this paper fills. Everyone agreed reasoning needs search-like behavior; k1.5 argues you can get it by scaling context and letting RL teach the model to spend that context well, instead of bolting a search algorithm onto the side.

## Contributions

Reading the report end to end, the substantive contributions are:

1. **Long-context RL scaling as the primary lever.** RL training is run all the way to a 131,072-token (128k) context, and the paper shows empirically that training accuracy and response length grow *together* across iterations, with harder benchmarks showing steeper length growth. This reframes "scale" from parameters/data to reasoning length.
2. **A deliberately minimal policy-optimization algorithm.** A variant of online (policy) mirror descent with a closed-form softmax optimum, approximated by a policy gradient with a *mean-reward baseline* and an L2 off-policy regularizer — and explicitly **no value network**.
3. **A length penalty that fights "overthinking."** RL has a tendency to blow up response length; a length reward (warmed up gradually) rewards shorter correct answers and penalizes long wrong ones, recovering token efficiency.
4. **Sampling strategy: curriculum + prioritized.** Start easy and progress to hard (curriculum), and sample problems in proportion to how often the model currently fails them (prioritized), focusing compute on the model's weak spots.
5. **long2short transfer.** Four methods — model merging, shortest rejection sampling, DPO, and a dedicated long2short RL phase — to distill the long-CoT reasoning prior into a token-efficient *short*-CoT model that still wins benchmarks.
6. **RL prompt-set curation with reward-hacking defenses.** Difficulty labeling via pass rate, plus explicit filters that exclude multiple-choice/true-false/proof questions and remove prompts a model can guess without any reasoning.
7. **Partial rollouts — the infra trick that makes long-CoT RL affordable.** A fixed token budget per rollout, with over-budget trajectory tails saved to a replay buffer and continued in later iterations, so a single 128k-token chain of thought does not stall the whole synchronous loop.

## Method

The method has two layers. The *algorithmic* layer is the RL update and the reward design. The *systems* layer is how you actually run long-CoT RL without setting your cluster on fire. The paper's quiet genius is that the two are co-designed: the simplicity of the algorithm is what makes the systems tractable, and the systems trick (partial rollouts) is what makes the long-context scaling — the whole point — affordable. Let us walk the pipeline first, then go deep on each component.

![Four stages from raw tokens to a 128k-context reasoner](/imgs/blogs/kimi-k1-5-2.png)

The pipeline above has four ordered stages. **Pretraining** runs in three sub-stages (vision-language pretraining to establish a language foundation and gradually integrate multimodality; a cooldown stage that consolidates with curated and synthetic reasoning/knowledge data; and a long-context activation stage that extends the context window to 131,072 tokens). **Vanilla SFT** fine-tunes on roughly 1M text examples plus roughly 1M text-vision examples. **Long-CoT SFT** is a small but high-quality warmup set of accurately verified reasoning paths, designed to teach the model the *shape* of good reasoning — planning, evaluation, reflection, exploration — across both text and image. Then **reinforcement learning**, the stage the paper actually focuses on, takes over. The fifth box, **long2short**, is a transfer step we will treat as its own topic; it takes the expensive long-CoT model and produces a cheap short-CoT one.

Two notes on the SFT stages, because they matter for the RL that follows. The vanilla SFT corpus splits into 500k general QA, 200k coding, 200k math and science, 5k creative writing, and 20k long-context examples (summarization, document QA, translation, writing). SFT is trained at 32k sequence length for one epoch, then 128k for one epoch — the learning rate decays from 2e-5 to 2e-6 in the 32k stage, then re-warms to 1e-5 and decays to 1e-6 in the 128k stage, with examples packed into single sequences for efficiency. The long-CoT SFT set is deliberately *small*: it is a warmup, not the main event. Its job is to give RL a starting policy that already knows how to write a structured, reflective chain of thought, so RL is refining a behavior rather than discovering it from scratch.

### Why explicit search is the wrong abstraction here

Before the math, it helps to be concrete about what the paper is rejecting. The figure below lays the two recipes side by side.

![What k1.5 deliberately throws away from the o1-style RL recipe](/imgs/blogs/kimi-k1-5-3.png)

On the left is the heavy recipe: Monte Carlo tree search to plan, a value network or critic to estimate per-step advantage, and a process reward model to assign dense step-level labels. Each of those is a real engineering object with real failure modes. A value network, in particular, has a subtle problem the paper calls out directly: it tends to penalize exploratory steps that *look* low-advantage in the moment but are exactly the trial-and-error moves that make a long chain of thought work. If your reasoning model is supposed to try a wrong approach, notice it is wrong, and recover, a critic that punishes the wrong approach mid-stream is fighting the behavior you want.

On the right is k1.5: a long chain of thought *is* the search, a mean-reward baseline replaces the value network, and a single outcome reward $r \in \{0, 1\}$ replaces the process reward model. The reward comes from rules and test-case execution where the answer is verifiable, and from a trained chain-of-thought reward model (98.5% spot-check accuracy on math) where it is free-form. The bet is that you lose almost nothing by going coarse on credit assignment, and you gain an enormous amount of simplicity and robustness. The results are the evidence for the bet.

### The policy-optimization objective

Now the core algorithm. We have a dataset $D$ of problems $x$ with reference answers $y^*$. The policy $\pi_\theta$ generates a chain of thought $z$ and a final answer $y$. The reward $r(x, y, y^*) \in \{0, 1\}$ scores whether the answer is correct. RL proceeds in iterations; at iteration $i$, the *current* model $\pi_{\theta_i}$ is frozen and used as a reference, and we optimize a relative-entropy-regularized objective (Eq. 2 in the paper):

$$
\max_\theta \; \mathbb{E}_{(x, y^*) \sim D} \, \mathbb{E}_{(y, z) \sim \pi_\theta} \Big[ r(x, y, y^*) \Big] \; - \; \tau \, \mathrm{KL}\big(\pi_\theta(x) \,\|\, \pi_{\theta_i}(x)\big)
$$

The first term says "get more reward." The second term, with temperature $\tau > 0$, says "do not drift too far from the reference policy in one step." This is the online-mirror-descent structure: each iteration is a *trust-region* update around the previous policy. The reason this matters is stability — unconstrained policy gradient on a language model is a fast route to collapse, and the KL term keeps each step honest.

This objective has a closed-form optimum. The optimal policy is a reward-tilted version of the reference:

$$
\pi^*(y, z \mid x) = \pi_{\theta_i}(y, z \mid x) \cdot \frac{\exp\big(r(x, y, y^*)/\tau\big)}{Z}, \qquad Z = \sum_{y', z'} \pi_{\theta_i}(y', z' \mid x) \cdot \exp\big(r(x, y', y^*)/\tau\big)
$$

You cannot use this directly — $Z$ sums over all possible responses — but it tells you the *target* the policy should move toward. Taking the log and rearranging gives a surrogate loss that the policy should satisfy at the optimum (a squared off-policy constraint):

$$
L(\theta) = \mathbb{E}\Big[ \big( r(x, y, y^*) - \tau \log Z - \tau \log \tfrac{\pi_\theta(y, z \mid x)}{\pi_{\theta_i}(y, z \mid x)} \big)^2 \Big]
$$

The term $\tau \log Z$ is the normalizer; the paper approximates it from $k$ sampled responses, and in practice uses the **empirical mean of sampled rewards** $\bar{r} = \mathrm{mean}\big(r(x, y_1, y^*), \dots, r(x, y_k, y^*)\big)$ as the baseline. That is the whole reason there is no value network: the mean reward over $k$ samples of the *same problem* is a perfectly good, zero-extra-parameters estimate of the baseline that a critic would otherwise provide. Differentiating the surrogate gives the gradient actually used in training (Eq. 3): for each problem, sample $k$ responses from the reference $\pi_{\theta_i}$, and update

$$
\frac{1}{k} \sum_{j=1}^{k} \Big[ \nabla_\theta \log \pi_\theta(y_j, z_j \mid x) \cdot (r_j - \bar{r}) \; - \; \frac{\tau}{2} \, \nabla_\theta \big( \log \tfrac{\pi_\theta(y_j, z_j \mid x)}{\pi_{\theta_i}(y_j, z_j \mid x)} \big)^2 \Big]
$$

Read that gradient in two pieces. The first term is a **policy gradient with a mean-reward baseline**: responses that beat the average ($r_j > \bar{r}$) get their log-probability pushed up, responses below average get pushed down. The second term is an **L2 regularizer** on the log-ratio against the reference — it is the gradient form of the trust region, penalizing the policy for moving too far. There is a subtle but important detail: the $k$ responses are sampled *off-policy* from the fixed reference $\pi_{\theta_i}$, not from the live $\pi_\theta$. That is what lets the system **reuse off-policy data** across the iteration — you sample once from the reference and can take multiple gradient steps — and it is what makes partial rollouts (later) sound.

Here is the update as a single training step, written as pseudocode so the moving parts are explicit:

```python
def k15_rl_step(policy, reference, batch, tau, length_reward_fn, k=8):
    """One online-mirror-descent iteration over a batch of problems.

    reference == pi_{theta_i} is frozen for the whole iteration.
    Responses are sampled OFF-POLICY from the reference, so the same
    rollouts can drive several gradient steps and feed the replay buffer.
    """
    total_loss = 0.0
    for x, y_star in batch:
        # sample k chains of thought from the FROZEN reference policy
        samples = reference.generate(x, num_samples=k)          # each = (z, y)

        # outcome reward in {0,1}: rules / test-exec if verifiable, else CoT RM
        rewards = [outcome_reward(x, y, y_star) for (z, y) in samples]

        # anti-overthinking length shaping (warmed up; see length_reward_fn)
        rewards = [r + length_reward_fn(z, correct=(r > 0)) for r, (z, _) in
                   zip(rewards, samples)]

        r_bar = sum(rewards) / k                                 # mean-reward baseline
        for (z, y), r in zip(samples, rewards):
            logp = policy.log_prob(x, z, y)                      # grad flows here
            logp_ref = reference.log_prob(x, z, y).detach()      # frozen
            log_ratio = logp - logp_ref

            pg = -(r - r_bar) * logp                             # policy gradient
            l2 = 0.5 * tau * (log_ratio ** 2)                    # trust-region term
            total_loss = total_loss + (pg + l2) / k

    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return total_loss.item()
```

One operational detail that is easy to miss but is load-bearing: **the optimizer is reset at the start of each iteration**. Because the reference policy changes every iteration, each iteration is genuinely a *different* optimization problem — carrying Adam's momentum and second-moment estimates across that boundary would be applying statistics from one objective to another. Resetting the optimizer state each iteration is the correct thing to do for an iterated trust-region method, and it is the kind of detail that separates a recipe that works from one that mysteriously diverges.

### The length penalty: fighting overthinking

There is a failure mode that shows up the moment you reward correctness and let length scale: **overthinking**. The model discovers that longer chains of thought tend to be more correct, so it inflates length without bound — burning tokens, latency, and money for diminishing returns. The paper introduces a *length reward* to counteract this, and the design is careful.

For the $k$ sampled responses to a problem, let `min_len` and `max_len` be the shortest and longest response lengths. If `max_len == min_len` (all the same length), the length reward is 0 — there is nothing to discriminate. Otherwise, for response $i$:

$$
\lambda = 0.5 - \frac{\mathrm{len}(i) - \mathrm{min\_len}}{\mathrm{max\_len} - \mathrm{min\_len}}
$$

Then the length reward is $\lambda$ for **correct** responses (so shorter correct answers get a higher reward — $\lambda$ is largest when `len(i)` is near `min_len`) and $\min(0, \lambda)$ for **incorrect** responses (so a wrong answer never gets a length *bonus*, and a long wrong answer is explicitly penalized). This length reward is added to the original reward with a weighting parameter. In code:

```python
def length_reward(lengths, correctness):
    """Per-response length shaping over the k samples of one problem.

    Shorter correct answers score higher; long wrong answers are penalized.
    Returns 0 for every response when all responses share a length.
    """
    lo, hi = min(lengths), max(lengths)
    if hi == lo:
        return [0.0 for _ in lengths]          # nothing to discriminate

    out = []
    for length, is_correct in zip(lengths, correctness):
        lam = 0.5 - (length - lo) / (hi - lo)  # in [-0.5, 0.5], higher = shorter
        if is_correct:
            out.append(lam)                    # reward brevity when right
        else:
            out.append(min(0.0, lam))          # never reward a wrong answer's length
    return out
```

The crucial implementation note is the **warmup**: the length penalty is *not* applied from the start. The paper trains without it first, then switches on a constant length penalty for the rest of training. The reason is that a length penalty applied too early slows the model's exploration — early in RL you *want* the model to ramble and discover long reasoning patterns, and clamping length prematurely starves that discovery. Let it learn to think long, *then* teach it to think tersely. This ordering — capability first, efficiency second — recurs throughout the paper and is one of its more transferable lessons.

### Sampling: curriculum and prioritized

Not all problems are worth the same compute at the same time. The paper uses two sampling strategies that stack.

**Curriculum sampling** uses the fact that the training data carries grade/difficulty labels. Training starts on easy problems and progresses to hard ones. The motivation is concrete: very early in RL, the model fails hard problems almost always, so a batch of hard problems yields almost no correct samples — and with a mean-reward baseline, a batch where everything is wrong (or everything is right) produces a near-zero advantage signal and a wasted step. Spend the early budget where the model can actually get a useful mix of right and wrong answers, then escalate.

**Prioritized sampling** tracks a per-problem success rate $s_i$ and samples problems in proportion to $1 - s_i$. Problems the model nails ($s_i \to 1$) get sampled rarely; problems it struggles with ($s_i \to 0$) get sampled often. This is a clean, parameter-free way to keep the gradient focused on the model's current frontier of competence — the problems where there is the most to learn. Here is the combined sampler as pseudocode:

```python
def sample_problem_batch(pool, success_rate, stage, batch_size):
    """Curriculum gate (by difficulty stage) + prioritized weighting (by 1 - s_i)."""
    # curriculum: only admit problems at or below the current difficulty stage
    eligible = [p for p in pool if p.difficulty <= stage]

    # prioritized: weight each eligible problem by how often we currently fail it
    weights = [max(1.0 - success_rate.get(p.id, 0.0), 1e-3) for p in eligible]
    total = sum(weights)
    probs = [w / total for w in weights]

    return weighted_sample(eligible, probs, k=batch_size)
```

These two strategies are individually simple and jointly powerful: curriculum controls *which difficulty band* is in play, prioritized controls *which problems within the band* get the compute. The ablations (later) show curriculum alone gives a large lift over uniform sampling.

### Reward design and prompt curation

The reward is the contract between the model and reality, and the paper spends real effort making it honest. The base reward is $r \in \{0, 1\}$. For verifiable problems — math with a checkable answer, code with test cases — the reward comes from rules or test-case execution, with a code execution service wired directly into the reward model. For free-form answers, a trained reward model scores correctness.

The reward-model choice is itself an ablation result. The paper compares a **Classic** value-head reward model (InstructGPT-style, ~800k training points) against a **Chain-of-Thought** reward model (~800k CoT-labeled examples) that emits step-by-step reasoning and then a correctness judgment in JSON. On a manual math spot-check, the Classic RM scores ~84.4% and the CoT RM scores ~98.5%. That gap is enormous for a component sitting in the reward loop — a 15-point error rate in your reward is a 15-point reward-hacking opportunity — so the CoT RM is the one used in RL.

Prompt-set curation enforces three properties: **diverse coverage** (STEM, coding, general reasoning), **balanced difficulty** (easy/moderate/hard), and **accurate evaluability**. Difficulty is labeled by a model: an SFT model answers each prompt 10 times at high temperature, and the pass rate becomes a difficulty proxy. The reward-hacking defenses are the part worth copying:

- Exclude **multiple-choice, true/false, and proof-based** questions — all three are easy to get right by accident or by gaming, which corrupts the reward signal.
- For general QA, prompt a model to *guess the answer with no chain of thought*. If it gets the answer right within $N = 8$ attempts, the prompt is "too easy to hack" — the answer is guessable without reasoning, so RL would learn to guess rather than reason — and the prompt is removed.

That last filter is the kind of thing you only build after watching RL find a degenerate shortcut. It directly targets the failure where the model learns "I do not need to reason; for this distribution of prompts, the answer is usually X."

The coding reward deserves its own note because the numbers show how much work honest verification takes. The paper auto-generates test cases using the CYaRon library plus the base Kimi k1.5 model. Per problem, it generates 50 test cases and samples 10 ground-truth submissions; a test case is *valid* only if at least 7 of 10 submissions agree on its output, and a problem is admitted to training only if at least 9 of 10 submissions pass the selected test set. The funnel is sobering: from 1,000 online-contest problems, ~614 do not need a special judge, 463 generators are built that produce at least 40 valid test cases each, and only **323 problems** survive into the training set. Roughly a third. That is the real cost of a trustworthy code reward — most problems get thrown out because you cannot verify them cheaply and reliably.

### Partial rollouts: making 128k-token RL affordable

Here is the systems problem that long-context RL creates. In a synchronous RL loop, every iteration has a rollout phase (generate trajectories) and a training phase (update on them). If you let trajectories run to 128k tokens, the *longest* trajectory in a batch dictates how long the whole rollout phase takes — and a single pathological 128k chain of thought can stall thousands of GPUs while they wait. Naively, long-context RL is dominated by stragglers.

![Partial rollouts: how a 128k CoT survives a fixed token budget](/imgs/blogs/kimi-k1-5-5.png)

Partial rollouts solve this with a fixed output-token budget per rollout per iteration. If a trajectory exceeds the budget, its unfinished tail is saved to a **replay buffer** and continued in a later iteration — the trajectory is segmented across iterations $n - m, \dots, n$. Only the *current* segment (iteration $n$) needs fresh on-policy compute; earlier segments are reused from the buffer. This is exactly why the off-policy sampling in the gradient matters: because the algorithm samples from a frozen reference within the iteration, a tail produced in an earlier iteration is still legitimate training data. Rollout workers run **asynchronously** so a long trajectory cannot block the short ones in its batch. And the system includes **repeat detection** — it identifies repeated sequences, terminates them early, and can assign extra penalties to discourage the degenerate "loop forever" behavior that long-context generation invites.

Put together, partial rollouts convert "the longest trajectory dominates the iteration" into "every iteration does a bounded amount of work, and long trajectories amortize across iterations." That is the trick that makes scaling context to 128k a budgeting decision rather than a wall.

### The training/inference system

The reward and rollout machinery sit on top of a hybrid deployment that is worth sketching because it is the kind of thing teams underestimate. Training uses **Megatron**; inference uses **vLLM**; the two are *collocated in one pod* via Kubernetes sidecar containers that share all GPUs. A **Checkpoint Engine** manages weight onload/offload: after a training step, Megatron offloads its GPU memory; vLLM starts from a dummy model and then has its weights updated through shared memory. Transferring checkpoints between peer nodes uses **Mooncake** over **RDMA**. The whole design goal is to keep GPUs busy — minimize idle hardware, let inference nodes scale elastically while the training footprint stays constant. This is the unglamorous half of "RL that works at scale," and the paper is right to document it.

A table to anchor the components and what each one replaces:

| Component | k1.5 choice | What it replaces / avoids | Why |
|---|---|---|---|
| Credit assignment | Outcome reward $r \in \{0,1\}$ | Process reward model | Avoids a hard-to-train reward and a reward-hacking surface |
| Baseline | Mean reward $\bar{r}$ over $k$ samples | Value network / critic | Zero extra params; no penalty on exploratory detours |
| Search | Long autoregressive CoT (to 128k) | Monte Carlo tree search | Implicit, learned search; no inference-time tree |
| Trust region | KL / L2 on log-ratio to reference | Unconstrained PG | Stability across iterations |
| Off-policy data | Sample from frozen reference $\pi_{\theta_i}$ | On-policy only | Enables data reuse and partial rollouts |
| Straggler control | Partial rollouts + replay buffer | Wait for longest trajectory | Bounds per-iteration compute |
| Inference/training | vLLM + Megatron collocated | Separate clusters | Minimizes idle GPUs |

## Experiments

The headline is that this recipe reaches o1-level long-CoT reasoning and produces a short-CoT model that demolishes the non-reasoning frontier on math. Let us take the long-CoT results first.

![Long-CoT k1.5 against the flagship reasoning models](/imgs/blogs/kimi-k1-5-4.png)

The matrix above is Table 2 from the report. Against QwQ-32B-Preview, o1-mini, and OpenAI o1, the long-CoT k1.5 model matches or beats o1 on four of the six shared benchmarks. Here is the table with exact numbers:

| Benchmark (Metric) | QwQ-32B-Prev | o1-mini | QVQ-72B-Prev | OpenAI o1 | **k1.5 (long)** |
|---|---|---|---|---|---|
| MATH-500 (EM) | 90.6 | 90.0 | – | 94.8 | **96.2** |
| AIME 2024 (Pass@1) | 50.0 | 63.6 | – | 74.4 | **77.5** |
| Codeforces (Percentile) | 62 | 88 | – | 94 | **94** |
| LiveCodeBench (Pass@1) | 40.6 | 53.1 | – | **67.2** | 62.5 |
| MathVista-Test (Pass@1) | – | – | 71.4 | 71.0 | **74.9** |
| MMMU-Val (Pass@1) | – | – | 70.3 | **77.3** | 70.0 |
| MathVision-Full (Pass@1) | – | – | 35.9 | – | **38.6** |

The wins are real and on hard benchmarks: +1.4 on MATH-500 over o1 (96.2 vs 94.8), +3.1 on AIME (77.5 vs 74.4), a tie at the 94th percentile on Codeforces, and +3.9 on MathVista (74.9 vs 71.0). The honest losses are LiveCodeBench (62.5 vs o1's 67.2) and MMMU-Val (70.0 vs 77.3). So the claim "o1-level reasoning" is well supported on math and competitive programming and reads as slightly generous on the broader multimodal-understanding benchmark (MMMU) and on the live coding benchmark. That is a fair scorecard, and the paper does not hide the two losses.

Now the more surprising result — the *short*-CoT model.

![Short-CoT k1.5 versus the non-reasoning frontier](/imgs/blogs/kimi-k1-5-7.png)

This matrix is Table 3, comparing the short-CoT k1.5 against the non-reasoning frontier — Qwen2.5-72B-Instruct, LLaMA-3.1-405B-Instruct, DeepSeek-V3, Claude 3.5 Sonnet, and GPT-4o. The full table:

| Benchmark (Metric) | Qwen2.5-72B | LLaMA3.1-405B | DeepSeek-V3 | Claude-3.5-Sonnet | GPT-4o | **k1.5 (short)** |
|---|---|---|---|---|---|---|
| MMLU (EM) | 85.3 | 88.6 | 88.5 | 88.3 | 87.2 | 87.4 |
| IF-Eval (Prompt Strict) | 84.1 | 86.0 | 86.1 | 86.5 | 84.3 | **87.2** |
| CLUEWSC (EM) | 91.4 | 84.7 | 90.9 | 85.4 | 87.9 | **91.7** |
| C-Eval (EM) | 86.1 | 61.5 | 86.5 | 76.7 | 76.0 | **88.3** |
| MATH-500 (EM) | 80.0 | 73.8 | 90.2 | 78.3 | 74.6 | **94.6** |
| AIME 2024 (Pass@1) | 23.3 | 23.3 | 39.2 | 16.0 | 9.3 | **60.8** |
| HumanEval-Mul (Pass@1) | 77.3 | 77.2 | 82.6 | 81.7 | 80.5 | 81.5 |
| LiveCodeBench (Pass@1) | 31.1 | 28.4 | 40.5 | 36.3 | 33.4 | **47.3** |
| MathVista-Test (Pass@1) | – | – | – | 65.3 | 63.8 | **70.1** |
| MMMU-Val (Pass@1) | – | – | – | 66.4 | 69.1 | 68.0 |
| MathVision-Full (Pass@1) | – | – | – | 35.6 | 30.4 | 31.0 |

The "+550%" headline lives in the AIME row: short-CoT k1.5 scores **60.8** against GPT-4o's **9.3** — roughly a +554% relative improvement — and Claude 3.5 Sonnet's 16.0. On MATH-500 the short model hits 94.6, beating DeepSeek-V3's 90.2 and GPT-4o's 74.6. On LiveCodeBench it reaches 47.3, ahead of DeepSeek-V3 (40.5) and GPT-4o (33.4). The framing matters: these are comparisons against *non-reasoning* models, so the right reading is "a short model that inherited a reasoning prior beats models that never had one." That is exactly what makes long2short interesting — the long-CoT capability survives compression into a model that answers at short-CoT token budgets.

### long2short, in detail

The distillation step is its own contribution with four methods, and the paper finds the RL variant most token-efficient.

![long2short: four ways to pour a long-CoT prior into a short model](/imgs/blogs/kimi-k1-5-6.png)

The four methods:

- **Model merging.** Average the weights of a long-CoT model and a short-CoT model. No training at all — just a weight average. It is the cheapest option and a surprisingly strong baseline.
- **Shortest rejection sampling.** Sample the same question $n = 8$ times and keep the *shortest correct* response for supervised fine-tuning. You are mining the model's own brief-but-right answers as distillation targets.
- **DPO.** Build preference pairs where the shortest correct solution is the positive, and the negatives are wrong long responses *or* correct-but-too-long ones (at least 1.5× longer than the chosen positive). Train DPO on these pairs to prefer brevity-with-correctness.
- **long2short RL.** After standard RL, pick the model with the best performance/efficiency balance as the base, then run a *separate* RL phase that applies the length penalty with a **significantly reduced max rollout length** — penalizing over-length responses even when they are correct.

The token-efficiency results are the punchline. The long2short RL model reaches **60.8 Pass@1 on AIME 2024** (averaged over 8 runs) using only **~3,272 tokens** on average, and the k1.5-shortest variant reaches **88.2 on MATH-500** at roughly the same token count as other short models. Across the four methods, long2short RL gives the **highest token efficiency**, and every k1.5-series model is more token-efficient than the baselines. A small table to make the efficiency concrete:

| Model | Benchmark | Score | Avg tokens | Note |
|---|---|---|---|---|
| k1.5-short (w/ RL) | AIME 2024 (Pass@1) | 60.8 | ~3,272 | avg over 8 runs; highest token efficiency |
| k1.5-shortest | MATH-500 (Pass@1) | 88.2 | ~same as short baselines | shortest variant |

### What is load-bearing, and what might not transfer

The single most load-bearing empirical claim is the **length-accuracy coupling**: performance correlates strongly with response/output length (Figure 6 in the report), with positive trend slopes across OMNI-MATH500, MATH500, AIME, ChatGLMMath, GAOKAO, and GPQA (slopes like 2.46e-05, 3.05e-05, 3.40e-05, 4.24e-05). If that correlation is causal — if longer reasoning *causes* higher accuracy rather than merely co-occurring with harder-won correct answers — then "scale context" is the right lever and everything follows. The paper treats it as causal and the scaling results are consistent with that, but the correlation itself is what the whole method leans on.

What might not transfer cleanly: the reward design is heavily tuned for *verifiable* domains. Math and code have ground-truth answers and executable tests; the CoT reward model gets to 98.5% precisely because math correctness is well-defined. In domains where correctness is fuzzy — open-ended writing, multi-step research, anything where "the answer" is a judgment call — the outcome-reward-only approach has a much thinner reward signal, and the reward-hacking filters (exclude multiple choice, remove guessable prompts) do not obviously generalize. The recipe is a *reasoning* recipe, and reasoning here means "problems with checkable answers." Carry it to soft domains and the load-bearing reward component is the part that breaks first.

## Critique

**What is strong.** The simplicity is the contribution, and it is genuinely strong. By removing the value network, MCTS, and process reward model, k1.5 eliminates three of the most failure-prone components in reasoning RL and still reaches frontier numbers. The off-policy sampling from a frozen reference, the optimizer reset per iteration, the length-penalty warmup, and partial rollouts are each the *correct* engineering choice for the problem they solve, and together they form a recipe that reads like it would actually reproduce. The reward-hacking defenses — especially the "can a model guess this with no CoT?" filter and the code-verification funnel that throws out two-thirds of problems — show a team that has been burned by reward hacking and built real defenses. And long2short is a genuinely useful idea: it makes the expensive long-CoT training pay off in a cheap-to-serve model.

**What is weak or unfalsifiable.** The biggest gap is disclosure. The report gives **no parameter count, no layer count, no attention type, and no MoE/expert configuration** for the flagship model; the ablation models are described only as "larger" and "smaller" with sizes withheld. There is **no GPU count, no GPU-hours, no cluster size, no wall-clock time, and no FLOPs**, and **no total pretraining-token or RL-sample count**. This matters because "a simplistic RL recipe reaches o1-level reasoning" is only meaningful if you know the *scale* it took to get there — a simple recipe on a 1T-parameter model trained for a year is a different claim than a simple recipe on a 70B model in a week. Without those numbers, the efficiency claim is partly unfalsifiable: we cannot tell how much of the result is the algorithm versus the scale.

The causal story behind length-accuracy is asserted more than proven. The paper shows length and accuracy rise together and frames length as "compute budget for implicit search," but a strictly correlational figure cannot rule out the alternative that the model simply *writes more when problems are hard* and the residual correctness is doing the work. A controlled experiment — fix the problem set, force different length budgets, measure accuracy — would settle it; the report leans on the training-time trend instead.

**What ablation is missing.** The ablations that *are* present are good: negative gradients matter (their method beats ReST, which only fits the best sample and uses no negative gradients, with a gap more pronounced for long CoT than in other domains); curriculum beats uniform sampling; the CoT RM beats the classic RM; the length penalty improves token efficiency; and there is a model-size-vs-context-length study showing a smaller model can reach a larger model's performance by using longer RL-optimized CoTs, while the larger model is generally more token-efficient. What is missing is an ablation isolating the **KL/L2 trust-region term** — how much does the off-policy regularizer actually buy versus plain policy gradient with a mean baseline? And there is no ablation on the **value of the long-CoT SFT warmup**: how much does RL depend on that small warmup set, and would cold-start RL from vanilla SFT collapse or merely slow down? Those two would tell us which parts of the recipe are essential versus incidental.

**What would change my mind.** If a controlled length-budget study showed accuracy *flat* across forced length budgets at fixed difficulty — i.e., the model gets the same score whether it writes 1k or 30k tokens — then "scaling context is the lever" would be wrong, and the gains would have to be attributed to something else (better SFT, better data, scale). That single experiment is the crux of the paper's thesis, and it is the one I would run first.

## What I'd build with this

1. **A verifiable-domain RL harness with the k1.5 update as the default.** The mean-reward-baseline mirror-descent update plus outcome reward is simple enough to implement in a few hundred lines and robust enough to be a sensible default for any domain with checkable answers — competitive programming, theorem proving, SQL generation, unit-test satisfaction. The first thing I would port is the "guess with no CoT" prompt filter, because reward-hacking-by-guessing is the failure that quietly ruins these setups.
2. **Partial rollouts as a library, not a research artifact.** The replay-buffer-plus-token-budget pattern is generally useful for any long-generation RL, agentic or otherwise. A clean implementation — fixed budget, async workers, replay buffer keyed by trajectory id, repeat detection — would be reusable far beyond reasoning, e.g. for long-horizon agent rollouts where a single episode can run for thousands of tool-call tokens.
3. **A long2short DPO recipe for any in-house reasoning model.** Even teams that cannot afford the full RL pipeline can apply shortest-rejection-sampling and the DPO variant (shortest-correct positive vs. ≥1.5×-longer negative) to an existing long-CoT model to recover token efficiency. This is the cheapest high-leverage extension in the paper.
4. **A curriculum + prioritized sampler bolted onto an existing trainer.** The $1 - s_i$ prioritized weighting plus a difficulty gate is a tiny change to a data loader and a measurable efficiency win; it is the kind of thing you add once and keep forever.
5. **A CoT reward model trained to emit JSON judgments.** The 84.4 → 98.5 jump from classic to CoT RM is large enough that, for any verifiable-ish domain, training a reward model that reasons step-by-step before judging is worth the extra labels — and the JSON-judgment format makes it trivial to wire into a reward loop.

## When to reach for this recipe (and when not to)

Reach for the k1.5 recipe when your problem has **verifiable answers** and you want a reasoning model: math, code, formal proof, structured generation with executable checks. In that regime the recipe's strengths line up perfectly — outcome rewards are honest, the reward-hacking filters apply, the length penalty recovers efficiency, and the long-context scaling gives you a clean knob to trade test-time compute for accuracy. If you are choosing between bolting MCTS and a process reward model onto your stack versus this, start here: it is dramatically simpler, and the burden of proof is now on the heavy machinery to justify itself.

Reach for it *cautiously* when correctness is fuzzy. The whole edifice rests on a reward you can trust, and in open-ended domains — creative writing, strategy, research synthesis — the outcome reward becomes a learned judge with all the reward-hacking exposure the paper works hard to avoid in verifiable domains. You can still use the *systems* pieces (partial rollouts, curriculum sampling, long2short) but the reward design needs rethinking, and the paper does not give you that.

Do **not** reach for it expecting a turnkey reproduction. The undisclosed scale — no parameter count, no GPU-hours, no token budgets — means you are reproducing an *algorithm*, not a *result*. Budget for the possibility that o1-level numbers required a frontier-scale base model and a large compute budget that the report simply does not quantify. The recipe is real and worth adopting; the specific benchmark numbers are aspirational targets that depend on scale you will have to supply yourself. As the authors themselves flag, improving the efficiency and scalability of long-context RL, sharpening credit assignment without re-introducing overthinking, and building better verification models all remain open — which is a fair statement of where the next round of work lives.

## References

- **Kimi k1.5: Scaling Reinforcement Learning with LLMs** — arXiv abstract: [arxiv.org/abs/2501.12599](https://arxiv.org/abs/2501.12599)
- **Code / report** — GitHub: [github.com/MoonshotAI/Kimi-k1.5](https://github.com/MoonshotAI/Kimi-k1.5)
- Related reads on this blog:
  - [Kimi K2: Open Agentic Intelligence](/blog/paper-reading/large-language-model/kimi-k2)
  - [Kimi K2 Thinking: An Open-Source Reasoning Model Built on K2](/blog/paper-reading/large-language-model/kimi-k2-thinking)
  - [Kimina-Prover: Large Formal Reasoning Models with RL](/blog/paper-reading/reasoning/kimina-prover)
  - [Kimi-Researcher: End-to-End RL for Autonomous Research](/blog/paper-reading/ai-agent/kimi-researcher)

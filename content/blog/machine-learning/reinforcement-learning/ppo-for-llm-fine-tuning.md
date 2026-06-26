---
title: "PPO for LLM Fine-Tuning: From Reward Model to Aligned Policy"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A from-first-principles guide to RLHF with PPO — the MDP framing of token generation, the clipped objective, the KL anchor that stops reward hacking, GAE over token sequences, and a complete runnable TRL training loop."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "rlhf",
    "ppo",
    "llm-alignment",
    "machine-learning",
    "pytorch",
    "trl",
    "actor-critic",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/ppo-for-llm-fine-tuning-1.png"
---

A supervised fine-tuned (SFT) language model will happily write you a polite, fluent paragraph that is also subtly wrong, evasive, or unhelpful — because nothing in its training signal ever told it that *this* answer was better than *that* one. Cross-entropy on a fixed corpus teaches a model to imitate. It does not teach the model to *prefer*. The first time I watched an SFT model confidently hallucinate a citation and then, when corrected, apologize and hallucinate a *different* fake citation, I understood the gap. Imitation learning has no mechanism for "the thing you just did was worse than the alternative." You need a reward.

Reinforcement Learning from Human Feedback (RLHF) closes that gap. We train a separate **reward model** to predict which of two responses a human prefers, then we use reinforcement learning to push the language model toward higher-reward responses. The dominant algorithm for that RL step — the one behind InstructGPT, the first few generations of Claude, and Llama-2-Chat — is **Proximal Policy Optimization (PPO)**. This post is about why PPO, specifically, became the workhorse of LLM alignment, and how to actually run it without the policy collapsing into a reward-hacking puddle of repeated tokens.

The whole series rests on one spine: **an agent interacts with an environment, collects rewards, and updates a policy; every algorithm is just a different answer to which objective to optimize and how to estimate its gradient.** RLHF is the most commercially important instance of that spine in existence. The agent is a language model. The environment is the prompt distribution plus a reward model. The policy update is PPO. By the end of this post you will be able to derive the full RLHF objective, explain exactly why the KL penalty term is non-negotiable, distribute a single response-level reward across hundreds of tokens with Generalized Advantage Estimation, and run a complete PPO fine-tune in the TRL library with a sentiment reward model. Figure 1 lays out the loop we will spend the whole post unpacking.

![Diagram of the PPO-RLHF episode loop showing a prompt flowing through policy generation, reward model scoring, a KL penalty, total reward, and the PPO update producing a new policy](/imgs/blogs/ppo-for-llm-fine-tuning-1.png)

If you want the map of where this sits among other RL methods, this post builds directly on [the policy gradient theorem](/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem) and on [Proximal Policy Optimization itself](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo); it is the bridge from generic continuous-control PPO to the LLM-specific RLHF setup, and it assumes you already know [why language models need RLHF](/blog/machine-learning/reinforcement-learning/why-language-models-need-rlhf). We will cross-link the unified map and the playbook at the end.

## 1. The RLHF-PPO setup: an LLM is a policy in an MDP

Before any math, get the framing exactly right, because almost every RLHF bug traces back to a fuzzy mental model of what the "episode" even is. Reinforcement learning operates on a **Markov Decision Process (MDP)**: a set of states $S$, actions $A$, a transition function, a reward function, and a discount $\gamma$. RLHF maps onto this cleanly once you stop thinking of the model as producing "a response" and start thinking of it as taking a *sequence of token-level actions*.

Here is the dictionary. The **policy** $\pi_\theta$ is the language model itself — its parameters $\theta$ define a probability distribution over the next token given everything so far. An **action** $a_t$ is the choice of the next token from the vocabulary (an action space of ~50,000 to ~150,000 discrete choices, depending on the tokenizer). A **state** $s_t$ is the prompt concatenated with all tokens generated up to step $t$: $s_t = (x, y_{<t})$ where $x$ is the prompt and $y_{<t}$ are the tokens generated so far. The **transition** is deterministic and trivial — appending the chosen token to the context is the entire dynamics, which is unusual and important. In most RL problems the environment dynamics are stochastic and unknown: you take an action and the world responds in a way you cannot predict or control, which is the entire source of difficulty in classical control. Here, given the state and the chosen token, the next state is *exactly* the concatenation — there is no environmental randomness in the transition at all. The only stochasticity in the whole MDP comes from the policy's own sampling and from the reward model's scoring. This is why some of the heavy machinery of general RL (model-based planning, exploration bonuses for unknown dynamics) is largely irrelevant to RLHF, and why the algorithm can be as simple as "sample, score, clip-update." The hard part of RLHF is not the dynamics; it is the credit assignment over a long deterministic chain with a single reward at the end. The **episode** is one complete response: start from the prompt $x$, generate tokens until an end-of-sequence token or a length limit, and you have a trajectory $\tau = (s_0, a_0, s_1, a_1, \dots, s_{T-1}, a_{T-1})$ of length $T$ = the number of generated tokens.

The **reward** is the part that makes RLHF different from a board game. In most RL problems, you get a reward at many steps. In RLHF, the reward model scores only the *complete* response. So the environment hands back a single scalar $r_{RM}(x, y)$ at the final token, and zero everywhere else. This is a **sparse, terminal reward** — one number for hundreds of decisions. A large fraction of the engineering in PPO-RLHF is about credit assignment: figuring out which of those hundreds of token choices deserve the praise or the blame for the final score.

There is one more reward component, and it is the secret sauce: a **per-token KL penalty** against a frozen reference model. We will derive why it must be there in section 3, but mechanically, at every token, we add a small negative reward proportional to how far the current policy's token probability has drifted from the original SFT model's probability. So the effective reward at step $t$ is mostly the KL cost, with the big reward-model score landing only at the end.

> The single most clarifying realization: in RLHF the "environment" includes a neural network (the reward model) that you trained, and that network is *gameable*. Unlike CartPole's physics, your reward function has exploitable holes. The KL term is what stops the policy from driving a truck through them.

#### Worked example: counting the MDP pieces for one response

Take the prompt `x = "Explain why the sky is blue."` Suppose the policy generates a 60-token response `y`. Then:

- The episode has $T = 60$ steps.
- There are 60 states: $s_0 = x$ (the bare prompt), $s_1 = x + \text{token}_1$, ..., $s_{59} = x + \text{tokens}_{1..59}$.
- There are 60 actions: each is one token id from the vocabulary.
- The reward model produces **one** scalar, say $r_{RM} = 2.3$, evaluated on the full $(x, y)$.
- At each of the 60 steps we also compute a KL cost $-\beta \cdot \text{KL}_t$. If the policy hasn't drifted, these are near zero; if it has, they bite.
- The total reward signal the PPO algorithm sees is: a stream of small KL costs at tokens 1–59, plus $2.3 - \beta\cdot\text{KL}_{60}$ at the final token.

That asymmetry — one big number at the end, tiny penalties along the way — is the crux of why we need a value function and GAE. Hold that thought.

## 2. Why PPO, specifically, for language models

The obvious approach is to fine-tune an LLM with vanilla policy gradient (REINFORCE): sample a response, score it, and push up the log-probability of every token in proportion to the reward. People tried exactly that, and it is a disaster for LLMs for two related reasons, both of which PPO fixes.

**Reason one: the variance is brutal and the updates are destructive.** A language model has billions of parameters and a softmax over a huge vocabulary. A single large gradient step can yank the policy into a region where it produces garbage, and unlike a tiny CartPole network, you cannot afford to throw away a half-trained 70B-parameter model and restart. REINFORCE has no built-in brake on step size. PPO's **clipped surrogate objective** is precisely a brake: it refuses to move the probability of any action more than a fixed ratio away from where it was when the data was collected. That single mechanism turns RLHF from "usually diverges" into "usually converges."

**Reason two: sample efficiency and reuse.** Generating responses from a large model is expensive — it is the dominant cost of an RLHF run. REINFORCE is strictly on-policy: you collect a batch, take one gradient step, and throw the data away. PPO lets you take *several* gradient epochs over the same batch of generated responses while staying safe, because the clip keeps you from over-optimizing on stale data. For LLMs, where generation dominates the compute budget, squeezing 4 epochs out of one batch of generations instead of 1 is a 4× win on the expensive part.

It is worth dwelling on *why* generation is the expensive part, because it reshapes how you think about the whole loop. A PPO step has three phases: rollout (generate responses — autoregressive, one forward pass per token, no parallelism across the sequence), scoring (the reward model and reference forward passes — fully parallel over the sequence), and the update (forward + backward on the policy — also parallel). For a 32-token response, the rollout is 32 sequential forward passes through a multi-billion-parameter model; the update is a couple of parallel passes. On a real run, rollout commonly eats 60–80% of wall-clock time. This is why every serious RLHF stack invests in fast generation (KV-cache reuse, paged attention, sometimes a dedicated inference engine like vLLM for the rollout phase) — and why the data-reuse property of PPO is not a minor convenience but a first-order efficiency lever. You paid dearly for those generations; PPO lets you learn from them four times instead of once.

PPO is an **actor-critic** method. The "actor" is the policy (the LM). The "critic" is a value function $V(s)$ that estimates the expected total future reward from a state. The critic is what lets us turn that single terminal reward into a useful, low-variance learning signal at every token, via the advantage estimate. We will build the critic in section 4.

### The policy gradient theorem, applied to tokens

To see exactly *why* the critic and the clip matter, start from where all of this begins: the policy gradient theorem. We want to maximize the expected return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$. The theorem says its gradient is

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\Big[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, A_t \Big]$$

This is the load-bearing equation of the whole field. Read it in words: to improve the policy, for every token you took, nudge its log-probability up or down in proportion to how *advantageous* that token was. The $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ term is the **score function** — it points in parameter space toward "make this token more likely." Multiplying by the advantage $A_t$ scales and signs that nudge: positive advantage pushes the token up, negative pushes it down.

Two facts fall out of this derivation that explain the rest of the post. First, **the gradient is a Monte Carlo estimate**, averaged over sampled trajectories — which is *why* it is high variance, and why we need a baseline (the value function) to subtract off the part of the return that has nothing to do with the action. Replacing the raw return with the advantage $A_t = Q(s_t,a_t) - V(s_t)$ is a variance reduction that leaves the gradient *unbiased* (subtracting any function of state only that doesn't depend on the action has zero expected gradient — that is the baseline theorem). For an LLM with a sparse terminal reward, the raw return of every token in a response is the *same* number (the final reward), so without a baseline you would push *every* token in a good response up by the same amount, including the filler tokens. The advantage is what discriminates "this token actually contributed" from "this token was just along for the ride."

Second, the policy gradient is **on-policy**: that expectation is over trajectories from $\pi_\theta$ *itself*. The moment you take a gradient step, $\theta$ changes, and your collected data is now slightly off-policy — sampled from the old policy, not the new one. Vanilla policy gradient handles this by throwing the data away after one step. PPO instead *corrects* for the mismatch with an importance-sampling ratio $r_t(\theta) = \pi_\theta(a_t\mid s_t) / \pi_{\theta_{old}}(a_t\mid s_t)$, which is exactly the ratio inside the clipped objective in section 6. The clip is what keeps that importance correction trustworthy: importance sampling blows up when the two distributions diverge, so clamping the ratio to $[1-\epsilon, 1+\epsilon]$ both bounds the step *and* keeps the importance weights from exploding. That is the deep reason the same mechanism gives you stability and data reuse at once — they are the same property viewed from two angles.

Here is the comparison that explains the design choice in one table.

| Method | Step-size control | Data reuse | Variance | Verdict for LLMs |
| --- | --- | --- | --- | --- |
| REINFORCE | none (raw gradient) | on-policy, 1 epoch | very high | diverges or trains slowly |
| Vanilla actor-critic (A2C) | none explicit | on-policy, 1 epoch | medium | unstable on huge nets |
| TRPO | hard KL trust region | on-policy | low | correct but heavy 2nd-order math |
| **PPO** | **soft clip on ratio** | **on-policy, several epochs** | **low** | **the practical sweet spot** |
| DPO (no RL) | implicit via loss | offline preferences | n/a | great when it fits; no online sampling |

PPO is the descendant of TRPO (Trust Region Policy Optimization) that replaced TRPO's expensive constrained optimization with a cheap clip. It is, almost exactly, "TRPO's safety guarantee, implemented with a `min` and a `clamp`." That is why it won. The clip is the entire reason PPO is stable enough to bet a multi-million-dollar training run on.

## 3. The full objective and the β tradeoff

Now the math, built up piece by piece. We want to maximize the reward model's score on responses our policy generates. Naively:

$$\max_\theta \; \mathbb{E}_{x \sim D,\; y \sim \pi_\theta(\cdot \mid x)} \big[\, r_{RM}(x, y) \,\big]$$

If you optimize *only* this, you get **reward hacking**: the policy discovers that the reward model — being an imperfect neural net trained on finite human comparisons — assigns spuriously high scores to certain patterns. Maybe it loves lists. Maybe it loves the phrase "Certainly! Here is". Maybe it loves long responses. The policy, being a relentless optimizer, will produce degenerate text that maxes out the reward model while being useless or unreadable to a human. I have seen a policy converge to repeating "helpful helpful helpful" because an under-trained reward model had a weak positive association with that token. The reward went up; the model became worthless.

The fix is to keep the policy *close to the original SFT model*, which we know produces fluent, on-distribution text. We add a penalty for drifting away, measured by **KL divergence** — the information-theoretic distance between the new policy's token distribution and the frozen reference's. The full RLHF objective becomes:

$$\max_\theta \; \mathbb{E}_{x \sim D,\; y \sim \pi_\theta} \Big[\, r_{RM}(x, y) \;-\; \beta \cdot \mathrm{KL}\big(\pi_\theta(\cdot\mid x) \,\|\, \pi_{SFT}(\cdot\mid x)\big) \,\Big]$$

In practice we implement the KL as a **per-token** penalty inside the reward stream. At token $t$, the KL contribution is approximated by the log-probability-ratio:

$$\mathrm{KL}_t \approx \log \pi_\theta(a_t \mid s_t) - \log \pi_{SFT}(a_t \mid s_t)$$

so the effective per-token reward fed to PPO is:

$$\tilde{r}_t = \underbrace{-\beta \big(\log \pi_\theta(a_t \mid s_t) - \log \pi_{SFT}(a_t \mid s_t)\big)}_{\text{per-token KL cost}} \;+\; \underbrace{r_{RM}(x, y) \cdot \mathbb{1}[t = T]}_{\text{terminal reward}}$$

That indicator $\mathbb{1}[t = T]$ is doing a lot of work: the reward-model score is added *only* at the last token. Everywhere else, the only reward is the KL cost. Figure 2 shows how these terms branch off the policy and the reward model and merge into the total loss.

![Diagram showing the PPO objective components where the policy branches into a clipped surrogate and a KL penalty, the value head produces a value loss, and all three merge into the total loss](/imgs/blogs/ppo-for-llm-fine-tuning-2.png)

### Why the KL term provably prevents reward hacking

Here is the rigorous version of the intuition. The objective with the KL penalty has a known closed-form optimal policy. For a fixed reward $r$, the policy that maximizes $\mathbb{E}[r] - \beta\,\mathrm{KL}(\pi \| \pi_{SFT})$ is:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \, \pi_{SFT}(y \mid x) \, \exp\!\Big(\tfrac{1}{\beta} r(x, y)\Big)$$

where $Z(x)$ is a normalizing constant. (This is the same identity DPO exploits to skip the RL step entirely.) Read that formula carefully: the optimal policy is the *reference model reweighted by an exponential of the reward*. It can never assign probability to a response that the reference model assigns *zero* probability to — because anything times zero is zero. The reference model's support is a fence. The policy can re-rank within the fence, sharpening toward high-reward responses, but it cannot escape into the wild, off-distribution garbage where the reward model's score is meaningless. The smaller $\beta$ is, the more aggressively it sharpens; the larger $\beta$ is, the closer it hugs the reference. That is the whole tradeoff, and it is a *provable* property of the objective, not a heuristic.

### The β coefficient tradeoff in practice

The coefficient $\beta$ is the most important hyperparameter in the entire run.

- **β too small** (say, 0.0–0.02): the KL fence is weak. The policy wanders off-distribution, finds the reward model's blind spots, and reward-hacks. You see the reward-model score shoot up while real quality craters and the KL divergence explodes past 30–40 nats. This is the failure mode on the left of Figure 3.
- **β too large** (say, > 1.0): the KL fence is a straitjacket. The policy can barely move from the SFT model, so you get almost no improvement — you paid for an expensive RLHF run and your win-rate barely budged.
- **β just right** (typically **0.1–0.5**, often starting around 0.2): the policy improves measurably on the reward while staying fluent and on-distribution, and the KL settles into a controlled range (single-digit to low-double-digit nats over the course of training).

![Before-and-after comparison contrasting a too-small KL coefficient causing reward hacking and KL explosion against an optimal coefficient yielding aligned responses and controlled KL](/imgs/blogs/ppo-for-llm-fine-tuning-3.png)

Many implementations go further and make $\beta$ **adaptive**: they set a target KL (say, 6 nats) and use a controller that nudges $\beta$ up when the measured KL overshoots the target and down when it undershoots. This is the "adaptive KL controller" from the original OpenAI RLHF papers, and TRL implements it as `adap_kl_ctrl`. It saves you from babysitting $\beta$ by hand across runs with different reward scales. The controller update is a simple proportional rule: $\beta \leftarrow \beta \cdot (1 + K_\beta \cdot e)$ where $e = \mathrm{clip}(\text{KL}/\text{KL}_{target} - 1, -0.2, 0.2)$ is the relative error. When the policy is drifting too far, $e > 0$ and $\beta$ rises to pull it back; when it has retreated too close to the reference, $e < 0$ and $\beta$ relaxes to allow more learning. The effect is a closed loop that holds the KL near your target regardless of the reward model's scale or the policy's size.

### Why the closed-form optimum is exactly the DPO identity

It is worth seeing the derivation of that optimal-policy formula because it is the single most reused identity in modern alignment. We are maximizing, per prompt $x$, the functional $\mathbb{E}_{y \sim \pi}[r(x,y)] - \beta\,\mathrm{KL}(\pi \| \pi_{SFT})$ over all distributions $\pi(\cdot\mid x)$. Write the KL out and add a Lagrange multiplier for the constraint that $\pi$ sums to one. Taking the functional derivative with respect to $\pi(y\mid x)$ and setting it to zero gives $r(x,y) - \beta(\log\pi(y\mid x) - \log\pi_{SFT}(y\mid x)) - \beta - \text{const} = 0$, which rearranges to $\log\pi(y\mid x) = \log\pi_{SFT}(y\mid x) + \tfrac{1}{\beta}r(x,y) - \log Z(x)$, i.e.

$$\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{SFT}(y\mid x)\,\exp\!\Big(\tfrac{1}{\beta}r(x,y)\Big).$$

PPO reaches this optimum *iteratively* by sampling and gradient-stepping. DPO observes that you can *solve for the reward* in terms of the policy ($r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{SFT}(y\mid x)} + \beta\log Z(x)$), substitute that into the Bradley-Terry preference loss (where the intractable $Z(x)$ cancels because it appears in both the winner and loser terms), and thereby optimize the policy *directly* on preference pairs without ever sampling or training a reward model. Same objective, same optimum, different path. Understanding this equivalence is what lets you reason about *when* the cheaper DPO suffices and when the online sampling of PPO is genuinely buying you something (chiefly: optimizing rewards you cannot express as offline preference pairs, and continuing to improve past the fixed dataset's coverage).

#### Worked example: the β tuning effect on one batch

Suppose on a batch of 256 prompts your reward model returns a mean raw score of 1.8, and the measured mean KL from the reference is 12 nats. The effective objective per response is roughly $1.8 - \beta \cdot 12$.

- With $\beta = 0.05$: penalized objective $= 1.8 - 0.6 = 1.2$. The KL barely costs anything, so the optimizer is happy to push KL even higher next batch — a slippery slope toward hacking.
- With $\beta = 0.2$: penalized objective $= 1.8 - 2.4 = -0.6$. Now drifting to 12 nats is *net negative*; the optimizer is pressured to find reward gains that don't cost so much KL. It will prefer responses that are better *and* close to the reference.
- With $\beta = 0.5$: penalized objective $= 1.8 - 6.0 = -4.2$. Severe. The policy will retreat toward the reference, KL will drop to maybe 3–4 nats, and reward gains will be tiny.

The art is picking $\beta$ so that, at equilibrium, the policy sits at a KL where the marginal reward gain equals the marginal KL cost — and that equilibrium produces genuinely better text. In the InstructGPT line of work, KL coefficients in the 0.1–0.2 range with target KLs around 6 nats were typical.

### Where the reward model comes from

We have been treating $r_{RM}$ as given, but it is worth a paragraph on where that scalar originates, because PPO's behavior is downstream of the reward model's quality. The reward model is trained *before* PPO, on a dataset of human **preference pairs**: for a prompt $x$, a human is shown two responses $y_w$ (preferred / "winner") and $y_l$ (dispreferred / "loser") and labels which is better. The reward model — typically the SFT model with its LM head replaced by a single scalar regression head — is trained to assign a higher score to the winner with the Bradley-Terry pairwise loss:

$$L_{RM}(\psi) = -\mathbb{E}_{(x, y_w, y_l)} \Big[ \log \sigma\big( r_\psi(x, y_w) - r_\psi(x, y_l) \big) \Big]$$

where $\sigma$ is the logistic sigmoid. This loss is maximized when the reward model assigns the winner a much higher score than the loser; it only ever learns *relative* scores, which is why the absolute scale of $r_{RM}$ is arbitrary (and why we often whiten rewards before PPO — see section 7). In TRL this is the `RewardTrainer`. The key thing to internalize for PPO: the reward model is a *learned, imperfect, and finite-data* approximation of human preference. It is accurate near the distribution of responses it was trained on and increasingly unreliable as the policy drifts away from that distribution — which is the deep reason both the KL penalty (keep the policy near the reference, where the RM is reliable) and Llama-2's iterative retraining (move the RM with the policy) exist.

```python
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Reward model = a sequence classifier with ONE output (the scalar score).
rm = AutoModelForSequenceClassification.from_pretrained(
    "gpt2", num_labels=1
)
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token

# Preference dataset must have columns: chosen, rejected (each tokenized).
# RewardTrainer applies the Bradley-Terry pairwise loss internally.
reward_config = RewardConfig(
    output_dir="./reward_model",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    learning_rate=1e-5,
)
reward_trainer = RewardTrainer(
    model=rm, args=reward_config,
    train_dataset=preference_dataset,  # has 'chosen' and 'rejected'
    tokenizer=tok,
)
reward_trainer.train()
# The trained rm's scalar output is what PPO's compute_rewards() will call.
```

A reward model that can't separate held-out good from bad responses by a clear margin will sabotage PPO no matter how well you tune $\beta$ — garbage reward in, garbage policy out. Before launching an expensive PPO run, always validate the reward model's accuracy on a held-out preference set (a well-trained RM typically hits 65–75% pairwise accuracy on hard human-preference data; much above that on easy data).

## 4. The value function for LLMs

PPO is actor-critic, so we need a critic: a value function $V(s)$ that estimates the expected total (KL-penalized) reward from state $s$ onward. Why do we need it at all? Because of the sparse terminal reward. If we only knew "the final response scored 2.3," we would have no idea whether token 17 was a good choice or token 43 was the brilliant one. The value function is our running estimate of "how good is the situation right now," and the *difference* between successive value estimates (the TD error) tells us whether each specific token made things better or worse than expected. That difference, accumulated, becomes the **advantage** — the low-variance learning signal we actually backprop through.

For LLMs, the value function is implemented as a **value head**: a single linear layer that maps the transformer's final hidden state at each token position to a scalar. So a PPO-RLHF model is the base transformer with *two* output heads — the usual language-modeling head (logits over the vocabulary, this is the actor) and a value head (one scalar per position, this is the critic). TRL's `AutoModelForCausalLMWithValueHead` wires this up for you.

A crucial implementation detail: the value head is typically **initialized from the reward model's head**, not randomly. The reward model already learned to map hidden states to a meaningful scalar score; that is almost exactly what the value head needs to do. Starting the value head from random weights means the critic spends the first thousands of steps producing noise, which injects enormous variance into the advantage estimates right when the policy is most fragile. Initializing from the reward model gives the critic a warm start.

Why does **sharing the transformer body** between the policy and the value head work well here, when in classic RL people often use separate networks? Two reasons. First, the representations a language model needs to predict the next token are nearly a superset of what it needs to predict response quality — both require deep understanding of the partial response. Second, memory: holding a separate full-size value network would mean a fifth large model in GPU memory, which we cannot afford (we are already juggling four). The pragmatic compromise some implementations use is a small detached value head with a *stop-gradient* so the value loss doesn't corrupt the policy's representations, but the most common setup shares the body and balances the two losses with a value-loss coefficient. Figure 4 stacks the four resident models so you can see where the value head sits.

![Stack diagram of the four RLHF system models: frozen reference policy, frozen reward model, value head critic, trainable policy actor, and the PPO update on top](/imgs/blogs/ppo-for-llm-fine-tuning-4.png)

The value function is trained with a simple regression loss — mean squared error between its prediction $V_\phi(s_t)$ and the empirical return (or, in PPO, a clipped version of it):

$$L_V(\phi) = \mathbb{E}_t \Big[ \big(V_\phi(s_t) - \hat{R}_t\big)^2 \Big]$$

where $\hat{R}_t$ is the return target computed during advantage estimation. PPO also clips the value update (to a `cliprange_value`) for the same stability reason it clips the policy — to stop a single batch from yanking the critic too far.

## 5. Token-level vs response-level rewards and GAE

This is the heart of the credit-assignment problem and the most LLM-specific part of the algorithm. The reward model gives us one scalar per response. PPO needs an **advantage** $A_t$ at every token to know how to nudge each token's probability. How do we get from one number to hundreds of per-token signals?

The bridge is **Generalized Advantage Estimation (GAE)**. First, define the per-token **TD error** (temporal-difference residual):

$$\delta_t = \tilde{r}_t + \gamma V(s_{t+1}) - V(s_t)$$

In words: the TD error at token $t$ is "the reward I just got, plus my estimate of the value of where I landed, minus my estimate of the value of where I was." If the situation turned out better than the critic expected, $\delta_t > 0$ and we should make that token *more* likely. The reward $\tilde{r}_t$ here is the per-token KL cost everywhere, plus the big reward-model score at the final token.

GAE then sets the advantage as an exponentially-weighted sum of future TD errors:

$$A_t^{GAE} = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \, \delta_{t+l}$$

The parameter $\lambda \in [0, 1]$ tunes the **bias-variance tradeoff**. At $\lambda = 0$, the advantage is just the single-step TD error $\delta_t$ — low variance, but biased because it trusts the critic completely. At $\lambda = 1$, it becomes the full Monte Carlo return minus the baseline — unbiased, but high variance because it sums all the noise from here to the end of the episode. For LLMs, $\lambda \approx 0.95$ and $\gamma \approx 1.0$ are standard ($\gamma = 1$ because there is no real reason to discount future tokens within a single short response). Figure 7 shows how each token's advantage is assembled from its own TD error plus the discounted tail of later ones.

![Diagram of GAE over a token sequence where token rewards and value estimates combine into a TD error and a discounted tail, merging into the per-token advantage and return target](/imgs/blogs/ppo-for-llm-fine-tuning-7.png)

The elegance of GAE for the sparse-reward LLM case: because almost all per-token rewards are just the small KL cost and the big reward only lands at the end, the terminal reward "flows backward" through the value-function bootstrapping. A well-trained critic will already have *anticipated* a good response — its value estimates will rise as the response heads somewhere good — so the credit gets distributed smoothly across the tokens that mattered, rather than dumped entirely on the last token. The critic is what makes the sparse reward usable.

Contrast the two extremes to feel why $\lambda$ exists. Set $\lambda = 1$ and the advantage of token 1 is the full discounted sum of *every* future reward minus its baseline — which, for a 200-token response, means token 1's learning signal carries the accumulated noise of 200 stochastic decisions and one noisy terminal reward. That is unbiased (it uses real returns) but its variance is enormous, and on a billion-parameter network high-variance gradients are how runs die. Set $\lambda = 0$ and token 1's advantage is just its one-step TD error $\delta_1$, which is low variance but leans entirely on the critic's $V(s_2)$ estimate being correct — if the critic is wrong, every advantage is biased in the same direction. The sweet spot $\lambda = 0.95$ keeps roughly the nearest $\sim 1/(1-\lambda) = 20$ tokens' worth of real signal while letting the critic absorb the long tail, which for typical RLHF response lengths is the right tradeoff. This is the same bias-variance dial that governs $\mathrm{TD}(\lambda)$ in classical RL; it is not an LLM-specific invention, just an LLM-specific tuning.

#### Worked example: GAE for a 4-token response

Let's make it fully concrete with a tiny 4-token response, $\gamma = 1.0$, $\lambda = 0.95$. Suppose the per-token rewards (mostly KL costs, with the reward-model score added at the final token) and value estimates are:

| Token $t$ | Reward $\tilde{r}_t$ | $V(s_t)$ | $V(s_{t+1})$ |
| --- | --- | --- | --- |
| 1 | −0.02 | 1.10 | 1.30 |
| 2 | −0.01 | 1.30 | 1.50 |
| 3 | −0.03 | 1.50 | 1.40 |
| 4 (terminal) | +2.00 | 1.40 | 0 (episode ends) |

Compute the TD errors $\delta_t = \tilde{r}_t + \gamma V(s_{t+1}) - V(s_t)$:

- $\delta_1 = -0.02 + 1.30 - 1.10 = +0.18$
- $\delta_2 = -0.01 + 1.50 - 1.30 = +0.19$
- $\delta_3 = -0.03 + 1.40 - 1.50 = -0.13$
- $\delta_4 = +2.00 + 0 - 1.40 = +0.60$

Now GAE with $\gamma\lambda = 0.95$, computed backward:

- $A_4 = \delta_4 = 0.60$
- $A_3 = \delta_3 + 0.95 \cdot A_4 = -0.13 + 0.95(0.60) = -0.13 + 0.57 = +0.44$
- $A_2 = \delta_2 + 0.95 \cdot A_3 = 0.19 + 0.95(0.44) = 0.19 + 0.418 = +0.608$
- $A_1 = \delta_1 + 0.95 \cdot A_2 = 0.18 + 0.95(0.608) = 0.18 + 0.578 = +0.758$

Notice what happened: token 3 had a *negative* TD error on its own ($\delta_3 = -0.13$, the critic thought the situation slightly worsened), but its GAE advantage is *positive* ($+0.44$) because the big terminal reward flowed backward through the discount. Meanwhile token 1's advantage ($+0.758$) is the largest, reflecting that it set up the entire good trajectory. The return targets for the value-function regression are $\hat{R}_t = A_t + V(s_t)$: $\hat{R}_1 = 0.758 + 1.10 = 1.858$, and so on. This backward recursion is exactly what the GAE code below computes in a single loop.

```python
import torch

def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    # rewards: [T] per-token rewards (KL cost + terminal RM score)
    # values:  [T+1] value estimates, with values[T] = 0 at episode end
    T = rewards.shape[0]
    advantages = torch.zeros(T)
    last_gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:T]   # regression targets for the value head
    return advantages, returns

rewards = torch.tensor([-0.02, -0.01, -0.03, 2.00])
values  = torch.tensor([1.10, 1.30, 1.50, 1.40, 0.0])   # length T+1
adv, ret = compute_gae(rewards, values)
print("advantages:", adv)   # ~[0.758, 0.608, 0.44, 0.60]
print("returns:   ", ret)
```

That `compute_gae` loop is, modulo batching and masking, exactly what runs inside TRL's `PPOTrainer` every step.

## 6. The clipped surrogate objective in code

We have advantages; now the actual PPO update. Define the probability **ratio** between the current policy and the policy that *collected* the data (the "old" policy, frozen at the start of the PPO epochs over this batch):

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$$

The clipped surrogate objective is:

$$L^{CLIP}(\theta) = \mathbb{E}_t \Big[ \min\big( r_t(\theta)\, A_t,\; \mathrm{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\, A_t \big) \Big]$$

with $\epsilon \approx 0.2$. Unpack the $\min$ of two terms. The first, $r_t A_t$, is the standard importance-weighted policy-gradient objective. The second clamps the ratio to $[0.8, 1.2]$. Taking the minimum means: when the advantage is positive (good token, want to increase its probability), the clip caps how much you can increase it — once the ratio hits 1.2, further increases give no additional objective, so the gradient is zero and the optimizer stops pushing. When the advantage is negative (bad token, want to decrease its probability), the clip caps how far down you can push before the gradient vanishes. Either way, **no single update can move any token's probability more than 20% from where it started in this batch.** That is the brake. It is what keeps a billion-parameter model from flying apart.

Here is the core update in PyTorch, written to be readable rather than maximally efficient. This is the inner loop; TRL wraps it with all the batching, masking, and distributed plumbing.

```python
import torch
import torch.nn.functional as F

def ppo_loss(logprobs, old_logprobs, advantages, values, returns,
             clip=0.2, clip_value=0.2, vf_coef=0.1):
    # logprobs, old_logprobs: [B, T] log pi(a_t|s_t) for chosen tokens
    # advantages, returns:    [B, T] from GAE
    # values:                 [B, T] current value-head outputs

    # --- policy (actor) loss: clipped surrogate ---
    ratio = torch.exp(logprobs - old_logprobs)            # r_t(theta)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()    # maximize -> negate

    # --- value (critic) loss: clipped MSE ---
    v_clipped = values + torch.clamp(values - returns, -clip_value, clip_value)
    v_loss_unclipped = (values - returns) ** 2
    v_loss_clipped = (v_clipped - returns) ** 2
    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

    total = policy_loss + vf_coef * value_loss
    # diagnostics worth logging every step:
    approx_kl = (old_logprobs - logprobs).mean()
    clip_frac = (torch.abs(ratio - 1.0) > clip).float().mean()
    return total, {"policy_loss": policy_loss.item(),
                   "value_loss": value_loss.item(),
                   "approx_kl": approx_kl.item(),
                   "clip_frac": clip_frac.item()}
```

Log `approx_kl` and `clip_frac` every step. A healthy run has `clip_frac` in the 0.1–0.3 range (some clipping is happening, which means the brake is doing its job) and `approx_kl` per-update small and stable. A `clip_frac` near zero means your learning rate is too low (nothing is moving); a `clip_frac` near 1.0 means every token is slamming the clip, which means your steps are wildly too big.

#### Worked example: the clip mechanism on one token

Make the clip concrete on a single token, $\epsilon = 0.2$. Say the policy currently assigns a token probability that gives a ratio $r_t(\theta) = 1.35$ relative to the old policy (i.e. the new policy already made this token 35% more likely than when the batch was collected).

- **If the advantage is positive**, $A_t = +0.8$: the unclipped term is $1.35 \times 0.8 = 1.08$; the clipped term is $\mathrm{clip}(1.35, 0.8, 1.2) \times 0.8 = 1.2 \times 0.8 = 0.96$. The $\min$ picks $0.96$. Because the *clipped* (smaller) term is selected, the gradient with respect to $\theta$ through the $1.35$ ratio is **zero** — the objective has flatlined, so the optimizer stops increasing this already-over-promoted token. The brake engaged exactly as designed.
- **If the advantage is negative**, $A_t = -0.8$: the unclipped term is $1.35 \times (-0.8) = -1.08$; the clipped term is $1.2 \times (-0.8) = -0.96$. The $\min$ picks $-1.08$ (the *more negative* one). Here the clip does **not** bind — and that asymmetry is intentional. PPO deliberately lets you keep pushing *down* the probability of a token that turned out bad even past the clip, because moving a bad token's probability further toward zero is safe; it is only the *upward* over-promotion that the clip restrains. This is the subtle part of the $\min$ most people miss: it is one-sided by design, conservative on the way up and permissive on the way down.

That single worked example is the entire intuition behind why PPO is stable: it caps optimism (never trust a too-large positive ratio) while staying responsive to mistakes.

## 7. The TRL library: PPOTrainer end to end

The TRL (Transformer Reinforcement Learning) library from Hugging Face implements all of the above so you don't reimplement the GAE recursion and the four-model juggling yourself. The central object is `PPOTrainer`. It expects: a policy model with a value head, a frozen reference model, a tokenizer, and a `PPOConfig`. You supply the reward externally — TRL deliberately does not own the reward model, so you can use a sequence-classification reward model, a rule-based reward, or even a human in the loop.

Let's install and set up. The 4-GPU minimum for real RLHF at scale comes from exactly the memory accounting we'll do in section 8 — at a 7B+ scale you need to shard the policy, hold a reference and a reward model, and still fit activations. For a learnable demo, GPT-2 fits on a single GPU.

```bash
pip install "trl>=0.8" transformers accelerate datasets torch
# For multi-GPU RLHF at 7B+, launch under accelerate with DeepSpeed ZeRO-3:
#   accelerate config        # choose multi-GPU + DeepSpeed stage 3
#   accelerate launch --num_processes 4 train_ppo.py
```

The minimal but complete setup — policy with value head, reference model, config:

```python
import torch
from transformers import AutoTokenizer
from trl import (PPOTrainer, PPOConfig,
                 AutoModelForCausalLMWithValueHead, create_reference_model)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Policy = LM + value head (value head can be init'd from a reward head)
policy = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
# Frozen reference for the KL term — TRL builds it by deep-copying & freezing
ref_model = create_reference_model(policy)

config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,     # small LR; LLMs are fragile
    batch_size=256,            # prompts per PPO step
    mini_batch_size=16,        # forward/backward chunk
    ppo_epochs=4,              # reuse each batch 4x (the data-reuse win)
    init_kl_coef=0.2,          # starting beta
    target=6.0,                # target KL for the adaptive controller
    adap_kl_ctrl=True,         # auto-adjust beta toward target KL
    cliprange=0.2,             # epsilon for the policy clip
    cliprange_value=0.2,       # clip for the value loss
    gamma=1.0,                 # no discount within a short response
    lam=0.95,                  # GAE lambda
)
```

Now the reward model integration. Here we use a real sentiment classifier as a stand-in reward model — the policy will learn to generate positive-sentiment continuations. This is the classic TRL "positive sentiment" demo and it actually trains in minutes on a single GPU, which makes it the best possible thing to run first.

```python
from transformers import pipeline

# Reward model: a frozen sentiment classifier. Higher "POSITIVE" logit = higher reward.
reward_pipe = pipeline(
    "sentiment-analysis",
    model="lvwerra/distilbert-imdb",
    device=0,
)

def compute_rewards(texts):
    # Return one scalar reward per full response (response-level reward).
    outputs = reward_pipe(texts, top_k=None)  # all class scores
    rewards = []
    for scores in outputs:
        pos = next(s["score"] for s in scores if s["label"] == "POSITIVE")
        rewards.append(torch.tensor(pos))      # in [0, 1]; scale if needed
    return rewards
```

And the training loop — this is the part that makes the whole MDP framing concrete. Each iteration: sample prompts, generate responses (this is the expensive on-policy step), score them with the reward model, and hand `(query_tensors, response_tensors, rewards)` to `ppo_trainer.step`, which internally computes per-token KL, runs GAE, and applies the clipped update.

```python
from datasets import load_dataset

# Prompts: first few tokens of IMDB reviews; policy continues them.
dataset = load_dataset("imdb", split="train").shuffle(seed=0).select(range(2000))
def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["text"])[:8]   # short prompt
    return sample
dataset = dataset.map(tokenize, batched=False)
dataset.set_format("torch")

ppo_trainer = PPOTrainer(config, policy, ref_model, tokenizer, dataset=dataset)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,            # MUST sample, not greedy — RL needs exploration
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

for epoch, batch in enumerate(ppo_trainer.dataloader):
    query_tensors = batch["input_ids"]

    # 1) ACT: generate responses from the current policy (on-policy rollout)
    response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, **generation_kwargs
    )
    batch["response"] = [tokenizer.decode(r) for r in response_tensors]

    # 2) SCORE: reward model on the full prompt+response text
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    rewards = compute_rewards(texts)

    # 3) LEARN: PPO step — KL penalty, GAE, clipped update all happen inside
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if epoch % 10 == 0:
        print(f"step {epoch}  mean_reward={torch.stack(rewards).mean():.3f}  "
              f"kl={stats['objective/kl']:.2f}  "
              f"policy_loss={stats['ppo/loss/policy']:.4f}")
```

That `stats` dict is your entire dashboard. The fields you watch most: `objective/kl` (the running KL from the reference — should rise then plateau, not explode), `ppo/mean_scores` (the reward, should rise), `ppo/policy/clipfrac` (should sit in 0.1–0.3), and `ppo/val/error` (value-function error, should fall). On the sentiment task, you should see mean reward climb from roughly 0.5 (random) toward 0.8+ within a few hundred steps while KL stays in the single digits — the policy genuinely learns to write positive continuations.

For a *real* preference-trained reward model rather than a sentiment proxy, you swap `reward_pipe` for an `AutoModelForSequenceClassification` you trained with TRL's `RewardTrainer` on a preference dataset (pairs of chosen/rejected responses). The PPO loop is identical; only the source of the scalar changes.

### Practical implementation tricks that decide whether it works

The difference between a PPO-RLHF run that converges and one that thrashes is mostly a handful of normalization and stability tricks. None of them is glamorous; all of them matter.

- **Whiten the advantages (and often the rewards).** Because the reward model's scale is arbitrary, raw rewards can be tiny or huge, which destabilizes the gradient. Standard practice is to normalize advantages within each batch to zero mean and unit variance before the policy update: $A_t \leftarrow (A_t - \mu_A) / (\sigma_A + 10^{-8})$. TRL does this by default (`whiten_rewards` / advantage whitening). It makes the effective learning rate insensitive to the reward model's arbitrary scale, which is exactly what you want.
- **Response-length normalization / bias.** Reward models frequently have a **length bias** — they spuriously prefer longer responses, so the policy learns to ramble. Two defenses: subtract a length penalty from the reward, or ensure the reward model's training data balanced lengths. Always plot mean response length over training; a steady climb with rising reward is the length-hacking signature.
- **A separate, frozen reference model for the KL.** The reference must be a *distinct* frozen copy (or, with LoRA, the adapter-disabled base). If you accidentally point the KL computation at the live policy, the KL is identically zero, the fence disappears, and the policy reward-hacks within a few steps. This is a shockingly common bug — verify your measured KL is nonzero in step 1.
- **Gradient clipping** on the policy (global-norm clip, e.g. `max_grad_norm=1.0`) on top of the PPO clip. The PPO clip bounds the *objective*; gradient clipping bounds the *step* in parameter space and catches the occasional pathological batch.
- **The frozen reward model.** As covered in section 8, freezing it keeps the optimization target fixed. A reward model that drifts is a moving goalpost the policy can never reach.

The reason all of these cluster together: PPO for LLMs is a high-variance optimization on a fragile, enormous network against a gameable reward. Every trick is a variance-reduction or a guardrail. Skip them and you will spend your debugging time (section 9) re-discovering why they exist.

## 8. Memory and compute: four models at once

The defining practical pain of PPO-RLHF is that you hold **four models in memory simultaneously**, and only one of them is being trained.

1. **Policy** $\pi_\theta$ — trainable, needs gradients, optimizer states (Adam = 2× params in fp32), and activations. The expensive one.
2. **Reference** $\pi_{SFT}$ — frozen, forward-only, for the KL term. Same architecture as the policy.
3. **Reward model** — frozen, forward-only, scores responses. Often the same size as the policy.
4. **Value function** — usually a head on the policy body (so it shares the policy's parameters), but it adds optimizer state and a separate set of activations for the value loss.

A rough memory accounting for a 7B policy in mixed precision (bf16 weights, fp32 optimizer states):

| Component | Approx memory (7B model) |
| --- | --- |
| Policy weights (bf16) | ~14 GB |
| Policy Adam states + fp32 master (training) | ~56 GB |
| Policy activations (batch-dependent) | ~10–40 GB |
| Reference model (bf16, frozen) | ~14 GB |
| Reward model (bf16, frozen, ~7B) | ~14 GB |
| **Total** | **~110–140 GB** |

That does not fit on a single 80 GB GPU — hence the **4-GPU minimum** for 7B-scale RLHF, with the policy sharded across GPUs via DeepSpeed ZeRO-3 or FSDP and the frozen models placed to balance memory. The standard mitigations, in order of impact:

- **LoRA / QLoRA on the policy.** Train only low-rank adapters. This collapses the trainable-parameter count by 100–1000×, which means the Adam-state and master-weight cost (the biggest line item) nearly vanishes. A LoRA RLHF run on a 7B model can fit on a single 24–48 GB GPU. As a bonus, the *reference* model is just the policy with the LoRA adapters disabled — so you can drop the separate reference model entirely and save another 14 GB. This is the single highest-leverage trick for making RLHF affordable.
- **Mixed precision (bf16).** Halves weight and activation memory versus fp32. Standard.
- **Gradient checkpointing.** Trades compute for memory by recomputing activations in the backward pass instead of storing them — typically cuts activation memory by 60–80% at the cost of ~30% more compute. Essential for long responses.
- **Shared body for the value head.** As discussed in section 4, avoids a fifth full model.
- **Smaller / distilled reward model.** Nothing requires the reward model to be the same size as the policy. A well-trained 1.3B reward model scoring a 7B policy is common and saves substantial memory.

```python
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],   # attention projections
    task_type="CAUSAL_LM",
)

# Policy with LoRA adapters + value head. The reference is this same model
# with adapters disabled (peft_config makes TRL skip a separate ref model).
policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    peft_config=lora_config,
    load_in_4bit=True,                      # QLoRA: 4-bit frozen base
    torch_dtype=torch.bfloat16,
)
# With peft_config set, pass ref_model=None to PPOTrainer; TRL toggles adapters.
```

A note on the **frozen reward model and frozen reference**: both must stay frozen for correctness, not just thrift. If the reward model trained alongside the policy, you would have a moving target — the policy would chase a reward that keeps changing, and the run would never stabilize. If the reference drifted, the KL term would measure distance to a moving anchor, defeating its entire purpose of keeping the policy near a *known-good* distribution. The reference is the SFT model at the moment training started; it never updates. Freezing both is what makes the optimization well-posed.

## 9. Training instabilities and how to debug them

RLHF is notoriously finicky. Here are the failure modes I have actually hit, what they look like in the logs, and the fix. Figure 8 turns this into a decision tree you can run from your dashboard.

![Decision tree for debugging a stuck PPO-RLHF run, branching on reward spiking, no improvement, mode collapse, and value divergence to the corresponding fix](/imgs/blogs/ppo-for-llm-fine-tuning-8.png)

**Reward spiking / policy collapse.** The reward-model score suddenly shoots up while the KL also explodes and the generated text turns to garbage (repeated tokens, degenerate phrases that the reward model happens to love). This is reward hacking, and it is the most common catastrophic failure. **Fix:** increase $\beta$ (or lower the adaptive controller's target KL), and verify your per-token KL is actually being applied. Sometimes the bug is that the KL penalty is silently zero because the reference model was accidentally set equal to the policy. If the reward shoots to its ceiling in a handful of steps, suspect the KL plumbing first.

**No improvement.** Reward stays flat across hundreds of steps. **Two causes:** (a) $\beta$ is too large, straitjacketing the policy — lower it. Or (b) the reward model itself is the problem — it might be poorly calibrated, or the policy's outputs are out-of-distribution for the reward model so it returns noise. Diagnose by checking the reward model's scores on a held-out set of known-good vs known-bad responses; if it can't separate them, fix the reward model before touching PPO.

**Mode collapse.** All responses converge to nearly identical text. The reward goes up (the model found one high-reward template) but diversity craters. **Fix:** add or increase an entropy bonus, raise sampling temperature during rollouts, and check that you are sampling (`do_sample=True`) rather than generating greedily. Track self-BLEU (section 10) to catch this early — a sharp rise in self-BLEU is the alarm.

**Value function divergence.** The value loss grows instead of shrinking, and advantages become wildly noisy, which then destabilizes the policy. **Fix:** lower the value-function learning rate (or its coefficient `vf_coef`), make sure you are using the value clipping (`cliprange_value`), and confirm the value head was initialized from the reward model rather than randomly. A diverging critic poisons every advantage estimate, so this can masquerade as a policy problem — always check `ppo/val/error` before blaming the actor.

The general debugging discipline: **log per-batch, and watch the reward–KL relationship as a pair, never reward alone.** Reward going up with controlled KL = real progress. Reward going up with exploding KL = hacking. Reward flat with low KL = over-constrained. The whole story is in those two curves. For deeper, modality-specific failure hunting, the debugging-training track has a full playbook on isolating where in the pipeline a bug hides; the same bisection mindset applies here.

| Symptom (in logs) | Likely cause | First fix |
| --- | --- | --- |
| Reward ↑, KL explodes, text degenerate | reward hacking | raise β / lower target KL |
| Reward flat, KL near zero | β too large | lower β |
| Reward flat, KL moderate | weak/miscalibrated reward model | fix the RM |
| Self-BLEU ↑, diversity ↓ | mode collapse | entropy bonus, higher temp |
| `clipfrac` ≈ 1.0 | learning rate too high | lower LR |
| `clipfrac` ≈ 0.0 | learning rate too low | raise LR |
| Value error grows | critic diverging | lower vf LR, value-clip, RM-init |

## 10. Evaluation: did it actually get better?

Reward-model score is *not* your evaluation metric — it is the thing you optimized, so by Goodhart's law it stops being a trustworthy measure of quality. You need independent evaluation.

**Win rate against the SFT baseline.** The gold standard. Take held-out prompts, generate from both the PPO policy and the original SFT model, and have a judge (humans, or a strong LLM judge) pick the better response. Report the fraction of prompts where the PPO model wins. InstructGPT's headline result was framed this way: labelers preferred the 1.3B PPO model's outputs over the 175B SFT model's outputs a majority of the time — a smaller aligned model beating a much larger unaligned one.

**Standardized benchmarks.** `MT-Bench` (multi-turn questions scored by GPT-4 as judge, on a 1–10 scale) and `AlpacaEval` (win rate against a reference model's responses, also LLM-judged) are the common automated proxies. They correlate reasonably with human preference and are cheap to run repeatedly during development.

**The reward–KL Pareto frontier.** Run training at several values of $\beta$ and plot final reward against final KL. You get a frontier: for each KL budget, there is a best achievable reward. This is the single most informative plot in RLHF because it separates "the policy improved because it's genuinely better" from "the reward went up because we let KL run wild." You want to operate at the knee of that frontier — the point past which buying more reward costs disproportionate KL (and therefore quality).

**Response diversity via self-BLEU.** Generate many responses to the same prompt (or across prompts) and compute BLEU of each against the others; high self-BLEU means the responses are near-duplicates, i.e. mode collapse. Track it across training so a collapse doesn't sneak past a rising reward curve.

```python
# Win-rate evaluation: PPO policy vs SFT baseline, judged pairwise.
import torch

@torch.no_grad()
def generate(model, tok, prompt, **kw):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=128, do_sample=True,
                         top_p=0.9, **kw)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def win_rate(ppo_model, sft_model, tok, prompts, judge):
    wins = 0
    for p in prompts:
        a = generate(ppo_model, tok, p)   # candidate
        b = generate(sft_model, tok, p)   # baseline
        # judge returns "A" or "B"; randomize order in practice to kill bias
        if judge(p, a, b) == "A":
            wins += 1
    return wins / len(prompts)

# rate = win_rate(ppo_model, sft_model, tok, eval_prompts, gpt4_judge)
# Report e.g. "PPO beats SFT on 68% of held-out prompts (n=500)."
```

## 11. Case study: InstructGPT's PPO training details

InstructGPT (Ouyang et al., "Training language models to follow instructions with human feedback," 2022) is the canonical, fully-documented PPO-RLHF run, and its recipe became the template the whole industry copied. The pipeline had three stages, and PPO was stage three.

**Stage 1 — SFT.** Fine-tune GPT-3 on ~13,000 human-written demonstrations of desired behavior. This produces $\pi_{SFT}$, which becomes both the starting point for the policy *and* the frozen reference for the KL term.

**Stage 2 — Reward model.** Collect human rankings of model outputs (labelers ranked 4–9 responses per prompt) and train a reward model on ~33,000 prompts to predict those preferences with the pairwise ranking loss. Notably, OpenAI used a **6B reward model** to train the larger policies — the reward model did not need to be as large as the policy.

**Stage 3 — PPO.** This is our subject. The reported details:

- **Policy sizes:** 1.3B, 6B, and 175B variants were all PPO-trained.
- **KL coefficient:** they used the per-token KL penalty against the SFT reference, with the KL term central to preventing over-optimization; the coefficient was tuned per model, in the small-but-nonzero regime we have been discussing (single-digit-tenths order), with the adaptive-KL variant targeting a fixed KL.
- **A "PPO-ptx" variant** mixed the pretraining gradient back in (a small fraction of pretraining log-likelihood added to the PPO objective) to prevent the "alignment tax" — a regression on standard NLP benchmarks caused by over-fitting to the human-preference distribution. This is a real and important detail: pure PPO-RLHF can make a model *better* at being helpful while quietly *worse* at, say, SQuAD, and mixing in pretraining gradients counteracts that.
- **Batch sizes** were large (hundreds of prompts per PPO step), with multiple PPO epochs per batch, exactly the data-reuse pattern from section 2.
- **Learning rates** were small (on the order of $10^{-6}$ for the largest model), reflecting the fragility of large policies.

The headline result, again: human labelers preferred the **1.3B PPO model over the 175B GPT-3** SFT baseline on the prompt distribution — a ~130× smaller model winning on the metric that mattered (human preference) because it was aligned. That is the entire commercial case for RLHF in one sentence: alignment beat scale.

A few design choices from InstructGPT are worth lifting out because they recur in every serious RLHF run since. First, the *same* SFT checkpoint served three roles — it initialized the policy, it was frozen as the KL reference, and (with the LM head swapped for a scalar head) it initialized the reward model. Reusing one base everywhere keeps the four models architecturally compatible and the value/reward heads well-matched. Second, they found the reward model's reliability degraded as the policy improved, which directly motivated keeping $\beta$ high enough that the policy stayed in the reward model's reliable region rather than wandering into its blind spots — the over-optimization phenomenon they explicitly measured and plotted as reward-vs-KL curves. Third, the PPO-ptx pretraining-mix term was not a minor footnote: without it, the aligned model regressed measurably on public NLP benchmarks (the "alignment tax"), and mixing pretraining gradients back in recovered most of that loss while keeping the human-preference gains. The lesson that propagated: optimizing hard for one reward silently costs you capabilities elsewhere, and you must actively defend the capabilities you care about.

### Case study: TRL's sentiment demo, measured

On the much smaller, fully-runnable scale of the TRL sentiment example from section 7 (GPT-2, distilbert-imdb as reward), a typical run looks like this: starting mean reward (positive-sentiment probability) around 0.50, climbing to roughly 0.85–0.90 over ~200 PPO steps, while the KL from the reference rises to the single-digit nats and plateaus there. The generated continuations shift from neutral or negative movie-review fragments toward enthusiastically positive ones — visibly so, and you can read them. It is the cheapest possible way to *see* RLHF work end to end on your own machine, and I recommend running it before you ever touch a 7B model.

### Case study: Llama-2-Chat's iterative RLHF

Llama-2 (Touvron et al., 2023) ran PPO-RLHF iteratively over five rounds, retraining the reward model each round on fresh human preferences as the policy improved (since the old reward model becomes less reliable on the new, better policy's outputs — a distribution-shift problem). They also used *two* reward models, one for helpfulness and one for safety, and combined them. The takeaway for practitioners: at the frontier, RLHF is not one PPO run but a *loop* of (collect preferences → train RM → PPO → collect preferences on the new policy → ...), because the reward model's reliability degrades as the policy moves away from the data the RM was trained on.

## 12. When to use PPO-RLHF (and when not to)

PPO-RLHF is powerful and expensive and finicky, so be deliberate about reaching for it.

**Use PPO-RLHF when:** you have (or can collect) a real human-preference signal, you need to optimize a reward that is *not* expressible as a fixed target sequence (helpfulness, harmlessness, style, factuality-as-judged), and you have the compute for the four-model setup. It remains the most flexible alignment tool — it can optimize *any* scalar reward, including non-differentiable or rule-based ones (e.g. "does this code pass the unit tests"), which preference-based offline methods cannot easily do.

**Prefer DPO (Direct Preference Optimization) when:** you have a static dataset of preference pairs and don't need online sampling. DPO derives the same KL-regularized optimal-policy identity from section 3 and turns it into a simple classification loss on the preference pairs — no reward model, no value function, no rollouts, no four-model juggle. It is dramatically simpler and cheaper, and for many alignment tasks it matches PPO's quality. If your problem fits the offline-preference-pairs mold, start with DPO and only escalate to PPO if you hit its limits. The full treatment is in [DPO: Direct Preference Optimization](/blog/machine-learning/reinforcement-learning/dpo-direct-preference-optimization).

**Prefer GRPO (Group Relative Policy Optimization) when:** you are doing RLHF at scale and want to *drop the value head entirely*. GRPO, which DeepSeek popularized in 2024, samples a *group* of responses per prompt and uses the group's mean reward as the baseline instead of a learned critic. This removes one of the four models (the value function) and its instabilities, cutting memory and a whole class of bugs. It has become the default for reasoning-focused RLHF (where you can generate many samples per prompt cheaply). If value-function divergence has been eating your runs, GRPO is the structural fix — see [GRPO: Group Relative Policy Optimization](/blog/machine-learning/reinforcement-learning/grpo-group-relative-policy-optimization).

**Don't use any of these when:** plain SFT on high-quality demonstrations gets you there. If you can write or collect good target responses, supervised fine-tuning is simpler, cheaper, and more stable than any RL method. RLHF earns its complexity only when you need to optimize a *preference* you cannot directly demonstrate. Figure 5 lays out how the main PPO hyperparameters trade off, and Figure 6 places PPO in the historical arc that leads to these successors.

![Matrix of PPO-RLHF hyperparameters showing how beta, learning rate, batch size, GAE lambda, and clip ratio each affect reward, KL, diversity, with typical values](/imgs/blogs/ppo-for-llm-fine-tuning-5.png)

![Timeline of PPO for LLMs from 2017 preference learning through InstructGPT, Claude, Llama-2, the TRL library, and GRPO replacing the value head in 2024](/imgs/blogs/ppo-for-llm-fine-tuning-6.png)

| Method | Needs reward model? | Needs value fn? | Online sampling? | Best for |
| --- | --- | --- | --- | --- |
| SFT | no | no | no | demonstrable targets |
| PPO-RLHF | yes | yes | yes | flexible reward, any scalar |
| DPO | no | no | no | static preference pairs |
| GRPO | yes | **no** | yes | scaled / reasoning RLHF |

## Key takeaways

- **An LLM is a policy in an MDP** where the state is the prompt-plus-generated-tokens, each action is a token, the episode is one response, and the reward is a sparse terminal score from a reward model.
- **PPO won because of the clip.** The clipped surrogate objective caps how far any token's probability can move per batch, turning fragile billion-parameter policy gradient into a stable, reusable update.
- **The KL penalty is non-negotiable.** It is what stops the policy from reward-hacking the imperfect reward model; provably, it confines the optimal policy to the reference model's support. $\beta$ in the 0.1–0.5 range, often adaptive against a target KL, is the single most important knob.
- **GAE solves credit assignment.** One terminal reward becomes a per-token advantage by flowing backward through value-function bootstrapping, with $\lambda \approx 0.95$ tuning bias against variance.
- **Initialize the value head from the reward model**, share the transformer body, and keep both the reward model and the reference frozen — a moving target or a drifting anchor breaks the optimization.
- **You hold four models in memory at once.** LoRA on the policy (which also doubles as the reference via adapter-toggling), a smaller reward model, gradient checkpointing, and bf16 are how you make it fit; 7B-scale RLHF wants 4+ GPUs.
- **Watch reward and KL as a pair, never reward alone.** Rising reward with exploding KL is hacking, not progress. Log `clipfrac`, `approx_kl`, and value error every step.
- **Evaluate independently of the reward model** — win rate vs SFT, MT-Bench, AlpacaEval, and the reward–KL Pareto frontier — and track self-BLEU to catch mode collapse.
- **Reach for the simpler tool first.** SFT if you can demonstrate, DPO if you have static preference pairs, GRPO to drop the value head at scale; PPO when you need maximal flexibility over an arbitrary reward.

## Further reading

- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), 2022 — the canonical PPO-RLHF recipe with measured win rates.
- Schulman et al., "Proximal Policy Optimization Algorithms," 2017 — the clipped surrogate objective and the trust-region intuition.
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation," 2016 — the GAE derivation underlying section 5.
- Christiano et al., "Deep Reinforcement Learning from Human Preferences," 2017 — the origin of RLHF with a learned reward model.
- Ziegler et al., "Fine-Tuning Language Models from Human Preferences," 2019 — the per-token KL penalty against a reference policy, formalized.
- Rafailov et al., "Direct Preference Optimization," 2023 — the closed-form optimal policy identity and how to skip the RL step.
- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," 2023 — iterative RLHF with helpfulness and safety reward models.
- Hugging Face TRL documentation — `PPOTrainer`, `PPOConfig`, `AutoModelForCausalLMWithValueHead`, and the runnable sentiment example.
- Within this series: the unified map of RL methods (`reinforcement-learning-a-unified-map`) and the capstone playbook (`the-reinforcement-learning-playbook`) place PPO-RLHF among its alternatives; the policy-gradient and PPO posts upstream derive the clipped objective in the general continuous-control setting that this post specializes to language.

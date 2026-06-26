---
title: "DPO: Direct Preference Optimization Without a Reward Model"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A from-first-principles derivation of Direct Preference Optimization — why the KL-constrained RLHF objective has a closed form, how that turns reward into a log-ratio, the full DPO loss, a runnable PyTorch and TRL implementation, the IPO/KTO/ORPO variants, and exactly when DPO beats PPO-RLHF and when it does not."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "rlhf",
    "dpo",
    "llm-alignment",
    "machine-learning",
    "pytorch",
    "preference-optimization",
    "trl",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/dpo-direct-preference-optimization-1.png"
---

The first time I aligned a 7B model with PPO-RLHF, the worst part was not the math. It was the *bookkeeping*. I had a policy model being trained, a frozen reference model for the KL penalty, a separate reward model that itself had to be trained on preference data, and a value head for advantage estimation. Four sets of weights, three of them on the GPU at once, a reward model whose accuracy I could never fully trust, and an online RL loop that would happily discover that appending "I hope this helps!" to every answer nudged the reward up by 0.3. Two weeks in, the model had learned to reward-hack the reward model rather than become more helpful. I had a beautiful training curve and a useless checkpoint.

That experience is the entire motivation for Direct Preference Optimization. DPO, introduced by Rafailov et al. in 2023, asks a deceptively simple question: if the whole point of RLHF is to make the model prefer responses that humans prefer, do we actually need the reward model and the RL loop in the middle? Or is there a way to optimize the preference data *directly* — with a single supervised-style loss, no sampling, no reward model, no value head? The answer, it turns out, is yes, and the derivation is one of the cleaner pieces of math in modern alignment. The reward model and the RL loop are not fundamental; they are an artifact of how we *thought* the problem had to be solved. Figure 1 shows the before-and-after at a glance: four models and an unstable online loop on the left, two models and one classification loss on the right.

![Side-by-side comparison of the four-model PPO-RLHF pipeline against the two-model DPO pipeline showing the removal of the reward model and the RL loop](/imgs/blogs/dpo-direct-preference-optimization-1.png)

By the end of this post you will be able to derive the DPO loss from the RLHF objective yourself, implement it in raw PyTorch in about forty lines, run it with Hugging Face TRL's `DPOTrainer`, reason about the `beta` hyperparameter the way you reason about a learning rate, choose between DPO and its cousins IPO, KTO, and ORPO based on the shape of your data, and — most importantly — know when to reach for DPO and when PPO-RLHF is still the right tool. This is a Reinforcement Learning series, so we will keep tying everything back to the spine: an agent (the language model policy) interacting with an environment (prompts and the implicit human judge), collecting rewards (preferences), and updating a policy. DPO's trick is that it never instantiates the reward or runs the loop — it solves for the optimal policy in closed form first, then fits it. For the RL framing of the same objective, this post builds directly on [Proximal Policy Optimization](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo), and you should read these alongside the unified map `reinforcement-learning-a-unified-map` and the capstone `the-reinforcement-learning-playbook` once they are published.

## 1. The RLHF bottleneck: why four models is a problem

Let us be precise about what standard RLHF actually requires, because "four models" is not hyperbole. The classic InstructGPT recipe (Ouyang et al., 2022) has three stages, and the third stage is where the cost concentrates.

**Stage 1 — Supervised fine-tuning (SFT).** Start from a pretrained base model and fine-tune on high-quality demonstrations of the desired behavior. This gives you $\pi_{\text{SFT}}$, a model that is already reasonably good at following instructions. Call it the *reference policy* $\pi_{\text{ref}}$ for what comes next.

**Stage 2 — Reward modeling.** Collect human preference data: for a prompt $x$, show annotators two responses $y_1$ and $y_2$ and ask which they prefer. Train a separate model $r_\phi(x, y)$ — usually the SFT model with the language-modeling head swapped for a scalar head — to predict a reward such that the preferred response scores higher. The standard loss here is the Bradley-Terry negative log-likelihood, which we will meet again shortly.

**Stage 3 — RL fine-tuning.** Now optimize the policy $\pi_\theta$ to maximize the learned reward while staying close to the reference, using PPO. The objective is the KL-constrained reward maximization

$$
\max_{\pi_\theta}\; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\big[r_\phi(x, y)\big] \;-\; \beta\, D_{\mathrm{KL}}\!\big(\pi_\theta(\cdot \mid x)\,\|\,\pi_{\text{ref}}(\cdot \mid x)\big).
$$

The KL term is not optional decoration. Without it, the policy will drift arbitrarily far from the reference to chase reward, and since the reward model is only accurate near the distribution it was trained on, the policy ends up exploiting reward-model errors — producing text that the reward model loves and humans find bizarre. This is *reward hacking*, and the KL leash is the standard defense. (For the deeper treatment of why KL prevents distribution shift, see the PPO post linked above; the RLHF connection there is the same objective we are about to solve in closed form.)

Now count the models you need on the GPU during stage 3:

1. The **policy** $\pi_\theta$ being trained — full gradients.
2. The **reference** $\pi_{\text{ref}}$ — frozen, but you need its log-probs for the KL term.
3. The **reward model** $r_\phi$ — frozen, evaluated on every generated sample.
4. The **value model** (critic) for PPO's advantage estimation — usually a separate head, sometimes a separate network, with its own gradients.

That is the bookkeeping I complained about. To make "four models is a problem" concrete rather than rhetorical, put rough numbers on it for a 7B model in mixed precision. Each model's weights are about 14 GB. The policy needs weights plus gradients plus optimizer state (Adam keeps two moments), so call it roughly 14 GB weights, 14 GB gradients, 28 GB optimizer state — on the order of 56 GB before activations. The value model, if it is a full separate network with its own gradients and optimizer, adds a comparable chunk. The reference and reward models are frozen, so they are "only" their 14 GB of weights each, but they still must live on the GPU to be queried every step. Add it up and a 7B PPO-RLHF run can demand well over 100 GB of accelerator memory, which is why it is typically a multi-GPU, multi-node affair with model sharding. DPO, with one trainable model (56 GB) and one frozen reference (14 GB), fits in roughly 70 GB — and with LoRA, as Section 8 shows, it collapses to the memory of a single model plus tiny adapters and runs on one consumer GPU. The memory argument alone moves DPO from "data-center project" to "weekend project," and that accessibility is a large part of why it took over open-source alignment.

Beyond memory, there are three structural pains. First, **the reward model is a lossy intermediary**: it is a finite-capacity approximation of human preference, and any systematic error in it becomes a target the policy will optimize toward. Second, **the RL loop is online**: every update requires sampling fresh responses from the current policy, scoring them, and computing advantages — generation is slow, and the on-policy data is expensive and high-variance (the policy gradient is a Monte Carlo estimator of the score function, with all the variance that implies). Third, **PPO is finicky**: the clip range, the KL coefficient, the value-loss weight, the number of epochs per batch, the advantage normalization — get any of them wrong and training either does nothing or collapses. I have spent more hours tuning PPO for RLHF than I care to admit.

The question DPO answers is: can we keep the *objective* — KL-constrained reward maximization on human preferences — but throw away the reward model, the value model, and the online loop? The insight is that the objective above has a known optimal solution, and that solution can be rearranged so that the reward never has to be computed explicitly.

## 2. The key insight: the RLHF objective has a closed-form optimum

Here is the pivot the whole method turns on. The KL-constrained objective in Section 1 is not some intractable thing you can only attack with gradient descent. For a *fixed* reward function $r(x, y)$, the policy that maximizes it has a known closed form. Figure 2 traces the full derivation as a dataflow graph; we will walk each node.

![Dataflow graph showing the RLHF objective branching into the analytic optimal policy and the KL constraint then merging into a log-ratio and combining with Bradley-Terry to produce the DPO loss](/imgs/blogs/dpo-direct-preference-optimization-2.png)

Consider the per-prompt objective. For a single prompt $x$, we want the distribution over responses $\pi(\cdot \mid x)$ that maximizes

$$
J(\pi) = \mathbb{E}_{y \sim \pi}\big[r(x, y)\big] - \beta\, D_{\mathrm{KL}}\big(\pi \,\|\, \pi_{\text{ref}}\big).
$$

Expand the KL divergence as $\mathbb{E}_{y \sim \pi}\big[\log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\big]$ and rewrite the whole thing as a single expectation:

$$
J(\pi) = \mathbb{E}_{y \sim \pi}\!\left[ r(x,y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right].
$$

Divide by $\beta$ (it does not change the argmax) and flip the sign so we are minimizing:

$$
-\frac{1}{\beta} J(\pi) = \mathbb{E}_{y \sim \pi}\!\left[ \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - \frac{1}{\beta} r(x,y) \right] = \mathbb{E}_{y \sim \pi}\!\left[ \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)\, e^{\,r(x,y)/\beta}} \right].
$$

Now define a normalized distribution by stuffing the exponential reward into the reference:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\left( \frac{1}{\beta} r(x,y) \right), \qquad Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x)\, \exp\!\left( \frac{1}{\beta} r(x,y) \right).
$$

The partition function $Z(x)$ is whatever constant makes $\pi^*$ sum to one over all responses $y$. With this definition, the objective becomes a KL divergence between $\pi$ and $\pi^*$:

$$
-\frac{1}{\beta} J(\pi) = \mathbb{E}_{y \sim \pi}\!\left[ \log \frac{\pi(y \mid x)}{\pi^*(y \mid x)} \right] - \log Z(x) = D_{\mathrm{KL}}\big(\pi \,\|\, \pi^*\big) - \log Z(x).
$$

The $\log Z(x)$ term does not depend on $\pi$, so minimizing this is exactly minimizing $D_{\mathrm{KL}}(\pi \| \pi^*)$. And a KL divergence is minimized — driven to zero — precisely when the two distributions are equal. Therefore the optimal policy is

$$
\boxed{\;\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\left( \frac{1}{\beta} r(x,y) \right).\;}
$$

This is the single most important equation in DPO, and it has a clean reading: **the optimal RLHF policy is the reference policy reweighted by the exponentiated reward.** High-reward responses get their reference probability multiplied up; low-reward responses get multiplied down; $\beta$ controls how aggressively (small $\beta$ means a sharp reweighting, large $\beta$ means gentle). This is the Gibbs/Boltzmann distribution, the same form that shows up in statistical mechanics and in maximum-entropy RL — it is not a coincidence, since the KL-regularized objective *is* a maximum-entropy objective in disguise.

But here is the catch that makes this look useless at first: $Z(x)$ is a sum over *all possible responses* $y$. For a language model, that is a sum over every string the model could generate — utterly intractable. You cannot compute $\pi^*$ directly because you cannot compute $Z(x)$. This is exactly why the original RLHF pipeline resorts to RL: you cannot sample from $\pi^*$ in closed form, so you train $\pi_\theta$ to approximate it via gradient ascent on the reward. DPO's genius is to sidestep $Z(x)$ entirely, which is what the next section does.

Two technical points are worth nailing down so the derivation is airtight rather than hand-waved. First, **the optimum is unique.** The objective $-\frac{1}{\beta}J(\pi)$ equals $D_{\mathrm{KL}}(\pi \| \pi^*) - \log Z(x)$, and KL divergence is *strictly* convex in its first argument and zero if and only if the two distributions are identical. So there is exactly one maximizer, $\pi = \pi^*$ — no local optima, no ambiguity. This is why DPO can be derived as the optimization of a well-defined target rather than as a heuristic. Second, **the derivation makes no approximation up to this point.** We did not Taylor-expand, we did not assume small KL, we did not linearize anything. The boxed $\pi^*$ is the *exact* solution to the exact KL-constrained objective that PPO-RLHF tries to reach by gradient ascent. Any gap between DPO and PPO-RLHF in practice therefore comes not from the objective — they share it exactly — but from three places: the Bradley-Terry assumption about how preferences are generated, the finiteness and coverage of the preference dataset, and the absence of online exploration. Keep those three sources of difference in mind; they are the entire story of when DPO and RLHF diverge, and we return to each of them.

It also helps to read the boxed equation as a statement about *temperature*. Rewrite the exponent as $\exp(r/\beta)$ and note that $\beta$ plays the role of a temperature on the reward. As $\beta \to \infty$ (infinite temperature), $\exp(r/\beta) \to 1$ for every response, the reweighting vanishes, and $\pi^* \to \pi_{\text{ref}}$: the optimal policy is just the reference, ignoring reward entirely. As $\beta \to 0$ (zero temperature), the exponential sharpens until all probability mass concentrates on the single highest-reward response — $\pi^*$ becomes a greedy argmax over reward, ignoring the reference. Real alignment lives in between, and $\beta$ is the dial. This is the same $\beta$ that will appear in the DPO loss, and it is the same trade-off Section 6 develops mechanically; seeing it here, in the optimal policy itself, explains *why* the dial behaves as it does downstream — it was baked in at the level of the objective, not bolted on at the level of the loss.

## 3. From optimal policy to the DPO loss

We have $\pi^*$ in terms of $r$. The trick is to run the relationship *backwards*: solve the boxed equation for the reward $r(x, y)$ in terms of the policy. Take logs of both sides of the boxed equation and rearrange:

$$
\log \pi^*(y \mid x) = \log \pi_{\text{ref}}(y \mid x) + \frac{1}{\beta} r(x, y) - \log Z(x),
$$

$$
\boxed{\;r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x).\;}
$$

Read this carefully, because it is the whole game. It says that **for the optimal policy, the reward of a response is exactly $\beta$ times the log-ratio of the policy to the reference, plus a term that depends only on the prompt** (through $Z(x)$). In other words, *the policy is its own reward model.* If we knew the optimal policy, we could read off the implied reward without ever training a separate reward network. The reward and the policy are two views of the same object, related by a log-ratio.

The remaining obstacle is still $\log Z(x)$ — we cannot compute it. This is where preferences save us. Recall that human preferences are modeled with the **Bradley-Terry** model. The Bradley-Terry model says the probability that response $y_w$ is preferred over $y_l$ (subscripts: $w$ for "winner/chosen", $l$ for "loser/rejected") given a latent reward $r$ is

$$
P(y_w \succ y_l \mid x) = \frac{\exp r(x, y_w)}{\exp r(x, y_w) + \exp r(x, y_l)} = \sigma\big( r(x, y_w) - r(x, y_l) \big),
$$

where $\sigma$ is the logistic sigmoid. The crucial feature: Bradley-Terry depends only on the **difference** of rewards, $r(x, y_w) - r(x, y_l)$. Now substitute the boxed reward expression for both responses:

$$
r(x, y_w) - r(x, y_l) = \Big(\beta \log \tfrac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x)\Big) - \Big(\beta \log \tfrac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \log Z(x)\Big).
$$

The two $\beta \log Z(x)$ terms — the intractable parts — **cancel exactly**, because $Z(x)$ depends only on the prompt $x$, which is shared by both responses. What survives is

$$
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}.
$$

Now we replace the unknown optimal policy $\pi^*$ with our trainable policy $\pi_\theta$ and plug the reward difference into the Bradley-Terry likelihood. We want to maximize the likelihood of the observed preferences, which is the same as minimizing the negative log-likelihood. The result is the **DPO loss**:

$$
\boxed{\;\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\,\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right].\;}
$$

That is it. No reward model. No partition function. No RL loop. No value head. The loss is a binary classification loss — log-sigmoid of a margin — over preference pairs, computed entirely from policy and reference log-probabilities. The reference log-probs are constants (the reference is frozen), so the only thing with gradients is $\pi_\theta$. You can train this with the same machinery you use for any supervised fine-tuning: a forward pass, a scalar loss, `loss.backward()`, an optimizer step.

It is worth pausing on what just happened. We started from the *RL* objective — the exact same KL-constrained reward maximization that PPO optimizes — and, by solving it analytically and exploiting the structure of Bradley-Terry, we arrived at a loss that has nothing RL-shaped left in it. DPO is not an approximation of RLHF that happens to be cheaper; under the Bradley-Terry assumption it optimizes *the same objective*, just reparameterized. The reward model in RLHF and the implicit reward $\beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$ in DPO are the same quantity, fit two different ways.

#### Worked example: computing the DPO loss by hand

Let us make the loss concrete with numbers. Take one preference pair: a prompt $x$, a chosen response $y_w$, a rejected response $y_l$. Suppose we have computed the sum of token log-probabilities (the sequence log-prob) for each response under both models. Let $\beta = 0.1$.

Under the **policy** $\pi_\theta$ (early in training, before it has learned the preference):

- $\log \pi_\theta(y_w \mid x) = -8.0$
- $\log \pi_\theta(y_l \mid x) = -7.5$ (the policy currently slightly *prefers* the rejected response — a problem)

Under the frozen **reference** $\pi_{\text{ref}}$:

- $\log \pi_{\text{ref}}(y_w \mid x) = -8.2$
- $\log \pi_{\text{ref}}(y_l \mid x) = -7.4$

Compute the two log-ratios:

- Chosen: $\log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} = -8.0 - (-8.2) = +0.2$
- Rejected: $\log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)} = -7.5 - (-7.4) = -0.1$

The margin inside the sigmoid:

$$
\beta\left( 0.2 - (-0.1) \right) = 0.1 \times 0.3 = 0.03.
$$

The loss for this pair:

$$
-\log \sigma(0.03) = -\log\!\big(0.5075\big) = 0.678.
$$

For reference, $-\log\sigma(0) = \log 2 \approx 0.693$, the loss when the model is exactly indifferent. Our loss of $0.678$ is just barely below that, meaning the model is only marginally on the right side. The gradient will push to *increase* the margin: raise $\log \pi_\theta(y_w)$ and/or lower $\log \pi_\theta(y_l)$, relative to the reference. After a few hundred steps you would expect the chosen log-ratio to climb toward, say, $+1.5$ and the rejected to fall toward $-1.0$, giving a margin of $0.1 \times 2.5 = 0.25$ and a loss of $-\log\sigma(0.25) = 0.575$. The "implied reward margin" the model has learned for this pair is $0.25$ in reward units. This is exactly the quantity a separate reward model would have tried to learn — except here it lives inside the policy's own log-ratios.

## 4. What DPO does mechanically: the gradient tells the story

The cleanest way to build intuition for any loss is to look at its gradient. Differentiate the DPO loss with respect to the policy parameters $\theta$. Writing $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ for the implicit reward, the gradient is

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\, \mathbb{E}_{(x, y_w, y_l)}\Big[ \underbrace{\sigma\big(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\big)}_{\text{weight}} \big( \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \big) \Big].
$$

Read the two pieces. The bracketed gradient term, $\nabla_\theta \log \pi_\theta(y_w) - \nabla_\theta \log \pi_\theta(y_l)$, **increases the log-probability of the chosen response and decreases the log-probability of the rejected response.** That is the entire mechanical action of DPO: push chosen up, push rejected down — *relative to the reference*, because the reference terms have no gradient. Figure 3 lays out this computation as a stack, from the two forward passes through the log-ratios to the final gradient.

![Layered stack showing the DPO loss computation from reference and policy forward passes through log-ratios and the sigmoid margin to the policy-only gradient update](/imgs/blogs/dpo-direct-preference-optimization-3.png)

The weight $\sigma\big(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\big)$ is the cleverer half. It is the model's estimated probability that it has the pair *wrong* — that it currently scores the rejected response higher than the chosen one. When the model already gets the pair right (chosen implied-reward much higher than rejected), this weight is near zero and the example contributes almost no gradient. When the model gets it badly wrong, the weight is near one and the example contributes a large gradient. **DPO automatically focuses learning on the pairs it is getting wrong** and stops pushing on pairs it has mastered. This is a built-in form of importance weighting, and it is why DPO does not catastrophically over-optimize easy examples the way a naive "increase chosen, decrease rejected" loss would.

There is a subtle and important consequence here that trips up newcomers. DPO does **not** necessarily increase the *absolute* probability of the chosen response. It increases the chosen log-ratio relative to the reference, and decreases the rejected log-ratio relative to the reference. In practice, what very often happens is that *both* $\log \pi_\theta(y_w)$ and $\log \pi_\theta(y_l)$ go **down** during training, with the rejected one falling faster. The model is not learning "the chosen response is great" so much as "the rejected response is even worse than the reference thought." If you log both quantities during training (and you should — see the implementation section), you will frequently see the chosen log-prob drift slowly downward while the margin grows. This is normal. It only becomes a problem when the chosen log-prob falls so far that the model starts producing degenerate text, which is one of the failure modes we will discuss with the `beta` parameter.

#### Worked example: how beta scales the same logprob change

Take the same pair as before but ask: how much does the *loss* care about a fixed improvement in the log-ratios, as a function of $\beta$? Suppose training moves the chosen log-ratio from $+0.2$ to $+1.2$ and the rejected from $-0.1$ to $-1.1$ — a raw log-ratio margin change from $0.3$ to $2.3$.

With $\beta = 0.1$: margin goes from $0.03$ to $0.23$; loss goes from $0.678$ to $-\log\sigma(0.23) = 0.583$. A drop of $0.095$.

With $\beta = 0.5$: margin goes from $0.15$ to $1.15$; loss goes from $0.625$ to $-\log\sigma(1.15) = 0.273$. A drop of $0.352$.

With $\beta = 1.0$: margin goes from $0.30$ to $2.30$; loss goes from $0.554$ to $-\log\sigma(2.3) = 0.0954$. A drop of $0.459$.

Same change in the underlying log-ratios, wildly different loss sensitivity. Larger $\beta$ makes the same policy movement count for far more in the loss, which means the optimizer drives the log-ratios harder before the gradient vanishes — the policy moves *further* from the reference. Smaller $\beta$ makes the loss nearly flat, so the policy barely budges from the reference. This is the mechanical reason $\beta$ behaves like a leash length, which Section 6 develops fully.

## 5. DPO versus RLHF: the honest trade-offs

It is tempting to read all this and conclude DPO simply dominates RLHF. It does not. They optimize the same objective under the Bradley-Terry assumption, but the *way* they optimize it leads to real differences in capability. Here is the comparison I actually use when deciding.

| Dimension | PPO-RLHF | DPO |
|---|---|---|
| Models in memory | 4 (policy, ref, reward, value) | 2 (policy, ref) |
| Reward model | Required, trained separately | None — implicit in log-ratios |
| Training style | Online RL (sample, score, update) | Offline supervised-style |
| Compute | High (generation in the loop) | ~10x lower (no generation) |
| Stability | Finicky (clip, KL coef, value loss) | Stable (one classification loss) |
| Can explore beyond data | Yes — generates novel responses | No — bounded by the pair distribution |
| Hyperparameters to tune | Many | Mainly `beta` and learning rate |
| Reward hacking | Real risk (exploits reward model) | Different failure (overfits pairs) |
| Best for | New capabilities, online feedback | Static preference datasets |

The deepest difference is the one that does not show up in compute numbers: **exploration**. PPO-RLHF is *online*. At every step it samples fresh responses from the current policy, and the reward model scores responses the policy generates — including responses that appear *nowhere in the original preference dataset*. This means RLHF can discover and reinforce behaviors that the human annotators never explicitly ranked, as long as the reward model generalizes to score them well. The reward model acts as a learned judge that can evaluate novel outputs.

DPO has no such judge and no sampling. It only ever sees the responses in the dataset — the chosen and rejected pairs that were collected up front. **DPO is fundamentally bounded by the distribution of its training data.** It can learn to prefer the chosen responses over the rejected ones, and it can generalize that preference to nearby responses, but it cannot reward a brilliant response that was never in the dataset, because it never generates and scores one during training. If your preference data does not contain examples of a behavior, DPO has no mechanism to discover it.

This is the single most important practical distinction, and people get it wrong constantly. They run DPO, see it match or beat PPO-RLHF on their benchmark, and conclude DPO is strictly better. What they usually have is a *static* alignment task — make the model more polite, less likely to refuse benign requests, better at a fixed style — where all the signal is already in the preference pairs. On those tasks DPO wins because it is cheaper and more stable and there is nothing to explore. But on tasks where the win comes from the model *discovering* a better response than any in the dataset — hard reasoning, agentic tool use, anything where on-policy generation surfaces new high-reward behaviors — the online exploration of PPO-RLHF (and its modern descendants like GRPO) is doing real work that DPO structurally cannot replicate. For the head-to-head decision framework across all three, see [GRPO vs DPO vs PPO: a decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide).

There is also a subtler theoretical wrinkle worth knowing. DPO's derivation assumed the preference data was generated by a Bradley-Terry model with some true reward, and that we have enough coverage. But the optimal policy DPO targets, $\pi^* \propto \pi_{\text{ref}} \exp(r/\beta)$, is only well-defined on the support of $\pi_{\text{ref}}$. For responses the reference assigns near-zero probability, the implied reward is essentially unconstrained by the loss — there is no data and no sampling to pin it down. RLHF's online sampling provides at least some signal in these regions; DPO provides none. This is part of why DPO can produce strange out-of-distribution behavior if pushed too hard, and why the choice of reference and `beta` matters so much.

## 6. The reference policy and the role of beta

Two design choices dominate DPO's behavior: which model you use as the reference, and how you set $\beta$. Figure 7 shows the $\beta$ trade-off as a branching graph.

![Branching graph showing small beta keeping the policy near the reference and safe but limited, large beta drifting far and risking overfitting, both tuned toward an aligned policy that improves on SFT](/imgs/blogs/dpo-direct-preference-optimization-7.png)

**The reference policy is almost always the SFT model.** This is not arbitrary. The reference appears in the loss as the denominator of the log-ratio, so it defines the "zero point" — the policy DPO measures movement *away from*. If you use the SFT model as the reference and also initialize the trainable policy from the same SFT model, then at step zero $\pi_\theta = \pi_{\text{ref}}$, both log-ratios are exactly zero, the margin is zero, and the loss is exactly $\log 2$ for every pair. Training then moves the policy away from this perfectly calibrated starting point. Using a *different* model as the reference (or initializing the policy from somewhere other than the reference) breaks this calibration and tends to destabilize training — I have seen losses start at bizarre values and the model immediately drift. Rule of thumb: **policy and reference both start as the SFT model.** A few methods (ORPO, discussed later) drop the reference entirely, but they change the loss to compensate.

**Beta is the leash.** It controls how far DPO is allowed to pull the policy from the reference. Look back at the loss: $\beta$ multiplies the log-ratio difference inside the sigmoid. Think through the two extremes carefully (this paragraph sits under Figure 7, which illustrates exactly this trade-off).

When $\beta$ is **too small** (say $0.005$), the margin inside the sigmoid is tiny even for large log-ratio differences, so the loss is nearly flat and its gradient nearly vanishes. The optimizer has little incentive to move the policy at all, and DPO **collapses toward the reference** — you get the SFT model back, barely changed, ignoring the preferences. The model is "safe but limited."

When $\beta$ is **too large** (say $1.0$ or more), the margin saturates the sigmoid quickly, so the loss aggressively rewards driving the log-ratios apart. The policy moves far from the reference, and because the only signal is the finite preference dataset, it **overfits the pairs**: it learns idiosyncrasies of the specific chosen/rejected responses rather than the general preference, and can produce degenerate or off-distribution text. The model is "creative but risky."

In practice, $\beta \in [0.1, 0.5]$ is the standard range, with $0.1$ being the most common default (it is the value in the original DPO paper and the TRL default). Smaller models and smaller datasets often want a slightly larger $\beta$ (more aggressive learning from limited data); larger models with lots of data tolerate smaller $\beta$. Treat it like a learning rate: sweep $\{0.05, 0.1, 0.3, 0.5\}$ and pick by validation win-rate. A diagnostic I rely on: log the **reward accuracy** (the fraction of validation pairs where the chosen implied-reward exceeds the rejected) and the **mean chosen log-prob**. Healthy training shows reward accuracy climbing toward $0.7$–$0.9$ while the chosen log-prob does not crater. If reward accuracy is stuck near $0.5$, $\beta$ is too small or the learning rate is too low. If reward accuracy is high but the chosen log-prob has fallen off a cliff, $\beta$ is too large and you are over-optimizing.

| Hyperparameter | Effect | Typical range |
|---|---|---|
| `beta` | KL leash; how far from reference | 0.05 – 0.5 (default 0.1) |
| learning rate | Step size; DPO is sensitive | 1e-6 – 5e-6 (lower than SFT) |
| epochs | Passes over preference data | 1 – 3 (more risks overfitting) |
| max length | Response truncation | model max; truncation biases loss |
| batch size | Pairs per step | 4 – 64 (gradient-accumulate if small) |

Note the learning rate row: DPO wants a learning rate roughly an order of magnitude *lower* than typical SFT (around $5\times 10^{-6}$ versus $5\times 10^{-5}$). The loss is sharp and the model is already well-trained, so large steps blow it up fast. This is the single most common DPO bug I see — people reuse their SFT learning rate and watch the model degenerate within a few hundred steps.

## 7. Implementing DPO in raw PyTorch

Let us build the loss from scratch so there is no magic. The core is three functions: compute the sequence log-prob of a response, run both models, and combine into the loss. Figure 4 shows the training loop these functions implement.

![Pipeline diagram of the DPO training loop loading the SFT model as policy and frozen reference computing four log-probs per pair and updating only the policy](/imgs/blogs/dpo-direct-preference-optimization-4.png)

First, the heart of everything: computing the log-probability of a response given a prompt. We run the model, get per-token logits, take the log-softmax, and gather the log-prob of each *actual* token in the response. We mask out the prompt tokens (we only score the response) and any padding.

```python
import torch
import torch.nn.functional as F

def sequence_logprobs(model, input_ids, attention_mask, response_mask):
    """Sum of token log-probs over the RESPONSE tokens only.

    input_ids:     (batch, seq_len) prompt + response tokens
    attention_mask:(batch, seq_len) 1 for real tokens, 0 for padding
    response_mask: (batch, seq_len) 1 for response tokens to score, 0 elsewhere
    Returns: (batch,) summed log-prob of each response.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]          # predict token t from positions < t
    labels = input_ids[:, 1:]                    # the actual next tokens
    mask = response_mask[:, 1:]                  # align mask to the shifted labels

    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)                                 # (batch, seq_len-1)

    return (token_logp * mask).sum(dim=-1)       # sum over response tokens only
```

The shifting (`logits[:, :-1]` against `input_ids[:, 1:]`) is the standard causal-LM alignment: the logits at position $t$ predict the token at position $t+1$. The `response_mask` is what restricts the score to the response — if you forget it and score the prompt too, your log-ratios are dominated by prompt tokens, which are identical for chosen and rejected, so the signal washes out. This is one of the most common implementation bugs.

Now the DPO loss itself, given the four log-probs:

```python
def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta=0.1):
    """All inputs are (batch,) summed sequence log-probs.
    ref_* must be computed with torch.no_grad() (frozen reference).
    """
    chosen_logratio   = policy_chosen_logp   - ref_chosen_logp     # log(pi/pi_ref) for y_w
    rejected_logratio = policy_rejected_logp - ref_rejected_logp   # log(pi/pi_ref) for y_l

    margin = beta * (chosen_logratio - rejected_logratio)          # the boxed loss argument
    loss = -F.logsigmoid(margin).mean()

    # Diagnostics you should always log:
    chosen_reward   = beta * chosen_logratio.detach()
    rejected_reward = beta * rejected_logratio.detach()
    reward_accuracy = (chosen_reward > rejected_reward).float().mean()
    reward_margin   = (chosen_reward - rejected_reward).mean()

    return loss, {
        "reward_accuracy": reward_accuracy.item(),
        "reward_margin":   reward_margin.item(),
        "chosen_reward":   chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
    }
```

Note `F.logsigmoid` rather than `torch.log(torch.sigmoid(...))` — the former is numerically stable for large negative margins, the latter underflows to `-inf`. The diagnostics dict is not optional in real training; `reward_accuracy` climbing past $0.6$ and `reward_margin` growing steadily are your "training is working" signals, and `chosen_reward` going sharply negative is your "beta too high, model degenerating" alarm.

Finally, the training loop tying it together:

```python
from transformers import AutoModelForCausalLM
import copy

policy = AutoModelForCausalLM.from_pretrained("path/to/sft-model")
reference = copy.deepcopy(policy)             # frozen snapshot of the SFT model
reference.eval()
for p in reference.parameters():
    p.requires_grad_(False)

optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-6)

for batch in dataloader:                      # batch has chosen + rejected, pre-tokenized
    # Policy forward (with gradients)
    pol_chosen = sequence_logprobs(policy, batch["chosen_ids"],
                                   batch["chosen_attn"], batch["chosen_resp_mask"])
    pol_rejected = sequence_logprobs(policy, batch["rejected_ids"],
                                     batch["rejected_attn"], batch["rejected_resp_mask"])

    # Reference forward (no gradients, frozen)
    with torch.no_grad():
        ref_chosen = sequence_logprobs(reference, batch["chosen_ids"],
                                       batch["chosen_attn"], batch["chosen_resp_mask"])
        ref_rejected = sequence_logprobs(reference, batch["rejected_ids"],
                                         batch["rejected_attn"], batch["rejected_resp_mask"])

    loss, metrics = dpo_loss(pol_chosen, pol_rejected,
                             ref_chosen, ref_rejected, beta=0.1)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    optimizer.step()
```

Four forward passes per batch (two per model), one backward pass on the policy only. The reference forwards are under `torch.no_grad()` so they cost memory but no gradient storage. That is the entire algorithm. Compare this to a PPO-RLHF loop with rollout collection, reward scoring, advantage estimation (GAE), the clipped surrogate, and value-loss balancing, and you understand why DPO is the method people reach for first.

**Common pitfalls**, each of which I have personally shipped to a failed run:

- **Response truncation.** If chosen and rejected have different lengths and you truncate to a fixed `max_length`, you may cut off the part of the rejected response that makes it bad. The model then sees a truncated rejected that looks fine and learns nothing useful. Set `max_length` generously and log how many examples get truncated.
- **Tokenizer mismatch.** The reference and policy must share a tokenizer. If you load them with different tokenizer configs (different special tokens, different chat templates), the log-probs are computed over different token sequences and the log-ratios are garbage.
- **Forgetting the response mask.** Already mentioned, but it bears repeating because it is the most insidious — training runs, loss decreases, but the model barely changes because the prompt tokens dominate.
- **Reusing the SFT learning rate.** DPO wants ~10x lower. High LR + sharp loss = degeneration in a few hundred steps.
- **Not deep-copying the reference correctly.** If the "frozen" reference shares parameters with the policy (a shallow copy), it updates as the policy trains, the log-ratio drifts toward zero, and DPO does nothing. Verify the reference parameters are actually frozen and independent.

A quick sanity check catches most of these at once. Right after constructing the policy and reference, run a single batch and confirm two things: the chosen and rejected log-ratios are *exactly zero* (because policy equals reference at initialization), and therefore the loss is *exactly* $\log 2 \approx 0.693$. If the initial loss is not $\log 2$, something is wrong before you have trained a single step — usually the reference is not the SFT model, the masking differs between the two models, or the tokenization diverged. I now treat "initial loss equals $\log 2$" as a hard pre-flight check on every DPO run; it has saved me from more than one silently broken pipeline where the loss decreased beautifully while the model learned nothing real. A second cheap check: after a few steps, assert that the reference model's parameters have *not* changed (compare a hash or a single tensor before and after `optimizer.step()`). If they moved, your reference is not frozen, and your log-ratios are meaningless.

## 8. Running DPO with Hugging Face TRL

In production you would not hand-roll the loss — you would use TRL's `DPOTrainer`, which handles tokenization, padding, the reference model, masking, and logging for you. First the dataset. TRL expects a dataset with three columns: `prompt`, `chosen`, and `rejected`.

```python
from datasets import Dataset

# Each example: a prompt, a preferred completion, a rejected completion.
raw = [
    {
        "prompt": "Explain why the sky is blue in one sentence.",
        "chosen": "Sunlight scatters off air molecules, and shorter blue "
                  "wavelengths scatter most, so the sky looks blue.",
        "rejected": "The sky is blue because it is blue, that's just how it is.",
    },
    # ... thousands more pairs ...
]
dataset = Dataset.from_list(raw)
```

The format is the same shape as the public preference datasets — `Anthropic/hh-rlhf`, `argilla/ultrafeedback-binarized`, `HuggingFaceH4/ultrafeedback_binarized` — so in practice you `load_dataset` one of those and map it into `prompt`/`chosen`/`rejected` columns. Then the trainer:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

model_name = "path/to/sft-model"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = DPOConfig(
    output_dir="dpo-aligned",
    beta=0.1,                       # the KL leash from the derivation
    learning_rate=5e-6,             # ~10x lower than SFT
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch 16
    max_length=1024,
    max_prompt_length=512,
    num_train_epochs=1,
    logging_steps=10,
    loss_type="sigmoid",            # standard DPO; "ipo", "kto_pair" select variants
)

trainer = DPOTrainer(
    model=model,                    # policy; reference is auto-created as a frozen copy
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

A few things TRL does for you that matter. It **creates the reference model automatically** as a frozen copy of `model` (unless you pass `ref_model=None` with a PEFT/LoRA setup, in which case it cleverly uses the base model with adapters disabled as the reference — saving the memory of a second full model). It handles the **response masking** and padding correctly. It logs `rewards/accuracies`, `rewards/margins`, `rewards/chosen`, and `rewards/rejected` to your tracker — the exact diagnostics from Section 7. And the `loss_type` flag is your gateway to the variants: `"sigmoid"` is standard DPO, `"ipo"` switches to the IPO loss, and there are entry points for KTO-style losses. To launch:

```bash
accelerate launch --multi_gpu --num_processes 4 train_dpo.py \
    --output_dir dpo-aligned \
    --beta 0.1 \
    --learning_rate 5e-6
```

With LoRA, the memory story gets even better — you train low-rank adapters on the policy and use the adapter-disabled base as the reference, so you are effectively running DPO with the memory of *one* model plus tiny adapters. This is why DPO on a single 24 GB GPU can align a 7B model, something PPO-RLHF with four models cannot dream of.

## 9. DPO variants: IPO, KTO, ORPO, and SLiC-HF

DPO opened a flood of follow-ups, each fixing a specific weakness. Figure 5 lays them out as a comparison matrix; the prose explains the *why* behind each.

![Matrix comparing DPO IPO KTO ORPO and SLiC-HF across whether a reference model is needed the data format the sycophancy risk and the compute cost](/imgs/blogs/dpo-direct-preference-optimization-5.png)

**IPO (Identity Preference Optimization)**, from Azar et al. (2023), addresses a real flaw in DPO: when preferences are nearly deterministic (annotators almost always pick $y_w$ over $y_l$), the Bradley-Terry model wants the implied reward difference to go to infinity, so DPO keeps pushing the log-ratio margin larger and larger with no natural stopping point. This is overfitting toward extremes, and it can manifest as sycophancy — the model learns to crank up whatever superficial feature distinguishes chosen from rejected. IPO replaces the log-sigmoid with a squared loss that targets a *finite* margin:

$$
\mathcal{L}_{\text{IPO}} = \mathbb{E}\left[ \left( \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)} - \frac{1}{2\beta} \right)^2 \right].
$$

It says "make the margin equal to $\frac{1}{2\beta}$, no more" rather than "make the margin as large as possible." This regularizes against the run-to-infinity behavior and reduces overfitting on near-deterministic preferences.

**KTO (Kahneman-Tversky Optimization)**, from Ethayarajh et al. (2024), changes the *data requirement*, which is often the binding constraint in practice. DPO needs *pairs* — for each prompt, a chosen and a rejected response judged against each other. That is expensive to collect. KTO needs only a **scalar label per example**: this single response is "good" (desirable) or "bad" (undesirable), no pairing required. It draws on prospect theory from behavioral economics (hence the names Kahneman and Tversky), modeling the human as having a reference point and being loss-averse, with asymmetric weighting of gains and losses. The upshot: you can train on unpaired thumbs-up / thumbs-down data, which is far more abundant — every "regenerate" click, every thumbs-down in a chat UI, is a KTO signal. When your feedback is naturally unpaired, KTO is the method, not DPO.

**ORPO (Odds Ratio Preference Optimization)**, from Hong et al. (2024), goes after the *reference model* itself. Every method so far needs $\pi_{\text{ref}}$ in memory and in the loss. ORPO eliminates it by combining the SFT loss and a preference term into a single objective, using a log-odds-ratio penalty on the rejected response:

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w) - \lambda \cdot \log \sigma\left( \log \frac{\text{odds}_\theta(y_w)}{\text{odds}_\theta(y_l)} \right), \qquad \text{odds}_\theta(y) = \frac{\pi_\theta(y)}{1 - \pi_\theta(y)}.
$$

Because it folds preference learning into the SFT stage and uses odds ratios rather than ratios against a reference, ORPO needs **no separate reference model and no separate SFT stage** — it does instruction tuning and preference alignment in one pass over the data. That is the lowest-memory, lowest-stage-count option, at the cost of being newer and less battle-tested.

**SLiC-HF (Sequence Likelihood Calibration)**, from Zhao et al. (2023), predates and parallels DPO. It uses a hinge/margin ranking loss on sequence likelihoods plus a regularizer, with the philosophy of *calibrating* the model's likelihoods to rank chosen above rejected by a margin, rather than the Bradley-Terry probabilistic framing. It typically works with ranked lists rather than just pairs.

| Variant | Key idea | Needs ref model | Data format | Best when |
|---|---|---|---|---|
| DPO | Bradley-Terry log-sigmoid | Yes | Pairs | Static paired preferences |
| IPO | Bounded squared margin | Yes | Pairs | Near-deterministic prefs, sycophancy worry |
| KTO | Prospect-theory scalar | Yes | Scalar good/bad | Unpaired thumbs-up/down data |
| ORPO | Odds ratio + SFT in one | No | Pairs (with SFT) | Want single stage, minimal memory |
| SLiC-HF | Margin ranking + calib | Yes | Ranked lists | Calibration framing, ranked data |

There is also **SimPO** (Meng et al., 2024), which drops the reference model differently — it uses the *length-normalized average* log-prob as the implicit reward and adds a target margin, achieving reference-free training while staying close to DPO's formulation. SimPO is partly a response to a quiet bias in vanilla DPO: because the loss sums token log-probs over the whole response, longer responses accumulate more (more negative) log-prob, and the optimizer can be nudged by length rather than quality. Length-normalizing the implicit reward removes that lever, which is why SimPO and length-controlled AlpacaEval often go together in evaluations. The fact that this many viable variants emerged within roughly a year tells you the design space DPO opened was rich.

The practical upshot for an engineer is that most of these are a one-line change in TRL, not a rewrite. The `DPOConfig.loss_type` field is the switch, and the rest of your pipeline — dataset, tokenizer, trainer, logging — stays identical:

```python
from trl import DPOConfig

# Standard DPO: log-sigmoid Bradley-Terry loss.
config = DPOConfig(output_dir="out", beta=0.1, loss_type="sigmoid")

# IPO: bounded squared-margin loss, robust to near-deterministic preferences.
config_ipo = DPOConfig(output_dir="out", beta=0.1, loss_type="ipo")

# For KTO (unpaired scalar labels) you switch to the dedicated KTOTrainer/KTOConfig,
# because the data format is fundamentally different (no pairs).
```

Because the data format for KTO is unpaired, it lives in a separate `KTOTrainer` rather than a `loss_type` flag — a useful reminder that the variant you pick is dictated first by the *shape of your data* (pairs versus scalars versus ranked lists), and only second by the loss formula. Decide what feedback you can actually collect, and the method narrows itself. Figure 6 places these methods on a timeline against the RLHF lineage so you can see how fast the space moved.

![Timeline of preference optimization from RLHF in 2017 and InstructGPT in 2022 through DPO IPO and KTO in 2023 to ORPO and SimPO in 2024](/imgs/blogs/dpo-direct-preference-optimization-6.png)

## 10. DPO through the RL lens, and why the data is everything

Before the decision rules, it is worth stepping back to the spine of this whole series — an agent interacting with an environment, collecting rewards, and updating a policy — and asking exactly where DPO sits in that picture, because the answer sharpens every practical decision that follows.

In standard RL, the agent acts, the environment responds with a reward, and the agent updates. The reward is a function we get to *query* during training: the agent generates an action, and the environment (or a learned reward model) tells it how good that action was. That query is the engine of exploration. The agent can try something it has never tried, get scored, and learn from the score. PPO-RLHF preserves this engine — the reward model is a queryable environment — which is exactly why it can explore.

DPO removes the queryable reward. There is no environment to ask "how good is this novel response?" during training. Instead, DPO has a *fixed dataset of past queries*: a finite set of prompts, each with a chosen and rejected response that some annotator already scored. In RL vocabulary, **DPO is an offline / batch RL method.** It learns a policy purely from a logged dataset of interactions, with no ability to collect new ones. Everything that is true of offline RL — that the policy is only trustworthy where the data has coverage, that extrapolating off-support is dangerous, that the achievable performance is capped by what the dataset reveals — is true of DPO. This is not a loose analogy; the boxed optimal policy $\pi^* \propto \pi_{\text{ref}} \exp(r/\beta)$ is precisely the policy that offline KL-regularized RL targets, and DPO is one way to fit it from logged preferences.

This reframing makes the practical priorities obvious. In online RL you spend your effort on the *algorithm* — the exploration strategy, the advantage estimator, the clip range — because the data is generated on the fly and its quality is partly the algorithm's responsibility. In offline RL, and therefore in DPO, **you spend your effort on the data**, because the data is a fixed ceiling on what you can learn. A perfect DPO run on mediocre preference pairs gives you a model that perfectly reflects mediocre preferences. Three things about the data matter most.

**Coverage.** Recall from Section 5 that DPO's implied reward is unconstrained for responses the reference assigns near-zero probability — there is no data and no sampling to pin it down. So your preference pairs must *cover the kinds of prompts and responses you care about at inference time*. If you only have preference pairs about coding questions and you deploy on creative writing, DPO has shaped nothing in the creative-writing region and the model's behavior there is whatever SFT left it as, at best. Build a preference set whose prompt distribution matches your deployment distribution.

**Margin quality, not just label correctness.** A preference pair where the chosen and rejected responses are nearly identical teaches the model almost nothing — the gradient is tiny and the "lesson" is noise. The most informative pairs are ones where the chosen is *clearly* better than the rejected for a *learnable* reason (the rejected hallucinates a fact, refuses unnecessarily, ignores part of the instruction). When I curate preference data now, I deliberately filter out near-tie pairs and pairs where the distinction is purely stylistic noise, because they dilute the signal. This is the data analog of the gradient weighting in Section 4: the model already learns most from pairs it gets wrong, so give it pairs where being right *means* something.

**Consistency.** If two annotators (or two LLM judges) would flip the chosen and rejected on the same pair, that pair is adding contradictory gradient. DPO has no robust way to average out heavily conflicting preferences within a single pass; it will chase whichever direction the noisy labels happen to push. Inter-annotator agreement on your preference set is a direct predictor of how cleanly DPO will train. Low agreement is a sign to either refine your annotation guidelines or move to a method (like KTO with abundant scalar labels) where the noise averages out over more examples.

#### Worked example: estimating the data ceiling

Suppose you have 20,000 preference pairs, and a quick audit shows that 15% are near-ties (chosen and rejected within a hair of each other) and another 10% have an annotator-agreement rate below 60% (essentially coin-flip labels). That means roughly 25% of your data — 5,000 pairs — is contributing noise or nothing. The *effective* dataset is closer to 15,000 clean, informative pairs. If you run DPO on the full 20,000 and get a disappointing AlpacaEval win rate of, say, 0.71 against SFT, the instinct is to lower the learning rate or sweep beta. But the more likely fix is to clean the data: drop the near-ties, re-judge the low-agreement pairs, and re-run on 15,000 high-margin, high-agreement pairs. In my experience that data cleaning moves the win rate more than any beta sweep — often from the low-0.70s to the mid-0.80s — precisely because, as an offline method, DPO's ceiling *is* the data. The lesson generalizes: when a DPO run underperforms, audit the data before you touch the hyperparameters.

This is the deepest reason the field split the way it did. Chat alignment is a domain where good static preference data is collectable and the target behavior is latent in the SFT model — so DPO, an offline method, thrives. Frontier reasoning is a domain where the best response often does *not* exist in any static dataset and must be discovered by generation-and-scoring — so online methods, which preserve the queryable-reward engine, are required. Both are the same RL principle viewed from two ends: when your logged data already contains the behavior, learn offline; when it does not, you must go online to collect it.

## 11. When DPO works, and when it does not

This is the section I wish someone had handed me before my two wasted weeks. DPO is not a strict upgrade over RLHF; it is a different point on a trade-off curve. Figure 8 is the decision tree I actually use.

![Decision tree for choosing a preference optimization method based on whether data is static or needs exploration whether it is paired or scalar and whether a reference model is available](/imgs/blogs/dpo-direct-preference-optimization-8.png)

**DPO excels when:**

- **You have a static preference dataset.** All the signal is in the pairs, collected up front. No new behavior needs to be discovered. This is the canonical DPO setting and it is extremely common — most "make the assistant nicer / safer / more on-brand" tasks are exactly this.
- **The SFT model already generates good responses.** If your reference policy already produces strong candidates and you just need to nudge its *ranking* of behaviors, DPO's reweighting of the reference is precisely the right operation. You are sculpting an already-good distribution, not building capability from scratch.
- **You are compute-constrained.** Two models, no generation in the loop, runs on hardware where four-model PPO is impossible. The ~10x compute saving is real and often decisive.
- **You want stable, reproducible training.** DPO's single classification loss is dramatically easier to get right than PPO's many-knobbed RL loop. Fewer ways to fail.

**DPO struggles or fails when:**

- **The task requires exploration to find new capabilities.** Hard math reasoning, agentic multi-step tool use, anything where the win comes from the model *generating* a better solution than exists in the dataset. DPO never generates during training, so it cannot reward what it never sees. This is where online methods — PPO-RLHF and especially **GRPO** (the workhorse of recent reasoning models, which samples multiple responses per prompt and ranks them online) — pull ahead decisively.
- **You need online feedback.** If preferences arrive continuously from live users and you want the model to adapt to responses it generates now, that is an online loop by definition. DPO is offline.
- **Your preference data has poor coverage.** Because DPO is bounded by its data distribution and unconstrained off-support, a dataset with narrow coverage produces a model that is well-aligned on the covered region and unpredictable elsewhere. RLHF's online sampling at least probes more of the space.
- **You need process supervision.** For step-by-step reasoning where you want to reward *correct intermediate steps*, a process reward model (PRM) plus PPO gives signal DPO's outcome-level pairs cannot.

A useful one-line heuristic: **DPO sculpts an existing distribution; RLHF and GRPO can grow a new one.** If the behavior you want is already latent in the SFT model and you just need to re-rank it, DPO. If you need to discover behavior that is not yet there, you need online generation. For the deeper offline-RL framing of why learning from a fixed dataset is fundamentally limited, see [Offline RL: learning from fixed datasets](/blog/machine-learning/reinforcement-learning/offline-rl-learning-from-fixed-datasets) — DPO is, in a precise sense, an offline-RL method for language, and inherits offline RL's distribution-coverage limitations.

## 12. Evaluation: does DPO actually match RLHF?

Numbers matter, so let us be concrete about how DPO is evaluated and what the evidence says. The standard alignment metrics:

**Win rate.** Generate responses from the DPO model and a baseline (the SFT model, or a PPO-RLHF model) on a held-out set of prompts, then have a judge — human, or increasingly a strong LLM like GPT-4 — pick the better response. The win rate against SFT tells you whether alignment helped at all; the win rate against PPO-RLHF tells you whether DPO matched the more expensive method.

**MT-Bench.** A multi-turn benchmark of 80 challenging questions across categories (writing, reasoning, math, coding, extraction, roleplay), scored 1–10 by GPT-4. It probes instruction-following quality.

**AlpacaEval.** An automated win-rate benchmark against a reference model (text-davinci-003 or GPT-4 Turbo), with a length-controlled variant to correct for the well-known bias where judges prefer longer answers.

The original DPO paper (Rafailov et al., 2023) reported that on controlled sentiment generation, summarization (Reddit TL;DR), and single-turn dialogue (Anthropic HH), DPO **matched or exceeded PPO-RLHF win rates** while being far simpler and cheaper to train. On the summarization task in particular, DPO's win rate against the reference summaries was competitive with or better than the PPO baseline at the same KL budget. Subsequent work — notably the Zephyr-7B model (Tunstall et al., 2023), which used DPO on UltraFeedback after distilled SFT — showed DPO producing chat models that beat much larger RLHF-tuned models on MT-Bench and AlpacaEval, cementing DPO as a practical default for the open-source community.

#### Worked example: reading a DPO training run

Here is what a healthy DPO run on UltraFeedback (roughly 60k pairs, a 7B SFT model, $\beta=0.1$, LR $5\times 10^{-6}$, one epoch) looks like in the logged metrics, and how to read it:

- **Step 0:** `rewards/accuracies` $= 0.50$, `rewards/margins` $= 0.00$, `loss` $= 0.693$. Exactly as predicted — policy equals reference, every margin is zero, loss is $\log 2$. If you do *not* see these starting values, your reference is not the SFT model or your masking is wrong.
- **Step 200:** accuracy $\approx 0.62$, margin $\approx 0.4$, loss $\approx 0.62$. Learning is happening; the model now ranks chosen above rejected on $62\%$ of pairs.
- **Step 1000:** accuracy $\approx 0.74$, margin $\approx 1.8$, loss $\approx 0.52$. `rewards/chosen` $\approx -0.5$, `rewards/rejected` $\approx -2.3$. Notice **both rewards are negative** — both log-probs fell relative to reference, the rejected one much further. This is the normal pattern from Section 4; it is not a bug.
- **End of epoch:** accuracy $\approx 0.78$, margin $\approx 2.5$. Evaluate: AlpacaEval win rate against the SFT model around $0.85$–$0.90$ on a successful run; MT-Bench up roughly $0.5$–$1.0$ points over SFT.

The **alarm signal** to watch: if `rewards/chosen` plunges past, say, $-5$ or $-10$ and keeps falling while the model's generations start repeating tokens or producing gibberish, you have over-optimized — lower $\beta$, lower the learning rate, or stop earlier. This degeneration is DPO's characteristic failure mode, the analog of reward hacking in PPO, and it traces directly to the off-support freedom we discussed in Section 5.

**The "DPO at scale" question.** Does DPO scale as well as PPO-RLHF to the largest models and hardest tasks? The honest answer in 2024–2025 was: for *chat-style alignment*, DPO scales beautifully and is the community default; but for *frontier reasoning capability* — the kind that made models good at competition math and complex agentic tasks — the leading labs converged on online methods (PPO-RLHF and GRPO with verifiable rewards), because those tasks need the exploration DPO cannot provide. Some research (e.g., work from Cohere and others comparing PPO and DPO carefully) found that PPO can still edge out DPO when the reward signal is rich and online exploration matters, while DPO wins on simplicity and cost when the task is well-covered by static data. Both findings are consistent with the exploration argument in Section 5 — they are the empirical face of the same theoretical distinction.

## 13. Case studies

**Zephyr-7B (Tunstall et al., 2023).** Hugging Face's H4 team distilled SFT data from a larger model, then applied DPO on the UltraFeedback preference dataset. The result, Zephyr-7B-β, scored about 7.34 on MT-Bench, beating Llama-2-Chat-70B (an RLHF-tuned model more than ten times its size) on that benchmark, and was competitive on AlpacaEval. This was a watershed: it showed a small team with modest compute could produce a top-tier chat model using DPO instead of the full RLHF pipeline. The recipe — distilled SFT, then DPO on public preference data — became the de facto open-source alignment playbook.

**The original DPO summarization result (Rafailov et al., 2023).** On the Reddit TL;DR summarization task, DPO trained on the same human preference data that PPO-RLHF used, and achieved comparable or higher win rates (judged by GPT-4) against the reference summaries across a range of sampling temperatures, while being substantially cheaper to train. This was the headline empirical claim that launched the method: same objective, simpler optimization, equal-or-better results on a static preference task.

**InstructGPT (Ouyang et al., 2022) — the PPO baseline.** It is worth naming the system DPO was reacting to. InstructGPT used the full three-stage RLHF pipeline (SFT, reward model, PPO) and demonstrated that a 1.3B RLHF model could be preferred by humans over the 175B GPT-3, showing the power of preference alignment. DPO does not dispute that result; it argues the *RL machinery* was not essential to achieve it for that class of static-preference task.

**The reasoning-model pivot (2024–2025).** When the frontier moved to reasoning — models trained to do long chains of thought on math and code — the leading systems did *not* use DPO. They used online RL with verifiable rewards: GRPO and PPO variants that sample many responses per prompt and reward the correct ones. This is the clearest real-world confirmation of the exploration boundary: for *discovering* reasoning behavior not present in static data, online generation was indispensable, and DPO sat that round out. The contrast — DPO dominating chat alignment, online RL dominating reasoning — is the trade-off of Section 10 written across the actual history of the field.

**Iterative and online DPO — closing the gap.** It would be wrong to leave the impression that DPO is permanently locked out of the exploration regime. A line of work on *iterative* DPO (and its online cousins) tries to recover some of the benefit of online sampling while keeping DPO's simple loss. The recipe: train with DPO on the current preference set, then use the improved policy to *generate fresh response pairs*, have a judge (a reward model or a strong LLM) rank them, append the new pairs to the dataset, and run DPO again. Each round injects new, on-policy responses into the static dataset, partially restoring the exploration that pure offline DPO lacks. This is, in RL terms, a way of turning an offline method into a semi-online one by periodically refreshing the logged data with the agent's own samples. It is more expensive than vanilla DPO (you are generating again) but still simpler than full PPO-RLHF, and it has produced strong results on chat benchmarks. The existence of iterative DPO underscores the framing from Section 10: the binding constraint was never the *loss* but the *data distribution*, and the moment you let the policy generate new data, you start buying back the exploration you gave up. If your vanilla DPO plateaus and you suspect the ceiling is data coverage rather than hyperparameters, an iterative round is often the highest-leverage next move.

## When to use this (and when not to)

Let me be decisive, because the whole point of understanding the derivation is to make this call confidently.

**Reach for DPO first** for any alignment task where you have (or can collect) static preference pairs and the desired behavior is already roughly within the SFT model's reach: style, tone, safety, helpfulness, brand voice, reducing over-refusal. It is cheaper, more stable, runs on commodity hardware, and matches RLHF's quality on these tasks. If you are an individual or small team without a battle-tested PPO-RLHF stack, DPO is almost certainly your starting point — there is a reason it became the open-source default.

**Reach for KTO instead of DPO** when your feedback is naturally *unpaired* — thumbs-up/thumbs-down logs, regenerate clicks, binary quality labels. Do not contort unpaired data into fake pairs; use the method designed for the data you have.

**Reach for ORPO** when you want to fold alignment into a single training pass with no reference model and minimal memory, and you are comfortable with a newer, less-proven method.

**Reach for PPO-RLHF or GRPO** when the task requires *discovering* behavior not present in your static data: hard reasoning, agentic tool use, anything where on-policy generation and a judge that scores novel outputs are doing real work. If you have verifiable rewards (math answers you can check, code that either passes tests or does not), online RL with those rewards will beat DPO, because DPO has no way to explore toward the correct answer.

And the simplest negative rule of all: **if you cannot collect or generate any preference signal, none of these apply** — you are back to supervised fine-tuning, and you should make sure your SFT data is excellent before reaching for any preference method, since every method here is fundamentally a *refinement* of the SFT policy, not a replacement for good SFT.

## Key takeaways

- **The RLHF objective has a closed-form optimum:** $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$. The optimal policy is the reference reweighted by exponentiated reward. This is the seed of everything.
- **Rearranging that gives reward as a log-ratio:** $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$. The policy *is* its own reward model.
- **The intractable partition function cancels** when you substitute into Bradley-Terry, because it depends only on the prompt, shared by both responses. This is the trick that makes DPO possible.
- **The DPO loss is a binary classification loss** — log-sigmoid of a $\beta$-scaled difference of log-ratios — trained with two models and no RL loop, at roughly 10x lower compute than PPO-RLHF.
- **Mechanically, DPO pushes chosen up and rejected down relative to the reference**, weighted by how wrong the model currently is on each pair. Both absolute log-probs often fall; the *margin* is what matters.
- **Beta is the leash.** Too small collapses to the reference; too large overfits the pairs and degenerates. Default $0.1$, sweep $\{0.05, 0.1, 0.3, 0.5\}$, and use a learning rate ~10x below SFT.
- **DPO is bounded by its data distribution** — it never generates during training, so it cannot discover behavior absent from the preference pairs. This is the fundamental limit, not an implementation detail.
- **DPO sculpts an existing distribution; online RL (PPO, GRPO) can grow a new one.** Choose by whether the behavior you want is already latent in the SFT model.
- **Variants fix specific weaknesses:** IPO bounds the margin (anti-sycophancy), KTO uses unpaired scalar labels, ORPO drops the reference model, SimPO length-normalizes. Pick by your data shape and constraints.
- **Always log reward accuracy and chosen/rejected rewards.** Accuracy climbing toward $0.7$–$0.9$ with a growing margin means it is working; chosen reward plunging means you have over-optimized.

## Further reading

- Rafailov, Sharma, Mitchell, Manning, Ermon, Finn — "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023). The original paper; the derivation in Sections 2–3 follows its appendix.
- Ouyang et al. — "Training language models to follow instructions with human feedback" (InstructGPT, 2022). The three-stage RLHF pipeline DPO replaces.
- Ziegler et al. — "Fine-Tuning Language Models from Human Preferences" (2019). The earlier RLHF formulation and the KL-constrained objective.
- Azar et al. — "A General Theoretical Paradigm to Understand Learning from Human Preferences" (IPO, 2023).
- Ethayarajh, Xu, Muennighoff, Jurafsky, Kiela — "KTO: Model Alignment as Prospect Theoretic Optimization" (2024).
- Hong, Lee, Thorne — "ORPO: Monolithic Preference Optimization without Reference Model" (2024).
- Tunstall et al. — "Zephyr: Direct Distillation of LM Alignment" (2023). The DPO case study and open-source recipe.
- Hugging Face TRL documentation — `DPOTrainer`, `DPOConfig`, and the preference-dataset format used in Section 8.
- Within this series: [Proximal Policy Optimization](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) for the RL method DPO reformulates, [Offline RL: learning from fixed datasets](/blog/machine-learning/reinforcement-learning/offline-rl-learning-from-fixed-datasets) for why static-data methods are bounded, [GRPO vs DPO vs PPO: a decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) for the head-to-head, and the unified map `reinforcement-learning-a-unified-map` and capstone `the-reinforcement-learning-playbook` for the broader picture.

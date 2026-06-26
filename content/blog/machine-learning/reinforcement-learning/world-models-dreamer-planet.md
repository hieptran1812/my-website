---
title: "World Models, PlaNet, and Dreamer: Learning to Plan in Latent Space"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "How model-based RL learns a compact latent world from pixels, plans inside it with PlaNet, and trains a policy purely in imagination with Dreamer v1 through v3 — with runnable RSSM code and measured benchmarks."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "model-based-rl",
    "world-models",
    "dreamer",
    "machine-learning",
    "pytorch",
    "planning",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/world-models-dreamer-planet-1.png"
---

A reinforcement learning agent staring at a screen of raw pixels is in a strange position. It sees 64×64×3 numbers — roughly twelve thousand of them — every frame, and somewhere inside that flood is the position of a cart, the angle of a pole, the velocity of a cheetah's leg. Model-free methods like PPO or SAC handle this by brute force: feed the pixels through a convolutional network, predict an action, collect a reward, and let gradient descent slowly carve a policy out of millions of frames. It works, eventually. On a visual control task it can take 100 million environment steps to do what a human masters in minutes. The agent never builds a *model* of how the world changes — it only memorizes which pixel patterns precede high reward.

There is another way, and it is the oldest idea in control theory wearing modern clothes: learn how the world works, then plan inside your head. If the agent could compress those twelve thousand pixels into a few hundred numbers that capture *everything that matters* — and if it could predict how those numbers evolve when it takes an action — then it could rehearse thousands of imagined futures cheaply, pick the best one, and only spend real environment steps on the plan it actually trusts. This is model-based RL, and the figure below shows its beating heart: a world model that takes pixel frames, squeezes them through an encoder into a compact latent, and trains itself to both reconstruct what it saw and predict the reward it got.

![Pipeline showing pixel frames flowing through a CNN encoder into an RSSM latent that feeds a decoder and a reward head, with an ELBO loss updating the model](/imgs/blogs/world-models-dreamer-planet-1.png)

By the end of this post you will understand the three architectures that made latent world models work — Ha & Schmidhuber's original World Models, the Recurrent State Space Model (RSSM) at the core of PlaNet, and the Dreamer family that trains a policy entirely inside imagined rollouts. You will be able to write an RSSM cell in PyTorch, derive the ELBO objective that trains it, and reason about *when* dreaming beats acting. We will keep returning to the same spine that runs through this whole series: an agent interacts with an environment, collects rewards, and updates a policy. World models change only *what the agent practices against* — a learned simulator in its own head instead of the real world. For the broader taxonomy of where model-based RL sits, see the series map at `reinforcement-learning-a-unified-map`.

It is worth naming the intellectual lineage before we start, because it will keep the engineering grounded. The dream of learning a model and planning inside it is older than deep learning by decades. Richard Bellman's dynamic programming assumed you *had* the model. Adaptive control and system identification in the 1960s and 70s tried to *fit* a model online. Sutton's Dyna architecture in 1991 interleaved learning a tabular model with planning against it. What changed in 2018–2023 is not the idea but the *medium*: we learned to fit a model not in a hand-engineered state space but in a learned latent space carved directly out of pixels, and to make that latent both compact enough to plan in and accurate enough to dream in. That is the thread this post follows from Ha & Schmidhuber through Dreamer v3.

## 1. The visual control challenge

Let us be concrete about why pixels break the methods you already know. Tabular RL — value iteration, Q-learning over a table — needs a finite, enumerable state space. A 64×64 RGB image has $256^{64 \times 64 \times 3}$ possible values. You cannot enumerate it, you cannot hash it, you cannot build a table. So we reach for function approximation, and the question becomes: *what should the network represent?*

A model-free agent represents a policy $\pi(a \mid o)$ or a value $Q(o, a)$ directly as a function of the observation $o$. The trouble is that every gradient update has to flow all the way from a scalar reward, back through the value estimate, back through the convolutional stack, to nudge the pixels-to-action mapping. The reward is sparse and noisy, the credit assignment is long, and the network has no incentive to learn *structure* — only to learn the part of the structure that happens to correlate with reward right now.

To feel how thin that signal is, count the bits. A single scalar reward of, say, +1 carries on the order of one bit of information per step. A single 64×64×3 frame carries, before compression, on the order of $64 \times 64 \times 3 \times 8 \approx 98{,}000$ raw bits, and even after the redundancy of natural images is removed, hundreds to thousands of bits of *structured* information about object positions, velocities, and contacts. A model-free learner throws almost all of that away — it only keeps whatever fraction of the frame happens to be linearly predictive of the next reward gradient. A world-model learner keeps it, because *every pixel of every frame is a label* for the reconstruction objective. The supervision is dense, immediate, and present on the very first frame, before a single reward has ever arrived. That density is the entire reason world models can be sample-efficient.

A world model flips the incentive. Instead of asking "what action maximizes reward?", it first asks "what is going on in this scene, and what happens next?" That is a far richer training signal: every pixel of every frame is supervision for the reconstruction objective, and every transition is supervision for the dynamics objective. Reward is just one more thing to predict. The agent ends up with a *compressed, predictive representation* of the world — a latent state $s_t$ of maybe 200–300 dimensions that throws away the texture of the background and keeps the cart position, the pole angle, the velocities. Once you have that, planning becomes tractable because you are searching over 300 numbers instead of 12,000.

The compression is not optional decoration. It is the whole game. A world model is a compression-plus-prediction machine: it learns the smallest description of the world that still lets it predict the future and the reward. Information theory gives us the language for this — the model is trading off reconstruction fidelity against the description length of the latent, which is exactly what the ELBO (evidence lower bound) we derive in section 4 balances. A latent that is too small cannot reconstruct the scene; a latent that is too large memorizes noise and fails to generalize its dynamics. The ELBO is the dial that finds the middle.

There is a second, sharper reason compression matters that is easy to miss: *planning cost is exponential in dimensionality for some methods and at best linear in it for all of them.* Cross-Entropy Method planning, which we will use in PlaNet, samples action sequences and rolls them forward; the number of samples it needs to cover a search space grows with the dimension of that space. If the planner had to reason about all 12,000 pixels — for instance, to predict the next frame pixel-by-pixel inside the search loop — every candidate rollout would cost a full image decode, and a thousand candidates over a twelve-step horizon would be twelve thousand image decodes *per decision*. In a 230-dimensional latent, each rollout step is a couple of small matrix multiplies. The compression does not merely make the model smaller; it makes the *search inside the model* feasible at all. This is the single fact that separates "a nice idea from control theory" from "an agent that learns to walk from pixels in an afternoon."

#### Worked example: how much does compression buy you?

Suppose a CartPole-from-pixels task gives you 96×96×3 = 27,648-dimensional observations. A model-free CNN policy on this task might need on the order of 1–2 million frames to reach an average return near 195/200 on `CartPole-v1`. PlaNet-style latent control on comparable visual tasks reaches strong performance in roughly 100–500 thousand frames — a 2–10× sample-efficiency improvement on visual control benchmarks, because every frame trains the dynamics model densely rather than waiting for sparse reward gradients. The latent is around 230 dimensions (a 30-dim stochastic part plus a 200-dim deterministic part, the standard RSSM sizing). That is a ~120× dimensionality reduction, and the planner runs inside the small space.

Put a price on that. Suppose each real environment step on a physical robot costs you one second of wall-clock and a small amount of wear on the hardware. One million steps is roughly eleven and a half days of continuous operation; one hundred thousand steps is a little over a day. The sample-efficiency multiplier is not an abstract benchmark number — it is the difference between a research project that finishes in a day and one that runs for two weeks and burns through a robot's actuators. The same logic applies to any setting where a "step" is expensive: a slow physics simulator, a market you can only sample in real time, a chemistry experiment, a recommendation system where each "step" is a real user impression you cannot replay. The more a step costs, the more a world model earns its complexity.

## 2. Ha & Schmidhuber World Models (2018)

The paper that named the field, "World Models" by David Ha and Jürgen Schmidhuber (2018), is worth studying precisely because it is *staged* — it splits the problem into three modules trained one after another, which makes each piece legible. Modern Dreamer fuses these stages into one jointly-trained model, but the staged version is the clearest possible introduction to *what each piece is for*, and it is genuinely the right mental scaffold even when you later collapse it.

The three modules are V, M, and C:

- **V (Vision)** is a Variational Autoencoder. It encodes each frame $o_t$ into a small latent vector $z_t$ (32 dimensions in the paper) and can decode $z_t$ back to a reconstructed frame. The VAE is trained on a buffer of frames collected by a random policy. Its job is pure compression: turn the image into a compact code.
- **M (Memory)** is a Mixture Density Network combined with a recurrent network — an MDN-RNN. Given the current code $z_t$, the action $a_t$, and a hidden state $h_t$, it predicts a probability distribution over the *next* code $z_{t+1}$. Because the future is uncertain, it predicts a mixture of Gaussians rather than a single point. This is the dynamics model: it learns how the compressed world evolves.
- **C (Controller)** is a tiny linear policy — just a single layer mapping $[z_t, h_t]$ to an action. It is so small (a few hundred to a few thousand parameters) that it is optimized not by gradient descent but by CMA-ES, an evolutionary strategy that perturbs the weights, evaluates returns, and moves toward better populations.

Why split it this way, and why these three pieces in particular? Each module isolates one of the three jobs any model-based agent must do: *perceive* (V turns pixels into a state), *predict* (M turns a state and action into a next state), and *decide* (C turns a state into an action). By training them separately, Ha & Schmidhuber could collect a buffer of random-policy frames once, train V to compress them, freeze V and train M on the resulting codes, then freeze both and search over C. The staging makes the system embarrassingly easy to debug — if reconstructions are blurry the VAE is the suspect, if dream rollouts drift the MDN-RNN is the suspect, if the dreamed agent transfers poorly the controller or the dream's fidelity is the suspect. You never have to wonder which of three coupled losses is fighting which.

The key insight, the one that echoes through everything that follows, is in the phrase *dream in latent space*. Because M is a generative model of $z_{t+1}$, you can roll it forward without ever touching the real environment or the pixel decoder. Start from a real $z_0$, sample $a_0$ from C, ask M for $z_1$, feed it back, sample $a_1$, and so on. You have a fully imagined trajectory in the 32-dimensional code space. Ha & Schmidhuber famously trained the controller for the `CarRacing-v0` task partly *inside the dream* and transferred it back — the agent learned to drive in its own hallucination.

Let us look at the Vision module in code. The encoder is a stack of strided convolutions that halve the spatial resolution each layer, ending in two linear heads that produce the mean and log-variance of a Gaussian over the 32-dim code. The reparameterization trick — sampling $z = \mu + \sigma \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$ — is what makes the sampling step differentiable, so the reconstruction gradient can flow back through it into the encoder.

```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    """Module V: compress a frame to a 32-dim code."""
    def __init__(self, z_dim=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(2 * 2 * 256, z_dim)
        self.fc_logvar = nn.Linear(2 * 2 * 256, z_dim)

    def encode(self, x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)  # reparameterization
        return z, mu, logvar
```

The MDN-RNN predicts the next code as a mixture, which matters because the world is stochastic: a car at an intersection might go several ways, and a single-Gaussian prediction would average them into a blur that satisfies nobody. The mixture density network outputs, for each of the 32 latent dimensions, a set of $K$ mixture weights $\pi_k$, means $\mu_k$, and standard deviations $\sigma_k$. The predicted next-code distribution is $p(z_{t+1}) = \sum_k \pi_k \,\mathcal{N}(\mu_k, \sigma_k^2)$. At sampling time you first sample which component fires from the categorical $\pi$, then sample from that Gaussian. A useful trick from the paper: scaling the mixture temperature lets you make the dream more or less stochastic — a hotter dream is harder to game, which matters when training a controller inside it.

```python
class MDNRNN(nn.Module):
    """Module M: predict next-code distribution as a mixture of Gaussians."""
    def __init__(self, z_dim=32, a_dim=3, h_dim=256, n_mix=5):
        super().__init__()
        self.rnn = nn.LSTM(z_dim + a_dim, h_dim, batch_first=True)
        self.n_mix, self.z_dim = n_mix, z_dim
        # per mixture component: weight pi, mean mu, log-std
        self.head = nn.Linear(h_dim, n_mix * (1 + 2 * z_dim))

    def forward(self, z, a, hidden=None):
        x = torch.cat([z, a], dim=-1)
        out, hidden = self.rnn(x, hidden)
        params = self.head(out)
        pi, mu, log_sig = torch.split(
            params, [self.n_mix, self.n_mix * self.z_dim,
                     self.n_mix * self.z_dim], dim=-1)
        return pi, mu, log_sig, hidden
```

The controller, by contrast, is almost embarrassingly simple — and that simplicity is deliberate. Because V and M have already done the hard work of turning pixels into a compact, predictive code, the policy that maps that code to an action can be a single linear layer of a few hundred to a few thousand parameters. With so few parameters, you do not need gradients at all; you can search the weight space directly with an evolution strategy. CMA-ES (Covariance Matrix Adaptation Evolution Strategy) maintains a Gaussian distribution over the controller's weight vector, samples a population of candidate controllers, evaluates each by running it (in the real environment or in the dream) and recording the return, then updates the Gaussian's mean and covariance toward the high-return candidates. It needs no reward gradient and no differentiable simulator — it only needs to be able to *score* a controller. That makes it a natural fit for training inside a dream, where the score is just the sum of MDN-RNN-predicted rewards over a rolled-out trajectory.

```python
import numpy as np

def cma_es_controller_search(score_fn, n_params, pop=64, iters=300):
    """Evolve a tiny linear controller; score_fn runs it (in env or dream)."""
    mean = np.zeros(n_params)
    sigma = 0.5
    for _ in range(iters):
        # sample a population of candidate weight vectors
        pop_w = mean + sigma * np.random.randn(pop, n_params)
        scores = np.array([score_fn(w) for w in pop_w])    # returns per candidate
        # keep the top half as "parents"
        elite = pop_w[scores.argsort()[::-1][: pop // 2]]
        mean = elite.mean(axis=0)                           # move toward elites
        sigma = elite.std(axis=0).mean() + 1e-6             # adapt step size
    return mean
```

#### Worked example: a full dream rollout in the 32-dim code space

Concretely, here is what "training in a dream" looks like step by step on `CarRacing`. You start from a real initial code $z_0 \in \mathbb{R}^{32}$ obtained by encoding the first real frame with V. The MDN-RNN hidden state is $h_0$. The controller, a $34 \times 3$ matrix plus bias (input is $[z_t, h_t]$ where $h_t$ is summarized, output is the 3-dim CarRacing action: steer, gas, brake), maps $[z_0, h_0]$ to $a_0 = [0.1, 0.8, 0.0]$ — slight right, mostly gas, no brake. You feed $z_0, a_0$ to M, which outputs five mixture components; you sample one, say component 2 fires with mean $\mu_2$, and draw $z_1 = \mu_2 + \sigma_2 \odot \epsilon$. Now you have $z_1$ without ever rendering a frame or touching the real game. Repeat: $a_1 = C(z_1, h_1)$, $z_2 \sim M(z_1, a_1)$, and so on for, say, a thousand dream steps. Sum the rewards M also predicts along the way, and you have a *score* for this controller, computed entirely inside the model. CMA-ES uses that score to evolve the controller. The agent learned to drive having spent zero real frames during the controller search — the real frames were only spent once, up front, to train V and M.

The architecture is clean and the result was a landmark, but staging V, M, and C separately has a cost. The VAE is trained to reconstruct *everything* in the frame, including details that do not matter for control, and it has no idea that some pixels are more decision-relevant than others. The dynamics model M is stuck with whatever code the VAE produced — if V decided to spend its 32 dimensions on background scenery rather than on the precise position of the car, M can only predict scenery accurately and the controller suffers. There is no feedback path from "the dynamics are hard to predict" back to "encode the frame differently." The next leap — PlaNet — fixes this by training the encoder and the dynamics model *together*, so the latent learns to be exactly as detailed as prediction requires, and no more.

## 3. PlaNet and the Recurrent State Space Model (2019)

PlaNet, from "Learning Latent Dynamics for Planning from Pixels" (Hafner et al., 2019), made two moves that defined the next five years of model-based RL. First, it trained the encoder, the latent dynamics, the decoder, and the reward predictor jointly in one model with one loss. Second, it threw away the parametric policy entirely and *planned online* with the Cross-Entropy Method (CEM) directly in the learned latent space.

The dynamics model is the Recurrent State Space Model, and its design is the single most important idea in this post. Earlier latent dynamics models were either purely deterministic (an RNN that predicts the next state exactly) or purely stochastic (a state-space model where every transition is a random sample). Hafner's team found that *both* fail. A purely deterministic model cannot represent uncertainty — it cannot say "the ball might bounce left or right" and so it collapses multimodal futures into their average, which is a blurry state that corresponds to no real situation. A purely stochastic model has trouble remembering information across many steps, because every step injects noise that washes out long-range memory; information about "the door is locked" that was observed twenty steps ago has to survive twenty rounds of resampling to still be in the state, and it usually does not.

The RSSM keeps both. The state is split into two parts:

- A **deterministic** recurrent state $h_t$, carried forward by a GRU. This is the model's reliable memory — it can store "the pole has been tilting right for three frames" with no noise. Because it is updated by a deterministic recurrence, information placed in $h_t$ persists indefinitely until something explicitly overwrites it. This is the channel that carries long-range memory.
- A **stochastic** latent $z_t$, sampled from a distribution conditioned on $h_t$. This is where uncertainty lives. Because it is resampled every step, it can represent "I genuinely do not know which way the ball will bounce" as a wide or multimodal distribution, and the model is not forced to commit to a single wrong guess.

The figure below shows how they interlock inside a single RSSM cell, and why there are *two* distributions over $z_t$ — a prior used for dreaming and a posterior used for fitting real observations.

![Graph of the RSSM cell where previous state and action feed a GRU producing the deterministic state, which branches into a prior distribution and an observation-conditioned posterior that both produce the stochastic latent](/imgs/blogs/world-models-dreamer-planet-2.png)

The transition equations are:

$$h_t = f_{\text{GRU}}(h_{t-1}, z_{t-1}, a_{t-1})$$
$$z_t \sim p(z_t \mid h_t) \quad \text{(prior — predict without seeing the frame)}$$
$$z_t \sim q(z_t \mid h_t, o_t) \quad \text{(posterior — correct using the actual frame)}$$

Read that carefully, because the *prior* and the *posterior* are the crux of the entire architecture, and the source of most confusion when people first meet it. The prior $p(z_t \mid h_t)$ predicts the next stochastic latent using only the recurrent state — it never sees the real frame. The posterior $q(z_t \mid h_t, o_t)$ also gets to look at the encoded observation $o_t$ and so produces a sharper, corrected estimate. During training we use the posterior (we have the frames, so why not use them). During imagination — when we plan or dream — we have no future frames, so we *must* use the prior. Training pushes the prior to match the posterior, which is precisely what lets the model dream accurately: the prior learns to predict what the posterior *would have said* if it could see the frame.

This is worth a concrete analogy that I find clarifying. The prior is the model with its eyes closed, predicting what it expects to see next from memory and physics alone. The posterior is the same model with its eyes open, snapping its belief to what actually appeared. Every training step asks: "Eyes closed, what did you expect? Now open your eyes — were you right?" The KL term in the loss is the penalty for being surprised, and minimizing surprise on real data is exactly what makes the eyes-closed prediction trustworthy enough to plan with. When the agent later dreams, it keeps its eyes closed for the whole rollout, and it can only afford to do that because training drilled the prior to expect correctly.

There is one more subtlety in the GRU equation that the notation hides: the recurrence takes $z_{t-1}$, the *stochastic sample from the previous step*, as an input. This is what couples the two channels. The deterministic memory does not evolve in a vacuum; it is updated using the sampled latent, so uncertainty that was resolved at step $t-1$ (the ball went left, not right) gets baked into the deterministic memory and carried forward losslessly from then on. The architecture lets uncertainty live in $z$ until it is resolved, then promotes the resolved fact into $h$ where it persists. That hand-off is the quiet engineering that makes long-horizon prediction work.

Once you have this model, PlaNet plans with CEM. CEM is embarrassingly simple and works shockingly well in low dimensions. To choose an action at the current latent state, you:

1. Sample $N$ candidate action sequences (each of horizon $H$) from a Gaussian.
2. Roll each one forward through the RSSM *prior* (no rendering, no real env), summing predicted rewards.
3. Keep the top $K$ ("elites").
4. Refit the Gaussian to the elites' mean and variance.
5. Repeat a few iterations, then execute the first action of the best plan.

```python
import torch

def cem_plan(rssm, reward_model, h, z, act_dim,
             horizon=12, n_candidates=1000, n_elites=100, iters=10):
    """Plan an action sequence in latent space with CEM."""
    mean = torch.zeros(horizon, act_dim)
    std = torch.ones(horizon, act_dim)
    for _ in range(iters):
        # sample candidate action sequences
        acts = mean + std * torch.randn(n_candidates, horizon, act_dim)
        acts = acts.clamp(-1, 1)
        returns = torch.zeros(n_candidates)
        h_b = h.expand(n_candidates, -1).clone()
        z_b = z.expand(n_candidates, -1).clone()
        for t in range(horizon):                 # roll out the PRIOR
            h_b, z_b = rssm.prior_step(h_b, z_b, acts[:, t])
            returns += reward_model(h_b, z_b).squeeze(-1)
        elite_idx = returns.topk(n_elites).indices
        elites = acts[elite_idx]
        mean = elites.mean(dim=0)               # refit the distribution
        std = elites.std(dim=0) + 1e-4
    return mean[0]                              # execute first action only
```

Notice that CEM is *derivative-free*. It never asks the model for a gradient; it only asks "given these actions, what reward do you predict?" and sorts candidates by the answer. This is both a strength and a weakness. The strength is robustness: CEM cannot be fooled by a sharp non-differentiable cliff in the reward landscape, and it naturally handles bounded action spaces by clamping. The weakness is that it scales poorly with action dimension and horizon — the number of candidates needed to find a good plan grows with $H \times \text{act\_dim}$, which is why PlaNet caps the horizon at twelve and re-plans every step rather than committing to a long plan. Re-planning every step is also what gives PlaNet a quiet form of robustness to model error, a point we will return to in the comparison section: even if the model's twelve-step prediction drifts, PlaNet only ever executes the *first* action and then re-grounds itself in a fresh real observation.

There is a detail in the code above worth dwelling on: the loop rolls out the *prior*, not the posterior. During planning the agent is imagining futures it has not observed, so there are no frames to feed a posterior — the prior is the only thing it can use. This is exactly why training spent so much effort pulling the prior toward the posterior. A PlaNet whose prior had not been trained to match its posterior would plan against a hallucination that bears no relation to what its eyes would have seen, and the plans would be worthless.

Because the rollouts happen in a ~230-dimensional latent and never touch the decoder, CEM here is cheap. The contrast with planning in pixel space is stark, and it is the reason latent world models are practical at all.

![Before-after figure contrasting expensive high-dimensional pixel-space planning with cheap low-dimensional latent-space planning that runs about fifty times faster](/imgs/blogs/world-models-dreamer-planet-3.png)

PlaNet also introduced one more training trick that matters for accuracy over long horizons: **latent overshooting**. The naive ELBO only trains one-step predictions — the prior at step $t$ is trained to match the posterior at step $t$, conditioned on the true state at $t-1$. But during planning the model has to predict twelve steps ahead, compounding its own predictions, and a model that is only ever trained on one-step transitions can drift badly when chained. Latent overshooting fixes this by also training *multi-step* predictions: unroll the prior $d$ steps forward from a state and penalize the KL between that $d$-step-ahead prior and the posterior that the real frames would have produced at that future step. In effect it tells the model, "do not just be right one step ahead; be right $d$ steps ahead when forced to rely only on your own predictions." This directly improves the quality of the long rollouts CEM depends on.

```python
def latent_overshooting_kl(rssm, posteriors, actions, max_dist=4):
    """Train d-step-ahead priors to match the real posteriors (d = 1..max_dist)."""
    T = len(posteriors)
    total_kl = 0.0
    for t in range(T - 1):
        h, z = posteriors[t]["h"], posteriors[t]["sample"]
        for d in range(1, min(max_dist, T - t)):
            # roll the prior forward d steps from the posterior at t
            h, z, (pmu, pstd) = rssm.prior_step_with_stats(h, z, actions[t + d - 1])
            qmu, qstd = posteriors[t + d]["mu"], posteriors[t + d]["std"]
            post = torch.distributions.Normal(qmu.detach(), qstd.detach())
            prior = torch.distributions.Normal(pmu, pstd)
            total_kl = total_kl + torch.distributions.kl_divergence(post, prior).sum(-1).mean()
    return total_kl
```

#### Worked example: a CEM planning step on the DeepMind Control Suite

Take the `cheetah-run` task from the DeepMind Control Suite, where PlaNet was evaluated. The action space is 6-dimensional. With horizon $H=12$, $N=1000$ candidates, and 10 CEM iterations, each planning decision rolls out $1000 \times 12 = 12{,}000$ latent transitions per iteration, $120{,}000$ total — all in the 230-dim latent, none rendered. On a single GPU this is milliseconds. PlaNet reached roughly 500–600 average return on `cheetah-run` within about 500k–1M environment steps, comparable to model-free methods like A3C that needed tens of millions of frames on the same suite (Hafner et al., 2019, report PlaNet matching the final performance of top model-free agents with 50× fewer episodes on several tasks). The headline of the paper: similar final scores, dramatically fewer real interactions.

Trace the shapes once so the cost is concrete. The action tensor `acts` has shape $[1000, 12, 6]$. The batched deterministic state `h_b` is $[1000, 200]$ and the stochastic `z_b` is $[1000, 30]$. Each `prior_step` does a couple of linear maps and a GRU cell update on a batch of a thousand — on the order of a few million floating-point operations, trivial for a GPU. Twelve such steps per candidate, ten CEM iterations, and the whole decision is well under a tenth of a second. Now imagine doing this in pixel space: each `prior_step` would have to produce a $1000 \times 3 \times 64 \times 64$ tensor of predicted frames, a transposed-convolution decode for every candidate at every step, which is roughly *two thousand times* more compute per step and would push a single decision into the seconds. That ratio — milliseconds versus seconds — is the "fifty times faster" of the figure made arithmetic, and it is why the latent is not a nicety but a precondition.

## 4. The RSSM objective: ELBO from first principles

Why does the RSSM train with a reconstruction loss *plus* a KL divergence? This is not arbitrary — it falls directly out of variational inference, and deriving it tells you exactly what each term does and where the failure modes hide.

We want a model that explains the sequence of observations $o_{1:T}$ given actions $a_{1:T}$. The model is generative: it posits latent states $s_{1:T}$ (here $s_t = (h_t, z_t)$, but the deterministic $h_t$ is a function of the past so the only random variable is $z_t$) and a likelihood $p(o_t \mid s_t)$. We would like to maximize the log-likelihood $\log p(o_{1:T} \mid a_{1:T})$, but that requires integrating over all latent paths, which is intractable — there are uncountably many continuous latent trajectories, and you cannot sum over them.

So we introduce a variational posterior $q(z_t \mid h_t, o_t)$ — our encoder — and use Jensen's inequality to lower-bound the log-likelihood. The move is the standard variational one: multiply and divide inside the log by $q$, then push the log inside the expectation using Jensen's inequality (the log of an average is at least the average of the logs). For a single step, dropping conditioning on actions for readability:

$$\log p(o_t) = \log \mathbb{E}_{q}\!\left[\frac{p(o_t, z_t)}{q(z_t)}\right] \geq \mathbb{E}_{q}\!\left[\log \frac{p(o_t, z_t)}{q(z_t)}\right] = \mathbb{E}_{q(z_t \mid h_t, o_t)}\big[\log p(o_t \mid h_t, z_t)\big] - D_{\text{KL}}\big(q(z_t \mid h_t, o_t)\,\|\,p(z_t \mid h_t)\big)$$

That right-hand side is the ELBO, the evidence lower bound. The gap between the true log-likelihood and the ELBO is exactly $D_{\text{KL}}(q \,\|\, p_{\text{true posterior}})$, which is why maximizing the ELBO both improves the model *and* tightens the bound — a satisfying property that means we are never optimizing the wrong thing, only a guaranteed lower estimate of the right thing. Read the two terms physically:

- The first term is the **reconstruction** likelihood. It says: a latent sampled from the posterior should let the decoder reproduce the actual frame. This forces the latent to *carry information about the world*. If you drop the cart position from the latent, the decoder cannot redraw the cart, the reconstruction term punishes you, gradient descent puts the cart position back. This is the term that makes the latent *informative*.
- The second term is the **KL divergence** between the posterior (which saw the frame) and the prior (which did not). Minimizing it pulls the prior toward the posterior. This is the term that makes dreaming work: it trains the prior to predict what the posterior would have said, so that at imagination time, rolling the prior forward gives accurate latents. This is the term that makes the latent *predictable*.

Those two pressures — informative and predictable — pull in opposite directions, and that tension is the heart of representation learning. A latent that is maximally informative would copy the frame verbatim (perfect reconstruction, useless dynamics). A latent that is maximally predictable would be a constant (trivial dynamics, useless reconstruction). The ELBO balances them, and the coefficient $\beta$ on the KL term is the dial: large $\beta$ favors predictability and risks throwing away detail, small $\beta$ favors fidelity and risks a latent the prior cannot track. PlaNet and Dreamer sit at $\beta = 1$ with additional tricks (free bits, KL balancing) to keep the balance honest.

Summed over the whole sequence and adding a reward-prediction term (PlaNet and Dreamer also decode reward from the latent), the full training loss is:

$$\mathcal{L} = \sum_{t=1}^{T} \Big[ \underbrace{-\log p(o_t \mid h_t, z_t)}_{\text{image recon}} \;\underbrace{-\log p(r_t \mid h_t, z_t)}_{\text{reward}} \;+\; \beta \underbrace{D_{\text{KL}}\big(q \,\|\, p\big)}_{\text{predictability}} \Big]$$

There is a subtle and important failure mode here that the Dreamer line spent two papers fixing: the KL term can be minimized two ways — by making the posterior *less* informative (collapse) or by making the prior *more* accurate (what we want). Look at $D_{\text{KL}}(q \,\|\, p)$: it goes to zero when $q$ equals $p$, and gradient descent does not care which one moves. If the posterior collapses toward the prior, the latent stops carrying observation detail, reconstruction suffers, and you have effectively trained a model that ignores its eyes. This is **posterior collapse**, the classic VAE pathology, and it is especially dangerous here because the prior is also being trained — the two distributions can quietly meet in the middle at an uninformative point. The fix, **KL balancing**, scales the gradient so that most of the KL pressure pushes the *prior* toward the posterior rather than the reverse. A complementary fix, **free bits**, clips the KL below a small floor so the model never bothers compressing already-predictable latents. We will return to both in section 6.

```python
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

def world_model_loss(obs, recon, reward, reward_pred,
                     post_mu, post_std, prior_mu, prior_std, beta=1.0):
    """ELBO: reconstruction + reward + KL(posterior || prior)."""
    recon_loss = F.mse_loss(recon, obs, reduction='none').sum(dim=[-1, -2, -3]).mean()
    reward_loss = F.mse_loss(reward_pred, reward)
    post = Normal(post_mu, post_std)
    prior = Normal(prior_mu, prior_std)
    kl = kl_divergence(post, prior).sum(dim=-1).mean()
    return recon_loss + reward_loss + beta * kl, recon_loss, kl
```

One practical note on the reconstruction term. We write it as a Gaussian log-likelihood, which reduces to a sum of squared errors over pixels — that is the `F.mse_loss` with `reduction='none'` summed over the channel and spatial dimensions. The *sum* over pixels matters: if you average over pixels instead, the reconstruction term shrinks relative to the KL as image resolution grows, and a 64×64 model and a 96×96 model would need different $\beta$ values. Summing keeps the per-frame reconstruction magnitude roughly resolution-independent, which is part of why Dreamer's recipe transfers across observation sizes. Small choices like this — sum versus mean, where the stop-gradients go — are where a lot of the practical difficulty of training world models actually lives.

## 5. Dreamer v1: learning a policy in imagination (2020)

PlaNet plans online with CEM. That is elegant but expensive at decision time — every single action requires a fresh CEM search of thousands of rollouts. Dreamer ("Dream to Control", Hafner et al., 2020) asked the obvious next question: if we already have a differentiable world model, why not *train a policy* inside it, so that at decision time we just do one forward pass through the actor?

The shift from PlaNet to Dreamer is a shift from *planning* to *learning a policy*, and it mirrors a much older distinction in classical control between model-predictive control (re-solve an optimization at every step, like PlaNet's CEM) and learning an explicit feedback law (compute the policy once, evaluate it cheaply forever after, like Dreamer's actor). Both use the model; they differ in *when* they spend the compute. PlaNet spends it at decision time, every step, online. Dreamer spends it at training time, amortizing thousands of imagined rollouts into a policy network that then answers in a single forward pass. For a robot that must react in real time, that amortization is the difference between a controller that runs at 1 Hz and one that runs at 100 Hz.

Dreamer keeps the RSSM world model unchanged. What it adds is an **actor** $\pi_\phi(a_t \mid s_t)$ and a **critic** $v_\psi(s_t)$, both trained entirely on trajectories *imagined* by the world model. The training loop alternates three jobs, shown in the figure below.

![Stack diagram of the three Dreamer training phases: fitting the world model on real data, imagining latent rollouts, and updating the actor-critic on imagined returns, then acting in the real environment](/imgs/blogs/world-models-dreamer-planet-4.png)

The three jobs are: (1) fit the world model on a batch of real, replayed experience using the ELBO of section 4; (2) imagine a large batch of latent rollouts under the current actor; (3) update the actor and critic on those imagined returns. Then the improved actor goes back to the real environment, collects a little more experience, and the cycle repeats. The crucial accounting fact is that step (2) produces *far* more training data than step (1) consumed — a single batch of replayed sequences, each timestep of which becomes an imagination start state, seeds thousands of imagined transitions. Real data is the bottleneck; imagined data is nearly free; the whole architecture is organized to convert a trickle of the former into a flood of the latter.

The imagination step is the clever part. Starting from every latent state in a batch of real, replayed experience, Dreamer unrolls the world model forward for a fixed **imagination horizon** $H$ (typically 15 steps), using the *actor* to choose actions and the *prior* to predict next latents. Crucially, this entire rollout is differentiable: because the latents are produced via the reparameterization trick and the actor outputs a reparameterized action, gradients of the imagined return flow straight back through the dynamics into the actor's parameters. The actor is trained to maximize the imagined return; the critic is trained to predict it, providing a bootstrap value at the horizon so the agent can reason beyond $H$ steps. Without the critic's bootstrap, a 15-step horizon would make the agent myopic — it would only optimize for what it can see in fifteen steps and ignore everything after. The critic's value at the horizon is a learned summary of "everything good or bad that happens after step 15," and it is what lets a short imagination horizon support long-horizon behavior.

The imagined return uses a $\lambda$-return (a TD($\lambda$) style mixture of $n$-step returns) to balance bias and variance:

$$V^\lambda_t = r_t + \gamma\Big[(1-\lambda)\,v_\psi(s_{t+1}) + \lambda V^\lambda_{t+1}\Big]$$

with $V^\lambda_H = v_\psi(s_H)$ at the horizon. The $\lambda$ knob interpolates between two extremes. At $\lambda = 0$ the return is a one-step TD target $r_t + \gamma v_\psi(s_{t+1})$ — low variance because it leans on the critic, but biased by whatever errors the critic has. At $\lambda = 1$ it is the full Monte Carlo sum of imagined rewards out to the horizon plus the bootstrap — unbiased given the model but high variance because it trusts a long chain of the model's own predictions. Dreamer uses $\lambda = 0.95$, leaning toward the long view while still pulling in the critic to tamp down variance. The actor loss maximizes this return; the critic regresses toward it.

```python
def imagine_rollout(rssm, actor, h, z, horizon=15):
    """Unroll the world model under the actor for `horizon` steps."""
    states_h, states_z, actions, log_probs, entropies = [], [], [], [], []
    for _ in range(horizon):
        s = torch.cat([h, z], dim=-1)
        dist = actor(s)                 # a torch.distributions object
        a = dist.rsample()              # reparameterized -> differentiable
        log_probs.append(dist.log_prob(a).sum(-1))
        entropies.append(dist.entropy().sum(-1))
        h, z = rssm.prior_step(h, z, a) # dream forward via the PRIOR
        states_h.append(h); states_z.append(z); actions.append(a)
    return (torch.stack(states_h), torch.stack(states_z),
            torch.stack(actions), torch.stack(log_probs), torch.stack(entropies))


def lambda_return(rewards, values, gamma=0.99, lam=0.95):
    """Compute TD(lambda) returns over an imagined horizon."""
    H = rewards.shape[0]
    out = torch.zeros_like(rewards)
    next_val = values[-1]
    for t in reversed(range(H)):
        bootstrap = (1 - lam) * values[t] + lam * next_val
        out[t] = rewards[t] + gamma * bootstrap
        next_val = out[t]
    return out
```

The actor update in Dreamer v1, because everything is differentiable, can simply backpropagate the $\lambda$-return through the dynamics — no high-variance policy-gradient estimator needed for continuous control. This is one reason Dreamer is so sample-efficient on the DeepMind Control Suite: the learning signal is a low-variance analytic gradient through a learned simulator, not a Monte Carlo score-function estimate. The contrast is worth making precise. A REINFORCE-style policy gradient estimates $\nabla_\phi \mathbb{E}[R]$ by weighting the log-probability of sampled actions by the return — it works for any reward function, including non-differentiable ones, but its variance is large because it learns only from the *correlation* between random action choices and outcomes. Dreamer's reparameterized gradient instead asks the model directly, "if I nudge this action's parameters, how does the predicted return change?" — a much more informative signal, because it knows *which direction* to move the action, not just whether the sampled action turned out well. The price is that you need a differentiable model of the reward and dynamics, which is exactly what the world model provides. If you want the contrast with score-function policy gradients and why they are high-variance, the policy-gradient post in this series covers that derivation; Dreamer sidesteps it for continuous actions by differentiating through the model.

```python
def actor_critic_losses_v1(returns, values, entropies, ent_coef=3e-4):
    """Dreamer v1 continuous-control actor/critic objectives."""
    # Actor: maximize the differentiable lambda-return (+ entropy for exploration).
    # 'returns' carries gradients back through the dynamics into the actor.
    actor_loss = -(returns + ent_coef * entropies).mean()
    # Critic: regress toward the (detached) returns so it cannot chase a moving target.
    critic_loss = 0.5 * (values - returns.detach()).pow(2).mean()
    return actor_loss, critic_loss
```

#### Worked example: one actor update from imagination

Take a batch of 50 replayed real latent states from a `walker-walk` agent. From each, imagine $H=15$ steps under the current actor: that is $50 \times 15 = 750$ imagined transitions, all in latent space, all differentiable. Suppose the imagined $\lambda$-returns average 6.2 and the critic currently predicts 4.8 — the actor gradient pushes toward the actions that produced the higher imagined return, and the critic regresses its 4.8 toward 6.2 with an MSE loss. No real environment step was taken during this update. Dreamer collects a few hundred real frames, then performs dozens of such imagined updates per real step, which is exactly where the sample efficiency comes from — real data is precious, imagined data is free.

Trace the gradient path once, because it is the most conceptually surprising part of Dreamer. At imagination step 7, the actor outputs an action $a_7$ via `rsample()`, which is $a_7 = \mu_\phi(s_7) + \sigma_\phi(s_7) \odot \epsilon$. That action feeds the RSSM `prior_step`, producing $s_8$, which feeds the reward head and the actor again to produce $a_8$, and so on. When we compute the loss as the negative $\lambda$-return and call `.backward()`, autograd walks this entire chain backward: the gradient of the return with respect to $a_7$ flows through the dynamics (how does changing $a_7$ change $s_8, s_9, \ldots$?) and through every downstream reward, then through the reparameterization into $\mu_\phi$ and $\sigma_\phi$. In one backward pass the actor learns not "action 7 was good" but "moving action 7 in *this* direction would raise the predicted return, accounting for all of its downstream consequences as the model understands them." That is a vastly richer update than a scalar reward could ever provide, and it is only possible because the simulator — the world model — is differentiable.

## 6. Dreamer v2: categorical latents and KL balancing (2021)

Dreamer v2 ("Mastering Atari with Discrete World Models", Hafner et al., 2021) was the version that beat a single GPU's worth of model-free agents on the Atari 200M benchmark, reaching human-level median performance. Two changes carried most of the weight.

First, **categorical latents**. Instead of a Gaussian stochastic latent $z_t$, Dreamer v2 uses 32 categorical variables each with 32 classes — a vector of 32 one-hot draws, which you can think of as 32 little 32-way switches. Why categorical? Atari worlds are full of discrete events: a brick is there or gone, a life is lost or not, an alien is in column 3 or column 4, a key is collected or not. A Gaussian latent has to smear these binary or few-valued facts across a continuous axis, and the model pays a representational tax for pretending a discrete world is continuous — the prior has to place probability mass on the impossible in-between states, and the dynamics have to learn to avoid them. A categorical latent can represent a sharp "either-or" cleanly: the switch is in class 3 or it is not. Empirically, categorical latents gave large gains on Atari and did no harm on continuous control, which is why every Dreamer since has kept them even for robotics.

There is a deeper reason categoricals help that goes beyond matching discrete worlds. A multimodal future — "the ghost will turn left or right" — is naturally expressed by a categorical distribution placing mass on two classes, but it is awkward for a single Gaussian, which can only be unimodal. With 32 categoricals of 32 classes each, the joint latent can express an astronomical number of distinct discrete configurations ($32^{32}$) while keeping each individual factor sharp and multimodal. The continuous Gaussian RSSM of PlaNet had to lean on its mixture-of-Gaussians cousins or accept blur; the categorical RSSM gets multimodality for free.

The catch is that sampling a categorical variable is not differentiable — you cannot backprop through an `argmax`, and the whole Dreamer machinery depends on differentiating through the latent. Dreamer v2 uses **straight-through gradients**: in the forward pass it samples a hard one-hot, but in the backward pass it pretends the operation was the identity on the softmax probabilities, so gradients flow. The trick is the little algebraic identity `one_hot + (probs - probs.detach())`: in the forward pass `probs - probs.detach()` is exactly zero, so the value is the hard `one_hot`; in the backward pass the `one_hot` and the `detach()`'d term contribute no gradient, so the gradient is simply that of `probs`. You get a discrete sample forward and a smooth gradient backward — a small lie that works because the softmax probabilities are a reasonable local linearization of the sampling operation.

```python
import torch
import torch.nn.functional as F

def straight_through_sample(logits):
    """Sample a one-hot categorical, but pass gradients through the softmax."""
    probs = F.softmax(logits, dim=-1)
    # hard sample via Gumbel-style argmax
    index = torch.distributions.Categorical(probs=probs).sample()
    one_hot = F.one_hot(index, probs.shape[-1]).float()
    # straight-through: forward = one_hot, backward = probs
    return one_hot + (probs - probs.detach())
```

Second, **KL balancing**, the fix for the failure mode we flagged in section 4. The KL term $D_{\text{KL}}(q \| p)$ trains both the posterior $q$ and the prior $p$. We want the *prior* to chase the posterior (so dreaming is accurate), not the posterior to collapse to the prior (which would throw away observation detail and starve the reconstruction). The mechanism is a pair of stop-gradients: compute the KL twice, once with the posterior detached (so only the prior moves toward it) and once with the prior detached (so only the posterior moves toward it), and weight the first far more heavily than the second. With $\alpha = 0.8$, eighty percent of the KL pressure makes the prior a better predictor of the posterior, and only twenty percent gently regularizes the posterior toward the prior. The prior does most of the chasing; the posterior keeps most of its freedom to be informative.

```python
def balanced_kl(post_logits, prior_logits, alpha=0.8):
    """KL balancing: most pressure pushes the prior toward the posterior."""
    post = torch.distributions.Categorical(logits=post_logits)
    prior = torch.distributions.Categorical(logits=prior_logits)
    sg = lambda d, l: torch.distributions.Categorical(logits=l.detach())
    # train prior toward (stopped) posterior
    kl_prior = torch.distributions.kl_divergence(sg(post, post_logits), prior)
    # gently regularize posterior toward (stopped) prior
    kl_post = torch.distributions.kl_divergence(post, sg(prior, prior_logits))
    return alpha * kl_prior.sum(-1).mean() + (1 - alpha) * kl_post.sum(-1).mean()
```

Dreamer v2 made a handful of smaller but important changes too. The actor update mixed two gradient estimators rather than relying on one: a *reparameterized* (dynamics-backprop) gradient for the smooth, continuous part of the objective, and a *REINFORCE* (score-function) gradient for the discrete-action case where you cannot differentiate through an `argmax` action choice, blended with a learned baseline from the critic to keep its variance down. The critic was trained with a target network — a slowly-updated copy of the value function — to stop it from chasing its own moving predictions, the same stabilization trick that DQN introduced for model-free value learning. And layer normalization throughout the recurrent model made training markedly more stable. None of these is glamorous, but together they are the difference between a method that works in a paper and one that works on all 55 Atari games from a single configuration.

On the Atari 200M benchmark, Dreamer v2 reached a median human-normalized score above 100% (i.e. above human-level on the median game) training on a single GPU — the first agent to do so with a learned world model, and competitive with the heavily-tuned model-free Rainbow and IQN agents that needed far more compute. The categorical-latent ablation in the paper was striking: removing the categorical latent (reverting to Gaussian) cost a large fraction of the Atari gains, confirming that the discrete representation, not just more parameters, was doing the work. The evolution of the line is worth seeing as one arc.

![Timeline of world-model methods from Ha and Schmidhuber 2018 through PlaNet 2019 and Dreamer v1, v2, and v3, ending at a single recipe spanning all domains in 2023](/imgs/blogs/world-models-dreamer-planet-5.png)

## 7. Dreamer v3: one recipe for every domain (2023)

The deep frustration of RL is that every task needs its own hyperparameters. Change the reward scale, the observation type, or the action space, and you re-tune learning rates, entropy bonuses, and clipping ranges. This is not a minor inconvenience — it is arguably the single biggest reason RL has been hard to deploy outside research. A method that needs a week of per-task tuning by an expert is not a method a practitioner can pick up. Dreamer v3 ("Mastering Diverse Domains through World Models", Hafner et al., 2023) set out to kill this: *one* set of hyperparameters, *one* network size, that works across more than 150 tasks spanning Atari, the DeepMind Control Suite (both proprioceptive and visual), DMLab, Crafter, and Minecraft — without per-task tuning. It was the first agent to collect a diamond in Minecraft from scratch, a famously hard long-horizon exploration problem, with no human data and no curriculum.

The robustness came from a handful of careful normalization tricks, each addressing a specific way the old recipe broke when reward and observation scales changed. The unifying theme is *making the loss landscape invariant to scale*, so that the same learning rate, the same entropy bonus, and the same clipping work whether rewards arrive in the thousands or in fractions.

- **Symlog predictions.** Reward and value magnitudes vary wildly across domains — some games give rewards in the thousands, control tasks in fractions. Dreamer v3 predicts the *symlog* of targets, $\text{symlog}(x) = \text{sign}(x)\ln(1 + |x|)$, which compresses large magnitudes and expands small ones, so the same loss scale works everywhere. It is symmetric (handles negative rewards), it is the identity near zero (so small rewards are not distorted), and it tames large outliers logarithmically. At inference you apply the inverse, $\text{symexp}$.

```python
import torch

def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))
```

- **Free bits.** The KL term is clipped below a floor (typically 1 nat) so the model does not waste capacity over-compressing already-predictable latents — below the floor the KL gradient is switched off, preventing posterior collapse on easy frames. Without free bits, on a frame that is trivially predictable (a static menu screen, a paused game) the model would keep pushing the KL toward zero, collapsing the latent on exactly the frames where it is most tempted to. The floor says "stop optimizing the KL once it is already small enough; spend your gradient budget elsewhere."
- **Return normalization** by a running estimate of the range between the 5th and 95th percentiles of returns. This decouples the entropy bonus and learning rate from the absolute reward scale, so the same entropy coefficient works on a sparse-reward Minecraft and a dense-reward cheetah. The percentile range is used rather than the standard deviation because it is robust to the rare huge return (the diamond) that would otherwise dominate a variance estimate and crush the normalized signal from ordinary steps.
- **Twohot reward and value heads** with symlog: instead of regressing a scalar, the head predicts a distribution over a fixed set of exponentially-spaced bins, placing weight on the two bins adjacent to the true value (hence "twohot"). Predicting a distribution over bins is more robust to outliers than squared-error regression to a scalar, because a single extreme target cannot drag a softmax-over-bins prediction the way it drags an MSE.

```python
def twohot_encode(x, bins):
    """Encode a scalar as weights on its two nearest (symlog-spaced) bins."""
    x = symlog(x).clamp(bins.min(), bins.max())
    # index of the bin just below x
    below = (bins <= x.unsqueeze(-1)).sum(-1).clamp(0, len(bins) - 2)
    above = below + 1
    lo, hi = bins[below], bins[above]
    w_hi = (x - lo) / (hi - lo + 1e-8)        # linear interpolation weight
    target = torch.zeros(*x.shape, len(bins), device=x.device)
    target.scatter_(-1, below.unsqueeze(-1), (1 - w_hi).unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), w_hi.unsqueeze(-1))
    return target                              # a soft, two-nonzero distribution
```

The result is a genuine engineering milestone: scaling the model size *monotonically improves* both final performance and sample efficiency across every domain — a clean scaling property that RL rarely exhibits. Most RL methods get *less* stable as you scale them up, not more; the fact that a bigger Dreamer v3 is uniformly a better Dreamer v3, with no re-tuning, is the property that makes it feel like a foundation rather than a trick. For more on why such scaling behavior is precious and rare in RL, see the scaling-laws series at `/blog/machine-learning/scaling-laws/reinforcement-learning-scaling`. Dreamer v3's reported Atari 100k and 200M results, DMC visual scores, and the Minecraft diamond all came from the same configuration — that uniformity is the headline, more than any single score.

#### Worked example: why symlog rescues a mixed-domain run

Imagine training one agent on both `Pong` (rewards in the range −21 to +21) and a Minecraft task with a one-time reward of +100 for a diamond. Without rescaling, the value head's loss is dominated by the Minecraft +100 target — the MSE gradient from a 100-magnitude error dwarfs the gradient from Pong's ±1 rewards, and the Pong policy starves. With symlog, +100 becomes $\ln(101) \approx 4.6$ and +1 stays near $\ln(2) \approx 0.69$, so the two domains contribute comparable gradients and a single learning rate trains both. This is the mechanism behind "one recipe, all domains" — not a smarter policy, but a loss landscape that no longer changes shape when the reward scale does.

Make the gradient ratio explicit, because it is the whole point. Under plain MSE, the loss is $(\hat v - v)^2$ and its gradient magnitude scales linearly with the error $|\hat v - v|$. A Minecraft target of 100 with an initial prediction near 0 contributes a gradient on the order of 100; a Pong target of 1 contributes a gradient on the order of 1. The Minecraft signal is *a hundred times louder*, so the shared value head spends nearly all its capacity fitting Minecraft and treats Pong as rounding error. Under symlog the targets become 4.6 and 0.69 — a ratio of under 7 instead of 100 — and after the twohot encoding distributes them over bins, the effective per-domain gradient magnitudes are comparable. The agent can now learn both with one optimizer state, one learning rate, one entropy coefficient. Multiply this across 150 domains with reward scales spanning many orders of magnitude and you see why the normalization tricks, unglamorous as they are, are the actual content of the "one recipe" claim.

## 8. The imagination training loop in detail

Let us slow down and trace exactly how Dreamer trains the actor and critic, because the data flow is the thing people get wrong. The figure below shows the loop: a frozen world model generates imagined latent states, the actor branches an action at each one, and the critic scores the trajectory so both networks can be updated.

![Graph of the Dreamer imagination loop where a frozen world model unrolls imagined latents, the actor picks actions at each step, and the critic backs up a lambda-return to update both networks](/imgs/blogs/world-models-dreamer-planet-7.png)

The procedure, step by step:

1. **Sample real start states.** Draw a batch of sequences from the replay buffer, run the world model's *posterior* over them (using the real frames) to get a batch of latent states $s_t = (h_t, z_t)$. These are your imagination start points — and there are many of them, one per timestep per sequence, which is why a small amount of real data seeds a huge amount of imagined data. A batch of 50 sequences of length 50 gives 2,500 distinct start states, and each will spawn a 15-step imagined rollout.
2. **Freeze the world model.** During the actor-critic update the world model parameters are held fixed. Gradients flow *through* the dynamics into the actor, but they do not change the dynamics — that would let the actor cheat by warping the simulator toward optimism. This is a subtle but essential separation of concerns: the model's job is to be accurate (trained in step 0 on real data), and the actor's job is to be good *given an accurate model*. If the same gradient step could change both, the actor would learn to make the model predict high reward rather than to act well, and the whole thing collapses into wishful thinking.
3. **Imagine $H$ steps.** From each start state, unroll the prior under the current actor for $H$ steps, producing imagined latents, actions, and predicted rewards. No real environment, no decoder, no rendering — the decoder is only needed during world-model training to compute the reconstruction loss, and it sits idle during imagination because the actor and critic operate directly on the latent.
4. **Compute $\lambda$-returns.** Decode reward from each imagined latent, query the critic for bootstrap values, and compute the $\lambda$-returns of section 5.
5. **Update the actor** to maximize the imagined return — by differentiating the return through the dynamics for continuous actions, or with a REINFORCE-style estimator plus the critic baseline for discrete actions (Dreamer v2/v3 mix both with entropy regularization).
6. **Update the critic** to regress toward the (stop-gradient) $\lambda$-returns, typically against a slow-moving target network for stability.
7. **Repeat** many times per batch of real data — this is the imagination-to-real ratio, often dozens of imagined updates per real environment step.

The reason no gradient flows through the real environment is fundamental: the real environment is not differentiable. You cannot backpropagate through Atari, through a physics engine, or through a robot's contact dynamics — they are black boxes that take an action and return an observation, with no derivative to offer. The world model exists precisely to provide a *differentiable surrogate* — a smooth, learned stand-in for the un-differentiable world that you can backprop through to your heart's content. This is the deepest payoff of learning a model: it converts a black-box environment into a white-box simulator you own, and once you own a white-box simulator you can do all the gradient-based optimization that model-free RL must approximate with high-variance sampling.

```python
def dreamer_update(world_model, actor, critic, replay,
                   horizon=15, gamma=0.99, lam=0.95, ent_coef=3e-4):
    obs, acts, rews = replay.sample()
    # 1. posterior over real data -> imagination start states
    with torch.no_grad():
        h0, z0 = world_model.observe(obs, acts)
    h0, z0 = h0.detach(), z0.detach()

    # 2-3. imagine under the (frozen) world model
    sh, sz, a, logp, ent = imagine_rollout(world_model.rssm, actor, h0, z0, horizon)
    states = torch.cat([sh, sz], dim=-1)
    rew_pred = world_model.reward_head(states).squeeze(-1)
    values = critic(states).squeeze(-1)

    # 4. lambda-returns
    returns = lambda_return(rew_pred, values, gamma, lam).detach()

    # 5. actor maximizes imagined return (+ entropy bonus)
    actor_loss = -(returns + ent_coef * ent).mean()
    # 6. critic regresses toward returns
    critic_loss = 0.5 * (values - returns).pow(2).mean()
    return actor_loss, critic_loss
```

Note the two stop-gradients in that code, because they encode the separation of concerns. The world model is wrapped in `torch.no_grad()` and the start states are `.detach()`'d, so the actor-critic update never touches the model. And the `returns` fed to the critic loss are `.detach()`'d, so the critic regresses toward a fixed target rather than chasing a value that moves as it learns. Get either stop-gradient wrong and the symptoms are insidious: forget to freeze the model and the actor learns to hallucinate reward; forget to detach the critic target and the value estimate diverges. These two lines are most of the difference between a Dreamer that trains and one that silently produces garbage.

## 9. PlaNet vs Dreamer vs model-free SAC

It helps to put the three philosophies side by side. PlaNet learns a world model and *plans* in it online with CEM — no policy network at all. Dreamer learns a world model and trains a *policy* inside it via imagination. SAC ignores the model entirely and learns a policy and Q-function directly from real transitions. The trade-offs fall out cleanly, and they are best read along two axes: *where you spend compute* (decision time versus training time) and *how much you trust the model*.

![Matrix comparing PlaNet, Dreamer v1, v2, v3, and SAC across latent type, decision rule, sample efficiency, and multi-domain coverage](/imgs/blogs/world-models-dreamer-planet-6.png)

| Method | Latent / state | Decision rule | Sample efficiency | DMC visual score | Atari (200M median) | Multi-domain |
| --- | --- | --- | --- | --- | --- | --- |
| Ha & Schmidhuber (2018) | 32-dim Gaussian VAE code | CMA-ES controller in dream | Moderate | n/a (CarRacing 906 vs human ~844) | n/a | No |
| PlaNet (2019) | 30 stoch + 200 det Gaussian | Online CEM, no policy net | High (~50× fewer eps) | strong (e.g. cheetah-run ~500–600) | n/a | Single suite |
| Dreamer v1 (2020) | RSSM Gaussian | Actor net (1 forward pass) | High | matches/beats model-free | n/a | Continuous control |
| Dreamer v2 (2021) | RSSM categorical (32×32) | Actor net | High | strong | >100% (human-level median) | Atari + control |
| Dreamer v3 (2023) | RSSM categorical + symlog | Actor net | Very high | state-of-the-art | strong, same config | 150+ tasks, no tuning |
| SAC, model-free (pixels) | none (raw conv features) | Q-greedy / stochastic actor | Low–medium | catches up with much more data | (DQN-family territory) | Per-task tuning |

The headline comparison on visual DeepMind Control: Dreamer-family agents typically match or beat model-free SAC's *final* performance while using roughly 5–20× fewer environment steps to get there, because every real frame feeds many imagined updates. SAC catches up given enough data — if you can simulate millions of steps cheaply, the model's sample-efficiency advantage evaporates and SAC's simplicity wins. This is the single most important practical question when choosing between them: *how expensive is a real environment step?* If steps are nearly free, model-free wins on simplicity and robustness. If steps are expensive — a physical robot, a slow simulator, a costly real-world interaction — the model's ability to turn a trickle of real data into a flood of imagined data is decisive.

PlaNet matches Dreamer's data efficiency but pays a much higher *decision-time* cost, because CEM re-searches at every step; Dreamer amortizes that search into a policy network once. On a robot that must close its control loop at 50 or 100 Hz, PlaNet's per-step CEM search may simply be too slow, and Dreamer's single forward pass is the only viable option. On an offline planning problem where you have all the time in the world per decision, PlaNet's re-planning can be an advantage.

A second, subtler axis is *robustness to model error*. PlaNet's online CEM can partly correct for a flawed model by re-planning every step against fresh observations — even if the twelve-step prediction is wrong, only the first action is executed before the agent re-grounds itself. Dreamer's policy, by contrast, is trained against the model's predictions, so a systematically biased world model can teach the actor to exploit hallucinated reward — the actor, being an optimizer, will find and lean on exactly the states where the model wrongly predicts high reward. This is the model-based RL analog of reward hacking: the actor is doing its job perfectly, optimizing the objective it was given, but the objective (the model's reward prediction) is wrong in a way the actor has learned to exploit. It is why KL balancing, free bits, return normalization, and a well-fit reward head matter so much: they keep the dreamed world honest, narrowing the gap between what the model predicts and what the real environment delivers so there are fewer blind spots for the actor to find.

There is a useful way to summarize the whole landscape. PlaNet and Dreamer are the same *world model* with two different *decision strategies* bolted on; Ha & Schmidhuber is the staged ancestor of that world model; SAC is the alternative that refuses to build a model at all. The progression from Ha & Schmidhuber to Dreamer v3 is the story of (1) fusing the staged modules into one jointly-trained model, (2) replacing online planning with an imagination-trained policy, and (3) normalizing everything so a single recipe spans every domain. Each step traded a little conceptual simplicity for a lot of generality and robustness.

## 10. A full RSSM sketch in PyTorch

Here is a compact but complete RSSM cell with encoder, prior, posterior, GRU recurrence, decoder, and reward head — enough to train the world-model loss from section 4. It is written for clarity over speed.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    def __init__(self, obs_feat=1024, act_dim=6, h_dim=200, z_dim=30):
        super().__init__()
        self.h_dim, self.z_dim = h_dim, z_dim
        # deterministic recurrence: (z, a) -> GRU -> h
        self.gru_input = nn.Linear(z_dim + act_dim, h_dim)
        self.gru = nn.GRUCell(h_dim, h_dim)
        # prior  p(z | h): predict latent stats from h alone
        self.prior_net = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ELU(),
                                       nn.Linear(h_dim, 2 * z_dim))
        # posterior q(z | h, obs): also see encoded observation
        self.post_net = nn.Sequential(nn.Linear(h_dim + obs_feat, h_dim), nn.ELU(),
                                      nn.Linear(h_dim, 2 * z_dim))

    def _stats(self, net, x):
        mu, logstd = net(x).chunk(2, dim=-1)
        std = F.softplus(logstd) + 0.1
        return mu, std

    def _sample(self, mu, std):
        return mu + std * torch.randn_like(std)  # reparameterized

    def prior_step(self, h, z, a):
        """One imagination step: no observation available."""
        x = F.elu(self.gru_input(torch.cat([z, a], dim=-1)))
        h = self.gru(x, h)
        mu, std = self._stats(self.prior_net, h)
        return h, self._sample(mu, std)

    def obs_step(self, h, z, a, obs_feat):
        """One training step: correct the latent with the real observation."""
        x = F.elu(self.gru_input(torch.cat([z, a], dim=-1)))
        h = self.gru(x, h)
        pri_mu, pri_std = self._stats(self.prior_net, h)
        post_mu, post_std = self._stats(self.post_net, torch.cat([h, obs_feat], dim=-1))
        z = self._sample(post_mu, post_std)
        return h, z, (post_mu, post_std), (pri_mu, pri_std)


class ConvEncoder(nn.Module):
    def __init__(self, out=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2), nn.ELU(),
            nn.Conv2d(32, 64, 4, 2), nn.ELU(),
            nn.Conv2d(64, 128, 4, 2), nn.ELU(),
            nn.Conv2d(128, 256, 4, 2), nn.ELU(),
            nn.Flatten(), nn.Linear(2 * 2 * 256, out))

    def forward(self, x):
        return self.net(x)


class ConvDecoder(nn.Module):
    def __init__(self, feat=230):  # h_dim + z_dim
        super().__init__()
        self.fc = nn.Linear(feat, 1024)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, 2), nn.ELU(),
            nn.ConvTranspose2d(128, 64, 5, 2), nn.ELU(),
            nn.ConvTranspose2d(64, 32, 6, 2), nn.ELU(),
            nn.ConvTranspose2d(32, 3, 6, 2))

    def forward(self, h, z):
        x = self.fc(torch.cat([h, z], dim=-1)).view(-1, 1024, 1, 1)
        return self.net(x)
```

A few design choices in that cell are worth flagging because they are easy to get wrong and hard to debug when you do. The `softplus(logstd) + 0.1` in `_stats` adds a minimum standard deviation of 0.1 to every latent — this is a guard against the variance collapsing to zero, which would make the KL explode and the sampling degenerate; a hard floor on $\sigma$ is one of the unglamorous tricks that keeps RSSM training from diverging. The prior and posterior share the *same* deterministic state $h$ (computed once per step from the GRU) and differ only in whether they additionally see the encoded observation — this is what makes the prior and posterior comparable, and it is the structural reason the KL between them is meaningful. And `prior_step` and `obs_step` deliberately compute the GRU update identically; the only difference is that `obs_step` also runs the posterior network. That symmetry means a model trained with `obs_step` can be rolled out with `prior_step` and the deterministic backbone behaves exactly the same — the dream and the waking experience share a spine.

And the training step that ties it together — observe a real sequence with the posterior, decode, and minimize the ELBO:

```python
def train_world_model(rssm, encoder, decoder, reward_head,
                      obs_seq, act_seq, rew_seq, opt, beta=1.0, free_nats=1.0):
    T, B = obs_seq.shape[:2]
    h = torch.zeros(B, rssm.h_dim)
    z = torch.zeros(B, rssm.z_dim)
    recon_loss = reward_loss = kl_loss = 0.0
    for t in range(T):
        feat = encoder(obs_seq[t])
        a = act_seq[t] if t > 0 else torch.zeros(B, act_seq.shape[-1])
        h, z, (qmu, qstd), (pmu, pstd) = rssm.obs_step(h, z, a, feat)
        recon = decoder(h, z)
        recon_loss = recon_loss + F.mse_loss(recon, obs_seq[t])
        reward_loss = reward_loss + F.mse_loss(
            reward_head(torch.cat([h, z], -1)).squeeze(-1), rew_seq[t])
        post = torch.distributions.Normal(qmu, qstd)
        prior = torch.distributions.Normal(pmu, pstd)
        kl = torch.distributions.kl_divergence(post, prior).sum(-1).mean()
        kl_loss = kl_loss + torch.clamp(kl, min=free_nats)  # free bits
    loss = (recon_loss + reward_loss + beta * kl_loss) / T
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(rssm.parameters(), 100.0)
    opt.step()
    return loss.item()
```

Read that loop as the data-flow diagram from section 8 made concrete. We carry $h$ and $z$ across timesteps so the recurrence has memory; we encode each real frame and run `obs_step` (the posterior path) because we are training on real data; we decode and accumulate the three loss terms; and we apply free bits via `torch.clamp(kl, min=free_nats)` so the KL gradient switches off once the latent is predictable enough. The gradient clip at norm 100 is the last line of defense against the occasional exploding update that recurrent models are prone to. Note the asymmetry with the imagination loop: here we use `obs_step` and the decoder, because the model is learning from reality; in `imagine_rollout` we use `prior_step` and no decoder, because the actor is learning from the model. Same RSSM, two modes, distinguished entirely by whether the eyes are open.

There is also a deliberate omission in the loss worth discussing: notice that the actor and critic are trained *without* any reconstruction signal of their own. The decoder shapes the latent during world-model training, but the actor only ever sees the latent and the predicted reward — it never decodes a frame. This is a hint of a debate we take up next: how much should the latent be shaped by reconstruction at all?

If you would rather not hand-roll this, the reference implementations are public: the original PlaNet and Dreamer code from Danijar Hafner, and community PyTorch ports. But writing the RSSM cell once, by hand, is the fastest way to internalize the prior/posterior split that everything else rests on.

## 11. Reconstruction or not? Dreamer vs value-equivalent models

A natural and important question is whether the latent *needs* a reconstruction loss at all. In Dreamer, the latent is shaped by three forces: it must reconstruct the frame, predict the reward, and be predictable by the prior. The reconstruction loss is by far the heaviest — a 64×64×3 image has thousands of pixels, so the reconstruction term dominates the gradient and the latent ends up organized largely around *what the scene looks like*. That is mostly good: it forces the latent to be rich. But it has a cost. The latent spends capacity on decision-irrelevant detail — the texture of the grass, the color of the sky — because the reconstruction loss does not know that those pixels do not matter for control. On a task where the visually-dominant part of the scene is irrelevant and the decision-relevant part is a few small pixels, reconstruction-driven latents can underweight exactly what matters.

This is the opening for a fundamentally different design philosophy, embodied by MuZero (Schrittwieser et al., 2020): the **value-equivalence** principle. MuZero learns a latent model too, but it does *not* reconstruct observations at all. Its latent is shaped purely by predicting the quantities that matter for decisions — the reward, the value, and the policy. The latent does not need to contain enough information to redraw the frame; it only needs to contain enough to predict what a good agent would do and how much reward will follow. Two observations that look completely different but lead to identical optimal behavior can collapse to the same latent, and two observations that look identical but matter differently for value can separate — neither of which a reconstruction-driven latent would do.

The trade-off between the two philosophies is real and not yet fully settled:

- **Reconstruction (Dreamer).** Dense supervision from every pixel makes the model easy to train and sample-efficient, and the latent is interpretable — you can decode it and *see* what the model believes. The risk is wasted capacity on irrelevant detail and vulnerability when the decision-relevant signal is visually tiny.
- **Value-equivalence (MuZero).** The latent is laser-focused on decisions and cannot be distracted by irrelevant pixels. The risk is sparser supervision (only reward and value, no per-pixel signal), which makes the model harder to train and more prone to collapse, and an opaque latent you cannot visualize. MuZero leans on heavy Monte Carlo Tree Search at decision time to compensate, which is its own cost.

In practice Dreamer's reconstruction-driven approach has proven remarkably robust and general — the dense pixel supervision is a big part of why it trains stably across 150+ domains — while MuZero's value-equivalent approach has dominated where lookahead search pays off, like board games and discrete planning. A useful way to hold them in mind: Dreamer learns *what the world is and looks like*, then acts; MuZero learns *only what it needs to decide and search*. For a fuller treatment of MuZero and search-based latent models, see the MuZero post in this series; here the point is just that "shape the latent with reconstruction" is a *choice*, not a law, and the alternative is live and competitive.

```python
def value_equivalent_loss(reward_pred, reward, value_pred, value_target,
                          policy_logits, policy_target):
    """A MuZero-style loss with NO reconstruction term — only decision quantities."""
    reward_loss = F.mse_loss(reward_pred, reward)
    value_loss = F.mse_loss(value_pred, value_target)
    policy_loss = F.cross_entropy(policy_logits, policy_target)
    # notice: nothing here asks the latent to redraw the observation
    return reward_loss + value_loss + policy_loss
```

## 12. Case studies

**PlaNet on the DeepMind Control Suite (Hafner et al., 2019).** Evaluated on six visual continuous-control tasks — `cartpole-swingup`, `reacher-easy`, `cheetah-run`, `finger-spin`, `cup-catch`, `walker-walk` — all from 64×64 pixels. PlaNet reached performance comparable to top model-free agents (A3C, D4PG) using on the order of 50× fewer episodes; on several tasks it matched D4PG's final score within about 500k–2M environment steps where D4PG needed tens of millions. The single world model was shared across all six tasks in a multi-task variant, demonstrating that one RSSM could capture six distinct dynamics — a foreshadowing of Dreamer v3's much broader generality. The ablations in the paper are the most instructive part: removing the deterministic path hurt long-horizon prediction, removing the stochastic path hurt multimodal tasks, and removing latent overshooting degraded the long rollouts that CEM depends on — each ablation confirming exactly the design rationale of section 3.

**Dreamer v2 on Atari 200M (Hafner et al., 2021).** The first world-model agent to exceed human-level *median* performance on the 55-game Atari benchmark at the 200M-frame budget, trained on a single GPU in about 10 days per game. It was competitive with the strongest single-GPU model-free baselines of the time (Rainbow, IQN) while using a learned discrete world model rather than direct value learning. The categorical-latent ablation in the paper showed the discrete latent was responsible for a large fraction of the Atari gains — replacing it with a Gaussian latent of matched capacity dropped the median score substantially, which is the cleanest available evidence that the *representation*, not merely the parameter count, was doing the work. This was also the result that shifted the field's default mental model from "world models are a nice idea for continuous control" to "world models can win the canonical discrete-action benchmark."

**Dreamer v3 on Minecraft and beyond (Hafner et al., 2023).** Dreamer v3 collected a diamond in Minecraft from scratch — no human demonstrations, no curriculum, no per-task tuning — a long-horizon, sparse-reward exploration task that had resisted prior end-to-end methods. Collecting a diamond requires a long chain of subgoals (gather wood, craft tools, mine stone, find iron, smelt it, descend to diamond depth, mine diamond) with reward only at the very end, which is precisely the credit-assignment nightmare that model-free methods struggle with and that a long-horizon imagination-trained agent is well suited to. The same fixed configuration achieved strong results across 150+ tasks (Atari, DMC proprio and visual, DMLab, Crafter, Atari 100k), and the paper documented monotonic improvement with model size, an unusually clean scaling result for RL. This generality — one recipe, no knobs — is the practical reason Dreamer v3 became a default starting point for visual and embodied RL, including as a base for robotics work where real-world sample efficiency is paramount.

**Ha & Schmidhuber's CarRacing and VizDoom (2018).** The original World Models reached a score of around 906 on `CarRacing-v0` — clearing the task's ~900 "solved" threshold and well above a typical human score in the mid-800s — by training the tiny CMA-ES controller on top of the frozen VAE-plus-MDN-RNN representation. On the `DoomTakeCover` task it went further and trained the controller substantially *inside the MDN-RNN's dream*, then transferred it to the real environment, where it performed well — the proof of concept that a controller could be learned entirely in a hallucinated world and still work in reality. A delightful detail from that work: the agent could exploit the dream if the dream was too deterministic, "cheating" by finding adversarial action sequences that the imperfect MDN-RNN rewarded but that would fail in reality — an early, vivid demonstration of the model-exploitation failure mode that section 9 discussed and that Dreamer's later honesty tricks were designed to suppress.

## When to use this (and when not to)

Model-based RL with a learned latent world model is not a universal upgrade. Reach for it when these conditions hold, as the decision tree below summarizes.

![Decision tree for choosing a model-based approach based on whether observations are pixels, low-dimensional state, a discrete game tree, or a cheap simulator](/imgs/blogs/world-models-dreamer-planet-8.png)

- **Pixel or high-dimensional observations + sample efficiency matters → Dreamer v3.** This is the sweet spot. If real interactions are expensive (a physical robot, a slow simulator, a costly market) and you observe images, a learned world model amortizes data brutally well. Dreamer v3's no-tuning property makes it the sane default — you are unlikely to do better with less effort, and the absence of per-task tuning means you can actually ship it.
- **Low-dimensional, continuous state with smooth dynamics → PETS or PILCO.** If your state is already a clean 10-dim vector (joint angles, velocities), you do not need a pixel encoder, and an RSSM's reconstruction machinery is overkill. Gaussian-process or ensemble dynamics models (PILCO, PETS) plan beautifully here and are simpler than an RSSM — they model the dynamics directly in the state space you already have, and ensembles give you a principled handle on model uncertainty for safe planning.
- **Discrete actions with a known or learnable game tree → MuZero.** For board games and discrete planning where lookahead search pays off, MuZero learns a latent model (value-equivalent, no reconstruction, as section 11 discussed) and runs Monte Carlo Tree Search inside it. It is a world model too, but tuned for search rather than imagination-trained policies. If your problem rewards deep lookahead more than fast reactive control, search beats imagination.
- **You can simulate cheaply and abundantly → model-free PPO or SAC.** If environment steps are nearly free (a fast simulator, a cheap game), the world model's sample-efficiency advantage shrinks to nothing and you pay only its complexity and its model-bias risk. Model-free methods are simpler, more robust to the absence of a good model, and harder to fool. For the model-free options, see the PPO and SAC posts in this series. The honest rule of thumb: if you can comfortably afford tens of millions of real steps, start model-free and only reach for a world model if sample efficiency becomes the bottleneck.

Honest caveats. World models add real failure modes the model-free world does not have. The actor can learn to exploit model errors (hallucinated reward) — the Ha & Schmidhuber dream-cheating story is the canonical example, and it recurs whenever a policy is trained against an imperfect model. The reconstruction loss can spend latent capacity on decision-irrelevant background detail, starving the parts of the latent that actually drive value. A poorly-fit reward head silently corrupts every imagined return, because every $\lambda$-return is built from its predictions. And the whole system has more moving parts to monitor: reconstruction quality, prior-posterior agreement, reward-prediction accuracy, and actor-critic stability all have to be healthy at once. These are debuggable — KL balancing, free bits, return normalization, reconstruction monitoring, and held-out prediction checks all exist precisely to catch them — but they are extra machinery. If you cannot afford to debug a generative model, a model-free baseline is the responsible first move, and you can always graduate to a world model once you have a working baseline to compare against.

## Key takeaways

- A world model is a compression-plus-prediction machine: it learns the smallest latent that reconstructs observations and predicts reward, turning intractable pixel-space planning into cheap latent-space search. The compression is not decoration — it is what makes planning inside the model feasible at all.
- The RSSM splits state into a deterministic recurrent part (reliable memory) and a stochastic part (uncertainty). You need both — pure-deterministic loses multimodality, pure-stochastic loses long memory. The deterministic backbone promotes resolved uncertainty into persistent memory.
- The prior $p(z \mid h)$ dreams without seeing frames; the posterior $q(z \mid h, o)$ corrects using the real frame. Training pulls the prior toward the posterior, which is exactly what makes imagination accurate. Eyes closed, predict; eyes open, correct.
- The training loss is the ELBO: reconstruction + reward likelihood + KL divergence. KL balancing ensures the prior chases the posterior rather than the posterior collapsing, and free bits stop the model over-compressing already-predictable latents.
- PlaNet plans online with CEM (no policy net) and re-plans every step, which buys robustness to model error at a high decision-time cost. Dreamer trains an actor-critic *inside imagination*, amortizing the search into one cheap forward pass at decision time.
- The world model is a differentiable surrogate for a non-differentiable environment — that is the deep payoff, letting analytic gradients flow through the dynamics into the actor for low-variance continuous control, instead of the high-variance score-function estimates model-free methods need.
- Dreamer v2 added categorical latents (with straight-through gradients) for discrete worlds; Dreamer v3 added symlog, free bits, and return normalization to get one recipe that works across 150+ domains without tuning, with the rare property that bigger is uniformly better.
- Reconstruction is a choice, not a law: Dreamer shapes its latent with pixel reconstruction (dense, robust, but capacity spent on irrelevant detail), while MuZero's value-equivalent latent is shaped only by decision quantities (focused, but sparser supervision and an opaque latent).
- Use model-based RL when real data is expensive and observations are high-dimensional; use model-free when you can simulate cheaply, and always watch for the actor exploiting model errors — the keep-the-dream-honest tricks exist for exactly this reason.

## Further reading

- Ha, D. & Schmidhuber, J. (2018). "World Models." *NeurIPS*. The paper that named the field; V/M/C decomposition and training inside the dream.
- Hafner, D. et al. (2019). "Learning Latent Dynamics for Planning from Pixels" (PlaNet). *ICML*. The RSSM and latent CEM planning, with the latent-overshooting and prior/posterior ablations.
- Hafner, D. et al. (2020). "Dream to Control: Learning Behaviors by Latent Imagination" (Dreamer v1). *ICLR*. Actor-critic trained in imagination via differentiable rollouts and $\lambda$-returns.
- Hafner, D. et al. (2021). "Mastering Atari with Discrete World Models" (Dreamer v2). *ICLR*. Categorical latents, straight-through gradients, KL balancing.
- Hafner, D. et al. (2023). "Mastering Diverse Domains through World Models" (Dreamer v3). *arXiv:2301.04104*. Symlog, free bits, return normalization, twohot heads; the Minecraft diamond and the model-size scaling result.
- Schrittwieser, J. et al. (2020). "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero). *Nature*. The value-equivalent, search-based cousin of latent world models.
- Chua, K. et al. (2018). "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models" (PETS). *NeurIPS*. The ensemble-dynamics alternative for low-dimensional state.
- Sutton, R. & Barto, A. "Reinforcement Learning: An Introduction" (2nd ed.). Chapter 8 on planning and learning with models, including the Dyna architecture that is the conceptual ancestor of all of this.
- Within this series: the taxonomy at `reinforcement-learning-a-unified-map`, the model-free counterparts in the PPO and SAC posts, the search-based MuZero post, and the capstone `the-reinforcement-learning-playbook`.

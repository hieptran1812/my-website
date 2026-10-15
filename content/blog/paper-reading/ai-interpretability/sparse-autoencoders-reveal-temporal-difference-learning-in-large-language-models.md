---
title: "Sparse Autoencoders Reveal Temporal Difference Learning in Large Language Models"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How Demircan et al. used sparse autoencoders to find a latent in Llama 3 70B that tracks the temporal difference error of Q-learning, then showed that switching it off breaks in-context reinforcement learning."
category: "paper-reading"
subcategory: "AI Interpretability"
tags:
  - paper-reading
  - sparse-autoencoders
  - temporal-difference-learning
  - in-context-learning
  - reinforcement-learning
  - successor-representation
  - mechanistic-interpretability
  - llama-3
  - iclr-2025-poster
author: "Hiep Tran"
featured: true
readTime: 19
image: "/imgs/blogs/sparse-autoencoders-reveal-temporal-difference-learning-in-large-language-models-fig1.webp"
excerpt: "Llama 3 70B learns simple reinforcement learning tasks from the prompt alone. Sparse autoencoders trained on its residual stream contain a latent that moves like a TD error, and zeroing that single latent makes the model a worse learner."
paper:
  title: "Sparse Autoencoders Reveal Temporal Difference Learning in Large Language Models"
  authors: "Can Demircan, Tankred Saanum, Akshay K. Jagadish, Marcel Binz, Eric Schulz"
  venue: "ICLR 2025; arXiv 2024"
  url: "https://arxiv.org/abs/2410.01280"
---

> [!tldr]
> - **Claim.** Llama 3 70B, trained only to predict the next token, solves small reinforcement learning (RL) problems inside its prompt, and its residual stream carries a signal that behaves like the **temporal difference (TD) error** of Q-learning.
> - **Mechanism.** Train a sparse autoencoder (SAE) per transformer block, correlate every latent with the TD error of a Q-learning model fitted to the same episodes, pick the best match, then zero or clamp it and splice the edited reconstruction back into the model.
> - **Why it matters.** It is a correlational finding with a causal check attached: zeroing one latent in block 34 lowers the returns Llama earns and makes its choices less Q-learning-like, while zeroing a non-TD latent does not.
> - **Most surprising.** The same story holds with **no rewards at all**. On a random walk over a graph, SAE latents match the successor representation learned by TD (max r = 0.62) better than a counted transition matrix (r = 0.49), and the TD error beats a surprise signal 0.60 to 0.18.
> - **Where it is thin.** One model, task-specific SAEs, moderate correlations (r = 0.58 for the headline latent). It shows *TD-like* signals, not a TD circuit.

![Figure 1 from Demircan et al. (2024): the three tasks (A Two-Step Task, B Grid World, C community graph) and the pipeline D: prompt Llama, record activations, train an SAE, compare latents to TD error, then lesion](/imgs/blogs/sparse-autoencoders-reveal-temporal-difference-learning-in-large-language-models-fig1.webp)

The diagram above is the whole paper at a glance: panels A to C are the three tasks, and panel D is the recipe run on each. The rest of this post unpacks it.

## The problem

Large language models learn from examples in the prompt with no gradient step. That in-context learning extends to RL: earlier work showed LLMs improving their choices across episodes of a small game written out in text.

What nobody had shown is *how*. Is the model copying the last good action, counting outcomes, or running something close to a textbook RL algorithm? The authors bet on the textbook. Its central learning signal is the **TD error**, which also has a famous biological echo in dopamine neurons.

Finding a scalar signal in an 8,192 dimensional residual stream is hard: thousands of comparisons cost statistical power, and concepts are smeared across neurons (polysemanticity). So the paper first decomposes the stream into sparse latents with an SAE, then searches that cleaner space.

## Contributions

1. **Behavior:** Llama 3 70B learns a two-step decision task in context and is best fit by **Q-learning**; Llama 3 8B stays near chance.
2. **Method:** per-block SAEs on the residual stream, correlated with variables from a fitted RL model, then targeted lesions through the SAE.
3. **Evidence** for TD-like latents in a two-step task, a 5×5 grid world, and a reward-free graph task.
4. **Causality:** zeroing or clamping TD latents degrades behavior and downstream representations; control lesions do little.
5. **Without reward:** Llama's graph representations resemble the **successor representation**, itself learned by TD.

## Method

Five pieces carry the paper: the TD error, the SAE, the correlation search, the lesion, and the successor representation.

### Technique 1: the TD error and Q-learning

**The problem it solves.** In a multi-step task the reward arrives at the end. A choice made at step one gets no direct feedback, so the learner needs a way to push credit backward in time.

**Intuition.** A forecaster updates Monday's forecast using Tuesday's better-informed forecast instead of waiting for the weather. In RL the forecast is a value estimate, and the gap between what you expected and what you now believe is the TD error.

**Mechanism.** The agent keeps a value estimate per state and action. After acting, it builds a target: the reward plus the discounted value of the best action from the new state. The TD error is target minus old estimate, and the estimate moves a small step toward the target.

**The math.** The paper works in a Markov decision process with states $s_t$, actions $a_t$, a reward function $r(s_t, a_t)$ and a discount factor $\gamma$. The Q-value $Q(s_t, a_t)$ is the expected discounted future reward for taking action $a_t$ in state $s_t$. The TD error (the paper's Equation 1) is

$$
\delta_{TD} = r(s_t, a_t) + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)
$$

The first two terms form the target: immediate reward plus the discounted best value one step ahead. The last term is the current belief. A positive $\delta_{TD}$ means "better than I expected", a negative one means "worse". The update (Equation 2) nudges the estimate by a learning rate $\alpha$:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \, \delta_{TD}
$$

Setting $\gamma = 0$ gives a **myopic** learner that only tracks immediate reward. In animal learning that special case is the Rescorla-Wagner model, and the paper uses it as the main rival hypothesis.

![Anatomy of one TD update: reward and discounted next value form the target, the current estimate is subtracted to give delta, and alpha times delta updates Q](/imgs/blogs/sparse-autoencoders-reveal-temporal-difference-learning-in-large-language-models-1.webp)

**Worked example.** Suppose $Q(s,a) = 2$, the reward is 0, $\gamma = 0.99$ and the best next value is 5. The target is ${0 + 0.99 \times 5 = 4.95}$, so $\delta_{TD} = 4.95 - 2 = 2.95$. With $\alpha = 0.1$ the new estimate is ${2 + 0.1 \times 2.95 = 2.295}$.

The first-stage action gained value even though it earned nothing directly. A myopic learner would compute $\delta = 0 - 2 = -2$ and *lower* it.

That is why the two-step task separates the models: the first choice always pays 0, so only a bootstrapping learner can learn which first move is good.

**A detail the paper adds.** Because Llama sees many past episodes at once, the model's TD error is an **average over the last $k$ transitions**, like a small replay batch: $k = 4$ for the two-step task (chosen for fit), full history elsewhere. Also $\gamma = 0.99$, and $\alpha = 0.1$ (two-step, grid) or 0.05 (graph).

**When it fails.** The TD error correlates with reward and with value by construction, so a latent that tracks "reward just arrived" will also correlate with $\delta_{TD}$. The myopic comparison addresses this, but it remains the main confound.

### Technique 2: the sparse autoencoder

**The problem it solves.** Each token in Llama 3 70B is an 8,192 dimensional residual vector, and a concept such as "value of this state" may be spread across many coordinates. Correlating each raw dimension against a TD error is both underpowered and likely to miss a distributed signal.

**Intuition.** An SAE is a translator with a huge vocabulary and a strict word budget. It must rewrite each residual vector as a sum of a few dictionary entries. Because few entries may be used at once, each tends to specialize on one recognizable thing.

**Mechanism.** A dense encoder maps the residual vector to a wider latent vector, a ReLU zeroes negative entries, and a dense decoder maps back. Training balances reconstruction against small latent activations.

**The math.** Let $h \in \mathbb{R}^{8192}$ be one residual stream vector, $f$ the encoder (a single dense layer), $\sigma$ the ReLU and $g$ the decoder (another dense layer). The reconstruction is $\tilde{h} = g(\sigma(f(h)))$. The paper's loss (Equation 3) is

$$
\mathcal{L}_{SAE} = \lVert h - \tilde{h} \rVert_2^2 + \beta \, \lVert \sigma(f(h)) \rVert_1^2
$$

The first term is squared reconstruction error. The second is the sparsity penalty: the L1 norm of the latent activations, which the paper writes squared, weighted by $\beta$. Every SAE in the paper uses $\beta = 10^{-5}$.

The latent space is doubled to 16,384 for the two-step and grid tasks and kept at 8,192 for the graph task. Inputs are rescaled so the average vector norm is $\sqrt{d}$, and the scaling is undone before an edited reconstruction goes back into Llama.

**Worked example.** One SAE per block (80 blocks) per experiment, trained with Adam, batch 256, learning rate ${10^{-4}}$. The two-step SAEs see 18,000 vectors for 30 epochs: about 70 batches per epoch, roughly 2,100 steps per block. The codes use only "a few dozen active features" out of 16,384 (Appendix A.2.1).

```python
import torch, torch.nn as nn

class SAE(nn.Module):
    def __init__(self, d=8192, m=16384):
        super().__init__()
        self.f = nn.Linear(d, m)   # encoder
        self.g = nn.Linear(m, d)   # decoder

    def forward(self, h):
        z = torch.relu(self.f(h))  # sparse latents, shape (batch, m)
        return self.g(z), z

def sae_loss(h, h_hat, z, beta=1e-5):
    recon = ((h - h_hat) ** 2).sum(-1)
    sparsity = z.abs().sum(-1) ** 2      # squared L1, as in Eq. 3
    return (recon + beta * sparsity).mean()

sae = SAE()
opt = torch.optim.Adam(sae.parameters(), lr=1e-4)
h = torch.randn(256, 8192)               # one batch of rescaled residuals
h_hat, z = sae(h)
loss = sae_loss(h, h_hat, z)
loss.backward(); opt.step()
```

**When it fails.** These SAEs are **task-specific**: trained on thousands of vectors from one prompt family, good at isolating that task's variables, useless as general dictionaries. The authors list this as their first limitation. For the wider debate on how to judge SAEs, see the earlier post on [scaling and evaluating sparse autoencoders](/blog/paper-reading/ai-interpretability/scaling-and-evaluating-sparse-autoencoders).

### Technique 3: finding the TD latent by correlation

**The problem it solves.** An SAE gives thousands of latents per block. We need a principled way to say which one, if any, behaves like a TD error.

**Intuition.** A police lineup. The suspect description is the TD error time series from the fitted Q-learning model; each latent offers its own series over the same steps, and the closest match is picked.

**Mechanism.** Record the residual stream at the token before each action ("pick" in the two-step prompt, "go" in the grid world, the comma in the graph task). Encode with that block's SAE, correlate every latent with Q-values, myopic values, TD errors and myopic errors, and report the maximum per block.

**The math.** For latent $j$ in block $\ell$, let $z^{(\ell)}_{j,t}$ be its activation at step $t$ and $\delta_t$ the model's TD error at the same step. The score is the Pearson correlation

$$
\rho^{(\ell)}_j = \operatorname{corr}_t\left(z^{(\ell)}_{j,t}, \; \delta_t\right), \qquad j^\star = \arg\max_j \rho^{(\ell)}_j
$$

and the per-block curve in Figure 2C and 2D plots $\max_j \rho^{(\ell)}_j$ against $\ell$. Only latents with non-zero variance enter the comparison.

**Worked example.** The authors added two task variants where the reward function or the transitions change halfway. Each change triggers large TD errors, exactly the variance a correlation can lock onto. The winner sits in **block 34 with r = 0.58**.

![Figure 3 from Demircan et al. (2024): the TD error and the best matching block 34 SAE latent across six runs of the standard, reward-disruption and transition-disruption Two-Step variants; dashed lines mark the change](/imgs/blogs/sparse-autoencoders-reveal-temporal-difference-learning-in-large-language-models-fig3.webp)

Figure 3 is the most persuasive image in the paper. In the bottom two rows the dashed line marks the rule change, and the TD trace and the latent jump together. The transition-disruption row matters most: a surprising transition with no reward still produces a TD error, which a pure reward detector would miss.

**When it fails.** A maximum over 16,384 latents and 80 blocks inflates correlations; some latent will match any wiggly series. The paper guards against this by comparing candidate models rather than with a shuffled null, so read the absolute r values as upper bounds.

### Technique 4: lesioning through the SAE

**The problem it solves.** Correlation does not show use. A latent could mirror the TD error while the model ignores it. The test is to remove it and see whether behavior breaks.

**Intuition.** A neuroscientist's lesion study, with a scalpel that cuts one concept instead of one brain region: change one number in the SAE code, decode, hand the vector back.

![The paper's pipeline as a loop: record activations, train SAEs, correlate latents with TD error, pick the TD latent, lesion it, splice the reconstruction back, and measure behavior and downstream correlations, with a least-correlated latent as control](/imgs/blogs/sparse-autoencoders-reveal-temporal-difference-learning-in-large-language-models-2.webp)

**Mechanism.** Before each choice, encode the block 34 residual, set the TD latent to 0 (deactivation) or a fixed value (clamp), decode, undo the scaling, and splice the result back so Llama finishes its forward pass. The control does the same to the latent with the *lowest* TD correlation.

**The math.** With $z = \sigma(f(h))$ and $j^\star$ the TD latent, the lesioned residual is

$$
\tilde{h}_{\text{lesion}} = g\left(z \odot (\mathbf{1} - e_{j^\star}) + c \, e_{j^\star}\right)
$$

where $e_{j^\star}$ is the one-hot vector for latent $j^\star$, $\odot$ is elementwise product, and $c$ is the value written into that slot: $c = 0$ for deactivation and $c = -10$ for the clamp, a choice the authors take from Templeton (2024).

**Worked example.** In the two-step task the readouts are return per episode and the log-likelihood of Llama's choices under the fitted Q-learning model. Figure 2E and 2F show both moving the wrong way after the TD lesion and staying put after the control lesion. Figure 2G and 2H then show the clamp at $-10$ lowering the *downstream* correlation of later blocks with Q-values and TD errors.

![Figure 2 from Demircan et al. (2024): (A) returns for Llama 70B and 8B, (B) model fits, (C, D) max correlation with values and errors across blocks, (E to H) the same measures after the TD lesion](/imgs/blogs/sparse-autoencoders-reveal-temporal-difference-learning-in-large-language-models-fig2.webp)

**When it fails.** Every lesion passes the residual through the SAE, so reconstruction error rides along with the edit; the control lesion, which pays the same cost, partly answers this. And a ReLU latent is never negative in training, so a clamp at $-10$ is a strong push off distribution: it shows the direction matters, not how it behaves naturally.

### Technique 5: the successor representation

**The problem it solves.** TD learning is not only about reward. It can learn any quantity that is "what happens next, plus a discounted version of what happens after". The graph experiment uses this to ask whether Llama runs TD even when there is no reward to predict.

**Intuition.** The successor representation (SR) is a map of where you tend to end up. Standing on one node, it lists how often, with discounting, you will visit each other node in the future. Nodes in the same community light up together, which is why the SR exposes community structure that a one-step transition table hides.

**Mechanism.** After each move from $s_t$ to $s_{t+1}$, row $s_t$ of the SR is pushed toward "one visit to $s_t$, plus the discounted SR row of where I arrived". The rival counts transitions, and its error signal is surprise under the counted table.

**The math.** The SR is a state by state matrix $M$ with

$$
M(s, s') = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \, \mathbb{I}(s_t = s') \,\middle|\, s_0 = s\right]
$$

where $\mathbb{I}$ is the indicator function. Its TD error and update (Equations 4 and 5) mirror the Q-learning ones, with a vector in place of a scalar:

$$
\delta_{TD} = \mathbf{1}_{s_t} + \gamma M(s_{t+1}, :) - M(s_t, :), \qquad M(s_t, :) \leftarrow M(s_t, :) + \alpha \, \delta_{TD}
$$

Here $\mathbf{1}_{s_t}$ is a one-hot vector at index $s_t$. The counting rival estimates $T(s, s') = \mathbb{E}[\mathbb{I}(s_{t+1} = s') \mid s_t = s]$ and scores each step with $-\log T(s_t, s')$ for the observed next state and $-\log(1 - T(s_t, s'))$ for the others.

**Worked example.** Take a three-node chain A, B, C and $\gamma = 0.99$ with $\alpha = 0.05$ (the graph-task rate), starting from $M = 0$. The walk goes A to B. The TD error for row A is $\mathbf{1}_A + 0.99 \cdot M(B,:) - M(A,:) = [1, 0, 0]$, so row A becomes $[0.05, 0, 0]$.

Suppose the walk has since passed through B to C a few times, so row B now reads $[0, 0.05, 0.04]$. The next A to B move gives row A a TD error of $\mathbf{1}_A + 0.99 \cdot [0, 0.05, 0.04] - [0.05, 0, 0] = [0.95, 0.0495, 0.0396]$, and row A becomes about $[0.0975, 0.0025, 0.0020]$.

Row A now carries mass on C even though A has never been followed directly by C. That transfer is the "global" information a counted table never sees. (This micro-example is mine, not the paper's.)

**When it fails.** On a small graph the SR and counted transitions are both dominated by who is next to whom, so the representation gap is modest (0.62 versus 0.49). The error signals separate far more cleanly, which is the stronger evidence.

## Experiments and results

The two-step task asks Llama to act, the grid world asks it to predict another agent, and the graph task removes reward entirely.

![Three tasks, one recipe: what Llama does in each task, the learning models compared, the best TD latent, and the lesion effect](/imgs/blogs/sparse-autoencoders-reveal-temporal-difference-learning-in-large-language-models-3.webp)

### Two-Step Task

Llama picks Left or Right to reach an Apple or Orange state (always 0 reward), then picks again to reach a rewarded terminal state. Everything is deterministic: 100 experiments of 30 episodes, temperature 1 sampling, first 7 episodes random.

| Measure | Value | Source |
| --- | --- | --- |
| Llama 70B returns | well above chance, often optimal | Fig. 2A |
| Llama 8B returns | roughly chance | Fig. 2A |
| NLL, Q-learning model | 2729 | Sec. 3 |
| NLL, myopic model ($\gamma = 0$) | 2864 | Sec. 3 |
| NLL, repetition model | 5745 (worse than chance) | Sec. 3 |
| Best TD latent | block 34, r = 0.58 | Sec. 3 |
| TD lesion | lower returns, less Q-learning-like | Fig. 2E, 2F |
| Control lesion | no drop in Q-learning likelihood | Sec. 3 |

The 135-point NLL gap between Q-learning and myopic is modest, but it is the evidence that Llama propagates value back from the second stage.

### Grid World

Fifty Q-learning agents walk from a fixed start to a fixed goal on a 5×5 grid with four actions: +1 at the goal, $-1$ per other step, $\epsilon$-greedy exploration decayed over 15 episodes. Llama reads the transcript and is scored on predicting the agent's next action over the first 320 observations.

![Figure 4 from Demircan et al. (2024): (A) action prediction with true versus random rewards, (B, C) max correlations with values and errors, (D) action prediction after TD versus control lesions, (E, F) downstream correlations after lesioning](/imgs/blogs/sparse-autoencoders-reveal-temporal-difference-learning-in-large-language-models-fig4.webp)

The key control keeps the **actions identical but randomizes rewards**. Llama predicts the same sequence better with true rewards (Figure 4A), so it uses reward rather than copying actions. Removing rewards or swapping their signs (Appendix A.4) points the same way.

SAE latents correlate most strongly with the Q-values and TD errors of the generating agent, beating both the myopic model and a Q-learner trained on the random rewards. For the lesion the authors went bigger: **4 latents across 4 blocks, all from block 40 onward, each with r ≥ 0.75**. Zeroing them degrades action prediction, while zeroing the least-correlated latents in the same blocks barely changes it (Figure 4D).

### Community graph

The graph has three communities of five nodes, four neighbors per node, and two bottleneck nodes per community that link out. Llama sees a 401-step random walk over single-token object names and predicts the next node, across 20 runs.

| Comparison | Llama latent match (max r) | Rival | Rival max r |
| --- | --- | --- | --- |
| Representation | SR learned by TD: **0.62** | counted transition matrix | 0.49 |
| Error signal | TD error for the SR: **0.60** | surprise under the counted matrix | 0.18 |
| Next-node accuracy | 24.3% average | ceiling of the task | 25% |
| After block 64 TD lesion | 18.9% | before lesion | 24.3% |

![Figure 5 from Demircan et al. (2024): (A) MDS projections of node representations across blocks, (B) next-state accuracy, (C) bottleneck decoding by block, (D, E) correlations with the SR and TD versus the transition matrix and surprise](/imgs/blogs/sparse-autoencoders-reveal-temporal-difference-learning-in-large-language-models-fig5.webp)

Accuracy reaches the ceiling after about 100 transitions. Multidimensional scaling shows community structure sharpening across blocks toward the SR's 2D layout (Figure 5A). A linear SVM, trained leave-one-run-out so node names cannot leak, decodes bottleneck nodes from around block 20, though next-token prediction never requires that.

The lesion in block 64 drops accuracy from 24.3% to 18.9% and lowers later correlations with the SR. Oddly, the TD correlation itself recovers quickly after the lesion (Figure 6C), while the community structure in block 65 is visibly distorted.

![Figure 6 from Demircan et al. (2024): (A) next-state accuracy after the block 64 lesion, (B) SR correlations, (C) TD correlations, (D, E) block 65 representations before and after lesioning](/imgs/blogs/sparse-autoencoders-reveal-temporal-difference-learning-in-large-language-models-fig6.webp)

The SAE result is replicated with centered kernel alignment (CKA), a representational similarity measure, in Appendix A.5:

$$
\text{CKA}(X, Y) = \frac{\lVert Y^\top X \rVert_F^2}{\lVert X^\top X \rVert_F \, \lVert Y^\top Y \rVert_F}
$$

where $X$ and $Y$ are row-stacked representations of Llama and a learning model and $\lVert \cdot \rVert_F$ is the Frobenius norm. CKA also favors the SR, without picking one maximal latent.

### What is load-bearing and might not transfer

- **Scale.** Only the 70B model learns the task, and everything mechanistic is one model at one size.
- **The TD model's window.** $k = 4$ was chosen for fit; a different $k$ smooths the trace (Appendix Figure 7) and shifts the r values.
- **Tiny SAE training sets.** 18,000 vectors with engineered rule changes is far from a general SAE trained on billions of tokens.

## Critique

**What is strong.** Each task pairs a "find" step with a "break" step and a control lesion, the right shape for an interpretability claim. The model comparisons isolate one thing each: bootstrapping (myopic versus Q-learning), reward use (random rewards), and TD without reward (SR versus counting). The graph result is the most interesting, because it suggests TD-like computation serves next-token prediction itself.

**What is weak.**

- **Max-over-many correlations.** With up to 16,384 latents and 80 blocks, the best r is biased upward. A shuffled-time null, or a held-out set of runs for selecting the latent, would make the r values interpretable on their own.
- **Moderate effect sizes.** r = 0.58 means the headline latent explains about a third of the TD error's variance. "Resembles" is the honest word, and the paper mostly uses it.
- **Fitted hyperparameters on both sides.** The TD window $k$ and the learning rates were tuned for fit, so the Q-learning model had flexibility the rival models may not have had.
- **Lesion baseline ambiguity.** It is not clear whether the no-lesion condition runs through the SAE, and the $-10$ clamp is off-distribution for a ReLU latent.
- **Small inconsistencies.** The graph task text says 20 runs; the Figure 5 caption says accuracy is averaged over 100 runs. This is the v1 preprint.

**Missing ablations.** A general-purpose SAE, a second model family, a dose-response curve of lesion strength against returns, and a lesion of a *reward* latent that is not a TD latent.

**What would change my mind.** If a shuffled-time null or held-out selection pushed the best TD correlations down to the level of the myopic model's, or if a reward-only latent lesion damaged behavior just as much as the TD latent lesion, I would read these latents as reward or surprise detectors rather than TD errors.

## What I would build with this

These are my extrapolations, not claims from the paper.

1. **A cross-task TD probe.** Train one SAE on a mixed corpus of RL prompts (bandits, grid worlds, card games) and test whether the *same* latent tracks TD error across all of them. That is the generality test the paper calls for.
2. **Circuit tracing upstream of block 34**, to test the authors' question of whether induction heads produce these signals.
3. **A steering knob for agents.** If the TD latent is causal, scaling it up after good outcomes might speed in-context learning in an LLM agent. Background on the update rule: [TD(0) and SARSA](/blog/machine-learning/reinforcement-learning/temporal-difference-learning-td0-and-sarsa) and [Q-learning as off-policy TD control](/blog/machine-learning/reinforcement-learning/q-learning-off-policy-td-control).

## References

1. Demircan, C., Saanum, T., Jagadish, A. K., Binz, M., and Schulz, E. *Sparse Autoencoders Reveal Temporal Difference Learning in Large Language Models.* ICLR 2025. [arXiv:2410.01280](https://arxiv.org/abs/2410.01280)
2. Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning*.
3. Dayan, P. (1993). Improving generalization for temporal difference learning: the successor representation. *Neural Computation*.
4. Templeton, A. et al. (2024). Scaling Monosemanticity: extracting interpretable features from Claude 3 Sonnet. Transformer Circuits Thread.
5. Related posts on this site: [Temporal difference learning: TD(0) and SARSA](/blog/machine-learning/reinforcement-learning/temporal-difference-learning-td0-and-sarsa), [Q-learning: off-policy TD control](/blog/machine-learning/reinforcement-learning/q-learning-off-policy-td-control), [Scaling and evaluating sparse autoencoders](/blog/paper-reading/ai-interpretability/scaling-and-evaluating-sparse-autoencoders).

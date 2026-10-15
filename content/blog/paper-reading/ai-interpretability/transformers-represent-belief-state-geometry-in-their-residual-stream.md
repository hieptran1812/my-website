---
title: "Transformers Represent Belief State Geometry in their Residual Stream"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How a small transformer trained only on next-token prediction ends up carrying the full Bayesian belief state of the hidden process, fractal and all, in a linear subspace of its residual stream."
tags: ["paper-reading", "interpretability", "belief-state", "residual-stream", "computational-mechanics", "mixed-state-presentation", "hidden-markov-model", "linear-probing", "transformer", "neurips-2024-poster"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 18
paper:
  title: "Transformers Represent Belief State Geometry in their Residual Stream"
  authors: "Adam S. Shai, Sarah E. Marzen, Lucas Teixeira, Alexander Gietelink Oldenziel, Paul M. Riechers"
  venue: "NeurIPS 2024"
  url: "https://arxiv.org/abs/2405.15943v3"
---

> [!tldr]
> - **Claim.** A transformer trained on next-token prediction learns more than the hidden structure of its data. It learns how an ideal observer *updates its beliefs* about that hidden structure, and those beliefs sit in a linear subspace of the residual stream.
> - **Why it matters.** The theory (computational mechanics) predicts the exact geometry of the activations *before* you look at the network. Interpretability rarely gets to make a prediction that sharp and then check it.
> - **Most surprising result.** For the Mess3 process the predicted belief geometry is a fractal, and a plain linear regression from the 64-dimensional residual stream recovers that fractal. A shuffled-label control collapses it to a single point.
> - **Beyond next token.** On the RRXOR process, distances in the residual stream follow true belief distances (R² 0.95) far better than next-token distances (R² 0.31), and the geometry only shows up when all layers are concatenated.
> - **Where it fails.** Toy processes with 3 to 5 hidden states and 2 to 3 tokens. No causal test that the network actually *uses* these beliefs. Real language would need a simplex far larger than any residual stream.

![Figure 1 from Shai et al. (2024): a hidden generator predicts a fractal belief geometry (top), and the same fractal emerges in a linear projection of the residual stream during training (bottom)](/imgs/blogs/transformers-represent-belief-state-geometry-in-their-residual-stream-fig1.webp)

Figure 1 is the whole paper in one picture: a small hidden Markov model predicts a fractal of points in a triangle (top), and a linear projection of a trained transformer's residual stream grows the same fractal during training (bottom). The rest of this post explains why theory predicts that triangle and how much the result should move you.

## The question: what does next-token prediction build?

Two popular stories describe what a language model learns: surface statistics (a "stochastic parrot") or a "world model" of whatever produced the text. Neither tells you what the activations should *look like*.

This paper borrows a formal theory of optimal prediction, computational mechanics, and asks: if a predictor were perfect, what internal states would it need? Not the world's hidden states, but its **beliefs** about them, plus the rules for updating those beliefs.

Knowing how a machine works (generation) is easy. Figuring out which state it is in right now, from a finite string of its output (prediction), is real computational work. The authors bet that a transformer must do that work, and that it leaves a geometric fingerprint.

## Technique 1: data from a hidden Markov model

**The problem.** To predict the geometry of a network's activations you need the ground-truth structure of its training data. Natural language does not come with one, so the authors generate data from a machine they fully control.

**The analogy.** Picture a board game where a hidden token moves between squares. Each move, the game shouts out a word that depends on the move it made. You hear the words, never the squares.

**The mechanism.** The machine is an edge-emitting hidden Markov model (HMM). It has hidden states $s \in \mathcal{S}$ and a vocabulary of tokens $x \in \mathcal{X}$. Each step it picks a transition and emits a token *on the edge*, so the token and the next state are chosen together.

**The math.** The whole machine is a set of token-labeled matrices, one per token:

$$
T^{(x)}_{ij} = \Pr\left(x, s_j \mid s_i\right)
$$

Here $i$ indexes the current state, $j$ the next state, and $x$ the emitted token. Each $T^{(x)}$ is an $|\mathcal{S}| \times |\mathcal{S}|$ matrix. Summing over tokens gives the plain state transition matrix $T = \sum_x T^{(x)}$, whose rows sum to one.

**Worked example: the Z1R process.** The paper's running toy is Zero-One-Random (Z1R), with three states $S_0$, $S_1$, $S_R$:

| From | Emits | Goes to | Probability |
| --- | --- | --- | --- |
| $S_0$ | 0 | $S_1$ | 1.0 |
| $S_1$ | 1 | $S_R$ | 1.0 |
| $S_R$ | 0 or 1 | $S_0$ | 0.5 each |

So the data looks like `...01R01R01R...`, where `R` is a fair coin. Generating this is trivial. Predicting it is not, because when you start reading mid-stream you do not know where in the `0 1 R` cycle you are.

**When it fails.** A small, finite, stationary HMM is a strong idealization. Any token dataset can be written as an HMM only if you allow infinitely many states and non-stationarity.

## Technique 2: the mixed-state presentation (beliefs, not states)

**The problem.** A predictor never sees the hidden state. It only sees tokens. So what does it actually carry from one token to the next?

**The analogy.** A detective does not know who the culprit is. They hold a probability for each suspect and revise all of them every time a clue arrives. The *list of probabilities* is the detective's state of mind, and it evolves by fixed rules.

**The mechanism.** Call the predictor's belief $\eta$: a row vector with one probability per hidden state. When a token $x$ arrives, push the belief through $T^{(x)}$ and renormalize. That is Bayes' rule written for an HMM.

**The math.** The paper's update rule (Eq. 1) is

$$
\eta' = \frac{\eta \, T^{(x)}}{\eta \, T^{(x)} \mathbf{1}}
$$

where $\eta$ is the current belief (a ${1 \times |\mathcal{S}|}$ row vector), $\eta'$ is the updated belief, and $\mathbf{1}$ is a column vector of ones, so the denominator $\eta T^{(x)} \mathbf{1}$ is exactly the probability of seeing token $x$ from belief $\eta$. Chaining the rule over a whole context $x_0, x_1, \dots, x_N$ gives Eq. 2:

$$
\eta = \frac{\eta^{\varnothing} \, T^{(x_0)} T^{(x_1)} \cdots T^{(x_N)}}{\eta^{\varnothing} \, T^{(x_0)} T^{(x_1)} \cdots T^{(x_N)} \mathbf{1}}
$$

The starting belief $\eta^{\varnothing}$ is the stationary distribution $\pi$ of $T$, the left eigenvector with eigenvalue one. For Z1R that is uniform: one third on each state.

The set of all reachable beliefs, together with the token-labeled jumps between them, is itself an HMM. Computational mechanics calls it the **mixed-state presentation (MSP)**. The generator HMM describes how data is *made*. The MSP describes how data is *predicted*.

**Worked example.** Feed the Z1R observer the context `1 1 0 1`, starting from uniform belief. Every number below comes from applying Eq. 1 directly:

- Start: $\pi = (\tfrac13, \tfrac13, \tfrac13)$. $S_0$ always emits 0 and $S_R$ does so half the time, so $P(\text{next}=0) = \tfrac13 + \tfrac16 = 0.5$.
- See `1`: a 1 can only come from $S_1$ (to $S_R$) or from $S_R$ (to $S_0$, half the time). Belief becomes $(\tfrac13, 0, \tfrac23)$.
- See `1` again: from $(\tfrac13, 0, \tfrac23)$, state $S_0$ cannot emit a 1, so the only path is $S_R \to S_0$. Belief becomes $(1, 0, 0)$. The observer is now certain.
- See `0`, then `1`: belief walks deterministically to $(0,1,0)$, then $(0,0,1)$. Once synchronized, it stays on the vertices.

![Annotated grid showing the Z1R observer's belief over S0, S1 and SR after each token of the context 1 1 0 1, collapsing from uniform to a single certain state](/imgs/blogs/transformers-represent-belief-state-geometry-in-their-residual-stream-1.webp)

The hard part of prediction is the first few tokens, where the observer is *synchronizing*. That is the paper's phrase for what a pretrained model must learn.

**When it fails.** The MSP assumes the generator is known. A real model must also *learn* the generator from data. The paper separates these two jobs (learning the structure, then inferring the current state) and only studies whether the second one shows up in activations.

## Technique 3: belief state geometry in the simplex

**The problem.** A belief is a vector of probabilities. To compare it against activations we need to know where beliefs *live* as geometric objects.

**The analogy.** A three-way election poll is a point in a triangle: each corner is a landslide for one candidate, the center is a three-way tie. Every possible poll result is somewhere inside.

**The mechanism.** Any belief over $|\mathcal{S}|$ states lives in the probability simplex. With three states that simplex is a triangle; with five it is a 4-simplex. The arrangement of all MSP beliefs inside it is what the paper calls the **belief state geometry**.

**The math.** The simplex (Appendix A.4) is

$$
\Delta^{|\mathcal{S}|-1} = \left\{ b \in \mathbb{R}^{|\mathcal{S}|} : \sum_{i=1}^{|\mathcal{S}|} b_i = 1, \; b_i \ge 0 \text{ for all } i \right\}
$$

where $b$ is a belief written as a column of probabilities. Corners are certainty about one state. The perpendicular distance from a point to an edge reads off the probability of the opposite corner, which is how the paper's Figure 3C labels it.

![Figure 3 from Shai et al. (2024): (A) the Z1R generator, (B) its mixed-state presentation, (C) how to read a belief off the simplex, (D) the resulting belief state geometry](/imgs/blogs/transformers-represent-belief-state-geometry-in-their-residual-stream-fig3.webp)

Panel B is the MSP, with belief states labeled by the shortest context that reaches them, such as $\eta^{10}$. Panel D drops them into the triangle.

**Why this is more than next-token information.** Each belief implies a full distribution over all futures, not just the next token. The paper points out that $\pi$, $\eta^{10}$ and $\eta^{01}$ in Z1R are distinct beliefs with the *same* next-token prediction. Here is why that happens geometrically:

$$
P(\text{next}=0 \mid b) = b_{S_0} \cdot 1 + b_{S_1} \cdot 0 + b_{S_R} \cdot 0.5
$$

The next-token probability is a *linear* function of the belief. Its level sets are straight lines through the triangle. Our three beliefs are $\pi = (\tfrac13,\tfrac13,\tfrac13)$, $\eta^{10} = (\tfrac12,\tfrac12,0)$ and $\eta^{01} = (0,0,1)$. Plug each in and you get exactly 0.5 every time: they all lie on the median from $S_R$ down to the midpoint of the opposite edge.

![The Z1R belief triangle with pi, eta 10 and eta 01 on one dashed median line of constant next-token prediction, and eta 0 and eta 1 off the line](/imgs/blogs/transformers-represent-belief-state-geometry-in-their-residual-stream-2.webp)

This picture is why the second experiment exists. Next-token prediction squashes the triangle onto one axis, so a model storing only next-token probabilities could not separate the highlighted points. Yet they predict different *futures*: from $\eta^{01}$ the observer is surely in $S_R$, from $\eta^{10}$ the next state is $S_0$ or $S_1$.

**When it fails.** The simplex grows with the number of hidden states. For natural language it would dwarf the residual stream, so any real model must hold beliefs in compressed form.

## Technique 4: finding the simplex with a linear probe

**The problem.** The residual stream has 64 dimensions here. The belief simplex for Mess3 is a 2D triangle. If the geometry is in there, we need a principled way to pull it out, and a way to make sure we did not just paint the answer onto noise.

**The analogy.** You suspect a city map is hidden in a pile of satellite photos. You fit one flat projector that best lines the photos up with the known street grid, then check whether the projection looks like a map on photos you did not fit on.

**The mechanism.** Run every possible input sequence through the trained model and record the residual activation at every context position. Each activation has a known ground-truth belief, computed with Eq. 2. Fit an affine map from activation to belief.

![The probing recipe: the HMM drives both a transformer, whose residual activations are recorded, and a Bayesian update that gives the true belief, and the two meet in a linear regression](/imgs/blogs/transformers-represent-belief-state-geometry-in-their-residual-stream-4.webp)

**The math.** For activations $a_i \in \mathbb{R}^{d_{\text{resid}}}$ and true beliefs $b_i \in \mathbb{R}^{|\mathcal{S}|}$, the paper solves (Appendix A.5)

$$
\min_{W,\, c} \; \sum_i \left\lVert b_i - \left(W a_i + c\right) \right\rVert^2
$$

where $W \in \mathbb{R}^{|\mathcal{S}| \times d_{\text{resid}}}$ is the projection, $c \in \mathbb{R}^{|\mathcal{S}|}$ is a bias, and the sum runs over all input sequences. For Mess3, $W$ is ${3 \times 64}$: just 192 weights plus 3 biases. Because each $b_i$ sums to one, the fitted points sit on a plane, so this is effectively the "2D subspace" the paper reports.

**Worked micro-example.** Why does the shuffle control work? Suppose the labels $b_i$ are randomly permuted across inputs, so they carry no information about $a_i$. The best affine predictor of an unrelated target is its mean. Every activation then maps to the average belief, which for a stationary process is $\pi$, the center of the triangle. A real fractal in the projection therefore cannot come from regression flexibility alone. This is my derivation of the control's logic; the paper reports the empirical outcome (collapse to the center).

**When it fails.** A linear probe finds *linearly decodable* information, not *used* information. The paper is careful here: it establishes representation, not that a Bayesian-update circuit reads these coordinates. It also cites a hand-built transformer that solves RRXOR perfectly *without* a linear belief representation, so linearity is an empirical fact about SGD-trained models, not a necessity.

## Result 1: the Mess3 fractal appears in the final layer

The first test uses Mess3, a 3-state HMM over tokens `{A, B, C}`. Its transition matrices (Appendix A.3) put 0.765 on a "stay and emit my own letter" entry and small values elsewhere, so every token is weak evidence, never proof.

**Why a fractal?** Run Eq. 1 on Mess3 and the belief never lands on a corner. From uniform belief, seeing `A` gives about $(0.85, 0.075, 0.075)$. A second `A` gives $(0.975, 0.013, 0.013)$, and then a `B` jumps back to about $(0.54, 0.42, 0.04)$. Each token applies one of three maps that pull the point toward a corner without reaching it. Infinitely many contexts give infinitely many distinct beliefs, and repeatedly applying three contracting maps is exactly how classic fractals like the Sierpinski triangle are built. (That iterated-map framing is my gloss; the paper's language is that the Bayesian map is "folded into itself time and again".)

The setup: the final residual stream (before the last LayerNorm and unembedding) of a 4-layer transformer with $d_{\text{model}} = 64$. Linear regression finds a 2D subspace whose geometry closely matches the predicted fractal. Appendix Figure S1 repeats the fit *after* the final LayerNorm and gets an MSE of 0.0003, versus 0.0004 before it.

![Figure 6 from Shai et al. (2024): (A) the belief geometry emerging over training, (B) cross-validated projection, (C) shuffled-label control collapsing to the center, (D) regression MSE](/imgs/blogs/transformers-represent-belief-state-geometry-in-their-residual-stream-fig6.webp)

The controls in Figure 6 are what make this more than a pretty picture:

| Control | What they did | What happened |
| --- | --- | --- |
| Training trajectory | Fit the probe at several checkpoints | Fractal emerges gradually, MSE drops over training |
| Cross-validation | Fit on 20% of input-activation pairs, evaluate on held-out 80%, repeated 1000 times | Geometry still clear; MSE similar to the full fit |
| Shuffle | Keep the fractal's shape, permute which input gets which belief, repeated 1000 times | Everything maps to the simplex center |
| Loss link (Fig. S2) | Plot probe MSE against validation loss normalized to the theoretical optimum | Lower loss goes with lower probe MSE, for both processes |

The training-trajectory panel answers an obvious worry: maybe a random 4-layer network already arranges inputs in a fractal because of its architecture. It does not. The structure grows with learning.

## Result 2: RRXOR keeps information next-token prediction throws away

Mess3 has a weakness as a test: its distinct beliefs mostly have distinct next-token predictions, so a model tracking only next-token probabilities would look the same. The second experiment breaks that tie.

The Random-Random-XOR (RRXOR) process emits two random bits and then their XOR, over and over. It has 5 hidden states, zero pairwise correlation between tokens, and structure only at higher order. Its MSP has 36 distinct belief states living in a 4-simplex, and many of them share the same optimal next-token distribution.

![Figure 7 from Shai et al. (2024): (A) the RRXOR generator, (B) its belief geometry projected from a 4-simplex, (C) the transformer's representation, (D) distance scatter plots, (E) per-layer regression error](/imgs/blogs/transformers-represent-belief-state-geometry-in-their-residual-stream-fig7.webp)

The test is a pairwise-distance comparison:

1. For every pair of belief states, compute the Euclidean distance between the true beliefs.
2. Compute the distance between the corresponding representations inside the transformer.
3. Separately, compute the distance between the two states' next-token distributions.
4. Ask which of the two ground truths better explains the representation distances.

$$
R^2_{\text{belief}} = 0.95 \qquad \text{versus} \qquad R^2_{\text{next-token}} = 0.31
$$

Here $R^2$ is the fraction of variance in the transformer's pairwise distances explained by each ground truth. For RRXOR the residual stream follows the beliefs and largely ignores the next-token view. For Mess3 both correlations are high, which is exactly what the theory predicts when beliefs rarely share predictions.

![Bar chart: for RRXOR, R squared of residual-stream pairwise distances is 0.95 against true belief distances and 0.31 against next-token distances](/imgs/blogs/transformers-represent-belief-state-geometry-in-their-residual-stream-5.webp)

**Why a next-token trainer keeps future information.** The paper's argument is short. Two beliefs that agree on the next token can still disagree on what comes three tokens later. If the model merged them now, it would get that later token wrong. To minimize loss over *every* position, a model has to distinguish every pair of pasts that implies different futures. That set of distinctions is the belief state.

## Result 3: the belief geometry can be spread across layers

For RRXOR there is a twist. The belief geometry is **not** recoverable from the final residual stream, and not from any single layer. It only appears when the residual activations of all layers are concatenated and probed jointly. Mess3 behaves differently: its geometry is well represented in individual layers and survives to the final layer.

![Grid: Mess3 belief geometry is found at every layer and in the concatenation, while RRXOR is found only in the concatenation of all layers](/imgs/blogs/transformers-represent-belief-state-geometry-in-their-residual-stream-3.webp)

The grid is a qualitative read of Figure 7E, not its values. The theory explains the split: the last layer only has to produce the next-token distribution, so beliefs that share one can be merged there. The framework only requires the information to exist somewhere at that position.

This matches the "Future Lens" finding the paper cites: far-future information in natural-language models peaks in intermediate layers.

## Why belief geometry, and not the generator?

A subtle point: many different HMMs generate the same token process, so which simplex should the model use? The paper's answer is that every process has a canonical representation, the minimal HMM, whose recurrent belief states match the **causal states** of the process's ε-machine. Those are the states a predictor must distinguish, even when a smaller generator exists.

Z1R makes this concrete: three states generate it, but its MSP adds transient beliefs like $\eta^{1}$ and $\eta^{10}$ that the observer passes through while synchronizing. A next-token trainer learns the predictor's machine, not the generator's.

The authors expect the same geometry in any residual-stream model trained on future-token prediction, and cite follow-up work finding it in RNNs.

## Setup and results at a glance

| Item | Value |
| --- | --- |
| Model | Transformer, 4 layers, 1 attention head per layer |
| Width | $d_{\text{model}} = 64$, head dimension 8, MLP 256, ReLU, LayerNorm, causal mask |
| Context window | 10 tokens |
| Optimizer | SGD, learning rate 0.01, no weight decay, batch 64, "1,000,000 epochs" as reported |
| Data | Fresh sequences sampled from the HMM each batch, start state from $\pi$ |
| Mess3 | 3 hidden states, tokens `{A,B,C}`, fractal belief geometry |
| RRXOR | 5 hidden states, tokens `{0,1}`, 36 belief states in a 4-simplex |
| Mess3 probe | Final layer, 2D subspace of 64; MSE 0.0004 pre-LayerNorm, 0.0003 post |
| RRXOR probe | Only all-layer concatenation works; $R^2$ 0.95 (beliefs) vs 0.31 (next token) |
| Controls | 20/80 cross-validation and label shuffle, each repeated 1000 times |
| Tooling | TransformerLens |

**What is load-bearing and might not transfer.** Three things in this setup make the result clean, and none of them hold in a real LLM:

- **Enumerable inputs.** With a 10-token window and 2 or 3 tokens, every input can be listed and every true belief computed exactly. For a 50,000-token vocabulary the MSP cannot be computed at all.
- **Tiny simplex.** A 2- or 4-dimensional simplex fits easily inside 64 dimensions. In real data the simplex is larger than the residual stream, so the representation must be compressed, and the paper does not say what that compression looks like.
- **Stationary, ergodic data.** Natural language is neither. The authors argue the framework extends (in-context learning is inherently non-ergodic), but that is a claim for future work.

## Critique

**What is strong.** The paper makes a falsifiable, quantitative prediction from first principles and then checks it. The fractal is a high bar: you do not get a Sierpinski-like structure in a linear projection by accident, and the shuffle control rules out the obvious "regression can draw any shape" objection. The RRXOR experiment is well designed because it separates beliefs from next-token predictions, which is exactly the confound that would otherwise sink the claim.

**What is weak or missing.**

- **Representation is not use.** Nothing ablates or steers the belief subspace to show the model's predictions depend on it. The authors flag this themselves.
- **One model size, one seed (as far as the paper says).** No sweep over depth, width, or random seeds, so we cannot tell how robust the layer-spreading result for RRXOR is.
- **The RRXOR concatenation probe is more powerful.** Concatenating four layers quadruples the probe's input dimension. A fairer comparison would also report a single layer probed with an equally large random projection or a matched-dimension baseline. The paper does not.

**What would change my mind.** I would weaken my belief in the paper's central claim if a causal intervention that moves activations along the probed simplex direction did *not* move the model's next-token predictions in the way Eq. 1 predicts, or if repeating the RRXOR experiment across several seeds and depths showed the "only in the concatenation" result appearing and disappearing at random. In the other direction, finding a compressed version of this geometry in a real pretrained model on a natural-language task with a computable latent structure would make me take the framework much more seriously as a general theory of what pretraining builds.

**How it fits with neighbors on this site.** The result is a concrete instance of the [linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis), except the represented object is a whole shape, not one direction. It pairs with [features that are not one-dimensionally linear](/blog/paper-reading/ai-interpretability/not-all-language-model-features-are-one-dimensionally-linear), whose circular days-of-week result this paper names as a target. And it offers a data-driven reason for the [Platonic representation hypothesis](/blog/paper-reading/ai-interpretability/position-the-platonic-representation-hypothesis): if every good predictor must carry the same canonical belief geometry, models trained on the same data should converge on it.

## References

1. Shai, A. S., Marzen, S. E., Teixeira, L., Gietelink Oldenziel, A., Riechers, P. M. *Transformers Represent Belief State Geometry in their Residual Stream.* NeurIPS 2024. [arXiv:2405.15943v3](https://arxiv.org/abs/2405.15943v3)
2. Shalizi, C. R., Crutchfield, J. P. *Computational mechanics: Pattern and prediction, structure and simplicity.* J. Stat. Phys. 104, 2001. (The theory of causal states the paper builds on.)
3. Pal, K. et al. *Future Lens: Anticipating subsequent tokens from a single hidden state.* [arXiv:2311.04897](https://arxiv.org/abs/2311.04897)
4. Engels, J. et al. *Not all language model features are linear.* [arXiv:2405.14860](https://arxiv.org/abs/2405.14860)
5. On this site: [The linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis), [Not all language model features are one-dimensionally linear](/blog/paper-reading/ai-interpretability/not-all-language-model-features-are-one-dimensionally-linear), [The Platonic representation hypothesis](/blog/paper-reading/ai-interpretability/position-the-platonic-representation-hypothesis)

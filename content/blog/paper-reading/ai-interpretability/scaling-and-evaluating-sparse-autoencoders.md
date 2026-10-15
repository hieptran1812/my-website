---
title: "Scaling and Evaluating Sparse Autoencoders: TopK, Dead Latents, and a 16M-Latent SAE on GPT-4"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How OpenAI trained a 16 million latent sparse autoencoder on GPT-4: the TopK activation, the two fixes for dead latents, the scaling laws they found, and the four metrics they propose for judging whether an SAE is any good."
category: "paper-reading"
subcategory: "AI Interpretability"
tags:
  - paper-reading
  - sparse-autoencoders
  - topk-sae
  - scaling-laws
  - representation-learning
  - mechanistic-interpretability
  - large-language-models
  - iclr-2025-oral
author: "Hiep Tran"
featured: true
readTime: 22
image: "/imgs/blogs/scaling-and-evaluating-sparse-autoencoders-fig1.webp"
excerpt: "Swap the L1 penalty for a TopK activation, add two tricks that keep latents alive, and sparse autoencoders start obeying clean scaling laws all the way to 16 million latents on GPT-4."
paper:
  title: "Scaling and Evaluating Sparse Autoencoders"
  authors: "Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, Jeffrey Wu"
  venue: "ICLR 2025 (Oral); arXiv 2024"
  url: "https://arxiv.org/abs/2406.04093"
---

> [!tldr]
> **Claim.** Sparse autoencoders (SAEs) become easy to scale once you replace the L1 sparsity penalty with a TopK activation that keeps exactly $k$ latents per token, and add two tricks (tied initialization and an auxiliary loss) that stop latents from dying.
> **Why it matters.** With those changes, reconstruction error follows clean power laws in the number of latents $n$ and in $k$, which turns "how big should my SAE be?" into a forecasting problem. The authors train a 16 million latent SAE on GPT-4 for 40 billion tokens with only 7% dead latents.
> **Most surprising result.** Splicing that 16M SAE into GPT-4 still costs as much language modeling loss as a model trained with only 10% of GPT-4's pretraining compute. Even the biggest SAE ever trained at the time leaves a lot of the model unexplained.
> **Where it fails.** TopK overfits to the $k$ it was trained with, forces every token to use the same number of latents, and the paper's quality metrics (probes, n-gram explanations, ablation sparsity) are cheap proxies, not proof that features are meaningful.

Interpretability researchers had a scaling problem in 2024. Sparse autoencoders were the best unsupervised tool for pulling human-readable features out of a language model, but a frontier model plausibly knows millions of concepts, and SAEs at that width were miserable to train. Two knobs fought each other (reconstruction versus sparsity), and a growing share of the latents simply stopped firing.

This paper from OpenAI's Superalignment interpretability team is mostly an engineering paper, and that is its strength. It gives a recipe that makes SAE training boring, then uses the boredom to measure scaling laws and to ask a harder question: once the loss goes down, how do we know the features got better?

## The setup: what an SAE is asked to do

An SAE rewrites one activation vector from inside a language model as a sparse combination of learned directions. Think of a dictionary where every sentence must be described with only a handful of words: the dictionary is the decoder, picking the words is the encoder.

The paper reads activations from the residual stream, the running vector each transformer layer adds into. For GPT-4 family models the layer sits ${5}/{6}$ of the way through the network; for GPT-2 small it is layer 8, about ${3}/{4}$ of the way. Every context is 64 tokens long. Before the SAE sees a vector, the mean over the $d_{\text{model}}$ dimension is subtracted and the vector is scaled to unit norm.

The baseline is the ReLU autoencoder used by Anthropic's "Towards Monosemanticity" work:

$$
z = \text{ReLU}\big(W_{\text{enc}}(x - b_{\text{pre}}) + b_{\text{enc}}\big), \qquad \hat{x} = W_{\text{dec}} z + b_{\text{pre}}
$$

Here $x \in \mathbb{R}^d$ is the residual vector, $n$ is the number of latents, $W_{\text{enc}} \in \mathbb{R}^{n \times d}$ maps the input to $n$ latent scores, $b_{\text{enc}} \in \mathbb{R}^n$ is the encoder bias, $W_{\text{dec}} \in \mathbb{R}^{d \times n}$ holds one dictionary direction per column, and $b_{\text{pre}} \in \mathbb{R}^d$ is a learned center subtracted on the way in and added back on the way out. The training loss is

$$
\mathcal{L} = \lVert x - \hat{x} \rVert_2^2 + \lambda \lVert z \rVert_1
$$

The first term is reconstruction error; the second is an L1 penalty that pushes most entries of $z$ to zero, weighted by a coefficient $\lambda$ you have to tune.

Two evaluation numbers recur throughout. **L0** is the average count of nonzero latents per token. **Normalized MSE** is the reconstruction error divided by the error of always predicting the mean activation, so 1.0 means "no better than guessing the average" and 0 means perfect.

| Symbol | Meaning | Typical value in the paper |
| --- | --- | --- |
| $d$ | residual stream width | 768 for GPT-2 small |
| $n$ | number of latents | ${2^{11}}$ up to 16M |
| $k$ | active latents per token (TopK) | 32 to 512 |
| L0 | measured nonzeros per token | equals $k$ for TopK |
| $\lambda$ | L1 coefficient (baseline only) | swept, e.g. 1.7e-3 to 1.7e-2 |

## Technique 1: the TopK activation

**The problem.** With an L1 penalty, sparsity is a side effect. You pick $\lambda$, train, and only then learn what L0 you got. Comparing two SAEs at the same sparsity means sweeping $\lambda$ until the L0s happen to match. Worse, L1 penalizes the size of every active value, so it quietly shrinks the activations that survive.

**The analogy.** L1 is a tax on every word you use in your description, charged per unit of emphasis. You end up using fewer words, but you also whisper the ones you keep. TopK is a hard word budget: say exactly $k$ words, at full volume, and nothing else.

**The mechanism.** Compute the $n$ pre-activation scores as before, sort them, keep the $k$ largest, and set everything else to zero. The decoder is unchanged. Because sparsity is now enforced by the activation, the loss no longer needs a sparsity term at all.

![TopK SAE forward pass: the residual vector is centered, encoded to n scores, cut to the top k, and decoded back](/imgs/blogs/scaling-and-evaluating-sparse-autoencoders-1.webp)

**The math.** The TopK encoder is

$$
z = \text{TopK}\big(W_{\text{enc}}(x - b_{\text{pre}})\big), \qquad \mathcal{L} = \lVert x - \hat{x} \rVert_2^2
$$

where $\text{TopK}(\cdot)$ keeps the $k$ largest entries of its $n$-dimensional input and zeroes the rest, so $z \in \mathbb{R}^n$ has exactly $k$ nonzeros. The encoder bias $b_{\text{enc}}$ is dropped. In the released code a ReLU is also applied so values stay non-negative, but the authors report the training curves are indistinguishable, since the top $k$ scores are almost always positive.

**Worked example.** Take six pre-activation scores $[2.0, 1.2, 0.9, 0.3, 0.1, -0.4]$ and $k = 3$. TopK returns $[2.0, 1.2, 0.9, 0, 0, 0]$: three latents fire at their full values. Now compare L1. For a single latent with a unit-norm decoder direction, minimizing $(a - z)^2 + \lambda z$ over $z \ge 0$ gives the soft threshold $z = \max(0, a - \lambda/2)$ (my derivation, not the paper's). With $\lambda = 0.4$ the six scores become $[1.8, 1.0, 0.7, 0.1, 0, 0]$. Every survivor lost 0.2, and a weak 0.3 score survived as a small, probably noisy, 0.1.

![L1 penalty versus TopK on the same six scores: L1 shrinks every survivor and lets weak values through, TopK keeps exactly k values unchanged](/imgs/blogs/scaling-and-evaluating-sparse-autoencoders-2.webp)

**Why it works.** The paper lists four benefits: no L1 bias, L0 is set directly (so model comparison and iteration get simpler), a better sparsity-reconstruction frontier that widens with scale, and more monosemantic random activations because small activations are clamped to zero.

Section 5.1 tests the shrinkage claim directly: freeze which latents fire, then re-optimize only their values to minimize MSE. For ReLU SAEs the refined values move up on average, as expected if L1 had pushed them down; for TopK the refinement is centered on zero and smaller. Refinement closes only part of the ReLU to TopK gap, so the advantage is not just shrinkage.

![Figure 2 from Gao et al. (2024): at fixed n = 32768 TopK beats ReLU and ProLU on the MSE versus L0 frontier and roughly ties Gated (left); at fixed L0 = 128 its scaling curve in n is steeper than ReLU (right)](/imgs/blogs/scaling-and-evaluating-sparse-autoencoders-fig2.webp)

**When it fails.** TopK forces every token to use exactly $k$ latents. A token like "the" probably needs fewer concepts than a dense line of code, and the authors list this as their first limitation: ideally you would constrain the expected L0, not the per-token L0. We will also see below that a TopK SAE overfits to its training $k$.

## Technique 2: keeping latents alive

**The problem.** In big SAEs, a growing fraction of latents stop activating entirely. The paper counts a latent as dead if it has not fired in 10 million tokens. Anthropic's 34M-latent SAE had only 12M alive latents, and the authors' own ablations hit up to 90% dead latents with no mitigation. A dead latent is wasted compute and a dictionary word nobody can use, which also makes MSE worse.

**The analogy.** A teacher who only calls on raised hands never teaches the quiet students, so they never learn enough to raise a hand. The fix: seat new students where the action is, and occasionally give the quiet ones a problem only they may answer.

**The mechanism.** The paper uses exactly two ingredients.

- **Tied initialization.** The encoder is initialized as the transpose of the decoder, so each latent starts by reading the same direction it writes. The weights are not tied during training; this is only a starting point.
- **AuxK loss.** Take the main model's leftover error $e = x - \hat{x}$. Ask the top $k_{\text{aux}}$ dead latents to reconstruct that error. Their gradient flows only from this side task, which drags dead directions toward whatever the live latents are currently missing.

**The math.** With $\hat{e} = W_{\text{dec}} z$ computed from only the top-$k_{\text{aux}}$ dead latents,

$$
\mathcal{L}_{\text{aux}} = \lVert e - \hat{e} \rVert_2^2, \qquad \mathcal{L}_{\text{total}} = \mathcal{L} + \alpha \, \mathcal{L}_{\text{aux}}
$$

where $\alpha$ is a small weight, typically ${1}/{32}$, and $k_{\text{aux}}$ is a power of two near $d_{\text{model}}/2$ (512 for GPT-2 small). Because the expensive encoder forward pass is shared with the main loss, AuxK adds only about 10% compute.

![Two fixes for dead latents: tied initialization and the AuxK loss both feed into a run where only 7% of 16M latents end up dead](/imgs/blogs/scaling-and-evaluating-sparse-autoencoders-3.webp)

**Worked example.** Suppose $x = [1.0, 0.0]$ and the live latents reconstruct $\hat{x} = [0.7, 0.0]$. The residual is $e = [0.3, 0.0]$. If one dead latent has decoder direction $[1, 0]$ and its encoder score on $x$ is the largest among dead latents, AuxK asks it to output 0.3 along $[1, 0]$. Its loss is then $\alpha \cdot 0 = 0$ after fitting, and on the way there it receives a real gradient. Next time this input pattern appears, that latent has a fighting chance of making the main top $k$.

**Why it works and when it fails.** The Appendix C ablation (GPT-2 small, 2M latents, $k = 32$) attributes the reduction in dead latents mostly to the combination of AuxK and tied init. The result: even the 16M-latent GPT-4 SAE ends with only 7% dead. One caveat the paper is candid about: the baselines it compares against use resampling or L1 warm-up and often end up with as few dead latents as TopK (Figure 14), so dead-latent prevention is a property of this recipe, not something only TopK can do.

## Technique 3: scaling laws for SAEs

**The problem.** If you want an SAE for a frontier model, you need to know how many latents to buy and how many tokens to train on before you spend the compute. Language model scaling laws answer that for pretraining; nothing comparable existed for SAEs.

**The analogy.** Think of the SAE as a camera sensor with $n$ pixels, where each photo may light up at most $k$ of them. More pixels and a bigger per-photo budget both sharpen the image, but with diminishing returns, and some blur comes from the lens itself (noise no sensor fixes).

![Figure 1 from Gao et al. (2024): scaling laws for TopK SAEs on GPT-4 activations. Left: the compute-optimal loss L(C) follows a power law in training FLOPs. Right: loss at convergence for each pair of total latents n and active latents k](/imgs/blogs/scaling-and-evaluating-sparse-autoencoders-fig1.webp)

**The mechanism.** The paper studies two training regimes and then fits a joint law.

1. **$L(C)$, the compute frontier.** Train each size for whatever token budget minimizes MSE at a given compute, ignoring convergence. On GPT-4 the fit in Figure 1 (left) is $L(C) = 0.09 + 0.056\,C^{-0.084}$, with $C$ the training compute as a fraction of GPT-4 pretraining compute. The authors call this regime arguably unprincipled for SAEs, because a wider SAE has a looser bottleneck and lower MSE comes cheap.
2. **$L(N)$, training to convergence.** Train every size until it stops improving. This bounds the best reconstruction the method can achieve. Here the largest stable learning rate scales as ${1}/\sqrt{n}$, the best $L(N)$ learning rate is about four times smaller than the best $L(C)$ one, and tokens to convergence grow roughly as $\Theta(n^{0.6})$ for GPT-2 small and $\Theta(n^{0.65})$ for GPT-4.
3. **An irreducible term.** Fits improve substantially once you add a constant floor, the familiar $y = \alpha x^{\beta} + e$ form. One explanation the authors offer: activations contain some unstructured noise, which they show in Appendix G scales with an exponent near zero.

**The math.** The joint fit on GPT-4 SAEs, valid only for small $k$ (reconstruction becomes trivial as $k$ approaches $d_{\text{model}}$), is

$$
L(n, k) = \exp\big(\alpha + \beta_k \log k + \beta_n \log n + \gamma \log k \log n\big) + \exp\big(\zeta + \eta \log k\big)
$$

with $\alpha = -0.50$, $\beta_k = 0.26$, $\beta_n = -0.017$, $\gamma = -0.042$, $\zeta = -1.32$ and $\eta = -0.085$. The first exponential shrinks as the SAE grows; the second is the irreducible floor. Because $\gamma$ is negative, the effective slope in $\log n$ is $\beta_n + \gamma \log k$, which gets steeper as $k$ grows. Because $\eta$ is negative, the floor itself falls as $k$ rises.

**Worked example.** Plugging numbers into the fit (my arithmetic, natural logs):

| $n$ | $k$ | reducible term | floor | predicted normalized MSE |
| --- | --- | --- | --- | --- |
| 32,768 | 32 | 0.276 | 0.199 | 0.474 |
| 32,768 | 512 | 0.169 | 0.157 | 0.326 |
| 1,048,576 | 32 | 0.157 | 0.199 | 0.356 |
| 16,777,216 | 32 | 0.100 | 0.199 | 0.299 |

Going from 32K to 16M latents at $k = 32$ is a 512× increase in width, and it cuts the reducible error by about 2.8×. The floor does not move with $n$ at all, so at 16M latents two thirds of the predicted error is floor. That extrapolation is mine: the 16M SAE was never trained to convergence, so it is not part of the authors' $L(N)$ series.

**Subject model size.** At fixed $k = 32$, bigger GPT-4 family models need more latents for the same MSE, with a worse exponent (Figure 4). That is the uncomfortable direction.

**When it fails.** A scaling law for MSE is only useful if MSE tracks what we care about. The paper says this itself: an infinitely wide SAE with $k = 1$ could reconstruct perfectly by tiling $\mathbb{R}^d$ densely with latents, and it would be completely uninteresting. That is why the second half of the paper exists.

## Technique 4: grading an SAE beyond reconstruction

The authors propose four quality metrics, each catching a different way an SAE can look good on MSE and still be bad.

![Four ways to grade an SAE, and how each responds to more latents (n) and more active latents (k)](/imgs/blogs/scaling-and-evaluating-sparse-autoencoders-4.webp)

### Downstream loss: does the model still work?

**Problem.** An SAE can have low MSE while dropping exactly the small directions the model relies on. **Mechanism.** Replace the residual stream with the SAE reconstruction during the forward pass and measure the damage to next-token prediction, as KL divergence or as delta cross-entropy (the gap to a perfect reconstruction).

The twist is how to report it. Prior work reported the fraction of loss recovered relative to zero-ablating the residual stream, which is so catastrophic that almost anything scores well: the 16M SAE gets 98.2%. The authors instead ask what pretraining compute a model needs to match the spliced-in loss. On that scale the 16M SAE turns GPT-4 into roughly a model trained with 10% of its compute. Same SAE, very different story.

They also find that TopK improves downstream loss more than it improves MSE over prior methods (Figure 5), and that MSE and downstream loss follow a clean power law relationship when L0 is fixed and only $n$ varies.

### Probe loss: did it find features we expected?

**Problem.** If sentiment or language identity are real features, a good SAE should have a latent for each. **Mechanism.** Over 61 binary classification datasets, fit a one-dimensional logistic probe on every single latent and keep the best one:

$$
\min_{i,\,w,\,b} \; \mathbb{E}\Big[\, y \log \sigma(w z_i + b) + (1 - y)\log\big(1 - \sigma(w z_i + b)\big) \Big]
$$

where $z_i$ is the pre-activation of latent $i$, $y \in \{0, 1\}$ is the task label, $\sigma$ is the sigmoid, and $w, b$ are the scalar probe weight and bias fit by Newton-Raphson. (The paper writes the minimization over this expression; read it as minimizing the cross-entropy, which is its negation.) The score is the loss of the best single latent, so this asks "is the concept sitting in one latent?" rather than "is it linearly decodable somewhere?"

On GPT-2 small, probe loss improves and then worsens as $k$ grows, and TopK beats ReLU, with both beating raw residual stream channels. On the GPT-4 SAEs it improves throughout training with no supervised signal at all. The weakness is built in: it assumes we know in advance which features are natural, and the authors call the metric noisy.

### Explainability: precision, not just recall

**Problem.** A latent that fires on "don't stop" and "can't stop" can get a high automated interpretability score from the explanation "fires on stop", which has great recall and terrible precision. The authors warn this illusion gets worse as SAEs scale and features become narrower.

**Mechanism.** Simulating precision with GPT-4 explanations is too expensive, so they use an improved Neuron to Graph (N2G) method. For each activating sequence, find the shortest suffix that still activates the latent at half its original strength or more, try replacing each position with padding to create wildcards, and check whether absolute position matters. Up to 16 activating examples build the explanation; another 16 measure recall.

Two findings stand out. SAEs with more total latents and fewer active latents are easiest to explain this way. And TopK latents have more than 1.5× better recall at only slightly worse precision (above 0.9×) than ReLU latents at the same $n$ and similar L0, which the authors read as TopK having fewer spurious activations.

They also simulate each latent from its N2G explanation, decode, and measure downstream loss. That captures more of GPT-2 small than a bigram baseline, and larger, sparser SAEs do better.

![Figure 6 from Gao et al. (2024): probe loss (left) and logit diff sparsity (right) across total latents n and active latents k for GPT-2 small SAEs; both are worse at L0 = 512](/imgs/blogs/scaling-and-evaluating-sparse-autoencoders-fig6.webp)

### Ablation sparsity: are the effects as sparse as the activations?

**Problem.** If a model's computation is sparse, a good feature should affect only a few output logits when removed. **Mechanism.** At a token, zero one latent at a time, record how the logits change over the vocabulary of size $V$, subtract the median change per position (a constant shift does not change probabilities), and concatenate across the next $T$ tokens to get a vector of $V \cdot T$ numbers. Its sparsity is

$$
S = \frac{1}{V T}\left(\frac{\lVert v \rVert_1}{\lVert v \rVert_2}\right)^2
$$

where $v$ is that concatenated vector. The squared ratio counts the "effective number of affected tokens"; dividing by $VT$ turns it into a fraction, so smaller is sparser.

**Worked example.** If $v = [3, 0, 0, 0]$, then $\lVert v \rVert_1 = 3$, $\lVert v \rVert_2 = 3$ and the ratio squared is 1: one token affected, $S = 0.25$. If $v = [1, 1, 1, 1]$, the ratio squared is $({4}/{2})^2 = 4$: every token affected, $S = 1$.

With $T = 16$ on GPT-2 small layer 8, SAE latents land at 10 to 14%, while ablating raw residual channels gives 60%, close to the roughly ${2}/\pi$ expected for random vectors. Larger $k$ makes effects sparser, until $k = 512$, where it reverses because $k$ is approaching $d_{\text{model}} = 768$ and latents turn dense.

## Technique 5: progressive codes and Multi-TopK

**The problem.** Can you change sparsity after training? If latents are ordered by magnitude, a good code should degrade gracefully as you keep fewer of them and improve as you keep more, like a progressive JPEG that sharpens as it loads.

**The mechanism.** Swap the trained activation for $\text{TopK}(k')$ at test time with a different $k'$, or for a JumpReLU that fires above a fixed threshold $\theta$:

$$
J_\theta(x) = x \cdot \mathbb{1}(x > \theta)
$$

where $\mathbb{1}(\cdot)$ is 1 when the condition holds and 0 otherwise. Unlike TopK, JumpReLU lets the number of active latents vary per token.

![Figure 10 from Gao et al. (2024): changing sparsity at test time. A TopK SAE degrades once k exceeds its training value; Multi-TopK, ReLU and Gated SAEs stay on a smooth curve, with TopK and JumpReLU test-time curves nearly overlapping for Multi-TopK](/imgs/blogs/scaling-and-evaluating-sparse-autoencoders-fig10.webp)

**What happens.** A plain TopK SAE is progressive only up to roughly its training $k$. MSE keeps improving slightly past $k$, then gets substantially worse, which the authors describe as overfitting to $k$. The fix is to sum losses at several sparsities. Their example:

$$
\mathcal{L}_{\text{multi}} = \mathcal{L}(k) + \tfrac{1}{8}\,\mathcal{L}(4k)
$$

That is enough to get a progressive code over all $k'$, at the price of slightly worse MSE at the training $k$ itself. With Multi-TopK, test-time TopK and JumpReLU curves nearly coincide, so the same SAE works with fixed or variable per-token sparsity.

**Worked example.** With $k = 32$, Multi-TopK reconstructs each input twice, from the top 32 and from the top 128 latents, weighting the second loss by ${1}/{8}$. The 32 are a subset of the 128, so the top latents must carry most of the signal on their own.

This idea later grew into Matryoshka SAEs, which nest dictionaries rather than sparsity levels; see our post on Matryoshka sparse autoencoders.

## The systems recipe that made 16M latents possible

| Ingredient | Setting from the paper | Why it matters |
| --- | --- | --- |
| Batch size | 131,072 tokens (prior work: 2,048 to 4,096) | parallelism; $L(N)$ barely depends on it once hyperparameters are right |
| Optimizer | Adam, $\beta_1 = 0.9$, $\beta_2 = 0.999$, constant LR | decay schedules gave no consistent gain |
| Adam epsilon | ${6.25} \times 10^{-10}$, lower for the largest runs | batch-averaged gradients are tiny; a large epsilon breaks scale invariance |
| Decoder norm | unit-norm columns, gradient parallel to them projected out | keeps scale in the encoder, not the dictionary |
| Weight EMA | coefficient 0.999 | tolerates slightly higher learning rates |
| Gradient clipping | needed for GPT-4 SAEs | prevented divergence at high learning rates |
| Sparse kernels | dense-sparse matmul for decoder and gradients | up to 6× compute saving in the sparse limit; only the encoder forward stays dense |
| $b_{\text{pre}}$ init | geometric median of a data sample | starts the center in the right place |

With $k$ nonzeros out of $n$, the decoder and most backward passes touch only $k$ columns, so the dense encoder forward pass dominates compute. The authors suggest mixture-of-experts style encoders to cut that remaining cost.

## Results at a glance

| Comparison | Setting | Result |
| --- | --- | --- |
| TopK vs ReLU, ProLU, Gated (MSE vs L0) | GPT-2 small, $n = 32768$ | TopK better than ReLU and ProLU, comparable to Gated |
| Scaling in $n$ at fixed L0 = 128 | GPT-2 small | TopK curve steeper than ReLU |
| Downstream loss vs L0 | GPT-2 small, $n = 2^{17}$ | TopK best among ReLU, ProLU-ReLU, ProLU-STE, Gated |
| Dead latents, no mitigation | ablation | up to 90% dead |
| Dead latents, full recipe | GPT-4, 16M latents | 7% dead |
| Compute-equivalent downstream loss | GPT-4, 16M latents | about 10% of GPT-4 pretraining compute |
| Zero-ablation fidelity | GPT-4, 16M latents | 98.2% |
| N2G vs ReLU at same $n$ | GPT-2 small | recall above 1.5×, precision above 0.9× |
| Ablation sparsity $S$ | GPT-2 small layer 8, $T = 16$ | 10 to 14% for SAE latents, 60% for residual channels |

**What is load-bearing and might not transfer.** Everything uses a 64-token context, which the authors flag as possibly too short for GPT-4's interesting behavior. Most metric comparisons are on GPT-2 small at a single layer. And none of the metrics says which SAE is best for a concrete task such as steering or circuit finding, which the authors list as future work.

## My take

**What is strong.** The paper makes SAE training a solved engineering problem at a scale nobody had reached, and it publishes the boring details (epsilon, EMA, batch size, kernels) that usually live in someone's head. The compute-equivalent framing of downstream loss is the most useful single idea in it: 98.2% of loss recovered sounds finished, while "as good as a model with a tenth of the compute" says plainly how far there is to go. The probe and ablation metrics improving with $n$ under no supervision is decent evidence that wider SAEs are not just memorizing.

**What is weak.** Each metric is a proxy with a known escape hatch. Single-latent probes reward a dictionary that happens to isolate 61 hand-picked concepts. N2G can only express n-gram patterns, so it rewards token-level latents and is blind to abstract ones. Ablation sparsity measures output effects at one layer over 16 tokens. None of these tests whether a feature is the model's own unit of computation. Later work makes the gap concrete: [feature absorption](/blog/paper-reading/ai-interpretability/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders) shows SAE latents can silently fail on exactly the cases a clean explanation promises, and sparse probing studies question whether SAE features beat simple baselines on real tasks.

**Missing ablations.** The joint scaling law omits the 16M SAE because it never converged, so the headline model sits outside the headline law. There is no comparison of TopK against a well-tuned Gated or JumpReLU SAE on the four new quality metrics at GPT-4 scale. And the irreducible floor is fit, not explained: Appendix G shows noise would produce one, but nobody checks how much of GPT-4's residual stream is actually unstructured.

**My extrapolation.** If the floor really is structure-free noise, buying more latents stops paying off near the widths reported here, and the next gains must come from variable per-token sparsity or training on downstream KL instead of MSE. That is my reading of the fit, not the authors' claim.

**What would change my mind.** I would stop treating these metrics as proxies if an SAE ranked best by probe loss, N2G precision and ablation sparsity were also shown to be best at a concrete task, such as finding a circuit that a human then verifies by intervention, or steering a behavior more reliably than a difference-of-means vector. Conversely, if a far wider SAE trained to convergence on GPT-4 broke through the predicted floor, I would read the irreducible term as an artifact of the training recipe rather than of the activations.

## References

1. Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, Jeffrey Wu. [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093). arXiv:2406.04093, 2024; ICLR 2025 (Oral).
2. Code and released GPT-2 small autoencoders: [github.com/openai/sparse_autoencoder](https://github.com/openai/sparse_autoencoder).
3. Alireza Makhzani and Brendan Frey. k-Sparse Autoencoders. arXiv:1312.5663, 2013.
4. Trenton Bricken et al. Towards Monosemanticity: Decomposing Language Models with Dictionary Learning. Transformer Circuits Thread, 2023.
5. Related on this site: Matryoshka sparse autoencoders, [feature splitting and absorption in SAEs](/blog/paper-reading/ai-interpretability/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders), are sparse autoencoders useful?.

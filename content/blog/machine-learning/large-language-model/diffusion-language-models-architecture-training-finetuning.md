---
title: "Diffusion Language Models: A First-Principles Deep Dive into Architecture, Loss, and Finetuning"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "How diffusion language models actually work — the discrete-diffusion math, the masked cross-entropy loss derived properly, the bidirectional architecture, parallel decoding, block diffusion, and how to SFT and RL them, with runnable PyTorch and production case studies."
tags:
  [
    "diffusion-language-models",
    "masked-diffusion",
    "mdlm",
    "llada",
    "discrete-diffusion",
    "finetuning",
    "loss-function",
    "block-diffusion",
    "diffu-grpo",
    "score-entropy",
    "parallel-decoding",
    "llm-architecture",
  ]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 51
---

For seven years, "language model" has meant one thing: a stack of causal transformer layers predicting the next token, left to right, one forward pass at a time. The autoregressive (AR) factorization $p(x) = \prod_i p(x_i \mid x_{\lt i})$ is so dominant that most engineers treat it as synonymous with the problem itself. It is not. It is *one* way to decompose a joint distribution over text, and it carries a structural cost that no amount of kernel engineering removes: generation latency is linear in output length, because token $i+1$ cannot start until token $i$ is committed.

Diffusion language models (DLMs) make a different bet. Instead of factorizing left to right, they learn to **denoise**: corrupt a sequence by masking tokens, then train a bidirectional transformer to recover the originals. At inference, you start from a fully masked sequence and iteratively un-mask it over a *fixed* number of steps — and crucially, every position is predicted in parallel at each step. The output length stops gating the number of forward passes. In 2025 this stopped being a research curiosity: [LLaDA-8B](https://arxiv.org/abs/2502.09992) matched a same-size LLaMA3 baseline trained from scratch, and commercial systems like Mercury and Gemini Diffusion shipped diffusion decoders running at over a thousand tokens per second.

This post is the from-the-loss-up version. We will derive the forward process, the loss function (properly, not hand-waved), the architecture, the sampling loop, the inference tricks that make it fast, and the recipes for supervised finetuning and reinforcement learning. There is runnable PyTorch for each core mechanism and a set of production case studies at the end. If you want the library-level tour — schedulers, samplers, distributed training wiring — read its companion, [dllm: an engineer's deep dive into the diffusion LM library](/blog/machine-learning/open-source-library/dllm-diffusion-language-models-deep-dive). This post is about understanding the paradigm well enough to re-implement it.

## Why diffusion for language is a different bet

Start with the mismatch between what people assume about diffusion-for-text and what is actually true.

| Common assumption | The naive view | The reality |
| --- | --- | --- |
| "Autoregression is the only way to model language." | Next-token prediction is fundamental. | It is one factorization. Diffusion models the *joint* via iterative denoising and reaches competitive likelihoods. |
| "Diffusion is for images; text is discrete and diffusion needs continuous noise." | You can't add Gaussian noise to a token id. | Discrete diffusion corrupts tokens by *masking* (an absorbing state), no continuous latent required. |
| "Diffusion LMs must be slower — they take many denoising steps." | Many steps × full sequence = expensive. | Each step decodes *all* positions in parallel; with 32–128 steps a 1000-token output costs far fewer passes than 1000 AR steps. |
| "They can't do in-context learning or instruction following." | Only AR models follow instructions. | LLaDA-8B-Instruct does few-shot ICL and instruction following on par with AR peers. |
| "You can't reuse the KV cache, so serving is hopeless." | Bidirectional attention recomputes everything. | Block diffusion restores a block-causal KV cache; training-free methods like Fast-dLLM cache the prefix. |

The four wins that actually motivate the bet are concrete:

1. **Parallel decoding.** A diffusion step predicts every masked position at once. Latency is set by the *step budget* $K$, not the sequence length $L$. When $K \ll L$ you get a genuine wall-clock win.
2. **Bidirectional context.** Every position attends to every other, so a token's prediction is informed by what comes *after* it. This is exactly what you want for infilling, editing, and constraint satisfaction.
3. **Iterative self-correction.** A token committed at step 3 can be revisited if the model's confidence in it collapses — the decoder is not forced to live with an early mistake the way an AR sampler is.
4. **Any-order generation.** Because the loss trains the model to fill arbitrary masked subsets, a diffusion LM is natively an any-order model. Infilling a function body given its signature *and* its call sites is the same operation as left-to-right generation.

None of this is free. Diffusion LMs pay in training compute (each example is seen at many noise levels), in likelihood evaluation that is not directly comparable to AR perplexity, and in a serving stack that the entire industry has *not* spent five years optimizing. The rest of this post is about where each cost lives and how to pay it down.

![Autoregressive generation emits one token per forward pass while a diffusion model fills every position in parallel over a fixed step budget](/imgs/blogs/diffusion-language-models-architecture-training-finetuning-1.png)

The diagram above is the mental model for the whole article. On the left, the AR model: one token per forward pass, a causal mask, $N$ tokens cost $N$ passes, and no revising a token once emitted. On the right, the diffusion model: all positions filled in parallel, $K$ denoising steps with $K$ potentially much smaller than $N$, full bidirectional context, and low-confidence tokens revised on the next step. Everything that follows — the corruption process, the loss, the sampler — is a tour of how the right column is trained and run.

> A diffusion LM is not "BERT that generates." It is a model of $p(x_0)$ defined by a learned reverse-time denoising process. The bidirectional encoder is the *parametrization*; the diffusion objective is what turns it into a generative model.

## From continuous to discrete diffusion

Diffusion for text did not arrive fully formed. It took three years and three distinct ideas to get from "spray Gaussian noise on word embeddings" to the masked objective that scales today.

![The lineage of diffusion language models from D3PM and Diffusion-LM through SEDD and MDLM to LLaDA and production systems](/imgs/blogs/diffusion-language-models-architecture-training-finetuning-2.png)

The first attempts were **continuous**. [Diffusion-LM](https://arxiv.org/abs/2205.14217) (Li et al., 2022) mapped tokens to continuous embeddings, ran a standard Gaussian DDPM in that space, and rounded back to tokens at the end. It worked well enough for controllable generation but fought a fundamental impedance mismatch: text is discrete, the embedding space is not, and the rounding step is lossy and unstable. If you want the continuous-diffusion foundations, the [DDPM deep dive](/blog/paper-reading/denoising-diffusion-probabilistic-models) and the [diffusion models primer](/blog/machine-learning/deep-learning/diffusion-models) cover the Gaussian machinery this lineage started from.

The second idea was to make the noise itself **discrete**. [D3PM](https://arxiv.org/abs/2107.03006) (Austin et al., 2021) defined the forward process as a Markov chain over the vocabulary with a transition matrix $Q_t$: at each step, every token has some probability of flipping to another token, to a uniform random token, or — the variant that mattered — to a special absorbing **mask** state. D3PM gave a clean variational objective but left a lot of design space (which $Q_t$? how to parametrize the reverse?) and its early instantiations underperformed AR baselines badly.

The third idea narrowed the design space twice. [SEDD](https://arxiv.org/abs/2310.16834) (Lou, Meng & Ermon, 2024) reframed discrete diffusion as learning *score ratios* between states and trained them with a **score entropy** loss, closing much of the gap to AR perplexity and giving a principled likelihood. Then [MDLM](https://arxiv.org/abs/2406.07524) (Sahoo et al., 2024) showed that if you commit to the **absorbing-state (masking)** process specifically, the whole variational objective collapses to a simple, weighted cross-entropy on masked tokens — no transition matrices, no score parametrization, no timestep embedding required. That simplification is what made scaling tractable: [LLaDA](https://arxiv.org/abs/2502.09992) trained an 8B masked diffusion model from scratch and matched a LLaMA3-8B-scale AR baseline, and within a year Mercury, Gemini Diffusion, and [Seed Diffusion](/blog/paper-reading/large-language-model/seed-diffusion-a-large-scale-diffusion-language-model-with-high-speed-inference) turned it into shipping products.

The takeaway from the lineage: **the field converged on masked / absorbing-state diffusion** because it is the variant where the math is simplest, the architecture is a vanilla bidirectional transformer, and the loss is something you already know how to optimize. The rest of this post lives almost entirely in that world. We will mention SEDD's score entropy where it clarifies the picture, but the working objective is MDLM's masked cross-entropy.

## 1. The forward process: corruption by masking

**Senior rule of thumb: in masked diffusion, "adding noise" means replacing tokens with a single absorbing mask symbol, independently per position, at a rate the schedule controls.**

Let the data be a sequence $x_0 = (x_0^1, \dots, x_0^L)$ of token ids from a vocabulary $V$. Augment the vocabulary with one extra symbol, $\mathbf{m}$ = `[MASK]`, so the model's embedding table has $|V| + 1$ rows. Define a continuous time $t \in [0, 1]$ and a **noise schedule** $\alpha_t$ that decreases monotonically from $\alpha_0 = 1$ (clean) to $\alpha_1 = 0$ (fully masked). The forward marginal — the distribution of the corrupted sequence at time $t$ given the clean one — factorizes over positions and is dead simple:

$$
q(x_t^i \mid x_0^i) = \alpha_t \, \delta(x_t^i = x_0^i) + (1 - \alpha_t)\, \delta(x_t^i = \mathbf{m}).
$$

In words: at time $t$, each token *independently* either keeps its original value (with probability $\alpha_t$) or becomes `[MASK]` (with probability $1 - \alpha_t$). The mask state is **absorbing**: once a token is masked it carries no information about its identity, and the only thing the model can do is predict it back. There is no "wrong token" corruption in the absorbing process — a position is either itself or a blank.

![The forward process replaces each token with a mask symbol at a rate that grows with the noise level until every token is masked](/imgs/blogs/diffusion-language-models-architecture-training-finetuning-3.png)

The figure traces one sentence through increasing noise. At $t = 0$ everything is clean; at $t = 0.3$ a single token has flipped to `[MASK]`; at $t = 0.6$ most are masked; at $t = 1$ the sequence is entirely mask symbols. The model never sees the corruption *trajectory* — only a single $(t, x_t)$ pair per training example — but across the dataset it sees every corruption level.

### 1.1 A worked numerical example

Take the linear schedule $\alpha_t = 1 - t$, a sequence of $L = 10$ tokens, and a sampled time $t = 0.3$. The mask probability per token is $1 - \alpha_t = t = 0.3$. The expected number of masked tokens is $t \cdot L = 3$. Concretely you would draw 10 independent Bernoulli$(0.3)$ variables, replace the tokens where the draw is 1 with `[MASK]`, and feed the result to the model with the instruction "predict the originals where you see a mask." On a different example you might sample $t = 0.85$ and mask roughly 8.5 of the 10 tokens. Over a training run, $t \sim \mathcal{U}(0, 1)$, so the model is trained uniformly across "almost clean" and "almost fully masked" inputs.

This is the entire forward process. There is no per-step Markov chain to simulate at training time — the marginal at any $t$ is available in closed form, so you sample $t$, sample the mask, and you are done. Contrast that with continuous diffusion, where the forward process is also closed-form but the reverse parametrization (predicting $\epsilon$, $x_0$, or $v$) and the variance schedule carry far more baggage.

### 1.2 The noise schedule, and why it is a knob

The schedule $\alpha_t$ decides how fast tokens disappear as $t$ grows. The two common choices are linear ($\alpha_t = 1 - t$) and cosine ($\alpha_t = \cos(\tfrac{\pi}{2} t)$), borrowed from image diffusion.

![Linear masks tokens at a constant rate while a cosine schedule keeps them unmasked longer early then masks fast near the end](/imgs/blogs/diffusion-language-models-architecture-training-finetuning-5.png)

A linear schedule masks at a constant rate: a uniformly sampled $t$ gives a uniformly distributed masking fraction. A cosine schedule keeps tokens unmasked longer at low $t$ (the curve hugs the top) and then masks quickly near $t = 1$, so the model spends proportionally more training steps on lightly-corrupted, easier-to-denoise inputs. The schedule does **not** change the per-token loss weight directly — as we will see, the $1 / (1 - \alpha_t)$ factor in the loss exactly offsets the expected number of masked tokens so that every $t$ contributes evenly to the gradient. What the schedule changes is the *distribution of difficulty* the model trains on, which empirically affects sample quality and the step-count/quality tradeoff at inference. For language, the differences between linear and cosine are smaller than in image diffusion; LLaDA and MDLM both work well with a near-linear schedule, and the schedule becomes a second-order tuning knob rather than a make-or-break choice.

### 1.3 Why the absorbing state beat the alternatives

D3PM allowed any forward kernel $Q_t$, and the obvious candidates were *uniform* (a token flips to a random other token) and *embedding-nearest* (flips to a token nearby in representation space). Both lost to the absorbing (mask) kernel, and the reasons are worth internalizing because they explain the whole design.

A uniform kernel corrupts by *substitution*: a position holds a real but wrong token. The model then faces an ambiguous job — is this token correct context, or noise to be overwritten? It cannot tell from the input alone, so it must hedge, and the reverse process has to model "detect and replace this plausible-looking token," which is strictly harder than "fill this blank." The absorbing kernel removes the ambiguity entirely: a `[MASK]` is unambiguously a slot to fill, and a real token is unambiguously clean context. That clean separation is exactly what lets the loss collapse to cross-entropy on masked positions, and what lets the model infer the noise level from the mask fraction instead of a timestep embedding.

There is an information-theoretic point too. Under the absorbing process, a token still present at time $t$ carries its *full* original information — no corruption has touched it. Under a substitution process every position is partially corrupted, so the model can never fully trust any input token. Masked diffusion's "either perfectly clean or completely gone" structure means the model always has a subset of perfectly reliable context to condition on, which empirically trains faster and samples better. This is why every production diffusion LM uses the absorbing kernel, and why the rest of this post can assume it without loss of generality.

## 2. The loss function, derived properly

**Senior rule of thumb: the masked-diffusion loss is a cross-entropy on masked positions only, reweighted by $1/(1-\alpha_t)$, integrated over noise levels — and that is a tight bound on the negative log-likelihood.**

Here is where masked diffusion earns its place. The training objective for a general discrete diffusion model is the variational bound (ELBO) on $\log p_\theta(x_0)$, a sum of KL terms over the diffusion steps. For the absorbing-state process specifically, MDLM showed that bound telescopes into something you can write in one line. With a continuous-time formulation, the negative ELBO is

$$
\mathcal{L}_{\text{NELBO}}
= \mathbb{E}_{t \sim \mathcal{U}(0,1)} \;
\mathbb{E}_{q(x_t \mid x_0)}
\left[
\frac{\alpha_t'}{1 - \alpha_t}
\sum_{i=1}^{L} \mathbf{1}[x_t^i = \mathbf{m}]
\; \log p_\theta\!\left(x_0^i \mid x_t\right)
\right],
$$

where $\alpha_t' = \tfrac{d\alpha_t}{dt} < 0$ (so the coefficient is positive), and $p_\theta(\cdot \mid x_t)$ is the model's predicted distribution over the *clean* vocabulary $V$ at each position, given the corrupted sequence $x_t$. Read it slowly:

- **Only masked positions contribute.** The indicator $\mathbf{1}[x_t^i = \mathbf{m}]$ kills the loss at every position that is still its original token. Clean tokens are pure *conditioning context* — the model attends to them but is never asked to predict them.
- **The target is the clean token $x_0^i$.** The model outputs a categorical over $V$ at each masked slot, and the loss is the negative log-probability it assigns to the truth. That is just cross-entropy.
- **The weight $\alpha_t' / (1 - \alpha_t)$ comes from the diffusion bound**, not from a heuristic. It is the rate at which mask-mass changes relative to how much is already masked.

If you adopt LLaDA's convenient parametrization — sample $t \sim \mathcal{U}(0,1)$ and let $t$ *be* the masking probability (equivalently $\alpha_t = 1 - t$) — the objective takes the form everyone implements:

$$
\mathcal{L}(\theta)
= - \, \mathbb{E}_{t \sim \mathcal{U}(0,1)} \;
\mathbb{E}_{x_0,\, x_t}
\left[
\frac{1}{t}
\sum_{i=1}^{L}
\mathbf{1}[x_t^i = \mathbf{m}]
\; \log p_\theta\!\left(x_0^i \mid x_t\right)
\right].
$$

The $1/t$ factor is doing something subtle and important. At a given $t$, the expected number of masked tokens is $t \cdot L$, and each is weighted by $1/t$, so the *expected total weight per sequence* is $(t L)(1/t) = L$, **independent of $t$**. The reweighting makes the Monte-Carlo estimate over a randomly sampled $t$ an unbiased estimator of a per-token objective: a step that happens to mask very few tokens does not get drowned out by a step that masks almost all of them. This objective is provably an upper bound on the negative log-likelihood $-\log p_\theta(x_0)$, so minimizing it is principled likelihood training, not a surrogate.

### 2.1 Three objectives, one family

It helps to see masked diffusion next to its cousins, because they trade simplicity for generality.

![D3PM SEDD and MDLM differ in what they regress and how they weight masked positions inside the training loss](/imgs/blogs/diffusion-language-models-architecture-training-finetuning-4.png)

| Objective | What the model regresses | Loss form | Weighting | Where it shines |
| --- | --- | --- | --- | --- |
| **D3PM** (2021) | The reverse posterior $q(x_{t-1} \mid x_t, x_0)$ | Variational ELBO over the Markov chain | Implicit, per transition | General discrete state spaces, any $Q_t$ |
| **SEDD** (2024) | Concrete score ratios $\tfrac{p_t(y)}{p_t(x)}$ | Score entropy (a Bregman divergence) | Via score parametrization | Any-order likelihood, principled scoring |
| **MDLM / LLaDA** (2024–25) | The clean token $x_0$ given masked $x_t$ | Weighted cross-entropy | $1/(1-\alpha_t)$ | Scalable LLM pretraining |

D3PM is the most general and the hardest to tune. SEDD is the most principled for *scoring* — its score-entropy loss generalizes continuous score matching to discrete data, and it gives a clean likelihood you can compare across orders. For the *absorbing* process, though, SEDD and MDLM provably coincide up to parametrization, and MDLM's cross-entropy form is both simpler to implement and friendlier to scale. That is why every large open diffusion LM in 2025 — LLaDA, Dream, and the production systems — trains the MDLM-style objective. Score entropy remains the better lens when you care about exact likelihoods or any-order ratios; the [any-order flexible-length masked diffusion](/blog/paper-reading/large-language-model/any-order-flexible-length-masked-diffusion) work is a good entry point there.

### 2.2 The training step in code

The objective is shorter as code than as prose. This is a complete, runnable training step for a masked diffusion LM:

```python
import torch
import torch.nn.functional as F

MASK_ID = vocab_size  # the appended absorbing token; embeddings sized vocab_size + 1

def masked_diffusion_loss(model, x0, eps: float = 1e-3):
    """One training step of a masked (absorbing-state) diffusion LM.

    model: a *bidirectional* transformer mapping (B, L) ids -> (B, L, vocab_size) logits.
    x0:    (B, L) clean token ids.  Returns a scalar loss.
    """
    B, L = x0.shape
    # 1. Sample a per-sequence noise level t ~ U(eps, 1).  Here t IS the mask prob.
    t = torch.rand(B, 1, device=x0.device).clamp_(eps, 1.0)
    # 2. Forward (corruption): mask each token independently with probability t.
    mask = torch.rand(B, L, device=x0.device) < t          # True where masked
    xt = torch.where(mask, MASK_ID, x0)
    # 3. One bidirectional forward pass predicts the clean token at every position.
    logits = model(xt)                                      # (B, L, vocab_size)
    # 4. Cross-entropy on masked positions only, reweighted by 1/t.
    ce = F.cross_entropy(logits.transpose(1, 2), x0, reduction="none")  # (B, L)
    ce = ce * mask                                          # zero out clean positions
    loss = (ce.sum(dim=1) / (t.squeeze(1) * L)).mean()      # 1/t weighting, per-token
    return loss
```

Four lines of actual math. Note what is *absent*: no causal mask, no shifted labels, no timestep embedding passed to `model` (more on that below), and no separate "noise prediction" head — the model's vocabulary logits *are* the prediction. If you have written an AR training loop, the differences are: the labels are the input itself (not shifted by one), the loss is gated to masked positions, and there is a $1/t$ reweighting. That is the whole change.

### 2.3 Second-order optimization: the variance of the estimator

The $1/t$ weight is unbiased but **high variance** for small $t$. When $t \approx 0.01$, you mask roughly one token in a hundred and divide its loss by $0.01$ — a single hard token can dominate the batch gradient. Practitioners clamp $t$ away from zero (the `eps` in the code), and some use **antithetic** or **low-discrepancy** sampling of $t$ across the batch (e.g., a stratified grid plus jitter) so each minibatch covers the noise range evenly instead of clustering by chance. MDLM also showed that you can switch to a *continuous-time* loss that integrates over $t$ analytically for the parts that admit it, further cutting variance. None of this changes the expected gradient; it changes how many steps you need to estimate it well, which is the difference between a model that trains in a week and one that thrashes.

### 2.4 A worked example, and the connection to BERT

Make the loss concrete. Suppose $L = 4$, the clean sequence is `[the, cat, sat, down]`, and we sample $t = 0.5$. The forward process masks each token with probability $0.5$; say it masks positions 2 and 4, giving $x_t = $ `[the, MASK, sat, MASK]`. The model does one forward pass and outputs, at position 2, a distribution that puts $0.6$ on `cat`, and at position 4, $0.3$ on `down`. The per-sequence loss contribution is

$$
\frac{1}{t}\Big( -\log 0.6 \;-\; \log 0.3 \Big) = \frac{1}{0.5}\,(0.51 + 1.20) = 3.42.
$$

Positions 1 and 3 (`the`, `sat`) contribute nothing — they are clean conditioning context. If instead we had sampled $t = 0.25$ and masked only position 2, the contribution would be $\frac{1}{0.25}\,(-\log p(\texttt{cat}))$ — the larger $1/t$ factor compensating for masking fewer tokens. Average enough of these over random $t$ and you recover an unbiased per-token objective.

If that procedure feels familiar, it should: **masked diffusion is masked language modeling — BERT's MLM — generalized to every masking rate and reweighted into a likelihood bound.** BERT masks a *fixed* ~15% of tokens and minimizes cross-entropy on them, which is essentially the $t \approx 0.15$ slice of the diffusion objective without the $1/t$ weight. The two differences are exactly what make diffusion generative: (1) training across *all* mask rates from near-0 to near-1, so the model can denoise a fully-masked sequence and not just a lightly-corrupted one; and (2) the $1/t$ reweighting that turns the collection of MLM losses into a proper bound on $\log p(x_0)$. Seen this way, a diffusion LM is the generative model that BERT's objective was always implicitly defining — the missing ingredient was training at high mask rates and running the reverse process to sample. This is also why you cannot take an off-the-shelf BERT and sample from it: it never saw mask rates above ~15%, so its predictions on a near-fully-masked canvas are nonsense.

### 2.5 The reverse process: what the model is allowed to change

The loss trains $p_\theta(x_0^i \mid x_t)$, but turning that into a sampler requires deciding what a reverse step does to *non-masked* positions. MDLM's parametrization — sometimes called the "carry-over" or zero-masking-probability trick — is the clean answer: in a reverse step, **already-clean tokens are copied through unchanged, and the model only ever predicts at masked positions.** Formally, the reverse transition is constrained so that $p_\theta(x_{t-\Delta}^i = x_t^i \mid x_t) = 1$ whenever $x_t^i \neq \mathbf{m}$ — a position that already holds a real token stays put.

Two consequences follow. First, the model's output distribution never needs to assign probability to the mask token itself — it predicts real tokens to *un-mask*, and a forward reverse step never re-introduces a mask — which is why the output head can stay at $|V|$ even though the input embedding is $|V|+1$. Second, the sampler's only freedom is *which* masked positions to commit and *what* to fill them with; the clean positions are frozen. This is exactly the structure the sampler in Section 4 exploits: it scores the predicted distributions at masked positions, commits the confident ones (turning them clean and therefore frozen), and leaves the rest masked for the next step. The low-confidence *re-masking* strategies are a sampler-side relaxation of this — they let a tentatively-committed token be re-masked when a later step's context makes it look wrong — but the underlying model parametrization is the clean carry-over above.

## 3. The architecture

**Senior rule of thumb: a diffusion LM is a vanilla bidirectional transformer with a mask token in its vocabulary — the only architectural change from GPT is deleting the causal mask.**

People expect diffusion LMs to need exotic machinery. They do not. Take a standard decoder-only transformer — RMSNorm, rotary embeddings, SwiGLU, grouped-query attention, the works — and make exactly two changes:

1. **Remove the causal attention mask.** Every position attends to every other position. This is the single defining change.
2. **Add the `[MASK]` token to the embedding table and the output head**, growing both from $|V|$ to $|V| + 1$ rows.

That is it. The output head still projects each position's hidden state to a categorical over $V$; you simply never need a logit for the mask token on the output side (the model predicts real tokens), so the head can stay at $|V|$ while the input embedding is $|V| + 1$.

![An autoregressive model multiplies attention by a lower-triangular mask so each query sees only keys up to its own position](/imgs/blogs/diffusion-language-models-architecture-training-finetuning-6.png)

The figure is the attention mask, drawn as the 0/1 matrix that multiplies the attention scores. An AR model uses the **lower-triangular** mask: query $q_i$ may attend to keys $k_0 \dots k_i$ and nothing later, which is what enforces the left-to-right factorization. A diffusion LM throws this matrix away — every entry is 1, every position attends to every other. That single change is why the KV cache, which the entire AR serving stack depends on, does not apply directly (Section 5), and it is also why a diffusion LM can use the *right* context of a token when predicting it.

### 3.1 The time-conditioning that turned out to be optional

Image diffusion models inject the timestep $t$ everywhere — sinusoidal embeddings added to every block, FiLM-style scale-and-shift on every norm — because the model genuinely needs to know "how noisy is this input?" to denoise correctly. A natural assumption is that diffusion LMs need the same. MDLM made the surprising and useful observation that for the absorbing process, **the model can infer the noise level from the input itself**: the fraction of `[MASK]` tokens in $x_t$ is a sufficient statistic for $t$, so a network with enough capacity learns the dependence implicitly. LLaDA dropped explicit time conditioning entirely and lost nothing. This matters practically because it means an off-the-shelf transformer implementation works unchanged — you do not need to thread a timestep through every layer, and you do not need the adaptive-norm plumbing that image UNets carry.

### 3.2 Bidirectional ≠ encoder-only-forever

A common confusion: "isn't this just BERT?" The architecture resembles a BERT encoder, but the *training objective and use* are completely different. BERT masks ~15% of tokens at a fixed rate and is trained for representation, never as a generative sampler. A diffusion LM masks at *every* rate from 0 to 100% (that uniform-over-$t$ sampling), and the loss is a likelihood bound, so the trained model defines a proper generative distribution you can sample from by running the reverse process. The encoder backbone is shared; the objective is what makes one a feature extractor and the other a generator. This is also why you cannot simply take a pretrained BERT and sample from it — it was never trained at high mask rates, so its predictions on a near-fully-masked sequence are garbage.

### 3.3 Handling sequence length

Autoregressive models decide their own length by emitting an end-of-sequence token and stopping. A diffusion LM has no natural stopping rule — it denoises a *fixed-size* canvas of positions, so the length has to come from somewhere. Three approaches are in use.

The simplest, used by LLaDA, is **fixed-length generation with padding**: pick a maximum generation length, initialize that many `[MASK]` positions, and let the model fill them — including emitting `[EOS]` and padding past the real answer. The model learns to place `[EOS]` where the content ends and pad the remainder, and you truncate at the first `[EOS]` post-hoc. It is simple but wasteful: you pay for the full canvas even when the answer is short, and the canvas size is itself a quality/cost tradeoff (too short truncates answers, too long burns compute on padding).

The second is **semi-autoregressive / block generation** (the block-diffusion structure from Section 5): generate one block at a time, deciding after each block whether to emit a stop signal and end or to allocate another block. This recovers AR's variable-length behavior while keeping intra-block parallelism, and it is how most practical long-form diffusion generation actually works.

The third, explored in any-order and flexible-length variants, lets the model **insert and delete** positions during denoising (edit-flow style), so the canvas size is itself dynamic. This is the most flexible and the least mature; the [any-order flexible-length masked diffusion](/blog/paper-reading/large-language-model/any-order-flexible-length-masked-diffusion) work is the reference point. For most deployments today, fixed-length for short answers and block generation for long ones is the pragmatic combination.

## 4. Sampling and decoding

**Senior rule of thumb: decoding is iterative denoising — predict all masked tokens, commit the most confident, re-mask the rest, repeat — and the number of steps is a latency-vs-quality dial you control at inference.**

Training gives you a model that, shown any partially-masked sequence, predicts the clean tokens at the masked slots. Sampling turns that into generation by running the reverse process: start from a fully-masked sequence and progressively un-mask it.

![Diffusion decoding predicts every masked token scores confidence commits the confident ones and re-masks the rest over K steps](/imgs/blogs/diffusion-language-models-architecture-training-finetuning-7.png)

The loop, one step at a time: take the current sequence $x_t$ (some positions clean, some masked); run one forward pass to get a predicted distribution at every masked position; **score the confidence** of each prediction (the max softmax probability, or its negative entropy); **commit** the highest-confidence predictions by writing those tokens in; **re-mask** everything else; and move to $x_{t-1}$ with fewer masks. After $K$ steps the sequence is fully un-masked and you return it. Each step is exactly one forward pass over the whole sequence — so total cost is $K$ passes, regardless of how many tokens you generated.

The decision of *which* tokens to commit each step is the heart of the sampler. The schedule says roughly $(\alpha_{t-1} - \alpha_t) \cdot L$ tokens should be un-masked per step, but *which* ones?

| Remasking strategy | How it picks tokens to commit | Tradeoff |
| --- | --- | --- |
| **Random** | Un-mask a random subset at the schedule's rate. | Simplest; ignores the model's own certainty, so it commits bad guesses early. Worst quality. |
| **Low-confidence remasking** | Keep the highest-confidence predictions; re-mask the rest. | LLaDA's default. Lets the model fill the easy slots first and defer hard ones. Best quality/step. |
| **Top-$k$ / entropy** | Commit the $k$ lowest-entropy positions per step. | Similar to low-confidence; entropy is a smoother signal than max-prob. |
| **Confidence-threshold (parallel)** | Commit *every* token whose confidence exceeds $\tau$, however many that is. | Fast-dLLM's trick — variable tokens per step, can finish in far fewer steps when the model is sure. |

The low-confidence strategy is the one that made diffusion LMs competitive. The intuition: a fully-masked sequence has many positions whose answer is nearly determined by the prompt (function words, obvious continuations) and a few that are genuinely hard. Committing the easy ones first turns the hard ones into a smaller, better-conditioned infilling problem on the next step. Random remasking throws that structure away.

### 4.1 The sampler in code

```python
import math

@torch.no_grad()
def generate(model, prompt_ids, gen_len: int = 256, steps: int = 64, temperature: float = 0.0):
    """Iterative denoising with low-confidence remasking (LLaDA-style)."""
    device = prompt_ids.device
    P = prompt_ids.shape[1]
    # Start: the prompt is clean conditioning; the generation region is all [MASK].
    x = torch.cat(
        [prompt_ids, torch.full((1, gen_len), MASK_ID, device=device)], dim=1
    )
    gen = slice(P, P + gen_len)

    for step in range(steps):
        if not (x[:, gen] == MASK_ID).any():
            break                                         # everything committed
        logits = model(x)[:, gen]                          # (1, gen_len, vocab)
        if temperature > 0:
            probs = (logits / temperature).softmax(-1)
            pred = torch.multinomial(probs[0], 1).squeeze(-1)[None]
            conf = probs[0].gather(-1, pred[0, :, None]).squeeze(-1)[None]
        else:
            conf, pred = logits.softmax(-1).max(-1)        # greedy + its confidence

        still_masked = x[:, gen] == MASK_ID
        n_mask = int(still_masked.sum())
        # Commit ~ this fraction of the remaining masks this step (linear schedule).
        n_keep = max(1, math.ceil(n_mask / (steps - step)))

        conf = conf.masked_fill(~still_masked, -1.0)       # never re-pick a clean slot
        keep = conf[0].topk(n_keep).indices                # highest-confidence masked slots
        region = x[:, gen].clone()
        region[0, keep] = pred[0, keep]
        x[:, gen] = region
    return x[:, gen]
```

A few things worth noticing. The prompt sits in the sequence as permanently-clean conditioning — it is never masked, so the model treats it as fixed context. The number of tokens committed per step (`n_keep`) follows a simple schedule; with `steps = gen_len` you commit one token per step (which recovers something close to AR behavior), and with `steps = 16` you commit ~16 tokens per step (aggressive parallelism). Temperature 0 is greedy; for diverse sampling you draw from the per-position categorical, and because positions are sampled *independently* within a step, **temperature matters more here than in AR** — a high temperature can make many positions simultaneously wrong, and there is no left-context to keep them coherent. Most diffusion LMs default to low temperature with confidence-based commitment.

### 4.2 Second-order optimization: the step-count cliff

The clean knob is "fewer steps = faster." The catch is that quality degrades *non-linearly* as you cut steps, and the cliff depends on the content. For text where each position is nearly determined (boilerplate code, formulaic prose) you can drop to 8–16 steps for hundreds of tokens with little loss. For text where positions are tightly coupled (math derivations, where token $i$ constrains token $i+5$) the same step count produces locally-plausible but globally-incoherent output, because committing many positions in one parallel step assumes they are conditionally independent given what is already un-masked — and they are not. The practical recipe is to **profile steps-vs-quality on your actual task** rather than trusting a single default, and to lean on confidence-threshold decoding so the model itself spends more steps where it is unsure.

### 4.3 Classifier-free guidance and conditioning strength

Diffusion LMs inherit one more tool from image diffusion: **classifier-free guidance (CFG)**. The idea is to sharpen the prompt's influence by extrapolating away from the unconditional prediction. At each denoising step you run the model twice — once with the prompt ($\ell_{\text{cond}}$) and once with the prompt dropped or masked ($\ell_{\text{uncond}}$) — and combine the logits:

$$
\ell_{\text{guided}} = \ell_{\text{uncond}} + w \,\big(\ell_{\text{cond}} - \ell_{\text{uncond}}\big),
$$

with a guidance scale $w \geq 1$. At $w = 1$ this is ordinary conditional decoding; larger $w$ pushes the sample toward tokens the prompt makes *more* likely relative to the unconditional model, which can improve instruction adherence on some tasks. The cost is a second forward pass per step (CFG roughly doubles decoding cost), and — the part that bites people — **CFG has a narrow sweet spot.** Push $w$ too high and the extrapolation leaves the region where the model's logits are calibrated; the per-position distributions become spiky and mutually inconsistent, and because positions are committed in parallel, you get locally-confident but globally-incoherent text — repetition, degenerate loops, or outright gibberish. The CFG-gibberish incident in the case studies is exactly this. The practical guidance: treat $w$ as a tuned hyperparameter in roughly $[1, 3]$, validate it on held-out prompts, and remember it interacts with temperature and the remasking threshold.

## 5. The KV-cache problem and block diffusion

**Senior rule of thumb: full bidirectional attention breaks the AR KV cache; block diffusion buys it back by making attention causal *between* blocks while staying bidirectional *within* a block.**

The single biggest practical obstacle to serving diffusion LMs is the [KV cache](/blog/machine-learning/large-language-model/kv-cache). In an AR model, the keys and values of all previous tokens are fixed once computed, so you cache them and each new token costs one cheap incremental attention. That is the entire reason AR decoding is affordable. In a diffusion LM with full bidirectional attention, every denoising step re-attends over the *entire* sequence, and because masked positions change every step, their keys and values change too — so there is nothing stable to cache. Naively, a $K$-step diffusion decode does $K$ full-sequence forward passes with no reuse.

![Full bidirectional attention recomputes all keys and values every step while block-causal attention caches finalized blocks and reuses them](/imgs/blogs/diffusion-language-models-architecture-training-finetuning-8.png)

The figure lays out the two regimes. Split the sequence into blocks. With **full bidirectional** attention (MDLM as originally trained), there is no causal structure, so you recompute all KV every step and get no speedup from caching. With **block-causal** attention — the [BD3LM](https://arxiv.org/abs/2503.09573) (Block Discrete Denoising Diffusion LM, Arriola et al., 2025) construction — tokens attend *bidirectionally within their own block* but *causally across blocks*: block $i$ sees the finalized blocks $0 \dots i-1$ and itself, but not future blocks. Now you generate block by block (semi-autoregressively), and once a block is finalized its keys and values are frozen — exactly the condition the KV cache needs. Each diffusion step only recomputes the *active* block while reusing the cached prefix.

Block diffusion is a knob, not a binary. With block size = the whole sequence you recover pure diffusion (maximum parallelism, no cache). With block size = 1 you recover autoregression (full cache, no parallelism). In between you trade parallel decoding for cache reuse. This is also exactly the structure that [DFlash](/blog/machine-learning/large-language-model/dflash-block-diffusion-speculative-decoding) exploits to build a parallel *drafter* for speculative decoding — a block-diffusion model proposes a whole block of draft tokens in one pass, which a standard AR target then verifies.

### 5.1 Fast-dLLM: caching without retraining

You do not always have to retrain for block-causal attention. [Fast-dLLM](https://arxiv.org/abs/2505.22618) (2025) is a *training-free* acceleration for existing diffusion LMs. It does two things. First, an **approximate KV cache**: it caches the keys and values of the already-decoded prefix and refreshes them only periodically, accepting a small approximation error because finalized tokens' representations barely move. Second, **confidence-aware parallel decoding**: instead of committing a fixed number of tokens per step, it commits *every* token whose confidence clears a threshold $\tau$, which lets confident regions finish in one step. Together these report 20–30× throughput improvements over naive diffusion decoding on LLaDA and Dream, with negligible quality change, and they require no weight updates — they are inference-time wrappers.

### 5.2 The throughput arithmetic

It is worth doing the back-of-envelope that decides whether diffusion's parallelism is a real win on your workload. Let $L$ be the output length, and assume a forward pass over a length-$L$ sequence costs roughly one "pass-unit" whether you are an AR model decoding incrementally with a KV cache or a diffusion model attending to the whole canvas. (This understates AR's per-token efficiency, but it is the right first cut.)

An AR model emits one token per pass, so generating $L$ tokens costs $L$ *sequential* pass-units. A diffusion model costs $K$ pass-units, where $K$ is the step budget, and each pass is *parallel* across positions. The crude comparison is $K$ versus $L$:

| Output length $L$ | AR sequential passes | Diffusion passes ($K$) | Pass-count ratio |
| --- | --- | --- | --- |
| 128 | 128 | 32 | 4× |
| 512 | 512 | 64 | 8× |
| 2048 | 2048 | 128 | 16× |

The win only materializes when $K \ll L$, which requires the content to be parallel-decodable — confident at many positions per step. The diffusion pass is also more expensive than an AR incremental step (it attends over the full canvas without the cache shortcut), which is exactly what block diffusion and Fast-dLLM exist to fix. The honest summary: the *latency* ceiling for structured single-stream generation is genuinely lower for diffusion, but *throughput* at high batch sizes is dominated by the same FLOPs as AR, so the win is largest exactly where AR hurts most — interactive, low-concurrency, structured output. This is the same shape the Mercury Coder case study shows in production.

### 5.3 Second-order optimization: the threshold is a footgun

The confidence threshold $\tau$ that makes parallel decoding fast is also the thing most likely to make it slow or wrong. Set $\tau$ too high and almost no tokens clear it per step, so you fall back to roughly one-token-per-step — *slower* than a well-tuned fixed schedule, because you also pay the cache-refresh overhead. Set $\tau$ too low and the model commits unsure tokens that it cannot revise (committed tokens are frozen), and quality collapses. The threshold interacts with temperature (which rescales confidence) and with the block size, so it has to be tuned jointly, on the target workload, not copied from a paper's table.

## 6. Reusing autoregressive weights (AR → diffusion conversion)

Training an 8B diffusion LM from scratch costs as much as training an 8B AR model from scratch. A cheaper path, **AR-to-diffusion conversion** (sometimes "a2d"), bootstraps a diffusion LM from a *pretrained AR checkpoint*. The insight is that the bulk of what an AR model learned — token embeddings, world knowledge, the shape of the representation space — transfers; what has to change is the attention pattern and the objective.

The recipe: initialize the diffusion model from the AR weights, add the mask token's embedding, and continue training under the MDLM loss while **annealing the attention mask** from causal toward full bidirectional. Early in conversion the model attends mostly causally (close to its pretrained behavior) and is gradually allowed to attend to the right context as the loss adapts. [Dream-7B](https://arxiv.org/abs/2508.15487) (2025) used this to convert Qwen2.5-7B into a strong diffusion LM at a fraction of from-scratch cost, and the approach is the standard way to get a competitive diffusion model without a full pretraining budget.

The economics are the whole point. A from-scratch 7–8B diffusion LM is a multi-hundred-thousand-GPU-hour pretraining run; conversion from a strong AR checkpoint reaches comparable quality in a small fraction of that, because token embeddings, factual knowledge, and most of the representation geometry transfer directly — only the attention pattern and the readout have to adapt. What does *not* transfer for free is the model's calibration at high mask rates: the AR checkpoint has never seen a mostly-masked input, so early in conversion its predictions on heavily-corrupted sequences are poor and the MDLM loss has to teach that regime from scratch. This is why conversion still needs a non-trivial continued-pretraining budget — tens of billions of tokens, not trillions — and why the attention-mask anneal must be gradual: flip to full bidirectional attention too fast and you destroy the causal representations before the model has learned to denoise without them. The failure mode — annealing too fast — is in the case studies below, because it is a real and expensive mistake.

## 7. Finetuning diffusion language models

**Senior rule of thumb: SFT a diffusion LM exactly like pretraining, with one rule — never mask the prompt; corrupt and compute the loss only on the response.**

A pretrained diffusion LM is an unconditional (or prefix-conditioned) model of text. To make it follow instructions, you SFT it on (prompt, response) pairs. The only thing that changes from the pretraining loss is *what is eligible to be masked*. If you let the forward process mask prompt tokens too, the model learns to *reconstruct the prompt* as part of its job — and at inference, where the prompt is given and clean, that capacity is wasted at best and, at worst, the model hallucinates a different prompt and answers *that*. The fix is to freeze the prompt as clean conditioning and only diffuse the response.

![Supervised finetuning keeps prompt tokens clean as conditioning and computes the diffusion loss only on the corrupted response tokens](/imgs/blogs/diffusion-language-models-architecture-training-finetuning-9.png)

The figure shows the layout: the prompt region ("What is RAG?") is never masked and contributes no gradient — it is pure conditioning — while the response region is corrupted by the usual forward process and the loss is computed only on its masked positions. This turns the unconditional model $p_\theta(x)$ into a conditional one $p_\theta(\text{response} \mid \text{prompt})$, which is exactly instruction following. Everything else — the bidirectional attention, the $1/t$ reweighting, the sampling loop — is unchanged.

### 7.1 SFT in code

```python
def sft_loss(model, prompt_ids, response_ids, eps: float = 1e-3):
    """SFT for a diffusion LM: freeze the prompt, diffuse only the response."""
    x0 = torch.cat([prompt_ids, response_ids], dim=1)        # (B, P + R)
    B, L = x0.shape
    P = prompt_ids.shape[1]
    is_resp = torch.zeros(B, L, dtype=torch.bool, device=x0.device)
    is_resp[:, P:] = True                                    # the loss region

    t = torch.rand(B, 1, device=x0.device).clamp_(eps, 1.0)
    # Only response tokens are eligible for masking; the prompt stays clean.
    mask = (torch.rand(B, L, device=x0.device) < t) & is_resp
    xt = torch.where(mask, MASK_ID, x0)

    logits = model(xt)                                       # full bidirectional attn
    ce = F.cross_entropy(logits.transpose(1, 2), x0, reduction="none")
    ce = ce * mask                                           # loss on masked response only
    return (ce.sum(1) / (t.squeeze(1) * is_resp.sum(1))).mean()
```

The diff from pretraining is two lines: build an `is_resp` region mask and `& is_resp` it into the corruption mask. **LoRA** works directly on this — the bidirectional transformer is still a stack of linear layers, so you attach low-rank adapters to the attention and MLP projections exactly as you would for an AR model. The general principles in [effective LLM finetuning techniques](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques) carry over; the only diffusion-specific subtlety is the prompt-masking rule above. A second subtlety: pad/format with a consistent chat template *before* deciding the response region, or you will accidentally mask special tokens that the model needs as clean anchors.

### 7.2 Reinforcement learning: the log-prob problem

RL post-training — the [GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) family that powers reasoning models — needs one thing from the policy: the **log-probability of a sampled sequence**, so it can push probability toward high-reward completions. For an AR model this is trivial: $\log p(y) = \sum_i \log p(y_i \mid y_{\lt i})$, computed in one teacher-forced pass. For a diffusion LM it is *intractable*: the model defines $p(y)$ through a sum over all masking orders, and there is no cheap exact sequence log-prob.

![diffu-GRPO samples completions by diffusion scores them normalizes advantages and updates the policy with a mean-field log-prob estimate](/imgs/blogs/diffusion-language-models-architecture-training-finetuning-10.png)

[diffu-GRPO](https://arxiv.org/abs/2504.12216), introduced in the **d1** paper (Zhao et al., 2025), solves this with a **mean-field, one-step estimate**. Instead of the true sequence log-prob, it masks the completion at some rate, does *one* forward pass, and reads off the per-token log-probabilities of the (now-masked) tokens — treating positions as approximately independent given the rest. Averaging this over a few random masks gives a low-cost, low-variance surrogate log-prob that is good enough to drive the policy gradient. The rest of the pipeline is standard GRPO: sample a group of $G$ completions per prompt by diffusion decoding, score each with a verifier or rule-based reward, normalize advantages within the group, and update. d1 stacks this on top of SFT to scale reasoning in diffusion LLMs; the [d1 paper deep dive](/blog/paper-reading/large-language-model/d1-scaling-reasoning-in-diffusion-large-language-models-via-reinforcement-learning) traces the full recipe and its results on LLaDA-8B.

### 7.3 The log-prob estimate in code

```python
def diffu_logprob(model, seq, n_mask_samples: int = 4, mask_rate: float = 0.5):
    """Mean-field per-token log-prob estimate used by diffu-GRPO.

    Average one-step denoising log-probs over a few random masks of the sequence.
    Returns (B, L) per-position log-probs to feed the GRPO objective.
    """
    B, L = seq.shape
    total = torch.zeros(B, L, device=seq.device)
    counts = torch.zeros(B, L, device=seq.device)
    for _ in range(n_mask_samples):
        mask = torch.rand(B, L, device=seq.device) < mask_rate
        xt = torch.where(mask, MASK_ID, seq)
        logits = model(xt)                                   # one forward pass
        lp = logits.log_softmax(-1).gather(-1, seq[..., None]).squeeze(-1)
        total += lp * mask                                   # only score masked slots
        counts += mask
    return total / counts.clamp_min(1.0)                     # average per position
```

The `mask_rate` and `n_mask_samples` are the variance/cost knobs: more samples and a moderate mask rate give a smoother estimate at higher cost. d1 also found that **random prompt masking** during RL acts as a useful regularizer — masking part of the prompt on some rollouts prevents the policy from overfitting to surface prompt features and stabilizes training. The whole thing is one of those cases where the "diffusion-specific" part is small (estimating log-probs) and the surrounding RL machinery is unchanged from the AR recipe.

### 7.4 Preference optimization for diffusion LMs

GRPO is not the only post-training route. Preference optimization — [DPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) and its relatives — also adapts to diffusion LMs and sidesteps the rollout cost of online RL. DPO needs the log-probability ratio between the policy and a frozen reference model on a *given* (chosen, rejected) pair: $\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$. The only diffusion-specific piece is, again, estimating those sequence log-probs — and you reuse the same mean-field, masked-forward-pass estimator from Section 7.3 for both the policy and the reference.

Because DPO scores fixed responses rather than sampling new ones, the estimator's variance matters less than in diffu-GRPO — there is no rollout feedback loop to destabilize — so DPO is often the *easier* first preference-tuning method to get working on a diffusion LM. The catch is the familiar one: DPO optimizes a proxy for the reward and can over-fit to the preference dataset's quirks, so it is best used as a lightweight alignment pass rather than a substitute for verifier-grounded RL on tasks with checkable answers. A common stack is SFT, then a DPO pass for general helpfulness, then diffu-GRPO on the subset of tasks where a programmatic reward exists — exactly the progression the [d1 work](/blog/paper-reading/large-language-model/d1-scaling-reasoning-in-diffusion-large-language-models-via-reinforcement-learning) uses for reasoning.

## Case studies from production

These are the incidents — some milestones, some failures — that teach how diffusion LMs behave when they leave the notebook. Versions, sizes, and mechanisms are concrete because the lessons live in the details.

### 1. LLaDA-8B and the "diffusion can't scale" assumption

For years the standing wisdom was that discrete diffusion would never close the gap to AR at scale — the early D3PM and SEDD models trailed same-size AR baselines on perplexity and downstream tasks, and the gap looked structural. LLaDA-8B was the existence proof that it is not. Trained from scratch on 2.3T tokens with the masked-diffusion objective, an 8B bidirectional transformer, no time conditioning, and low-confidence remasking at inference, it matched a LLaMA3-8B-class AR baseline across a broad benchmark suite and *beat* it on some reversal/infilling tasks where bidirectional context helps. The lesson is not "diffusion wins" — it does not, uniformly — but that the paradigm scales, and the earlier gap was about *immature objectives and recipes*, not a ceiling. Every architecture decision in this post (vanilla transformer, no time embedding, MDLM loss) is downstream of what LLaDA showed you can get away with.

### 2. The prompt that leaked through the mask

A team finetuning a diffusion LM for a QA assistant reused their pretraining data pipeline, which masked tokens uniformly across the whole sequence. SFT loss dropped nicely, but at inference the model would sometimes answer a *different* question than the one asked — paraphrasing or mutating the prompt before responding. The wrong first hypothesis was a decoding bug. The actual root cause: because the prompt tokens had been eligible for masking during SFT, the model had learned to treat the prompt as *something to reconstruct*, not as fixed conditioning, so during parallel decoding it would occasionally "denoise" a prompt token into a different word and then answer the mutated prompt. The fix was the prompt-masking rule from Section 7: build a response-region mask and only corrupt response tokens. Loss curves looked nearly identical before and after; only the behavior changed. This is the most common diffusion-SFT bug, and it is invisible in the training metrics.

### 3. The block-size train/inference mismatch

A group adopted block diffusion (BD3LM-style) to get KV caching in serving. They trained with a block size of 32 but, to hit a latency target, configured inference with a block size of 8. Outputs were locally fluent but fell apart at block boundaries — repeated phrases, dropped clauses, an answer that restarted mid-sentence. The symptom looked like a sampling-temperature problem; it was not. The model's block-causal attention had learned the statistics of 32-token blocks (how much intra-block context to expect, where boundaries fall), and running it at block 8 put it permanently off-distribution at every boundary. The general rule, which mirrors what [DFlash](/blog/machine-learning/large-language-model/dflash-block-diffusion-speculative-decoding) found for drafters: a model trained at a larger block size usually generalizes *down* to smaller blocks at inference, but a model trained at a small block size cannot scale *up*. They retrained at block 8 (or, equivalently, could have trained at block 32 and inferred at 32) and the boundaries healed.

### 4. The confidence-threshold stall

After deploying Fast-dLLM-style confidence-threshold parallel decoding, an inference team found their p99 latency had gotten *worse* than the fixed-step baseline on a fraction of requests. Those requests turned out to be high-entropy generations (open-ended creative prompts) where the model was rarely confident enough to clear the threshold $\tau = 0.9$. With almost no tokens committed per step, decoding crawled to nearly one-token-per-step *and* paid the periodic KV-cache-refresh overhead on top — strictly worse than the schedule it replaced. The fix was an adaptive threshold: start at a high $\tau$ and decay it if too few tokens commit in a step, with a hard floor on tokens-per-step so the worst case degrades to the fixed schedule rather than below it. The lesson: confidence-threshold decoding is a *throughput* optimization for confident workloads and needs a fallback for the rest.

### 5. The remasking-strategy quality cliff

A reproduction of a diffusion-LM result came out far worse than the paper, with no obvious bug — same weights, same step count, same prompts. The difference was the remasking strategy: the reproduction used *random* remasking (commit a random subset each step) while the paper used *low-confidence* remasking (commit the most confident, defer the rest). On a 256-token generation at 64 steps, random remasking commits guesses on hard positions early, when the model has little context to constrain them, and those early mistakes are frozen for the rest of the decode. Switching to low-confidence remasking — three lines of code — recovered the paper's numbers. The general principle: in diffusion decoding, *order of commitment is a quality decision*, not just a scheduling detail, and "commit what you're sure of first" is almost always right.

### 6. The diffu-GRPO reward collapse

An RL run with diffu-GRPO on a math-reasoning task collapsed after a few hundred steps: rewards spiked briefly, then the policy degenerated into short, repetitive outputs that gamed the reward. The first suspect was the reward function; the actual driver was the **variance of the mean-field log-prob estimate**. With too few mask samples (`n_mask_samples = 1`) and an aggressive mask rate, the surrogate log-prob was noisy enough that the policy gradient occasionally pointed the wrong way, and once the policy started shortening outputs (which made the estimate easier and lower-variance) the dynamics reinforced it. Raising the number of mask samples, lowering the learning rate, and adding a length-normalized reward term stabilized it — along with the random-prompt-masking regularizer that d1 recommends. RL on diffusion LMs is more sensitive to log-prob estimator variance than AR RL is to anything analogous, because the AR log-prob is exact and the diffusion one is not.

### 7. Dream-7B and the attention-mask anneal done right (and wrong)

Dream-7B converted Qwen2.5-7B into a diffusion LM via AR-to-diffusion adaptation. The make-or-break detail was the **schedule for annealing the attention mask** from causal to full bidirectional. An early internal attempt (per the broader a2d folklore) flipped to full bidirectional attention immediately at the start of conversion; the model's loss spiked and never fully recovered, because suddenly allowing every token to attend to the future invalidated the causal representations the AR pretraining had built — a form of catastrophic forgetting localized to the attention pattern. The working recipe anneals gradually: keep attention mostly causal early, widen the bidirectional window over many steps, and let the MDLM loss pull the representations along. Done right, conversion reaches near-from-scratch quality at a fraction of the compute. The lesson is that the *thing you are changing* (the attention structure) is exactly where the pretrained model's assumptions are baked in, so you change it slowly.

### 8. The time-conditioning that wasn't needed

A team porting an image-diffusion mindset to text built elaborate timestep conditioning into their diffusion LM — sinusoidal $t$ embeddings, adaptive layer norm on every block, the full UNet-style apparatus. It trained fine but was slower and more memory-hungry than necessary, and ablations showed the time conditioning contributed *nothing* to quality. The reason, from Section 3.1: for the absorbing process the mask *fraction* in the input is a sufficient statistic for the noise level, so the network infers $t$ from the input and the explicit conditioning is redundant. Ripping out the time-embedding plumbing simplified the model, sped up training, and let them use a stock transformer implementation. The lesson is that intuitions from continuous image diffusion do not transfer wholesale — the discrete masked process is genuinely simpler, and importing complexity costs without buying anything.

### 9. Mercury Coder and the parallel-decode latency win

[Mercury](https://arxiv.org/abs/2506.17298) (Inception Labs, 2025) is a commercial diffusion LM family whose code model, Mercury Coder, reported throughput on the order of 1000+ tokens/second on H100s — multiples faster than comparable-quality AR code models — by leaning on the fact that code is full of high-confidence, locally-determined tokens (brackets, keywords, boilerplate) that parallel decoding commits many-per-step. The instructive part is *where the win is largest*: at low concurrency and for structured, predictable output, the parallel decode is a big win; at high batch sizes the hardware is already compute-bound from batching, so the per-request parallelism matters less, and for highly unpredictable output the step count climbs back up. The same shape shows up in Gemini Diffusion's public demos and in [Seed Diffusion](/blog/paper-reading/large-language-model/seed-diffusion-a-large-scale-diffusion-language-model-with-high-speed-inference). The takeaway for capacity planning: diffusion LMs shine on latency for structured single-stream generation, and that is exactly the regime (interactive coding, low-latency tool use) where they are being deployed first.

### 10. The likelihood-evaluation trap

A team tried to compare their diffusion LM to an AR baseline by perplexity and concluded the diffusion model was much worse — its reported "perplexity" was far higher. The comparison was meaningless. A diffusion LM's training loss is an *upper bound* (the negative ELBO) on the negative log-likelihood, not the exact NLL, and it is computed by Monte-Carlo over masking levels, so the number is both an over-estimate and noisy. Comparing it directly to an AR model's exact per-token perplexity is apples to oranges. The right comparisons are either (a) downstream task accuracy under matched decoding budgets, or (b) a carefully estimated likelihood using many Monte-Carlo samples and the same any-order convention for both models (which SEDD's framing makes precise). The lesson, and it bites every team once: **do not rank a diffusion LM against an AR model by raw perplexity** — pick a metric that is defined the same way for both.

### 11. The classifier-free-guidance gibberish

A team enabled CFG to improve instruction adherence and, after modest gains at $w = 2$, cranked the guidance scale to $w = 6$. Output quality fell off a cliff: responses became repetitive, dropped into degenerate loops, or produced word salad that was locally plausible and globally meaningless. The first hypothesis was a decoding bug or a corrupted checkpoint. The actual cause was CFG extrapolation past the calibrated regime (Section 4.3): at $w = 6$ the guided logits $\ell_{\text{uncond}} + 6(\ell_{\text{cond}} - \ell_{\text{uncond}})$ became extremely spiky, and because the diffusion sampler commits many positions in parallel per step, many positions simultaneously locked onto over-sharpened, mutually-inconsistent tokens with no left-context to keep them coherent. Lowering $w$ to $2.5$ and validating on a held-out set restored quality. The lesson: CFG is a tuned knob with a narrow useful range, and its failure mode in diffusion is *worse* than in AR, because parallel commitment removes the sequential coherence that would otherwise paper over a few over-confident tokens.

### 12. The cosine-vs-linear scheduler regression

A model that trained well under a linear schedule regressed silently after a teammate switched to cosine "to match the image-diffusion papers," without re-tuning anything else. Training loss looked fine — slightly lower, even — but downstream accuracy dropped a few points, and nobody noticed for two weeks because the loss curve was healthy. The root cause: the cosine schedule changes the distribution of mask rates the model trains on (more low-corruption steps), which shifts where the model is most accurate, while the *inference* step schedule and remasking thresholds had been tuned for the linear-schedule model. Train-time and inference-time assumptions were now mismatched. The fix was to either revert to linear or re-tune the inference schedule for the cosine model. The lesson is twofold: schedule changes are not free even when the loss says they are, and a diffusion LM's quality depends on *train/inference schedule consistency*, not the schedule in isolation — exactly parallel to the block-size mismatch in case study 3.

### 13. The 4-bit + LoRA divergence

Finetuning a diffusion LM with 4-bit quantization plus LoRA — the standard QLoRA recipe that works reliably for AR models — diverged within a few hundred steps, loss exploding to NaN. The same data and hyperparameters were stable in bf16. The wrong first guess was the learning rate. The actual issue was an interaction between the masked-diffusion loss's $1/t$ reweighting and 4-bit quantization noise: when a small-$t$ batch masks few tokens, the $1/t$ factor amplifies both the legitimate gradient *and* the quantization error in the logits, and the amplified noise occasionally produced a gradient large enough to destabilize the low-rank adapters. The fixes that worked were clamping $t$ further from zero (raising `eps`), capping the per-token loss weight, and lowering the LoRA learning rate. The general lesson: recipes ported from AR finetuning are *mostly* transferable, but the diffusion loss carries a high-variance reweighting term that interacts badly with anything that adds noise to the logits — quantization, aggressive dropout, a too-high learning rate — so the numerically-sensitive knobs need re-tuning, not copying.

## When to reach for a diffusion LM — and when not to

![Latency infilling needs output length and tooling maturity decide between an autoregressive model a diffusion model or a block-diffusion hybrid](/imgs/blogs/diffusion-language-models-architecture-training-finetuning-11.png)

The decision tree above compresses the tradeoffs. The honest summary: diffusion LMs are not a drop-in replacement for AR everywhere, but there are regimes where they clearly win.

**Reach for a diffusion LM when:**

- **Latency on structured, single-stream output matters** and the output has many high-confidence positions (code, templated text, structured data). Parallel decoding turns that predictability into a real wall-clock win.
- **You need infilling, editing, or constraint satisfaction.** Filling a hole given both left and right context, or generating text that must satisfy a downstream constraint, is native to a bidirectional any-order model and awkward for AR.
- **The task benefits from iterative refinement.** When an early token mistake is expensive and revisable refinement helps (some planning and reasoning tasks), the diffusion decoder's ability to re-mask and re-predict is an advantage.
- **You want any-order generation** — the same model handles left-to-right, right-to-left, and arbitrary-order filling without retraining.

**Skip it (stick with autoregression) when:**

- **You need maximum open-ended quality and the most mature tooling.** AR has a five-year head start on serving infrastructure, quantization, speculative decoding, and RLHF recipes; for a general chat assistant on commodity infra, AR is still the safer default.
- **Outputs are long and unpredictable.** When few positions are confident, the step count climbs toward the sequence length and the parallelism advantage evaporates.
- **You are serving at very high concurrency**, where batching already saturates the hardware and per-request parallel decoding buys little.
- **Your team cannot absorb the operational novelty.** The KV-cache story, the step-count tuning, the likelihood-evaluation caveats — these are real costs in engineering time that a mature AR stack does not impose.

The middle path is **block diffusion**: tune the block size to trade parallel decoding against KV-cache reuse, and you get a hybrid that is faster than AR on structured output while keeping a cache for long generations. For most teams evaluating diffusion LMs today, a block-diffusion model or a training-free accelerator like Fast-dLLM on top of an existing diffusion checkpoint is the pragmatic entry point — it lets you measure the latency win on your own workload before committing to a from-scratch training run.

A closing word on where the paradigm is heading. The 2021–2024 question — *can discrete diffusion match autoregression at all?* — has been answered: at the 7–8B scale it can, and on infilling and parallel-decode latency it can win outright. The open 2026 questions are operational, not existential: how far block size and confidence thresholds can be pushed before quality breaks, whether RL on diffusion LMs closes the reasoning gap that AR models opened with verifier-grounded training, and whether the serving stack matures enough to make the latency advantage routine rather than artisanal. None of those require a new objective or architecture — they are engineering on the foundation this post laid out. If you understand the masked loss, the bidirectional backbone, the iterative sampler, and the block-causal cache, you understand the machine; the rest is tuning it for a workload.

## Further reading

- [D3PM — Structured denoising diffusion models in discrete state-spaces](https://arxiv.org/abs/2107.03006) (Austin et al., 2021): the discrete-diffusion foundation.
- [SEDD — Discrete diffusion modeling by estimating ratios of the data distribution](https://arxiv.org/abs/2310.16834) (Lou et al., 2024): the score-entropy objective.
- [MDLM — Simple and effective masked diffusion language models](https://arxiv.org/abs/2406.07524) (Sahoo et al., 2024): the simplification that made scaling tractable.
- [LLaDA — Large language diffusion models](https://arxiv.org/abs/2502.09992) (Nie et al., 2025): the 8B from-scratch proof point.
- [BD3LM — Block discrete denoising diffusion language models](https://arxiv.org/abs/2503.09573) (Arriola et al., 2025): block-causal attention and the KV cache.
- [Fast-dLLM](https://arxiv.org/abs/2505.22618) (2025): training-free KV caching and confidence-aware parallel decoding.
- [d1 / diffu-GRPO](https://arxiv.org/abs/2504.12216) (Zhao et al., 2025): reinforcement learning for diffusion LLMs — see the [deep dive](/blog/paper-reading/large-language-model/d1-scaling-reasoning-in-diffusion-large-language-models-via-reinforcement-learning).
- Sibling posts on this blog: the [dllm library deep dive](/blog/machine-learning/open-source-library/dllm-diffusion-language-models-deep-dive), [DFlash block-diffusion speculative decoding](/blog/machine-learning/large-language-model/dflash-block-diffusion-speculative-decoding), and [any-order flexible-length masked diffusion](/blog/paper-reading/large-language-model/any-order-flexible-length-masked-diffusion).

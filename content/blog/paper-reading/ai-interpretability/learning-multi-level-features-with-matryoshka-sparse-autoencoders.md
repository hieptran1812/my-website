---
title: "Matryoshka Sparse Autoencoders: Nesting Dictionaries So General Features Survive Scale"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How training several nested SAE dictionaries at once stops sparsity from splitting and absorbing high-level features, with the loss, a worked toy example, and the Gemma-2-2B results."
tags:
  - paper-reading
  - matryoshka
  - sparse-autoencoders
  - multi-level-features
  - feature-absorption
  - mechanistic-interpretability
  - batchtopk
  - icml-2025-poster
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 18
paper:
  title: "Learning Multi-Level Features with Matryoshka Sparse Autoencoders"
  authors: "Bart Bussmann, Noa Nabeshima, Adam Karvonen, Neel Nanda"
  venue: "ICML 2025 (arXiv 2503.17547)"
  url: "https://arxiv.org/abs/2503.17547"
---

> [!tldr]
> **Claim.** A sparse autoencoder (SAE) gets *worse* at its real job as you make it bigger, because the sparsity objective rewards folding general concepts into specific ones. Matryoshka SAEs fix this by making the first few thousand latents reconstruct the input on their own, then the first few more, and so on, all inside one model.
>
> **Why it matters.** You can train one large SAE and still find the coarse features ("female names", "starts with E") intact, instead of training a sweep of sizes and hoping one of them is right.
>
> **Most surprising result.** On Gemma-2-2B at 40 active latents per token, the feature absorption rate drops from 0.49 (BatchTopK) to 0.05, and a first-letter concept needs about one latent instead of three. The only change is the loss.
>
> **Where it fails.** Reconstruction gets slightly worse (70% vs 72% variance explained at L0 = 40), training costs roughly 50% more with five nested groups, and on a chess model with a small hidden size it degrades at high L0.

Sparse autoencoders were supposed to be the microscope for language models. The awkward discovery of 2024 was that the microscope gets blurrier as you add magnification. Bigger SAEs reach lower training loss, but on downstream tasks like probing and concept erasure they often get worse. This paper by Bussmann, Nabeshima, Karvonen and Nanda names the culprit as a mismatch between a *flat* sparsity objective and the *hierarchical* structure of real features, then fixes it with a dozen lines of PyTorch.

## 1. The problem: sparsity eats the general features

### 1.1 What an SAE is

The paper starts from the standard SAE, which is an encoder followed by a linear decoder:

$$
f(x) = \sigma\left(W_{\text{enc}}\, x + b_{\text{enc}}\right), \qquad \hat{x} = W_{\text{dec}}\, f(x) + b_{\text{dec}}
$$

Here $x \in \mathbb{R}^{n}$ is one activation vector from the model (for Gemma-2-2B's residual stream, $n$ is the model width). $W_{\text{enc}} \in \mathbb{R}^{m \times n}$ maps it into $m$ latents, with $m$ much larger than $n$. The activation $\sigma$ (ReLU, TopK, JumpReLU, BatchTopK) keeps $f(x)$ non-negative and mostly zero. $W_{\text{dec}} \in \mathbb{R}^{n \times m}$ maps the sparse code back, and each of its $m$ columns is the "direction" that latent writes.

Training minimizes reconstruction error while keeping few latents active. That count of active latents per token is called **L0**, and it is the sparsity knob every result in this paper is plotted against.

### 1.2 Three pathologies of flat sparsity

The paper lists three ways the sparsity pressure distorts the dictionary. All three get worse as $m$ grows.

| Pathology | What happens | Paper's example |
| --- | --- | --- |
| **Feature splitting** | A general concept fragments into specialized latents; the general latent disappears | "punctuation" becomes separate period, comma, question-mark latents |
| **Feature absorption** | A general latent survives but develops holes where a specific latent took over | a "starts with E" latent that skips "Elephant" |
| **Feature composition** | Two independent concepts get merged into one combo latent | "red triangle" instead of "red" + "triangle" |

The splitting row cites a striking finding from Leask et al. (2025): when you double an SAE's dictionary, only about one third of the latents in the larger SAE represent new concepts; the rest re-express old ones more sparsely.

### 1.3 Why sparsity *wants* absorption

The intuition is an accounting argument. Think of L0 as a budget of how many latents you are allowed to light up per token. If "Lily" always implies "female name", there are two ways to reconstruct it:

- **Clean:** light the "female name" latent *and* a "Lily, minus the female part" latent. Cost: 2.
- **Absorbed:** light one latent whose decoder direction is "female name + Lily". Cost: 1.

Both reconstruct "Lily" perfectly. The sparsity penalty only sees the cost, so it picks the absorbed solution. The "female name" latent now only needs to fire on the other female names. That is precisely a hole.

![Before/after: under flat sparsity the Lily token fires one absorbed latent and leaves a hole in the female-names latent; under Matryoshka nesting the early-group parent stays on and a later latent adds the Lily refinement](/imgs/blogs/learning-multi-level-features-with-matryoshka-sparse-autoencoders-1.webp)

The figure above is the whole paper in one picture. Flat sparsity saves one active latent by folding the parent into the child. The Matryoshka objective makes that trade unprofitable, and the rest of this post is about why.

### 1.4 It shows up in real models

The paper's TinyStories study makes this concrete. The authors train reference SAEs of 30, 100, 300, 1k, 3k and 10k latents on a 4-layer Transformer (hidden size 768) and track how a latent's role changes as the dictionary grows.

![Figure 4 from Bussmann et al. (2025): a general "female words" latent in a 300-latent SAE develops holes in a 1000-latent SAE as specialized Lily and Sue latents absorb those cases (red circles)](/imgs/blogs/learning-multi-level-features-with-matryoshka-sparse-autoencoders-fig3.webp)

In the 300-latent SAE, latent S/2/65 fires on "she", "her", "girl", "Lily" and "Sue". In the 1000-latent SAE, its counterpart S/3/66 stops firing on "Lily" because S/3/359 took that case, and S/3/861 took "Sue". A 25k-latent Matryoshka SAE at the same average sparsity keeps a female-tokens latent without holes *and* still develops Lily and Sue child latents.

## 2. The method: nested dictionaries

### 2.1 Intuition: one set of dolls, many sizes

A Matryoshka doll set is one object that is also several smaller complete objects. The paper borrows the name and the idea from Matryoshka Representation Learning (Kusupati et al., 2024), where the first $k$ dimensions of an embedding must work as a usable embedding on their own.

Applied to an SAE: the first $m_1$ latents must be a usable SAE on their own. So must the first $m_2$. So must all $m$. The mapping is direct: each doll is a prefix of the latent vector, and "usable on its own" means "reconstructs the input with nothing else".

Why this kills absorption is now visible. The smallest doll cannot call on the "Lily" latent because that latent lives further out. To reconstruct "Lily" at all, the small doll must keep a latent that fires on it, and the cheapest such latent is the general one. Absorption would leave the small doll blind on exactly those tokens.

![Figure 1 from Bussmann et al. (2025): (left) nested reconstructions that each use only a prefix of the latents; (right) at L0 = 40, Matryoshka trades a little variance explained for much lower absorption and lower decoder cosine similarity](/imgs/blogs/learning-multi-level-features-with-matryoshka-sparse-autoencoders-fig1.webp)

### 2.2 Mechanism in words

Pick a maximum dictionary size $m$ and a short list of prefix sizes. For each training batch:

1. Encode $x$ once, exactly like a normal SAE, and apply the sparsity activation once. There is a single sparse code.
2. Starting from the decoder bias, add the contribution of latents 0 through $m_1 - 1$. That is reconstruction number one.
3. Add the next group of latents to the running sum. That is reconstruction number two. Continue until all $m$ latents are in.
4. Compute a mean-squared error for *every* reconstruction and add them up with equal weight.

### 2.3 The math

Define the nested sizes $\mathcal{M} = \{m_1, m_2, \ldots, m_k\}$ with $m_1 \lt m_2 \lt \cdots \lt m_k = m$. (The paper writes the list as $M$; I add a calligraphic letter so it does not collide with the scalar $m$.) Using the subscript ${0{:}m_i}$ to mean "the first $m_i$ latents", the $i$-th reconstruction is

$$
\hat{x}_i = W^{\text{dec}}_{0:m_i,\,:}\; f(x)_{0:m_i} + b_{\text{dec}}
$$

where $f(x)_{0:m_i} \in \mathbb{R}^{m_i}$ is the first $m_i$ entries of the sparse code and $W^{\text{dec}}_{0:m_i,\,:}$ is the matching slice of the decoder (the paper stores the decoder row-per-latent in this equation, so "first $m_i$ rows" means "first $m_i$ latent directions"). The training loss sums the error of every doll:

$$
\mathcal{L}(x) = \sum_{m_i \in \mathcal{M}} \left\lVert x - \left( f(x)_{0:m_i}\, W^{\text{dec}}_{0:m_i} + b_{\text{dec}} \right) \right\rVert_2^2 \;+\; \alpha\, \mathcal{L}_{\text{aux}}
$$

$\mathcal{L}_{\text{aux}}$ is the standard auxiliary loss from Gao et al. (2024) that revives dead latents, weighted by $\alpha$. There is no explicit sparsity penalty in the Gemma runs: BatchTopK enforces sparsity by construction, and the paper says no regularization is used beyond what the nested objectives impose.

Read the sum as a vote count. A latent in group 1 appears in all $k$ terms, so it receives gradient from every doll and is pushed toward features that are useful everywhere. A latent in the last group appears only in the final term, so it is free to specialize.

![The Matryoshka loss: one sparse code split into five contiguous groups; a running reconstruction adds each group in turn, each partial reconstruction gets its own MSE, and the five losses are summed with equal weight](/imgs/blogs/learning-multi-level-features-with-matryoshka-sparse-autoencoders-2.webp)

The group boundaries above are the ones the paper uses on Gemma-2-2B: $\mathcal{M} = \{2048, 6144, 14336, 30720, 65536\}$. As fractions of the total they are $\tfrac{1}{32}, \tfrac{1}{16}, \tfrac{1}{8}, \tfrac{1}{4}$ and $\tfrac{1}{2} + \tfrac{1}{32}$ for the *new* latents in each group. You can check the arithmetic: 2,048 + 4,096 = 6,144, + 8,192 = 14,336, + 16,384 = 30,720, + 34,816 = 65,536.

### 2.4 The code

Appendix B of the paper gives the forward pass. Here it is as a self-contained module, with the paper's structure kept and the BatchTopK step written out:

```python
import torch
import torch.nn as nn

class MatryoshkaBatchTopKSAE(nn.Module):
    def __init__(self, d_model: int, dict_sizes=(2048, 6144, 14336, 30720, 65536), k: int = 40):
        super().__init__()
        m = dict_sizes[-1]
        self.dict_sizes = list(dict_sizes)
        self.k = k                                    # target average L0 per token
        self.encoder = nn.Parameter(torch.randn(d_model, m) * 0.01)
        self.decoder = nn.Parameter(torch.randn(m, d_model) * 0.01)  # one row per latent
        self.bias = nn.Parameter(torch.zeros(d_model))                # shared pre/post bias

    def batch_top_k(self, acts: torch.Tensor) -> torch.Tensor:
        B = acts.shape[0]
        flat = acts.flatten()
        keep = torch.topk(flat, B * self.k).indices   # B*K largest values across the batch
        mask = torch.zeros_like(flat)
        mask[keep] = 1.0
        return (flat * mask).view_as(acts)

    def forward(self, x: torch.Tensor):
        feats = torch.relu((x - self.bias) @ self.encoder)
        sparse = self.batch_top_k(feats)              # one sparse code for every doll
        out = self.bias.expand_as(x).clone()
        losses = []
        for i, end in enumerate(self.dict_sizes):
            start = 0 if i == 0 else self.dict_sizes[i - 1]
            out = out + sparse[:, start:end] @ self.decoder[start:end]
            losses.append(((out - x) ** 2).mean())    # MSE of this prefix
        return sum(losses), out                       # equal weights, final = full SAE
```

The running `out` is why the overhead is moderate: each group's decode is computed once and reused by every larger doll.

### 2.5 BatchTopK, the sparsity mechanism underneath

For the Gemma experiments the activation is BatchTopK (Bussmann et al., 2024). Instead of keeping the top $K$ latents in every token, it keeps the top $B \times K$ activations across a whole batch:

$$
\text{BatchTopK}(X) = X \odot \mathbb{1}\left[X \ge \tau(X)\right]
$$

$X \in \mathbb{R}^{B \times m}$ is the batch of pre-activations for $B$ tokens, $\tau(X)$ is the $(B \times K)$-th largest value in $X$, $\odot$ is elementwise product and $\mathbb{1}[\cdot]$ is the indicator. At inference, $\tau(X)$ is replaced by a fixed threshold $\theta$ calibrated on training data, so a token's code no longer depends on its batch-mates.

**Micro-example.** Take $B = 2$ tokens, $K = 2$, and three latents. Pre-activations are $[3.0, 0.5, 1.0]$ for token A and $[2.0, 0.1, 0.2]$ for token B. The four largest values in the batch are 3.0, 2.0, 1.0 and 0.5, so $\tau = 0.5$. Token A keeps three latents and token B keeps one. The average is still two, but an "easy" token can spend less and a "busy" one more.

## 3. A worked example: why the nested loss refuses to absorb

Let me make the accounting argument from 1.3 numeric. This toy is mine, built to mirror the paper's synthetic setup; the numbers are not from the paper.

**Setup.** One parent direction $p$ and three child directions $c_1, c_2, c_3$, all orthonormal. Every input contains the parent. With probability 0.4 nothing else is on, so $x = p$. Otherwise exactly one child is on, each with probability 0.2, so $x = p + c_j$.

**Flat SAE, four latents.** Two exact solutions exist:

| Solution | Decoder directions | Avg L0 | Reconstruction error |
| --- | --- | --- | --- |
| Clean | $p,\ c_1,\ c_2,\ c_3$ | ${0.4 \times 1 + 0.6 \times 2 = 1.6}$ | 0 |
| Absorbed | $p,\ p{+}c_1,\ p{+}c_2,\ p{+}c_3$ | 1.0 | 0 |

Same error, lower L0. Any sparsity pressure at all prefers the absorbed dictionary, and in it the parent latent is silent on 60% of the inputs that contain the parent.

**Matryoshka SAE, prefix of size 1 plus the full dictionary.** The first latent must now reconstruct every input alone. Compare the two candidates for slot one by expected squared error of the size-1 doll. Fitting the best scalar coefficient $a = x \cdot d$ for a unit direction $d$ leaves error $\lVert x \rVert^2 - (x \cdot d)^2$.

- If slot one is the clean parent $p$: parent-only inputs give error 0. Child inputs have $\lVert x \rVert^2 = 2$ and $x \cdot p = 1$, so error 1. Expected error: ${0.6 \times 1 = 0.6}$.
- If slot one is the parent latent of the *absorbed* dictionary, it fires only on parent-only inputs. On child inputs the size-1 doll outputs nothing and pays $\lVert p + c_j \rVert^2 = 2$. Expected error: ${0.6 \times 2 = 1.2}$.

The absorbed solution now costs an extra 0.6 of squared error per sample in the first doll's term. That penalty is what buys back the parent. The later latents then only need to add $c_j$, because $p$ is already in the running sum, which is exactly the clean dictionary.

A caveat worth stating: if the children together dominated (say the parent almost never appeared alone and there were only one or two children), the size-1 doll could prefer a tilted direction between $p$ and a child. Nesting prefers features that are *broadly* useful, which matches "general" only when general features really are the common ones.

## 4. Evidence from the toy model

The paper's synthetic test follows Chanin et al. (2024). It builds a tree of binary features, each mapped to a unique direction in $\mathbb{R}^{20}$. The root is always present, and every other feature can only fire if its parent fires, with an edge probability $p_{\text{parent} \to \text{child}}$. The input is the sum of the active directions, so "child implies parent" holds by construction.

![The toy model: a feature tree in 20 dimensions where children fire only when their parent fires; the vanilla SAE folds the parent into child latents while the Matryoshka SAE keeps the parent latent on and child latents add only their own direction](/imgs/blogs/learning-multi-level-features-with-matryoshka-sparse-autoencoders-3.webp)

Both SAEs use 20 latents (matching the true number of features), ReLU, Adam at learning rate 3e-2, batch size 200 and 40,000 steps. The Matryoshka variant samples 10 prefix lengths per batch from a truncated Pareto(0.5) distribution, always includes the full SAE, and adapts its $\ell_1$ penalty to hit the ground-truth average $\ell_0$ of 1.2338.

![Figure 2 from Bussmann et al. (2025): the toy model (left) and cosine similarity between learned decoder vectors and ground-truth features for the Matryoshka SAE (middle) and the vanilla SAE (right)](/imgs/blogs/learning-multi-level-features-with-matryoshka-sparse-autoencoders-fig2.webp)

The Matryoshka heatmap is close to a clean diagonal: each latent matches one true feature. The vanilla heatmap has bright off-diagonal blocks between parents and children, the signature of child latents that carry their parent's direction.

## 5. Results on Gemma-2-2B

### 5.1 Setup

| Item | Value |
| --- | --- |
| Model and site | Gemma-2-2B, residual stream, layer 12 |
| Training data | 500M tokens from The Pile |
| Dictionary | 65,536 latents, 5 fixed nested groups |
| Sparsity levels | L0 = 20, 40, 80, 160, 320 (one SAE each) |
| Optimizer | Adam, learning rate 3e-4, batch size 2048 |
| Baselines | SAE Bench suite: ReLU, TopK, BatchTopK, Gated, JumpReLU, P-Annealing |
| Comparability | Same data ordering and hyperparameters as the baselines |

### 5.2 Headline numbers

| Metric (Gemma-2-2B, L0 = 40) | BatchTopK | Matryoshka | Better is |
| --- | --- | --- | --- |
| Variance explained | 72% | 70% | higher |
| Absorption rate (first-letter task) | 0.49 | 0.05 | lower |
| Latents per first letter (splitting) | about 3 | about 1 | lower |
| Meta-SAE variance explained on decoder (largest size) | 55% | 42% | lower |
| Training time (5 groups) | baseline | about +50% | lower |

The meta-SAE row comes from Appendix D.1 and is reported at the largest dictionary size tested there. A meta-SAE is an SAE trained on another SAE's decoder vectors. If it explains a lot of their variance, the latents share structure, which is the composition pathology.

### 5.3 Absorption and splitting

Absorption is measured with the first-letter task from Chanin et al. Logistic-regression probes on the raw activations find the "starts with letter X" directions. The metric then counts how often the matching SAE latents fail to fire on such tokens because a token-specific latent took over.

![Figure 6 from Bussmann et al. (2025): feature absorption rate and feature splitting across L0 for Matryoshka and six baseline architectures](/imgs/blogs/learning-multi-level-features-with-matryoshka-sparse-autoencoders-fig4.webp)

Matryoshka sits at the bottom of both panels across sparsity levels.

### 5.4 Reconstruction vs downstream loss

The cost shows up in Figure 5 of the paper. Variance explained is slightly worse than the best baselines at every L0. But when the reconstructed activations are fed back into Gemma, the increase in cross-entropy loss is comparable, especially at larger L0. The authors read this as Matryoshka spending its budget on directions the model actually uses; the paper does not test that reading directly.

### 5.5 Probing and concept isolation

Three SAE Bench tasks probe whether latents are *useful* rather than just reconstructive:

- **Sparse probing:** 35 binary tasks over professions, sentiment, language, code and news. Pick the most relevant latent, fit a logistic probe.
- **SCR (spurious correlation removal):** train a classifier on biased data (for example "nurse" correlated with "female"), ablate the latents for the spurious signal, and measure recovered accuracy on the real task.
- **TPP (targeted probe perturbation):** ablate latents for one class and check that only that class's probe drops.

![Figure 7 from Bussmann et al. (2025): sparse probing accuracy, SCR and TPP across L0](/imgs/blogs/learning-multi-level-features-with-matryoshka-sparse-autoencoders-fig5.webp)

Matryoshka beats every baseline on SCR and TPP. On sparse probing it is best at lower L0 and beats BatchTopK at every L0.

### 5.6 The scaling result

This is the result the title promises. The authors train SAEs at 4k, 16k and 65k latents and average each metric over L0 = 40, 80 and 160.

![Figure 10 from Bussmann et al. (2025): evaluation metrics across dictionary sizes 4k, 16k and 65k for Matryoshka and baseline SAEs](/imgs/blogs/learning-multi-level-features-with-matryoshka-sparse-autoencoders-fig6.webp)

On almost all metrics the Matryoshka line stays flat or improves as the dictionary grows, while several baselines slope downward.

### 5.7 Ablations

| Variant (65k, 200M tokens) | What changed | Finding |
| --- | --- | --- |
| Weighted Matryoshka | Loss weights proportional to new latents per group | Cross-entropy close to BatchTopK, but worse absorption and SCR |
| Stop gradients | Each partial reconstruction detached before the next | Better absorption at low L0; worse loss recovered, probing, SCR |
| 3 groups | Ratios 1/8, 1/4, 5/8 | Better loss recovered; worse absorption and SCR |
| 10 groups | Log-spaced sizes | Slightly better absorption; much worse loss recovered and autointerp |
| Random prefixes (16k) | Pareto-sampled sizes per batch | Similar; slightly worse SCR and sparse probing |

The pattern is a dial. Fewer groups or reweighting toward the full dictionary interpolates back toward BatchTopK, buying reconstruction with feature quality. Five equally weighted groups is the authors' chosen balance, and they say plainly that the exact number is not critical.

## 6. What's load-bearing, and what might not transfer

**The group schedule is a hyperparameter you inherit.** The geometric schedule fixes how many latents can be "general". On Gemma, the first doll has 2,048 latents. If the model has many more than 2,048 genuinely broad features, some of them will be pushed into later groups, where they are again exposed to absorption. The ablations vary the number of groups but never where the first cut lands, so this is my extrapolation, not a finding.

**Early latents fire more.** Appendix D shows latent frequency falls with index: group-1 latents are used by every doll, so they are denser. Any tool that assumes latents are exchangeable (random subsampling for autointerp, for example) should sample per group.

**Small models at high L0.** On Chess-GPT (hidden size 512) Matryoshka matches others at L0 at or below 150 but degrades above it. The authors attribute this to high L0 being a large fraction of a small residual stream. That is plausible, and it means the method is tuned for the regime where the dictionary is much wider than the model.

## 7. Critique

**What's strong.** The intervention is tiny, the motivation is mechanistic rather than empirical, and the toy model shows the mechanism directly before the benchmarks show the effect. Training the Matryoshka SAEs with identical data ordering and hyperparameters to the SAE Bench baselines removes most of the usual "better tuning" confound.

**What's weak.**

- **One model, one layer.** Every main result is Gemma-2-2B layer 12. TinyStories is qualitative, chess is mixed. We do not know how the absorption gain holds across depth or across model families.
- **Absorption is measured on one task family.** First-letter classification is a clean probe, but it is a single hierarchy ("starts with E" over specific tokens). Semantic hierarchies like "animal over dog" may behave differently.
- **"Better features" is inferred from downstream proxies.** The paper admits the evaluation leans on automated metrics. A human interpretability study is listed as future work.

**What would change my mind.** If a Matryoshka SAE trained on a different model family showed absorption rates comparable to BatchTopK on a *semantic* hierarchy task, I would read the first-letter result as specific to token-level hierarchies.

## 8. When to reach for it

Use a Matryoshka SAE when you need one dictionary that serves both coarse and fine questions: steering on a broad concept, probing a category, or building circuits that should not silently miss sub-cases. Skip it when raw reconstruction fidelity is the goal.

For the measurement methodology behind absorption, see our analysis of [A is for Absorption](/blog/paper-reading/ai-interpretability/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders). For the TopK SAE and the auxiliary loss this paper builds on, see [Scaling and Evaluating Sparse Autoencoders](/blog/paper-reading/ai-interpretability/scaling-and-evaluating-sparse-autoencoders). If the idea of features packed into fewer dimensions than concepts is new, start with [What is superposition](/blog/machine-learning/ai-interpretability/what-is-superposition).

## References

1. Bussmann, B., Nabeshima, N., Karvonen, A., and Nanda, N. *Learning Multi-Level Features with Matryoshka Sparse Autoencoders.* ICML 2025. [arXiv:2503.17547](https://arxiv.org/abs/2503.17547)
2. Reference implementation: [dictionary_learning, Matryoshka BatchTopK trainer](https://github.com/saprmarks/dictionary_learning/blob/main/dictionary_learning/trainers/matryoshka_batch_top_k.py)
3. Toy model code: [noanabeshima/matryoshka-saes](https://github.com/noanabeshima/matryoshka-saes)
4. Kusupati, A. et al. *Matryoshka Representation Learning.* [arXiv:2205.13147](https://arxiv.org/abs/2205.13147)
5. Chanin, D. et al. *A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders.* [arXiv:2409.14507](https://arxiv.org/abs/2409.14507)
6. Bussmann, B., Leask, P., and Nanda, N. *BatchTopK Sparse Autoencoders.* [arXiv:2412.06410](https://arxiv.org/abs/2412.06410)
7. Karvonen, A. et al. *SAEBench: A Comprehensive Benchmark for Sparse Autoencoders.* 2024.

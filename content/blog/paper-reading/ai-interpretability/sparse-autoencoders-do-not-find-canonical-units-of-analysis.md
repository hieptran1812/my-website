---
title: "Sparse Autoencoders Do Not Find Canonical Units of Analysis"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "Why no single sparse autoencoder width gives you a unique, complete and atomic list of features: the SAE stitching and meta-SAE experiments of Leask et al., explained from intuition to math."
tags: ["paper-reading", "sparse-autoencoders", "canonical-units", "sae-stitching", "meta-sae", "feature-splitting", "mechanistic-interpretability", "critique", "iclr-2025-poster"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 21
paper:
  title: "Sparse Autoencoders Do Not Find Canonical Units of Analysis"
  authors: "Patrick Leask, Bart Bussmann, Michael Pearce, Joseph Bloom, Curt Tigges, Noura Al Moubayed, Lee Sharkey, Neel Nanda"
  venue: "ICLR 2025"
  url: "https://arxiv.org/abs/2502.04878"
---

> [!tldr]
> **The claim.** A hope in mechanistic interpretability is that a big enough sparse autoencoder (SAE) will land on the "true features" of a model: a list that is unique, complete and atomic. This paper argues no SAE width gets you there.
> **The two tools.** *SAE stitching* moves latents between SAEs of different sizes and shows that small SAEs miss information entirely (not complete). *Meta-SAEs* are SAEs trained on another SAE's decoder directions, and they show that large-SAE latents break into smaller parts (not atomic).
> **Most surprising result.** An "Albert Einstein" latent decomposes into meta-latents for science, famous people, Germany, space and astronomy, and a few more. Those meta-latents look almost exactly like the latents of a much smaller SAE trained on the real activations.
> **Where it is thin.** Two models (GPT-2 Small and Gemma 2 2B), one layer each, a cosine threshold chosen by eye, and a meta-SAE that explains only 55.47% of the variance it decomposes.
> **The practical takeaway.** Treat dictionary size as a lens you choose per task, not a dial you turn until the truth appears.

![Figure 1 from Leask et al. (2025): a GPT-2 SAE latent for "Einstein" decomposed into a sparse set of interpretable meta-latents](/imgs/blogs/sparse-autoencoders-do-not-find-canonical-units-of-analysis-fig1.webp)

The figure above is the paper's punchline in one picture. A single latent that a large SAE learned for "Einstein" is, on closer inspection, a bundle of more general pieces. If the latent you were treating as a basic unit is itself made of parts, then it was never the basic unit.

Below we walk through the two techniques the authors built to show it, and what the numbers do and do not support.

## The problem: what would a canonical unit even be

Interpretability needs a unit of analysis, the way chemistry needs atoms. Early work tried individual MLP neurons and attention heads. Both turned out to be **polysemantic**: one neuron fires for several unrelated concepts, so you cannot read it as one thing.

Sparse autoencoders were the answer. An SAE re-expresses a model's activation vector as a sparse sum of many learned directions, and each direction (a **latent**) tends to be monosemantic. Bricken et al. (2023) floated the idea that with a large enough dictionary, an SAE might find the "true features". The paper calls such a set **canonical** and pins it to three properties.

![The three properties of a canonical unit, how the paper probes each, and what it finds](/imgs/blogs/sparse-autoencoders-do-not-find-canonical-units-of-analysis-1.webp)

| Property | What it means | How the paper probes it | Verdict |
| --- | --- | --- | --- |
| Unique | No variation across runs | Same-size SAEs with different seeds (appendix) | Mostly holds: 94% of latents match at cosine above 0.7 |
| Complete | Nothing the model uses is missing | SAE stitching | Fails: larger SAEs contain novel latents |
| Atomic | Cannot be split into smaller parts | Meta-SAEs | Fails: large-SAE latents decompose |

The paper's real targets are the last two rows. Uniqueness only shows up in an appendix figure, and it comes out reasonably well. Keep that in mind when the title says SAEs "do not find" canonical units: the case rests on completeness and atomicity.

Two earlier observations set this up. In **feature splitting**, a base64 latent in a small SAE becomes three latents (letters, digits, encoded ASCII) in a bigger one. And Templeton (2024) found latents for specific chemical elements in a large SAE that a smaller one lacked. Nobody had a systematic way to compare latents across widths. That is the gap this paper fills.

## Background: the SAE in three equations

An SAE has an encoder and a decoder. Here is the paper's notation:

$$
f(x) = \sigma(W_{\text{enc}} x + b_{\text{enc}}), \qquad \hat{x}(f) = W_{\text{dec}} f + b_{\text{dec}}
$$

where $x \in \mathbb{R}^n$ is one activation vector from the model (for GPT-2 Small's residual stream, $n = 768$), $f(x) \in \mathbb{R}^m$ is the latent code with $m$ entries (the **dictionary size**), $W_{\text{enc}}$ and $b_{\text{enc}}$ are the encoder weights and bias, $W_{\text{dec}}$ is the $m \times n$ decoder matrix, and $b_{\text{dec}} \in \mathbb{R}^n$ is the decoder bias. The activation $\sigma$ keeps $f$ non-negative and sparse. Latent $i$ is **active** on $x$ when $f_i(x) > 0$.

Training balances reconstruction against sparsity:

$$
\mathcal{L}(x) = \underbrace{\lVert x - \hat{x}(f(x)) \rVert_2^2}_{\text{reconstruction}} + \lambda \, \underbrace{S(f(x))}_{\text{sparsity}} + \alpha \mathcal{L}_{\text{aux}}
$$

Here $S$ penalizes dense codes (the L1 norm for ReLU SAEs), $\lambda$ sets how hard we push for sparsity, and $\mathcal{L}_{\text{aux}}$ is an optional auxiliary loss with small weight $\alpha$, used for example to revive dead latents in TopK SAEs. The number of active latents per input, averaged over a batch, is the **L0**.

Each decoder row $W^{\text{dec}}_i$ is a direction in activation space, and the reconstruction is a sum of those directions scaled by their activations. That one fact powers both techniques: you can swap pieces between SAEs, and you can treat the pieces themselves as data. For the deeper story on TopK SAEs and how width scales, see our analysis of [Scaling and Evaluating Sparse Autoencoders](/blog/paper-reading/ai-interpretability/scaling-and-evaluating-sparse-autoencoders).

## Why sparsity pushes toward composed latents

Before the techniques, it helps to see *why* a larger SAE might learn composite latents at all. The paper borrows a toy example from Smith (2024) and Wattenberg and Viégas (2024).

**The setup.** A model represents two independent properties: color (red, green, blue) and shape (circle, square, triangle). The true underlying features are those six values. Every input is one colored shape.

**Small SAE.** With six latents, the natural solution is one latent per value. A blue square fires "blue" and "square": L0 = 2 per input.

**Large SAE.** With nine latents, the SAE can learn one latent per combination: "red circle", "blue square", and so on. A blue square now fires one latent: L0 = 1 per input.

![How the sparsity penalty rewards a composed dictionary over an atomic one](/imgs/blogs/sparse-autoencoders-do-not-find-canonical-units-of-analysis-4.webp)

Both dictionaries reconstruct the input perfectly. But the loss does not only score reconstruction. With an L0-style count, the per-input sparsity cost is

$$
S_{\text{atomic}} = 2, \qquad S_{\text{composed}} = 1
$$

so the composed dictionary halves the sparsity term at zero reconstruction cost. The optimizer has no reason to prefer the "true" six features, and a concrete reason to prefer the nine compositions. More capacity does not buy you finer atoms. It buys you the freedom to pre-multiply atoms that co-occur.

**When this argument fails.** It assumes the combinations are few enough to fit in the dictionary. With 10 colors and 10 shapes you need 100 composed latents; an SAE with fewer slots must keep some atomic ones. So composition is a pressure, not a guarantee, and how much of it you get depends on width. That width-dependence is exactly what the next two techniques measure. The same pressure produces the related failure called feature absorption, covered in our post on [A is for Absorption](/blog/paper-reading/ai-interpretability/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders).

## Technique 1: SAE stitching

### The problem it solves

Take a 768-latent SAE and a 1,536-latent SAE trained on the same activations. Which latents in the big one are "the same" as latents in the small one, which are refinements, and which are brand new? Without an answer, "bigger SAEs find more features" stays an anecdote.

### The analogy

Two editions of a dictionary: a pocket edition and an unabridged one. Some entries in the big edition are the same word with the same meaning. Some are new words the pocket edition never had. Stitching is the experiment of pasting entries from the big edition into the pocket one and seeing if the pocket edition gets better (a new word helps) or just redundant (a duplicate adds clutter).

### The mechanism

Because the SAE output is a sum of decoder directions, you can build a hybrid SAE: keep some latents from SAE 0, bring in some from SAE 1, and add up their contributions. Then measure reconstruction error (MSE) and L0 of the hybrid. Two outcomes matter:

- **MSE improves** after adding a latent: it carried information the small SAE lacked. That is a **novel latent**.
- **MSE gets worse**: the latent overlaps with something already there, and you are now representing the same thing twice. That is a **reconstruction latent**, and the right move is to *swap* it in for its counterparts rather than add it.

### The math

The SAE output decomposes as a sum over latents:

$$
\hat{x} = \sum_{i=0}^{d} W^{\text{dec}}_i f_i(x) + b_{\text{dec}}
$$

where $d$ is the dictionary size, $W^{\text{dec}}_i \in \mathbb{R}^n$ is latent $i$'s decoder direction, and $f_i(x)$ its activation. A stitched SAE takes a set $L_0$ of latents from SAE 0 and a set $L_1$ from SAE 1:

$$
\hat{x} = \alpha b^{\text{dec}}_0 + (1-\alpha) b^{\text{dec}}_1 + \sum_{l_0 \in L_0} W^{\text{dec}}_{0,l_0} f_{0,l_0}(x) + \sum_{l_1 \in L_1} W^{\text{dec}}_{1,l_1} f_{1,l_1}(x)
$$

with $\alpha = |L_0| / (|L_0| + |L_1|)$, so the two decoder biases are blended in proportion to how many latents each SAE contributes. Unlike classic model stitching, no learned adapter is needed: decoder directions already live in the same activation space. The paper checks that the bias blend is harmless: decoder biases of SAEs trained on the same activations have a minimum cosine similarity of 0.9970 and differ by less than 0.1% in magnitude.

Testing every subset of latents is impossible, so the paper uses a cheap proxy. Latent $i$ of the larger SAE is classified as **novel** when its best match in the smaller SAE is weak:

$$
\max_j \frac{W^{\text{dec}}_{1,i} \cdot W^{\text{dec}}_{0,j}}{\lVert W^{\text{dec}}_{1,i} \rVert \, \lVert W^{\text{dec}}_{0,j} \rVert} \lt \theta
$$

and as a **reconstruction latent** otherwise. The threshold is $\theta = 0.7$ for the GPT-2 SAEs and $\theta = 0.4$ for the Gemma Scope SAEs, both "based on our exploratory analysis". Decoder cosine is a stand-in for behavioral similarity; on the GPT-2 768 and 1,536 pair, it correlates with activation similarity at $R^2 = 0.87$ for cosines above 0.5.

Reconstruction latents are swapped in groups. The authors build a bipartite graph: small-SAE latents on one side, large-SAE latents on the other, an edge wherever cosine exceeds $\theta$. Each **connected subgraph** is a family. Swapping a family removes its small-SAE members and inserts its large-SAE members.

![The SAE stitching procedure: classify each larger-SAE latent, then add novel ones and swap reconstruction families](/imgs/blogs/sparse-autoencoders-do-not-find-canonical-units-of-analysis-2.webp)

### A worked micro-example

Take a 3-dimensional toy activation space. A large-SAE latent has unit decoder direction $W^{\text{dec}}_{1,i} = (0.6, 0.8, 0)$. The small SAE has two unit directions, $(1, 0, 0)$ and $(0, 1, 0)$.

- Cosine with the first: ${0.6}$. Cosine with the second: ${0.8}$.
- The max is ${0.8}$, which is at least $\theta = 0.7$, so this is a **reconstruction latent**. It joins a family with the second small-SAE latent.
- Change the large latent to $(0.5, 0.5, 0.707)$ (also unit length). Cosines are ${0.5}$ and ${0.5}$, max ${0.5 \lt 0.7}$, so it is **novel**: most of its mass points along a third axis the small SAE never covered.

For the bias blend: keep 700 small-SAE latents and stitch in 50 large-SAE latents, and $\alpha = 700 / 750 \approx 0.93$.

Here is the classification and family-building step as PyTorch-shaped code:

```python
import torch
import networkx as nx

def stitch_plan(W_small: torch.Tensor, W_large: torch.Tensor, theta: float = 0.7):
    """W_small: [m0, n] decoder rows, W_large: [m1, n] decoder rows."""
    A = torch.nn.functional.normalize(W_small, dim=1)   # [m0, n]
    B = torch.nn.functional.normalize(W_large, dim=1)   # [m1, n]
    cos = B @ A.T                                       # [m1, m0]
    best = cos.max(dim=1).values                        # [m1]

    novel = torch.nonzero(best < theta).flatten().tolist()

    # bipartite graph over reconstruction latents
    g = nx.Graph()
    rows, cols = torch.nonzero(cos >= theta, as_tuple=True)
    for i, j in zip(rows.tolist(), cols.tolist()):
        g.add_edge(("large", i), ("small", j))

    families = []
    for comp in nx.connected_components(g):
        small = sorted(j for side, j in comp if side == "small")
        large = sorted(i for side, i in comp if side == "large")
        families.append((small, large))   # swap: remove small, insert large
    return novel, families

# Phase 1: add each novel latent (L0 up, MSE down on average)
# Phase 2: swap each family (L0 down on average)
```

### What the experiments show

The paper stitches eight GPT-2 Small SAEs trained on the layer-8 residual stream (widths 768 to 98,304) and two Gemma Scope SAEs on Gemma 2 2B (16,384 and 32,768).

![Figure 4 from Leask et al. (2025): change in MSE from adding each GPT2-1536 latent to GPT2-768, against its max cosine similarity to GPT2-768](/imgs/blogs/sparse-autoencoders-do-not-find-canonical-units-of-analysis-fig2.webp)

Latents with low max cosine (left) mostly reduce MSE when added: they bring new information. Latents with high max cosine (right) mostly increase it: they double-count. The crossover sits near 0.7 for GPT-2. An appendix ROC curve shows max cosine predicts the sign of the effect, and adding *all* the GPT2-1536 latents with cosine at or below 0.7 into GPT2-768 cuts MSE by almost 10%.

One novel latent gets a close look: a GPT2-1536 latent for "make sure" whose nearest GPT2-768 neighbors sit around cosine 0.3. The small SAE reconstructs noticeably worse exactly where that latent fires:

| SAE | MSE, latent inactive | MSE, latent active | Gap |
| --- | --- | --- | --- |
| GPT2-1536 | 2.225 | 2.518 | 0.293 |
| GPT2-768 | 2.703 | 3.292 | 0.589 |

The small SAE's error gap is about twice the large one's (0.589 versus 0.293): it simply lacks the concept. The paper reports the same pattern averaged over all 657 low-cosine latents in GPT2-1536.

![Figure 5 from Leask et al. (2025): interpolating from GPT2-768 to GPT2-12288 by adding novel latents, then swapping reconstruction families](/imgs/blogs/sparse-autoencoders-do-not-find-canonical-units-of-analysis-fig3.webp)

The interpolation figure is the cleanest summary. Each transition (768 to 1,536 to 3,072 to 6,144 to 12,288) has two phases. Adding novel latents raises L0, because you are inserting extra active units, and lowers MSE. Swapping reconstruction families lowers L0 on average, because the larger SAE encodes the same information more sparsely. Both phases improve MSE on average. You can walk smoothly from one SAE to the next, and the Gemma Scope pair interpolates the same way at $\theta = 0.4$.

### Why it works, and when it fails

Stitching works because SAE decoder directions share one coordinate system, so no alignment step is needed. The weak point is the threshold. Max cosine is a single number per latent and it is blind to families where one large latent is a blend of three small ones, each at a cosine of 0.5. Such a latent can be classified "novel" even though it adds little. The threshold is also model-specific (0.7 versus 0.4) and set by inspection, so the exact count of novel latents depends on a knob the paper does not sweep in the main text. The *existence* of novel latents is robust to this; the *number* is not.

## From stitching to composition

Stitching surfaced something odd. Feature splitting predicts one small latent becoming several large ones, but some large-SAE latents had high cosine to *several* small-SAE latents at once. That is the blue-square story showing up in a real model, and the next technique checks it directly.

## Technique 2: meta-SAEs

### The problem it solves

If a large-SAE latent is really a composition, we should be able to *find* its parts. Stitching only measures overlap with one smaller SAE. We need a tool that takes any latent and returns its ingredients.

### The analogy

Take every recipe in a big cookbook and ask: what is the smallest pantry of base ingredients from which each recipe is a short list? A meta-SAE is that pantry-finder. The recipes are the large SAE's decoder directions; the pantry is the set of meta-latents.

### The mechanism

A meta-SAE is an ordinary SAE whose training data is not model activations but the decoder matrix of another SAE. Each of the 49,152 decoder rows of the GPT-2 SAE becomes one training sample. The meta-SAE learns to rebuild each row as a sparse sum of meta-latent directions. A row that needs "science" plus "Germany" plus "famous people" gets those three meta-latents switched on.

![A meta-SAE pipeline: decoder rows of a large SAE become training data for a small sparse dictionary](/imgs/blogs/sparse-autoencoders-do-not-find-canonical-units-of-analysis-3.webp)

### The math

The paper does not write a separate equation for the meta-SAE, so here is the standard SAE objective with the data swapped (my notation). Each training sample is a decoder row $w_i = W^{\text{dec}}_i \in \mathbb{R}^{768}$, for $i = 1, \dots, 49152$:

$$
g(w_i) = \text{BatchTopK}(V_{\text{enc}} w_i + c_{\text{enc}}), \qquad \hat{w}_i = V_{\text{dec}} \, g(w_i) + c_{\text{dec}}
$$

where $g(w_i) \in \mathbb{R}^{2304}$ is the meta-latent code, $V_{\text{dec}}$ is the ${2304 \times 768}$ meta-decoder, and $V_{\text{enc}}$, $c_{\text{enc}}$, $c_{\text{dec}}$ are the remaining weights. The meta-SAE minimizes $\lVert w_i - \hat{w}_i \rVert_2^2$.

**BatchTopK** keeps the top $b \cdot k$ activations across a batch of $b$ samples rather than the top $k$ per sample, so samples can share the sparsity budget. At inference, where batch-dependence is unwanted, it uses a global threshold estimated on training data:

$$
\theta = \mathbb{E}_X \left[ \min \{ z_{i,j}(X) \mid z_{i,j}(X) > 0 \} \right]
$$

that is, the average over batches of the smallest activation that survived the batch-level top-k. Anything below $\theta$ at test time is zeroed.

The configuration, straight from the paper:

| Setting | Value |
| --- | --- |
| Source SAE | GPT-2 Small, layer 8, 49,152 latents |
| Training samples | 49,152 decoder directions |
| Meta-latents | 2,304 |
| Average active meta-latents per SAE latent | about 4 |
| Epochs | 2,000 (few samples, so many passes) |
| Optimizer | Adam, learning rate 1e-4, batch size 4,096 |
| Variance of decoder directions explained | 55.47% |

### A worked micro-example

Back to the toy world. Suppose the large SAE learned nine composed directions in a 6-dimensional space, one axis per base value: "blue square" is $(0, 0, 1, 0, 1, 0)$ normalized, with axes ordered red, green, blue, circle, square, triangle. A meta-SAE with six meta-latents and $k = 2$ can rebuild every one of the nine rows exactly: "blue square" fires the blue meta-latent and the square meta-latent, each at about 0.707. The meta-dictionary is the original six features. Nine training rows, six meta-latents, two active per row: the meta-SAE has recovered the atoms the large SAE fused.

### What the experiments show

![Table 1 from Leask et al. (2025): GPT-2 SAE latents and their meta-latent decompositions](/imgs/blogs/sparse-autoencoders-do-not-find-canonical-units-of-analysis-fig4.webp)

The decompositions read as sensible:

- **Albert Einstein**: Science and Scientists, Famous People, Space and Astronomy, Germany, Electricity, Words starting with a capital E.
- **Rugby**: Sports activities, Words starting with "R", References to Ireland, References to sports leagues, activities and actions.
- **Android Operating System**: Mobile phones, operating systems, Californian cities.

Interpretable-looking is not evidence, so the authors test it. GPT-4o-mini explains each SAE latent from its top activations, then each meta-latent from the SAE latents it fires on. Given only the meta-latent explanations active on a latent, the model must pick that latent's explanation out of five (one correct, four random). On 1,000 random latents it is right **73%** of the time, against 20% chance.

![Figure 6 from Leask et al. (2025): max cosine similarity between meta-SAE latents and latents of standard SAEs of several sizes](/imgs/blogs/sparse-autoencoders-do-not-find-canonical-units-of-analysis-fig5.webp)

The second test is the striking one. The meta-SAE never saw a model activation. Yet its 2,304 meta-latents closely match the latents of a standard 3,072-wide SAE trained on the actual layer-8 activations (high maximum cosine in the figure). The authors push it further: replace each meta-decoder direction with its most similar SAE decoder direction, retrain only the encoder, and reconstruction drops only a little. So the parts of a 49,152-wide SAE's latents are, to a good approximation, the latents of a 3,072-wide SAE.

### Why it works, and when it fails

It works because sparse coding on decoder directions finds their shared sub-directions, just as sparse coding on activations does. The caveat is the 55.47% figure. Almost half of the decoder variance is not explained by the meta-SAE, so "Einstein is science plus Germany plus famous people" is a sparse approximation of the latent, not an identity. A meta-SAE also has a sparsity penalty of its own; it will return *some* sparse decomposition of any matrix you feed it. The evidence that the decomposition is real rests on the 73% auto-interp score and on the match to the 3,072 SAE, not on the decomposition existing.

## Experiments and results at a glance

Here are the SAEs behind every result, from the paper's Table 2. CELR is cross-entropy loss recovered when the model's activations are replaced by the SAE reconstruction, relative to zero or mean ablation.

| SAE | Model | Width | L0 | MSE | CELR (zero) | CELR (mean) |
| --- | --- | --- | --- | --- | --- | --- |
| GPT2-768 | GPT-2 Small | 768 | 35.2 | 2.72 | 0.915 | 0.876 |
| GPT2-1536 | GPT-2 Small | 1,536 | 39.5 | 2.22 | 0.942 | 0.915 |
| GPT2-3072 | GPT-2 Small | 3,072 | 42.4 | 1.89 | 0.955 | 0.937 |
| GPT2-6144 | GPT-2 Small | 6,144 | 43.8 | 1.63 | 0.965 | 0.949 |
| GPT2-12288 | GPT-2 Small | 12,288 | 43.9 | 1.46 | 0.971 | 0.958 |
| GPT2-24576 | GPT-2 Small | 24,576 | 42.9 | 1.33 | 0.975 | 0.963 |
| GPT2-49152 | GPT-2 Small | 49,152 | 42.4 | 1.21 | 0.978 | 0.967 |
| GPT2-98304 | GPT-2 Small | 98,304 | 43.9 | 1.14 | 0.980 | 0.970 |
| GemmaScope-16384 | Gemma 2 2B | 16,384 | 43.4 | 1.72 | 0.983 | 0.980 |
| GemmaScope-32768 | Gemma 2 2B | 32,768 | 41.5 | 1.59 | 0.985 | 0.982 |

Two things stand out. First, L0 stays in a narrow 35 to 44 band across a 128x range of widths, so the stitching comparisons are not confounded by wildly different sparsity. Second, MSE keeps falling with width: from 768 to 98,304 latents it drops from 2.72 to 1.14, a 58% reduction (my arithmetic: ${1 - 1.14/2.72 \approx 0.58}$). Nothing in the table plateaus, which is consistent with the claim that there is no width at which the dictionary is "done".

All GPT-2 SAEs were trained on 300M tokens of OpenWebText (sequences up to 128 tokens) with Adam at learning rate 4e-4, batch size 4,096, and sparsity penalty 8e-5.

The appendix adds two downstream checks. Single-latent **sparse probing** on eight datasets finds the best width "complex and dataset dependent". **Targeted probe perturbation**, a concept-removal benchmark built on the SHIFT method from [Sparse Feature Circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models), trends down with width but not monotonically. Neither crowns a winner.

**What might not transfer.** One layer per model (layer 8 of GPT-2, layer 12 of Gemma 2 2B), a largest SAE of 98,304 latents (far from the millions-wide dictionaries in Templeton, 2024), per-model cosine thresholds, and a single meta-SAE configuration.

## Critique

**What is strong.** Both tools are simple, cheap and reusable. The interpolation in Figure 5 is a genuinely new view: SAEs of different widths sit on a continuum you can walk. The "make sure" case and the 657-latent average make incompleteness concrete. And the conclusion admits the paper finds neither canonical units nor the right width.

**What is weak or under-tested.**

- **The threshold carries the novel/reconstruction split.** A heuristic cut at 0.7 (or 0.4) classifies every latent. The ROC curve shows it predicts the *sign* of the effect, but the counts of novel latents move with $\theta$.
- **"Atomic" is relative to a basis you chose.** Meta-latents are SAE latents too. Train a meta-meta-SAE and they will probably decompose again. The paper shows large latents are not atomic *relative to smaller dictionaries*; it does not show any dictionary is atomic, which is arguably the more damning reading, but it is not stated that way.
- **Composition versus genuine structure.** If the model represents Einstein as a sum of "scientist", "German" and "famous" directions, the composite latent is an artifact. If it has a dedicated Einstein direction that merely correlates with those, the meta-SAE is imposing structure. The evidence cannot fully tell these apart.
- **Uniqueness gets one appendix figure.** 94% of latents match across seeds at equal width. That is a result in SAEs' favor, and it deserves more than a footnote. Work such as Archetypal SAE treats run-to-run stability as its central problem.

**The missing experiment.** Take the meta-latents, use them (or the matched 3,072-SAE latents) as a compositional code for activations, and compare downstream cross-entropy and L0 against the 49,152-wide SAE. That would turn "decomposes into" from a geometric statement into a functional one.

**What would change my mind.** Two results would reopen the canonical-units hope for me. If the fraction of novel latents per width doubling fell toward zero at some width (the census flattening rather than continuing), that would point to a complete dictionary. And if a meta-SAE explained close to all of a large SAE's decoder variance with meta-latents that, used directly on activations, matched its loss at equal L0, composition would be a removable artifact of the sparsity penalty rather than a fundamental limit.

## What I would build with this

These are my extrapolations, not the paper's claims.

1. **A novel-latent census.** For every width doubling, report the share of novel latents at several thresholds. A curve that flattens or keeps climbing is the cleanest single answer to "is there a complete width".
2. **Width selection as a diagnostic.** Before using an SAE for steering or unlearning, stitch the task-relevant latents across two or three widths. If the concept is a reconstruction latent everywhere, any width works; if it only appears as novel at large widths, you need the big one.
3. **Training objectives that resist composition.** Train one SAE whose prefixes of the dictionary must each reconstruct well on their own, so general latents are forced to exist alongside specific ones. Follow-up work along exactly these lines (Matryoshka SAEs) came out after this paper; I mention it as context, not as something this paper evaluates.

## References

- Leask, Bussmann, Pearce, Bloom, Tigges, Al Moubayed, Sharkey, Nanda. *Sparse Autoencoders Do Not Find Canonical Units of Analysis.* ICLR 2025. [arXiv:2502.04878](https://arxiv.org/abs/2502.04878)
- Interactive meta-SAE dashboard from the authors: [metasaes.streamlit.app](https://metasaes.streamlit.app)
- Bricken et al. *Towards Monosemanticity: Decomposing Language Models with Dictionary Learning.* Transformer Circuits Thread, 2023.
- Bussmann, Leask, Nanda. *BatchTopK Sparse Autoencoders.* [arXiv:2412.06410](https://arxiv.org/abs/2412.06410)
- Related posts on this site: [Scaling and Evaluating Sparse Autoencoders](/blog/paper-reading/ai-interpretability/scaling-and-evaluating-sparse-autoencoders), [A is for Absorption](/blog/paper-reading/ai-interpretability/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders), [Sparse Feature Circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models)

---
title: "Dense SAE Latents Are Features, Not Bugs: What the Always-On Latents Actually Encode"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "Why sparse autoencoders keep learning latents that fire on 10 to 50 percent of tokens, the ablation that shows those latents track a real dense subspace of the residual stream, and the six kinds of signal they turn out to carry."
category: "paper-reading"
subcategory: "AI Interpretability"
tags:
  - paper-reading
  - sparse-autoencoders
  - dense-latents
  - features
  - mechanistic-interpretability
  - gemma-scope
  - antipodal-pairs
  - neurips-2025-poster
author: "Hiep Tran"
featured: true
readTime: 18
image: "/imgs/blogs/dense-sae-latents-are-features-not-bugs-fig1.webp"
excerpt: "Sparse autoencoders keep producing latents that fire on a large share of tokens. This paper ablates them, measures their geometry, and sorts them into position, context-binding, nullspace, alphabet, part-of-speech and PCA classes."
paper:
  title: "Dense SAE Latents Are Features, Not Bugs"
  authors: "Xiaoqing Sun, Alessandro Stolfo, Joshua Engels, Ben Wu, Senthooran Rajamanoharan, Mrinmaya Sachan, Max Tegmark"
  venue: "NeurIPS 2025; arXiv 2025"
  url: "https://arxiv.org/abs/2506.15679"
---

> [!tldr]
> **Claim.** The sparse autoencoder (SAE) latents that fire on a large fraction of tokens are not training noise. They reconstruct a specific dense subspace of the residual stream, and they carry real signals: position, the active topic, output entropy, the next token's first letter, part of speech, and the top principal component.
>
> **Why it matters.** Recent SAE work treats dense latents as a defect and adds penalties to suppress them. If they encode genuine model computation, those penalties throw information away.
>
> **Most surprising result.** Remove the subspace spanned by dense latents, retrain the SAE, and almost no dense latents come back. Remove an equally sized subspace of sparse latents and the dense latents return as before.
>
> **Where it falls short.** The taxonomy explains less than half of the dense latents, and almost all of the interpretation work uses one SAE family (Gemma Scope JumpReLU) on one model (Gemma 2 2B).

Anyone who has browsed a sparse autoencoder has met them: latents that light up on a large share of every document, with top examples that look like a wall of unrelated text.

This paper asks whether these latents are an artifact of SAE training, or the SAE faithfully reporting that the model has signals that are always on. The answer is mostly the second, backed by a causal ablation, a geometric regularity, and a taxonomy. For why models pack many features into few dimensions at all, see the [superposition explainer](/blog/machine-learning/ai-interpretability/what-is-superposition).

## The setup: what an SAE is and what "dense" means

**The problem.** A transformer's residual stream is a vector of a few thousand numbers per token, and individual neurons respond to many unrelated things. An SAE tries to rewrite each such vector as a sum of a few interpretable directions.

**The analogy.** A paint mixer with sixteen thousand pigment tubes should mix any color (an activation) from a handful of tubes. A "dense" tube is one you squeeze into almost every color.

![An SAE reads one residual activation through an encoder into mostly-zero latents and decodes it back; a latent counts as dense when it fires on more than 10 percent of tokens](/imgs/blogs/dense-sae-latents-are-features-not-bugs-1.webp)

**The mechanism.** The encoder applies a wide matrix, a bias and a nonlinearity that zeroes most entries; the decoder sums the surviving latents' decoder vectors plus a bias.

**The math.** Following the paper's notation:

$$
f(x) = \sigma\left(W_{\text{enc}}\, x + b_{\text{enc}}\right), \qquad \hat{x}(f) = W_{\text{dec}}\, f + b_{\text{dec}}
$$

Here $x \in \mathbb{R}^{d_{\text{model}}}$ is the residual-stream activation for one token, $f \in \mathbb{R}^{d_{\text{sae}}}$ is the non-negative latent vector with $d_{\text{sae}} \gg d_{\text{model}}$, and $\sigma$ is the activation function (JumpReLU for Gemma Scope, TopK for the SAEs the authors train). $W_{\text{enc}}$ has shape $d_{\text{sae}} \times d_{\text{model}}$, $W_{\text{dec}}$ has shape $d_{\text{model}} \times d_{\text{sae}}$, and the $i$-th rows and columns are written $W^{(i)}_{\text{enc}}$ and $W^{(i)}_{\text{dec}}$. Training minimizes $\lVert x - \hat{x}(f(x)) \rVert_2^2$ under a sparsity constraint.

The density of latent $i$ is simply the fraction of tokens on which $f_i(x) > 0$:

$$
\text{density}_i = \frac{1}{N} \sum_{t=1}^{N} \mathbb{1}\left[f_i(x_t) > 0\right]
$$

where $N$ is the number of tokens and $\mathbb{1}[\cdot]$ is 1 when the condition holds. The paper calls a latent **dense** when this exceeds 0.1.

**A worked example.** A latent firing on 1,500 of 10,000 tokens has density 0.15 and counts as dense. A "Golden Gate Bridge" latent firing on 3 of 10,000 has density 0.0003, three orders of magnitude lower.

**When the definition misleads.** A latent tracking something always present (sentence position) and a noisy blend of many sparse features can both land at 0.15. Separating those cases is the real job of this paper, and the authors admit they only partly manage it.

### The experimental setup at a glance

| Item | What the paper uses |
| --- | --- |
| Main model | Gemma 2 2B |
| Main SAEs | Gemma Scope residual-stream JumpReLU SAEs, 16k latents |
| Own SAEs | TopK SAEs on layer 25, trained on 1B tokens of OpenWebText |
| Compute | about 30 A6000 hours, plus under USD 20 of LLM-judge calls |

## Technique 1: the subspace ablation test

**The problem.** An optimization quirk would reappear whatever you feed the SAE. A real property of the activations, once removed from the data, should stop dense latents from forming.

**The analogy.** A smoke detector goes off constantly. Remove the toaster; if the alarm stops, the detector was fine and was reporting the toaster.

![The ablation test: train SAEs on original activations, on activations with a sparse-latent subspace removed, and on activations with the dense-latent subspace removed](/imgs/blogs/dense-sae-latents-are-features-not-bugs-3.webp)

**The mechanism.** The authors take a TopK SAE trained on layer 25 of Gemma 2 2B and collect the decoder directions of its dense latents. They zero-ablate the subspace these directions span in the training activations and train a fresh SAE. As a control, they pick an equally sized set of non-dense latents, ablate their subspace instead, and train a third SAE. They repeat everything at two dictionary sizes, 16,384 and 32,768.

**The math.** Let $D \in \mathbb{R}^{d_{\text{model}} \times m}$ hold the $m$ dense decoder directions as columns and let $P_D$ be the orthogonal projector onto their span. Zero-ablating that subspace means training the new SAE on

$$
x' = \left(I - P_D\right) x
$$

where $I$ is the identity. Every component of $x$ inside the dense subspace is deleted; everything orthogonal to it is untouched. (The paper describes this in words; the projector notation is mine.)

**A worked example.** In a toy 3-dimensional residual stream, suppose the only dense direction is $e_1 = (1, 0, 0)$. An activation $x = (2.0, 0.3, -0.1)$ becomes $x' = (0, 0.3, -0.1)$. If the large, always-present first coordinate is what made a latent fire on most tokens, the new SAE has nothing left to be dense about.

![Figure 1 from Sun et al. (2025): (a) latent density distributions for the original SAE, the sparse-latent ablation and the dense-latent ablation; (b) encoder cosine similarity among the 50 densest latents; (c) antipodality score against density](/imgs/blogs/dense-sae-latents-are-features-not-bugs-fig1.webp)

**The result.** In panel (a) of Figure 1, at both dictionary sizes, the dense-ablation SAE (teal) has far fewer high-density latents than the original (blue) and the sparse-ablation control (orange). The appendix adds that the dense count plateaus within about 100k training steps, so longer training does not shrink it, and that GPT-2 Small and LLaMA 3.2 1B behave the same way.

**When it fails.** The test shows that dense latents need the dense subspace. It does not show that every dense latent inside that subspace is individually meaningful. A subspace can be real while the SAE carves it into an arbitrary basis, a caveat the authors raise themselves in their limitations.

## Technique 2: antipodal pairs and the antipodality score

**The problem.** Many dense latents come in pairs whose decoder vectors point in nearly opposite directions. The paper needs a number for how strongly a latent has such a twin.

**The analogy.** A one-way street sign can only say "go north." To describe traffic in both directions you need two signs, one pointing north and one pointing south. A ReLU-style SAE latent is a one-way sign: its activation is never negative.

![Why dense latents come in antipodal pairs: a ReLU-style SAE needs two latents with opposite decoder directions to represent a signed signal, while an AbsoluteTopK SAE covers both signs with one latent](/imgs/blogs/dense-sae-latents-are-features-not-bugs-2.webp)

**The mechanism.** If the model keeps a signed scalar along direction $d$ on almost every token, one non-negative latent cannot represent it. The SAE allocates two, with decoders near $+d$ and $-d$, and whichever sign the signal takes, one fires. Since the signal is always present, each member is dense. To score this, compare latent $i$ with every other latent on both encoder and decoder similarity and keep the strongest match.

**The math.** The antipodality score of latent $i$ is

$$
s_i = \max_{j \neq i} \; \operatorname{sim}\left(W^{(i)}_{\text{enc}}, W^{(j)}_{\text{enc}}\right) \cdot \operatorname{sim}\left(W^{(i)}_{\text{dec}}, W^{(j)}_{\text{dec}}\right)
$$

where $\operatorname{sim}(u, v) = u^\top v / (\lVert u \rVert \lVert v \rVert)$ is cosine similarity. If both cosines are near $-1$, their product is near $+1$, so $s_i$ is high. If $j$ points in an unrelated direction, both cosines are near 0 and so is the product.

**A worked example.** Take a dense latent and suppose its best partner has encoder cosine $-0.95$ and decoder cosine $-0.97$. Then $s = (-0.95)(-0.97) = 0.92$. A random partner with cosines ${0.05}$ and $-0.08$ gives $s = -0.004$. Note that a near-duplicate latent with cosines $+0.95$ and $+0.97$ would also score high; the authors check this case in the appendix and find that $s$ approaches 1 only when both cosines are close to $-1$, so high scores here really mean opposites.

**The result.** Panel (c) of Figure 1 shows a strong positive correlation between $s_i$ and density. Most latents with density above 0.3 have $s$ above 0.9. The trend holds for TopK and JumpReLU SAEs, for GPT-2 and Gemma, and across layers.

**The clinching control.** An AbsoluteTopK SAE allows negative activations and keeps the $k$ latents with the largest absolute values. Trained with the same seed as a TopK SAE, it shows no antipodal dense pairs, exactly as the one-way-sign story predicts.

**When it fails.** Antipodality explains why dense latents come in twos, not what the shared direction means.

## Technique 3: a taxonomy of dense latents

With the persistence and the geometry established, the paper sorts the dense latents of the Gemma Scope SAEs, layer by layer, into classes using automated tests with fixed cutoffs.

![Figure 3 from Sun et al. (2025): the share of dense latents assigned to each taxonomy class at every layer of Gemma 2 2B](/imgs/blogs/dense-sae-latents-are-features-not-bugs-fig2.webp)

| Class | Test used to assign it | Cutoff |
| --- | --- | --- |
| Position (sentence, paragraph, context) | Spearman correlation with distance to a boundary | $\lvert\rho\rvert > 0.4$ |
| Context-binding | Fraction of correct flips under steering | above 0.75 |
| Nullspace | Encoder norm in the bottom 10 singular directions of $W_U$ | above 0.2 |
| Alphabet | Share of top or bottom 100 logit tokens with the same first letter | at least 90% |
| Meaningful-word | AUC of "is a meaningful word" predicting firing | above 0.75 |
| PCA-aligned | Cosine with the top principal component | above 0.75 |

Only 3.6% of dense latents land in more than one class; ties follow a fixed priority order. Keep the large "Unknown" band at the top of Figure 3 in view throughout.

### Position latents

**The problem.** Knowing how far into a sentence or paragraph you are helps predict periods and line breaks, on every token.

**The mechanism.** Over 5,000 contexts of 1,024 tokens, project residual activations onto each dense latent's decoder vector and correlate that projection with distance since the last period (sentence start), last newline (paragraph start) and beginning of input (context start). The projection is used instead of the JumpReLU output because the direction, not the threshold, should carry position.

**The math.** Spearman's rank correlation between the projection $p_t$ and the distance $\delta_t$ over $n$ tokens is

$$
\rho = 1 - \frac{6 \sum_{t=1}^{n} r_t^2}{n\left(n^2 - 1\right)}
$$

where $r_t$ is the difference between the rank of $p_t$ and the rank of $\delta_t$. It measures whether the two move together monotonically, without assuming a straight line.

**A worked example.** Five tokens after a period have distances 1, 2, 3, 4, 5 and projections 0.9, 0.7, 0.8, 0.4, 0.1. The projection ranks (largest is 5) are 5, 3, 4, 2, 1 against distance ranks 1, 2, 3, 4, 5. The squared rank differences sum to ${16 + 1 + 1 + 4 + 16 = 38}$, so $\rho = 1 - 228/120 = -0.9$. A strong negative correlation: this latent decays as the sentence goes on, and it would clear the $\lvert\rho\rvert > 0.4$ bar.

**The findings.** Sentence and paragraph trackers are prominent before layer 10; context-position trackers, like earlier "position neurons", appear throughout. One paragraph tracker ignores artificially inserted newlines, so it tracks semantic paragraph breaks.

### Context-binding latents

**The problem.** Some mid-layer dense latents fire on long runs of tokens, but their apparent concept changes from document to document.

**The analogy.** A CPU register has no fixed meaning; it holds whatever the current function needs. The hypothesis: these directions are registers for "the concept currently in focus."

**The mechanism.** Within an antipodal pair, the active member tends to switch when the topic changes. To test causality, the authors generate unsteered, steered-to-feature-1 and steered-to-feature-2 completions from RedPajama prompts. An LLM judge (Gemini 2.5 Flash) labels each as closer to feature 1, feature 2, or unclear, given activating examples from the same context. The score is the fraction of flips from the unsteered label that go the steered way.

**A worked example.** Suppose 50 completions change label relative to the unsteered baseline, and 41 of those changes match the steering direction. The fraction of correct flips is ${41/50 = 0.82}$. The paper reports values of this size for several mid-layer pairs, for instance 81.5% for the layer 13 pair (15275, 11449) and 76.5% for the layer 12 pair (14906, 14599).

**The telling contrast.** With in-context examples, "unclear" is rare: 0.024 to 0.051 across the eleven top pairs in Table 1. With out-of-context examples it jumps to 0.531 to 0.798. Specific inside a document, inconsistent across documents: what a register would look like.

**When it fails.** A generic uninterpretable direction could look the same once an eager LLM judge finds a story. The authors say they cannot rule this out.

### Nullspace latents

**The problem.** Earlier work found a quasi-nullspace of the unembedding matrix $W_U$: directions that carry a large share of the residual norm yet barely change the logits. If they carry so much norm, an SAE minimizing reconstruction error should spend latents on them.

![Figure 6 from Sun et al. (2025): (a) how much encoder norm each latent places in the last 10 singular directions of the unembedding; (b) entropy change when ablating nullspace latents versus random latents; (c) the same ablation with RMSNorm scaling frozen](/imgs/blogs/dense-sae-latents-are-features-not-bugs-fig3.webp)

**The mechanism.** Take the SVD $W_U = U \Sigma V^\top$ and look at the last $k$ left singular vectors, the directions $W_U$ ignores most. For each latent, measure how much of its encoder vector lies in that subspace.

**The math.** Writing $w = W^{(i)}_{\text{enc}}$ and $U_{-j}$ for the $j$-th singular vector counted from the end,

$$
\alpha_k = \frac{\left\lVert \sum_{j=1}^{k} U_{-j}\, U_{-j}^\top w \right\rVert}{\lVert w \rVert}
$$

is the fraction of the encoder's norm that lies in the bottom-$k$ subspace. (I write the numerator as the norm of the projection, which is what the paper's description says; its typeset equation compresses this.) The paper uses $k = 10$.

**A worked example.** A unit-norm encoder vector with components ${0.2}$, ${0.2}$ and ${0.1}$ along three of the bottom singular vectors and nothing on the other seven has $\alpha_{10} = \sqrt{0.04 + 0.04 + 0.01} = 0.3$. That clears the 0.2 cutoff, so the latent is called nullspace-aligned.

**The findings.** At layer 25, 99.6% of latents have $\alpha_{10} \lt 0.2$. Of the rest, 75% are dense, making up 40% of the layer's dense latents. Ablating each candidate's decoder direction (setting it to the decoder bias value) shifts output entropy far more than 50 random latents do for some of them, with latent 14325 standing out. Freezing the RMSNorm scale shrinks the effect. The mechanism, as I read it: the model pushes norm into a direction the unembedding ignores, so the final RMSNorm divides by a bigger number and the logits flatten. The pair 13748 and 14325 tracks output entropy closely.

**When it fails.** Not all nullspace latents move entropy. The authors speculate that some track other signals, such as the attention sink, but leave it open.

### Alphabet latents

**The mechanism.** Project each latent's decoder onto the vocabulary (through the unembedding) and take its 100 most promoted and 100 most suppressed tokens. If at least 90% of either set start with the same character, ignoring the leading space marker, the latent is an alphabet latent.

![Table 2 from Sun et al. (2025): alphabet latents at layer 25 of Gemma 2 2B, with density, the share of top tokens sharing the letter, and example tokens](/imgs/blogs/dense-sae-latents-are-features-not-bugs-fig4.webp)

**The findings.** Layer 25 has 114 such latents; 21 have density above 0.1, which is 20% of its dense latents, and they are dense only at the final layer. Table 2 shows the granularity: a dense "R" latent (density 0.16) beside a sparse "B then R" latent (density 0.006), which the authors attribute to feature splitting. These concern the next token's first letter, not the current one's.

### Meaningful-word latents

**The mechanism.** Using the Brown Corpus, the authors collapse part-of-speech tags into high-level groups, collect dense-latent activations on about 200k tokens, and compute the AUC-ROC of predicting "latent fires" from "token is in this group." Single groups like "verb" predict poorly. The union {nouns, proper nouns, verbs, adjectives, adverbs}, which they call "meaningful words," gets an AUC of about 0.8 for many early-layer dense latents.

AUC is the probability that a random firing token outscores a random silent token under the predictor: 1.0 is perfect, 0.5 is a coin flip. At 0.8, the "is a content word" bit separates firing from silent tokens well but far from perfectly.

### PCA latents

Do dense latents just reconstruct the top principal components? Only partly: one antipodal pair reconstructs most of the first PC (cosine above 0.75), but other dense latents have little norm even in the top 5 PCs, and changing L0 or dictionary size does not change this.

## Technique 4: layer-wise dynamics

**The problem.** Do dense latents change in number and in subspace from layer to layer?

![Figure 9 from Sun et al. (2025): (a) the fraction of latents above four density thresholds at each layer; and the median principal angle between the dense-latent subspaces of every pair of layers](/imgs/blogs/dense-sae-latents-are-features-not-bugs-fig5.webp)

**The mechanism.** Count latents above density thresholds of 0.05, 0.1, 0.2 and 0.3 per layer. Then, for each pair of layers, take the subspaces spanned by latents with density above 0.2 and compute their principal angles.

**The math.** For subspaces $\mathcal{A}$ and $\mathcal{B}$ with orthonormal bases $Q_A$ and $Q_B$, the principal angles $\theta_1 \le \theta_2 \le \dots$ satisfy

$$
\cos \theta_\ell = \sigma_\ell\left(Q_A^\top Q_B\right)
$$

where $\sigma_\ell$ is the $\ell$-th singular value. Angles near 0° mean the subspaces overlap; near 90° means they are orthogonal. The paper summarizes each pair of layers by the median angle.

**A worked example.** If $Q_A^\top Q_B$ has singular values 0.98, 0.9 and 0.2, the principal angles are about 11°, 26° and 78°, and the median is 26°: the two subspaces share most of their structure.

**The findings.** The dense count spikes in layers 0 to 4 just above the 0.05 and 0.1 thresholds (mostly part-of-speech latents), stays flat through layers 5 to 23, and rises in the last two layers. The angles form three clusters: layers 0 to 4, layers 10 to 22, and the last few layers, which break away as alphabet and nullspace latents appear. Subspaces of 100 random non-dense latents sit near 90° everywhere, so the clustering is not a dimensionality artifact.

![What dense latents encode by depth in Gemma 2 2B: structural signals early, semantic registers in the middle, output-oriented signals at the end](/imgs/blogs/dense-sae-latents-are-features-not-bugs-4.webp)

The arc: structural signals early, semantic signals in the middle, output-oriented signals at the end.

## Results summary

| Claim | Evidence | Scale |
| --- | --- | --- |
| Dense latents are intrinsic | Dense-subspace ablation leaves almost no dense latents; sparse-subspace ablation does not | Gemma 2 2B L25 at 16k and 32k latents; replicated on GPT-2 Small and LLaMA 3.2 1B |
| They form antipodal pairs | Most latents with density above 0.3 have $s$ above 0.9 | TopK and JumpReLU, GPT-2 and Gemma |
| Pairs come from non-negativity | AbsoluteTopK removes antipodal dense latents | Same-seed TopK comparison |
| Nullspace latents regulate entropy | Ablation shifts entropy; effect shrinks with RMSNorm frozen | Gemma 2 2B L25, 10k C4 tokens |
| Alphabet latents are output signals | 21 dense of 114 alphabet latents, 20% of dense latents | Gemma 2 2B L25 |
| Context-binding is in-context only | "Unclear" 0.024 to 0.051 in-context, 0.531 to 0.798 out-of-context | 11 pairs, layers 12 to 17 |

**What is load-bearing and might not transfer.** Interpretation uses one SAE suite, one sparsity per layer and one 2B model, with hand-set cutoffs behind Figure 3. My untested guess: nullspace and alphabet latents transfer best, being tied to architecture (final RMSNorm, a subword vocabulary), while context-binding is most sensitive to the judge and steering strength.

## Critique

**What is strong.** The ablation has the right control, an equally sized sparse subspace. The AbsoluteTopK result turns an observation (pairs) into a confirmed prediction (no pairs once negativity is allowed). The nullspace analysis reaches a mechanism by freezing RMSNorm and watching the effect shrink.

**What is weaker.**

- The authors explain less than half of the dense latents. "Some dense latents are features" is what the evidence supports.
- The ablation shows the dense subspace is real, not that the SAE's basis inside it is meaningful. The limitations section concedes that dense latents co-occur so often that a rotated basis reconstructs equally well.
- Context-binding rests on an LLM judge with no reported human agreement rate.

**A missing ablation.** Train an SAE with the frequency regularizer the paper argues against and measure whether entropy, position or next-letter probes from its latents get worse. That would turn "do not penalize dense latents" into a measured cost.

**What would change my mind.** If a frequency-penalized SAE matched the original on reconstruction and downstream probing while having no dense latents, I would read dense latents as one convenient basis, not a necessary one. Conversely, if the "Unknown" share in Figure 3 shrank sharply with better tests while staying non-random under the ablation, the headline would earn its full strength.

**For practitioners.** Before filtering dense latents out of a dashboard or circuit analysis, check for antipodal pairs and read the projection rather than the activation: a pair often reads as one signed variable. A related failure of SAE bases is covered in the [feature absorption analysis](/blog/paper-reading/ai-interpretability/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders).

## References

1. Sun, Stolfo, Engels, Wu, Rajamanoharan, Sachan, Tegmark. [Dense SAE Latents Are Features, Not Bugs](https://arxiv.org/abs/2506.15679). NeurIPS 2025.
2. Lieberum et al. (2024), Gemma Scope, the SAE suite analyzed.
3. Gao et al. (2025), the TopK SAE. On this site: [Scaling and Evaluating Sparse Autoencoders](/blog/paper-reading/ai-interpretability/scaling-and-evaluating-sparse-autoencoders).
4. Stolfo et al. (2024), the entropy-regulation mechanism behind the nullspace analysis.
5. Related on this site: [What is superposition](/blog/machine-learning/ai-interpretability/what-is-superposition) and [A is for Absorption](/blog/paper-reading/ai-interpretability/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders).

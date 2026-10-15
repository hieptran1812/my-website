---
title: "Linear Representations of Political Perspective Emerge in Large Language Models"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How Kim, Evans and Schein find a single liberal to conservative direction inside a handful of attention heads, read it out token by token, test it on news outlets it never saw, and push generated text along it."
excerpt: "Ridge probes on 1,024 attention heads recover lawmakers' DW-NOMINATE scores at a Spearman correlation near 0.87, transfer without retraining to news outlet slant, and double as steering vectors. A walk through every technique, from intuition to the math."
category: "paper-reading"
subcategory: "AI Interpretability"
tags:
  - paper-reading
  - linear-representations
  - political-perspective
  - probing
  - activation-steering
  - inference-time-intervention
  - large-language-models
  - mechanistic-interpretability
  - iclr-2025-oral
author: "Hiep Tran"
featured: true
readTime: 19
image: "/imgs/blogs/linear-representations-of-political-perspective-emerge-in-large-language-models-fig1.webp"
paper:
  title: "Linear Representations of Political Perspective Emerge in Large Language Models"
  authors: "Junsol Kim, James Evans, Aaron Schein"
  venue: "ICLR 2025 (oral)"
  url: "https://arxiv.org/abs/2503.02080"
---

> [!tldr]
> - **Claim:** three 7B chat models (Llama-2-7b-chat, Mistral-7b-instruct, Vicuna-7b) carry a *linear* direction for the American liberal to conservative axis, concentrated in a few middle-layer attention heads.
> - **Mechanism:** fit one ridge regression per attention head (1,024 of them per model) that predicts a lawmaker's DW-NOMINATE score from the head's output when the model is asked to "generate a statement by [NAME]".
> - **Why it matters:** the same probe, never retrained, ranks 400 news outlets by slant (Spearman 0.798 on Llama-2), reads a stance out of each generated token, and doubles as a steering vector.
> - **Most surprising:** non-linear probes buy almost nothing (0.854 linear vs 0.855 MLP on Llama-2's best head), and steering only works in layers below 22.
> - **Where it is thin:** 7B models only, a U.S.-centric axis that drops to 0.531 on non-U.S. parties, and steering effects rated by GPT-4o against just 21 human-rated essays.

![Figure 1 from Kim, Evans and Schein (2025): Mistral-7b-instruct essays on immigration and abortion, each token shaded by the probe on Layer 16, Head 1](/imgs/blogs/linear-representations-of-political-perspective-emerge-in-large-language-models-fig1.webp)

The image above is the whole paper in one picture: a probe trained only on lawmaker prompts shades each token of an open-ended essay toward one end of a single axis. The rest of this post unpacks how that probe is built, tested and turned into a dial.

## The problem: does "political perspective" live somewhere specific?

Chat models readily write in the voice of a named senator or news outlet, and researchers already use this to simulate survey respondents. The open question is what the model is doing inside when it does that: is there a reusable internal variable for perspective, or does each persona live in its own corner of activation space?

Interpretability has a working answer for many other concepts. Sentiment, latitude and longitude, time, truthfulness and refusal have all been found as **directions** in activation space, which is the claim of the [linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis). This paper asks the same question about something subjective and contested, and it brings a measuring stick from political science to do it.

The stick is **DW-NOMINATE**. It is a score fitted to a bills-by-lawmakers voting matrix, and its first dimension places each U.S. lawmaker on a line from -1 (most liberal) to +1 (most conservative). It is not a judgment about anyone's views; it is a compression of roll-call votes. That gives the probe a numeric target with decades of validation behind it.

| What the paper needs | What it uses |
| --- | --- |
| A label for "perspective" | First dimension of DW-NOMINATE, range -1 to +1 |
| A population with labels | All N = 552 members of the 116th Congress (2019–2021) plus the President |
| Models with open activations | Llama-2-7b-chat, Mistral-7b-instruct-v0.1, Vicuna-7b-v1.5 |
| An out-of-domain test | Ad Fontes Media slant scores for N = 400 outlets |

The 116th Congress was chosen because it sits before Llama-2's September 2022 pre-training cutoff, so the model has had a chance to read about every member.

## Contributions

1. **Per-head probing**: ridge probes on middle-layer heads predict held-out lawmakers' DW-NOMINATE at Spearman about 0.85, near 0.87 with a 32-head ensemble.
2. **Linearity checks**: MLP probes do no better, and scrambled or non-monotonic labels do not fit.
3. **Generalization**: a token-level readout of open-ended essays, and zero-shot transfer to news-outlet slant.
4. **Steering**: adding the probe's weights to head outputs shifts the rated slant of generated text, but only before layer 22.

## Method

![The probing pipeline: one prompt per lawmaker, 1,024 head outputs recorded, 1,024 ridge probes scored by cross-validated Spearman, and the top 32 averaged](/imgs/blogs/linear-representations-of-political-perspective-emerge-in-large-language-models-1.webp)

The figure above is the full probing loop. Each technique below is one box in it.

### 1. Where to look: the output of a single attention head

**The problem.** "Activation" is ambiguous in a transformer. You could probe the residual stream, the MLP output, or one of the 32 heads in each layer, and the choice changes what you can find.

**Intuition.** Think of each layer as a meeting where 32 specialists each add a short note to a shared document. The residual stream is the document after everyone has written in it. Probing one head means reading one specialist's note before it gets mixed in with the other 31.

**Mechanism.** The paper writes the residual stream $r_\ell$ after layer $\ell$ as the previous stream plus the projected outputs of all heads plus an MLP. Head $h$ reads a projection of the stream, runs attention, and emits a small vector $x_{\ell,h}$. That vector, taken before layer normalization, is what gets probed.

**The math.** The next-token distribution is

$$
P(w_t = v \mid w_{\lt t}) \propto \exp\!\left(u_v^\top r_L\right)
$$

where $w_t$ is the token at position $t$, $u_v \in \mathbb{R}^D$ is the unembedding of vocabulary item $v$, and $r_L \in \mathbb{R}^D$ is the final residual vector. Each layer updates the stream as

$$
r_\ell = r_{\ell-1} + \sum_{h=1}^{H} Q_{\ell,h}\, x_{\ell,h} + \mathrm{MLP}_\ell\!\left(r_{\ell-1} + \sum_{h=1}^{H} Q_{\ell,h}\, x_{\ell,h}\right), \qquad x_{\ell,h} = \mathrm{ATTN}_{\ell,h}\!\left(P_{\ell,h}\, r_{\ell-1}\right)
$$

Here $H$ is the number of heads per layer, $P_{\ell,h} \in \mathbb{R}^{d \times D}$ maps the stream down into head space, and $Q_{\ell,h} \in \mathbb{R}^{D \times d}$ maps the head's output back up. For these models $D = 4096$ and there are 32 heads, so (my arithmetic, not stated in the paper) each head output has $d = 4096 / 32 = 128$ dimensions.

**Worked example.** 32 layers of 32 heads means ${32 \times 32 = 1{,}024}$ separate probes per model, each seeing one 128-dimensional vector per lawmaker prompt.

**When it fails.** A concept spread thinly across many heads, or written only into the MLP path, would look weak under per-head probing even if the residual stream encodes it well. The paper's ensemble step (below) partly compensates.

### 2. The probe: ridge regression on one head

**The problem.** We want to know whether a head's output *linearly* predicts ideology. With 128 features and only about 276 training lawmakers per fold, ordinary least squares would overfit and suffer from correlated features.

**Intuition.** A probe is a ruler laid across the cloud of activations. If ideology is a direction, there is some angle at which the ruler's readings line up with the DW-NOMINATE scores. Ridge regression keeps the ruler from twisting itself to chase individual points.

**Mechanism.** Feed every lawmaker prompt through the model, record $x^{(i)}_{\ell,h}$ for each head, then for each head fit weights $\theta_{\ell,h}$ so that the dot product with the activation approximates that lawmaker's score.

**The math.** The probe assumes

$$
\mathbb{E}\!\left[y^{(i)} \mid x^{(i)}_{\ell,h}\right] = \hat{y}^{(i)}_{\ell,h} \triangleq \theta_{\ell,h}^\top x^{(i)}_{\ell,h}
$$

and is fitted by minimizing the L2-penalized squared loss

$$
\mathcal{L}_\lambda(\theta_{\ell,h}) = \sum_{i=1}^{N} \left(y^{(i)} - \theta_{\ell,h}^\top x^{(i)}_{\ell,h}\right)^2 + \lambda \lVert \theta_{\ell,h} \rVert_2^2
$$

where $y^{(i)} \in [-1, 1]$ is lawmaker $i$'s DW-NOMINATE score, $\theta_{\ell,h} \in \mathbb{R}^{d}$ are the coefficients, $N$ is the number of training prompts, and $\lambda$ sets how hard large weights are penalized. The authors picked $\lambda = 1$ by 2-fold cross-validation over $\{0, 0.001, 0.01, 0.1, 1, 100, 1000\}$. The fitted $\hat\theta_{\ell,h}$ is then read as a *direction* in that head's activation space.

**Worked example.** Shrink the head to 2 dimensions. Suppose the fitted probe is $\hat\theta = (0.5, -0.25)$ and a lawmaker prompt produces $x = (0.8, 0.4)$. The prediction is ${0.5 \times 0.8 - 0.25 \times 0.4 = 0.4 - 0.1 = 0.3}$, a point slightly toward the conservative end of the scale.

The per-model, per-head loop is short. This sketch mirrors the paper's setup with scikit-learn (hook names are illustrative):

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

# acts: array [N_lawmakers, n_layers, n_heads, d_head], recorded before layer norm
# y:    array [N_lawmakers], DW-NOMINATE first dimension in [-1, 1]
def score_heads(acts, y, lam=1.0, seed=0):
    N, L, H, d = acts.shape
    rho = np.zeros((L, H))
    folds = list(KFold(n_splits=2, shuffle=True, random_state=seed).split(y))
    for l in range(L):
        for h in range(H):
            X = acts[:, l, h, :]
            fold_rhos = []
            for train, test in folds:
                probe = Ridge(alpha=lam, fit_intercept=False).fit(X[train], y[train])
                fold_rhos.append(spearmanr(probe.predict(X[test]), y[test]).correlation)
            rho[l, h] = np.mean(fold_rhos)   # cross-validated Spearman per head
    return rho

rho = score_heads(acts, y)
top = np.dstack(np.unravel_index(np.argsort(-rho, axis=None), rho.shape))[0]
print("best head (layer, head):", top[0], "rho =", rho.max().round(3))
```

**When it fails.** A high score says the head *contains* information that predicts the label, not that the model *uses* it. That distinction is why the paper later intervenes on the same heads.

### 3. Scoring: cross-validated Spearman, then a top-K ensemble

**The problem.** We need one number per head that says how well its probe generalizes to lawmakers it did not train on, and we care about ordering more than exact values.

**Intuition.** Spearman correlation asks only "did you put the lawmakers in the right order?" It ignores whether the predictions are squashed or stretched, which suits a probe whose output scale is arbitrary.

**Mechanism.** Split lawmakers randomly into two equal folds. Train on one, predict the other, compute the rank correlation, swap, and average. Call the result $\hat\rho^{CV}_{\ell,h}$. Then sort all heads by it and average the predictions of the best $K$.

**The math.** With $T_K$ the set of $(\ell, h)$ pairs holding the $K$ highest $\hat\rho^{CV}_{\ell,h}$, the ensemble prediction is

$$
\hat{y}^{(i)}_K \triangleq \frac{1}{K} \sum_{(\ell,h) \in T_K} \hat{y}^{(i)}_{\ell,h}
$$

and it gets its own cross-validated score $\hat\rho^{CV}_K$.

**Worked example.** Three held-out lawmakers have true scores $(0.5, -0.4, -0.1)$ and the probe predicts $(0.3, -0.6, 0.1)$. Ranking both lists gives $(3, 1, 2)$ and $(3, 1, 2)$. The ranks match, so Spearman is ${1.0}$ even though no single prediction is exact.

![Figure 2 from Kim, Evans and Schein (2025): cross-validated Spearman for every head in every layer, heads sorted within each layer, for (a) Llama-2-7b-chat, (b) Mistral-7b-instruct, (c) Vicuna-7b](/imgs/blogs/linear-representations-of-political-perspective-emerge-in-large-language-models-fig2.webp)

The dark band in Figure 2 is the headline. For all three models, many heads in roughly layers 10 to 20 reach $\hat\rho^{CV}_{\ell,h}$ near 0.8, while early layers stay pale. The best single heads are listed below (from Table A2).

| Model | Best head (layer, head) | Best single-head $\hat\rho^{CV}$ | Ensemble $\hat\rho^{CV}_{K=32}$ |
| --- | --- | --- | --- |
| Llama-2-7b-chat | (15, 18) | 0.854 | 0.870 |
| Mistral-7b-instruct | (16, 3) | 0.846 | 0.864 |
| Vicuna-7b | (24, 8) | 0.861 | 0.885 |

**When it fails.** The ensemble gains are small. Table A3 shows Llama-2 moving from 0.8537 at $K = 1$ to 0.8703 at $K = 32$ and back down to 0.8476 at $K = 512$, which is the "rises, then falls as weak heads are averaged in" shape the authors predicted. Selecting the best of 1,024 heads by a score also inflates that maximum a little, which I return to in the critique.

### 4. Is it really linear? Two robustness checks

**The problem.** A high linear score could mean ideology is linear, or it could mean 128 dimensions are enough to fit almost anything on 276 points. The authors test both readings.

**Intuition.** If a ruler fits, try a bendy ruler. If the bendy one does no better, the straight ruler was not just lucky. Then try fitting the ruler to nonsense labels: if it fits those too, the fit meant nothing.

**Check A: non-linear probes.** Following Gurnee and Tegmark, they fit a one-hidden-layer MLP per head:

$$
\hat{y}^{(i)}_{\ell,h} = A_{\ell,h}\, \mathrm{ReLU}\!\left(B_{\ell,h}\, x^{(i)}_{\ell,h} + b_{\ell,h}\right) + a_{\ell,h}
$$

where $B_{\ell,h}$ and $A_{\ell,h}$ are weight matrices and $b_{\ell,h}$, $a_{\ell,h}$ are bias vectors. The best MLP probe barely moves the needle: 0.855 vs 0.854 on Llama-2, 0.872 vs 0.861 on Vicuna, and *worse* on Mistral, 0.838 vs 0.846.

**Check B: transformed labels.** They refit the linear probes on three distortions of $y$: a monotonic cube $y^3$, a non-monotonic $\sin(10y)$, and a random permutation of $y$ across lawmakers.

![Figure from Table A4 of Kim, Evans and Schein (2025): Spearman and R-squared for linear probes trained on the original label, a random permutation, sin(10y) and y cubed](/imgs/blogs/linear-representations-of-political-perspective-emerge-in-large-language-models-fig5.webp)

| Label (Llama-2-7b-chat) | Spearman | R² |
| --- | --- | --- |
| Original $y$ | 0.854 | 0.821 |
| Cube $y^3$ | 0.842 | 0.613 |
| $\sin(10y)$ | 0.551 | 0.298 |
| Random permutation | 0.152 | 0.019 |

**Worked reading.** Spearman cannot tell $y$ from $y^3$, because cubing preserves order, so the 0.842 is expected. R² can tell them apart, and it drops from 0.821 to 0.613: the activations line up with $y$ itself, not with a bent version of it. The permutation row is the noise floor, roughly 0.15 Spearman.

**When it fails.** These checks rule out "the head fits anything", but not "the model memorized DW-NOMINATE from pre-training", since the prompt is little more than a name. The next two experiments attack that.

## Generalization: does the direction mean "ideology" or "this list of names"?

### 5. Token-by-token readout

**The problem.** If the probe only learned a lookup from names to scores, it should produce noise when pointed at anything else.

**Mechanism.** Ask a model to write an essay on a policy topic with no persona at all. At each generated token $t$, record the head output $x^{(i,t)}_{\ell,h}$ and apply the *unchanged* lawmaker probe, $\hat{y}^{(i,t)}_{\ell,h} = \hat\theta_{\ell,h}^\top x^{(i,t)}_{\ell,h}$. Color the token by the result.

**What it shows.** In Figure 1, the readout leans liberal on spans describing the arguments of immigration supporters and abortion-rights advocates, and leans conservative on spans describing the arguments of immigration critics and abortion opponents. The model writes a balanced essay; the probe tracks which side's argument is being voiced. The authors call this qualitative and leave systematic evaluation to future work.

Over 90 unprompted essays per model (Appendix A.2), the mean readout is 0.054 for Llama-2, 0.048 for Vicuna and 0.143 for Mistral, whose spread is also about double ($\sigma = 0.649$ vs 0.349 and 0.356).

### 6. Zero-shot transfer to news outlets

**The problem.** A token-level picture is persuasive but anecdotal. A clean test needs a second labeled population the probe never saw.

**Mechanism.** Take 400 popular U.S. outlets with Ad Fontes Media scores, rescaled to the same -1 (left) to +1 (right) range. Prompt each model with "Generate a statement from a news source in the United States" and the completion stub "[OUTLET] reported that". Apply the top 32 lawmaker-trained probes as-is and compare rankings.

![Figure 3 from Kim, Evans and Schein (2025): Llama-2-7b-chat ensemble predictions against DW-NOMINATE for lawmakers (left) and against Ad Fontes scores for news outlets (right)](/imgs/blogs/linear-representations-of-political-perspective-emerge-in-large-language-models-fig3.webp)

Ad Fontes scores come from article ratings by politically balanced panels of at least three analysts (one left-leaning, one centrist, one right-leaning). With no retraining, the probes reach Spearman 0.798 on Llama-2, 0.764 on Mistral and 0.720 on Vicuna.

![Train once on lawmakers, test on what it never saw: the same probe transfers to news outlets, essay tokens and non-U.S. parties with very different fidelity](/imgs/blogs/linear-representations-of-political-perspective-emerge-in-large-language-models-2.webp)

The diagram above collects every place the lawmaker probe was pointed. The third arm comes from Appendix A.10: on 411 parties from the Manifesto Project, Llama-2's probe manages 0.531 overall. Some countries track closely (New Zealand 0.920, Australia 0.916, Canada 0.883, the United Kingdom 0.845) while others are weak or negative. The direction is an *American* axis that happens to line up with some other two-party-like systems.

## Steering: turning the probe into a dial

### 7. Inference-time intervention with the probe weights

**The problem.** Correlation inside a head is not proof the model uses that head. If the direction is causal, pushing activations along it should change what the model writes.

**Intuition.** The probe weight vector points from "more liberal" to "more conservative" in head space. Steering is nudging the needle along that same arrow, at every step of generation, and watching where the text goes.

**Mechanism.** Following the inference-time intervention method of Li et al. (2023), which the [multi-attribute steering post](/blog/paper-reading/ai-interpretability/multi-attribute-steering-of-language-models-via-targeted-intervention) builds on, replace each targeted head's output with a shifted copy. Apply it at all $K$ top heads and at every generated token.

**The math.**

$$
x^{(\alpha)}_{\ell,h} \triangleq x_{\ell,h} + \alpha\, \hat\sigma_{\ell,h}\, \hat\theta_{\ell,h}
$$

where $\hat\sigma_{\ell,h}$ is the estimated standard deviation of that head's activations (so the push is scaled to the head's natural spread) and $\alpha \in \mathbb{R}$ sets direction and strength. Negative $\alpha$ should yield more liberal-sounding text, positive $\alpha$ more conservative-sounding text.

**Worked example.** Reuse the 2-dimensional head: $x = (0.8, 0.4)$, $\hat\theta = (0.5, -0.25)$, and suppose $\hat\sigma = 0.2$ and $\alpha = 10$. The shift is ${10 \times 0.2 \times (0.5, -0.25) = (1.0, -0.5)}$, so $x^{(\alpha)} = (1.8, -0.1)$. The probe now reads ${0.5 \times 1.8 - 0.25 \times (-0.1) = 0.925}$, up from ${0.3}$. In general the readout rises by exactly $\alpha \hat\sigma \lVert\hat\theta\rVert^2$, here ${10 \times 0.2 \times 0.3125 = 0.625}$.

![Steering adds the probe direction scaled by alpha; the usable window is about -30 to +30, and only layers below 22 respond](/imgs/blogs/linear-representations-of-political-perspective-emerge-in-large-language-models-3.webp)

**When it fails.** The diagram above marks the two edges. Push too hard and the text collapses: for $|\alpha| > 30$ the share of coherent essays falls below 40% in all three models (Appendix A.8), with outputs that loop on a single phrase. Push in the wrong place and nothing happens: on Llama-2, intervening only in layers below 22 gives a 0.540 correlation between $\alpha$ and rated slant, while layers 22 and up give -0.022.

### 8. Measuring the steered text

**The study.** Nine policy topics, prompted with "Write a statement about [ISSUE]" and the stub "Regarding [ISSUE], I believe that", crossed with $K \in \{16, 32, 48, 64, 80, 96\}$ and $\alpha \in \{-30, -20, -10, 0, 10, 20, 30\}$, for 1,134 essays in total.

**The rater.** GPT-4o (gpt-4o-2024-08-06) scores each essay on a 7-point scale. It was validated against 10 CloudResearch annotators (3 Democrats, 4 independents, 3 Republicans) on 21 Llama-2 essays, reaching an intraclass correlation of 0.91 and a Spearman of 0.952 with the human average.

![Figure 4 from Kim, Evans and Schein (2025): (a) rated slant against alpha for each model, (b) per-issue correlation between alpha and slant, (c) per-issue correlation between alpha and essay length](/imgs/blogs/linear-representations-of-political-perspective-emerge-in-large-language-models-fig4.webp)

All three models move monotonically with $\alpha$ in panel (a). With $K \in \{64, 80, 96\}$, the correlation between $\alpha$ and slant is 0.607 for Llama-2, 0.396 for Mistral and 0.381 for Vicuna, and more heads steer more strongly. At $\alpha = 0$ every model averages below the scale midpoint of 4, which the authors read as a default lean toward liberal-sounding text. Panel (b) shows steering works best on immigration and abortion and weakest, though still positive, on free speech and trust in science.

Panel (c) is a side observation: for some topics, notably gun control and climate change, essays steered toward the liberal end run longer. The authors flag it as a lead, not a finding. On the ADVANCE Act, a 2024 nuclear energy law past Llama-2's cutoff, both steered essays support the law but for different stated reasons (emissions and equity at negative $\alpha$, energy security at positive $\alpha$). Both events past the cutoff still steer, at $\rho = 0.479$ and ${0.470}$.

## Experiments and results at a glance

| Experiment | Llama-2-7b-chat | Mistral-7b-instruct | Vicuna-7b | Baseline or reference |
| --- | --- | --- | --- | --- |
| Best single linear head, lawmakers | 0.854 | 0.846 | 0.861 | Permuted labels about 0.15 |
| Best single MLP head, lawmakers | 0.855 | 0.838 | 0.872 | Linear probe |
| Top-32 ensemble, lawmakers | 0.870 | 0.864 | 0.885 | Single best head |
| Top-32 ensemble, news outlets (no retrain) | 0.798 | 0.764 | 0.720 | n/a |
| Steering, $\alpha$ vs rated slant ($K \geq 64$) | 0.607 | 0.396 | 0.381 | $\alpha = 0$ |
| Non-U.S. parties (Manifesto, N = 411) | 0.531 | not reported | not reported | U.S. lawmakers 0.870 |

All numbers are Spearman correlations from Sections 3, 5, 6 and Appendices A.10 and Tables A2 to A4. Compute is modest: three 7B models, one forward pass per prompt for probing, and 1,134 generations for steering. Appendix A.11 reports the middle-layer pattern replicating on Gemma-2-2b.

**What is load-bearing and might not transfer.** Three choices carry the result. The prompt template ("In 2019, [NAME] said that") puts the model in the act of voicing a specific person. DW-NOMINATE is a single-axis summary of a highly sorted two-party legislature. And the evaluation is rank correlation, which forgives any monotonic distortion. Change any of these, say a multi-party system or a probe that must hit exact values, and the numbers should be expected to fall, as the Manifesto result already shows.

## Critique

**What is strong.** The validation ladder is well built, and each step closes a loophole the previous one left open: held-out lawmakers, a non-linear control, a scrambled-label floor, an R² check, a second labeled population, and a causal intervention. The politically balanced human panel behind the GPT-4o rater is a thoughtful touch for a contested construct.

**What is weak or under-tested.**

- **Selection on the test score.** Heads are ranked by the same cross-validated Spearman that is then reported as the best result. Picking the maximum of 1,024 noisy estimates biases it upward. A nested split (choose heads on one set, report on another) would settle how much.
- **Memorization is only partly addressed.** The lawmaker prompt is little more than a name, so the probe may be reading "this name co-occurs with this voting record" from pre-training. Token-level shading and outlet transfer argue against pure lookup, but outlets are also named entities with documented reputations.
- **One rater, small anchor.** The 0.91 ICC rests on 21 essays from one model. Steering numbers for Mistral and Vicuna lean entirely on GPT-4o.
- **A scale-direction inconsistency.** The GPT-4o prompt in Appendix A.3 lists 1 as strongly conservative and 7 as strongly liberal, while Section 6, Appendix A.6 and Figure A4 treat low values as liberal. The plots are internally consistent, so ratings were presumably recoded, but the paper does not say so.
- **Steering strength is moderate.** A 0.607 correlation on the best model leaves much of the rating variance unexplained, and the effect depends heavily on the issue.

**Missing ablation.** A random-direction steering control at matched norm, and a probe-direction steering control on a *non*-political topic, would show how specific the effect is. So would comparing the per-head probe directions with each other: if the top 32 heads point the "same way" after projection to the residual stream, that is much stronger evidence for one shared axis.

**What would change my mind.** If a nested cross-validation dropped the top-32 lawmaker score well below 0.8, or if steering along a random unit direction at the same $\alpha \hat\sigma$ scale shifted GPT-4o ratings comparably, I would read the paper as "the models know a lot about named politicians" rather than "the models carry a reusable ideology axis".

This also sits next to work showing that [linear representations can drift dramatically over a conversation](/blog/paper-reading/ai-interpretability/linear-representations-in-language-models-can-change-dramatically-over-a-conversation). A probe fitted on single-turn prompts is a snapshot; nothing here tests whether the same heads keep the same meaning twenty turns into a dialogue.

## What I would build with this

These are my extrapolations, not the paper's claims.

1. **A slant monitor for evaluation suites.** Run the token-level readout over a model's answers to a fixed bank of open-ended policy prompts and report the mean and spread per release, as Appendix A.2 does for three models.
2. **A neutrality regression test.** Pair the readout with the steering knob to check whether a fine-tune moved the $\alpha = 0$ baseline, a cheap complement to closed-form survey benchmarks.
3. **Multi-axis probes.** Fit probes on the second DW-NOMINATE dimension, or on Manifesto sub-scales, to see whether the model separates economic and social axes or collapses them into one.

## References

1. Kim, J., Evans, J., and Schein, A. (2025). *Linear Representations of Political Perspective Emerge in Large Language Models.* ICLR 2025 (oral). [arXiv:2503.02080](https://arxiv.org/abs/2503.02080)
2. Li, K., et al. (2023). *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model.*
3. Gurnee, W., and Tegmark, M. (2024). *Language Models Represent Space and Time.*
4. Poole, K. T., and Rosenthal, H. (1985). *A Spatial Model for Legislative Roll Call Analysis.*
5. On this site: [The linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis), [Multi-attribute steering via targeted intervention](/blog/paper-reading/ai-interpretability/multi-attribute-steering-of-language-models-via-targeted-intervention), and [Linear representations can change dramatically over a conversation](/blog/paper-reading/ai-interpretability/linear-representations-in-language-models-can-change-dramatically-over-a-conversation).

---
title: "Controlling Language and Diffusion Models by Transporting Activations"
publishDate: "2025-09-30"
category: "paper-reading"
subcategory: "Computer vision"
tags: ["diffusion-model", "text-to-image", "activation-steering"]
date: "2025-09-30"
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20250930005013.png"
excerpt: "The main contribution: Provide a unifying interpretation of activation steering methods under the framework of optimal transport (OT) ..."
---

![](/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20250930005013.png)

## Motivation

Fine tuning is a common strategy for aligning Generative Models (GMs). As model sizes grow, this process becomes increasingly expensive and can unintentionally reduce performance on other tasks. This challenge has encouraged research into inference time interventions, which provide lighter and more efficient ways to control model behavior.

Existing approaches such as applying constant vector shifts or suppressing specific neurons often fail to preserve the activation distributions that the model learned during training. Because GMs are brittle, these shifts can move activations out of distribution, resulting in undesirable behaviors and weaker overall performance.

To address these issues, the authors propose Activation Transport (AcT), a framework based on optimal transport. AcT steers activations from a source distribution to a target distribution in a principled way, preserving activation distributions while improving controllability and robustness.

The main contribution:

- Provide a unifying interpretation of activation steering methods under the framework of optimal transport (OT).

- Introduce Linear-AcT, an inference-time intervention based on OT that preserves activation distributions, with controllable strength and transport support to avoid out-of-distribution activations.

- Demonstrate that Linear-AcT matches or outperforms existing inference-time methods on LLM tasks such as toxicity mitigation, concept induction, and truthfulness.

- Show that Linear-AcT is also effective for T2I diffusion models in fine-grained style control and concept negation, and adapt ITI for T2I.

- Present the first inference-time intervention that works effectively on both LLMs and Diffusion Models.

## Transporting neural activations

The authors represent the activations of a Generative Model (GM) for a given input sentence $x \in \mathcal{S}$ as a tensor

$$
\mathbb{R}^{M \times L \times K},
$$

where

- $M$ is the number of activations per layer,
- $L$ is the number of layers,
- $K$ is the number of tokens decoded.

To simplify, the authors reduce the token dimension $K$ into one value using a pooling operator $\phi$. This gives a mapping

$$
Z : \mathcal{S} \to \mathbb{R}^{M \times L},
$$

which turns a sentence into a matrix of activation statistics.

The authors then consider two probability distributions on sentences, $p$ and $q$. For example, $p$ could represent toxic sentences, while $q$ represents non-toxic sentences.

Through the pushforward operator $Z_{\sharp}$, the distributions become

$$
\mu := Z_{\sharp}p, \quad \nu := Z_{\sharp}q.
$$

With samples $x^1, \ldots, x^n \sim p$ and $y^1, \ldots, y^n \sim q$, the activations are

$$
a^i := Z(x^i), \quad b^i := Z(y^i).
$$

Thus, there are $n+n$ activation matrices of size $M \times L$. The goal is to learn a transport map

$$
T : \mathbb{R}^{M \times L} \to \mathbb{R}^{M \times L},
$$

that approximately pushes $\mu$ to $\nu$, meaning

$$
T_{\sharp}\mu \approx \nu.
$$

### Low budget estimators for transport maps

Since modern GMs have millions of activations, directly computing transport maps is infeasible. The challenges are:

- **Curse of dimensionality**: Estimating high-dimensional OT maps is very difficult. As the number of dimensions grows, the estimates often fail to generalize well.
- **Layer composition**: In practice, the method applies transport maps layer by layer. That means the activations for a given layer are already influenced by maps from previous layers, making things even more complicated.

To handle these problems, the authors choose a simpler approach:

- Instead of learning one huge transport map for all activations, they factorize it into independent univariate maps.
- In other words, treat each activation dimension separately and learn a small 1D map for it.

So, if there are $M \times L$ activations, the method just learns $M$ $L$ small univariate transport maps.

Each of these maps takes the marginal distribution of $\mu$ (source) in that coordinate and maps it to the marginal distribution of $\nu$ (target).

For two univariate distributions $\rho, \tau \in \mathcal{P}(\mathbb{R})$, the optimal transport map $T$ under any submodular cost is

$$
T^* = Q_{\tau} \circ F_{\rho},
$$

where $F_{\rho}$ is the **CDF** of $\rho$, and $Q_{\tau}$ is the **quantile function** of $\tau$.

To make the method practical, the authors simplify the transport map to an **affine function**:

$$
T(a; A, B) = \omega a + \beta,
$$

where $A = (a^1, \ldots, a^n)$ and $B = (b^1, \ldots, b^n)$.

The closed-form solutions for $\omega$ and $\beta$ are:

$$
\omega = \frac{\sum_i \tilde{a}^{(i)} \tilde{b}^{(i)}}{\sum_i (\tilde{b}^{(i)})^2},
\quad
\beta = m_b - \omega m_a,
$$

with

$$
m_a = \frac{1}{n}\sum_i a^i, \quad
m_b = \frac{1}{n}\sum_i b^i,
$$

and

$$
\tilde{a}^{(i)} = a^{(i)} - m_a, \quad
\tilde{b}^{(i)} = b^{(i)} - m_b.
$$

Here, $a^{(i)}$ and $b^{(i)}$ denote **sorted values** of $a$ and $b$.

If the distributions are Gaussian with equal variance, then the affine map reduces to a simple **mean shift**:

$$
T(a) = a + m_b - m_a.
$$

This is referred to as **Mean-AcT**.

![](/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20250930011753.png)

The authors test different mapping methods on toy Gaussian data. They find that **mean-shift style methods** (like ACTADD, ITI-C, Mean-AcT) can **overshoot or undershoot**, meaning the transformed samples can end up **outside the target distribution** (out-of-distribution, OOD). In contrast, **Linear-AcT** shows a **good balance**:

- It maps distributions well.
- It has very low computational cost.

Real-world GM activations are usually **unimodal** (one main peak) but with different spread (standard deviation) depending on the behavior. For such cases, a linear map is suitable. If activations were **multimodal** (several peaks), then **non-linear maps** would be necessary, but that is beyond this paper.

The mapping defined in Linear-AcT is learned from **n pairs of samples** (often just a few hundred). This means it’s only a rough approximation of the true transport between distributions.

- Problem: the tails (extreme values) of the source distribution $\mu$ have very few samples, so errors are larger there.
- If you try to transport those rare “tail” values, you may produce **unexpected or unstable results** (OOD behavior).

So the solution is:

- Only apply transport to values **within the observed support** of the source samples:
  $$
  \mathcal{Q}_o = [\min A, \max A]
  $$
- This ensures you don’t move points that the model has never really “seen.”

Different cases:

- For **mitigation tasks** (like reducing toxicity), they use bounded support $\mathcal{Q}_o$.
- For **induction tasks** (like adding new features), they allow unbounded support:
  $$
  \mathcal{Q}_\infty = (-\infty, \infty).
  $$

![](/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20251001144036.png)

### Sequential iterative maps

Instead of simply applying univariate transport maps independently to each activation, the authors point out that this ignores the **causal relationship between layers** in a neural network. Specifically, the activations produced at layer $\ell$ become the input to layer $\ell+1$:

$$
a_{m,\ell+1} = f_\ell(a_{m,\ell})
$$

This means that any intervention at one layer directly affects all subsequent layers. Therefore, transport maps cannot be estimated in isolation for each layer without considering this dependency.

To respect this causality, the authors propose estimating transport maps incrementally, layer by layer:

1. Estimate the transport map for the first layer.
2. Apply this map and run inference again to update the activations.
3. Use these updated activations to estimate the map for the second layer.
4. Repeat this process for all layers until every map is estimated.

In this way, each transport map is conditioned on the interventions applied at the earlier layers.

Appendix C shows that causal estimation (sequential layer-by-layer estimation) provides more effective conditioning and achieves better results than estimating all maps simultaneously under the assumption of independence.

![](/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20251001145636.png)

In this paper, the authors use causal estimation for both **Mean-AcT** (mean-shift mapping) and **Linear-AcT** (affine mapping).

The authors describe how to extend an optimal transport (OT) map into an **interpolating transport map**, which provides a smooth way to move between the source distribution $\mu$ and the target distribution $\nu$. Instead of always applying the full transport, they introduce a parameter $\lambda \in [0,1]$ to control the degree of transport:

$$
T(a, \lambda) = (1 - \lambda)a + \lambda T(a).
$$

- When $\lambda = 0$, the mapping returns the original value $a$ (no transport).
- When $\lambda = 1$, it applies the full transport $T(a)$.
- For values between 0 and 1, the output is a **weighted interpolation** between $a$ and its transported version $T(a)$.

This formulation gives users a **continuous and interpretable knob** $\lambda$ to control how strongly a concept appears during generation. It eliminates the need for costly parameter searches and avoids being stuck with fixed, uncontrollable conditioning.

Such interpretability is especially important in applications like diffusion models, where it is difficult to measure model utility. By adjusting $\lambda$, one can fine-tune the strength of intervention with clear semantics: “0% transport” to “100% transport.”

The authors also note that previous methods like **ACTADD, CAA, or ITI-C** include a conditioning strength parameter, but those are usually applied in the form

$$
T(a, \lambda) = a + \lambda \beta.
$$

In that setup, $\lambda$ is unbounded and depends on the choice of $\beta$, which makes it **harder to interpret, less consistent across models, layers, and tasks, and not robust**. In contrast, the interpolation approach based on OT ensures that $\lambda$ has a consistent meaning within the bounded interval $[0,1]$.

### Generalization of prior inference-time interventions work

![](imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20251001150609.png)

Table above compares different inference-time intervention methods from the literature. The main point the authors make is that **all these methods can actually be seen as special cases of Linear-AcT**, because they are all just forms of linear transport.

Most existing methods modify activations by adding a **bias term**. What differs between them is **how this bias is computed**. But in all cases, the conditioning strength parameter $ \lambda $ multiplies this bias, making it unbounded and often hard to interpret.

What makes AcT different:

- **Linear transformation, not just bias-addition:** AcT applies a proper linear transformation $ T(a) = \omega a + \beta $ to activations, which better preserves the internal distribution.
- **Interpolatable parameter $ \lambda $:** With AcT, $ \lambda $ interpolates smoothly between the original activation $ a $ and the transported one $ T(a) $. This keeps $ \lambda \in [0,1] $ and makes it interpretable as a percentage of transport strength.
- **Support selection:** Other methods rely on heuristics to choose the support set of activations (for example, using observed input ranges $ \mathcal{Q}\_o $), while AcT simply uses all activations or $ \mathcal{Q}\_o $.

Relation to Mean-based methods:

- Methods like CAA, ITI-M, and Mean-AcT all use a **difference in means** to define their bias, i.e. $ \beta = m_b - m_a $.
- The authors group these into a single family of approaches and show that Mean-AcT already captures their essence, but with the added benefit of an interpretable $ \lambda $.

Another technical difference lies in how activations across tokens are aggregated:

- **Many earlier methods:** use **the last token only**, written as $ \phi(z) = z[\ldots, -1] $.
- **Det$_{\text{zero}}$** and **AURA:** use **max pooling**, i.e. $ \phi(z) = z.\max(-1) $.
- **AcT:** uses the **mean across tokens**, i.e. $ \phi(z) = z.\text{mean}(-1) $, which the authors found to be **more robust**.

![](/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20251001152617.png)

## Experiments on LLMs

### Toxicity Mitigation in LLMs

![](/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20251009004212.png)

The experiments evaluate how well AcT (Activation Transport) mitigates toxic language generation in large language models (LLMs) like Gemma2-2B and Llama3-8B.

- Linear-AcT achieves the best toxicity reduction, lowering toxic outputs by up to 7.5× on Gemma2-2B and 4.3× on Llama3-8B, with minimal impact on perplexity (PPL) and reasoning accuracy (MMLU).

- Compared to other methods (AURA, AcTADD, and ITI-C), Linear-AcT and Mean-AcT provide the most stable and robust performance across layer choices and the intervention strength parameter (λ).

- ITI-C also performs strongly (up to 5.6× reduction) but is highly sensitive to λ and layer type.

- AURA shows moderate improvement (up to 3.1×), while AcTADD provides the weakest mitigation.

In short, Linear-AcT is both effective and robust, showing strong toxicity reduction without hurting model fluency or general performance.

### Concept Induction in LLMs

![](/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20251009004250.png)

This section tests AcT’s ability to induce specific semantic concepts into model generations (e.g., “football”, “cloud”, “baby”).

- Linear-AcT can reliably inject arbitrary concepts at a consistent λ = 1, maintaining low perplexity and strong concept presence in generated text.

- Compared to ITI-C and Mean-AcT, Linear-AcT exhibits smoother and more interpretable control of concept strength, aligning with the theoretical optimal transport (OT) formulation.

- ITI-C peaks at λ ≈ 2.5, but is less stable across tasks and layers.

Thus, Linear-AcT generalizes well for concept control, achieving high concept presence (p(yes) ≈ 0.87) while keeping text quality consistent.

### Inducing Truthfulness in LLMs

![](/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20251009004342.png)

The final experiment applies AcT to improve truthfulness in language generation using the TruthfulQA benchmark.

- Both Linear-AcT and Mean-AcT significantly increase factual accuracy (MC1 and MC2) compared to baselines.

- On Gemma2-2B, Linear-AcT raises MC1 by about +5%, and on Llama3-8B, by nearly +8%.

- The improvement comes with minimal trade-off in general reasoning ability (MMLU decreases by less than 0.5%).

- Again, λ = 1 is found to be a stable and effective default setting, corresponding to full transport in AcT’s formulation.

Overall, AcT—especially Linear-AcT—can induce truthfulness and desirable behaviors in LLMs while preserving general performance and text fluency.

## Controlling image diffusion models

### Fine-Grained Style Control

![](/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20251009004823.png)

A key challenge in T2I models is precisely controlling visual style attributes (e.g., sketchiness, impressionism, watercolor) without distorting image semantics.

- Linear-AcT significantly improves fine-grained control, increasing the presence of the desired style from ~12% to ~95% on SDXL while maintaining ~80% similarity to the original image (measured by CLIP similarity).

- The optimal strength λ = 1 yields the best results for both SDXL and FLUX, consistent with AcT’s theoretical framework and prior LLM experiments.

- ITI-C can also achieve style control but performs inconsistently across models, requiring different λ values (λ = 2 for SDXL, λ = 1 for FLUX) and often exaggerating visual traits or distorting semantics.

### Concept Negation

![](/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20251009004847.png)

Concept negation refers to preventing the model from generating undesired objects (e.g., instructing it not to draw a “pink elephant”). This is a persistent challenge for diffusion models like SDXL and DALL·E 3.

- The study found that Linear-AcT effectively suppresses unwanted concepts while preserving image semantics, performing better than ITI-C and native “negative prompt” mechanisms in SDXL and FLUX.

- Quantitatively, Linear-AcT achieved higher CLIP similarity scores (indicating better semantic preservation) and lower unintended concept presence than ITI-C.

- In contrast, ITI-C required stronger interventions (higher λ) to suppress undesired concepts, which often caused semantic degradation or over-suppression.

## Some stuff that I interest when reading this paper :))

The paper takes a brilliant and refreshing approach to model control. Instead of using ad-hoc vector shifts or neuron suppression, it frames the whole problem under Optimal Transport (OT). This is a principled mathematical view that connects and generalizes all previous activation-steering methods. I find this idea elegant because it not only unifies earlier techniques but also gives a clear, interpretable way to control model behavior through a single parameter λ that smoothly adjusts intervention strength.

What really stands out is how Linear-AcT performs across both language and image diffusion models. It manages to reduce toxicity, induce truthfulness, and even control fine-grained visual styles or remove unwanted concepts, all with consistent results and minimal side effects. This kind of cross-domain robustness is rare, and it shows that AcT isn’t just a theoretical idea but a genuinely practical tool for generative model control.

That said, I do think the paper relies on some strong assumptions. The linear mapping between activations is a bit too simplistic for such high-dimensional, nonlinear spaces. Also, treating each activation dimension independently ignores important correlations inside neural layers. So while AcT is a powerful and interpretable first step, it still feels like an approximation, maybe a good one, but not yet the full story of how activations could be transported in complex generative systems.

## References

1. [CONTROLLING LANGUAGE AND DIFFUSION MODELS BY TRANSPORTING ACTIVATIONS](https://arxiv.org/pdf/2410.23054)

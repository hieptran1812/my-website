---
title: "Unveiling Concept Attribution in Diffusion Models (Method review)"
publishDate: "2025-12-08"
category: "paper-reading"
subcategory: "AI Interpretability"
tags: ["diffusion-model"]
date: "2025-12-08"
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/unveiling-concept-attribution-in-diffusion-models-20251208174653.png"
excerpt: ""
---

## TLDR

This post summarizes a paper that makes diffusion models more interpretable by assigning an attribution score to individual model components, showing whether each component encourages or suppresses a given concept. 

The authors introduce CAD, a fast framework that estimates each component’s counterfactual effect using a linear approximation via gradients, avoiding expensive brute force ablations. 

With these scores, they show concept knowledge is localized and comes in two forms: positive components that induce a concept and negative components that inhibit it. They then use this structure for lightweight editing at inference time: CAD Erase removes a concept by zeroing the top positive components, and CAD Amplify strengthens a concept by zeroing the top negative components, with experiments demonstrating these edits are effective while mostly preserving other behaviors.

## Motivation

The paper addresses the lack of interpretability in diffusion models. Specifically, it aims to answer the general question: "How do components in diffusion models contribute to a generated concept?"

Existing work on interpreting generative models often focuses only on:

- Knowledge storage (components responsible for generating concepts).
- Coarse-grained components, such as entire layers (e.g., UNet components).

This limited perspective, inspired by the "distributed hypothesis," overlooks more subtle properties and other types of components that also influence the output. The authors seek a holistic understanding of how model components activate concepts (e.g., objects, styles, or explicit contents) by considering both positive and negative contributions.

## Contribution

- **Proposing Component Attribution for Diffusion Model (CAD):** A comprehensive framework that efficiently computes the attribution scores of diffusion model components using a linear counterfactual estimator.
- **Confirming and Extending the Localized Nature:** CAD confirms the existence of concept-inducing (positive) components and is the first work to uncover the existence of concept-amplification (negative) components, providing a more holistic understanding.
- **Developing Lightweight Editing Algorithms:** Leveraging the localized nature of positive and negative components, the authors develop CAD-Erase for concept erasing and CAD-Amplify for concept amplification.
- **Empirical Analysis and Validation:** The paper includes extensive experiments to analyze and evaluate the effectiveness and practicality of the proposed editing algorithms.

## Concept Attribution in Diffusion Models

### Decomposing Knowledge in Diffusion

This authors proposes the CAD framework for concept attribution by formulating how individual components (parameters $\mathbf{w}$) contribute to a generated concept $c$.

- **Quantifying Contribution:** Concept generation is measured by a function $J(c, \mathbf{w})$.
- **Counterfactual Measurement:** The contribution of a component $w_i$ is determined by measuring the counterfactual effect, how $J(c, \mathbf{w})$ changes when $w_i$ is "knocked out" (set to 0).
- **Linear Approximation:** The counterfactual function $g(\mathbf{O}_{\mathbf{\tilde{w}}}; c) = J(c, \mathbf{\tilde{w}})$ is approximated using a linear model:

![](/imgs/blogs/unveiling-concept-attribution-in-diffusion-models-20251208182713.png)

The coefficients $\alpha_{c, i}$ then represent the contribution of each component $w_i$ to the concept $c$.

### CAD: Component Attribution for Diffusion Model

Figuring out how much each piece of a diffusion model matters is hard if we do it the “brute force” way, because we'd have to mask pieces on and off, generate tons of samples, and train a regression model. That’s super slow for diffusion models.

This part says: figuring out how much each piece of a diffusion model matters is hard if we do it the “brute force” way, because we'd have to mask pieces on and off, generate tons of samples, and train a regression model. That’s super slow for diffusion models.

So instead, they use a quick math shortcut: approximate the impact of turning off a component using a first order Taylor approximation. In practice, a component’s contribution can be estimated as its value times the gradient of the objective with respect to it $w_i \cdot \frac{\partial J}{\partial w_i}$.

Big win: we can get these importance scores with basically one forward pass and one backward pass, instead of tons of expensive runs.

## Editing Diffusion Models with CAD

### Localizing and Erasing Knowledge

Once we can score which parts of a diffusion model “cause” a concept, we can edit the model at inference time to remove or strengthen that concept.

The model has some components that push it toward generating a concept and some that push it away. If we identify those components, we can tweak the output without retraining.

Two simple editing tricks:

1. CAD Erase: Find the top k components that most strongly support the target concept, then set them to zero -> the model becomes less likely to generate that concept.

2. CAD Amplify: Find the top k components that most strongly oppose the concept, then set them to zero -> the model becomes more likely to generate that concept.

Why this matters? The authors argue that concept related knowledge is often localized, meaning only a small number of components really matter for that concept. So removing just those few can erase the concept while mostly leaving other concepts alone.

![](/imgs/blogs/unveiling-concept-attribution-in-diffusion-models-20251208185257.png)

Why not just use the training loss? The authors say a simple idea is to use the model’s normal training loss as $J$. But past work on concept erasing found that optimizing this can work poorly, so they use a different objective instead.

They define a concept attribution objective:

$$
J_{c_b}(c,w)=\mathbb{E}_{x_t,t,\epsilon}\left|\Phi(x_t,c_b,t;w).\text{sg()}-\Phi(x_t,c,t;w)\right|^2
$$

Meaning: compare the model’s predicted noise when conditioned on the base condition $c_b$ versus when conditioned on the target concept $c$, and measure the squared difference.

$c$ is the target concept, like “parachute”

$c_b$ is the base condition, like an empty prompt

$\text{sg}()$ is stop gradient, so gradients do not flow through the base prediction term

$\Phi(\cdot)$ is the model’s noise prediction

They try to make the model’s concept conditioned noise prediction look close to the unconditional one. If conditioning on “parachute” does not change the noise prediction much, the denoising process will not move toward images that express the parachute concept. That makes it easier to erase the concept cleanly.

### Amplifying Knowledge in Diffusion Models

The authors say the model has different internal components that affect whether a concept appears in the generated image. Some components have positive influence and help produce the concept. Others have negative influence and suppress it, making the concept less likely.

Their Hypothesis 2 is: negative components exist, and if we remove or weaken them, the model will show stronger knowledge of the concept and generate it more often.

![](/imgs/blogs/unveiling-concept-attribution-in-diffusion-models-20251209134302.png)

They then introduce a method called CAD Amplify. It tries to amplify a target concept by finding and ablating those negative components, using a training loss objective. The scatter plot shows that the attribution scores predicted by their method correlate moderately with the true objective values, meaning the scores are somewhat reliable indicators.

![](/imgs/blogs/unveiling-concept-attribution-in-diffusion-models-20251209134745.png)

## References

1. [Unveiling Concept Attribution in Diffusion Models](https://arxiv.org/pdf/2412.02542)

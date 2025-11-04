---
title: "Steering Large Language Models using Conceptors: Improving Addition-Based Activation Engineering (The idea)"
publishDate: "2025-11-03"
category: "paper-reading"
subcategory: "AI Interpretability"
tags: ["model-interpretation", "activation-engineering"]
date: "2025-11-03"
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/steering-large-language-models-using-conceptors-improving-addition-based-activation-engineering-20251103232342.png"
excerpt: "The authors propose using conceptors (a type of matrix operation) instead of vector addition to steer model activations. This allows smoother and more controlled manipulation."
---

## Motivation

Large language models (LLMs) can generate powerful results, but they sometimes spread misinformation, amplify bias, or behave undesirably. Traditional ways to control them, like reinforcement learning from human feedback (RLHF), fine-tuning, or prompt engineering,.. are costly and unreliable. A newer idea called activation engineering modifies the internal activations of models directly, but its usual approach (adding or subtracting activation vectors) often gives inconsistent results.

The paper seeks a more reliable, efficient, and flexible method to steer LLM behavior without retraining or heavy computation. The contribution are:

1. **New Steering Method**: The authors propose using conceptors (a type of matrix operation) instead of vector addition to steer model activations. This allows smoother and more controlled manipulation.
2. **Application to Function Vectors**: They test this technique on GPT-NeoX and GPT-J, showing it works effectively for learned function vectors.
3. **Combining Steering Goals**: They also explore how Boolean algebra on conceptors can merge multiple steering objectives in GPT-J, allowing flexible control over different model behaviors.

## Conceptors as Steering Matrices

### What a conceptor is?

A conceptor is a neuro computational mechanism that captures and controls the state space of neural activations. Mathematically it is a positive semi definite matrix, so it represents an ellipsoid in a high dimensional space. That ellipsoid encodes the overall shape and extent of the activation cloud, not just a single location.

![](/imgs/blogs/steering-large-language-models-using-conceptors-improving-addition-based-activation-engineering-20251104102636.png)

### Limits of point based steering

In activation engineering, a common practice is to average many activation vectors to obtain one steering vector, or to subtract the mean of a negative set from the mean of a positive set.

This yields a single point or a single direction. It tells you where the center is, but discards how the activations spread and co vary across dimensions.

Because real activations are dispersed and correlated, this point based summary can steer unreliably and react inconsistently across inputs.

### But, why does conceptor takes the shape of an ellipsoid?

A conceptor takes the shape of an ellipsoid because it represents how activation vectors from a neural network are distributed in high dimensional space.

When we collect activation vectors from a model, each vector can be viewed as a point in space. These points usually form a cloud rather than clustering at a single location. The goal of the conceptor is to describe this cloud — its overall spread and orientation.

The conceptor is derived from the correlation matrix
$$R = \frac{X^T X}{n}$$
where $X$ contains the activation vectors. This matrix captures how each dimension varies and how different dimensions co vary. When $R$ is decomposed into its eigenvalues and eigenvectors,
$$R = Q \Lambda Q^T$$
each eigenvector shows the direction of maximum variation, and each eigenvalue tells how strong that variation is.

Geometrically, these directions and magnitudes form an ellipsoid:

- Long axes correspond to directions where activations vary widely.
- Short axes correspond to directions with little variation.
- The tilt of the ellipsoid shows how the dimensions are correlated.

Because the conceptor $C$ is a positive semi definite matrix derived from $R$, it inherits this ellipsoidal geometry. In simple terms, the conceptor defines a soft “boundary” — an ellipsoid that encloses most of the activation patterns.

This shape is crucial for steering because it allows the conceptor to project new activations smoothly into a natural region of the model’s activation space. Instead of forcing the model toward one fixed direction, it ensures the activations remain consistent with the learned distribution, producing stable and realistic behavior.

### Formation and Use of Conceptors

A conceptor $C$ is computed by optimizing a balance between two objectives: maintaining the ability of $C$ to reconstruct the original activation set $X$, and limiting its complexity through a regularization parameter $\alpha$, known as the _aperture_. The closed-form solution of the conceptor is given by

$$C(R, \alpha) = R(R + \alpha^{-2}I)^{-1}$$

where $R = \frac{X^T X}{n}$ is the correlation matrix of the activation vectors.

When $\alpha$ is large, the conceptor allows more signal variation, meaning $C$ approaches the identity matrix and becomes more flexible. When $\alpha$ is small, the conceptor becomes tighter, keeping only the most important and consistent patterns, as $C$ approaches the zero matrix. This aperture parameter therefore controls the trade-off between _generalization_ and _precision_.

To steer a large language model using conceptors, the process typically follows three steps.

- First, activation vectors $h_t^{(p)}$ are collected from the model on a set of representative examples.

- Second, a conceptor $C_f$ is computed from these activations.

- Finally, when steering, the model’s new activations are obtained by projecting them through the conceptor using

$$h_t' = \beta_c C_f h_t$$

where $\beta_c$ is a scaling factor that adjusts the steering intensity.

Conceptors provide a softer and more stable steering mechanism compared to the traditional method of vector addition. Instead of forcing the model to move strictly along a predefined direction, a conceptor projects activations into a subspace that corresponds to the desired behavioral pattern.

This projection maintains the internal structure of activations while guiding the model smoothly and consistently toward the intended output behavior.

### Boolean Operations on Conceptors

Conceptors can be combined using Boolean-like operations—OR, AND, and NOT—which make it possible to merge or refine multiple steering targets. These operations were originally introduced by Jaeger and allow flexible control over conceptor behavior.

**1. OR Operation (Merging Concepts)**

The OR operation ($C_1 \lor C_2$) merges two conceptors $C_1$ and $C_2$, effectively combining the activation regions that each conceptor represents.

If $C_1$ and $C_2$ are computed from covariance matrices $R_1$ and $R_2$, the merged conceptor can be written as:

$C_1 \lor C_2 = (R_1 + R_2)(R_1 + R_2 + \alpha^{-2}I)^{-1}$

or equivalently,

$C_1 \lor C_2 = \left(I + (C_1(I - C_1)^{-1} + C_2(I - C_2)^{-1})^{-1}\right)^{-1}$

This means the OR operation broadens the conceptor’s coverage, it merges the activation spaces of $C_1$ and $C_2$, capturing both patterns.

**2. NOT Operation (Inverting a Concept)**

The NOT operation produces a conceptor that represents the complementary activation space, i.e., the directions not covered by the original conceptor. It is defined as:

$\neg C = R^{-1}(R^{-1} + \alpha^{-2}I)^{-1}$

which simplifies to

$\neg C = I - C$

Intuitively, this operation inverts the conceptor so that it highlights the parts of the activation space that the original $C$ suppresses.

**3. AND Operation (Intersection of Concepts)**

The AND operation ($C_1 \land C_2$) captures the overlap between two conceptors, representing shared activation patterns. Using De Morgan’s law $;a \land b = \neg(\neg a \lor \neg b)$, it can be computed as:

$C_1 \land C_2 = (R_1^{-1} + R_2^{-1})^{-1}((R_1^{-1} + R_2^{-1})^{-1} + \alpha^{-2}I)^{-1}$

or equivalently,

$C_1 \land C_2 = (C_1^{-1} + C_2^{-1} - I)^{-1}$

This operation narrows the focus to the intersection of the activation spaces captured by $C_1$ and $C_2$.

Key Insight

These Boolean operations allow conceptors to be compositional—you can combine, invert, or intersect different steering targets in a mathematically consistent way. This makes conceptors powerful tools for steering LLMs along multiple, overlapping, or complementary behaviors with fine-grained control.

### Computational Complexity of Conceptor Steering

The main cost of conceptor steering comes from computing the conceptor matrix, which requires inverting and multiplying the activation correlation matrix $R = X X^T / n$. This has a complexity of $O(n^3)$ but is done offline, while the matrix itself uses $O(n^2)$ memory (about 17 MB for $n=2048$, 67 MB for $n=4096$, 268 MB for $n=8192$).

During inference, steering adds one extra multiplication $C x$, but this cost can be removed by fusing the conceptor with existing weight matrices ($W_x^C = W_x C$). The runtime overhead is minimal for single-sample generation and only matters for large batch sizes.

## References

1. [Steering Large Language Models using Conceptors: Improving Addition-Based Activation Engineering](https://arxiv.org/pdf/2410.16314)

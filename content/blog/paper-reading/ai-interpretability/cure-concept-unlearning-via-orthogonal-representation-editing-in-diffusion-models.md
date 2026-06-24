---
title: >-
  CURE: Concept Unlearning via Orthogonal Representation Editing in Diffusion
  Models
publishDate: '2025-12-22'
category: paper-reading
subcategory: AI Interpretability
tags:
  - activation-steering
  - mechanistic-interpretability
  - diffusion-model
  - neurips-2025
date: '2025-12-22'
author: Hiep Tran
featured: false
image: >-
  /imgs/blogs/cure-concept-unlearning-via-orthogonal-representation-editing-in-diffusion-models-20251222161926.png
excerpt: ''
---

![](/imgs/blogs/cure-concept-unlearning-via-orthogonal-representation-editing-in-diffusion-models-20251222154300.png")

## TLDR

The paper presents CURE, a method that makes diffusion models forget specific unwanted concepts such as unsafe content, copyrighted styles, or private identities.

Instead of retraining the model, CURE directly edits the model weights in one fast step. Its core idea, called Spectral Eraser, uses linear algebra to isolate and remove features linked to the target concept while keeping other abilities intact. The method is simple, efficient, and takes only a few seconds, achieving stronger and cleaner concept removal than prior approaches with minimal impact on overall image generation quality.


## Method

### Preliminaries

### Concept Unlearning via Orthogonal Representation Editing


## Experiments

## My thoughts

## Appendix

## References

1. [CURE: Concept Unlearning via Orthogonal Representation Editing in Diffusion Models](https://openreview.net/forum?id=4ad6c490a670e8ac6a02e05c9a1fc800e6a3d4bc)
2. [CURE: Concept Unlearning via Orthogonal Representation Editing in Diffusion Models (arXiv:2505.12677v2)](https://arxiv.org/abs/2505.12677v2)

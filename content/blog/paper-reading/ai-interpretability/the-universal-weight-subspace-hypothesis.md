---
title: The universal weight subspace hypothesis
publishDate: '2025-12-06'
category: paper-reading
subcategory: AI Interpretability
tags:
  - model-interpretation
date: '2025-12-06'
author: Hiep Tran
featured: false
image: /imgs/blogs/the-universal-weight-subspace-hypothesis-20251216112621.png
excerpt: ''
---

## TLDR

Researchers found that even though different AI models (like Mistral or LLaMA) are trained on completely different tasks, they all tend to organize their information in the same mathematical "neighborhoods" (called low-dimensional subspaces).

They tested over 1,100 models, including 500 Vision Transformers and 550 Large Language Models. Regardless of how the model started (initialization) or what it learned (task), the models converged to the same shared structures. Most of the important information in these massive models is actually captured in just a few key directions, rather than being spread out randomly.

This discovery is exciting because it suggests we can make AI much better in the following ways:

- **Model Merging**: It becomes easier to combine two different AI models into one.
- **Efficiency**: We can develop algorithms that require less power and data, making AI cheaper to run.
- **Sustainability**: By making training more efficient, we can significantly reduce the carbon footprint (environmental impact) of AI technology.

## Motivation



## Contribution

## Notation, definitions and theoretical analysis

## Analysis

### Analysis methodology


### Result from joint subspace's analysis

#### Lower-rank joint subspaces in CNN, LoRA and Finetuned models

### Low rank shared universal subspaces in classical weights

#### Finding universal subspaces and applying them to future tasks

## References

1. [The universal weight subspace hypothesis (arXiv:2512.05117v2)](https://arxiv.org/abs/2512.05117v2)

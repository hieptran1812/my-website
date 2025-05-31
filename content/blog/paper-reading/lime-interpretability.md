---
title: "LIME: Local Interpretable Model-Agnostic Explanations"
excerpt: "Deep dive into LIME, a technique for explaining predictions of any machine learning classifier in an interpretable manner."
category: "paper-reading"
subcategory: "AI Interpretability"
tags: ["AI Interpretability", "LIME", "Explainable AI", "Machine Learning"]
date: "2024-10-18"
readTime: "22 min read"
featured: true
author: "Hiep Tran"
---

# LIME: Local Interpretable Model-Agnostic Explanations

## Overview

LIME (Local Interpretable Model-agnostic Explanations) provides a novel approach to understanding individual predictions of black-box machine learning models by learning local interpretable models.

## Core Methodology

### Local Surrogate Models

- Generate perturbations around the instance of interest
- Train interpretable models on these perturbations
- Use the surrogate model to explain the original prediction

### Model-Agnostic Approach

LIME works with any machine learning model:

- Neural networks
- Random forests
- Support vector machines
- Any other black-box classifier

## Applications

### Text Classification

- Highlight important words/phrases
- Show feature contributions
- Explain sentiment analysis decisions

### Image Classification

- Identify important image regions
- Visualize decision boundaries
- Understand CNN predictions

## Limitations and Extensions

While LIME provides valuable insights, it has limitations:

- Local explanations may not reflect global behavior
- Perturbation strategy affects explanation quality
- Computational overhead for real-time applications

## Impact on Explainable AI

LIME has become a foundational tool in the explainable AI toolkit, inspiring numerous follow-up works and practical applications in high-stakes domains.

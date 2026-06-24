---
title: Evolutionary Strategies lead to Catastrophic Forgetting in LLMs
publishDate: '2026-02-01'
category: paper-reading
subcategory: AI Interpretability
tags:
  - ai-interpretability
  - evolutionary-strategies
  - catastrophic-forgetting
  - paper-reading
  - optimization
date: '2026-02-01'
author: Hiep Tran
featured: false
image: ''
excerpt: >-
  Research examining how evolutionary strategies, when applied to large language
  models, can unexpectedly lead to catastrophic forgetting and degrade
  previously learned capabilities.
---

## Introduction

Evolutionary strategies (ES) have emerged as an alternative optimization method for training and fine-tuning large language models. While ES offers potential advantages like parallelization and gradient-free optimization, this paper reveals a critical drawback: they can lead to catastrophic forgetting, where models lose previously acquired knowledge and capabilities.

## Background

### Evolutionary Strategies in LLMs

Evolutionary strategies optimize model parameters by:

- Evaluating multiple parameter perturbations in parallel
- Selecting and combining successful variations
- Iterating without explicit gradient computation

### Catastrophic Forgetting

Catastrophic forgetting occurs when a neural network, upon learning new information, abruptly loses previously learned knowledge. This phenomenon is particularly problematic for large language models that need to maintain broad capabilities.

## Key Findings

The research demonstrates that:

- **ES-Induced Forgetting**: Models optimized with evolutionary strategies show significantly higher rates of catastrophic forgetting compared to gradient-based methods
- **Performance Degradation**: Tasks that were previously well-learned can see dramatic performance drops after ES-based fine-tuning
- **Scale Dependency**: The forgetting effect becomes more pronounced as model size and population size in ES increase

## Why Does This Happen?

The paper proposes several mechanisms:

1. **Lack of Gradient Information**: ES doesn't directly use gradient information that helps preserve existing knowledge
2. **Population-Based Sampling**: Random perturbations may accidentally disrupt critical parameter configurations
3. **Selection Pressure**: Optimizing for specific tasks creates selection pressure that can override multi-task representations

## Experimental Results

The authors demonstrate catastrophic forgetting through:

- Benchmark evaluations showing capability loss across multiple domains
- Comparison with gradient descent methods (SGD, Adam)
- Analysis of representation stability before and after ES optimization

## Mitigation Strategies

Potential solutions include:

- **Hybrid Approaches**: Combining ES with gradient-based regularization
- **Memory Mechanisms**: Incorporating experience replay or elastic weight consolidation
- **Constrained Search**: Limiting the parameter space ES can explore
- **Multi-Objective Optimization**: Explicitly optimizing for retention alongside new task performance

## Implications

This research has important implications for:

- **LLM Training**: Caution needed when applying ES to pre-trained models
- **Fine-tuning Strategies**: Alternative methods may be safer for preserving capabilities
- **Optimization Research**: Need for forgetting-aware evolutionary algorithms
- **Model Deployment**: Importance of continuous evaluation across all capabilities

## Conclusion

While evolutionary strategies offer appealing properties for LLM optimization, this work reveals their susceptibility to causing catastrophic forgetting. Understanding and mitigating this phenomenon is crucial for developing robust and reliable evolutionary approaches to language model training.

The findings underscore the importance of considering knowledge retention when designing optimization algorithms for large-scale models with diverse capabilities.

## References

1. [Evolutionary Strategies lead to Catastrophic Forgetting in LLMs (arXiv:2601.20861v1)](https://arxiv.org/abs/2601.20861v1)
2. [Paper details to be added]

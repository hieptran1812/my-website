---
title: >-
  Do Natural Language Descriptions Of Model Activations Convey Privileged
  Information?
publishDate: '2025-11-05'
category: paper-reading
subcategory: AI Interpretability
tags:
  - model-interpretation
date: '2025-11-05'
author: Hiep Tran
featured: false
image: >-
  /imgs/blogs/do-natural-language-descriptions-of-model-activations-convey-privileged-information-20251105210200.png
excerpt: ''
---

## Motivation

The paper is motivated by a key challenge in large language model (LLM) interpretability research — how to understand the internal representations of LLMs, which are often “opaque,” meaning not easily interpretable by humans. A recent approach called verbalization aims to decode these internal activations into natural language, typically using a second LLM (the verbalizer) to describe the behavior of the target model.

However, the authors question whether this process truly reveals the target model’s inner workings. They observe that in many cases, the verbalizer might rely on its own background knowledge rather than accessing privileged information from the target model’s activations. 

As a result, the generated explanations may not accurately reflect the target model’s reasoning, making the interpretation potentially unfaithful or misleading.


![](/imgs/blogs/do-natural-language-descriptions-of-model-activations-convey-privileged-information-20251105212658.png)

For example: 

M₁ is the original model (the target model) that we want to understand. M₂ is the verbalizer, a model used to explain M₁’s inner workings in natural language.

The input (x_input) to M₁ is the sentence “My name is Alice.”

We then ask M₂: “What is the country of hˡ?” where hˡ represents M₁’s internal activation.

The goal is to determine whether M₂ truly reads the hidden information encoded in M₁ or just guesses based on its own background knowledge.

In case (a), the verbalizer uses privileged information.

Here, the verbalizer genuinely interprets M₁’s internal activations and identifies that M₁ has encoded the fact “Alice is from the United States.”

M₂ correctly answers: “The country of Alice is the United States.”

This means M₂ actually accessed and described internal knowledge from M₁, information not present in the original input text. This is the ideal scenario where the verbalizer truly helps us understand how M₁ represents knowledge internally.

In case (b), the verbalizer does not use privileged information.

In this case, the verbalizer fails to access M₁’s internal activations and instead relies on its own prior knowledge (for instance, M₂ thinks Alice is from the UK because in its training data, the name Alice often co-occurs with the UK).

M₂ gives the wrong answer: “The country of Alice is the UK.”

Here, M₂ does not actually read M₁’s internal state but simply guesses based on its own associations.
Therefore, the verbalizer is not reliable for interpretability purposes because it does not reflect the true behavior of M₁.

## Contribution

The paper contributes three main points.

First, it identifies that current verbalization benchmarks are flawed because verbalizers can score well without accessing the target model’s internal information.

Second, through controlled experiments, the authors show that verbalizers often rely on their own knowledge instead of the target model’s behavior.

Third, they propose a new evaluation framework to detect when a verbalizer adds its own knowledge, revealing that most verbalizations come from the verbalizer itself rather than the target model.


## References

1. [Do Natural Language Descriptions Of Model Activations Convey Privileged Information? (arXiv:2509.13316v3)](https://arxiv.org/abs/2509.13316v3)
2. [Do Natural Language Descriptions Of Model Activations Convey Privileged Information?](https://arxiv.org/pdf/2509.13316)

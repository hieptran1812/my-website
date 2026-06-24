---
title: 'AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders'
publishDate: '2026-01-09'
category: paper-reading
subcategory: AI Interpretability
tags:
  - axbench
  - steering
  - sparse-autoencoders
  - benchmark
  - icml-2025-spotlight
date: '2026-01-09'
author: Hiep Tran
featured: false
image: >-
  /imgs/blogs/axbench-steering-llms-even-simple-baselines-outperform-sparse-autoencoders-20260109153655.png
excerpt: >-
  AXBENCH introduces a comprehensive benchmark for evaluating LM steering
  methods at scale, revealing that simple baselines like difference-in-means
  outperform Sparse Autoencoders (SAEs), while their novel ReFT-r1 method
  achieves competitive performance with finetuning and prompting.
---

## Motivation

In order to be useful, language models must follow user instructions and be aligned with human goals and values. While **prompting** and **finetuning** are widely used to instill such behavior, both methods have significant limitations:

- **Circumvention via jailbreaks** and continued training
- **Reliance on dataset quality**
- **Uninterpretability**

Interpretability researchers have proposed a new class of **representation-based interventions** for steering LMs, which hope to address these issues. These methods include learning steering vectors from small labelled datasets and self-supervised sparse autoencoders (SAEs). Since steering may enable lightweight and interpretable control over model outputs, it has emerged as a potential alternative to finetuning and prompting.

However, **existing benchmarks for steering only evaluate a few methods at toy scales**. To assess whether representation steering is a viable alternative to existing model control techniques, we need to evaluate it in a more realistic setting:

- Over **open-vocabulary concepts**
- On **long-form generation**
- Compared to **prompting and finetuning baselines**

## Contribution

This paper introduces **AXBENCH**, a benchmark for evaluating LM control methods at scale using synthetic data. The key contributions are:

1. **AXBENCH Benchmark**: Takes in a list of natural language descriptions of concepts and samples relevant training and evaluation data from an LLM. It evaluates model-control methods along two utility axes:

   - **Concept Detection (C)**: Uses labelled synthetic data as ground truth
   - **Model Steering (S)**: Evaluates long-form generations using an LLM judge

2. **Comprehensive Evaluation**: The benchmark includes tasks generated from SAE concept lists for GemmaScope, covering two layers each from instruction-tuned Gemma-2-2B and Gemma-2-9B. AXBENCH is extensible to arbitrary concept descriptions.

3. **ReFT-r1**: A novel weakly-supervised steering method that is **competitive with finetuning and prompting baselines** - the only steering method to achieve this level of performance.

4. **Key Finding on SAEs**: Sparse Autoencoders fall behind both ReFT-r1 and simple difference-in-means baselines on both evaluation axes, demonstrating that even simple baselines can outperform SAEs for steering tasks.

5. **Supervised Dictionary Learning (SDL)**: Along with AXBENCH, the authors train and publicly release SAE-scale feature dictionaries for ReFT-r1 and DiffMean

## AxBench

AXBENCH is a benchmark that takes in a list of **natural language descriptions of concepts** and synthetically generates the appropriate training and evaluation data for each concept using an LLM. The training and evaluation data consists of labelled pairs of instructions and responses, where the responses are either:

- **Positive examples**: expressing the presence of the concept of interest
- **Negative examples**: representing the unsteered behaviour of the model

### Evaluation Axes

The benchmark evaluates methods along two axes:

1. **Concept Detection (C)**: Measures classification performance on a held-out set of labelled data
2. **Model Steering (S)**: Uses an LLM judge to rate steered outputs on three relevant axes

### Benchmark Setup

In this work, the authors use natural language concept lists for **GemmaScope SAEs** as input, generating training and evaluation data for the following representation sites:

- Layers 10 and 20 of instruction-tuned **Gemma-2-2B**
- Layers 20 and 31 of instruction-tuned **Gemma-2-9B**

They sample **500 concepts** for each task to generate data, termed **CONCEPT500**. These eight tasks (4 sites × 2 axes) form the core training and evaluation testbeds for AXBENCH.

![](/imgs/blogs/axbench-steering-llms-even-simple-baselines-outperform-sparse-autoencoders-20260109161019.png)

### Synthetic Concept Dataset Generation

The authors construct a small training dataset $\mathcal{D}\_{\text{train}}$ with $n = 144$ examples and a concept detection evaluation dataset $\mathcal{D}\_{\text{concept}}$ with the same structure but harder examples. They use **gpt-4o-mini-2024-07-18** to generate the data.

The data generation process involves the following steps:

1. **Genre labelling & seed instructions**: Consider three genres: _text_, _code_, and _math_. The LLM picks the genre $g_c$ for each concept, then randomly select seed instructions from an instruction pool belonging to that genre.

2. **Positive examples**: For each randomly sampled instruction, prompt the LLM to generate a response that incorporates the concept $c$. The generated concept-conditioned responses concatenated with their instructions (using the LM's chat template) form the positive set.

3. **Negative examples**: To evaluate generalisation ability, independently sample seed instructions from all genres for negatives. These instructions are shared across concepts to save generation costs. Sample responses from the LM to steer (not the LLM) without any additional instructions. The paired instructions and responses form the negative set.

4. **Hard negative examples** _(evaluation only)_: For each concept, find contrasting concepts that are semantically related but should not activate the concept. This is done by:

   - (a) Generating a list of phrases semantically relevant to the concept
   - (b) Filtering for polysemous words
   - (c) Finding alternative senses of those words which the concept should not activate on

   This results in a set of contrast concepts $c\_{\text{contrast}}$, each representing a specific sense of a polysemous word $w\_{\text{contrast}}$. The LLM generates responses incorporating $w\_{\text{contrast}}$ expressing the sense related to $c\_{\text{contrast}}$. These contrastive responses paired with instructions form the hard negative set.

> **Note**: The negative training set is not applicable to all methods (e.g., full finetuning only needs the positive training set for model steering).

### Concept Detection

A popular LM interpretability method is to train _probes_ that measure to what extent LM representations encode properties of interest (e.g., linguistic features). In recent years, the goal of concept detection has broadened to the **open-vocabulary setting**, with unsupervised methods becoming more common.

**Task description**: Given a Transformer-based LM with hidden dimension size $d$, define a concept classifier as a parameterized function $\Psi\_{\text{Detect}}$ that maps a model representation $h \in \mathbb{R}^d$ into a _binary_ label $\hat{y}$ indicating the relative presence of a concept:

$$\Psi\_{\text{Detect}}(h) = \hat{y} \in \mathbb{R}^1$$

where $\Psi$ is any function, e.g., a neural network.

**Evaluation dataset**: Measure how accurately the classifier can predict ground-truth labels on the labelled evaluation set from $\mathcal{D}\_{\text{concept}}$.

**Evaluation metrics**: Since labels are at the sequence-level, aggregate token-label scores from $\Psi$ using **max-pooling** over the sequence of token representations $\mathbf{h}^l = [h_1^l, h_2^l, \ldots, h_n^l]$ with $n$ tokens at layer $l \in [1, m]$:

$$\hat{y}\_{\text{Detect}} = \max(\Psi\_{\text{Detect}}(\mathbf{h}^l))$$

Then normalize $\hat{y}_{\text{Detect}}$ between $[0, 1]$ by min-max normalisation over the evaluation dataset for each concept. The predicted score represents how strongly a concept is present in a sequence, which can be compared to the true label.

### Model Steering

Representation-based steering has emerged as a potential alternative to existing model-control methods (e.g., finetuning and prompting) and a practical application of various interpretability methods. Unlike concept detection, model steering assesses **causal efficacy** in controlling model behaviour.

**Task description**: Given a prompt $\mathbf{x}$, the model's original generation can be written as $\hat{\mathbf{y}} = \text{LM}(\mathbf{x})$. Produce the model's **counterfactual generation** conditioned on the concept-based intervention $\Phi_{\text{Steer}}(\mathbf{h})$:

$$\hat{\mathbf{y}}\_{\text{Steer}} = \text{LM}(\mathbf{x}, \mathbf{h} \leftarrow \Phi\_{\text{Steer}}(\mathbf{h}))$$

where $\mathbf{h} \leftarrow \Phi_{\text{Steer}}(\mathbf{h})$ is an **in-place representation modification**. The authors use the open-source intervention library **pyvene** to perform such interventions on PyTorch implementations of models.

**Evaluation dataset**: Evaluate steering methods in the **instruction-following setting**, sampling instructions from **Alpaca-Eval** and prompting the LM to generate a response while intervening on its forward pass in-place using one of the steering methods.

**Evaluation metrics**: For the intervened model generation, evaluate $\hat{y}_{\text{Steer}}$ based on the **harmonic mean** of the following scores, each rated by an LLM with a discrete score of 0, 1, or 2:

1. **Concept score**: How well the concept is incorporated into the response
2. **Instruct score**: How well the response is related to the instruction
3. **Fluency score**: How fluent the response is

Since the harmonic mean is used, the overall score ranges from 0 to 2 but **heavily penalises poor performance** on any of the three subscores. For each concept, 10 instructions are randomly sampled from Alpaca-Eval, and sample continuations for each steering factor. To ensure fair comparison, instructions are partitioned into two equally sized sets, selecting the best factor from one set and evaluating on the holdout set.

## Methods

## Experiments

## My thoughts

## References

1. [AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders (arXiv:2501.17148v3)](https://arxiv.org/abs/2501.17148v3)

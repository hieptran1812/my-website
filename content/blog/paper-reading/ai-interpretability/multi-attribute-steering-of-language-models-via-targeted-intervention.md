---
title: "Multi-Attribute Steering of Language Models via Targeted Intervention"
publishDate: "2026-01-07"
category: "paper-reading"
subcategory: "AI Interpretability"
tags:
  [
    "steering",
    "language-models",
    "interpretability",
    "targeted-intervention",
    "multi-attribute",
    "acl-2025",
  ]
date: "2026-01-07"
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/multi-attribute-steering-of-language-models-via-targeted-intervention-20260107210248.png"
excerpt: "MAT-STEER introduces a gating mechanism for selective token-level intervention, enabling LLMs to be steered across multiple attributes (truthfulness, toxicity, bias) without conflicts — outperforming fine-tuning methods with less than 20% training data."
---

## Motivation

Despite strong performance across many tasks, LLMs still generate undesirable outputs — harmful, biased, or factually inaccurate responses. Inference-time intervention (ITI), particularly steering vectors, offers a cost-effective way to modify model behavior by adding offset vectors to token representations during inference, without expensive retraining.

However, **steering vectors do not scale well to multi-attribute settings**. A vector that improves one attribute may harm another. For example, when steering a model to be both helpful and unbiased:

- A model aligned **only for helpfulness** might accept harmful presuppositions (e.g., _"How has immigration harmed the job market?"_), increasing bias
- A model aligned **only to be unbiased** may give unhelpful responses like _"I can't answer that question"_

Applying interventions uniformly on all tokens causes the dominant attribute (e.g., helpfulness) to overshadow others, inadvertently increasing bias. Moreover, uniform intervention risks overcorrecting and pushing the model too far in one direction.

## Contribution

The paper introduces **MAT-Steer (Multi-Attribute Targeted Steering)**, a parameter-efficient approach that addresses these challenges through two key innovations:

1. **Selective Token-Level Intervention**: MAT-Steer uses a gating mechanism to identify _which tokens to intervene on_ and _determines the appropriate intervention intensity_ based on each token's relevance to specific attributes. For example, in _"How has immigration harmed the job market?"_, only _"harmed"_ is relevant to bias and requires intervention, while _"How has"_ and _"the"_ do not.

2. **Alignment Objective with Conflict Mitigation**: A new optimization objective that shifts internal representations of undesirable outputs closer to desirable ones while explicitly mitigating attribute conflicts through **sparsity and orthogonality constraints**.

**Key Results**:

- Outperforms fine-tuning approaches (DPO, SFT) and state-of-the-art ITI methods (LITO) on TruthfulQA, ToxiGen, and BBQ
- Achieves **67.59% win rate** over in-context learning and **71.56% win rate** over ITI on generation tasks (HelpSteer)
- Requires **less than 20% of training data** to match fine-tuning baselines while generalizing to other tasks without degrading LLM capabilities

## Problem Setting and Background

### Inference-time Intervention (ITI)

ITI can be thought of as adding a carefully designed "hint" to tokens that steers the model's internal activations in the desired direction — a subtle instruction that guides the model without changing its parameters.

Formally, given an LLM $\mathcal{M}$ with $L$ layers, each attribute has two sides:

- **Positive (p)**: desirable side (e.g., truthfulness)
- **Negative (n)**: undesirable side (e.g., untruthfulness)

The core idea is to define a transformation function $f(a_i | \theta) = a_i + \alpha\theta$, where:

- $a_i$ is the activation vector at token $i$
- $\theta \in \mathbb{R}^d$ is the steering vector
- $\alpha$ scales the magnitude of intervention

### Problem Setting

Given $T$ distinct attributes, each with its own dataset $\mathcal{D} = \{\mathcal{D}_1, \mathcal{D}_2, ..., \mathcal{D}_T\}$ containing prompt-response pairs that exhibit positive or negative demonstrations.

**Goal**: Learn a set of steering vectors $\mathcal{V} = \{\theta_1, \theta_2, ..., \theta_T\}$ where each $\theta_t$ shifts the activation space toward the positive attribute.

### Limitations of Naive Approaches

A naive extension would be to merge all datasets and learn a single global steering vector $\theta$, or use a linear combination $\theta = \sum_{t=1}^{T} \theta_t$. However:

1. **Conflicting steering directions**: Such approaches risk introducing conflicts that reduce performance on both attributes
2. **Uniform intervention**: Prior methods apply the same editing strength uniformly across tokens, ignoring that different tokens contribute differently to each attribute

**MAT-Steer's Solution**: Use attribute-specific gating functions that modulate each steering vector's contribution on a per-token basis, combined with an objective function to align representations and avoid conflict.

## Multi-Attribute Targeted Steering (MAT-Steer)

![](/imgs/blogs/multi-attribute-steering-of-language-models-via-targeted-intervention-20260107235959.png)

MAT-Steer focuses on three critical components:

- **Gating Function**: An attribute-aware, token-level mechanism determining the degree to which each steering vector influences the activation
- **Representation Alignment**: An objective function that encourages edited activations to align with those derived from positive samples
- **Conflict Avoidance**: Regularization terms that minimize interference among steering vectors and prevent interventions on activations already exhibiting positive attributes

### Gating Function

The gating function enables **soft, token-level determination of intervention strength**. It allows selective intervention _only_ when a token's activation deviates from the desired attribute.

For example, the word _"harmed"_ may prime the model to exhibit bias, so the gating function assigns it a high intervention weight. In contrast, unrelated tokens like _"the"_ receive low weight and are left largely unaltered.

For attribute $t$, the gating function is defined as:

$$G_t(a_i) = \sigma(w_t \cdot a_i + b_t)$$

where:

- $w_t \in \mathbb{R}^{1 \times d}$ and $b_t \in \mathbb{R}$ are learnable weight and bias
- $\sigma(\cdot)$ is the sigmoid function, ensuring output lies in $(0, 1)$

**Key insight**: If a token's activation is already aligned with the desired attribute, $G_t(a_i) \approx 0$ (little to no intervention). If the activation deviates, $G_t(a_i) \approx 1$ (stronger intervention).

The overall steering function becomes:

$$f(a_i | \theta_1, ..., \theta_T) = a_i + \sum_{t=1}^{T} G_t(a_i) \cdot \theta_t$$

### Representation Alignment

The goal is to intervene on activations corresponding to negative traits (e.g., untruthfulness) so they more closely resemble those associated with positive traits (e.g., truthfulness).

**Challenge**: Paired data with both positive and negative responses $(x, y^p, y^n)$ may not exist for all attributes. The paper uses **Maximum Mean Discrepancy (MMD)** loss, which compares entire distributions without requiring explicit pairings.

**Why MMD over conventional losses?** Prior ITI methods typically match lower-order statistics (e.g., the mean), risking missing critical higher-order differences like variance. MMD maps data into a reproducing kernel Hilbert space (RKHS), capturing higher-order moments for a richer representation.

The MMD loss:

$$L_{MMD} = \sum_{t=1}^{T} \left\lVert \sum_{a_i \in A_t^p} \frac{\phi(a_i)}{|A_t^p|} - \sum_{a_i \in A_t^n} \frac{\phi(f(a_i))}{|A_t^n|} \right\rVert_{H}^2$$

This drives negative activations toward the desired positive region.

### Avoiding Conflicts

When combining multiple attribute-specific steering vectors, conflicts may arise. For example, a vector designed to suppress bias might conflict with one intended to enhance helpfulness if both are applied to the same token.

**Three complementary strategies**:

#### Preservation of Positive Samples

For activations already positively aligned, we want to avoid unnecessary intervention:

$$L_{pos} = \sum_{t=1}^{T} \sum_{a_i \in A_t^p} [G_t(a_i)]^2$$

This forces gating outputs to be near zero for positive activations, preserving original semantic information and preventing over-correction.

#### Sparsity for Negative Samples

Since not every steering vector is relevant to every activation, we apply an $\ell_1$ penalty to encourage sparsity:

$$L_{sparse} = \sum_{t=1}^{T} \sum_{a_i \in A_t^n} |G_t(a_i)|$$

This ensures only the most relevant attribute-specific steering vectors are applied, reducing the chance of conflicts.

#### Orthogonality of Steering Vectors

Two attribute-specific vectors acting on the same token may interfere destructively (cancel out components in opposite directions). We impose an orthogonality constraint:

$$L_{ortho} = \sum_{t=1}^{T} \sum_{t' \neq t}^{T} \left( \frac{\theta_t^T \theta_{t'}}{\lVert\theta_t\rVert_2 \lVert\theta_{t'}\rVert_2} \right)^2$$

**Why this works**: By encouraging steering vectors to be orthogonal, each vector operates in a distinct, complementary direction. Interventions for one attribute do not spill over to adversely affect others. Given the large activation space ($d = 4096$), it's feasible for all steering vectors to be orthogonal as long as $T \ll d$.

### Normalization and Overall Loss

After applying the steering function, we normalize the edited activation to maintain the original $\ell_2$-norm:

$$\tilde{a}_i \leftarrow \tilde{a}_i \cdot \frac{\lVert a_i \rVert_2}{\lVert \tilde{a}_i \rVert_2}$$

This ensures the intervention shifts the _direction_ rather than the _scale_ of the activation.

**Overall loss function**:

$$L_{total} = L_{MMD} + \lambda_{pos} L_{pos} + \lambda_{sparse} L_{sparse} + \lambda_{ortho} L_{ortho}$$

where $\lambda_{pos}$, $\lambda_{sparse}$, and $\lambda_{ortho}$ are hyperparameters balancing each term.

## Experiments

### Experimental Setup

**Models**: Experiments are conducted on Llama-3.1-8B, Llama-3.1-8B-Chat, and Qwen2.5-7B. The main paper reports results for Llama-3.1-8B, with other models showing similar trends in Appendix C.

**QA Datasets** (measuring multiple-choice accuracy):

- **TruthfulQA**: Assesses the model's ability to provide truthful responses
- **Toxigen**: Evaluates capability to avoid generating toxic outputs
- **BBQ**: Measures bias in generated answers

**Generation Dataset**:

- **HelpSteer**: Evaluates five human-annotated attributes — Helpfulness, Correctness, Coherence, Complexity, and Verbosity (rated 0-4). GPT-4o serves as LLM-as-a-judge, and win rates are computed by comparing average attribute scores.

**Baselines**:

- **In-Context Learning (ICL)**: Tests whether prompt engineering alone yields improvements
- **Fine-Tuning Methods**: SFT and DPO with LoRA adapters
- **Multiple-Adapters Methods**: Merge (combining attribute-specific adapters) and RAdapt (router-based adapter selection)
- **Intervention/Steering Methods**: ITI, ICV, NL-ITI, and LITO

### Main Results

![](/imgs/blogs/multi-attribute-steering-of-language-models-via-targeted-intervention-20260108001336.png)

**QA Tasks Performance**: MAT-STEER achieves the highest accuracy on all three datasets:

| Dataset    | MAT-STEER | Improvement over LITO |
| ---------- | --------- | --------------------- |
| TruthfulQA | 61.94%    | +3.31%                |
| Toxigen    | 57.59%    | +3.51%                |
| BBQ        | 60.32%    | +2.18%                |

Fine-tuning approaches (SFT, DPO) and model merging yield inconsistent improvements — boosting one attribute while failing to generalize across all targeted attributes.

![](/imgs/blogs/multi-attribute-steering-of-language-models-via-targeted-intervention-20260108001421.png)

**Generation Tasks Performance**: MAT-STEER consistently achieves higher win rates compared to all baselines on the HelpSteer dataset. This demonstrates that MAT-STEER not only enhances desired attributes (factual correctness, helpfulness) but also effectively preserves fluency and coherence.

![](/imgs/blogs/multi-attribute-steering-of-language-models-via-targeted-intervention-20260108001447.png)

**Data Efficiency**: MAT-STEER with **less than 20% of training data** achieves the same or better performance than SFT and DPO using 100% of the data. For example, on TruthfulQA:

- MAT-STEER with 10% data: **60.05%**
- DPO with 100% data: 55.98%
- SFT with 100% data: 54.12%

This highlights MAT-STEER's effectiveness in low-data scenarios, making it practical for real-world applications where labeled data is scarce.

## Analysis

### Internal Mechanism of MAT-STEER

To understand how MAT-STEER's gating mechanism works in practice, the authors analyze its behavior on the **ParaDetox dataset** (toxicity-focused). They sample 100 toxic sentences (requiring intervention) and 100 neutral sentences (no intervention needed), measuring:

1. **Gating weights** for toxicity vs. other attributes (truthfulness, bias)
2. **Number of intervened tokens**
3. **Toxicity flip rate**: percentage of toxic samples that become neutral after intervention

![](/imgs/blogs/multi-attribute-steering-of-language-models-via-targeted-intervention-20260108002238.png)

**Key findings**:

- For **toxic samples**: The toxicity steering vector receives high gating weight (0.61), while unrelated attributes remain largely inactive (0.14). This leads to **86% toxicity decrease**.
- For **neutral samples**: Gating weights remain low across all attributes (0.08-0.12), demonstrating MAT-STEER's ability to **preserve already-aligned outputs** without unnecessary intervention.

This validates that the gating function correctly identifies _which_ tokens need intervention and _which_ steering vector to apply.

### Impact on General LLM Capabilities

![](/public/imgs/blogs/multi-attribute-steering-of-language-models-via-targeted-intervention-20260108002300.png)

A critical concern with steering methods is whether they degrade the model's general capabilities (e.g., fluency, coherence). The authors evaluate this on TruthfulQA using open-ended generation and measure **BLEU accuracy** (closeness to correct vs. incorrect references).

**Results**:

- MAT-STEER: **45.97** BLEU accuracy
- SFT (LoRA): 43.83
- ITI: 41.58

MAT-STEER produces more factually correct and coherent outputs than both fine-tuning and prior intervention methods.

### Generalization to Other Tasks

To test whether MAT-STEER generalizes beyond its training distribution, the authors evaluate on **FaithEval** — a counterfactual QA benchmark where the context contains statements contradicting common facts. Importantly, FaithEval was _not_ used to construct the steering vectors.

**Results**:

- MAT-STEER: **56.89%** accuracy
- DPO: 51.20%
- ICL: 48.68%

This shows MAT-STEER selectively focuses on context positions containing factual inconsistencies, reinforcing model faithfulness even on unseen tasks.

### Ablation Study

![](/imgs/blogs/multi-attribute-steering-of-language-models-via-targeted-intervention-20260108002333.png)

The ablation study on TruthfulQA reveals the contribution of each component:

| Configuration                            | Accuracy   |
| ---------------------------------------- | ---------- |
| Base model                               | 49.91%     |
| + Representation alignment (MMD)         | 53.82%     |
| + Positive preservation ($L_{pos}$)      | 55.48%     |
| + Sparsity penalty ($L_{sparse}$)        | 56.73%     |
| + Orthogonality constraint ($L_{ortho}$) | 54.37%     |
| Full MAT-STEER (w/o normalization)       | 59.88%     |
| **Full MAT-STEER (with normalization)**  | **61.94%** |

**Key takeaways**:

- Each component contributes meaningfully to performance
- **Normalization** plays a crucial role (+2.06% over unnormalized version)
- Removing positive preservation causes a **3.86% drop**, highlighting its importance in preventing over-correction

## Conclusion

MAT-STEER is a parameter-efficient inference-time intervention method that dynamically steers LLMs across multiple potentially conflicting attributes. Through its **gating mechanism** and **conflict-aware optimization objective**, it selectively adjusts token-level representations to mitigate undesirable outputs while preserving model capabilities.

**Key achievements**:

- Outperforms fine-tuning (SFT, DPO) and prior steering methods across QA and generation tasks
- Achieves robust generalization with **significantly less training data** (<20%)
- Preserves fluency and coherence while improving targeted attributes

**Limitations**:

- May struggle with highly complex or "unsteerable" attributes
- Still relies on a small dataset to construct steering vectors (though more data-efficient than baselines)
- Does not eliminate LLM risks entirely — mitigates but doesn't solve bias/toxicity

## References

1. [Multi-Attribute Steering of Language Models via Targeted Intervention (arXiv:2502.12446)](https://arxiv.org/abs/2502.12446)
2. [GitHub: MAT-Steer](https://github.com/duykhuongnguyen/MAT-Steer)

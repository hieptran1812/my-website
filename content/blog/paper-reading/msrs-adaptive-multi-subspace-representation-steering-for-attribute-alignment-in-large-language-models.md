---
title: "MSRS: Adaptive multi-subspace representation steering for attribute alignment in large language models"
publishDate: "2025-09-01"
category: "paper-reading"
subcategory: "AI Interpretability"
tags: ["model-interpretation", "activation-steering"]
date: "2025-09-01"
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/msrs-adaptive-multi-subspace-representation-steering-for-attribute-alignment-in-large-language-models-20250901175838.png"
excerpt: "This paper introduces MSRS (Multi-Subspace Representation Steering) to solve these challenges. The core motivation is to mitigate conflicts between different attribute controls while enabling multiple attributes..."
---

![](/imgs/blogs/msrs-adaptive-multi-subspace-representation-steering-for-attribute-alignment-in-large-language-models-20250901175838.png)

# Motivation

Large Language Models (LLMs) have shown remarkable capabilities in tasks like text generation, question answering, and dialogue. However, when applied in real-world and sensitive contexts, their behavior can become problematic. For example, they may generate toxic, biased, or factually incorrect outputs. Imagine asking a model a simple question about a political figure: instead of giving a neutral answer, it might respond with a biased or misleading statement. This happens because the internal representations learned during training are very complex and not transparent, making it difficult to directly control or align their behavior with desired attributes such as truthfulness and fairness.

Researchers have tried to address these issues by steering model activations after training, instead of retraining the whole model. This is attractive because it is lightweight and more scalable. Yet, existing steering methods usually focus on one attribute at a time, say, reducing toxicity, but fail to manage multiple attributes simultaneously. If we try to naively combine steering directions, for example steering for truthfulness and for fluency at once, the model can get confused: improving one attribute might unintentionally hurt another. For instance, making outputs more truthful might reduce fluency, or steering for fairness might conflict with persuasiveness.

Moreover, prior approaches like clustering or orthogonal constraints often struggle to cleanly separate attributes at the fine-grained level. They either blend steering signals, leading to interference, or fail to allocate enough expressive capacity for complex attributes. As a result, steering for multiple attributes remains suboptimal, simple attributes may need small adjustments, while complex ones require larger and more specialized representation spaces.

This paper introduces MSRS (Multi-Subspace Representation Steering) to solve these challenges. The core motivation is to mitigate conflicts between different attribute controls while enabling multiple attributes to be steered simultaneously in a more precise and adaptive way. In other words, instead of forcing different behavioral adjustments into the same space and risking interference, MSRS adaptively assigns separate subspaces for each attribute, while also capturing their shared parts in a common space. This makes it possible to balance attributes like truthfulness, fairness, and fluency together without sacrificing performance.

In short, the motivation is to make LLMs not only powerful but also trustworthy and controllable in real-world applications, especially when multiple values or goals must be aligned at the same time.

# Background

ReFT is a method that fine-tunes a low-dimensional subspace in the model’s hidden representations to guide outputs toward certain attributes, like making responses more truthful or less toxic. It works well when steering for a single attribute, but struggles when multiple attributes are involved. The problem is that all steering directions are forced into the same space, which creates interference. This makes it hard for the model to balance different needs, for example improving truthfulness while also keeping fluency.

Some approaches try to fix this by splitting the space into equal parts for each attribute, but this is inefficient. Different attributes need different amounts of capacity: simple ones (like avoiding repetition) may need less space, while complex ones (like fairness or truthfulness) require more. Giving each attribute the same fixed space can waste capacity or limit performance.

To overcome these issues, the authors propose MSRS (Multi-Subspace Representation Steering). MSRS creates separate subspaces for each attribute, adapting their sizes based on how expressive they need to be. It uses Singular Value Decomposition (SVD) to dynamically allocate space more efficiently. In addition, MSRS introduces a shared subspace that captures common steering directions across attributes and models their interactions. This makes it possible to manage multiple attributes in a flexible and balanced way, leading to better control over LLM behavior compared to older methods.

# Methodology

## MULTI-ATTRIBUTE STEERING DIRECTION EXTRACTION

To control multiple attributes at the same time, the method extracts steering directions that separate what is shared across attributes from what is specific to each attribute. This ensures precise and flexible steering in the activation space.

For each attribute $i$, the model first computes the average activation across its dataset $D_i$. The intermediate activation of sample $j$ at layer $l$ is $h^{l}_{i,j}$. The average activation is:

$$
\tau_i = \frac{1}{|D_i|} \sum_{j=1}^{|D_i|} h^{l}_{i,j}
$$

This captures the main feature representation for attribute $i$. To combine all attributes, the average activations are concatenated:

$$
\tau_c = [\tau_1 \|\ \tau_2 \|\ \dots \|\ \tau_n] \in \mathbb{R}^{d \times n}
$$

To separate shared and attribute-specific information, Singular Value Decomposition (SVD) is applied to $\tau_c$:

$$
\tau_c = U_c \Sigma_c V_c^\top
$$

From this, the shared subspace is obtained:

![](/imgs/blogs/msrs-adaptive-multi-subspace-representation-steering-for-attribute-alignment-in-large-language-models-20250917164629.png)

Here, $r_s$ is chosen so that the top $r_s$ singular values capture at least 90% of the total energy in $\Sigma_c$. This shared subspace represents the dominant steering directions common to all attributes.

For each attribute $i$, the method removes the shared part to isolate the unique attribute-specific directions. The residual is:

![](/imgs/blogs/msrs-adaptive-multi-subspace-representation-steering-for-attribute-alignment-in-large-language-models-20250917164946.png)

Then, applying SVD:

$$
H^{(i)}_{\text{res}} = U^{(i)} S^{(i)} V^{(i)\top}
$$

The top $r_i$ singular values are chosen (again covering at least 90% of the energy), and the private subspace is defined as:

![](/imgs/blogs/msrs-adaptive-multi-subspace-representation-steering-for-attribute-alignment-in-large-language-models-20250917165149.png)

Each $B_i$ captures directions unique to attribute $i$, orthogonal to the shared space.

Finally, the shared subspace and all attribute-specific subspaces are combined into the alignment matrix:

$$
S_{\text{align}} = [B_{\text{shared}}, B_1, B_2, \dots, B_n] \in \mathbb{R}^{(r_s + \sum_{i=1}^n r_i) \times d}
$$

In simple terms, the method first finds the average signal for each attribute, then separates what is common across all attributes from what is unique to each one. By adapting the size of each subspace, it gives more capacity to complex attributes and less to simpler ones, before merging everything into one alignment matrix $S_{\text{align}}$, which allows the model to steer multiple attributes at once without interference.

## ADAPTIVE SUBSPACE SELECTING

This section introduces a method called **Adaptive Subspace Selecting** to control multiple attributes without them interfering with each other. Instead of forcing all attributes to share the same space, it uses a **mask network** $m(h) = \text{sigmoid}(\text{MLP}(h))$ that assigns weights to each subspace dimension. These weights decide which parts of the space should be emphasized for each attribute. The model applies these weights through a diagonal matrix to adjust how much each dimension contributes when combining transformations. This allows the model to **adaptively select and combine different subspaces**, making multi-attribute control more effective and reducing conflicts between attributes.

So we have steering function:

![](/imgs/blogs/msrs-adaptive-multi-subspace-representation-steering-for-attribute-alignment-in-large-language-models-20250917172248.png)

## OPTIMIZATION OBJECTIVE

This section explains how they train the model so it can control different attributes clearly and without mixing them up.

First, they apply their steering function to the model’s hidden representation. This changes the model’s internal features, and then they check how well it does on the task using normal cross-entropy loss — called $L_{task}$.

But to make sure each attribute only uses the **right subspace** and doesn’t interfere with others, they add two extra losses:

1. **Regularization loss ($L_{reg}$)**
   They create a “prior mask” saying which dimensions are supposed to be used for this attribute. The model’s learned mask $m(h)$ is trained to match this prior.
   → This pushes the model to only activate dimensions relevant to that attribute and ignore others.

2. **Alignment loss ($L_{align}$)**
   They also make the learned representation $R$ align with a known structured basis $S_{align}$.
   → This ensures the model’s space lines up nicely with both shared and attribute-specific directions.

Finally, the **total training loss** combines all three:

$$
L = L_{task} + \lambda_1 L_{reg} + \lambda_2 L_{align}
$$

where $\lambda_1, \lambda_2$ are weights to balance the terms.

Okay, in short, they train the model to (1) do the task well, (2) stay in the right subspace for each attribute, and (3) align the subspaces properly, so it can control different attributes cleanly and consistently.

From my thought, while this design is clever and theoretically appealing, it also brings some practical challenges:

- **High training cost**: Adding multiple extra loss terms and subspace constraints increases computation. The model must learn masks $m(h)$, representations $R$, and align them, which can slow down training and require more GPU memory.
- **Not straightforward to tune**: The method introduces **new hyperparameters** ($\lambda_1, \lambda_2$), and the choice of the prior mask $m_{prior}$ also affects results. Finding the right balance is tricky and may need a lot of trial-and-error or expensive hyperparameter search.
- **Risk of instability**: Because these losses push in different directions (task performance vs. disentanglement), training might become unstable if the weights are not well tuned.

## DYNAMIC INTERVENTION POSITION SELECTION

Normally, older methods always apply the intervention (the steering) at the same token position for all attributes (for example, always at the last token). But this can cause interference, because not all tokens are equally important for every attribute.

So instead, they propose a smarter approach:

For each attribute $i$, they **search for the most relevant token position $p_i$** in the sequence. Here’s how they do it:

1. Each token has a hidden representation $h_t$.
2. They **project** each token’s representation onto the attribute’s specific subspace $R_i$:

   $$
   \text{proj}_{R_i}(h_t) = R_i^\top R_i h_t
   $$

3. They measure how **strongly this token aligns** with the attribute’s subspace by taking the **L2 norm** (basically the size/strength) of that projection:

   $$
   s_{i,t} = \|\text{proj}_{R_i}(h_t)\|_2
   $$

4. They pick the **token with the highest score** $s_{i,t}$ as the best place to intervene:

   $$
   p_i = \arg\max_{t} s_{i,t}
   $$

This means: for each attribute, find the token that is **most related to that attribute**, and apply the steering there.

# Experimental results

## Multi-Attribute Steering Performance

![](/imgs/blogs/msrs-adaptive-multi-subspace-representation-steering-for-attribute-alignment-in-large-language-models-20250917174942.png)

MSRS was evaluated on multiple large language models, including Llama2-7B, Llama3-8B-Instruct, Qwen2-7B-Instruct, and Mistral-7B-v0.3, and compared with several strong baselines such as In-Context Learning (ICL), Contrastive Activation Addition (CAA), Inference-Time Intervention (ITI), Representation Fine-Tuning (ReFT), Multi-Task Learning with LoRA (MTL-LoRA), and Multi-Attribute Steering (MAT-STEER). The results show that MSRS outperformed all baselines in balancing multiple conflicting attributes. In tasks targeting truthfulness and bias, using datasets like TruthfulQA and BBQ, MSRS achieved much better trade-offs than existing methods, which often improved one attribute at the cost of harming another. For example, on Llama3-8B-Instruct, MSRS reached an MC2 score of 56.32 and a BBQ accuracy of 0.645, both higher than other methods. Moreover, for instruction-following, refusal, and generation quality tasks, MSRS successfully harmonized different objectives. It performed competitively on Alpaca win rates, while also maintaining strong refusal behavior on Sorry-Bench and high output quality across HelpSteer dimensions, including helpfulness, coherence, and verbosity.

## Preservation of General Capabilities

![](/imgs/blogs/msrs-adaptive-multi-subspace-representation-steering-for-attribute-alignment-in-large-language-models-20250917174954.png)

An important observation from the experiments is that MSRS can steer model behavior toward targeted attributes while preserving, or even improving, the model’s general natural language understanding abilities. Across standard benchmarks such as HellaSwag, RACE, MMLU, OpenBookQA, and GLUE, MSRS consistently matched or surpassed the baselines. On Llama3-8B-Instruct, for example, it achieved an average GLUE score of 0.775, outperforming ITI at 0.742 and ReFT at 0.757. This shows that MSRS can guide specific behavior without weakening the model’s overall performance.

## Ablation Studies and Analysis

A series of ablation studies confirmed the importance of each component in the MSRS framework. The adaptive subspace selection mechanism consistently outperformed fixed allocation strategies, while the dynamic token positioning method worked better than static last-token interventions on all models and datasets. Additionally, the layer-wise analysis showed that steering is most effective in the mid-to-upper layers of the transformer (around layer 15). These layers provide the best balance between semantic abstraction and controllability. In contrast, lower layers do not contain enough semantic information, while deeper layers show a tendency to overfit when steering is applied there.

# My thoughts

MSRS is a well-designed approach that effectively reduces interference between attributes by assigning each one its own subspace while still modeling their shared components. Its adaptive subspace allocation and dynamic token positioning make it more flexible and precise than prior methods.

However, the method also comes with practical drawbacks. It significantly increases training cost due to additional modules like mask networks, SVD computation, and multiple loss terms. It is also hard to tune, as it introduces several hyperparameters ($\lambda_1$, $\lambda_2$, $r_s$, $r_i$) and relies on carefully chosen prior masks. This complexity may cause training instability if the losses are not well balanced. Future work could explore progressive training, adaptive loss weighting, lightweight mask network or use some unsupervising designs to reduce computational overhead and make the method easier to use in large-scale settings.

# References

1. [MSRS: ADAPTIVE MULTI-SUBSPACE REPRESENTATION STEERING FOR ATTRIBUTE ALIGNMENT IN LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2508.10599)
2. ChatGPT :)))

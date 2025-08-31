---
title: "Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning"
publishDate: "2025-09-01"
category: "paper-reading"
subcategory: "AI Interpretability"
tags: ["model-interpretation", "unsupervised-learning"]
date: "2025-09-01"
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/decomposing-representation-space-into-interpretable-subspaces-with-unsupervised-learning-20250807133820.png"
excerpt: "This paper introduces a new, unsupervised method called Neighbor Distance Minimization (NDM). The idea is to divide the model’s internal representation space into smaller “subspaces”..."
---

![](/imgs/blogs/decomposing-representation-space-into-interpretable-subspaces-with-unsupervised-learning-20250807133820.png)

# Motivation

The researchers are trying to understand how neural networks think. More specifically, how they represent different ideas or features inside their “brain” (also known as the representation space). Because these models are super high-dimensional (tons of numbers representing information), it's hard to tell which parts of the network are responsible for what.

They asked:

- Can we find natural “subspaces” (like sections of the network that focus on certain ideas) without any supervision?
- Can these parts be interpretable. It means can we understand what they’re doing?

And the surprising answer is: Yesss!

This paper introduces a new, unsupervised method called Neighbor Distance Minimization (NDM). The idea is to divide the model’s internal representation space into smaller “subspaces” by measuring how close different activations are to each other. Unlike previous approaches, NDM doesn’t need any labels or supervision. It tries to organize the space so that each subspace captures a different, independent kind of information. The result is a meaningful and interpretable set of subspaces, like discovering the “natural categories” the model uses internally.

They tested this in GPT-2 (a well-known language model) and found:

- These subspaces match known circuits (kind of like logic gates or functions inside the model).
- The model’s "thinking parts" map well to these interpretable subspaces.
- It even scales up to large models (2 billion parameters).

They found subspaces that handle things like context tracking or routing knowledge to the right part of the model.

One of the big ideas proposed is to treat these subspaces as new “building blocks” for understanding and analyzing models. Because subspaces group related features, they’re like variables, it means each can take on different values depending on the input. This opens the door to building input-independent circuits, which are stable and easier to interpret. If we can understand how subspaces interact across layers using model weights, we could build clearer mental maps of how models process information. In short, this method could significantly improve how we understand and debug complex AI systems.

# Background

One key idea in interpretability research is superposition. This means a model can represent more features than the actual number of dimensions it has. It does this by encoding features as non-orthogonal directions in a lower-dimensional space. In other words, even though the model has fewer “slots” (dimensions), it can still store many features by letting them overlap in clever ways.

For example, imagine you have a notebook with only 3 pages, but you need to store 6 different drawings. You could overlay two drawings on each page. If you know the trick to separate them later, you can still reconstruct each original drawing. That’s what superposition is doing with features.

![](/imgs/blogs/decomposing-representation-space-into-interpretable-subspaces-with-unsupervised-learning-20250831142845.png)

The authors also tested this idea using a simple toy model. They split features into groups and ensured that only one feature from each group was active at a time. The results showed that features inside the same group caused very little interference, while groups themselves stayed independent. This confirmed that **mutual exclusiveness** and grouping are what really make superposition possible.

In short, superposition is a clever way for models to pack more knowledge into fewer dimensions. But for it to work well, the model relies on **mutual exclusiveness within groups of features**, which prevents overlap from causing confusion and helps preserve information.

A model’s activation, written as $h$, can be the result of multiple features being active, but importantly, each of those features must come from a different group. Within the same group, features tend to overlap, while between different groups they are more distinct or orthogonal. This creates a structure where each group forms its own subspace, and while superposition happens within a subspace, the subspaces themselves remain separate and orthogonal to one another.

The authors argue that this kind of exclusivity is not only theoretical but also happens in the real world. For example, if the previous token in a sentence is “the”, it cannot simultaneously be “cat”. If the subject of a sentence is Alice, it cannot at the same time be Bob. If the sum of two digits equals 5, it cannot also equal 6. Similarly, in models, variables can only take on one value at a time, and each value belongs to a specific group or subspace.

This structure has important consequences. It suggests that models naturally form interpretable partitions of features, where groups behave like independent variables. The authors also connect this idea to the Multi-Dimensional Superposition Hypothesis, which proposes that each feature group can be seen as one irreducible unit of meaning. This highlights both exclusivity within groups and independence across groups.

To make this more intuitive, imagine a clothing store. There are groups like shirts, pants, and shoes. Within the shirt group, you can only wear one shirt at a time. But wearing a shirt does not prevent you from also choosing pants and shoes. Each clothing category acts as a separate subspace. Inside each group you must pick only one, but across groups you can freely combine them. This is exactly how the authors describe features in models: exclusive within groups, but independent across groups.

To further validate the concept of feature superposition and its relation to orthogonality, the authors conduct a toy experiment using a simplified model setup. The experiment explores the concept of superposition in interpretability, which means a model can represent more features than its dimensionality by encoding them as non-orthogonal directions.

They use a toy model from _Elhage et al. (2022)_, defined as:

$$
h = W x
$$

$$
x' = \text{ReLU}(W^T h + b)
$$

where:

- \$x, x' \in \mathbb{R}^z\$
- \$h \in \mathbb{R}^d\$
- \$W \in \mathbb{R}^{d \times z}\$
- \$d < z\$

Here, \$x\$ is the **high-dimensional ground truth feature vector**. Each entry \$x_i\$ represents a “feature” with values in $\[0,1]\$, which gets encoded into the low-dimensional \$h\$. The model output \$x'\$ is trained to reconstruct \$x\$, meaning that each column of \$W\$ corresponds to a direction in the lower-dimensional space that represents a feature \$x_i\$.

Toy Model of Intra-group Superposition:

- They set \$z = 40\$ (number of features) and \$d = 12\$ (lower-dimensional encoding).
- Features are divided into **two groups**, each containing 20 features.
- When sampling \$x\$:

  - With probability \$0.25\$, no feature in a group is activated.
  - With probability \$0.75\$, one feature is randomly chosen and its value is uniformly sampled from $\[0,1]\$.

- All features are equally important (no weighting applied during reconstruction loss).

After training:

- The **fraction of variance unexplained (FVU)** is:

$$
\text{FVU} = 0.059
$$

which indicates the model reconstructs the features quite well.

- The matrix \$W^T W\$ shows the relationships between features:

  - Products of features from the same group are non-zero (indicating overlap / superposition inside the group).
  - Features from different groups are orthogonal (no overlap).

Thus, the model effectively learns two orthogonal subspaces, each corresponding to one feature group. A closer look at the singular values reveals that each feature group is encoded in a 5-dimensional space.

In summary, this toy model verifies that when features are grouped with mutual exclusivity, the model naturally organizes them into orthogonal subspaces, preserving structure across groups while allowing superposition within groups.

# Methodology

## Goal

The goal of this study is to identify orthogonal structure in the representation space. This means the space can be divided into multiple orthogonal subspaces, and each subspace contains one feature group.

In the toy setting, this is simple: we can take all feature vectors from the same group, and the subspace spanned by them is the desired result. However, in real-world models we do not have access to the ground truth features x or the projection W. Therefore, we need a method that only relies on model activations h to find the correct subspace. This method is called NDM.

The idea can be illustrated with an example in 3D space. Suppose the xy plane contains a group of three features (represented by blue arrows), while the z dimension contains another group of two features (orange arrows). In this case, there are two orthogonal subspaces, each containing a feature group.

If the property of intra-group mutual exclusiveness holds, each data point is formed by only one blue and one orange feature. As a result, the points will only lie on specific pink planes rather than being spread across the whole space.

With this partition, the projections of the data points onto the xy subspace will lie along a few specific lines (the blue arrows). Similarly, the projection onto the z subspace will also be concentrated in fixed positions. This makes the distance between points within the same subspace small, so finding the nearest neighbor is easy.

In contrast, if the partition is incorrect (for example, xz and y), then the data points projected onto the xz plane will cover the entire plane. This leads to larger distances between points, making clustering more difficult.

In other words, the correct partition ensures that data points within each subspace stay close together, while incorrect partitions cause the points to spread out because they mix features from different groups.

![](/imgs/blogs/decomposing-representation-space-into-interpretable-subspaces-with-unsupervised-learning-20250831151355.png)

## Optimization

The authors define a partition as a pair consisting of:

1. An orthogonal matrix \$R\$, which rotates and reflects the space.
2. A dimension configuration \$c\$, which specifies the number of subspaces and their dimensionalities.

The idea is that after applying the transformation \$R\$, the representation space can be partitioned into orthogonal subspaces according to \$c\$. Given \$N\$ model activations \$h_n \in \mathbb{R}^d\$, the transformed activations are:

$$
[\hat{h}_1 \cdots \hat{h}_N] = R [h_1 \cdots h_N]
$$

Each transformed activation \$\hat{h}\_n\$ is divided into components \$\hat{h}\_n^{(s)}\$, corresponding to different subspaces of dimension \$d_s\$:

$$
\hat{h}_n =
\begin{bmatrix}
\hat{h}_n^{(1)} \\
\vdots \\
\hat{h}_n^{(S)}
\end{bmatrix}, \quad \hat{h}_n^{(s)} \in \mathbb{R}^{d_s}
$$

For each data point \$n\$, its nearest neighbor \$n^\*\$ is found within the same subspace:

$$
n^* = \arg \min_{m=1,\ldots,N, m \neq n} \, \text{dist}\left(\hat{h}_n^{(s)}, \hat{h}_m^{(s)}\right)
$$

The optimization objective is to minimize the average distance to the nearest neighbor across subspaces (I don't know why the latex render doesn't work :)):

![](/imgs/blogs/decomposing-representation-space-into-interpretable-subspaces-with-unsupervised-learning-20250901002802.png)

Here, $\text{dist}(\cdot)$ is a distance metric, such as Euclidean distance.

From another perspective, this method can also be interpreted as minimizing total correlation between subspaces. Total correlation generalizes mutual information (MI) from two variables to multiple variables, measuring redundancy and dependency among them. Intuitively, neighbor distance reflects entropy: the method reduces entropy within subspaces, while the total entropy of the full space remains unchanged under orthogonal transformation. As a result, the subspaces become more independent and interpretable, serving as fundamental units of representation.

## Determining the Subspace Dimension Configuration

A key question is how to find the correct dimension configuration \$c\$ (the number of subspaces and their dimensions). This is not straightforward, since the objective is not differentiable.

To address this, the authors use mutual information (MI). The procedure is:

1. Start with many small, equally sized subspaces and train the orthogonal matrix \$R\$.
2. Regularly measure the MI between subspaces during training, using the KSG estimator (Kraskov et al., 2004).
3. If the MI between two subspaces is above a threshold, merge them into a single subspace, and continue training \$R\$.
4. Repeat this process until no pair of subspaces has MI above the threshold.

This procedure progressively reduces dependency between subspaces by merging those that are strongly correlated.

In the end, NDM produces a partition that contains both the orthogonal matrix \$R\$ and the dimension configuration \$c\$.

# Experiments in Toy Models

In this experiment, the authors test their method on a simple toy model to see if it can discover the hidden structure of features. They use Euclidean distance to check whether different parts of the model’s hidden activations can be grouped into separate subspaces, each corresponding to one feature group. Ideally, each subspace should only represent a single group of features.

The key question is whether these subspaces can be found just by minimizing neighbor distances, without any prior knowledge of the groups. The results show that the method, called Neighbor Distance Matching (NDM), is able to do this successfully.

Starting from an identity matrix, the learned transformation after training is almost perfect, and additional tests with different numbers of features and groups confirm its effectiveness. Overall, the study shows that in toy settings where the ground truth subspaces are orthogonal, NDM can automatically recover them using only the model’s activations.

# Experiments in Language Models

## Quantitative Evaluation Based on GPT-2 Circuits

This section evaluates NDM in language models (GPT-2 Small), which is more challenging than toy models because there are no ground-truth feature groups.

The goal is to test whether NDM can discover interpretable subspaces that correspond to meaningful concepts. To do this, the authors use subspace activation patching, a method where activations are swapped or replaced to see how information is localized. The intuition is that important information (for example, the previous token in an induction head) should ideally lie within a single subspace.

They construct a test suite of five evaluation tasks based on known GPT-2 circuits, such as testing how previous tokens, subject names, positions, or year digits are represented in the residual stream. The effect of patching is measured with the Gini coefficient, which captures whether the influence of information is concentrated in one subspace (high Gini) or spread out (low Gini).

Results show that NDM successfully finds subspace partitions in GPT-2 Small. The discovered subspaces demonstrate a strong concentration of target information, with average Gini coefficients significantly above 0.6 (around 0.71), outperforming baselines. Importantly, NDM works in an unsupervised way, relying only on model activations, without labels or ground-truth features. Additional analysis shows that the mutual information between subspaces decreases after training, which further supports NDM’s ability to separate meaningful information.

![](/imgs/blogs/decomposing-representation-space-into-interpretable-subspaces-with-unsupervised-learning-20250831234448.png)

## Qualitative Analysis of Obtained Subspaces in GPT-2

![](/imgs/blogs/decomposing-representation-space-into-interpretable-subspaces-with-unsupervised-learning-20250831235626.png)

In this part, the authors analyze the subspaces discovered in GPT-2 to see whether they are meaningful and interpretable. They use the idea of a **preimage**, which is the set of inputs that produce the same activation in the model. Instead of looking at the whole space, they focus on subspaces and collect a large number of activations to find which inputs correspond to each one.

Their results show that for the same token at the same model layer, different aspects of context—such as the current token, its position, the previous token, or even the topic—are separated into different subspaces.

Importantly, the interpretation of each subspace is consistent: for example, one subspace always represents the current token across different cases. While the meaning of a subspace can sometimes vary depending on the situation, most of the time the structure is stable and interpretable. Overall, this analysis demonstrates that GPT-2 organizes information in a clear and systematic way across subspaces, making the internal representations easier to understand.

## Applicability to Larger Models

![](/imgs/blogs/decomposing-representation-space-into-interpretable-subspaces-with-unsupervised-learning-20250901000136.png)

The authors test whether NDM works on larger language models like Qwen2.5-1.5B and Gemma-2-2B. Since no detailed ground-truth circuits are available, they design knowledge conflict prompts (e.g., “Max Planck won the Nobel Prize in Literature” vs. “Max Planck won the Nobel Prize in Physics”) to check if the model uses different subspaces for context knowledge and parametric knowledge.

To evaluate this, they use subspace patching with two types of counterfactual tests:

- Param Corrupt – replace the target entity with a fake one to disrupt parametric knowledge.
- Context Corrupt – change only the last mention of the target entity to break context knowledge.

Results:

- NDM successfully identifies subspaces that behave differently under these two conditions, meaning certain subspaces specialize in context knowledge while others specialize in parametric knowledge.
- Example: In Qwen2.5-1.5B, one subspace handles parametric knowledge, while another handles context routing.
- Some subspaces show clear and consistent meanings, such as encoding the current entity directly or propagating knowledge from earlier tokens via attention.

# My thoughts

Reading this paper sparks several new directions worth exploring. First, instead of training or fine-tuning across the full parameter space, one could restrict optimization to the subspaces identified by NDM. This could lead to more efficient fine-tuning and safer model adaptation, since core knowledge subspaces could be preserved while task-specific subspaces are adjusted.

Second, the clear separation between parametric knowledge (facts stored in weights) and contextual knowledge (information from the prompt) opens the door to more controlled knowledge management. For example, one might lock parametric subspaces to prevent factual drift, while allowing contextual subspaces to update freely.

Third, the idea of subspaces as “information drawers” makes interpretability more tangible. Building visualization tools that map these drawers could make neural models more transparent to both researchers and end users.

Finally, while the paper focuses on language models, the same approach could be extended to multimodal systems. In vision–language or video models, NDM might separate subspaces for color, object identity, motion, or semantic roles. Beyond analysis, this hints at the possibility of designing interpretable-first models, where subspaces are not only discovered after training but are explicitly embedded into the architecture from the start.

# References

1. [Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](https://arxiv.org/pdf/2508.01916)
2. [Superposition, Memorization, and Double Descent](https://transformer-circuits.pub/2023/toy-double-descent/index.html)

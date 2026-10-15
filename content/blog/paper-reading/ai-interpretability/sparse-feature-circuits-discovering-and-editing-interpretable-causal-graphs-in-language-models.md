---
title: "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How Marks et al. turn sparse autoencoder features into causal circuits you can read, score with gradients, and then edit to remove a gender shortcut from a classifier."
tags: ["paper-reading", "sparse-feature-circuits", "sparse-autoencoders", "causal-graphs", "attribution-patching", "mechanistic-interpretability", "spurious-correlations", "large-language-models", "iclr-2025-oral"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 19
paper:
  title: "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models"
  authors: "Samuel Marks, Can Rager, Eric J. Michaud, Yonatan Belinkov, David Bau, Aaron Mueller"
  venue: "ICLR 2025 (oral)"
  url: "https://arxiv.org/abs/2403.19647"
---

> [!tldr]
> - **Claim.** You can explain a language model behavior as a small causal graph whose nodes are sparse autoencoder (SAE) features, not neurons or attention heads, and you can find that graph cheaply with gradient approximations.
> - **Why it matters.** Features are readable, so a circuit made of features is something a human can inspect, label, and then edit.
> - **Most surprising result.** A human who simply reads the features in a classifier's circuit and deletes the gender-related ones (the method is called SHIFT) takes a Pythia-70M profession classifier from 61.9% to 88.5% accuracy on balanced data, and to 93.1% after retraining the head, matching an oracle trained on balanced labels (93.0%). No balanced labeled data was used to pick the features.
> - **Where it fails.** Everything rests on the SAEs. Parts of the model they do not capture show up as opaque "error nodes", feature labeling is subjective, and much of the evaluation is qualitative.

Mechanistic interpretability has a granularity problem. The units that are easy to point at (attention heads, MLP blocks, neurons) usually do several unrelated jobs at once, so a circuit built from them is hard to read and harder to act on. The units that are easy to read (probe directions, hypothesized variables) require you to guess the mechanism before you look.

This paper by Marks, Rager, Michaud, Belinkov, Bau and Mueller proposes a middle path. Use sparse autoencoders to get many fine-grained, readable directions. Then use linear approximations of causal effects to find which of those directions, and which connections between them, actually drive a behavior. The result is a **sparse feature circuit**.

![Figure 1 from Marks et al. (2025): the overall pipeline. Contrastive pairs, classification data, or auto-discovered behaviors go in; a feature circuit comes out, gets interpreted, and can be debugged by ablating features.](/imgs/blogs/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models-fig1.webp)

The paper makes three contributions, and this post follows them in order:

| Contribution | Section here | What it shows |
| --- | --- | --- |
| A scalable method to discover feature circuits | Techniques 1 to 4 | Small circuits explain most of a behavior |
| SHIFT, editing a classifier through its circuit | Technique 5 | Removing a spurious cue without disambiguating labels |
| Unsupervised discovery at scale | Technique 6 | Thousands of circuits for auto-discovered behaviors |

## Why neurons are the wrong unit

Before the method, the motivation. If you have read about [superposition](/blog/machine-learning/ai-interpretability/what-is-superposition), you know the short version: a model packs more concepts than it has dimensions, so individual neurons end up responding to several unrelated things. The paper uses "neuron" in a broad sense, meaning any basis-aligned direction in a latent space, whether or not a nonlinearity precedes it.

Prior circuit work built on heads and MLP modules (see the [neuron-basis circuits explainer](/blog/machine-learning/ai-interpretability/language-model-circuits-neuron-basis)) tells you *where* a computation happens, rarely *what* it computes. The authors name two challenges: find a fine-grained unit that is interpretable without a prior hypothesis, and search causal structure over a huge number of such units cheaply.

## Technique 1: Treat SAE features as nodes in the model

**The problem.** We want readable units, and we want them to be *part of* the model's computation, so we can intervene on them and measure what changes.

**The analogy.** Think of an activation vector as a smoothie. A sparse autoencoder is a recipe card that says "this smoothie is mostly 2 parts banana, 1 part spinach, a pinch of ginger", using a very large menu of possible ingredients where only a few are ever used at once. Whatever the recipe misses is written down separately as "unexplained residue".

**The mechanism.** Given an activation $x \in \mathbb{R}^d$ at some submodule (attention output, MLP output, or residual stream), the SAE writes it as

$$
x = \hat{x} + \epsilon(x) = \sum_{i=1}^{d_{\text{SAE}}} f_i(x)\, v_i + b + \epsilon(x)
$$

Every symbol, defined:

- $d$ is the width of the model's latent space at that submodule.
- $d_{\text{SAE}}$ is the number of dictionary features, much larger than $d$.
- $v_i \in \mathbb{R}^d$ is the $i$-th feature direction, a unit vector (a column of the decoder).
- $f_i(x) \ge 0$ is the feature's activation; for a given input almost all of them are zero.
- $b \in \mathbb{R}^d$ is a bias, and $\hat{x}$ is the reconstruction.
- $\epsilon(x) = x - \hat{x}$ is the **SAE error**, everything the dictionary failed to capture.

The key move is in the last term. Most SAE work discards $\epsilon$. This paper keeps it as its own node, so the decomposition is exact and the circuit can say "this much of the behavior comes from parts the SAE does not explain yet".

![Redrawn: one activation is split by the SAE into a few active features plus an error term, and each of those becomes a separate node in the circuit graph.](/imgs/blogs/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models-1.webp)

**The two SAE suites.** The paper uses two sets of dictionaries:

| Model | SAEs | Encoder | Width | Where |
| --- | --- | --- | --- | --- |
| Pythia-70M | Trained by the authors | ReLU-linear | ${64 \times d}$ | Every attention, MLP, residual sublayer, plus embeddings |
| Gemma-2-2B | Gemma Scope (open source) | JumpReLU-linear | ${8 \times d}$ | Every sublayer, embeddings excluded |

The Pythia SAEs minimize $\lVert \hat{x} - x \rVert_2 + \lambda \lVert f \rVert_1$ with $\lambda = 0.1$, trained on about 2 billion tokens from The Pile. Scaling SAEs themselves is explicitly out of scope: the paper takes a trained suite as given.

**Making it differentiable.** The authors re-insert the decomposition into the forward pass, $x \leftarrow \sum_i f_i(x) v_i + b + \epsilon(x)$, so autograd sees $f_i$ and $\epsilon$ as variables. A stop-gradient on $\epsilon$ keeps the feature gradients from cancelling to zero, and a pass-through gradient keeps upstream gradients exact.

**When it fails.** If the SAE is poor, the error node absorbs most of the effect and the circuit becomes "a few features plus a big black box". The paper's own results show that removing residual-stream error nodes badly hurts faithfulness, which is a direct measure of how much the dictionaries still miss.

## Technique 2: Score each node by its indirect effect

**The problem.** A circuit should contain the nodes that *cause* the behavior, not just the ones that light up. That requires an intervention.

**The analogy.** To learn whether a single ingredient matters to a cake, you bake two cakes that differ only in that ingredient. Here the two "cakes" are two inputs that differ minimally, and we transplant one internal value from one run into the other.

**The mechanism.** Pick a scalar metric $m$ and a pair of inputs. For subject-verb agreement, $x_{\text{clean}}$ is "The teacher", $x_{\text{patch}}$ is "The teachers", and $m(x) = \log P(\text{"are"} \mid x) - \log P(\text{"is"} \mid x)$. The **indirect effect** (IE) of a node $a$ is

$$
\mathrm{IE}(m; a; x_{\text{clean}}, x_{\text{patch}}) = m\big(x_{\text{clean}} \mid \mathrm{do}(a = a_{\text{patch}})\big) - m(x_{\text{clean}})
$$

Here $a_{\text{patch}}$ is the value node $a$ takes when the model runs on the patch input, and $\mathrm{do}(\cdot)$ means we run the clean input but force $a$ to that value. A large IE means the node, by itself, moves the model's choice between "is" and "are".

When there is only one input (classification and next-token data later in the paper), the patch value is replaced by zero: $\mathrm{IE}(m; a; x) = m(x \mid \mathrm{do}(a = 0)) - m(x)$, a zero-ablation.

**When it fails.** Computed exactly, this is one forward pass per node. With ${64 \times d}$ features per sublayer, every sublayer, every token position, that is far too many runs. Hence the next technique.

## Technique 3: Approximate the effect with gradients

**The problem.** We need the IE of tens of thousands of nodes at once.

**The analogy.** Instead of re-baking the cake for every ingredient, measure how sensitive the taste is to each ingredient (a slope) and multiply by how much that ingredient would change. One sensitivity measurement covers every ingredient.

**The mechanism.** **Attribution patching** takes a first-order Taylor expansion of the IE around the clean activation:

$$
\widehat{\mathrm{IE}}_{\text{atp}}(m; a; x_{\text{clean}}, x_{\text{patch}}) = \nabla_a m \big|_{a = a_{\text{clean}}} \cdot \left(a_{\text{patch}} - a_{\text{clean}}\right)
$$

$\nabla_a m$ is the gradient of the metric with respect to the node, evaluated on the clean run. The whole thing needs two forward passes (clean and patch) and one backward pass, for every node simultaneously.

The paper finds this is accurate for most SAE features and errors, but it **underestimates** the true effect in the layer-0 MLP and early residual stream. So the experiments use the more expensive **integrated gradients** variant, which averages the gradient along the straight line between the two activations:

$$
\widehat{\mathrm{IE}}_{\text{ig}}(m; a; x_{\text{clean}}, x_{\text{patch}}) = \left( \frac{1}{N} \sum_{\alpha} \nabla_a m \big|_{\alpha a_{\text{clean}} + (1-\alpha) a_{\text{patch}}} \right) \cdot \left(a_{\text{patch}} - a_{\text{clean}}\right)
$$

with $N = 10$ equally spaced values $\alpha \in \{0, \tfrac{1}{N}, \dots, \tfrac{N-1}{N}\}$. It parallelizes across nodes that do not depend on each other, so its extra cost scales with $N$ times the serial depth of the network, not with the number of nodes.

**Worked micro-example (my numbers, for illustration).** Suppose a feature "plural subject" has activation $a_{\text{clean}} = 0$ on "The teacher" and $a_{\text{patch}} = 3$ on "The teachers". The gradient of $m$ with respect to it on the clean run is ${0.4}$. Attribution patching estimates $\widehat{\mathrm{IE}} = 0.4 \times (3 - 0) = 1.2$ nats of shift toward "are". If $m$ actually saturates as the feature grows, the gradient at 0 overstates or understates the change, and averaging gradients at $a = 0, 0.3, 0.6, \dots, 2.7$ (integrated gradients) gives a better estimate.

![Redrawn: exact patching, attribution patching, and integrated gradients compared on formula, cost, and accuracy. The paper uses integrated gradients for its experiments.](/imgs/blogs/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models-2.webp)

**When it fails.** Linear approximations break where the metric is strongly nonlinear in the node. On 30 across-relative-clause inputs (Appendix H), attribution patching undershoots exactly in those early components, and integrated gradients significantly tightens the fit.

## Technique 4: From nodes to a circuit, with edges

**The problem.** A list of important features is not yet a mechanism. We want to know which features feed which.

**The mechanism.** The full procedure (Figure 2) has four steps:

1. Cache activations and the metric on clean and patch inputs.
2. Backpropagate and store gradients for every feature and error node.
3. Compute each node's approximate IE, average over the dataset, and keep nodes with $|\widehat{\mathrm{IE}}| > T_N$, a node threshold.
4. Compute edge effects the same way and keep edges with $|\widehat{\mathrm{IE}}| > T_E$, an edge threshold.

![Figure 2 from Marks et al. (2025): the model is a computation graph of SAE features and errors. Activations are cached, gradients stored, node effects computed and thresholded at T_N, then edges are computed and filtered.](/imgs/blogs/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models-fig2.webp)

**Edge weights.** For an edge from an upstream node $u$ to a downstream node $d$, with $\mathcal{M}$ the set of nodes in between, the edge's effect is the effect on $m$ of patching $u$ while holding every intermediate node at its clean value. Its linear approximation is

$$
\widehat{\mathrm{IE}}(m; e) = \nabla_d m \big|_{d_{\text{clean}}} \; \nabla_{u,\, \mathrm{stop}(\mathcal{M})} d \big|_{u_{\text{clean}}} \left(u_{\text{patch}} - u_{\text{clean}}\right)
$$

Read it right to left: how much $u$ changes, times how much that change moves $d$ *directly* (gradients through intermediate nodes are stopped), times how much $d$ moves the metric. When $d$ is an SAE error (a full $d$-dimensional vector), a naive computation would need $d_{\text{model}}$ backward passes; a Jacobian-vector product trick, treating $\nabla_d m$ as a constant row vector, reduces it to two.

**Aggregation.** On templatic data, the same feature at different positions counts as different nodes; on free-form data, effects are summed over positions first, so each feature appears once.

## How good are the circuits?

The authors test circuits for subject-verb agreement in four sentence structures, on held-out pairs not used to discover the circuit.

| Structure | Example clean input | Metric |
| --- | --- | --- |
| Simple | The parents | p(is) minus p(are) |
| Within RC | The athlete that the managers | p(likes) minus p(like) |
| Across RC | The athlete that the managers like | p(do) minus p(does) |
| Across PP | The secretaries near the cars | p(has) minus p(have) |

**Faithfulness** asks how much of the model's behavior the circuit alone reproduces. Mean-ablate every node outside circuit $C$ (set it to its position-specific average) and compute

$$
\text{faithfulness}(C) = \frac{m(C) - m(\varnothing)}{m(M) - m(\varnothing)}
$$

where $m(M)$ is the full model and $m(\varnothing)$ is the empty circuit, everything mean-ablated. A score of 1 means the circuit does everything the model does. **Completeness** is the faithfulness of the complement $M \setminus C$; ideal is 0, meaning nothing is left outside.

**Worked example (my numbers).** If the full model gives $m(M) = 4.0$, the fully ablated model $m(\varnothing) = 0.5$, and the circuit $m(C) = 2.25$, faithfulness is $(2.25 - 0.5) / (4.0 - 0.5) = 0.5$. The circuit recovers half the gap between "knows nothing input-specific" and "full model".

One methodological caveat: early-layer components mostly process specific tokens, and train and test splits do not share tokens, so the authors ignore the first third of the circuit and evaluate only the latter two thirds.

![Figure 3 from Marks et al. (2025): faithfulness (left) and completeness (right) as the node threshold sweeps, for Pythia-70M (top) and Gemma-2-2B (bottom). Feature circuits reach high faithfulness with far fewer nodes than neuron circuits.](/imgs/blogs/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models-fig3.webp)

The headline comparison:

| | Pythia-70M | Gemma-2-2B |
| --- | --- | --- |
| Feature nodes for majority of behavior | about 100 | about 500 |
| Neurons for half the behavior | about 1,500 | about 50,000 |

Completeness tells the same story from the other side: ablating only a few feature nodes wipes out task performance, even with every SAE error left in place, while it takes hundreds (Pythia) or thousands (Gemma) of neurons to do the same.

Error nodes are coarse and high-dimensional, so keeping them flatters feature circuits. The authors therefore also plot circuits with error nodes removed: dropping residual-stream errors badly hurts faithfulness, while dropping attention and MLP errors hurts much less.

**Interpretability.** On a 0 to 100 scale (Pythia SAEs, Appendix F), crowdworkers rated random neurons 32.6 and agreement-circuit neurons 30.2, against 52.8 for random SAE features, 62.3 for agreement-circuit features and 81.5 for Bias in Bios circuit features. Raters came from the ARENA Slack channel, which the authors flag as a likely source of bias.

## Case study: agreement across a relative clause

To make a circuit small enough to annotate by hand, the authors raise the node threshold until faithfulness is just above 0.2. That yields 86 nodes for Pythia (faithfulness 0.21) and 223 nodes for Gemma (faithfulness 0.21).

![Figure 4 from Marks et al. (2025): summarized circuits for "The girl/girls that the teacher sees has/have" in Pythia (a) and Gemma 2 (b). Squares count feature nodes, triangles count error nodes, shading shows summed effect.](/imgs/blogs/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models-fig4.webp)

Both models use two pathways:

- **Direct.** Features detect the number of the main subject ("girl" versus "girls") and generically promote the matching verb form.
- **Relay.** The same number detection is carried forward by PP/RC boundary detectors to the end of the relative clause, where verb-form discriminators fire.

Gemma 2 adds **noun-phrase number trackers**, features that pick up the number of the noun heading a noun phrase and stay active on every token until the phrase ends, promoting matching verbs most strongly at its last token.

The across-RC circuit overlaps heavily with the across-PP circuit, so the models treat two syntactically different structures mostly uniformly. As Finlayson et al. (2021) found, there is less overlap with the simple and within-RC circuits.

## Technique 5: SHIFT, editing a classifier through its circuit

**The problem.** Classifiers learn shortcuts. Almost every fix for spurious correlations needs *disambiguating* labeled data, where the shortcut and the real signal come apart. Sometimes that data does not exist: the two signals are perfectly correlated in everything you have.

**The analogy.** Rather than collecting a balanced dataset to retrain a biased screener, you open it up, read what it pays attention to, and cross out "gender" with a pen.

**The mechanism.** Spurious Human-interpretable Feature Trimming (SHIFT), for a labeled dataset $D = \{(x_i, y_i)\}$ and classifier $C$:

1. Discover a feature circuit for $C$'s accuracy, using metric $m = -\log C(y \mid x)$ and the zero-ablation IE.
2. A human inspects every feature in the circuit and judges whether it is task-relevant.
3. Zero-ablate the task-irrelevant features to get $C'$.
4. Optionally, fine-tune $C'$ on the same data $D$.

**The setup.** The Bias in Bios dataset classifies a person's profession from their biography, and also labels gender. The authors build a worst case:

- **Ambiguous set (training):** male professors (label 0) and female nurses (label 1). Gender predicts profession perfectly.
- **Balanced set (evaluation only):** equal numbers of male and female professors and nurses.

A linear head trained on the ambiguous set has no reason to prefer profession over gender. The feature circuit for that classifier contains 67 features in Pythia and 46 in Gemma. Reading each one on Neuronpedia (top-activating text plus its effect on output logits), the authors judge 55 Pythia features and 43 Gemma features task-irrelevant, for example features that promote female-associated language in biographies of women.

![Redrawn: the SHIFT pipeline on Pythia-70M, from the ambiguous training set through circuit discovery, human judgment of 55 of 67 features, ablation, and retraining, with the balanced-set results.](/imgs/blogs/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models-3.webp)

**Baselines.** To check that the human judgment and the features both matter, the paper compares:

- **CBP**, concept bottleneck probing adapted from Yan et al. (2023).
- **Random**: ablate the same number of randomly chosen SAE features.
- **Feature skyline**: ablate the 55 (or 43) circuit features most implicated in *gender* accuracy, chosen using balanced data, so this cheats.
- **Neuron skyline**: the same, with neurons, mean-ablated.
- **Oracle**: a classifier trained on balanced labels.

![Table 2 from Marks et al. (2025): profession, gender, and worst-group accuracy on the balanced set for Pythia-70M and Gemma-2-2B.](/imgs/blogs/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models-fig5.webp)

The Pythia-70M numbers in plain form (gender accuracy near 50% means the classifier no longer encodes gender):

| Method | Profession (higher better) | Gender (lower better) | Worst group |
| --- | --- | --- | --- |
| Original | 61.9 | 87.4 | 24.4 |
| Random ablation | 61.8 | 87.5 | 24.4 |
| Neuron skyline | 75.5 | 73.2 | 41.5 |
| Feature skyline | 88.5 | 54.3 | 62.9 |
| SHIFT | 88.5 | 54.0 | 76.0 |
| SHIFT + retrain | 93.1 | 52.0 | 89.0 |
| Oracle | 93.0 | 49.4 | 91.9 |

Three readings:

- **Human selection matches the cheating skyline.** SHIFT without retraining ties the feature skyline on profession accuracy (88.5) while never seeing balanced labels.
- **Features, not ablation per se, do the work.** Random feature ablation changes nothing.
- **Neurons cannot do this.** The authors abandoned SHIFT-with-neurons because essentially no neuron was interpretable. Even the neuron skyline, which uses balanced data, lags, because the neurons most tied to gender are also useful for profession.

On Gemma-2-2B, the ablation alone damages profession accuracy (76.0) but cuts gender accuracy to 51.5; retraining the head restores profession to 95.0 with gender staying at 52.4, matching the 95.0 oracle.

**When it fails.** SHIFT only removes signals a human can recognize in feature form. A shortcut that lives in SAE error nodes, or in features whose label does not reveal the problem, survives. And "task-irrelevant" is a judgment call that may differ across annotators.

## Technique 6: Unsupervised circuit discovery at scale

**The problem.** Every circuit so far needed a human-built dataset defining the behavior. Models do many things humans would never think to test.

**The mechanism.** Two steps, run on Pythia-70M over a large subset of The Pile:

1. **Behavior discovery by clustering.** Treat the corpus as (context, next token) pairs. Keep only pairs the model predicts confidently and correctly (cross-entropy below 0.1 or 0.3 nats). Map each pair to a vector, such as the training gradient $\nabla_\theta \log P_\theta(y_i \mid x_i)$ following Michaud et al. (2023), or vectors built from SAE activations, and cluster. Each cluster is a candidate "behavior".
2. **Circuit discovery.** For each cluster, run the zero-ablation version of the method with $m = -\log P(y_i \mid x_i)$.

The output is thousands of clusters, each with its own feature circuit, published at feature-circuits.xyz.

![Figure 5 from Marks et al. (2025): two auto-discovered clusters and features in their circuits. Blue shading marks where a feature is active, red marks the tokens it promotes.](/imgs/blogs/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models-fig6.webp)

The examples are illuminating because circuits split what looks like one behavior into several mechanisms:

- **Cluster 382, incrementing sequences** ("Step 1 ... Step 2 ... Step 3 ... Step"). The circuit uses general *succession* features plus several narrow *induction* features that recognize patterns like "A3 ... A" and copy from the "3".
- **Cluster 475, "to" as infinitive object.** One feature promotes "to" after words that take infinitive objects; a separate feature fires on objects of verbs or prepositions and promotes "to" as an object complement.

The pipeline also rediscovers attention features for succession and induction, previously studied at the head level with hand-curated data: a useful sanity check.

**When it fails.** The paper calls evaluating these clusters and circuits an open problem; the published circuits are unlabeled, and "interesting" is an informal judgment.

## What transfers and what might not

The load-bearing assumptions in this setup:

- **Good SAEs exist for every sublayer.** A model without such a suite first needs a large one-time training cost.
- **Small models.** The quantitative results are on a 70M and a 2B model. My extrapolation: the method's cost scales with the number of SAE features and the serial depth (because of integrated gradients), so on much larger models the per-circuit cost and the size of circuits a human must read both grow.
- **Templatic evaluation.** Faithfulness and completeness are measured only on structured agreement data; the free-form settings are judged downstream or qualitatively.

## Critique

**What is strong.**

- Keeping SAE errors as nodes is a quietly important design choice. It turns "how much does the SAE miss" into a measurable part of every circuit rather than a hidden assumption.
- SHIFT is the clearest demonstration I know of that SAE features are *useful*, not just pretty: a downstream number moves, against strong and cheating baselines.
- The paper reports where its own approximations break (early layers) and fixes them rather than hiding it.

**What is weak.**

- Faithfulness evaluation drops the first third of each circuit. That is justified by token mismatch, but it means the early part of every circuit is effectively unevaluated.
- The case-study circuits are tuned to faithfulness of only about 0.21 so they are small enough to label. The readable story covers about a fifth of the behavior.
- The SHIFT result is one dataset with one spurious attribute, and the human annotators are the authors. A study with independent annotators would make the human-in-the-loop claim much stronger.
- Interpretability ratings come from a self-selected group of alignment-interested researchers.

**Missing ablation.** I would want SHIFT run with a range of thresholds and with multiple independent annotators, reporting variance in which features are cut and in the resulting accuracy.

**What would change my mind.** If an independent group of annotators, given the same 67 Pythia features, disagreed substantially on which ones are gender-related and their choices produced much worse balanced-set accuracy, I would read SHIFT's success as partly author knowledge rather than feature legibility. Conversely, if SHIFT worked on a second spurious attribute where the authors did not know the cue in advance, I would take the "without knowing the signal ahead of time" claim at full strength.

## References

- Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., Mueller, A. *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models.* ICLR 2025. [arXiv:2403.19647](https://arxiv.org/abs/2403.19647)
- Code, data and autoencoders: [github.com/saprmarks/feature-circuits](https://github.com/saprmarks/feature-circuits)
- Browsable clusters and circuits: [feature-circuits.xyz](https://feature-circuits.xyz)
- On this site: [What is superposition?](/blog/machine-learning/ai-interpretability/what-is-superposition), [Language model circuits in the neuron basis](/blog/machine-learning/ai-interpretability/language-model-circuits-neuron-basis), and [Decomposing representation space into interpretable subspaces](/blog/paper-reading/ai-interpretability/decomposing-representation-space-into-interpretable-subspaces-with-unsupervised-learning)

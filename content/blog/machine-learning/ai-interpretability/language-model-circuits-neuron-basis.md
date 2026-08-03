---
title: "Language Model Circuits in the Neuron Basis: Why MLP Activations Can Be Sparse and Faithful"
date: "2026-08-03"
publishDate: "2026-08-03"
description: "An intuition-first guide to Transluce's result that raw MLP neurons, combined with RelP attribution, can recover sparse and causally useful language-model circuits without first training an SAE."
tags: ["ai-interpretability", "mechanistic-interpretability", "circuit-tracing", "neuron-circuits", "sparse-autoencoders", "relp", "attribution", "llama", "causal-analysis"]
category: "machine-learning"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 28
image: "/imgs/blogs/language-model-circuits-neuron-basis-1.webp"
excerpt: "Raw neurons are often dismissed as too polysemantic for circuit tracing. This deep dive explains why MLP activations and RelP make that assumption worth revisiting."
---

There is a familiar interpretability workflow that feels almost inevitable. Take a language model, collect a large activation dataset, train a sparse autoencoder, give the learned features friendly names, and only then start tracing computation. The raw neurons are treated as an inconvenient implementation detail: too many of them, too entangled, too unreliable to serve as a basis for explanation.

That workflow may be backwards more often than we think.

Transluce's study, *Language Model Circuits Are Sparse in the Neuron Basis*, revisits the claim that neurons are intrinsically a poor unit of analysis. The authors make two changes to the baseline that matter more than the headline suggests: they use post-nonlinearity MLP activations rather than MLP outputs, and they replace the usual Integrated Gradients attribution with RelP. On the subject–verb agreement benchmark, this combination produces circuits that are much smaller than circuits built from MLP outputs and competitive with SAE-based circuits. The same neuron-level approach also recovers recognizable structure in multi-hop reasoning, addition, multilingual antonyms, and a synthetic user-modeling task [Transluce, 2025](https://transluce.org/neuron-circuits).

![A sparse causal circuit connects input pairs to selected MLP neurons, weighted edges, and an intervention-tested output.](/imgs/blogs/language-model-circuits-neuron-basis-1.webp)

The diagram above is the mental model: we are not asking a neuron description to sound plausible. We are selecting a small subgraph, intervening on its complement, and asking whether the model's behavior moves as the graph predicts. The important object is therefore a causal graph with measurable obligations, not a collection of appealing labels.

## 1. The assumption that neurons are not enough

The strongest version of the old argument is easy to understand. A model may represent one human concept across many coordinates, and one coordinate may participate in many unrelated concepts. If a feature is distributed, looking at one neuron gives an incomplete story. If a neuron is polysemantic, its name is not a faithful explanation. Learned features appear attractive because an SAE explicitly searches for a sparse dictionary in which individual latents might be easier to interpret.

The argument becomes less decisive when we separate three questions that are often bundled together:

| Question | Naive interpretation | More precise engineering question |
| --- | --- | --- |
| What is a unit? | A neuron must mean one thing | Which basis yields a small causal graph for this task? |
| What is important? | A large activation is important | Which intervention changes the target metric? |
| What is an explanation? | A readable description is enough | Does the selected subgraph preserve and account for behavior? |

An SAE can be useful without being the only valid basis. It adds a learned coordinate system, but it also adds approximation error, a training cost, choices about width and sparsity, and the possibility that one learned feature splits or merges behaviors in unintuitive ways. The neuron basis has the opposite trade-off: it is already present in the model, but the coordinate system may be less semantically clean.

The study's claim is deliberately narrower than “neurons are monosemantic.” It does not establish that every neuron has one human-readable meaning. It asks whether raw MLP neurons can form sparse and faithful circuits for concrete tasks when the representation site and attribution method are chosen carefully. That is a much more useful question because circuits are task-conditional by construction.

> A unit does not need to be universally interpretable to be causally useful in a task-specific circuit.

## 2. What a circuit actually contains

A circuit is a sparse subgraph $C = (V, E)$ of the model's computational graph. $V$ is a set of nodes; in this study, a node can be an MLP neuron at a particular layer and token position. $E$ is a set of directed edges between nodes. The same neuron index at two token positions counts as two different circuit nodes because the computation happens at different locations in the sequence.

The distinction between a node and an edge is not cosmetic. A node can be active without carrying the task-relevant signal to the output. Conversely, a small number of strong edges can explain why one intermediate representation affects a later one. A node-only circuit answers “which units matter?” An edge-aware circuit also answers “which causal route does the information take?”

The evaluation procedure uses mean ablation. Given an input $x$ and a dataset $mathcal{D}$, the complement of the circuit is replaced by average activations while the selected circuit remains intact:

$$
C(x) := M\left(x;\operatorname{do}\left(v = \mathbb{E}_{d \sim \mathcal{D}}[v(d)]\right)\ \text{for}\ v \in \overline{C}\right).
$$

Here $M$ is the original model, $v$ is a circuit component, and $\overline{C}$ is the complement of the selected circuit. The notation is written as an intervention because the test is asking what happens when we surgically replace internal computation, not merely observe a correlation.

![The original page's circuit-evaluation figure, cropped to the intervention and output-comparison procedure.](/imgs/blogs/language-model-circuits-neuron-basis-eval.webp)

There are two complementary metrics. Faithfulness asks whether retaining the circuit while ablating its complement preserves the original model's behavior. Completeness asks whether ablating the circuit itself removes approximately as much task-specific behavior as ablating the full model. In the paper's notation, with $m$ as the output metric and $\varnothing$ as the fully ablated baseline:

$$
\mathsf{Faithfulness}(C) = \frac{\mathbb{E}_{x \sim \mathcal{D}}[m(C,x)-m(\varnothing,x)]}{\mathbb{E}_{x \sim \mathcal{D}}[m(M,x)-m(\varnothing,x)]}
$$

$$
\mathsf{Completeness}(C) = \frac{\mathbb{E}_{x \sim \mathcal{D}}[m(\overline{C},x)-m(\varnothing,x)]}{\mathbb{E}_{x \sim \mathcal{D}}[m(M,x)-m(\varnothing,x)]}.
$$

The notation makes the desired corner visible: high faithfulness and low completeness. A perfect circuit preserves the model when its complement is removed, while removing the circuit causes the model to lose the target behavior. Sparsity is then a third requirement: keep $|V|$ and $|E|$ small enough that a human can inspect the result.

## 3. The basis choice: MLP activation versus MLP output

The paper compares several possible representation sites. A Transformer block produces attention outputs, post-nonlinearity MLP activations, MLP outputs after the down projection, and a residual stream that accumulates component outputs. These are not interchangeable views of the same information.

For an MLP layer, the activation vector lives in the feed-forward width $d_{\mathrm{ffn}}$. The down projection maps it back into the model width $d_{\mathrm{model}}$. If we trace the projected MLP output, many hidden neurons have already been mixed together by that projection. If we trace the activation coordinates, we inspect the coordinates immediately after the element-wise nonlinearity.

That location is what the paper calls a privileged basis. The architecture itself treats each activation coordinate separately at the nonlinearity. The claim is not that the coordinate is guaranteed to be a concept; it is that the architecture gives the coordinate a structural status that the post-projection mixture does not have.

![The original faithfulness/completeness comparison across representation bases, cropped from the source visualization.](/imgs/blogs/language-model-circuits-neuron-basis-fc-log-bases.webp)

The result is large enough to change the experimental default. With Integrated Gradients, MLP activations produce circuits about 100 times smaller than MLP outputs at comparable evaluation quality. The activation basis also closes much of the gap with SAE bases. The right interpretation is not “the number 100 is a universal law”; it is that a representation-site decision can dominate the apparent quality of a basis.

The paper's result can be summarized as a diagnostic table:

| Representation | What it preserves | Main risk for tracing |
| --- | --- | --- |
| MLP activation | Individual post-nonlinearity coordinates | A neuron can still be polysemantic |
| MLP output | A projected MLP contribution in model width | Many hidden coordinates are mixed together |
| Attention output | Sequence-mixing result | Attribution may be concentrated or irregular |
| Residual stream | Running sum of prior components | Many mechanisms are superposed in one stream |
| SAE feature | Learned sparse dictionary coordinate | Approximation error and retraining cost |

In production interpretability work, this is a cheap experiment to run before adopting a learned representation: hold the task and evaluation fixed, compare the same attribution budget at the most informative sites, and plot circuit size against causal quality.

## 4. SVA: a controlled test of circuit sparsity

The subject–verb agreement benchmark is useful because it creates paired examples whose relevant distinction is explicit. In the `simple` subset, the original input can be “The parents” with output “are,” while the counterfactual is “The parent” with output “is.” Other subsets insert relative clauses or noun phrases that act as distractors.

![The original SVA result, cropped to the before/after comparison of attribution and representation choices.](/imgs/blogs/language-model-circuits-neuron-basis-sva-result.webp)

The experiments use the Llama 3.1 8B base model, 300 training pairs, and 40 held-out validation pairs across four SVA templates. For a pair $(x,x')$, the metric compares the logit for the original target token $y$ against the counterfactual target $y'$:

$$
m(C,x) = [C(x)]_y - [C(x)]_{y'}.
$$

The circuit is built by greedily selecting the highest-attribution nodes. The evaluation then asks whether a small node set retains the grammatical-number decision on validation examples. This is a much more disciplined test than looking at a heatmap and declaring that a neuron “looks like plural.”

The node basis is also input-dependent. A circuit for “The parents” need not contain exactly the same nodes as a circuit for “The athlete that the managers like.” That variability is not automatically a defect; it is a consequence of tracing a conditional computation rather than searching for one universal static graph.

### Attribution distributions are the hidden lever

Greedy selection works best when attribution is concentrated: a few nodes should receive large scores while most nodes receive small scores. MLP activations have a wider attribution distribution and stronger outliers than the alternative representation sites. The practical implication is simple: a top-$k$ cutoff captures more of the target effect when the score distribution has a heavier tail.

![Neuron attribution scores across the SVA task, cropped from the source page.](/imgs/blogs/language-model-circuits-neuron-basis-neuron-scores-sva.webp)

The layer distribution matters too. MLP output scores are highly concentrated in the final two layers, while MLP activation scores are spread more evenly through depth. That spread is useful when the behavior depends on intermediate computation. A circuit that only lights up at the end may still be correct, but it provides less evidence about how the model built the answer.

![The layer distribution of selected neurons, cropped from the source page.](/imgs/blogs/language-model-circuits-neuron-basis-neuron-layer-sva.webp)

## 5. RelP: attribution that matches the computational graph

Integrated Gradients is principled but expensive for deep internal tracing. In the reported setup, IG uses 10 backward passes. It also estimates a path integral by sampling intermediate points between the original and counterfactual activations. That can be noisy when the model has many nonlinear interactions.

RelP takes a different route. It constructs a replacement model whose forward pass is the same as the original on the chosen input, but whose backward pass locally freezes nonlinearities. For Llama 3, this means treating the normalization scale, SiLU gate, and attention weights as fixed during the attribution calculation. The paper also applies the half rule to multiplicative interactions so attribution is conserved across layers.

For a node $v$, the exact RelP expression reported in the article is:

$$
\begin{aligned}
\mathsf{Attribution}_{\mathsf{RelP}}(v) &= \mathbb{E}_{(x,x')\sim\mathcal{D}}[\mathsf{RelP}_v(x;x')] \\
\mathsf{RelP}_v(x;x') &= (v(x)-v(x'))\frac{\partial m(M_{\mathrm{replacement}},x)}{\partial v(x)}.
\end{aligned}
$$

The equation has a useful intuition. The first term asks how much the node changed between the original and counterfactual examples. The second asks how sensitive the task metric is to that node under the locally linearized backward pass. Their product says: a node matters when it changes for the counterfactual distinction and the output cares about that change.

That last sentence is an explanatory abstraction, not a replacement for the paper's implementation. The actual method overwrites backward behavior for selected components and computes gradients on saved activations with `torch.autograd.grad`.

![The source comparison of attribution quality for IG and RelP, cropped to the main log-scale result.](/imgs/blogs/language-model-circuits-neuron-basis-fc-log-adag-ig-relp.webp)

The empirical result is the important part. On the MLP activation basis, RelP reaches near-perfect faithfulness and completeness with roughly 200 neurons. It also improves the other representation choices in most settings and closes the remaining gap between MLP activations and SAE features.

### An illustrative implementation

The following code is an explanatory abstraction of the score, not the paper's full replacement-model implementation. It shows the bookkeeping that matters: preserve the original and counterfactual activations, run the locally linearized model, and multiply activation difference by the metric gradient.

![The attribution comparison is the visual reference for the explanatory RelP score below.](/imgs/blogs/language-model-circuits-neuron-basis-fc-log-adag-ig-relp.webp)

```python
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RelPBatch:
    original: Tensor       # [batch, neurons]
    counterfactual: Tensor # [batch, neurons]
    gradient: Tensor       # [batch, neurons]


def relp_score(batch: RelPBatch) -> Tensor:
    """Educational RelP-style node score, not the production implementation."""
    delta = batch.original - batch.counterfactual
    per_example = delta * batch.gradient
    return per_example.mean(dim=0)


def select_circuit(scores: Tensor, fraction: float = 0.01) -> Tensor:
    """Keep the highest absolute attribution scores."""
    count = max(1, int(scores.numel() * fraction))
    _, indices = torch.topk(scores.abs(), k=count)
    return indices.sort().values


original = torch.randn(40, 16)
counterfactual = torch.randn(40, 16)
gradient = torch.randn(40, 16)
scores = relp_score(RelPBatch(original, counterfactual, gradient))
selected = select_circuit(scores, fraction=0.25)

print("selected neurons:", selected.tolist())
print("largest score:", scores.abs().max().item())
```

For real circuit tracing, the gradient must come from a replacement model with the correct local rules. Replacing that gradient with an ordinary gradient from the original nonlinear network would change the method, even if the surrounding Python looks similar.

## 6. The unpaired setting: when counterfactuals are unavailable

Paired inputs are convenient because they isolate a feature. Real datasets are messier. We may not know how to construct a clean counterfactual for “the model is implicitly inferring this user attribute,” or we may not have a templatic task at all.

The article tests a zero baseline in place of the counterfactual. In the unpaired setting, the exact node expressions become:

$$
\begin{aligned}
\mathsf{IGAct}_v(x) &= v(x)\int_{\alpha=0}^{1}\frac{\partial m(M,x;\operatorname{do}\,v=\alpha v(x))}{\partial v(x)}\,\mathrm{d}\alpha, \\
\mathsf{RelP}_v(x) &= v(x)\frac{\partial m(M_{\mathrm{replacement}},x)}{\partial v(x)}.
\end{aligned}
$$

The zero baseline is simpler but can create out-of-distribution activations during ablation. The authors therefore evaluate both mean ablation and zero ablation. MLP activations remain considerably more faithful than alternatives, although completeness can be worse than the residual stream under zero ablation.

![The unpaired sparse-circuit evaluation, cropped to the mean/zero-ablation comparison.](/imgs/blogs/language-model-circuits-neuron-basis-fc-log-bases-relp.webp)

This is a useful operational warning. “No counterfactual labels” does not mean “no evaluation design.” It means the baseline and intervention become part of the scientific claim. A circuit that looks sparse under zero ablation may be exploiting an unnatural internal state; mean ablation is usually a safer first comparison when a dataset mean is available.

## 7. From important neurons to important edges

Finding nodes is only half of circuit tracing. Suppose neuron A and neuron B both correlate with the answer. That does not tell us whether A influences B, whether both receive a common signal, or whether one is an irrelevant downstream echo.

For an edge from source $v_s$ to target $v_t$, the paper defines a RelP-style score using the source activation and the derivative of the target activation in the replacement model:

$$
\mathsf{RelP}_{v_s\to v_t}(x) = v_s(x)\frac{\partial v_t^{\mathrm{replacement}}(x)}{\partial v_s^{\mathrm{replacement}}(x)}.
$$

Scores are collected across the dataset and thresholded by percentage. A neuron is removed if all its incoming or outgoing edges fall below the threshold, ensuring that the retained graph contains connected units rather than isolated high-scoring points.

![Edge pruning results from the source page, cropped to the linear-scale faithfulness/completeness plot.](/imgs/blogs/language-model-circuits-neuron-basis-fc-edges-aggregated-linear.webp)

RelP reaches over 80% faithfulness while retaining roughly $10^5$ edges, about 10% of the 500,000 candidate edges in the reported SVA experiment. This is the difference between a feature inventory and a circuit. The inventory says “these neurons are relevant.” The circuit says “this directed flow is sufficient to reproduce the behavior.”

The edge graph also changes how we should read interpretability visualizations. A dense hairball is not automatically a richer explanation. If the causal edges cannot be thresholded without destroying faithfulness, the method has not found a compact mechanism yet. If the graph can be pruned to a small connected structure while preserving the metric, the visualization becomes an auditable hypothesis.

## 8. Case study: Dallas → Texas → Austin

The first case study tests multi-hop reasoning about state capitals. The prompt asks: “What is the capital of the state containing Dallas?” The intended chain is not a single lookup from Dallas to Austin. It is:

$$
\text{Dallas} \rightarrow \text{Texas} \rightarrow \text{Austin}.
$$

The dataset contains 50 questions in the same general style. For the Texas example, automatic tracing recovers 257 neurons, after which the authors manually identify 23 neurons with especially meaningful descriptions. Those neurons cluster into categories such as Dallas, Texas, capital, and Austin.

![The exact source crop of the Texas circuit and its clustered reasoning roles.](/imgs/blogs/language-model-circuits-neuron-basis-texas-circuit.webp)

The important evidence is not that every neuron has a perfect label. It is that the clusters correspond to distinct roles in the computation and that steering those clusters changes the output in interpretable directions. A capital-related cluster should affect the capital answer; a state-related cluster should affect the intermediate entity. The graph provides a place to test those expectations.

![The source crop showing steering effects for the state-capital clusters.](/imgs/blogs/language-model-circuits-neuron-basis-capitals-steering.webp)

This case study also explains why a task-specific circuit can tolerate polysemantic neurons. If one neuron participates in several behaviors, its inclusion is still useful when the selected edge path and intervention effect are specific to the state-capital task. We should avoid upgrading “useful in this circuit” into “has one universal meaning.”

## 9. Case study: two-digit addition

The addition task asks the model to solve two-digit arithmetic. The circuit analysis searches for neurons whose attribution scores separate labeled properties such as the ones digit, the tens digit, or the answer modulo 10.

For a class label $a$, the authors split examples into $\mathcal{D}^{+}$, where the label equals $a$, and $\mathcal{D}^{-}$, where it does not. They score a neuron using AUROC:

$$
\mathrm{AUROC}(v) = \mathbb{P}_{x^{+}\sim\mathcal{D}^{+},\,x^{-}\sim\mathcal{D}^{-}}\left[\mathsf{Attribution}_v(x^{+}) > \mathsf{Attribution}_v(x^{-})\right].
$$

AUROC is not a causal proof by itself. It is a discovery tool that identifies neurons worth inspecting, steering, and testing with ablation. The causal claim comes later, from intervening on the selected units and measuring the output.

![The source case-study crop containing the addition and state-capital result panels.](/imgs/blogs/language-model-circuits-neuron-basis-case-study-result.webp)

The strongest repeated pattern is the ones digit modulo 10. The authors repeat the analysis for moduli 2 through 9 as a robustness check. Most coprime moduli stay near random AUROC values between 0.4 and 0.6. Modulo 10 is the distinctive structure, with more neurons above 0.8 or below 0.2. A small number of exceptions occur for modulo 2 and modulo 5, which is plausible because both divide 10.

![The exact modulo histogram crop from the addition analysis.](/imgs/blogs/language-model-circuits-neuron-basis-mod-histogram.webp)

The tens-digit neurons are noisier. They may be approximating the overall sum rather than calculating the tens digit as a clean independent feature. That distinction is valuable: a circuit analysis does not need every discovered neuron to match the conceptual vocabulary we started with. It should reveal where the evidence is crisp and where it is approximate.

## 10. Case study: multilingual antonyms

The multilingual task contains 54 prompts: 9 languages multiplied by 6 concepts. The prompt asks for an antonym, such as “What is the opposite of big?” with the answer “small.” The analysis searches for three kinds of features:

| Feature | What it captures | Result |
| --- | --- | --- |
| Language | The language of the prompt or output | Replicated |
| Concept | Language-independent meaning such as “hot” | Replicated |
| Attribute | Semantic axis such as temperature | Replicated |

![The source distribution of maximum AUROCs for language, concept, and attribute features.](/imgs/blogs/language-model-circuits-neuron-basis-roc-aucs.webp)

The layer distributions make the hierarchy more concrete. Language neurons appear throughout depth with a large final-layer peak. Concept neurons appear more in early and middle layers. Attribute neurons arise mainly in middle layers.

![The source layer-distribution crop for multilingual feature types.](/imgs/blogs/language-model-circuits-neuron-basis-features-by-layer.webp)

There is no single universal “antonym neuron.” That absence is itself informative. In the state-capital example, one highly important state-capital neuron can appear across prompts. In multilingual antonyms, the relation is assembled from language, concept, and attribute signals. A good circuit explanation should preserve that difference instead of forcing every task into the same semantic template.

The study also finds that some automatic neuron descriptions are multilingual in a way that would be easy to miss from a single example. A neuron may respond to words related to “cold” in German and Spanish, or to language-specific grammatical forms. The best interpretation therefore combines descriptions, exemplars, attribution, and intervention.

## 11. Case study: user modeling and its uncomfortable edge

The final case study investigates the inferences a model makes about a user. The setup is deliberately synthetic: a user shares a fact, then asks for a hypothetical Wikipedia biography infobox. The assistant response is prefixed so the model must predict a demographic field such as gender, country of origin, occupation, or religion.

This forced-output format makes tracing possible because the target attribute appears as an output token. Given a gym-related prompt, for example, the model is asked to produce an infobox containing a gender field. The analysis traces the token corresponding to the predicted value back through the model.

The result is a useful demonstration and a limitation at the same time. It shows that neuron tracing can surface internal signals tied to inferred demographic attributes. It does not show that the model's inference is accurate, justified, or ethically acceptable. Nor does it show that the model stores a stable “belief state” at one token position. The authors note that these inferences are not stored at a fixed token position, which makes access difficult.

This is where causal interpretability becomes more than a visualization exercise. If a circuit exposes a protected-attribute inference, we need to ask what intervention changes it, whether the intervention harms other behavior, and whether the synthetic output format is measuring a behavior that occurs in normal interaction. A readable graph is an invitation to audit, not a license to trust.

## 12. What the result does and does not establish

The headline is easy to overstate, so the boundaries matter.

First, the work does not prove that raw neurons are universally more interpretable than SAE features. Learned features may still be more monosemantic on some tasks, especially when the extra dimensionality allows a cleaner decomposition. The study's narrower contribution is to challenge the prior that neurons are automatically an uninterpretable basis.

Second, the circuits are sparse relative to the alternatives, not necessarily small enough for a human to read without tooling. A circuit with hundreds of neurons and around 100,000 edges is better than a dense graph, but it is not a finished explanation.

Third, descriptions remain imperfect. The authors use automatic descriptions and top-activating exemplars, then manually inspect clusters. Subtle patterns can be missed by the description model and appear only in the exemplars. This means the natural-language label is evidence about a node, not the node's definition.

Fourth, efficiency is still a practical concern. The implementation relies on manual calls to `torch.autograd.grad`, sometimes with very small activations and low hardware utilization. Edge computation could not be fully batched across examples. Not loading SAEs into memory makes neuron tracing tractable for larger models, but “tractable” is not the same as cheap.

Finally, a circuit is input-dependent. A task-level claim should be supported across a dataset, not inferred from one attractive graph. When a circuit is traced for one example, we should call it an example circuit. When we aggregate across examples and validate on held-out data, we can make a stronger task-level statement.

## 13. A practical workflow for neuron-level circuit tracing

For an interpretability project, the result suggests a conservative sequence:

1. Define a behavior metric and, when possible, an original/counterfactual pair.
2. Compare MLP activations with MLP outputs before introducing a learned basis.
3. Use a locally faithful attribution method and record the backward-pass assumptions.
4. Select nodes by attribution, then compute and prune edges separately.
5. Evaluate faithfulness and completeness on held-out examples.
6. Use descriptions and exemplars to generate hypotheses, not to replace interventions.
7. Steer or ablate discovered clusters and record both desired and collateral effects.
8. Report the baseline, ablation distribution, circuit size, edge count, and compute cost.

The following table is a useful decision rule:

| Situation | Start with | Why |
| --- | --- | --- |
| Paired templatic benchmark | MLP activations + RelP | The counterfactual isolates a causal distinction |
| Messy unpaired data | MLP activations + zero/mean baseline comparison | It tests how sensitive conclusions are to the baseline |
| Need for a human-readable feature dictionary | Neuron circuit first, SAE second | Establish whether the extra basis is buying causal quality |
| Many candidate edges | Edge-level RelP and threshold sweep | Node relevance does not identify information flow |
| Sensitive user modeling | Circuit tracing plus intervention audit | Visibility must be separated from validity and safety |

### A small worked example

Imagine a four-neuron circuit with RelP scores `[0.42, 0.08, -0.31, 0.01]`. If we select by absolute score with a threshold of 0.1, neurons 0 and 2 remain. That selection is not yet an explanation. We still need to keep the edges between those nodes, ablate their complement, and measure whether the target logit difference remains close to the original model.

If faithfulness is 0.96 but completeness is 0.45, the selected graph preserves the behavior while removing it does not explain most of the model's loss. If faithfulness is 0.55 and completeness is 0.92, the graph captures a lot of the behavior when removed but is not sufficient on its own. The useful circuit is the region where both obligations are satisfied with a small node and edge count.

That is the core discipline: never confuse a ranking with a causal explanation.

## 14. How to read the result without fooling yourself

The most dangerous failure mode in interpretability is not a crash. It is a polished plot that answers a weaker question than the caption implies. A circuit can be sparse because the threshold was aggressive. A neuron can have a high AUROC because it tracks a proxy. A graph can preserve a logit difference while missing the actual computation that generalizes to a new prompt. The evaluation contract has to make those possibilities visible.

### Keep the axes honest

When a plot places circuit size on the horizontal axis, confirm what “size” means. It may be the number of neurons, the number of edges, the percentage of candidate edges, or a logarithmic coordinate. Comparing a 200-neuron node circuit to a 100,000-edge circuit without naming the unit is not a meaningful sparsity comparison. Nodes and edges answer different questions and should be reported separately.

The same applies to faithfulness and completeness. A line close to 1 is not automatically good if its denominator is nearly zero, if the baseline is unusual, or if the metric is only loosely related to the behavior we care about. The target metric should be defined before looking at the plot. For SVA, the difference between the original and counterfactual verb logits is a sensible metric because it directly encodes grammatical number. For user modeling, a forced demographic token is much narrower than “the model has formed a complete user model.”

### Separate discovery from confirmation

The AUROC scan is a discovery phase. It ranks neurons that separate labeled examples. It is useful because searching every neuron by hand is impossible. It is not confirmation because a separating score can arise from a correlated feature, a token-position artifact, or dataset leakage.

Confirmation should use a fresh intervention. If a neuron is supposed to encode the ones digit, steer or ablate it and measure whether the predicted ones digit changes while unrelated arithmetic behavior remains stable. If a cluster is supposed to represent Texas in the capital task, intervene on that cluster and compare the output against interventions on the Dallas and Austin clusters. If the effects are not selective, the label is still a hypothesis.

This distinction suggests a simple experiment log:

| Stage | Input to the analysis | Output to record | What it can support |
| --- | --- | --- | --- |
| Discovery | Attribution scores and labels | Candidate neurons | A ranking hypothesis |
| Structural tracing | Node and edge scores | Sparse graph | A candidate causal route |
| Confirmation | Mean/zero ablation or steering | Metric change | A causal contribution claim |
| Generalization | Held-out prompts or templates | Stable effect | A task-level claim |
| Safety audit | Sensitive prompts and collateral metrics | Side effects | A deployment or governance decision |

### Inspect the baseline before interpreting a failure

Suppose a circuit has good faithfulness under mean ablation but poor completeness under zero ablation. It is tempting to call the circuit incomplete. The more careful interpretation is that the intervention distributions differ. Setting a neuron to zero may produce a state the original model rarely visits, causing nonlinear downstream behavior that does not occur under a mean replacement.

That is why the paper reports both mean and zero ablation in the unpaired setting. The difference is not a footnote. It tells us whether the conclusion is robust to the intervention used to define “remove this component.” In a new project, we should plot the two baselines side by side and report how many examples move outside the normal activation range.

### Treat descriptions as searchable metadata

Automatic descriptions are valuable because they make a 500-neuron graph searchable. They can cluster neurons related to number, language, tense, or a named entity. But the description model is itself an interpreter with error modes. It may summarize a frequent surface pattern rather than the causal feature, or omit a subtle multilingual relationship that becomes obvious in the top-activating exemplars.

The reliable workflow is therefore:

1. Use attribution to select a small candidate set.
2. Use descriptions to find promising clusters.
3. Read positive and negative exemplars for each candidate.
4. Intervene on the cluster and evaluate the target metric.
5. Test on prompts that break the surface shortcut.

The fifth step is the one most often skipped. A neuron that appears to represent plural nouns may really respond to a particular determiner, language, or token position. Counterfactual templates and held-out variants are how we discover the difference.

## 15. The architectural reason this may scale

The paper offers a plausible architectural explanation for why MLP activations are a good basis. Transformer MLPs already use element-wise nonlinearities, and gated MLPs can express positive and negative contributions through multiplicative gates. A representation can therefore be sparse at the activation site without requiring a separately learned sparse dictionary.

Mixture-of-experts models make the architectural trend even more visible. Routing sends an input to only a subset of expert MLPs, so the model already contains a form of conditional sparsity. This does not prove that every expert neuron is interpretable. It does suggest that interpretability methods should measure the sparsity the architecture naturally provides before adding a new decomposition layer.

There is a second-order consequence for model development. If a model checkpoint changes, an SAE trained on the old representation may need to be retrained before it can be used reliably. Native neuron tracing can move with the checkpoint because the basis is part of the model itself. That advantage is strongest when the analysis is used during training or across many rapidly changing checkpoints.

The trade-off is compute. Native does not mean free. RelP avoids the repeated backward passes of IG, but edge tracing still requires many gradients and careful batching. The engineering target should not be “trace every neuron in every layer for every token.” It should be an incremental pipeline: narrow the task metric, score candidate nodes, prune edges, cache activation summaries, and reserve expensive interventions for a small set of hypotheses.

## 16. A reproducible reporting checklist

For a result that another team can audit, report more than a screenshot. The minimum useful record includes:

- the exact model name and checkpoint;
- the task dataset and train/validation split;
- whether inputs were paired, unpaired, or synthetically forced;
- the output metric and target/counterfactual tokens;
- the representation site, including activation versus output;
- the attribution method and replacement-model rules;
- the node and edge thresholds;
- the ablation baseline and intervention distribution;
- the number of retained nodes and edges;
- faithfulness and completeness on held-out examples;
- the compute cost and batching strategy;
- the steering or ablation tests used to validate semantic labels.

This list may look bureaucratic, but it prevents a common comparison error: two papers can both say “sparse circuit” while using different units, thresholds, baselines, and output metrics. Without the measurement contract, the adjective “sparse” is not portable.

It is also worth storing the raw artifacts beside the prose: the attribution arrays, threshold sweep, intervention configuration, and model commit. A future reader should be able to answer whether a plot changed because the method improved or because the candidate pool, tokenization, or ablation baseline changed. Interpretability results become much more useful when they behave like ordinary systems experiments: versioned inputs, explicit metrics, reproducible transformations, and a failure log.

The Transluce result is strongest precisely because it compares representation choices on a shared benchmark and then pushes beyond the benchmark into qualitative case studies. The quantitative SVA plots establish the sparsity/faithfulness claim. The state-capital, addition, and multilingual results establish that the selected units can be organized into recognizable task structure. The user-modeling experiment expands the scope while also exposing the ethical and methodological limits.

## When to reach for the neuron basis—and when not to

### Reach for it when

- You can define a causal output metric and an intervention baseline.
- The model exposes post-nonlinearity MLP activations at a manageable cost.
- You want to test whether a learned feature basis is necessary for the task.
- The behavior is likely to involve intermediate computation rather than only the final logits.
- You need to avoid retraining an SAE every time the base model or checkpoint changes.
- You are willing to inspect edges and ablation results, not only neuron descriptions.

### Keep a learned basis in the toolbox when

- A neuron-level graph remains too large after careful edge pruning.
- A task needs a stable, reusable feature vocabulary across many prompts or models.
- The learned representation provides a clear causal advantage on held-out evaluations.
- You need to compare concepts at a granularity the model's native coordinates do not expose.

### Do not make these shortcuts

- Do not call a neuron “the concept” because its top examples look coherent.
- Do not report sparsity without reporting faithfulness, completeness, and the ablation baseline.
- Do not infer a causal edge from two correlated node scores.
- Do not treat a synthetic forced-output user-modeling task as a complete description of normal conversation.
- Do not present RelP intuition as the exact formula or implementation when the local replacement model matters.

The durable takeaway is modest but important: the raw neuron basis deserves to be tested before we assume it is disqualified. In the Transluce experiments, the combination of MLP activations, RelP attribution, edge pruning, and intervention recovers circuits that are sparse enough to analyze and faithful enough to matter. The result does not eliminate SAEs. It changes the baseline they must beat.

## Further reading

- [Language Model Circuits Are Sparse in the Neuron Basis](https://transluce.org/neuron-circuits), the source article and visualizations for the results discussed here.
- [The Linear Representation Hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis), for the geometry of concept directions and activation interventions.
- [What Is Superposition?](/blog/machine-learning/ai-interpretability/what-is-superposition), for why multiple features can occupy a smaller native basis.
- [Transcoders Find Interpretable LLM Feature Circuits](/blog/paper-reading/ai-interpretability/transcoders-find-interpretable-llm-feature-circuits), for the learned-feature circuit perspective.

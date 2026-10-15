---
title: "Finding Transformer Circuits With Edge Pruning: Circuit Discovery as Optimization"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How Bhaskar et al. turn circuit discovery into a gradient-descent problem over edge masks, beat ACDC and EAP on GPT-2, and scale it to a 13B model."
tags: ["paper-reading", "edge-pruning", "circuit-discovery", "mechanistic-interpretability", "transformers", "acdc", "l0-regularization", "hard-concrete", "neurips-2024-spotlight"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 19
paper:
  title: "Finding Transformer Circuits With Edge Pruning"
  authors: "Adithya Bhaskar, Alexander Wettig, Dan Friedman, Danqi Chen"
  venue: "NeurIPS 2024 (spotlight)"
  url: "https://arxiv.org/abs/2406.16778"
---

> [!tldr]
> - **Claim.** Circuit discovery can be solved directly as an optimization problem: put a learnable mask on every edge between Transformer components, train the masks with gradient descent under a sparsity target, and round them to get the circuit.
> - **Why it matters.** The two standard tools each give something up. ACDC is a greedy search that gets slow fast. EAP is a one-shot linear approximation that is fast but less faithful. Edge Pruning is both reasonably fast and more faithful.
> - **Most surprising result.** On CodeLlama-13B, it finds circuits with 1,041 and 1,464 edges out of 3,872,820 (over 99.96% sparsity) that stay within 2.75 and 2 accuracy points of the full model on a Boolean logic task, and the instruction-prompted and few-shot circuits share 653 edges.
> - **Where it fails.** It needs a lot of GPU memory (32 H100s for the 13B run), it is slower than EAP on small datasets, and a circuit that matches the outputs can still misrepresent the computation.

A circuit is the interpretability equivalent of a wiring diagram: the small part of a network that actually does a given job. The tools that automate finding one force an awkward choice between slow and approximate.

This paper borrows the machinery of structured pruning and optimizes all edge decisions jointly. The trick that makes it possible: pull the residual stream apart so each edge can be masked on its own.

## What a circuit is, formally

### The computational graph

A Transformer is a stack of layers (attention and MLP) that read from and write to a shared residual stream. Layer $i$ reads the current stream $h_i$, computes its output $y_i = f_i(h_i)$, and adds it back: $h_{i+1} = h_i + y_i$. Unroll that recurrence and every layer's input is just a sum of everything written before it:

$$
y_i = f_i\Big(y_0 + \sum_{j=1}^{i-1} y_j\Big)
$$

Here $y_0$ is the input embedding and $y_j$ is the output of an earlier component $j$. Each term in that sum is an **edge** $j \to i$ of the computational graph. Following ACDC, the paper splits each attention head into four nodes (Q, K, V, O), so a head's query, key and value inputs each receive their own incoming edges.

![Figure 1 from Bhaskar et al. (2024): (a) a regular Transformer writes into one shared residual stream; (b) Edge Pruning keeps every earlier output separate and puts a learnable mask on each read; (c) rounding the masks yields a sparse circuit](/imgs/blogs/finding-transformer-circuits-with-edge-pruning-fig1.webp)

Panel (a) is the stream we just wrote down, panel (b) adds a mask (red dots) on every wire feeding a Q, K or V input, and panel (c) is what is left after rounding.

### The objective

A circuit $C$ is a subset of the edges $G$ of the full model. Circuit discovery asks for the smallest $C$ whose outputs still match the full model on a task:

$$
\arg\min_{C}\; \mathbb{E}_{(x,\tilde{x}) \in T}\Big[ D\big(p_G(y \mid x) \,\|\, p_C(y \mid x, \tilde{x})\big) \Big] \quad \text{s.t.} \quad 1 - \frac{|C|}{|G|} \ge c
$$

where $T$ is the task distribution, $x$ is a clean example, $\tilde{x}$ is a matched corrupted example, $p_G$ is the full model's output distribution, $p_C$ is the circuit's output, $D$ is a divergence (KL in practice), and $c$ is the target sparsity. The constraint says "keep at most a fraction ${1-c}$ of the edges."

The difficulty is that $C$ is a discrete choice over a huge set. Because edges grow with the square of the node count, the set is enormous: CodeLlama-13B has 3.88M edges under this graph. Every method in this space is a different way of approximating that combinatorial search.

## Why ACDC and EAP fall short

| Method | How it chooses edges | Cost | Weakness |
| --- | --- | --- | --- |
| ACDC (Conmy et al., 2023) | Greedy: ablate each edge alone, drop it if the effect is below a threshold | Steps grow linearly with edge count | Ignores interactions between edges; too slow for big models or datasets |
| EAP (Syed et al., 2023) | First-order (gradient) estimate of each edge's patching effect, keep top-k | One forward and backward pass | Linear approximation is often wrong; each score ignores which other edges are present |
| Node or head pruning | Remove whole heads or neurons | Cheap | Too coarse; Conmy et al. found these worse than ACDC |
| **Edge Pruning** | Learn a continuous mask per edge with gradient descent and a sparsity penalty, then round | Fixed number of training steps | High memory; needs tuning of the sparsity schedule |

Both ACDC and EAP score edges one at a time, ACDC exactly but sequentially, EAP all at once with a tangent-line estimate. Neither asks which **set** of edges, chosen together, best reproduces the output. Edge Pruning does, because every mask gets its gradient while every other mask is also in play.

## Technique 1: interchange ablation on a single edge

**The problem.** To score a circuit you have to run the model with some edges "missing." Setting a missing edge to zero pushes the activations far outside anything the model saw in training, so you end up measuring a broken model rather than a smaller circuit.

**The analogy.** To test whether a recipe needs its sugar, removing it entirely changes the dish in unrelated ways. Swapping in the same amount of an ingredient from a different recipe keeps the dish dish-shaped while removing what the sugar contributed.

**The mechanism.** Each clean prompt $x$ has a corrupted twin $\tilde{x}$ that changes the answer (same IOI template, different names). The full model runs on $\tilde{x}$ and caches every node's output $\tilde{y}_j$. Removing edge $j \to i$ feeds node $i$ the value $\tilde{y}_j$ in that one slot; every other reader of $y_j$ is unaffected.

![Removing an edge swaps the clean activation for the corrupted run's activation; a continuous mask interpolates between the two](/imgs/blogs/finding-transformer-circuits-with-edge-pruning-2.webp)

**The math.** To make this differentiable, the paper lets a continuous mask $z_{ji} \in [0,1]$ interpolate between the clean and corrupted activations at each edge:

$$
y_i = f_i\Big( z_{0i}\, y_0 + (1 - z_{0i})\, \tilde{y}_0 \;+ \sum_{\substack{1 \le j \lt i \\ j \text{ upstream of } i}} \big( z_{ji}\, y_j + (1 - z_{ji})\, \tilde{y}_j \big) \Big)
$$

$z_{ji} = 1$ keeps the edge (clean value flows through), $z_{ji} = 0$ removes it (corrupted value flows through), and anything in between is a blend. The full model is the case where every mask equals 1. The model weights are frozen; only the $z$ values are trained.

**Worked example.** Suppose for one example a clean head output projects to $y_j = 2.0$ along some direction, the corrupted run gives $\tilde{y}_j = -1.0$, and the current mask is $z_{ji} = 0.7$. Node $i$ reads ${0.7 \times 2.0 + 0.3 \times (-1.0) = 1.1}$. The gradient of the loss with respect to $z_{ji}$ is proportional to $y_j - \tilde{y}_j = 3.0$ times how much node $i$'s output matters, which is exactly the signal "how much does this edge's clean information help."

**When it fails.** The answer depends on the choice of corruption. A corruption that only swaps names tests name-routing edges; edges that carry template or syntax information look unimportant because both runs agree on them. For Tracr models the paper switches to zero ablation, following ACDC, which shows the choice is not neutral.

## Technique 2: the disentangled residual stream

**The problem.** Equation 3 asks each node to see its own blend of upstream outputs. A normal Transformer cannot do that. Once $y_j$ is added into $h$, the sum is shared, so every downstream node reads the same mixture, and you cannot remove $j \to i$ without also removing $j \to k$.

**The analogy.** A shared residual stream is a group chat: everything anyone posts lands in one feed, and every reader sees the whole feed. The disentangled version is a mailbox per sender. Each reader can choose, sender by sender, whose mail to open.

**The mechanism.** The paper modifies the model (following Friedman et al., 2023, and Lindner et al., 2023) so that instead of adding $y_i$ into a single tensor, it appends $y_i$ to a list $(y_0, y_1, \ldots, y_{i-1})$. At the input of each node, the list is aggregated on the fly with that node's own masks, as in equation 3.

![A regular residual stream sums all writers into one tensor; the disentangled stream keeps a list so each read has its own mask](/imgs/blogs/finding-transformer-circuits-with-edge-pruning-1.webp)

**The math.** For a model with $N$ nodes, a node at depth $i$ reads $O(i)$ separate tensors instead of one, so the number of edges grows as $O(N^2)$. That quadratic edge count is the point: it is what gives Edge Pruning finer granularity than node pruning. It is also the cost, because the activations behind those edges have to be kept for the backward pass.

**Worked example.** Take a toy model with an embedding and three components A, B, C in sequence (ignoring the Q, K, V split). The regular model has one residual tensor. The disentangled model has a list of up to four tensors; A has 1 incoming edge, B has 2, C has 3, the output has 4, for 10 edges total. Pruning node B removes all of its edges at once; Edge Pruning can keep $B \to C$ and drop $B \to \text{output}$.

**When it fails.** Memory. The 13B run needed 32 H100 GPUs with FSDP full sharding, BF16 and gradient checkpointing. The authors suggest running EAP first to shrink the candidate set, an idea they did not test.

## Technique 3: hard concrete masks

**The problem.** We want masks that end up at exactly 0 or 1, because a circuit is a yes-or-no set. But a hard step function has zero gradient almost everywhere, and a plain sigmoid never reaches exactly 0 or 1.

**The analogy.** A dimmer switch with a dead zone at each end: push it far enough and it clicks fully on or off, but in the middle it still dims, so you can feel which direction helps.

**The mechanism.** Each mask is a random variable from the hard concrete distribution of Louizos et al. (2018), as used in CoFi pruning. Sample noise, add a learned bias, squash with a sigmoid, stretch the result slightly past $[0,1]$, and clip. The clip turns the stretched tails into point masses at 0 and 1.

![The hard concrete sampling pipeline: noise and a learned log alpha are squashed, stretched past the unit interval, then clipped](/imgs/blogs/finding-transformer-circuits-with-edge-pruning-3.webp)

**The math.** Exactly as the paper writes it:

$$
u \sim \text{Uniform}(\epsilon, 1-\epsilon), \qquad s = \sigma\!\Big( \frac{1}{\beta} \cdot \log\frac{u}{1-u} + \log \alpha \Big)
$$

$$
\tilde{s} = s \times (r - l) + l, \qquad z = \min\big(1, \max(0, \tilde{s})\big)
$$

Here $\sigma$ is the sigmoid, $\epsilon = 10^{-6}$ keeps the log finite, ${1/\beta = 2/3}$ is a fixed temperature, $\log \alpha$ is the learnable parameter (one per mask), and $[l, r] = [-0.1, 1.1]$ is the stretch interval. Larger $\log \alpha$ shifts the sample toward 1. Note that the paper also uses $s$ for the current sparsity in the next equation; they are different quantities.

**Worked example.** Take $\log \alpha = 0$, so the mask is undecided.

| Noise $u$ | Logit $\tfrac{1}{\beta}\log\tfrac{u}{1-u}$ | $s = \sigma(\cdot)$ | Stretched $\tilde{s}$ | Mask $z$ |
| --- | --- | --- | --- | --- |
| 0.10 | ${1.5 \times (-2.197) = -3.30}$ | 0.036 | $-0.057$ | **0** |
| 0.80 | ${1.5 \times 1.386 = 2.08}$ | 0.889 | 0.967 | 0.967 |
| 0.95 | ${1.5 \times 2.944 = 4.42}$ | 0.988 | 1.086 | **1** |

Two of three draws land exactly on 0 or 1, and the middle one still carries a gradient back to $\log \alpha$.

**When it fails.** The final circuit is not read off the distribution directly. After training, the authors compute the average mask value, call that the desired density, and binary-search a threshold so that exactly that fraction of masks round to 1. Thresholds landed between 0.2 and 0.8. That procedure is sensible but adds a step whose effect on the reported circuits is not ablated.

## Technique 4: hitting a sparsity target with a Lagrangian

**The problem.** A plain L0 penalty with a fixed weight does not hit a specific sparsity; you get whatever the weight happens to produce.

**The analogy.** A thermostat turns the heater up the longer the room stays below target.

**The mechanism.** The paper adds a Lagrangian term on the gap between a reference value $t$ and the current sparsity $s$, and updates its multipliers by gradient **ascent**, so the penalty grows while the gap persists. The reference $t$ is ramped linearly from 0 to its final value over most of training.

**The math.**

$$
\mathcal{L}_s = \lambda_1 (t - s) + \lambda_2 (t - s)^2, \qquad \mathcal{L} = \mathcal{L}_{\mathrm{KL}} + \mathcal{L}_{\mathrm{edge},s}
$$

where $\lambda_1, \lambda_2$ are learned multipliers. Runs tend to settle slightly below $t$, so the authors set $t$ above 1 (up to 1.1 for GPT-2, 1.2 for CodeLlama) even though sparsity can never exceed 1. The linear term $\lambda_1$ deliberately lets momentum overshoot the target so sparsity does not lag behind by a constant gap.

**Worked example.** With $t = 0.9$, current $s = 0.8$, $\lambda_1 = 0.5$ and $\lambda_2 = 2$, the penalty is ${0.5 \times 0.1 + 2 \times 0.01 = 0.07}$. If sparsity stays stuck, ascent raises the multipliers and the penalty keeps growing until pruning resumes.

### Node masks: the accelerator

Edge masks alone struggled to converge to high sparsity. The fix is a second set of masks $z_n$ on nodes, so that an edge survives only if both it and its source node do:

$$
\tilde{z}_{(n_1, n_2)} = z_{(n_1, n_2)} \times z_{n_1}
$$

A node mask can only remove edges, so the edge Lagrangian still controls the target. For CodeLlama-13B the authors add a node Lagrangian too.

| Hyperparameter (from Appendix A) | GPT-2 Small | CodeLlama-13B |
| --- | --- | --- |
| Optimizer | Adam | Adam |
| Steps | 3,000 | 6,000 |
| Batch size | 32 | 1 |
| Mask learning rate | 0.8 (1 for IOI-t1) | 0.8 |
| Final edge target $t$ | up to 1.1 | 1.2 |
| Hardware and time | one A100 or H100, about 45 min | 32 H100, about 35 hours |

## Results on GPT-2 Small

The four tasks are IOI-t1 (one template), IOI (30 templates), Greater Than (GT) and Gendered Pronoun (GP), each split into train, validation and test so circuits are scored on unseen examples.

### Faithfulness

![Figure 2 from Bhaskar et al. (2024): KL divergence between the model and each circuit against edge sparsity; lower is better](/imgs/blogs/finding-transformer-circuits-with-edge-pruning-fig2.webp)

On the simple tasks (IOI-t1, GP) the three methods are close at low sparsity and Edge Pruning pulls ahead as circuits get sparser. On IOI and GT the gap is large. On IOI, ACDC's sparse circuits score worse than the 0.69 KL you would get by picking one of the two names at random, while Edge Pruning stays below it.

### Task performance

![Figure 3 from Bhaskar et al. (2024): logit difference (or probability difference for GT) of each circuit against edge sparsity; the dashed line is the full model](/imgs/blogs/finding-transformer-circuits-with-edge-pruning-fig3.webp)

The headline comparison: on IOI, Edge Pruning finds a circuit at 98.8% sparsity that is as faithful and performs as well as ACDC's at 96.8%. Keeping 1.2% of edges instead of 3.2% is the paper's "2.65× fewer edges." EAP sits far below the dashed full-model line on IOI-t1 and IOI, though on GT it holds up better than ACDC even at 99.5% sparsity.

### Scaling to 100K examples

![Table 1 from Bhaskar et al. (2024): KL and wall-clock time on IOI at 96.6% sparsity with 200, 400 and 100K training examples, one H100](/imgs/blogs/finding-transformer-circuits-with-edge-pruning-fig4.webp)

| Examples | ACDC KL / time | EAP KL / time | Edge Pruning KL / time |
| --- | --- | --- | --- |
| 200 | 0.92 / 18,783 s | 3.47 / 21 s | **0.25** / 2,756 s |
| 400 | 0.88 / 40,759 s | 3.66 / 43 s | **0.22** / 2,931 s |
| 100K | not run | 3.78 / 12,260 s | **0.20** / **3,042** s |

Edge Pruning's time barely moves because its step count is fixed (at 100K it sees about 96K unique examples once). ACDC's time doubles with the data. EAP is 130× faster at 200 examples but its KL is 14× worse, and at 100K it is the slower of the two.

### Ground truth from Tracr

Tracr compiles small programs into Transformers whose circuits are known exactly: `xproportion` (2 layers, fraction of x's so far) and `reverse` (3 layers, reverses a list).

![Figure 4 from Bhaskar et al. (2024): the ground-truth circuits for the Tracr-compiled xproportion and reverse programs, both recovered exactly](/imgs/blogs/finding-transformer-circuits-with-edge-pruning-fig5.webp)

Edge Pruning recovers both perfectly. That is a sanity check, not a stress test: two tiny models with clean, non-redundant circuits.

### Robustness and overlap with manual circuits

Across 12 random seeds at fixed targets, the resulting sparsities and KLs are tightly clustered (Appendix D). The pairwise intersection-over-union of chosen edges is "generally high (0.5–0.7), but still far from 1," which the authors read as some components being vital and others redundant.

Against hand-found circuits, node ROC AUC is 0.8256 (Edge Pruning) vs 0.8108 (ACDC) on IOI and 0.9245 vs 0.9597 on GT: no clear win, and the authors argue faithfulness is the better yardstick anyway.

## Case study: CodeLlama-13B on Boolean expressions

The task is a programmatically generated version of BBH Boolean Expressions, with prompts like "((not False) and False) or (False and True) is". The model is the instruction-tuned CodeLlama-13B, at 82.00% accuracy instruction-prompted (IP) and 89.25% with three few-shot (FS) demonstrations.

![Table 2 from Bhaskar et al. (2024): edge counts, accuracy and exact match of the full CodeLlama-13B, the instruction-prompt circuit, the few-shot circuit, and their intersection](/imgs/blogs/finding-transformer-circuits-with-edge-pruning-fig6.webp)

Three research questions, three answers:

- **RQ1, extreme sparsity.** The IP circuit keeps 1,041 of 3,872,820 edges (99.97% sparsity) and scores 79.25%. The FS circuit keeps 1,464 (99.96%) and scores 87.25%. Both stay within 2.75 points of the full model in their own setting.
- **RQ2, overlap.** The two circuits share 653 edges, 62.7% of the smaller one. The paper reports this as over 1,700× more than random chance. My own rough check: two random sets of those sizes would share about ${1041 \times 1464 / 3{,}872{,}820 \approx 0.39}$ edges, so the overlap is three orders of magnitude above chance either way. The intersection alone scores 72.50% (IP) and 68.25% (FS).
- **RQ3, cross-evaluation.** The FS circuit scores 75.75% when prompted IP, and the IP circuit scores 74.50% when prompted FS. Each works reasonably in the other's setting, but with drops of 11.5 and 4.75 points from its home setting.

The authors' reading is careful: the two prompting styles share a large common mechanism, but the gap between FS (87.25%) and the intersection (68.25%) says each also needs its own components.

Appendix F adds one manual finding: head L8.H16 attends from an operator to the token before it, and L10.H24 attends from the second operand back to that operator and reads L8.H16, suggesting the pair evaluates `a op b`. The authors conjecture a guess-and-verify scheme for nested expressions and flag it as a hypothesis.

## What is load-bearing, and what might not transfer

- **The corruption scheme defines the circuit.** Every result uses a matched corrupted twin. Tasks without a natural minimal-pair corruption (open-ended generation, say) do not have an obvious $\tilde{x}$, and the method has nothing to say there.
- **The graph granularity is fixed in advance.** Nodes are heads (split into Q, K, V, O) and MLPs. If a behavior lives in a direction inside an MLP, Edge Pruning will find the whole MLP. That is why feature-level work like [Sparse Feature Circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models) changes the nodes rather than the search.
- **Hyperparameters were hand-picked** from "a small number of pilot runs," and the `reverse` Tracr program needed a much longer schedule.
- **Compute.** About 5,000 GPU hours for the whole paper.

## Critique

**What is strong.** The framing is clean: optimize the actual objective rather than a greedy or linear proxy of it. Held-out test splits improve on prior circuit papers. Appendix C's edge-faithfulness check (removing an edge from the circuit hurts about as much as removing it from the full model, with near-linear correlation) is the right sanity test and passes.

**What is weak.** On GPT-2, Edge Pruning wins or ties on all four tasks, but the figures compare scatter points rather than matched sparsities: each method lands wherever its own knob (ACDC's threshold, EAP's top-k, Edge Pruning's target) puts it, so some comparisons are read across slightly different sparsities. That is my reading of the plots, not a point the paper makes. The CodeLlama study has no baseline at all, because ACDC and EAP were not run at 13B; we learn that Edge Pruning finds sparse, accurate circuits, not that it finds better ones than an alternative would at that scale.

**What is missing.** No ablation of the node masks' effect on final faithfulness, of the thresholding procedure, or of KL against other losses. The 12-seed IoU of 0.5 to 0.7 means "the circuit" is not unique, which sits uneasily with reading any one circuit as the mechanism. The authors also flag the interpretability illusion: output faithfulness does not prove the circuit computes the way the model does.

**What would change my mind.** If someone ran EAP or a refined attribution method (Kramár et al., 2024) with a cheap fine-tuning step on the selected edges and matched Edge Pruning's KL on IOI at a fraction of the memory, the case for full optimization would shrink to "a nice exact baseline." Conversely, if an intervention study on CodeLlama confirmed the L8.H16 and L10.H24 guess-and-verify story, the case study would turn from a scaling demo into evidence that Edge Pruning circuits point at real mechanisms.

## When to reach for Edge Pruning, and when not to

| Situation | Reach for | Why |
| --- | --- | --- |
| Small dataset, quick look, limited GPU | EAP | Seconds instead of minutes; good enough on simple tasks |
| Complex or mixed-template task, need a faithful circuit | Edge Pruning | Joint optimization handles interactions the linear proxy misses |
| Large dataset (thousands of examples or more) | Edge Pruning | Fixed step count, cost does not grow with data |
| Billion-parameter model | Edge Pruning with sharding | ACDC's per-edge loop is infeasible at millions of edges |
| Behavior lives in features, not heads | A feature-level method | Edge Pruning's nodes are whole heads and MLPs |

From the compression side, this is the L0 and hard concrete toolkit covered in [pruning LLMs and Transformers](/blog/machine-learning/edge-ai/pruning-llms-and-transformers), applied to the wires between components instead of the components themselves, and trained to preserve one task's outputs rather than general quality. For background on why heads and MLPs are imperfect units to begin with, see [language model circuits in the neuron basis](/blog/machine-learning/ai-interpretability/language-model-circuits-neuron-basis).

## References

1. Bhaskar, A., Wettig, A., Friedman, D., and Chen, D. *Finding Transformer Circuits With Edge Pruning.* NeurIPS 2024. [arXiv:2406.16778](https://arxiv.org/abs/2406.16778)
2. Code and data: [github.com/princeton-nlp/Edge-Pruning](https://github.com/princeton-nlp/Edge-Pruning)
3. Conmy, A. et al. *Towards Automated Circuit Discovery for Mechanistic Interpretability* (ACDC). NeurIPS 2023.
4. Syed, A., Rager, C., and Conmy, A. *Attribution Patching Outperforms Automated Circuit Discovery* (EAP). 2023.
5. Louizos, C., Welling, M., and Kingma, D. *Learning Sparse Neural Networks through L0 Regularization.* ICLR 2018.
6. Xia, M., Zhong, Z., and Chen, D. *Structured Pruning Learns Compact and Accurate Models* (CoFi). ACL 2022.
7. On this site: [Sparse Feature Circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models), [Pruning LLMs and Transformers](/blog/machine-learning/edge-ai/pruning-llms-and-transformers), [Language model circuits in the neuron basis](/blog/machine-learning/ai-interpretability/language-model-circuits-neuron-basis)

---
title: "The Non-Linear Representation Dilemma: Why Causal Abstraction Needs a Leash"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "Sutter et al. prove that once alignment maps may be arbitrarily non-linear, any network can be shown to implement any algorithm, then show it happening on randomly initialised MLPs and Pythia models. Here is the theorem, the experiments, and what it means for DAS-style interpretability."
tags: ["paper-reading", "causal-abstraction", "non-linear-representations", "distributed-alignment-search", "interchange-intervention", "linear-representation-hypothesis", "mechanistic-interpretability", "theoretical-foundations", "neurips-2025-spotlight"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 19
paper:
  title: "The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?"
  authors: "Denis Sutter, Julian Minder, Thomas Hofmann, Tiago Pimentel"
  venue: "NeurIPS 2025 (spotlight)"
  url: "https://arxiv.org/abs/2507.08802v2"
---

> [!tldr]
> **Claim.** Causal abstraction says a network "implements" an algorithm if you can find an alignment map that makes interventions on the network behave like interventions on the algorithm. Nothing in the definition says that map must be linear. Sutter et al. prove that if it may be arbitrarily complex, then under mild assumptions **every network implements every algorithm**, so the test says nothing.
>
> **Why it matters.** Most practical work (DAS and its descendants) quietly assumes linear maps. That assumption is not a convenience; it is the thing that gives the result meaning.
>
> **Most surprising result.** With a non-linear (RevNet) alignment map, randomly initialised Pythia models reach near-perfect interchange intervention accuracy on indirect object identification, including the 31M and 70M models that never learn the task even after training.
>
> **Where it fails.** The theorem is an existence proof built on memorisation, and the authors show the learned maps stop working when the test set uses names the training set never saw. The paper names the dilemma; it does not resolve it.

Interpretability researchers have a favourite experiment: swap a piece of a model's internal state with the piece from another input, and check whether the output changes the way a hypothesised algorithm predicts. This paper asks what happens if the "piece" can be defined so flexibly that the experiment always succeeds. The authors (ETH Zürich and EPFL) answer with a theorem, replications of the original distributed alignment search (DAS) results, and a name for the problem: the **non-linear representation dilemma**.

![Figure 1 from Sutter et al. (2025): as the alignment map grows from identity to linear to non-linear, intervention accuracy climbs toward 1.0 and the algorithm appears ever more faithfully implemented](/imgs/blogs/the-non-linear-representation-dilemma-is-causal-abstraction-enough-for-mechanistic-interpretability-fig1.webp)

Figure 1 is the whole argument: a neural classifier on the left, a two-step algorithm on the right, and alignment maps of growing power in between with illustrative accuracies of 0.8, 0.9 and 1.0. The punchline is that the last row is not evidence of anything.

## The problem: "implements an algorithm" needs a definition

To say a network implements an algorithm, we need a matching rule strict enough that a mismatch is possible. Probing hit a capacity problem here: a powerful enough probe reads off almost anything, including information the model never uses.

Causal abstraction was supposed to fix that by demanding that **intervening** on a hidden variable changes the output exactly as intervening on the algorithm's variable would. This paper shows the capacity problem did not go away. It moved into the alignment map.

## Background 1: algorithms as causal models

**Problem it solves.** We need an object we can intervene on, so the algorithm must be written as a causal model, not as a black-box function.

**Intuition.** Think of a spreadsheet. Some cells are inputs, some are formulas over other cells, and one cell is the answer. An intervention is typing a constant over one formula cell and letting everything downstream recompute.

**Mechanism.** The paper writes an algorithm $A$ as a directed acyclic graph with input nodes $\eta_x$, inner nodes $\eta_{\text{inner}}$ and output nodes $\eta_y$. Each non-input node $\eta$ holds a value $v_\eta$ computed by a local function of its parents. Running the graph in topological order gives the algorithm's output $f_A(x)$.

**Math.** An intervention forces node $\eta'$ to a constant $c$ and recomputes everything else:

$$
v_{\eta_x} = x,\qquad v_{\eta'} = c,\qquad v_\eta = f_A^{\eta}\big(v_{\text{par}_A(\eta)}\big)\ \ \forall \eta \in (\eta_{\text{inner}} \cup \eta_y)\setminus\{\eta'\},\qquad f_A\big(x, (v_{\eta'} \leftarrow c)\big) = v_{\eta_y}
$$

Here $\text{par}_A(\eta)$ is the set of parents of $\eta$ and $f_A^{\eta}$ is that node's local function. In an **interchange intervention**, $c = f_A^{:\eta'}(x')$: the value node $\eta'$ takes when the algorithm runs on a different input $x'$.

**Worked example.** The hierarchical equality task takes four vectors and outputs $y = (x_1 == x_2) == (x_3 == x_4)$. The **both equality relations** algorithm has two inner nodes, $\eta_1 = (x_1 == x_2)$ and $\eta_2 = (x_3 == x_4)$, and outputs $\eta_1 == \eta_2$. The grid below runs one interchange intervention by hand.

![One interchange intervention on hierarchical equality: the base input gives False, but swapping in eta1 from the source input flips the algorithm's answer to True](/imgs/blogs/the-non-linear-representation-dilemma-is-causal-abstraction-enough-for-mechanistic-interpretability-1.webp)

The base input $(A, B, C, C)$ gives $\eta_1 = \text{False}$, $\eta_2 = \text{True}$, output False. The source input $(D, D, E, F)$ gives $\eta_1 = \text{True}$. Overwrite the base's $\eta_1$ with the source's True and the output becomes True == True, which is True. That True is the **counterfactual label** the network will be graded against.

**When it fails.** The algorithm is the researcher's hypothesis. The paper tests three for this one task (both equalities, left equality only, and an "identity of first argument" variant that copies $x_1$), and the method can only rank hypotheses you thought to write down.

## Background 2: interventions on a network, through an alignment map

**Problem it solves.** The network has no cell labelled $\eta_1$. We need a way to say which part of a hidden vector plays that role, and neurons are often the wrong unit because they are polysemantic.

**Intuition.** Treat the hidden vector as mixed paint. An alignment map is a reversible recipe that separates it into labelled jars, lets you swap one jar, and remixes it exactly.

**Mechanism.** At layer $\ell$ the network produces a hidden vector $h$. An alignment map $\phi_\ell$ is a bijection from $\mathbb{R}^{d_\ell}$ to itself. We compute $\phi_\ell(h)$, split its coordinates into blocks (one block per algorithm node plus an unused block $\psi_\perp$), overwrite the block for $\eta$ with the same block computed from the source input, apply $\phi_\ell^{-1}$, and let the network finish its forward pass.

**Math.** Write $\psi^\phi_\eta$ for the coordinates assigned to node $\eta$ and $\Pi_\eta$ for the operation "replace the $\psi^\phi_\eta$ coordinates." The intervened hidden state is

$$
h' \;=\; \phi_\ell^{-1}\Big(\Pi_\eta\big(\phi_\ell(h_{\text{base}}),\ \phi_\ell(h_{\text{src}})\big)\Big), \qquad h,\ h' \in \mathbb{R}^{d_\ell}
$$

where $h_{\text{base}}$ and $h_{\text{src}}$ are the layer-$\ell$ activations on the base and source inputs. The size $|\psi^\phi_\eta|$ is what the paper calls the **intervention size**. The three families of $\phi$ the paper compares map onto three views of how networks store information:

| Alignment map | Form | Hypothesis it assumes | Freedom |
| --- | --- | --- | --- |
| $\phi_{\text{id}}$ | $\phi(h) = h$, greedy neuron selection | Privileged bases: one variable per set of neurons | None beyond picking neurons |
| $\phi_{\text{lin}}$ | $\phi(h) = W_{\text{orth}}\, h$, $W_{\text{orth}}$ orthogonal | Linear representation hypothesis | A rotation, about $d_\ell^2/2$ parameters |
| $\phi_{\text{nonlin}}$ | RevNet with $L_{rn}$ layers, hidden size $d_{rn}$ | Non-linear representation hypothesis | Grows with $L_{rn}$ and $d_{rn}$ |

The RevNet (Gomez et al., 2017) is the natural choice because it is invertible by construction. As background from that paper rather than this one: a RevNet block splits its input into halves $(a, b)$ and computes $a' = a + F(b)$, $b' = b + G(a')$, which you can undo exactly with $b = b' - G(a')$, $a = a' - F(b)$, no matter how complex $F$ and $G$ are.

**When it fails.** Every row of that table encodes an assumption about the network. The definition (Beckers and Halpern, 2019) only requires $\phi$ to be a layer-wise bijection; the rows are choices researchers made, not choices the theory forced.

## Background 3: distributed alignment search and IIA

**Problem it solves.** Given a hypothesised algorithm and a layer, we need to actually find a $\phi$, and we need a score.

**Intuition.** DAS turns the search into supervised learning. The algorithm generates counterfactual labels for free; we train $\phi$ until the intervened network predicts those labels.

![Distributed alignment search, one training step: base and source run through the frozen network to layer l, phi swaps the eta block in latent space, phi inverse maps back, and the loss compares the prediction with the algorithm's counterfactual label](/imgs/blogs/the-non-linear-representation-dilemma-is-causal-abstraction-enough-for-mechanistic-interpretability-2.webp)

**Mechanism.** Sample a base input $x_\emptyset$, pick which nodes to intervene on, and sample a source input for each. Run the algorithm with the interchange intervention to get the label $y$. Run the network with the matching distributed intervention $I_N$ to get a prediction. The network's weights stay frozen; only $\phi$ is trained.

**Math.** With $N$ such triples in a dataset $\mathcal{D}$, DAS minimises cross-entropy:

$$
\mathcal{L}(\phi) \;=\; -\sum_{(x_\emptyset^{(n)},\, I_N^{(n)},\, y^{(n)}) \in \mathcal{D}} \log p^{\phi}_N\big(y^{(n)} \mid x_\emptyset^{(n)}, I_N^{(n)}\big)
$$

where $p^{\phi}_N(y \mid x_\emptyset, I_N)$ is the network's output distribution when run on $x_\emptyset$ with the intervention applied through $\phi$. The score on a held-out set is the **interchange intervention accuracy**:

$$
\text{IIA} \;=\; \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{n} \mathbb{1}\Big[\arg\max_{y'} p^{\phi}_N\big(y' \mid x_\emptyset^{(n)}, I_N^{(n)}\big) = y^{(n)}\Big]
$$

**Worked example.** Take the intervention from the grid above: the algorithm says True. If the network also outputs True after the $\eta_1$ block swap, that pair scores 1. Get 9,700 of 10,000 pairs right and IIA is 0.97, which would usually be read as strong evidence for the algorithm.

**When it fails.** $\phi$ is trained to make the network output a specific label on a specific input, which is a classifier's objective. If $\phi$ is expressive enough, it can produce the label itself and the network becomes a passive conduit. That is the seam the paper pries open.

## Technique 1: the theorem that unbounded abstractions are vacuous

**Problem it solves.** We want to know whether the worry above is a practical nuisance or a fundamental flaw. A theorem settles it for the idealised case.

**Intuition.** Picture a library where every book (every possible input) gets a unique shelf position (its hidden state). The shelves are a continuum; the books are countable. There are infinitely more empty slots than books. A librarian with unlimited freedom can always reshelve things so that "go to the slot for book $X$ with chapter 3 swapped" lands on an empty slot she has secretly stocked with whatever ending the algorithm requires.

**The assumptions.** The theorem needs five things (formal versions are in the paper's Appendix F):

| # | Assumption | Authors' defence |
| --- | --- | --- |
| 1 | Countable input space | Holds for token strings and quantised pixels |
| 2 | Every layer is input-injective | Almost surely true for transformers at init (App. G) |
| 3 | Strict output-surjectivity from every layer | Can fail (softmax bottleneck) but mostly holds |
| 4 | Enough neurons to host the nodes in order, one spare per layer | Easy for large networks |
| 5 | The network solves the task | Otherwise IIA cannot be perfect |

**The statement.** Given these, the paper proves:

$$
\textbf{Theorem 1.}\quad \forall\, A,\ \forall\, N \text{ satisfying Assumptions 1 to 5}:\quad A \text{ is an input-restricted distributed abstraction of } N.
$$

"Input-restricted" means the interventions only use values that real inputs can produce, which is exactly what DAS does in practice. So the theorem applies to the notion practitioners use, not to some exotic variant.

![Theorem 1 intuition: reachable hidden states form a countable scatter inside an uncountable space, so the constructed phi can send each intervened state to an unused state whose downstream output is whatever the algorithm demands](/imgs/blogs/the-non-linear-representation-dilemma-is-causal-abstraction-enough-for-mechanistic-interpretability-4.webp)

**Mechanism (my paraphrase of the Appendix F proof).** The proof builds $\phi$ layer by layer by induction. Because inputs are countable and layers are injective, the set of hidden states real inputs can reach is countable, while $\mathbb{R}^{d}$ is uncountable. That leaves an uncountable reserve of "unused representations." When an intervention would produce some new combination of node values, the construction picks a fresh unused state (the paper's pseudo-code calls this `extract_unused_rep`) and, using strict output-surjectivity, makes sure the network's downstream layers turn it into the output the algorithm demands. One coordinate per node is enough, which is why Assumption 4 only asks for a single neuron per node plus a spare.

**Worked micro-example (my own toy).** Say a layer's reachable states are $\{0.1, 0.2, 0.3, 0.4\}$, one per input, and the algorithm wants an intervention on input 1 to yield input 4's output. The constructed $\phi$ places that intervened combination at ${0.25}$, which no real input reaches, so the downstream layers can map it to input 4's output without disturbing anything else. There is always another unused point for the next combination.

**Why it works, and why that is the problem.** The map never discovers structure. It memorises a lookup table and hides the entries in the gaps between real activations, which the authors call "a form of extreme overfitting." The theorem does not say DAS will find such a map from finite data; it says a perfect score cannot, by itself, rule one out.

**When it does not apply.** Restrict $\phi$ to a small family and the argument collapses: a rotation cannot carve out arbitrary pockets. That is the dilemma in its sharpest form. Linear maps give results with meaning but may miss non-linearly encoded variables; arbitrary maps catch everything, including variables that are not there.

![Linear versus non-linear alignment maps: a rotation can only isolate a variable that already lies along a direction, while a flexible bijection can bend space until any labelling becomes a single coordinate](/imgs/blogs/the-non-linear-representation-dilemma-is-causal-abstraction-enough-for-mechanistic-interpretability-3.webp)

## Technique 2: replicating DAS with a dial on map complexity

**Problem it solves.** A theorem about idealised maps could be irrelevant if gradient descent on finite data never finds them. The experiments check.

**Setup.** The hierarchical equality task, borrowed from Geiger et al. (2024b), uses inputs $x = x_1 \circ x_2 \circ x_3 \circ x_4$ with each $x_i \in [-0.5, 0.5]^4$, so 16 dimensions in total. The network is a 3-layer MLP with hidden size 16, trained with Adam at learning rate 0.001 on 1,048,576 samples. The alignment maps are trained on 1,280,000 interventions and tested on 10,000 held-out ones.

The sweep covers three maps, three candidate algorithms, layers 1 to 3, intervention sizes 1, 2 and 8, and network checkpoints from random initialisation to fully trained.

![Figure 2 from Sutter et al. (2025): IIA per layer and intervention size for three algorithms; linear maps drop sharply at layer 3 for the two equality algorithms while the non-linear map stays near ceiling everywhere](/imgs/blogs/the-non-linear-representation-dilemma-is-causal-abstraction-enough-for-mechanistic-interpretability-fig2.webp)

**Result 1: layer-wise "findings" vanish.** With $\phi_{\text{lin}}$ the paper reproduces Geiger et al.'s pattern: IIA for both equality relations and left equality relation falls substantially at layer 3, which earlier work read as the variables becoming hard to manipulate deeper in the network. With $\phi_{\text{nonlin}}$ ($L_{rn} = 10$, $d_{rn} = 16$) the drop disappears and IIA is near-optimal at every layer. The bars show the max over 10 seeds; black lines are means with 95% confidence intervals.

So the conclusion "the variable lives in layer 1, not layer 3" was a statement about linear accessibility, not about where the variable is. That is a useful finding in itself, but it is a different finding from the one usually reported.

**Result 2: the outlier that proves the rule.** The identity of first argument algorithm sits near 50% IIA for every map, which looks like a counterexample to Theorem 1. Appendix H explains it: on 1,280,000 training samples the RevNet never exceeds 0.55 train IIA, but on 1,000 samples it exceeds 0.99, and no two of 1,280,000 hidden states collided across 10 seeds. The RevNet simply lacks the capacity to memorise the table the proof describes.

## Technique 3: aligning algorithms to networks that cannot do the task

**Problem it solves.** The sharpest test of a method is a negative control. If a randomly initialised network "implements" the algorithm, the method has failed.

![Figure 3 from Sutter et al. (2025): left, mean IIA at layer 1 rises with RevNet hidden size and intervention size; right, the most complex maps reach high IIA even on the untrained network, while linear maps track training progress](/imgs/blogs/the-non-linear-representation-dilemma-is-causal-abstraction-enough-for-mechanistic-interpretability-fig3.webp)

**Result 3: random MLPs pass.** Figure 3 (left) shows IIA climbing with both the RevNet hidden size $d_{rn}$ and the intervention size. Figure 3 (right) tracks the network through training. Even at initialisation, the most complex map ($L_{rn} = 10$, $d_{rn} = 27$) achieves **over 80% IIA** on the both equality relations algorithm. As training proceeds, progressively simpler maps catch up.

Read the right panel as a family of curves, one per map complexity, plotted against the network's own accuracy. Wherever a map's IIA sits above what the network can actually do, that surplus is accuracy the map produced on its own. The simpler the map, the more closely its curve follows training progress.

**Result 4: Pythia on indirect object identification.** The IOI task gives a sentence like "Friends Juana and Kristi found a mango at the bar. Kristi gave it to" and asks for the first token of the indirect object, "Juana." The ABAB-ABBA algorithm has one inner node: whether the name order is ABAB or ABBA. Models are Pythia 31M, 70M, 160M and 410M, intervened at the middle layer, with $\phi_{\text{nonlin}}$ hidden size $d_{rn} = 64$ and intervention size set to half the model's hidden dimension.

![Figure 4 from Sutter et al. (2025): left, IIA across Pythia sizes at initialisation and after full training, with non-linear maps near 1.0 everywhere; right, IIA across Pythia-410m training checkpoints for linear and 1 to 8 layer RevNet maps](/imgs/blogs/the-non-linear-representation-dilemma-is-causal-abstraction-enough-for-mechanistic-interpretability-fig4.webp)

| Finding (Figure 4) | Number |
| --- | --- |
| Models that learn IOI after full training | 160M and 410M only |
| Models alignable with $\phi_{\text{nonlin}}$ | All four, including 31M and 70M |
| Random-init alignment with $\phi_{\text{nonlin}}$ | Near-perfect across all sizes |
| Pythia-410m starts solving IOI | Around training step 3000 |
| 8-layer RevNet on Pythia-410m | Near-perfect IIA at every checkpoint, including step 0 |
| Linear map | Approaches perfect only in the fully trained model |

A 4-layer RevNet is also near-perfect except for a dip at step 1000, and 1- and 2-layer maps reach perfect alignment later in training. The authors suggest the dip comes from the model over-fitting to unigram statistics at that point, so its hidden states carry little context. My reading, not the authors': the one time a powerful map stumbled, the network had temporarily stopped using context, which fits the theorem's reliance on hidden states that still carry the input.

A reproduction note: the alignment maps showed "quite severe grokking." On randomly initialised Pythia-31m the loss plateaued, then dropped suddenly after about 6k steps, and the small models needed float64 precision to converge.

## Technique 4: generalisation as the missing constraint

**Problem it solves.** If memorisation is what makes Theorem 1 work, then a test that punishes memorisation should separate real structure from manufactured structure.

**Intuition.** A student who memorised the answer key aces the practice exam and fails the real one. Change the names on the exam and see who still passes.

**Mechanism.** The main IOI split ensures no sentence appears in both train and test, but the same names can recur. Appendix I.2.3 reruns Pythia-410m at layer 12 with completely disjoint name sets for training and evaluation.

![Figure 12 from Sutter et al. (2025): with disjoint name sets, the randomly initialised Pythia-410m can no longer be aligned even with 8 or 16 layer RevNets, while the fully trained model still aligns perfectly with the non-linear map and reasonably well with the linear one](/imgs/blogs/the-non-linear-representation-dilemma-is-causal-abstraction-enough-for-mechanistic-interpretability-fig12.webp)

**Result 5.** With disjoint names, alignment on the **randomly initialised** model fails even with 8- and 16-layer RevNets. On the **fully trained** model, $\phi_{\text{nonlin}}$ still achieves perfect alignment and $\phi_{\text{lin}}$ achieves reasonably high alignment. That is the pattern you would want from a trustworthy method: the trained model, which really solves IOI, passes; the random one, which does not, fails.

**My extrapolation.** This suggests a cheap protocol the paper does not formalise: evaluate alignment maps on a split that changes the entities, not just the sentences, and always run a random-init control. One experiment does not prove it suffices, but it caught the failure here.

## The dilemma, stated plainly

Put the pieces together and the shape of the problem is clear:

| Choice of $\phi$ | What you gain | What you lose |
| --- | --- | --- |
| Identity | Results tied to neurons, easy to interpret | Misses anything in superposition |
| Linear | A positive result is hard to fake; matches the linear representation hypothesis | Misses variables that are genuinely non-linearly encoded, and there is growing evidence some are |
| Non-linear, unbounded | Can find any encoding | Finds encodings that are not there; Theorem 1 says a perfect score is guaranteed |
| Non-linear, bounded | Somewhere in between | No principled way to pick the bound |

The last row is the dilemma, and the authors note it is the old probing debate in new clothes: control tasks, complexity-accuracy Pareto curves, MDL probes and unsupervised probes were all proposed, and none ended it. Their practical advice is case by case. If you have reason to believe the feature is linearly encoded, constrain $\phi$ to be linear; if not, examine the accuracy-complexity trade-off explicitly instead of reporting a single IIA number.

## What is load-bearing in the setup

The paper also runs a distributive law task in Appendix I.3 (a 3-layer MLP of width 24) and reports the same qualitative pattern. Three details carry the results and might not transfer. The toy MLP has 16 hidden units, so a RevNet with $d_{rn} = 27$ has more capacity than the layer it is aligning. The IOI result relies on the training and test sets sharing names, as Appendix I.2.3 shows. And across five Pythia seeds, two misbehave: seed 3 learns IOI but cannot be aligned even with complex maps, seed 4 never learns IOI. The authors attribute the alignment failure to limited hyperparameter tuning, which is plausible but untested.

## Critique

**What is strong.** The theorem is clean and each assumption is defended, with Appendix G proving injectivity almost surely holds for transformers at initialisation. The experiments replicate the canonical DAS setup with one dial turned, the random-init runs are the right negative control, and the disjoint-names experiment shows the authors went looking for where their own claim breaks.

**What is weak.**

- **The theorem is about existence, and the experiments are about learnability.** The paper is candid about this, but the two halves are connected by analogy, not by a result. The Limitations section even notes the learned maps do generalise to unseen test data in the main experiments, and that we do not know why.
- **No complexity measure is proposed.** "Complexity" is just $L_{rn}$ and $d_{rn}$, never compared with layer width or training-set size, which is what would tell us when a map can memorise.
- **The discussion's own temptation goes unresolved.** The authors note that $\phi_{\text{lin}}$ tracks the network's real accuracy more closely in Figure 4 (right) and say they are inclined to read that as support for the linear representation hypothesis, then admit they "cannot make this intuition formal." Linear maps also sometimes exceed the network's own accuracy, so even they "learn the task" a little.
- **Missing ablation.** A sweep that varies training-set size for a fixed map, measuring train IIA minus test IIA, would directly quantify memorisation and is cheap on the MLP.

**What would change my mind.** If someone showed a bounded but genuinely non-linear map family (say a fixed-capacity RevNet sized relative to the layer) that passes on trained models and fails on random-init controls *and* on disjoint-entity splits across several tasks, the dilemma would become a calibration problem rather than a foundational one. Conversely, if linear DAS were shown to reach high IIA on random-init models with entity-disjoint splits, I would stop treating the linear constraint as a safe harbour.

For background, see our explainer on [the linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis), [HyperDAS](/blog/paper-reading/ai-interpretability/hyperdas-towards-automating-mechanistic-interpretability-with-hypernetworks) (a DAS descendant that must stop its own intervention model from "cheating"), and the companion critique asking [whether mechanistic interpretability is identifiable](/blog/paper-reading/ai-interpretability/everything-everywhere-all-at-once-is-mechanistic-interpretability-identifiable).

## References

1. Denis Sutter, Julian Minder, Thomas Hofmann, Tiago Pimentel. *The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?* NeurIPS 2025. [arXiv:2507.08802v2](https://arxiv.org/abs/2507.08802v2)
2. Code: [github.com/densutter/non-linear-representation-dilemma](https://github.com/densutter/non-linear-representation-dilemma)
3. Geiger et al. (2024b). *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations* (the DAS paper whose experiments are replicated here).
4. Beckers and Halpern (2019), AAAI. The formal definitions of mapping one causal model onto another that this paper builds on.
5. Gomez et al. (2017). *The Reversible Residual Network* (the RevNet used for $\phi_{\text{nonlin}}$).

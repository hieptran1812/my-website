---
title: "Emergence of a High-Dimensional Abstraction Phase in Language Transformers: Where an LM First Understands Its Input"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How measuring the intrinsic dimension of hidden states layer by layer exposes a central high-dimensional phase in five 7B language models, and why that phase is where syntax and semantics are first extracted, where different models agree, and where transfer learning starts to work."
tags: ["paper-reading", "intrinsic-dimension", "gride", "information-imbalance", "representation-geometry", "probing", "layer-analysis", "linguistic-probing", "iclr-2025-poster", "iclr-2025"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 19
paper:
  title: "Emergence of a High-Dimensional Abstraction Phase in Language Transformers"
  authors: "Emily Cheng, Diego Doimo, Corentin Kervadec, Iuri Macocco, Jade Yu, Alessandro Laio, Marco Baroni"
  venue: "ICLR 2025"
  url: "https://arxiv.org/abs/2405.15471v4"
---

> [!tldr]
> **The claim.** If you measure how many "real" directions the hidden states of a language model occupy at each layer (their intrinsic dimension, or ID), five different 7B-class models all show the same shape: a clear peak in ID in the early-to-middle layers. The authors argue this peak is a distinct *phase*, the place where the model builds its first full abstract analysis of the input.
> **Why it matters.** The peak is cheap to find (no labels, no gradients, just geometry), and it lines up with things you care about: syntactic and semantic probes saturate inside it, sentiment and toxicity transfer converges inside it, and the peak layers of different models encode nearly the same neighborhood structure.
> **Most surprising result.** Models whose peak starts earlier predict text better: across (model, corpus) pairs, surprisal correlates with relative peak onset at Spearman $\rho = 0.65$ ($p = 0.01$). Shuffled text and untrained checkpoints show no peak at all.
> **Where it fails.** Everything is correlational. There are no layer ablations, the plateau scale for the ID estimator is chosen by eye, and the probing evidence relies on five older sentence-level tasks. The "phase" boundary is a convention the authors define, not something the data hands you.

Most interpretability work zooms in on one circuit or feature. This paper zooms out. It asks a blunt question about the whole residual stream: as a sentence flows through 32 layers, how *complicated* is the cloud of representations at each layer, and does that complexity tell us what the layer is doing?

The answer is a surprisingly clean story. Representations start simple, expand into a high-dimensional phase where the model analyzes the input abstractly, and then contract again as the model prepares to emit a token. Below: the two measurement tools, the six findings, and what I think is and is not established.

## The setup in one table

The authors keep the experimental design deliberately uniform, so that differences between curves reflect the models and not the pipeline.

| Ingredient | Choice in the paper |
| --- | --- |
| Models | OPT-6.7B, Llama-3-8B, Pythia-6.9B, OLMo-7B, Mistral-7B (base) |
| Shared shape | 32 layers, hidden size 4096, pre-norm decoder-only |
| Representation | Last-token hidden state (HuggingFace `hidden_states`) at every layer |
| Corpora | Bookcorpus, the Pile (10k-document subsample), WikiText-103 |
| Inputs | 50k distinct 20-token sequences per corpus, split into 5 partitions of 10k |
| Control | 5 partitions of 10k sequences from token-shuffled corpora |
| Geometry tools | GRIDE intrinsic dimension; Information Imbalance $\Delta$ |
| Function tools | MLP probes on 5 Conneau et al. tasks; linear probes for sentiment and toxicity |

Why the last token? In a causal model only the final position has attended to the whole sequence, and it is the one position that gets decoded into the next-token prediction. The authors also replicated the Pile runs on Pythia and OPT with 128-token sequences and report very similar results. Code and data are public in [the authors' repository](https://github.com/chengemily1/id-llm-abstraction).

## Tool 1: intrinsic dimension, the number of directions that matter

**The problem.** A hidden state has 4096 coordinates, but that number says nothing about how much freedom the data actually uses. If 10k sentence representations lie on a thin curved sheet, the sheet's dimension is the honest measure, and linear PCA overcounts it because curvature needs many linear directions.

**The analogy.** Think of a long garden hose coiled up in a room. The room is three-dimensional and the coil fills a lot of it, but an ant walking on the hose only ever needs one number to say where it is. Intrinsic dimension is the ant's count, not the room's.

**The mechanism.** You cannot see the sheet directly, but you can feel its dimension through distances. In a space of dimension $d$, the volume of a ball grows like $r^d$. So in high dimension a small increase in radius sweeps in a lot of new volume, and your second neighbor sits barely farther away than your first. The ratio of neighbor distances is a dimension thermometer.

![Left: illustrative points lying on a curved ribbon inside a large ambient space. Right: around one point i, the k-th and 2k-th neighbor radii, whose ratio mu shrinks toward 1 as the intrinsic dimension grows.](/imgs/blogs/emergence-of-a-high-dimensional-abstraction-phase-in-language-transformers-1.webp)

**The math, starting from the simplest case.** The TwoNN estimator (Facco et al., 2017) uses only the first two neighbors. For each point $i$, let $r_{i,1}$ and $r_{i,2}$ be the Euclidean distances to its first and second nearest neighbors, and let $\mu_i = r_{i,2} / r_{i,1} \geq 1$. If density is roughly uniform up to the second neighbor, $\mu_i$ follows a Pareto law that depends only on $d$:

$$
f(\mu) = d\, \mu^{-(d+1)}, \qquad \mu \geq 1 .
$$

Maximizing the log-likelihood $N \log d - (d+1) \sum_i \log \mu_i$ over $d$ gives a closed form, where $N$ is the number of points:

$$
\hat d_{\text{TwoNN}} = \frac{N}{\sum_{i=1}^{N} \log \mu_i } .
$$

**GRIDE: the same idea at a chosen scale.** TwoNN looks only at the tiniest neighborhoods, where measurement noise dominates. GRIDE (Denti et al., 2022), which the paper uses, generalizes the ratio to $\mu_{i,2k,k} = r_{i,2k} / r_{i,k}$, the distance to the ${2k}$-th neighbor over the distance to the $k$-th. Under local uniformity up to the ${2k}$-th neighbor, this ratio follows a generalized Pareto density:

$$
f_{\mu_{i,2k,k}}(\mu) = \frac{d\,\left(\mu^{d} - 1\right)^{k-1}}{B(k,k)\;\mu^{\,d(2k-1)+1}} ,
$$

where $B(\cdot,\cdot)$ is the Beta function and $d$ is the intrinsic dimension. Set $k = 1$ and $B(1,1) = 1$, and this collapses to the TwoNN density above, which is a good sanity check. The estimate $\hat d$ comes from numerically maximizing the product of these densities over all points, treating the ratios as independent.

**A tiny worked example.** For $d = 1$, the median of the TwoNN ratio is ${2^{1/d}} = 2$: on a line, your second neighbor is typically twice as far as your first. For $d = 20$, the median is ${2^{1/20}} \approx 1.035$. Now take five measured ratios $\mu = (1.05, 1.10, 1.02, 1.08, 1.03)$. Their logs sum to about ${0.270}$, so $\hat d = 5 / 0.270 \approx 18.5$. Ratios hugging 1 mean high dimension.

**Choosing the scale $k$.** Every real dataset has an ID that depends on the zoom level. At tiny scales you measure noise, which looks high-dimensional; at large scales curvature and density changes bend the estimate. The authors sweep $k$ for each (model, corpus, layer), then pick by visual inspection a $k$ where the estimate plateaus. In practice the chosen scale sits around $k = 32$, and their appendix shows that halving or doubling it ($k = 16$ or $k = 64$) changes the curves only weakly.

**When it fails.** GRIDE assumes locally uniform density, which is never exactly true, and the plateau choice is a judgment call. ID is also a property of the *point cloud*, so the same layer can look lower-dimensional on different inputs, which is exactly the control the paper exploits.

## Tool 2: Information Imbalance, asking whether one layer predicts another

**The problem.** Knowing that layer 10 is high-dimensional does not tell you whether it carries the same information as layer 3 or layer 31. You want a similarity measure that tolerates different scales and dimensions, and that is *directional*: "layer 12 contains what layer 4 knows" differs from the reverse.

**The analogy.** Imagine two seating charts for the same 100 guests, one arranged by profession and one by hometown. Pick each guest's closest neighbor in the profession chart, then look up how close those two people sit in the hometown chart. If they usually sit together there too, the first chart "predicts" the second.

**The mechanism.** For each point $i$, find its nearest neighbor $j$ in space $A$. Then compute the *rank* of $j$ among $i$'s neighbors in space $B$ (rank 1 means $j$ is also $i$'s nearest neighbor in $B$). Average and normalize those ranks.

**The math.** Following Glielmo et al. (2022), with $r^{A}_{ij}$ the rank of $j$ among the neighbors of $i$ in space $A$ (rank 1 is the nearest):

$$
\Delta(A \to B) = \frac{2}{N^2} \sum_{i,j \,:\, r^{A}_{ij} = 1} r^{B}_{ij} .
$$

The sum has exactly $N$ terms, one per point, so $\Delta(A \to B)$ equals ${2/N}$ times the mean $B$-rank of the $A$-neighbors. If $B$ is unrelated to $A$, that rank is uniform over ${1, \dots, N-1}$ with mean about $N/2$, giving $\Delta \approx 1$. If $A$'s nearest neighbors are also $B$'s nearest neighbors, every rank is 1 and $\Delta = 2/N$, which is essentially 0 at $N = 10{,}000$.

![Information Imbalance on five points: the same nearest-neighbor pairs in space A, scored against a space B that keeps the neighborhoods and one that scrambles them, with the resulting Delta values placed on a 0 to 1 number line.](/imgs/blogs/emergence-of-a-high-dimensional-abstraction-phase-in-language-transformers-2.webp)

**Worked example.** Take five points where, in space $A$, the nearest neighbors are $p_1 \to p_2$, $p_2 \to p_1$, $p_3 \to p_4$, $p_4 \to p_3$, $p_5 \to p_4$. In a space $B$ that keeps these neighborhoods, the ranks of those partners are $(1, 1, 2, 1, 1)$, summing to 6, so $\Delta(A \to B) = 2 \cdot 6 / 25 = 0.48$. With only five points the floor is ${2/N = 0.40}$, so this is near-perfect. In a scrambled $B$ the ranks become $(3, 2, 3, 2, 2)$, summing to 12, and $\Delta = 24/25 = 0.96$: almost no information.

**Why the paper prefers it.** $\Delta$ is rank-based, so it is indifferent to scale and dimension mismatches between layers or models. It is also asymmetric, $\Delta(A \to B) \neq \Delta(B \to A)$, which lets the authors talk about one model's layer *containing* another's. They tried three alternatives (neighborhood overlap, RSA, and linear CKA); only CKA gave interpretable results, and those broadly agree with $\Delta$.

**When it fails.** $\Delta$ only looks at first neighbors in $A$, so it is blind to global structure, and it assumes Euclidean distance is the geometry that matters.

## Finding 1: a central high-ID phase appears in every model

With both tools in hand, the first result is simply the ID curve per layer, averaged over 15 partitions (5 per corpus).

![Figure 1 from Cheng et al. (2025): layer-wise intrinsic dimension for five LMs on natural corpora (left), on token-shuffled corpora (center), and for Pythia checkpoints over training (right).](/imgs/blogs/emergence-of-a-high-dimensional-abstraction-phase-in-language-transformers-fig1.webp)

Three things jump out of the left panel.

- **ID is tiny.** Every model's ID is on the order of 10, against an ambient dimension of 4096. That matches earlier work on representation manifolds and is not new.
- **There is a peak.** Each curve climbs to a maximum early-to-mid network, roughly from layer 6 or 7 to about layer 20. OPT is the exception, with its peak around layers 17 to 23.
- **There is a second, smaller bump.** Late layers show a second rise; only for OLMo is it as tall as the first. The paper does not analyze it further.

The center panel is the control that makes the peak meaningful. Feed the same models token-shuffled text (same unigram statistics, no syntax or coherence) and ID stays low through every layer. The right panel shows Pythia checkpoints at steps 512, 4000, 16000, 64000, and 143000: the ID grows and becomes more peaked over training, so the profile is learned, not architectural.

**How the peak is delimited.** The authors adopt a convention. The *end* of the peak is the closest inflection point after the maximum. The *beginning* is set by the end's value. As an illustration with made-up numbers, if layers 5 to 14 have ID $(20, 26, 33, 38, 40, 39, 35, 30, 27, 26)$, the maximum is at layer 9 and the second differences turn positive at layer 12 (ID 30), so the peak ends there. Reading the paper's rule as "where the rising side first reaches that end value", the peak begins at layer 7 (ID 33), giving a window of layers 7 to 12. That reading of the start rule is mine; the paper's one-sentence definition is terse.

Pythia-2.8B and Pythia-12B (hidden sizes 2560 and 5120) show the same profile, with peak height growing much less than proportionally to hidden size.

## Finding 2: the peak is where layers stop resembling the input

Now the Information Imbalance comes in. For each layer $l_i$, the authors compute $\Delta(l_i \to l_{\text{first}})$, how well layer $i$'s neighborhoods predict the input layer's, and $\Delta(l_i \to l_{\text{last}})$.

![Figure 2 from Cheng et al. (2025): ID (colored) overlaid with Information Imbalance to the first layer (gray) and to the last layer (brown), for Llama, OPT and Pythia.](/imgs/blogs/emergence-of-a-high-dimensional-abstraction-phase-in-language-transformers-fig2.webp)

The gray curve, $\Delta(l_i \to l_{\text{first}})$, peaks around the ID peak in all three models. A high value means that two sequences which are nearest neighbors at the peak were far apart at the input layer. In plain terms, the peak layers have reorganized the space: sentences that start out looking alike (similar tokens) are no longer grouped together, which suggests they are grouped by something more abstract.

The brown curve toward the last layer shows no clear relationship with the first peak, and the authors leave it alone.

A second geometric signal is the **forward $\Delta$ scope**: for each layer $l_n$, count how many following layers $l_{n+k}$ satisfy $\Delta(l_n \to l_{n+k}) \leq 0.1$. A long scope means the layer already contains what the next several layers hold. At the ID peak, the scope dips. Peak layers share information with each other, but that information differs from what later layers carry.

## Finding 3: at the peak, different models share a representation space

If the peak is where models build an abstract description of the input, and the input is the same, then different models' peak layers might agree. The authors compute cross-model $\Delta$ for every pair of layers.

![Figure 4 from Cheng et al. (2025): cross-model Information Imbalance between layers of Llama, OPT and Pythia; orange bands mark ID-peak layers, symbols mark Delta levels below 0.1 in each direction.](/imgs/blogs/emergence-of-a-high-dimensional-abstraction-phase-in-language-transformers-fig4.webp)

Only low values are drawn ($\Delta \leq 0.1$; everything above is omitted). The intersections of the two models' orange bands always contain a concentration of low-$\Delta$ points, so peak layers of different models have close neighborhood structure. Neither the first layers nor the last layers show this agreement, which the authors highlight as remarkable: the shared structure lives in the middle.

Two nuances. Low $\Delta$ values also appear outside the band intersections, so high ID may be sufficient for cross-model similarity rather than necessary (the authors' own hedge). And the asymmetry tracks ID height: Pythia and OPT, with the highest peaks, directionally contain the other models' representations, while between the two of them $\Delta$ is small in both directions.

This connects to work on shared intermediate representations across languages and modalities; see our analysis of [the semantic hub hypothesis](/blog/paper-reading/ai-interpretability/the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities), which finds a similar "middle layers are the shared space" pattern from a different angle.

## Finding 4: probes say the peak is where syntax and semantics arrive

Geometry says the peak layers are different. Probing says *how* they are different. The authors train a small MLP probe (one hidden layer of 200 units, logistic activation, L2 coefficient 0.0001, 5 seeds) on each layer for five tasks from Conneau et al. (2018).

| Task | Type | What the probe must detect |
| --- | --- | --- |
| Sentence Length | Surface | Input length in tokens, binned into 5 intervals |
| Word Content | Surface | Which of 1k preset words occurs in the sentence |
| Bigram Shift | Syntax | Whether two adjacent tokens were swapped |
| Coordination Inversion | Semantics | Whether two coordinated clauses were reordered |
| Odd Man Out | Semantics | Whether a noun or verb was replaced by a random one |

Five other Conneau tasks were dropped for ceiling effects or unreliable parses.

![Figure 5 from Cheng et al. (2025): probe accuracy overlaid on the ID profile. Row (a): surface-form probes fall through the peak. Row (b): syntactic and semantic probes reach their asymptote inside the peak.](/imgs/blogs/emergence-of-a-high-dimensional-abstraction-phase-in-language-transformers-fig5.webp)

**Surface information leaves.** Sentence Length accuracy falls as you climb the layers, and the sharp decrease starts at the onset of the ID peak or just before. OPT is the telling case: its peak is late, and its Sentence Length accuracy stays stable until that late onset, then drops. Word Content falls during ID expansion, then partially recovers after the peak as the model re-encodes lexical detail for its prediction.

**Abstract information arrives.** For Bigram Shift, Coordination Inversion and Odd Man Out, accuracy reaches its asymptote during the peak and then stays roughly flat. For OPT, again, the asymptote arrives later, in step with its later peak. The appendix confirms that probes trained on shuffled-text representations sit at chance, so the signal is in the model, not in the probe.

![Schematic summary: an illustrative ID curve over 32 layers with the high-ID abstraction band shaded, a flat shuffled-text baseline, and where the probe and transfer events land relative to the peak.](/imgs/blogs/emergence-of-a-high-dimensional-abstraction-phase-in-language-transformers-3.webp)

The schematic compresses Findings 1, 4 and 6 into one picture; its curve shape is mine, not data.

## Finding 5: earlier peaks, better language models

If the peak is where the model first understands its input, then the model has more layers left to exploit that understanding when the peak comes early. The authors test this with 15 points, one per (model, corpus) pair, and Spearman rank correlations against average surprisal (in bits per token).

![Figure 6 from Cheng et al. (2025): surprisal against maximum ID across layers (left) and against relative ID-peak onset (right), one point per model and corpus.](/imgs/blogs/emergence-of-a-high-dimensional-abstraction-phase-in-language-transformers-fig6.webp)

Spearman's $\rho$ compares the *rankings* of the two quantities, from $-1$ (reversed) to $+1$ (identical order), so surprisal and perplexity give identical results.

| Relationship | Spearman $\rho$ | $p$-value | Reading |
| --- | --- | --- | --- |
| Surprisal vs. max ID | $-0.46$ | 0.09 | Higher peak, somewhat better LM (marginal) |
| Surprisal vs. relative peak onset | ${0.65}$ | 0.01 | Earlier peak, better LM (significant) |

The onset result is the stronger of the two, and it is the one the authors lean on. Note what it is not: 15 points from 5 models, where the 3 corpora per model are not independent. The extreme controls (shuffled text, untrained checkpoints) have no peak and cannot predict tokens, which makes the direction plausible, but the evidence among trained models is thin.

## Finding 6: the peak is the first layer that transfers

If peak layers hold an abstract summary of the input, a simple classifier on them should already work downstream. The authors freeze the model (fine-tuning would change the ID they want to relate to performance) and train a linear probe per layer on two binary tasks: IMDb sentiment and a balanced 30,588-example subset of the Jigsaw toxic comments data, with an 80/20 train/validation split.

![Figure 7 from Cheng et al. (2025): validation accuracy of per-layer linear classifiers for sentiment and toxicity, overlaid on the ID profile, for Llama, OPT and Pythia.](/imgs/blogs/emergence-of-a-high-dimensional-abstraction-phase-in-language-transformers-fig7.webp)

In every model, both accuracies climb through the early layers and converge inside the ID peak, then stay high. The peak was computed on three generic corpora, yet it locates where transfer to two other datasets saturates, so you could pick a frozen-feature layer from geometry alone.

## What the results add up to

The claims, their evidence, and my read of its strength:

| Claim | Evidence in the paper | My read |
| --- | --- | --- |
| A central high-ID phase exists | ID peaks in 5 models, 3 corpora; Pythia 2.8B and 12B too | Strong, consistent |
| It reflects learned linguistic structure | Gone on shuffled text; grows over training | Strong |
| Peak layers depart from the input | $\Delta(l_i \to l_{\text{first}})$ peaks with ID | Solid, one metric |
| Peak layers agree across models | Low cross-model $\Delta$ at band intersections; CKA concurs | Solid, qualitative |
| Abstraction is extracted at the peak | Syntax/semantic probes saturate; surface probes fall | Moderate, 5 old tasks |
| Earlier peak means better LM | $\rho = 0.65$, $p = 0.01$ on 15 points | Suggestive |
| Peak is where transfer starts | Sentiment and toxicity converge inside the peak | Solid, 2 tasks |

**What is load-bearing and may not transfer.** All five models share 32 layers and a 4096 hidden size, which is what makes "layer 6 to 20" comparable at all. The inputs are 20-token snippets of in-distribution text, and only the last token is analyzed. Compute is modest (12 nodes of 5 A30 GPUs, about 20 days of wall-clock including failed runs), so this is easy to reproduce.

**Where this fits.** Data living on a low-dimensional manifold inside a huge ambient space is the same manifold hypothesis behind [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions). The low ID of *weight updates*, rather than activations, is what motivates LoRA (see [effective LLM fine-tuning techniques](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques)).

**A disagreement with prior work.** Valeriani et al. (2023), studying vision and protein transformers, found semantic information coalescing in a low-ID *valley* after the first peak; this paper puts it in the peak itself. The authors suggest a methodological cause: Valeriani and colleagues averaged over all tokens and used a nearest-neighbor probe.

**Proposed uses, none tested in the paper.** Pick frozen-feature layers from the start of the peak; avoid pruning inside it (cited work finds late layers are the cheaper ones to drop); use peak layers as interfaces for model stitching. My own extrapolation from Finding 5: ID profiles could be a cheap, label-free quality signal during training.

## Critique

**What is strong.** The convergence of independent evidence is the paper's real asset. A geometric statistic (ID), a relational one ($\Delta$), supervised probes, transfer accuracy, and a correlation with LM quality all point at the same band of layers, across five model families and three corpora. The shuffled-text and checkpoint controls rule out an architectural quirk.

**What is weak.**

- **No causal test.** The authors say so: no ablations or interventions inside versus outside the peak. The functional story is inferred from correlations across layers.
- **Researcher degrees of freedom.** The GRIDE scale is picked by eye per (model, corpus, layer), and peak boundaries come from a hand-specified inflection rule. The halve-and-double check on $k$ helps.
- **Thin quality correlation.** 15 points from 5 models, with non-independent corpora; the max-ID result misses conventional significance.
- **Dated probing suite.** Sentence-level tasks built for older encoders, several dropped for ceiling effects. Alignment with factual recall or reasoning probes would be far stronger.
- **Same-shape models only.** Every main model has 32 layers, so whether the peak sits at a fixed fraction of depth is untested beyond three Pythia sizes.

**What would change my mind.** If someone projected peak-layer representations onto a lower-dimensional subspace and syntactic, semantic *and* transfer accuracy were unaffected, I would stop believing high dimensionality itself does the work and read the peak as a side effect. Conversely, if constraining ID at the peak selectively broke the syntactic and semantic tasks while leaving surface tasks intact, I would treat the paper's story as close to established.

## References

1. Emily Cheng, Diego Doimo, Corentin Kervadec, Iuri Macocco, Jade Yu, Alessandro Laio, Marco Baroni. *Emergence of a High-Dimensional Abstraction Phase in Language Transformers.* ICLR 2025. [arXiv:2405.15471v4](https://arxiv.org/abs/2405.15471v4)
2. Code and data: the authors' GitHub repository, linked in the setup section above.
3. Denti et al. (2022), the GRIDE estimator; Facco et al. (2017), TwoNN; Glielmo et al. (2022), Information Imbalance; Conneau et al. (2018), the probing tasks. All as cited in the paper.
4. Related on this site: [the semantic hub hypothesis](/blog/paper-reading/ai-interpretability/the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities), [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions), [effective LLM fine-tuning techniques](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques).

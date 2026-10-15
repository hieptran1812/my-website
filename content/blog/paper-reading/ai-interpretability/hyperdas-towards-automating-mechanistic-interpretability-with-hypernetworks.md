---
title: "HyperDAS: Teaching a Hypernetwork to Find Where a Concept Lives Inside an LLM"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How HyperDAS replaces the brute-force search of distributed alignment search with a transformer that reads a concept description, picks the token positions, and builds the subspace to intervene on, and what keeps it from cheating."
tags: ["paper-reading", "hyperdas", "distributed-alignment-search", "mechanistic-interpretability", "hypernetworks", "interchange-intervention", "ravel", "causal-interpretability", "iclr-2025-poster"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 22
paper:
  title: "HyperDAS: Towards Automating Mechanistic Interpretability with Hypernetworks"
  authors: "Jiuding Sun, Jing Huang, Sidharth Baskaran, Karel D'Oosterlinck, Christopher Potts, Michael Sklar, Atticus Geiger"
  venue: "ICLR 2025"
  url: "https://arxiv.org/abs/2503.10894"
---

> [!tldr]
> **Claim.** Finding the features that carry a concept (say, "the country of a city") inside an LLM normally means a brute-force search: train a separate intervention at every layer and token position, then keep the best. HyperDAS trains one transformer hypernetwork that reads a plain-English description of the concept, looks at the target model's hidden states, and outputs both *which tokens* to intervene on and *which subspace* to swap.
>
> **Why it matters.** It turns a search problem into a forward pass, which is the step you need before interpretability can scale past hand-picked token positions.
>
> **Most surprising result.** On RAVEL with Llama3-8B it lifts the average Disentangle score from 76.0 (MDAS, the previous best) to 84.7, and at deep layers it chooses to intervene on JSON punctuation tokens nobody had suspected of storing entity attributes.
>
> **Where it fails.** It costs about 2.4x the training compute of MDAS, it only works if the concept is linearly encoded, and without careful masking and a sparsity loss it learns to "hack" the benchmark instead of finding real structure. The authors spend a whole section on that last risk, and it is the most useful part of the paper.

Most of mechanistic interpretability runs on one experiment: change something inside the model and check whether the output moves the way your hypothesis predicts. The hard part is knowing *where* to run it. This post walks through HyperDAS (Sun et al., ICLR 2025) one technique at a time: problem, mental model, mechanism, math, a tiny example, and how it breaks.

## The problem: interventions need an address

Take the concept "the birth year of a person". You suspect some features of Llama3-8B's hidden state store it. To test that, you run an **interchange intervention**: feed the model "Albert Einstein was born in", but overwrite the suspected features with the values they take when the model reads "Marie Curie was a chemist". If the output flips from 1879 to 1934, you have evidence those features mediate birth year.

That experiment has two free choices. You must pick a *location* (which layer, which token position) and a *feature* (which direction or subspace of the hidden vector at that location). Distributed alignment search (DAS) learns the feature for you. It does not pick the location.

So in practice people search. They train one DAS rotation per layer and per token position, compare, and keep the winner. Or they skip the search and use a heuristic, most often "the attribute lives on the last token of the entity". The paper points out that this heuristic, common in knowledge-editing work, does not hold for every entity.

![Side by side: on the left, DAS trains a separate fixed rotation at every layer and token site and compares them; on the right, HyperDAS reads the concept and both prompts once and emits a per-example token pairing and subspace](/imgs/blogs/hyperdas-towards-automating-mechanistic-interpretability-with-hypernetworks-1.webp)

HyperDAS keeps the layer fixed (you still choose it) but removes the token search and makes the subspace depend on the example.

## Background: the distributed interchange intervention

Every later piece of HyperDAS feeds this one operation, so it is worth getting exactly right.

**The problem it solves.** Swapping an *entire* hidden vector is too blunt. A city token's residual stream carries its country, continent, latitude and language at once; swap the whole vector and you move all of them.

**Mental model.** Treat the hidden vector as a mixing desk with 4,096 sliders. Some small group of slider combinations controls "country". The intervention copies the settings of only that group from the counterfactual run into the base run and leaves every other slider where it was.

**Mechanism.** Project both hidden vectors onto a small subspace. Take the difference of the two projections. Lift that difference back into the full space and add it to the base vector. The base vector now has the counterfactual's coordinates inside the subspace and its own coordinates everywhere else.

**The math.** Let $\bar{h} \in \mathbb{R}^{d}$ be the hidden vector at some site when the model reads the base prompt $\bar{x}$, and $\hat{h} \in \mathbb{R}^{d}$ the vector at the matched site for the counterfactual prompt $\hat{x}$. For Llama3-8B, $d = 4096$. Let $R \in \mathbb{R}^{k \times d}$ have orthonormal rows, so $R h$ gives the $k$ coordinates of $h$ inside the subspace. (The paper phrases $R$ as a matrix "whose orthogonal columns are the features"; writing it as $k \times d$ just makes the shapes in $R h$ line up.) The intervention is the paper's Equation 1:

$$
H \leftarrow \bar{h} + R^{\top}\big(R\hat{h} - R\bar{h}\big)
$$

Here $R\hat{h} - R\bar{h} \in \mathbb{R}^{k}$ is how far apart the two runs are inside the subspace, and $R^{\top}$ maps that $k$-vector back into $\mathbb{R}^{d}$. Since $R R^{\top} = I_k$, the patched vector satisfies $R H = R\hat{h}$ exactly, while the component of $\bar{h}$ orthogonal to the subspace is untouched.

![Data flow of a distributed interchange intervention: base and counterfactual hidden states are projected onto subspace R, their difference is lifted back with R transpose and added to the base vector, and the model continues to predict France](/imgs/blogs/hyperdas-towards-automating-mechanistic-interpretability-with-hypernetworks-2.webp)

**Worked micro-example.** Take $d = 3$, $k = 1$, and $R = [0, 1, 0]$ (the subspace is the second axis). Let $\bar{h} = (2, 5, 1)$ and $\hat{h} = (7, -3, 4)$. Then $R\bar{h} = 5$, $R\hat{h} = -3$, the difference is $-8$, and $R^{\top}(-8) = (0, -8, 0)$. The patched vector is $(2, -3, 1)$: the base's first and third coordinates, the counterfactual's second.

**How DAS learns $R$.** DAS trains $R$ with a frozen model on examples whose correct post-intervention answer $y$ is known. With $M_{\gamma}(\bar{x})$ meaning "run model $M$ on $\bar{x}$ while applying intervention $\gamma$", the loss is the paper's Equation 2:

$$
\mathcal{L}_{\text{DAS}} = \mathrm{CE}\Big(M_{H \leftarrow \bar{h} + R^{\top}(R\hat{h} - R\bar{h})}(\bar{x}),\; y\Big)
$$

Only $R$ receives gradients. **When it fails:** $R$ is one fixed subspace at one fixed site. Wrong site, and no $R$ can succeed, which is why the search exists at all.

### How RAVEL scores an intervention

RAVEL, the benchmark used throughout, asks a method to move one entity attribute without disturbing the others. Each example has a base prompt that queries some attribute ("Albert Einstein studied the field"), a counterfactual prompt with another entity of the same type ("Poland declared 2011 the Year of Marie Curie"), and a target attribute.

| Score | What it measures | Einstein example |
| --- | --- | --- |
| **Cause** | Fraction of interventions that *do* change the targeted attribute | Target = field of study; output should become "chemistry" |
| **Iso** | Fraction that do *not* change a non-targeted attribute | Target = birth year; the field answer should stay "physics" |
| **Disentangle** | Average of Cause and Iso | |

The halves pull against each other: swap everything and Iso dies; swap nothing and Cause dies.

## The HyperDAS architecture in one picture

![Figure 1 from Sun et al. (2025): the HyperDAS framework localizing the concept "country". A transformer hypernetwork with two cross-attention blocks reads the instruction, attends to the target model's states for "Vienna is in" and "I love Paris", selects the tokens "nna" and "Paris", builds a feature subspace, and patches it so the model predicts France](/imgs/blogs/hyperdas-towards-automating-mechanistic-interpretability-with-hypernetworks-fig1.webp)

Figure 1 runs left to right through four steps, which map onto the next four sections:

1. **Encode the concept.** A transformer reads "The country of a city" and, through cross-attention, the target model's hidden states for both prompts.
2. **Select token positions.** Using that encoding as a query, it scores every (base token, counterfactual token) pair and picks "nna" in "Vienna" and "Paris".
3. **Identify a subspace.** The same encoding is turned into a per-example rotation of a learned base subspace.
4. **Intervene.** The subspace content of "nna" is replaced with that of "Paris" at layer $l$, and the model now says "France" after "Vienna is in".

The target LLM stays frozen throughout. Everything that learns lives in the hypernetwork $\mathcal{H}$.

## Technique 1: encoding the concept as a vector

**The problem.** Both later decisions need one representation of "which concept, in which pair of sentences". A concept ID alone is not enough, because "country" may sit in different places in different prompts.

**Mental model.** A reader holding a question in mind ("what country?") while skimming two documents side by side. The question shapes what they notice; the documents supply the evidence.

**Mechanism.** The instruction text is embedded with the target model's own token embeddings. Then $N$ decoder blocks run. Each block does ordinary masked self-attention over the instruction, then cross-attends to the base model states, then cross-attends to the counterfactual model states. After the last block, the vector at the final instruction token is the concept encoding.

**The math.** The instruction $x$ has $E$ tokens and is embedded as $e_0 = \mathrm{Emb}(x) \in \mathbb{R}^{E \times d}$. The base and counterfactual hidden states are stacked across *all* layers of the target model: $\bar{h} \in \mathbb{R}^{B \times L \times d}$ and $\hat{h} \in \mathbb{R}^{C \times L \times d}$, where $B$ and $C$ are the two prompt lengths and $L$ is the target's layer count. Block $p$ computes (Equations 3 to 5):

$$
\begin{aligned}
e'_p &= \mathrm{MHA}\big(Q(e_p), K(e_p), V(e_p)\big) \\
e''_p &= \overline{\mathrm{MHA}}\big(\bar{Q}(e'_p), \bar{K}(\bar{h}), \bar{V}(\bar{h})\big) \\
e_{p+1} &= \widehat{\mathrm{MHA}}\big(\hat{Q}(e''_p), \hat{K}(\hat{h}), \hat{V}(\hat{h})\big)
\end{aligned}
$$

Each line's queries come from the hypernetwork; the keys and values come from itself, then from the base run, then from the counterfactual run. The output we keep is $e^{(N)}_{E} \in \mathbb{R}^{d}$, the residual vector at the last instruction token after all $N$ blocks. The paper uses $N = 8$.

**Worked example.** For "Vienna is in" against "I love Paris" with the instruction "The country of a city", $e^{(N)}_E$ is one 4,096-dimensional vector whose only job is to steer the next two steps.

**When it fails.** Because every block can read every layer of the target model, this encoder can see the answer. Section 6 below shows how it uses that to cheat, and how masking stops it. The ablation in the paper's Figure 12 also shows the encoder carries weight: replacing it with a plain vector lookup drops city-domain Disentangle from 82.4 to 31.0, and removing just the cross-attention drops it to 76.1.

## Technique 2: pairing base tokens with counterfactual tokens

**The problem.** Which base token should be overwritten, and from which counterfactual token? Picking positions is a discrete choice, and discrete choices do not pass gradients.

**Mental model.** A seating chart between two rows of guests. Each base token can either take a seat opposite one counterfactual token or stay home (no intervention). During training, guests are allowed to split their attention fractionally across seats; at test time everyone must commit to one.

**Mechanism.** For every pair (base token $b$, counterfactual token $c$) at the chosen layer $l$, concatenate their two hidden vectors and squeeze them back to $d$ dimensions with a linear map. Add one extra "column" per base token that holds the base vector itself, meaning "do not intervene". Score every pair against the concept encoding with multi-head dot-product attention. Normalize so each base token's weights over the $C$ counterfactual tokens plus "do not intervene" sum to one.

**The math.** The merged representation of a pair is (Equations 6 and 7):

$$
g_{(b,c)} = F\big([\bar{h}^{(l)}_b ; \hat{h}^{(l)}_c]\big), \qquad g_{(b,C+1)} = \bar{h}^{(l)}_b
$$

where $F: \mathbb{R}^{2d} \to \mathbb{R}^{d}$ is linear and $[\cdot\,;\cdot]$ is concatenation. The raw score uses $M$ heads, each with query and key matrices $Q^{(i)}, K^{(i)} \in \mathbb{R}^{d \times d/M}$ (Equation 8, the paper uses $M = 32$):

$$
G'_{(b,c)} = \frac{\sum_{i=1}^{M} \big(e^{(N)}_{E} Q^{(i)}\big) \cdot \big(g_{(b,c)} K^{(i)}\big)}{M\sqrt{d}}
$$

A softmax then turns $G' \in \mathbb{R}^{B \times (C+1)}$ into $G$, with each base token's $C+1$ weights summing to one. The paper calls it a column-wise softmax because its plots draw base tokens as columns. The weights then build the counterfactual vector each base token will receive (Equation 9):

$$
\tilde{h}^{(l)}_b = G_{(b,C+1)}\,\bar{h}^{(l)}_b + \sum_{c=1}^{C} G_{(b,c)}\,\hat{h}^{(l)}_c
$$

If $G_{(b,C+1)} = 1$, token $b$ is left alone. If $G_{(b,c)} = 1$ for one $c$, token $b$ receives exactly the counterfactual vector at position $c$. In between, it receives a blend.

![Figure 2 from Sun et al. (2025): the intervention score matrix for counterfactual "The city of Macheng's official language is" and base "Springfield is a city in the country of". Left: raw soft scores. Right: after the double argmax at inference, only the pair (eng, field) is selected](/imgs/blogs/hyperdas-towards-automating-mechanistic-interpretability-with-hypernetworks-fig2.webp)

**Worked example, straight from Figure 2.** The base token "field" (the end of "Springfield") spreads its weight as 0.67 on "eng", 0.24 on "Mach", 0.07 on "'s" and 0.02 on [SELF]. Those sum to 1.0, as they must. Nearly every other base token puts 0.95 or more on [SELF]: they are left alone. So the network has learned, without being told, that the country attribute lives on the final token of each city name.

**When it fails.** The softness that makes this trainable also makes it exploitable. A blend of many counterfactual vectors is no longer an interchange intervention; it is closer to writing a new vector into the model. The next technique exists to stop that.

## Technique 3: snapping soft scores to a one-to-one alignment

**The problem.** Training uses soft weights, but evaluation must be a real interchange intervention: each base token untouched or fed exactly one counterfactual token. A network never pushed toward that collapses when the weights are snapped.

**Mental model.** A practice exam where you may hedge across answers, followed by a real exam where you must circle one. The sparse loss is the teacher docking points for hedging during practice.

**Mechanism.** Two pieces. During training, a **sparse attention loss** penalizes any counterfactual token whose total weight across base tokens exceeds one, which is what "one counterfactual token feeds many base tokens" looks like. At evaluation, a **double argmax** keeps a pair only if it is the largest entry in both its row and its column, and any base token left with no pair gets [SELF].

**The math.** Writing $S_c = \sum_{b} G_{(b,c)}$ for the total weight counterfactual token $c$ hands out, the sparse loss is (Equation 13):

$$
\mathcal{L}_{\text{sparse}} = \frac{1}{C}\sum_{c=1}^{C} S_c \cdot \mathbb{1}[S_c > 1]
$$

The training objective adds it to the task loss with a scheduled weight $\lambda$: $\mathcal{L} = \mathcal{L}_{\text{RAVEL}} + \lambda\,\mathcal{L}_{\text{sparse}}$. The task loss is Equation 2 with the HyperDAS-built $\tilde{h}$ and subspace plugged in, applied at every base position of layer $l$. Its label $y$ is the counterfactual's attribute when the instruction targets the attribute the base prompt asks about, and the base's own attribute otherwise. That second case is where Iso comes from. The weight $\lambda$ rises linearly from 0 to 1.5, starting halfway through training, so the network first finds any working soft solution and only then is squeezed toward a sparse one.

At test time (Equations 14 and 15):

$$
G_{(b,c)} \leftarrow \begin{cases} 1 & \text{if } G_{(b,c)} = \max\big(G_{(b,*)}, G_{(*,c)}\big) \\ 0 & \text{otherwise} \end{cases}, \qquad G_{(b,C+1)} \leftarrow 1 \text{ if } \max_c G_{(b,c)} = 0
$$

**Worked example.** In Figure 2, "eng" gives 0.67 to "field" and 0.01 to "country", so $S_{\text{eng}} = 0.68$. That is below 1 and costs nothing. Now a toy case of my own: suppose "eng" had put 0.7 on "field" and 0.6 on "city". Then $S_{\text{eng}} = 1.3$, and with $C = 10$ counterfactual tokens it adds ${1.3 / 10 = 0.13}$ to the loss. At inference, 0.67 is the largest value in both the "eng" row and the "field" column, so that one entry becomes 1 and the rest of the matrix goes to zero or [SELF], exactly as the right panel of Figure 2 shows.

![Three panels: soft G during training with the sparse loss penalizing counterfactual tokens whose weight sums exceed one; the double argmax at test time snapping to a one-to-one pairing; and the two failure modes of too little or too much sparse loss](/imgs/blogs/hyperdas-towards-automating-mechanistic-interpretability-with-hypernetworks-4.webp)

**When it fails, in both directions.** The paper's Figure 7 trains three versions on an occupation example. All three reach a weighted-intervention Disentangle of about 94.0 during training, so the training metric alone cannot tell them apart.

- **No sparse loss:** the last subject token of the counterfactual is copied into most base tokens. Fine under soft evaluation, broken once pairings must be one-to-one.
- **Too much sparse loss:** the pairing attention degenerates and blends all hidden states. Near-perfect soft score, total failure at snapped test time, and no interpretable pattern.

The lesson: a soft intervention can score well without making any discrete causal claim, so the test protocol must force one.

## Technique 4: building the subspace with a Householder reflection

**The problem.** DAS learns one fixed $R$. HyperDAS wants a subspace that can bend per example and per concept, but whatever it outputs must still have orthonormal rows, or Equation 1 stops being a clean swap.

**Mental model.** A fixed set of mirrors you can tilt. The base subspace $R$ is learned once; for each example the network picks a mirror angle, and reflecting through a mirror never stretches or skews anything.

**Mechanism.** Feed the concept encoding through an MLP to get a vector $v$. Build the Householder matrix from $v$: the reflection across the hyperplane perpendicular to $v$. Multiply the learned base $R$ by it. A reflection is orthogonal, so the result is still orthonormal.

**The math.** With $v = \mathrm{MLP}(e^{(N)}_{E}) \in \mathbb{R}^{d}$ and $v \neq 0$ (Equation 10):

$$
\mathbf{H} = I - 2\,\frac{v v^{\top}}{v^{\top} v}, \qquad R' = R\,\mathbf{H}
$$

$\mathbf{H} \in \mathbb{R}^{d \times d}$ satisfies $\mathbf{H}^{\top}\mathbf{H} = I$, so $R' R'^{\top} = R\,\mathbf{H}\mathbf{H}^{\top} R^{\top} = R R^{\top} = I_k$. The final intervention at each base token is Equation 11, which is Equation 1 with $R'$ and the blended $\tilde{h}^{(l)}_b$ in place of $R$ and $\hat{h}$. The paper sets $k = 128$ of $d = 4096$ dimensions after trying sizes from 32 up to 2048.

![How the per-example subspace is built: the concept encoding passes through an MLP to give v, v defines a Householder reflection H, and the learned shared subspace R is multiplied by H to give the per-example R prime with 128 of 4096 dimensions](/imgs/blogs/hyperdas-towards-automating-mechanistic-interpretability-with-hypernetworks-3.webp)

**Worked micro-example.** In $d = 2$, take $v = (1, 1)$. Then $v^{\top}v = 2$ and $\mathbf{H} = I - (v v^{\top}) = \begin{pmatrix} 0 & -1 \\ -1 & 0 \end{pmatrix}$. With $R = [1, 0]$ (the first axis), $R' = R\mathbf{H} = [0, -1]$: the subspace is now the second axis. A different $v$ would land it somewhere else, always as a unit vector.

**Why it works, and its limit.** One reflection is cheap ($O(d)$ to apply) and keeps orthonormality for free, but it is a narrow family: every example's subspace is a mirror image of the same $R$. The paper reads the learned $v$ vectors as a window into the model and finds they cluster by attribute (average same-attribute cosine 0.97 to 1.00 on the city domain), with latitude and longitude sharing an unusually similar subspace (0.87 cosine). Whether that reflects the target model or the hypernetwork's parametrization is not something the paper separates.

## Technique 5: guarding against an interpretability method that cheats

Any supervised method strong enough to find structure is strong enough to manufacture it.

**The shortcut.** In Appendix A.3, the base prompt "Vienna, known for its Imperial palaces, is a city in the country of" asks about country. The instruction says "localize the latitude". The correct label is the base's own country, Austria, because latitude features should not affect a country question. A hypernetwork that can read the base hidden states notices that the queried attribute (country) differs from the instruction (latitude) and simply chooses not to intervene. That scores perfectly on Iso examples and has found nothing. The paper's Figure 9 shows the learned pattern: every base token mapped to [SELF].

**The fix.** Mask every base token from the attribute mention onward, so the hypernetwork cannot see what is being asked. Its choice of location can then depend only on the instruction and the entity.

| Guard | Hack it blocks | Evidence in the paper |
| --- | --- | --- |
| Mask base prompt from the attribute mention | "Skip the intervention when the attributes differ" | Figure 9, Appendix A.3 |
| Sparse attention loss | One counterfactual token copied into many base tokens | Figure 7, middle panel |
| Scheduled $\lambda$, not maximal | Blending all states, which acts like steering | Figure 7, right panel |
| Snapped one-to-one evaluation | Any soft-blend solution | Figure 2, Equations 14 and 15 |
| Orthonormal low-rank $R'$ | Writing arbitrary new vectors | Section 3.3 |

None of these prove faithfulness. Each closes a door the authors found open, which leaves open any door they did not find.

## Results: what moved on RAVEL

The setup: Llama3-8B as the target, frozen. The hypernetwork has 8 decoder blocks, 32 heads for pair scoring, and a 128-dimensional subspace, and it is trained from scratch (initializing from Llama3-8B weights gave no advantage in Appendix A.4). The baseline is **MDAS**, the previous state of the art, which uses a multi-task DAS objective at a hand-chosen position, the last entity token.

Two variants appear. **Asymmetric** HyperDAS may pick different positions for the base and counterfactual prompts. **Symmetric** randomly flips the order of the concatenation in Equation 6, so "get" and "set" use the same localization rule.

![Figure 3a from Sun et al. (2025): Cause and Iso scores on the five RAVEL domains with Llama3-8B for MDAS and four HyperDAS variants, plus average Disentangle](/imgs/blogs/hyperdas-towards-automating-mechanistic-interpretability-with-hypernetworks-fig3.webp)

| Method | City Cause / Iso | Nobel Cause / Iso | Occupation Cause / Iso | Avg Disentangle |
| --- | --- | --- | --- | --- |
| MDAS | 55.8 / 77.9 | 56.0 / 93.5 | 50.7 / 88.1 | 76.0 |
| HyperDAS, asymmetric | 70.8 / 93.9 | 55.4 / 95.1 | 50.4 / 99.1 | **84.7** |
| HyperDAS, asymmetric, all domains | 58.8 / 90.5 | 47.6 / 92.0 | 75.7 / 82.1 | 80.7 |
| HyperDAS, symmetric | 76.9 / 90.9 | 59.2 / 88.4 | 47.1 / 89.1 | 76.9 |
| HyperDAS, symmetric, all domains | 16.8 / 94.7 | 2.0 / 98.8 | 6.1 / 97.3 | 54.8 |

Three things stand out once you read past the bold number.

- **The gain is mostly Iso.** Averaged over the five domains, asymmetric HyperDAS moves Cause from 64.4 to 72.5 and Iso from 87.4 to 96.8 (my averages of the table's columns). On Nobel laureates and occupations, its Cause is slightly *below* MDAS. It wins each domain's Disentangle, but largely by disturbing less, not by moving more.
- **One model for all domains works, mostly.** Training a single asymmetric HyperDAS across all five domains loses 4.0 points to the per-domain models but still beats MDAS. The symmetric version trained on all domains collapses to near-zero Cause, which the paper reports but does not explain.
- **Symmetry costs.** Forcing the same localization for base and counterfactual matches MDAS on average and does worse on verbs (42.3 Cause against 93.0 for asymmetric).

![Figure 3b from Sun et al. (2025): Cause, Iso and Disentangle across the layers of Llama3-8B for the city domain, for MDAS, asymmetric HyperDAS and symmetric HyperDAS. All peak near layer 15](/imgs/blogs/hyperdas-towards-automating-mechanistic-interpretability-with-hypernetworks-fig4.webp)

The layer sweep on the city domain (every second layer) shows HyperDAS above MDAS at every layer tested, with both peaking near layer 15. One small inconsistency: the table caption says results come from "the best layer between 10 and 15", while the text says layer 15.

### Where it chose to intervene

![Figure 4 from Sun et al. (2025): token types chosen by HyperDAS in the counterfactual and base prompts at a shallow (7), middle (15) and deep (29) layer, split into BOS, entity, JSON syntax and other tokens](/imgs/blogs/hyperdas-towards-automating-mechanistic-interpretability-with-hypernetworks-fig5.webp)

The location analysis is where the method earns the word "automating":

- **Counterfactual side:** at every layer it reads the attribute from entity tokens. Attribute information is on the entity early.
- **Base side, layer 7:** erratic, often the beginning-of-sequence token or "is" (36% of the time).
- **Base side, layer 15:** consistently the entity token, matching prior findings by Huang et al. (2024) and Geva et al. (2023).
- **Base side, layer 29:** often JSON syntax tokens such as an opening brace, positions not previously known to carry attributes.

It also picks more than one token 53% of the time, unlike MDAS's single fixed position. And the asymmetric model splits roles on three-token entities: it reads from the last entity token of the counterfactual in at least 95% of cases but writes to the *second-to-last* entity token in the base.

### What it costs

| | MDAS | HyperDAS |
| --- | --- | --- |
| Training compute, one epoch, city country attribute | 193,833 TFLOPs | 468,923 TFLOPs (about 2.4x) |
| Extra memory | 4.1 GB per attribute | 52 GB total |
| Memory for all 23 RAVEL attributes | 23 x 4.1 + 16 = 110.3 GB | 52 + 16 = 68 GB |

More compute per task, but memory stops growing with the number of attributes.

**What is load-bearing and may not transfer.** RAVEL supplies labelled counterfactual pairs for every attribute; most real interpretability questions have no such dataset. You still choose the layer, and the JSON templates behind the layer-29 finding are RAVEL's. My reading: that finding is a property of Llama3-8B on these templates until shown elsewhere.

## Critique

**What is strong.** A clean answer to a real bottleneck, with a decisive ablation (drop the hypernetwork and Disentangle falls from 82.4 to 31.0). The authors also found two concrete ways their method cheats, showed them (Figures 7 and 9), and designed around them.

**What is weak.**

- **The headline gain is mostly Iso,** and a method that intervenes less scores better on Iso. Cause is roughly flat against MDAS on two of five domains.
- **Faithfulness is argued, not measured.** There is no test that would catch a third, unknown hack. A useful one: train HyperDAS on a randomly initialized Llama3-8B and check the score falls to chance.
- **Thin baselines.** The comparison is MDAS at one hand-picked position. A DAS sweep over all token positions at layer 15, the brute force HyperDAS claims to replace, is the natural baseline and is missing. The SAE baseline in Appendix A.5 appears only as a trade-off plot.
- **One target model, one benchmark.** Everything is Llama3-8B on RAVEL.
- **Unexplained collapse.** Symmetric all-domain training drops to 2.0 Cause on Nobel laureates, and the paper does not say why.
- **Linear only.** As the authors say, the method cannot find a concept that is not linearly encoded. Recent work argues some features are not, as covered in [this site's post on multi-dimensional features](/blog/paper-reading/ai-interpretability/not-all-language-model-features-are-one-dimensionally-linear).

**What would change my mind.** I would trust the localization claims much more if (1) a full DAS sweep over every layer-15 token position landed on the same tokens HyperDAS picks, and (2) the same HyperDAS training on a randomly initialized model gave near-zero Cause. If the sweep disagreed, or the random model still scored well, I would read HyperDAS as a strong editor rather than an interpreter.

For the hypothesis underneath all of this, that concepts sit in linear subspaces at all, see [the linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis).

## References

1. Sun, J., Huang, J., Baskaran, S., D'Oosterlinck, K., Potts, C., Sklar, M., Geiger, A. *HyperDAS: Towards Automating Mechanistic Interpretability with Hypernetworks.* ICLR 2025. [arXiv:2503.10894](https://arxiv.org/abs/2503.10894)
2. Code: [github.com/jiudingsun01/HyperDAS](https://github.com/jiudingsun01/HyperDAS)
3. Huang, J., Wu, Z., Potts, C., Geva, M., Geiger, A. *RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations.* ACL 2024.
4. Geiger, A., Wu, Z., Potts, C., Icard, T., Goodman, N. *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations.* CLeaR 2024 (the DAS paper).
5. On this site: [The linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis) and [Not all language model features are one-dimensionally linear](/blog/paper-reading/ai-interpretability/not-all-language-model-features-are-one-dimensionally-linear).

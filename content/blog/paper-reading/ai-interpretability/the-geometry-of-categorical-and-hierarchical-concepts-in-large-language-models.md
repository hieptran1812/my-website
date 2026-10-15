---
title: "The Geometry of Categorical and Hierarchical Concepts in LLMs: Vectors, Polytopes and Right Angles"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How Park, Choe, Jiang and Veitch turn concept directions into vectors, show that categories become polytopes and hierarchy becomes orthogonality, and test it on 900+ WordNet concepts in Gemma and LLaMA-3."
tags:
  - paper-reading
  - concept-geometry
  - hierarchical-concepts
  - linear-representation-hypothesis
  - causal-inner-product
  - representation-learning
  - large-language-models
  - mechanistic-interpretability
  - iclr-2025-oral
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 19
paper:
  title: "The Geometry of Categorical and Hierarchical Concepts in Large Language Models"
  authors: "Kiho Park, Yo Joong Choe, Yibo Jiang, Victor Veitch"
  venue: "ICLR 2025 (oral), arXiv 2406.01506"
  url: "https://arxiv.org/abs/2406.01506"
---

> [!tldr]
> **Claim.** If an LLM linearly represents a feature like *is animal*, that representation is not just a direction but a vector with a natural length. Categories such as {mammal, bird, fish} then become polytopes (convex hulls of those vectors), and a child concept's "extra" part is orthogonal to its parent.
> **Why it matters.** It gives the linear representation hypothesis a geometry for hierarchy, which is exactly the structure sparse autoencoders keep tripping over (absorption, splitting).
> **Most surprising result.** Across 593 noun synsets in Gemma-2B, the cosine between $\bar\ell_{\text{parent}}$ and $\bar\ell_{\text{child}} - \bar\ell_{\text{parent}}$ sits near zero, and this survives a control that breaks the trivial set-inclusion explanation.
> **Where it fails.** Everything lives at the final layer and depends on whitening the unembedding matrix. The worst-case "is this just set inclusion?" question needs a careful control, and the intervention test only covers three hand-picked pairs.

## The question: where does "animal" live?

Ask a model to finish "The ___ barked" and its last-layer vector must say *animal*, *mammal* and *dog* at once. The linear representation hypothesis says concepts like these are directions in that space.

That hypothesis worked nicely for contrasts with clean word pairs, like male ⇒ female from (man, woman) and (king, queen). It says nothing about a feature with no natural opposite, such as *is animal*. And it says nothing about how *animal*, *mammal* and *dog* relate to each other geometrically.

This paper answers both. Its answer fits in one picture: hierarchy is a stack of orthogonal subspaces, and each category is a simplex living inside one of them.

![Figure 1 from Park et al. (2025): hierarchically related concepts live in orthogonal subspaces and categorical concepts form polytopes, shown as a cartoon (top) and measured in Gemma (bottom)](/imgs/blogs/the-geometry-of-categorical-and-hierarchical-concepts-in-large-language-models-fig1.webp)

Read the top panel of Figure 1 as a tree drawn with vectors. The arrow from Organism to Animal is one direction; the arrows from Animal out to Mammal, Bird, Fish and Reptile live in a subspace at a right angle to it (the ⊕ means direct sum).

## Setup: one space, one inner product

### The problem

An LLM has two vector spaces at the output. The context vector $\lambda(x) \in \mathbb{R}^d$ is the final-layer, final-token hidden state for input text $x$. The unembedding $\gamma(y) \in \mathbb{R}^d$ is the row of the output matrix for token $y$. "Representation of a concept" could mean a vector in either space, and the ordinary dot product has no reason to respect meaning.

### The mechanism

The model only ever uses these two vectors through the softmax:

$$
P(y \mid x) = \frac{\exp\big(\lambda(x)^\top \gamma(y)\big)}{\sum_{y' \in \text{Vocab}} \exp\big(\lambda(x)^\top \gamma(y')\big)}
$$

Here $d$ is the hidden size (2,048 for Gemma-2B) and the sum runs over the whole vocabulary (256K tokens for Gemma). Because only the products $\lambda^\top\gamma$ matter, we can apply any invertible matrix $A$ to one side and its inverse transpose to the other, and shift the unembeddings by a constant, without changing a single probability.

Park et al. (2024), the precursor paper, showed there is a choice of $A$ that makes the Euclidean dot product a **causal inner product**: one under which concepts that can be varied independently (language and gender, say) are orthogonal. In practice that choice is whitening the unembeddings:

$$
g(y) = \mathrm{Cov}(\gamma)^{-1/2}\,\big(\gamma(y) - \mathbb{E}[\gamma]\big), \qquad \ell(x) = A^{-\top}\lambda(x)
$$

where $\gamma$ is the unembedding of a token drawn uniformly from the vocabulary, $\mathrm{Cov}(\gamma)$ is its $d \times d$ covariance, and $\mathbb{E}[\gamma]$ its mean. After this, $g(y)$ and $\ell(x)$ live in one space and the plain dot product means something.

### When it fails

Whitening is defined only for the unembedding matrix, so the whole paper is a statement about the **final layer**. The authors say so directly: making sense of internal-layer geometry is an open question. Appendix F also shows the payoff is real. With centering alone and no whitening (the naive Euclidean inner product), the hierarchical orthogonality disappears (their Figure 12).

## From directions to vectors

### The problem

Park et al. (2024) defined a linear representation as a direction: adding it to the context raises the probability of the concept and leaves every causally separable concept alone. A direction has no length. Without length you cannot add, subtract or take convex hulls, so you cannot even ask whether $\ell_{\text{bird}} - \ell_{\text{mammal}}$ means anything.

### A stronger definition

Picture a dimmer switch labeled *animal*. Turning it up should make the next token more likely to be some animal. It should not tilt the odds between mammal and bird; if it did, the switch would really be a blend of *animal* and *mammal*.

The paper formalizes "tilting the odds" with **subordination**. Each attribute $w$ gets a token set $\mathcal{Y}(w)$; $z$ is subordinate to $w$, written $z \prec w$, when $\mathcal{Y}(z) \subseteq \mathcal{Y}(w)$. Definition 3 then says $\bar\ell_W$ is a linear representation of binary concept $W$ if, for every context $\ell$ and every $\alpha > 0$,

$$
P(W = 1 \mid \ell + \alpha \bar\ell_W) > P(W = 1 \mid \ell), \qquad P(Z \mid \ell + \alpha \bar\ell_W) = P(Z \mid \ell)
$$

for every concept $Z$ that is **either subordinate to or causally separable with** $W$. The "subordinate" clause is the new part. It forbids the animal direction from reshuffling mammal versus bird.

![The paper's ladder: a direction becomes a vector once Theorem 4 gives it a length, and vectors compose into contrasts, polytopes and orthogonal hierarchy](/imgs/blogs/the-geometry-of-categorical-and-hierarchical-concepts-in-large-language-models-1.webp)

### Theorem 4: the length falls out

The strengthened definition forces a striking consequence. Take any two animal tokens, "dog" and "cat". The contrast dog ⇒ cat is subordinate to *animal*, so moving along $\bar\ell_W$ must not change their relative probability. That means $\bar\ell_W^\top\big(g(\text{dog}) - g(\text{cat})\big) = 0$: every animal token has the **same** projection. The same argument applies to any two non-animal tokens.

Choosing the unembedding origin to put the non-animal level at zero gives Theorem 4:

$$
\bar\ell_W^\top g(y) =
\begin{cases}
b_w & \text{if } y \in \mathcal{Y}(w) \\
0 & \text{if } y \notin \mathcal{Y}(w)
\end{cases}
$$

with a constant $b_w > 0$. Definition 5 then names the **vector representation** $\bar\ell_w$: the representation direction scaled to length $b_w$. Now the concept has a magnitude.

**Worked example.** Say the unit animal direction gives projections of 3.0 for "dog", 3.0 for "cat" and 0 for "table", so $b_{\text{animal}} = 3$. Add $\alpha = 0.5$ of that unit direction to the context. The logit gap between "dog" and "table" grows by ${0.5 \times (3 - 0) = 1.5}$ nats, while the gap between "dog" and "cat" changes by ${0.5 \times (3 - 3) = 0}$. That zero is the subordination constraint at work.

The paper is candid that exact equality will not come out of gradient descent. The realistic version is that projections cluster around $b_w$ and zero with spreads small compared with $b_w$.

## Categories as polytopes

### Contrasts are differences

Once features are vectors, Corollary 6 says the contrast $w_0 \Rightarrow w_1$ (for example mammal ⇒ bird) is represented by $\bar\ell_{w_1} - \bar\ell_{w_0}$. The reason is the theorem: on the subspace these two vectors span, mammal tokens sit at one point and bird tokens at another, so the difference moves you from one cluster to the other.

### Categories are hulls

Definition 7 defines the representation of a categorical concept $W = \{w_0, \dots, w_{k-1}\}$ as the **convex hull** of the vectors $\bar\ell_{w_0}, \dots, \bar\ell_{w_{k-1}}$. Mixing the vertices with nonnegative weights that sum to one gives every point inside, which matches a distribution over the $k$ categories.

Appendix C sharpens this. Proposition 9 says that if the model can realize **any** joint distribution over $k$ mutually exclusive attributes, their vectors must form a $(k-1)$-simplex: a triangle for three categories, a tetrahedron for four. The argument for $k = 3$ is short. If the three vertices were collinear, no direction could raise $P(w_2)$ while keeping the ratio $P(w_1)/P(w_0)$ fixed. A triangle always has one, perpendicular to the opposite side.

![Figure 6 from Park et al. (2025): Gemma tokens projected onto the span of the mammal, bird and fish vectors form a triangle orthogonal to the animal arrow (left), and the reptile, fish and bird contrasts form a simplex (right)](/imgs/blogs/the-geometry-of-categorical-and-hierarchical-concepts-in-large-language-models-fig6.webp)

In the left panel of Figure 6, colored token clouds sit near three vertices, and the black arrow for $\bar\ell_{\text{animal}}$ stands off the triangle's plane at a right angle. That right angle is the next theorem.

**When it fails.** "Polytope" is permissive. Definition 7 also builds a hull for {dog, sandwich, running}, which no one would call a natural category. The simplex result needs the strong assumption that every distribution over the categories is reachable, which the paper states but does not test.

## Hierarchical orthogonality (Theorem 8)

### The intuition

Go back to the dimmer switch. Turning up *animal* must not change mammal versus bird. In vector terms, adding $\bar\ell_{\text{animal}}$ must leave the dot product with $\bar\ell_{\text{bird}} - \bar\ell_{\text{mammal}}$ unchanged, which happens exactly when they are orthogonal. Theorem 8 turns that into four statements.

### The four statements

For $z \prec w$ and vector representations that exist:

| Part | Statement | Plain reading |
| --- | --- | --- |
| (a) | $\bar\ell_w \perp \bar\ell_z - \bar\ell_w$ | The child's extra part is at a right angle to the parent |
| (b) | $\bar\ell_w \perp \bar\ell_{z_1} - \bar\ell_{z_0}$ | A contrast inside $w$ is orthogonal to $w$ |
| (c) | $\bar\ell_{w_1} - \bar\ell_{w_0} \perp \bar\ell_{z_1} - \bar\ell_{z_0}$ | Parent contrast ⊥ child contrast (plant ⇒ animal vs mammal ⇒ bird) |
| (d) | $\bar\ell_{w_1} - \bar\ell_{w_0} \perp \bar\ell_{w_2} - \bar\ell_{w_1}$ for $w_2 \prec w_1 \prec w_0$ | Each step down a chain turns 90 degrees |

![Theorem 8 in two dimensions: the child-specific part of each vector sits at a right angle to the parent, so the mammal to bird contrast is orthogonal to the animal vector](/imgs/blogs/the-geometry-of-categorical-and-hierarchical-concepts-in-large-language-models-2.webp)

### The mechanism behind (a)

The proof uses Theorem 4 three times. For $z \prec w$, project any token onto $\bar\ell_z - \bar\ell_w$:

- a token in $\mathcal{Y}(z)$ gives $b_z - b_w$;
- a token in $\mathcal{Y}(w)$ but not $\mathcal{Y}(z)$ gives ${0 - b_w = -b_w}$;
- a token outside $\mathcal{Y}(w)$ gives ${0 - 0 = 0}$.

So $\bar\ell_z - \bar\ell_w$ only reshuffles mass **inside** $w$: it is the representation of "is $z$, given $w$". That concept is subordinate to $w$, and the causal inner product makes a concept's representation orthogonal to anything it must not disturb. Hence (a). Statement (b) follows by subtracting two instances of (a).

Statement (d) is Pythagoras. From (a), $\|\bar\ell_{w_1} - \bar\ell_{w_0}\|^2 = \|\bar\ell_{w_1}\|^2 - \|\bar\ell_{w_0}\|^2$, and likewise for the other pairs, so the three squared lengths satisfy

$$
\|\bar\ell_{w_1} - \bar\ell_{w_0}\|_2^2 + \|\bar\ell_{w_2} - \bar\ell_{w_1}\|_2^2 = \|\bar\ell_{w_2} - \bar\ell_{w_0}\|_2^2,
$$

which holds only when the two steps meet at a right angle.

### Worked example

Take three dimensions and let $\bar\ell_{\text{animal}} = (2, 0, 0)$, $\bar\ell_{\text{mammal}} = (2, 1, 0)$ and $\bar\ell_{\text{dog}} = (2, 1, 0.8)$.

| Quantity | Value | Check |
| --- | --- | --- |
| $\bar\ell_{\text{mammal}} - \bar\ell_{\text{animal}}$ | $(0, 1, 0)$ | dot with $(2,0,0)$ is 0, so (a) holds |
| $\bar\ell_{\text{dog}} - \bar\ell_{\text{mammal}}$ | $(0, 0, 0.8)$ | dot with $(0,1,0)$ is 0, so (d) holds |
| squared lengths of the two steps | 1 and 0.64 | sum 1.64 |
| $\lVert\bar\ell_{\text{dog}} - \bar\ell_{\text{animal}}\rVert^2$ | ${1^2 + 0.8^2 = 1.64}$ | Pythagoras matches |

Notice what this predicts for cosines. $\bar\ell_{\text{mammal}}$ and a sibling $\bar\ell_{\text{bird}} = (2, 0, 1.5)$ share the component $(2,0,0)$, so their cosine is ${4 / (\sqrt{5} \cdot 2.5) \approx 0.72}$. Siblings should look **similar**, while their child-specific parts are orthogonal. Both predictions get tested below.

![Figure 2 from Park et al. (2025): all 256K Gemma tokens projected onto 2D planes spanned by animal, mammal, bird and plant vectors; colored tokens line up with the predicted right angles](/imgs/blogs/the-geometry-of-categorical-and-hierarchical-concepts-in-large-language-models-fig2.webp)

In Figure 2(a), animal tokens share one coordinate along $\bar\ell_{\text{animal}}$ while mammals spread along the second axis. In panel (c), plant ⇒ animal and mammal ⇒ bird span a plane where the clusters sit at the corners of a rectangle, which is statement (c) in pixels. One caveat the appendix discloses: the token sets for Figures 1, 2 and 6 were generated with ChatGPT-4 and checked by hand, because WordNet's animal categories ("chordate", "aquatic vertebrate") are awkward to plot.

## Experiments: estimating 900+ concept vectors

### Building concepts from WordNet

WordNet organizes words into synsets connected by hyponym (is-a) links. For each synset $w$, the authors take $\mathcal{Y}(w)$ to be every word in any descendant synset, then add plurals, verb tenses and capitalization variants. They keep synsets with at least 50 words in Gemma's vocabulary, leaving **593 noun** and **364 verb** synsets. Where a synset has two parents (WordNet is not a perfect tree), they pick one.

### The LDA estimator

Theorem 4 asks for two things from $\bar\ell_w$: tokens in $\mathcal{Y}(w)$ should project well away from the rest, and their projections should be nearly equal. That is the job description of Linear Discriminant Analysis, which minimizes within-class variance. The estimator is

$$
\bar\ell_w = \big(\tilde g_w^\top \mathbb{E}(g_w)\big)\,\tilde g_w, \qquad \tilde g_w = \frac{\mathrm{Cov}(g_w)^{\dagger}\,\mathbb{E}(g_w)}{\big\|\mathrm{Cov}(g_w)^{\dagger}\,\mathbb{E}(g_w)\big\|_2}
$$

where $g_w$ is the whitened unembedding of a token drawn uniformly from $\mathcal{Y}(w)$, $\mathbb{E}(g_w) \in \mathbb{R}^d$ is the class mean, $\mathrm{Cov}(g_w)$ is the $d \times d$ class covariance and $\dagger$ is the pseudo-inverse. The unit vector $\tilde g_w$ points where class tokens agree most, and the scalar in front sets the length to the mean projection, playing the role of $b_w$.

With only 50 to a few thousand words per class in 2,048 dimensions, the raw covariance is singular, so they use the Ledoit-Wolf shrinkage estimator. Appendix F shows why the extra machinery matters: using the plain class mean $\mathbb{E}(g_w)$ gives much noisier projections (their Figure 11).

### Test 1: do vector representations exist?

For each synset they fit on 70% of $\mathcal{Y}(w)$ and project the held-out 30% plus random words onto $\bar\ell_w$, dividing by $\|\bar\ell_w\|_2^2$. Theory says held-out members should land near 1 and random words near 0.

![Figure 3 from Park et al. (2025): normalized projections of train, test and random words for every WordNet noun feature; test words sit near 1 and random words near 0 with real unembeddings, while shuffled unembeddings collapse test words to 0](/imgs/blogs/the-geometry-of-categorical-and-hierarchical-concepts-in-large-language-models-fig3.webp)

That is what Figure 3 shows on the left. The control on the right shuffles the unembedding vectors across tokens, which keeps the geometry of the cloud but destroys which word owns which vector. There, held-out projections drop to about 0: the LDA cannot generalize from the training words, so the structure on the left comes from semantics and not from the estimator.

### Test 2: does steering touch only the target?

Definition 3 is an interventional claim, so they test it directly. Add the normalized parent contrast $\bar\ell_W$ to the context and measure how the log-odds between token pairs change. Because the change is $\bar\ell_W^\top\big(g(y_1) - g(y_0)\big)$, it does not depend on the context at all.

| Parent contrast $W$ | Child contrast $Z$ | Parent pairs (mean ± sd) | Child pairs (mean ± sd) |
| --- | --- | --- | --- |
| plant ⇒ animal | mammal ⇒ reptile | 5.1265 ± 1.1731 | -0.0600 ± 1.2190 |
| fluid ⇒ solid | crystal ⇒ food | 9.8296 ± 1.1099 | 0.3770 ± 1.5410 |
| scientist ⇒ contestant | athlete ⇒ player | 14.4222 ± 0.9458 | -0.1545 ± 1.1426 |

The parent log-odds shift by 5 to 14 nats; the child shift averages within 0.4 of zero. The standard deviations deserve a second look, though. The child spread (about 1.1 to 1.5) is as large as the parent spread, so individual token pairs do move. The claim holds on average, not pair by pair.

### Test 3: does similarity mirror the tree?

If $\bar\ell_{\text{bird}} = \bar\ell_{\text{animal}} + (\text{orthogonal part})$ and likewise for mammal, siblings should have a substantial cosine (the 0.72 in our toy). Figure 4 compares the WordNet graph distance matrix, the cosine matrix of the estimated vectors, and the cosine matrix after shuffling.

![Figure 4 from Park et al. (2025): WordNet graph closeness (left), cosine similarity of Gemma's concept vectors (middle) and the shuffled control (right); the middle heatmap reproduces the block structure of the left one](/imgs/blogs/the-geometry-of-categorical-and-hierarchical-concepts-in-large-language-models-fig4.webp)

The middle heatmap reproduces the block structure of the left. The shuffled control is informative in a subtle way. Parent and child still correlate there, because "mammal" words remain a subset of "animal" words whatever vectors they get. Siblings, though, become nearly orthogonal, as random vectors in high dimension should. Sibling similarity is the part that needs real semantics.

### Test 4: is hierarchy orthogonal?

The headline test computes, for every noun synset $w$, the cosine between $\bar\ell_{\text{parent}(w)}$ and $\bar\ell_w - \bar\ell_{\text{parent}(w)}$. Theorem 8(a) says it should be 0.

![Figure 5 from Park et al. (2025): cosine between each child-minus-parent vector and its parent across the WordNet noun hierarchy, using all tokens (left) and a 70% split per synset (right), with random-parent and shuffled controls](/imgs/blogs/the-geometry-of-categorical-and-hierarchical-concepts-in-large-language-models-fig5.webp)

The blue curve (real unembeddings) hugs zero in both panels. The trouble is that "near zero" is cheap in 2,048 dimensions, so everything hinges on the controls.

## Controls: is the orthogonality real?

![Four controls behind Figure 5 and which boring explanation each one rules out](/imgs/blogs/the-geometry-of-categorical-and-hierarchical-concepts-in-large-language-models-3.webp)

### Control 1: random parent

Replace the true parent with a random synset. The cosine moves far from zero (orange). So the zero is not a generic property of high-dimensional vectors.

A note of my own, not the paper's: this control is weaker than it looks. For a random $r$ nearly orthogonal to child $c$, $\bar\ell_r^\top(\bar\ell_c - \bar\ell_r) \approx -\|\bar\ell_r\|^2$, which is strongly negative by construction. What the true-parent result really shows is that $\bar\ell_p^\top \bar\ell_c \approx \|\bar\ell_p\|^2$: the child's projection onto the parent equals the parent's length. That is still a nontrivial, theory-predicted fact.

### Control 2: shuffled, all tokens

Here is the surprise. With shuffled unembeddings and all tokens, the green curve on the left is **also** near zero. Appendix G explains why with a calculation worth doing yourself.

Let the parent's tokens be $a_1, \dots, a_{N_a}$ (shared with the child) plus $b_1, \dots, b_{N_b}$ (parent only), all i.i.d. standard Gaussians in $\mathbb{R}^d$. Use class means: $v_z = \frac{1}{N_a}\sum a_i$ and $v_w = \frac{1}{N_a + N_b}\big(\sum a_i + \sum b_i\big)$. Then

$$
(v_z - v_w)^\top v_w = \frac{N_b \big\|\sum a_i\big\|^2 - N_a \big\|\sum b_i\big\|^2}{N_a (N_a + N_b)^2} \approx \frac{N_b N_a d - N_a N_b d}{N_a (N_a + N_b)^2} = 0,
$$

using $\|\sum a_i\|^2 \approx N_a d$ for independent high-dimensional Gaussians. **Set inclusion alone forces orthogonality.** For a concrete check, take $N_a = 20$, $N_b = 30$, $d = 2048$: both terms in the numerator are ${30 \times 20 \times 2048}$, about 1.23 million,, and they cancel.

### Control 3: the 70% split

To break the inclusion, each synset is re-estimated from its own independent 70% sample, so a child's training words are no longer a subset of its parent's. Now the shuffled curve drops to about −0.2, which the authors rightly call highly nontrivial in this dimension, while the real-unembedding curve stays at zero. That gap is the actual evidence that semantics, and not bookkeeping, produce the right angles.

### Control 4: no whitening

Swap the causal inner product for the naive Euclidean one (center but do not whiten). The orthogonality disappears (Figure 12 in the appendix). The right angles exist only in the right coordinates.

## Results at a glance

| Test | Model | Prediction | Outcome |
| --- | --- | --- | --- |
| Vector representations exist (Fig. 3) | Gemma-2B, 593 noun synsets | held-out ≈ 1, random ≈ 0 | holds for most features; collapses under shuffling |
| Steering isolates the target (Table 1) | Gemma-2B, 3 parent/child pairs | child shift ≈ 0 | means within 0.4 nats; sd around 1.2 |
| Cosine mirrors hierarchy (Fig. 4) | Gemma-2B | siblings similar | holds; siblings orthogonal under shuffle |
| Theorem 8(a), 8(d) (Figs. 5, 10) | Gemma-2B | cosine ≈ 0 | holds, survives the 70% split control |
| Verb hierarchy (Figs. 13 to 16) | Gemma-2B, 364 verb synsets | same | analogous |
| Noun hierarchy (Figs. 17 to 20) | LLaMA-3-8B | same | analogous |
| Euclidean inner product (Fig. 12) | Gemma-2B | n/a | orthogonality fails |

### What might not transfer

The load-bearing assumptions are the final layer, a whitening estimated from the vocabulary, and token sets drawn from a lexical database. Concepts that are not well described by a set of single next tokens (sentiment over a sentence, a persona, a coding style) have no $\mathcal{Y}(w)$ in this sense. Whether the same geometry holds mid-network, where steering and SAEs usually operate, is untested here.

## Critique

### What's strong

- **A theory that earns its predictions.** The orthogonality is derived from a behavioral definition, not observed and then rationalized. Theorem 8 makes four falsifiable statements and the experiments target two of them at scale.
- **Honest controls.** The authors found that their own shuffled baseline also produced orthogonality, explained it in Appendix G, and designed the split control to get around it.
- **Two models, two parts of speech.** Gemma-2B nouns and verbs plus LLaMA-3-8B nouns, all analogous.

### What's weak

- **The intervention test is thin.** Three hand-chosen parent/child pairs carry the claim that steering is target-specific, and the per-pair spread is as big as the parent's. A test across all 593 synsets would have been easy given the setup.
- **"Near zero" is not quantified.** Figure 5 is read by eye. There is no statistic for how close to zero the blue curve is relative to the split-control green curve, or how many synsets fail.
- **The headline figures use ChatGPT-4 token sets**, not WordNet. They are hand-checked, but the most-shared visuals are on curated data.

### What would change my mind

I would doubt the hierarchy-as-orthogonality claim if a larger intervention study (every parent/child pair in WordNet, with token-level statistics) showed child log-odds shifting systematically, or if an internal-layer analogue with a suitable inner product showed no orthogonality at all. Conversely, a demonstration that an SAE trained on the reparameterized basis ($\bar\ell_{\text{animal}}$, $\bar\ell_{\text{mammal}} - \bar\ell_{\text{animal}}$, ...) stops absorbing features would make me much more confident that this geometry is what the model uses.

## Why this matters for sparse autoencoders

The discussion section makes a pointed observation. SAEs look for one latent per binary feature: *animal*, *mammal*, *bird*. By Theorem 8, those vectors are strongly co-linear, because each child contains its parent. The basis that actually separates cleanly is $\bar\ell_{\text{animal}}$, $\bar\ell_{\text{mammal}} - \bar\ell_{\text{animal}}$, $\bar\ell_{\text{bird}} - \bar\ell_{\text{animal}}$, and so on.

That is the geometry behind the failure documented in [A is for Absorption](/blog/paper-reading/ai-interpretability/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders), where a sparse dictionary lets a child latent swallow its parent. [Matryoshka SAEs](/blog/paper-reading/ai-interpretability/learning-multi-level-features-with-matryoshka-sparse-autoencoders) can be read as one engineering response: force general features to be learned first in a nested dictionary. Connecting those two lines of work through this paper's orthogonal decomposition is my synthesis, not a claim the authors test.

For the background on what "linear" means here and where the hypothesis comes from, see [the linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis).

## References

1. Kiho Park, Yo Joong Choe, Yibo Jiang, Victor Veitch. [The Geometry of Categorical and Hierarchical Concepts in Large Language Models](https://arxiv.org/abs/2406.01506). ICLR 2025 (oral). arXiv:2406.01506.
2. Code: [github.com/KihoPark/LLM_Categorical_Hierarchical_Representations](https://github.com/KihoPark/LLM_Categorical_Hierarchical_Representations).
3. Kiho Park, Yo Joong Choe, Victor Veitch. The Linear Representation Hypothesis and the Geometry of Large Language Models. ICML 2024 (the causal inner product this paper builds on).
4. George A. Miller. WordNet: A Lexical Database for English. Communications of the ACM, 1995.
5. On this site: [A is for Absorption](/blog/paper-reading/ai-interpretability/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders), [Matryoshka Sparse Autoencoders](/blog/paper-reading/ai-interpretability/learning-multi-level-features-with-matryoshka-sparse-autoencoders), [The linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis).

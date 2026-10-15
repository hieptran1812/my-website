---
title: "On the Origins of Linear Representations in LLMs: Why Softmax and Gradient Descent Make Concepts Point One Way"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How a toy latent-concept model, the log-odds of softmax, and the implicit bias of gradient descent together explain why concepts like gender or language end up as straight-line directions inside LLMs."
category: "paper-reading"
subcategory: "AI Interpretability"
tags:
  - paper-reading
  - large-language-model
  - linear-representations
  - linear-representation-hypothesis
  - representation-learning
  - interpretability
  - implicit-bias
  - gradient-descent
  - icml-2024-poster
author: "Hiep Tran"
featured: true
readTime: 18
paper:
  title: "On the Origins of Linear Representations in Large Language Models"
  authors: "Yibo Jiang, Goutham Rajendran, Pradeep Ravikumar, Bryon Aragam, Victor Veitch"
  venue: "ICML 2024"
  url: "https://arxiv.org/abs/2403.03867"
---

> [!tldr]
> - **Claim:** linear representations of concepts are not a quirk of the Transformer. They fall out of next-token prediction itself: softmax with cross-entropy, trained by gradient descent.
> - **Mechanism:** the paper builds a latent model where contexts and tokens are both driven by binary "concept" variables, then shows two routes to linearity: matching log-odds (Theorem 3) and the implicit bias of gradient descent toward max-margin directions (Theorems 4 and 5).
> - **Why it matters:** it gives the linear representation hypothesis a mechanism, and it predicts something new: the embedding (context) direction and the unembedding (token) direction for one concept should line up.
> - **Most surprising result:** linearity emerges even when the latent graph and its probabilities are random, so log-odds matching cannot be the whole story. Gradient descent on the "this token is impossible here" constraints does the work.
> - **Where it is thin:** the LLaMA-2 evidence is real but weak (Winograd matched-pair cosine averages 0.042 versus 0.011), and the theory covers a subproblem of training, not the full nonconvex objective.

![Figure 1 from Jiang et al. (2024): the latent conditional model, where a context fixes some concepts and the next token is read off the full concept vector](/imgs/blogs/on-the-origins-of-linear-representations-in-large-language-models-fig1.webp)

The diagram above is the whole model at a glance. A prompt like "The ruler of a kingdom is a" pins down some concepts (royalty) and leaves others open (gender), and the next token is a deterministic readout of the completed concept vector. The rest of this post unpacks why training such a model forces each concept to become a single direction.

## The problem: is "linear" real or an illusion?

For a decade, interpretability work has kept finding that human concepts sit along straight lines in model space. Word2vec gave us king minus man plus woman landing near queen; the background is laid out in [the linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis).

The trouble is that this was almost entirely an empirical observation. Nobody had a clean answer to "why would a nonlinear network trained on text bother to arrange concepts linearly?" The authors put the worry sharply: could the apparent linearity be illusory?

![A concept is linearly represented when every counterfactual pair moves along one direction, in both the token space and the context space](/imgs/blogs/on-the-origins-of-linear-representations-in-large-language-models-1.webp)

The figure above fixes the vocabulary. A **counterfactual pair** is two tokens (or two contexts) that differ in exactly one concept, like "man" and "woman". The difference of their representations is a **steering vector**, and "linear" means every such steering vector for the same concept points the same way.

## Contributions

1. **A latent conditional model** of next-token prediction in which both contexts and tokens are functions of a binary concept vector, so learning an LLM becomes learning conditional probabilities between concepts.
2. **Linearity from log-odds (Theorem 3):** if the model learns the right log-odds for a concept, its steering vectors are parallel inside a subspace. This recovers the classic word-embedding argument.
3. **Linearity from implicit bias (Theorems 4 and 5):** even without log-odds matching, gradient descent on the zero-probability constraints drives steering vectors of one concept to align, and aligns the embedding direction with the unembedding direction. This is the main technical contribution.
4. **Orthogonality (Theorem 7, Corollary 8):** concepts that are separated in the latent graph get orthogonal representations, which helps explain why plain Euclidean geometry carries meaning in LLMs at all.
5. **Experiments:** simulations confirm near-perfect linearity (average cosines of 0.97 to 0.99), and LLaMA-2-7B shows the predicted embedding and unembedding alignment, weakly but consistently.

![The paper's argument map: two independent routes lead to linear representations, and matched linear representations then imply orthogonality](/imgs/blogs/on-the-origins-of-linear-representations-in-large-language-models-2.webp)

## Method

### Technique 1: the latent conditional model

**The problem it solves.** You cannot prove anything about "concepts" in a real LLM because nobody can define them. The authors need a world where concepts are explicit objects, so they can check whether the trained model represents them linearly.

**The mechanism.** Concepts are binary random variables $C_1, \dots, C_m$ that form a Markov random field over a graph $G_C$, so each variable is conditionally independent of the rest given its neighbours. A token $y$ maps injectively to a concept vector $c = h_y(y) \in \{0,1\}^m$. A context $x$ maps to a partial vector $h_x(x) \in \{\diamond, 0, 1\}^m$, where $\diamond$ means "unknown".

The **core concepts** of a context are the ones that are fixed across every concept vector that could have produced it. Formally, with $\mathcal{C}^x$ the set of concept vectors $c$ with $P(x \mid C = c) \neq 0$:

$$
\mathrm{core}(x) = \{\, i \in [m] : c_i = 1 \ \forall c \in \mathcal{C}^x \ \text{ or } \ c_i = 0 \ \forall c \in \mathcal{C}^x \,\}
$$

The modelling choice that makes everything work is that the context only matters through its core concepts:

$$
p(y \mid x) = p(c \mid x) = p\big(c \mid c_{\mathrm{core}(x)}\big), \qquad c = h_y(y)
$$

The LLM learns an embedding $\bar f : \mathcal{X} \to \mathbb{R}^d$ for contexts and an unembedding $\bar g : \mathcal{Y} \to \mathbb{R}^d$ for tokens, and predicts with a softmax over inner products. Because both factor through concepts, write $\bar f = f \circ h_x$ and $\bar g = g \circ h_y$, and the training target becomes:

$$
\hat p(c \mid d) = \frac{\exp\!\big(f(d)^\top g(c)\big)}{\sum_{c'} \exp\!\big(f(d)^\top g(c')\big)}, \qquad c \in \{0,1\}^m,\ d \in \{\diamond,0,1\}^m
$$

Here $d$ is a "conditioning context" (some concepts known, some $\diamond$), $f(d) \in \mathbb{R}^d$ and $g(c) \in \mathbb{R}^d$ are free lookup-table vectors, and $d$ in $\mathbb{R}^d$ is the representation dimension.

**Tiny example.** Take $m = 3$ concepts: royalty, gender, plural. "The ruler of a kingdom is a" maps to $d = (1, \diamond, 0)$. The model must spread probability over $c = (1,0,0)$ ("king") and $c = (1,1,0)$ ("queen"), and assign zero to every vector with royalty $= 0$ or plural $= 1$.

**When it fails.** Injectivity from concepts to tokens is a strong assumption: real tokens are polysemous, and the authors flag this themselves (Remark 9). Binary concepts also cannot represent graded or circular features, a gap later work on [multi-dimensional features](/blog/paper-reading/ai-interpretability/not-all-language-model-features-are-one-dimensionally-linear) makes concrete.

### Technique 2: what "linearly represented" means, formally

Write $c^{(i \to t)}$ for $c$ with its $i$-th coordinate set to $t$. The unembedding steering vector for concept $i$ at base vector $c$ is

$$
\Delta_{c,i} = g\big(c^{(i\to 1)}\big) - g\big(c^{(i\to 0)}\big)
$$

**Definition 2** (adapted from Park et al.) says $C_i$ is linearly represented in the unembedding space if there is one unit vector $u$ with $\Delta_{c,i} \in \mathrm{Cone}(u) = \{\alpha u : \alpha > 0\}$ for every $c$. The same with $f$ and some $v$ gives linearity in the embedding space, and $u = v$ is a **matched representation**.

### Technique 3: linearity from log-odds (Theorem 3)

**The problem.** We want every $\Delta_{c,i}$ for a fixed concept $i$ to be parallel, regardless of the other concepts in $c$.

**The analogy.** If you know a vector's dot product with every axis of a coordinate system, you know the vector. If those dot products are the same for "king minus queen" and "man minus woman", the two vectors are the same vector.

**The mechanism.** Take any context $d$ that does not condition on concept $i$ ($d_i = \diamond$). The softmax normalizer cancels in a ratio, so the log-odds between a counterfactual pair is a single inner product:

$$
\ln \frac{\hat p\big(c^{(i\to 0)} \mid d\big)}{\hat p\big(c^{(i\to 1)} \mid d\big)} = -\, f(d)^\top \Delta_{c,i}
$$

**Theorem 3** assumes the model has learned these log-odds correctly and that they equal the marginal log-odds $\ln \frac{p(C_i = 0)}{p(C_i = 1)}$. Then $f(d)^\top \Delta_{c,i}$ is the same constant for every $c$ and every such $d$. Projecting onto $\mathrm{span}\{f(d) : d_i = \diamond\}$ with the operator $\Pi_i$, the projected steering vectors $\Pi_i \Delta_{c,i}$ are all parallel.

**Tiny example (mine).** Let that span be the plane with basis $f(d_1) = (1,0)$ and $f(d_2) = (0,1)$, and let $p(C_i = 1) = 0.6$. The required inner product is $\ln(0.6/0.4) \approx 0.405$ with both basis vectors. So every projected steering vector equals $(0.405, 0.405)$, whatever the other concepts are.

**When it fails.** The assumption holds exactly when concepts are jointly independent, because then the true distribution is a product. With dependent concepts in a Markov random field, Appendix B shows the steering vectors only land in a subspace whose dimension is at most ${2}^{|\mathrm{ne}(i)|}$, one direction per configuration of the neighbours. And the projection throws away exactly the directions $f(d)$ with $d_i \neq \diamond$, the contexts that do condition on concept $i$, which is where the next technique lives.

### Technique 4: the hidden binary classification (Theorem 4)

**The problem.** Log-odds matching is a stringent requirement, yet in the authors' simulations linearity appears even with random graphs and random probabilities. Something else is pushing.

**The analogy.** Every time a context rules a token out completely ("The ruler of a kingdom is a" can never be followed by a commoner word), the model must drive that token's probability to zero. With a softmax, zero is only reachable at infinity, so the model keeps pushing forever, and the direction it pushes in is a classifier's decision direction.

**The mechanism.** Take three concepts and only the two contexts that fix concept 1: $w_0 = f(0,\diamond,\diamond)$ and $w_1 = f(1,\diamond,\diamond)$. Let $v = g(1,1,1) - g(0,1,1)$ be the concept-1 steering vector. The true conditionals put zero mass on the "wrong" member of the pair, so training wants:

$$
\frac{\hat p(0,1,1 \mid c_1 = 1)}{\hat p(1,1,1 \mid c_1 = 1)} = \exp\!\big(-w_1^\top v\big) \to 0, \qquad
\frac{\hat p(1,1,1 \mid c_1 = 0)}{\hat p(0,1,1 \mid c_1 = 0)} = \exp\!\big(w_0^\top v\big) \to 0
$$

That is exactly the exponential loss of a linear classifier $v$ on two labelled points, $w_1$ with label $+1$ and $w_0$ with label $-1$:

$$
L(v) = \exp\!\big(-w_1^\top v\big) + \exp\!\big(w_0^\top v\big)
$$

On separable data, gradient descent on exponential loss converges in direction to the hard-margin SVM solution (Soudry et al., 2018). That direction depends only on $w_0, w_1$, not on the base vector $c$.

**Theorem 4** generalizes this. Fix $f$ on the two contexts $d_\diamond^{(i\to 1)}$ and $d_\diamond^{(i\to 0)}$, where $d_\diamond = (\diamond,\dots,\diamond)$, and minimize

$$
L = \sum_{c \in \mathcal{C}} \Big[ \exp\!\big(-\Delta_{c,i}^\top f(d_\diamond^{(i\to1)})\big) + \exp\!\big(\Delta_{c,i}^\top f(d_\diamond^{(i\to0)})\big) \Big]
$$

over $g$. Each $\Delta_{c,i}$ is its own copy of the same classification problem, so all of them converge to the same max-margin direction: $\lim_{t\to\infty} \cos(\Delta^t_{c_1,i}, \Delta^t_{c_2,i}) = 1$ for any $c_1, c_2$.

**When it fails.** The theorem uses full-batch gradient descent with a step size small enough; the authors appeal to stochastic approximation theory for SGD, which is a reasonable but unproven bridge here.

### Technique 5: gradient descent aligns embeddings with unembeddings (Theorems 5 and 12)

**The problem.** Theorem 4 froze the embeddings. In a real model $f$ and $g$ train together, and the paper wants a stronger, testable claim: the context direction and the token direction for a concept should match.

**The mechanism.** Proposition 11 first shows that if the conditionals are learned to within $\varepsilon$, the inner product between the embedding steering vector and the unembedding steering vector of the same concept must be large, growing without bound as $\varepsilon \to 0$. So the question becomes: what does gradient descent do when it simply maximizes an inner product through an exponential loss? **Theorem 12** answers it for two vectors:

$$
L(u, v) = \exp\!\big(-u^\top v\big), \qquad
u_{t+1} = u_t + \eta\, L_t\, v_t, \qquad
v_{t+1} = v_t + \eta\, L_t\, u_t
$$

Each vector is pulled toward the other. As long as $u_0 \neq -\alpha v_0$ for any $\alpha > 0$ and the step size satisfies $\eta \lt 1/L(u_0, v_0)$, three things hold: (a) $L_t \to 0$, (b) $\lVert u_t \rVert, \lVert v_t \rVert \to \infty$, and (c) $\cos(u_t, v_t)$ increases monotonically.

![Toy run of Theorem 12's dynamics (my numbers, not the paper's): two orthogonal vectors pulled toward each other by an exponential loss end up nearly parallel](/imgs/blogs/on-the-origins-of-linear-representations-in-large-language-models-3.webp)

**Tiny example (my own run).** Start with $u_0 = (1,0)$ and $v_0 = (0,1)$, orthogonal, so $L_0 = 1$, and use $\eta = 0.5$. After one step $u_1 = (1, 0.5)$, $v_1 = (0.5, 1)$ and the cosine jumps from 0 to 0.80. By step 8 the cosine is 0.97 and the loss has fallen to about 0.05, while both norms keep growing.

| Step | $u_t$ | cosine | loss $L_t$ |
| --- | --- | --- | --- |
| 0 | (1.00, 0.00) | 0.00 | 1.00 |
| 1 | (1.00, 0.50) | 0.80 | 0.37 |
| 2 | (1.09, 0.68) | 0.90 | 0.22 |
| 4 | (1.23, 0.90) | 0.95 | 0.11 |
| 8 | (1.39, 1.10) | 0.97 | 0.05 |

Here is the same thing as code you can paste into a REPL:

```python
import math
u, v, eta = [1.0, 0.0], [0.0, 1.0], 0.5
for t in range(9):
    dot = u[0] * v[0] + u[1] * v[1]
    L = math.exp(-dot)                       # exponential loss on the inner product
    cos = dot / (math.hypot(*u) * math.hypot(*v))
    print(t, round(cos, 3), round(L, 3))
    # simultaneous update: each vector moves toward the other
    u, v = [a + eta * L * b for a, b in zip(u, v)], [b + eta * L * a for a, b in zip(u, v)]
```

**Theorem 5** combines this with Theorem 4. If the steering vectors $\{\Delta_{c,i}\}$ and the two context embeddings start mutually orthogonal with equal norms, jointly training $f$ and $g$ gives both

$$
\lim_{t\to\infty} \cos\big(\Delta^t_{c_1,i}, \Delta^t_{c_2,i}\big) = 1, \qquad
\lim_{t\to\infty} \cos\Big(\Delta^t_{c,i},\ f^t\big(d_\diamond^{(i\to1)}\big) - f^t\big(d_\diamond^{(i\to0)}\big)\Big) = 1
$$

The first limit is linearity in token space; the second is the matched representation $u = v$. Remark 6 notes that Gaussian initialization approximately satisfies the starting condition, and experiments suggest it is unnecessary.

**When it fails.** The theorem covers one concept in isolation. Real training juggles many concepts and many contexts at once, and the paper's own Figure 2 shows what happens when the conditioning set is too small.

![Figure 2 from Jiang et al. (2024): unembeddings after training on (a) contexts fixing only concept 1, (b) contexts fixing concepts 1 and 2, (c) every context](/imgs/blogs/on-the-origins-of-linear-representations-in-large-language-models-fig2.webp)

In panel (a), training only on contexts that fix $C_1$ splits the eight unembeddings into two clusters by $C_1$. In panel (b), the clusters follow $C_1$ and $C_2$ jointly, but the two directions are coupled: $g(110) - g(100)$ is parallel to $g(111) - g(101)$ but not to $g(010) - g(000)$. Only the full conditioning set in panel (c) decouples them.

### Technique 6: orthogonality of unrelated concepts (Theorem 7, Corollary 8)

**The problem.** The softmax objective only identifies inner products up to an invertible linear map, so Euclidean geometry should be arbitrary. Yet in LLaMA-2, unrelated concepts look nearly orthogonal under the plain dot product. Why would Euclidean angles mean anything?

**The mechanism.** Say $C_i$ and $C_j$ are separated in the graph $G_C$. The Markov factorization makes the log-odds for concept $i$ unaffected by whether concept $j$ is known, so (assuming positivity and correct learning on the relevant contexts):

$$
g\big(c^{(i\to1)}\big) - g\big(c^{(i\to0)}\big) \ \perp\ f\big(d^{(j\to c_j)}\big) - f\big(d^{(j\to\diamond)}\big)
$$

In words: revealing concept $j$ in the context moves the context embedding in a direction that does not change the odds for concept $i$. **Corollary 8** adds matched representations ($u = v$ from Technique 5) and concludes that the two unembedding steering vectors themselves are orthogonal.

**When it fails.** Remark 9 is honest about it: injectivity may not hold, and matching between embedding and unembedding is only approximate in LLMs. In simulation the authors also see near-orthogonality between concepts that are *not* separated, which the theorem does not explain; their informal account is that large-norm steering vectors with small required inner products are forced toward cosine zero.

![Figure 3 from Jiang et al. (2024): cosine similarities between unembedding steering vectors in LLaMA-2, with same-concept blocks on the diagonal and near-zero off-diagonal values](/imgs/blogs/on-the-origins-of-linear-representations-in-large-language-models-fig4.webp)

## Experiments and results

### Simulations from the latent model

The setup: random DAGs over $m$ concepts, each conditional a Bernoulli whose parameter is drawn uniformly from 0.3 to 0.7. Training uses masked prediction: each sampled binary vector gets a random mask, masked entries are set to $-1$ (the numeric stand-in for $\diamond$), and the model predicts the full vector with cross-entropy. Inputs are one-hot encoded so the network cannot exploit the binary structure, $f$ and $g$ are linear maps, SGD runs at learning rate 0.1 with batch size 100, and the representation dimension equals $m$.

![Table 1 from Jiang et al. (2024): average cosine similarities among unembedding steering vectors, among embedding steering vectors, and between the two, for 3 to 7 concepts](/imgs/blogs/on-the-origins-of-linear-representations-in-large-language-models-fig3.webp)

| Setting | Unembedding | Embedding | Unembedding vs embedding |
| --- | --- | --- | --- |
| SGD, complete conditionals, $m = 3$ | 0.972 | 0.982 | 0.980 |
| SGD, complete conditionals, $m = 7$ | 0.995 | 0.972 | 0.981 |
| Adam, complete conditionals, $m = 3$ | 0.910 | 0.923 | 0.926 |
| Adam, complete conditionals, $m = 7$ | 0.966 | 0.918 | 0.929 |
| Subset of contexts, 10 concepts, 50 masks | 0.974 | 0.946 | 0.959 |
| Subset of contexts, 10 concepts, 100 masks | 0.957 | 0.915 | 0.934 |
| Missing concept vectors, 10 concepts | 0.951 | 0.777 | 0.855 |
| Missing concept vectors, 12 concepts | 0.896 | 0.551 | 0.696 |

Three readings matter. First, with the complete set of conditionals, linearity and matching are near perfect for every $m$ tested. Second, the result survives Adam (learning rate 0.001), subsets of contexts, and representation dimensions smaller than $m$ (their Figure 6 shows only a mild drop). Third, when some concept vectors have zero probability, unembedding linearity holds but embedding alignment falls to 0.551 at 12 concepts.

The authors offer two explanations for that drop: undertraining as the problem grows exponentially, and the new zero probabilities acting as extra, misaligned "forces" in the sense of Technique 4.

![Figure 7 from Jiang et al. (2024): the same block pattern of aligned same-concept steering vectors and near-orthogonal different-concept vectors in (a) simulation and (b) LLaMA-2](/imgs/blogs/on-the-origins-of-linear-representations-in-large-language-models-fig6.webp)

### LLaMA-2-7B

Prior work already showed linear token directions in LLMs. What is new here is the **embedding versus unembedding** comparison, the prediction of Theorem 5. Constructing counterfactual context pairs is hard, so the authors reuse two existing datasets.

**Multilingual contexts.** From OPUS Books they take four language pairs (French and Spanish, French and German, English and French, German and Spanish), 150 random samples, keeping contexts between 20 and 150 tokens. The embedding steering vector is the average difference between the same sentence in two languages; the unembedding steering vectors are the 27 token concepts from Park et al., built on the BATS analogy set. They report absolute cosine similarity between the two.

![Figure 4 from Jiang et al. (2024): cosine similarity between the French to Spanish context direction and the top 10 token concepts in LLaMA-2-7B](/imgs/blogs/on-the-origins-of-linear-representations-in-large-language-models-fig5.webp)

The language-translation token concepts sit at the top of the ranking, ahead of morphology concepts such as verb to third-person singular. The y-axis tops out around 0.35, so this is "clearly above the rest", not "aligned" in the simulation sense.

**Winograd Schema.** These are sentence pairs that differ in one or two words, where the embedding vector is the difference at the first differing token. Matching context and token pairs average 0.042 cosine (max 0.161), against 0.011 (max 0.081) for non-matching ones. The best single hit is "Anne gave birth to a daughter... charming woman/baby" aligning with the male to female concept at 0.311.

| LLM experiment | Matched | Non-matched |
| --- | --- | --- |
| Winograd, average cosine | 0.042 | 0.011 |
| Winograd, max cosine | 0.161 | 0.081 |

### What is load-bearing and might not transfer

The simulations use lookup-table embeddings, $d = m$, and complete or near-complete conditional sets, all chosen to make the theory's assumptions true. A real LLM has $d \ll$ the number of concepts, contextual embeddings computed by a deep network, and a concept-to-token map that is anything but injective. The 0.97 cosines are the theory confirming itself in its own world; the 0.04 cosines are what is left after the world gets messy.

## Critique

**What is strong.** The paper names a concrete, architecture-free cause for linearity, and the "zero probabilities are a hidden max-margin classifier" observation is genuinely clarifying. And Theorem 5 makes a new, falsifiable prediction about embeddings matching unembeddings, which is more than most "why linear" papers offer.

**What is weak.**
- The LLM evidence is thin. Average Winograd cosines of 0.042 versus 0.011 are a ratio of about four, but tiny in absolute terms, and there is no significance test or shuffled-pair baseline reported.
- The theorems analyze isolated subproblems (one concept, frozen or specially initialized embeddings). Nothing bounds how interference between thousands of concepts in $d \ll m$ dimensions degrades the result, which is precisely where [superposition](/blog/machine-learning/ai-interpretability/what-is-superposition) lives.
- Every concept is binary. Calendar-like or magnitude-like features are outside the model by construction.
- Orthogonality in simulation appears even for non-separated concepts, so Theorem 7 under-explains the very effect it is cited for.

**Missing ablation.** The cleanest test of route B would be to remove zero-probability constraints (smooth every target away from zero) and show linearity collapses in simulation. That experiment would separate "gradient descent on zeros" from everything else, and it is not in the paper.

**What would change my mind.** If label-smoothed or temperature-floored training (no exact zeros, ratios bounded away from zero) produced the same linearity and matching in the simulation, the implicit-bias story would be in trouble and log-odds structure would be doing the real work. Conversely, a shuffled-pair baseline in LLaMA-2 that matched the 0.042 figure would erase the empirical support for Theorem 5.

## What I'd build with this

These are my extrapolations, not the paper's claims.

1. **A label-smoothing probe.** Re-run the simulation with label smoothing at several strengths and plot average steering cosine against smoothing. If the theory is right, cosines should fall as the targets move away from zero.
2. **A matched-direction steering tool.** If context-side and token-side directions match, you can estimate a steering vector from unembedding pairs (cheap, from a word list) and apply it in the residual stream, then measure how much transfer you lose versus a context-derived vector.
3. **Testing the categorical extension.** The follow-up on [categorical and hierarchical concepts](/blog/paper-reading/ai-interpretability/the-geometry-of-categorical-and-hierarchical-concepts-in-large-language-models) generalizes the geometry; the same latent model with non-binary concepts would tell us whether polytopes also emerge from the zero-probability mechanism.

## References

1. Yibo Jiang, Goutham Rajendran, Pradeep Ravikumar, Bryon Aragam, Victor Veitch. [On the Origins of Linear Representations in Large Language Models](https://arxiv.org/abs/2403.03867). ICML 2024. arXiv:2403.03867.
2. Kiho Park, Yo Joong Choe, Victor Veitch. The Linear Representation Hypothesis and the Geometry of Large Language Models. 2023 (source of the 27 token concepts and the counterfactual-pair framing).
3. Daniel Soudry et al. The Implicit Bias of Gradient Descent on Separable Data. 2018 (the max-margin convergence result behind Theorem 4).
4. On this blog: [The linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis), [What is superposition](/blog/machine-learning/ai-interpretability/what-is-superposition), [Not all language model features are one-dimensionally linear](/blog/paper-reading/ai-interpretability/not-all-language-model-features-are-one-dimensionally-linear), and [The geometry of categorical and hierarchical concepts in LLMs](/blog/paper-reading/ai-interpretability/the-geometry-of-categorical-and-hierarchical-concepts-in-large-language-models).

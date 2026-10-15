---
title: "Token Embeddings Violate the Manifold Hypothesis: A Hypothesis Test for the Shape of an LLM's Vocabulary"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How Robinson, Dey and Chiang turn 'count the tokens inside a growing ball' into a statistical test for manifolds and fiber bundles, and why GPT2, Llemma7B, Mistral7B and Pythia6.9B all fail it."
tags:
  - paper-reading
  - embeddings
  - manifold-hypothesis
  - fiber-bundle
  - intrinsic-dimension
  - representation
  - large-language-model
  - mechanistic-interpretability
  - neurips-2025
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 19
paper:
  title: "Token Embeddings Violate the Manifold Hypothesis"
  authors: "Michael Robinson, Sourya Dey, Tony Chiang"
  venue: "NeurIPS 2025, arXiv 2504.01002"
  url: "https://arxiv.org/abs/2504.01002"
---

> [!tldr]
> **Claim.** The input token embeddings of GPT2, Llemma7B, Mistral7B and Pythia6.9B are not a manifold, and three of the four are not even a fiber bundle (a manifold-like space allowed one extra scale of structure).
> **How.** For each token, sort its distances to every other token, read "how many tokens fit in a ball of radius $r$" as a volume, and test whether the log-log slope (the local dimension) changes in a way a smooth space forbids.
> **Most surprising.** Llemma7B and Mistral7B share an identical vocabulary, yet their irregular tokens and dimension profiles look nothing alike. Llemma7B's small-radius median dimension is 4096, Mistral7B's is 48.
> **Where it fails.** The test can reject smoothness but never tells you what kind of singularity you found, and its tuning knobs ($v_{\min}$, $v_{\max}$, the reach) were set by inspecting a sample of tokens.

Most of us carry a picture of embeddings in our heads: a smooth, curved sheet floating in a high-dimensional space, with similar words near each other. UMAP plots reinforce that picture. So does almost every "the model learns a low-dimensional manifold" sentence in the literature.

This paper asks a blunt question: is that picture testable, and if so, does it survive the test? The answer is a statistical procedure that needs nothing but a distance matrix, and a fairly clear "no" for every model they tried.

## Why the shape of the token space matters

The first layer of an LLM is a lookup table. Each of the $n$ tokens in the vocabulary maps to a vector in $\mathbb{R}^{\ell}$, where $\ell$ is the embedding width (768 for GPT2, 4096 for the 7B models here). The paper calls the set of those $n$ points the **token subspace**.

Plenty of downstream reasoning quietly assumes this point cloud sits on or near a smooth manifold: dimension estimators, UMAP, geodesic distances, even the idea that a small nudge in latent space is a small change in meaning. If the assumption is false at some tokens, the authors argue, those tokens are places where small changes in input can produce large changes in behavior.

The motivating evidence comes from earlier work by the same group. Estimating local dimension token by token in GPT2 gives a histogram with at least three distinct peaks.

![Figure 1 from Robinson et al. (2025): histogram of local dimension estimates near GPT2 tokens, with peaks for numerics and calendar words, single English words, and word fragments](/imgs/blogs/token-embeddings-violate-the-manifold-hypothesis-fig1.webp)

A connected manifold has one dimension everywhere. A histogram with peaks near very different dimensions is already a red flag, unless the space is badly disconnected. What was missing was a rigorous test that says, token by token, "this neighborhood is not smooth, at this significance level."

## Three definitions you need

The paper leans on three pieces of geometry. Each one is short.

| Term | Plain meaning | Example |
| --- | --- | --- |
| Manifold | Every point has a neighborhood that looks like flat $\mathbb{R}^d$, same $d$ everywhere (if connected) | Hollow sphere in $\mathbb{R}^3$, $d$ = 2 |
| Singularity | A point where that fails | Tip of a cone, edge of a disk, where a stick meets a ball |
| Fiber bundle | Locally a product $B \times F$: a base manifold $B$ times a fiber $F$ (a manifold with boundary) | A narrow strip: long direction is the base, short width is the fiber |
| Reach $\tau$ | Largest distance from the space at which every point still has a unique nearest point on it | Convex set: $\tau = \infty$. Cusp: $\tau = 0$ |

Reach plays the role curvature normally plays. Small reach means the space bends sharply, so anything you measure at radii above the reach is not trustworthy. Both tests only look at radii below an estimated reach.

The fiber bundle is the key generalization. It lets a space look one-dimensional from far away and two-dimensional up close, as long as that pattern is consistent. Think of signal at long scales and noise at short scales: the paper's appendix frames it exactly that way.

## Theorem 1: volume grows like a power of radius

### The problem

We cannot see a 4096-dimensional point cloud. We need a number we can compute at each token that would behave in a known way if the space were smooth.

### The analogy

Stand at a point on a flat floor and draw circles of growing radius. The area grows like $r^2$. On a long thin hallway, the area you can reach grows like $r^2$ until the circle hits the walls, then roughly like $r$, because you can only grow along the hallway. The exponent tells you the dimension you "feel" at that scale.

![A strip in the plane is a fiber bundle: slope 2 while the ball fits inside, slope 1 once it sticks out both sides](/imgs/blogs/token-embeddings-violate-the-manifold-hypothesis-3.webp)

### The math

For a $d$-dimensional Riemannian manifold, the volume $v$ of a ball of radius $r$ around any point is

$$
v = K r^{d} + (\text{correction terms})\, r^{d+1},
$$

where $K > 0$ depends only on $d$ and large correction terms mean high curvature. Taking logs for small $r$:

$$
\log v \approx \log K + d \log r .
$$

So on a log-log plot of volume against radius, a manifold gives a straight line with slope $d$.

Theorem 1 extends this to a fiber bundle $p : T \to S$, where $T$ is the total space (the tokens) and $S$ is the base. With $e$ the embedding into $\mathbb{R}^{\ell}$, $e_* v$ the volume pushed into the latent space, $B_r(\psi)$ the ball of radius $r$ at token $\psi$, and $\rho(\psi) \in [0, \tau]$ the largest radius at which the ball still fits inside the space:

$$
(e_* v)\big(B_r(\psi)\big) =
\begin{cases}
O\!\left(r^{\dim T}\right) & 0 \le r \le \rho(\psi), \\[4pt]
(e_* v)\big(B_{\rho(\psi)}(\psi)\big) + O\!\left((r - \rho(\psi))^{\dim S}\right) & \rho(\psi) \le r .
\end{cases}
$$

Read it as: slope $\dim T$ while the ball is inside, slope $\dim S$ after it sticks out. Since $\dim S \le \dim T$, **the slope can only go down as $r$ grows**. The proof sketch is exactly the hallway: a ball that sticks out of the space captures less volume than it would inside, so the curve bends down, never up.

### A worked micro-example (mine, not the paper's)

Take a strip of width 1 in the plane. At $r$ = 0.1 and $r$ = 0.2 the disk sits inside, so the area is $\pi r^2$: 0.0314 then 0.1257. The slope is $\ln(0.1257 / 0.0314) / \ln 2 = \ln 4 / \ln 2$ = 2.

At $r$ = 10 and $r$ = 20 the disk spills out both sides, so the area is roughly width times length, about 20 then 40. The slope is $\ln 2 / \ln 2$ = 1. One space, two slopes, and the drop is allowed.

### When it fails

The theorem only holds for $r$ below the reach. Above it, a perfectly smooth but tightly curved surface can produce any slope pattern. That is why the choice of radii matters so much later.

## Reading the two tests off one curve

Theorem 1 hands us two different null hypotheses from one log-log curve per token:

- **Manifold test.** $H_0$: there is a unique dimension at $\psi$, so the slope is constant. Any slope change rejects.
- **Fiber bundle test.** $H_0$: the dimension at $\psi$ does not increase as $r$ grows. Only a slope *increase* rejects.

![Figure 2 from Robinson et al. (2025): the fiber bundle test detects slope increases, while the manifold test detects any slope change](/imgs/blogs/token-embeddings-violate-the-manifold-hypothesis-fig2.webp)

The asymmetry is the whole design. A downward kink is what a boundary or a multi-scale bundle looks like, so it breaks the manifold null but not the bundle null. An upward kink is something no fiber bundle can produce.

![Three curve shapes, two verdicts: a fiber bundle may bend down, never up](/imgs/blogs/token-embeddings-violate-the-manifold-hypothesis-2.webp)

A note on logic the authors stress: failing to reject is uninformative. A token whose curve looks clean has not been shown to sit on a manifold. A rejection at one token is enough to say the whole space is not a manifold, because the definition is global.

## Algorithm 1: counting tokens in growing balls

### The problem

Volume in 4096 dimensions sounds impossible to measure. The trick is that we do not need it in absolute units, only how it scales.

### The mechanism

The paper uses a Monte Carlo estimate: the volume of a ball is proportional to the number of tokens inside it. That turns the whole thing into sorting.

1. Compute the $n \times n$ Euclidean distance matrix $D$ between all token embeddings.
2. For each token (a column of $D$), sort the column. Now the row index $k$ is a volume (the number of tokens within that radius) and the entry is the radius $r_{(k)}$.
3. Keep rows $v_{\min}$ through $v_{\max}$, the window of neighbor counts you trust.
4. Estimate the slope of $\log k$ against $\log r_{(k)}$ at every row with a 3-point centered difference (`numpy.gradient`). Each slope is a local dimension estimate.
5. Slide two adjacent windows of $W$ slopes along the column and run a two-sample t-test between them.
6. Apply Holm-Bonferroni correction across all tokens.

![Algorithm 1 end to end: distances become radii, ranks become volumes, log-log slopes become dimension, and sliding-window t-tests flag slope changes](/imgs/blogs/token-embeddings-violate-the-manifold-hypothesis-1.webp)

### The math

The dimension estimate at row $k$ is the local log-log slope:

$$
\hat{d}_k = \frac{\Delta \log k}{\Delta \log r_{(k)}} \approx \frac{\log (k+1) - \log (k-1)}{\log r_{(k+1)} - \log r_{(k-1)}} .
$$

Here $k$ is the neighbor count and $r_{(k)}$ is the distance to the $k$-th nearest token. For two adjacent windows $A$ and $B$, each holding $W$ slope estimates with means $\bar{d}_A, \bar{d}_B$ and variances $s_A^2, s_B^2$, the statistic is the standard two-sample form:

$$
t = \frac{\bar{d}_B - \bar{d}_A}{\sqrt{s_A^2 / W + s_B^2 / W}} .
$$

The manifold test uses a two-sided p-value ("the slope is constant"). The fiber bundle test uses a one-sided one ("the slope decreases with row index"), so only $\bar{d}_B > \bar{d}_A$ counts against it. Holm-Bonferroni then sorts the $n$ p-values ascending and rejects the $j$-th smallest only if $p_{(j)} \lt \alpha / (n - j + 1)$, stopping at the first failure. The paper's tables use $\alpha = 10^{-3}$.

### A worked micro-example (mine)

Suppose a token has 10 neighbors within radius 1.0 and 80 within radius 2.0. Then $\hat{d} = \ln 8 / \ln 2$ = 3: the neighborhood feels three-dimensional at that scale.

Now the high-dimensional case. If $d$ = 389 (GPT2's median at small radius), doubling the neighbor count needs the radius to grow by a factor of ${2}^{1/389} \approx 1.0018$. A 0.18% change in distance doubles the count. That is why Figure 6 later shows all the action happening between radius 3.6 and 4.0: in high dimensions, everything lives at nearly the same distance.

### Cost and when it fails

The whole run took about 12 hours per model on a single Intel Core i7-3820 with 32 GB RAM and no GPU, almost all of it in the dense distance matrix. The stages after that take minutes. My own arithmetic for scale: GPT2's 50,257 tokens give about 2.5 billion pairwise distances, roughly 20 GB in float64, which is why this is a CPU-and-patience job rather than a GPU one.

The fragile parts are the knobs. $v_{\min}$ and $v_{\max}$ decide which scales you look at, and the authors chose them per model by inspecting a small random sample of tokens. A window that is too wide can contain two kinks; a window that is too narrow sees none.

## Does the test work? Three synthetic spaces

Before touching an LLM, the authors run the manifold test on shapes where the answer is known.

![Figure 3 from Robinson et al. (2025): the test applied to (a) a sphere, (b) a cusp surface, (c) a lollipop, and (d) the neighborhood of the token ember in Mistral7B, darker means lower p-value](/imgs/blogs/token-embeddings-violate-the-manifold-hypothesis-fig3.webp)

| Space | True structure | What the test found |
| --- | --- | --- |
| Unit sphere in $\mathbb{R}^3$ | Manifold | p close to 1 everywhere, no rejections of either test |
| Cusp surface $z^8 + x^2 + y^2 = 0$, $x^2 + y^2 \le 1$ (1200 points, 41 in cusp or boundary) | Fiber bundle, cusp at origin | Cusp detected, $p \lt 5 \times 10^{-8}$; boundary missed ($p > 0.001$); fiber bundle test never rejects |
| Lollipop: stick attached to a hollow sphere (3000 points) | Neither | Manifold rejected near stick end (75 points) and near the junction (58 points), $p \lt 10^{-5}$; fiber bundle rejected at 2 points near the junction |

The lollipop is the instructive one. The end of the stick is a boundary, which bends the curve down, so only the manifold test fires. The junction is where 1D meets 2D: a ball centered on the stick first sees a line, then suddenly the sphere, so the slope jumps up. That is exactly what the fiber bundle test catches.

The cusp row also shows the sensitivity trade-off. A larger $v_{\max}$ would catch the boundary, but the authors note it would make the test worse at pinpointing *which* points are singular.

## Results on four open LLMs

For each model they picked two $(v_{\min}, v_{\max})$ pairs, one small-radius and one large-radius, giving three tests: manifold, fiber bundle at small radius, fiber bundle at large radius.

![Table 2 from Robinson et al. (2025): manifold and fiber bundle rejections with quartiles of local dimension at small and large radius for GPT2, Llemma7B, Mistral7B and Pythia6.9B](/imgs/blogs/token-embeddings-violate-the-manifold-hypothesis-fig4.webp)

The same numbers in a form you can scan:

| Model | Vocab $n$ | Manifold rejects | Small-radius dim Q1 / Q2 / Q3 | Bundle rejects (small) | Large-radius dim Q1 / Q2 / Q3 | Bundle rejects (large) |
| --- | --- | --- | --- | --- | --- | --- |
| GPT2 | 50,257 | 66 | 20 / 389 / 531 | 12 | 8 / 14 / 32 | 7 |
| Llemma7B | 32,016 | 33 | 4096 / 4096 / 4096 | 0 | 8 / 11 / 14 | 1 |
| Mistral7B | 32,016 | 40 | 9 / 48 / 220 | 1 | 5 / 6 / 9 | 2 |
| Pythia6.9B | 50,254 | 54 | 2 / 108 / 235 | 0 | 2 / 5 / 145 | 0 |

The minimum corrected p-values for the manifold test range from about ${5} \times 10^{-9}$ (Llemma7B) to ${3} \times 10^{-7}$ (Mistral7B). Every model rejects the manifold hypothesis; GPT2, Llemma7B and Mistral7B also reject the fiber bundle hypothesis.

Three things jump out.

- **The rejections are rare.** 66 of 50,257 GPT2 tokens is about 0.13%. The appendix estimates roughly 0.1% of tokens are singular, but argues users probably hit them often.
- **Dimension is all over the place.** GPT2's small-radius interquartile range runs from 20 to 531. Pythia6.9B's large-radius Q3 is 145 against a median of 5.
- **Same vocabulary, different geometry.** Llemma7B (a math-tuned model) and Mistral7B use the identical token set, yet Llemma7B's small-radius dimension saturates at the full 4096 while Mistral7B's median is 48. The authors read this as a training "fingerprint", and warn that prompts touching irregular tokens may not port between the two.

The authors describe fiber bundle rejections as more frequent at large radius and connect that to polysemy. The counts support it for Llemma7B (0 vs 1) and Mistral7B (1 vs 2), but GPT2 goes the other way (12 vs 7), so I would hold this one loosely. Their argument: a word whose meanings live in different neighborhoods should look irregular at the scale where those neighborhoods meet.

## Which tokens break, and why

The appendix lists every violation. The manual summary is telling:

- **GPT2:** irregular tokens are ones that can only start a word.
- **Pythia6.9B:** almost all are word fragments that mean little alone.
- **Llemma7B and Mistral7B:** a mix of both.
- In Llemma7B and Pythia6.9B, tokens with unusually low fiber dimension often contain whitespace or non-printing characters, which the authors read as sensitivity to text layout.

The fiber bundle violations include clear homonyms: "affect" and "monitor" (noun or verb), "wins" (noun, verb, or the start of "winsome"), and "aunder", which becomes "launder" for clothes or for money. The asterisk is singular in Mistral7B, and the discussion gives a nice story for why: it is a footnote marker in prose, multiplication in code, and bold in Markdown. If each language were its own manifold with its own dimension, they would be glued together at tokens like this one.

![Figure 6 from Robinson et al. (2025): log volume versus log radius for the GPT2 tokens #, cent sign and dollar sign, showing no rejection, a manifold rejection, and both rejections](/imgs/blogs/token-embeddings-violate-the-manifold-hypothesis-fig5.webp)

This case study makes the test concrete. All three tokens sit above the $\alpha = 10^{-3}$ cutoff, so they are illustrations, not entries in the tables.

| GPT2 token | Curve shape | Reading |
| --- | --- | --- |
| # | Straight | No rejection; slope gives a dimension of about 53 |
| Cent sign | One kink near radius 3.9 | Manifold rejected, fiber bundle not |
| Dollar sign | Kinks at radii 3.8 and 4.0 | Small-radius kink breaks the manifold; the larger-radius kink is an increase, breaking the fiber bundle |

The paper's Figure 3(d) shows what this looks like in space for Mistral7B, projected to 3D with PCA around the token "ember". The authors point at "ember" and "P" as putative cusps and at ":" and "ly" as pinch points, like the joins in a chain of sausages. Neither shape can be fit by a manifold without extreme curvature.

## Theorem 2: context cannot paper over a singularity

### The problem

Perhaps singular tokens do not matter, because the transformer sees tokens in context and can smooth things out. Theorem 2 argues it cannot, at least not generically.

### The mechanism

Two steps. First, Lemma 1: if $X$ is a manifold and $X \times Y$ is a manifold, then $Y$ must be a manifold. Contrapositive: if the token space $T$ is not a manifold, then even if $T^{w-1}$ happened to be one, $T^{w}$ cannot be. Lengthening the window never persistently removes the singularity.

Second, an embedding result from the authors' earlier work: under the right size conditions, a generic transformer topologically embeds the space of whole context windows $T^{w}$ into its output. An embedding preserves topology, so the singularities come along.

![Theorem 2 as a chain: a singular token space, a long enough response, and a generic transformer let the singularity leak into the output](/imgs/blogs/token-embeddings-violate-the-manifold-hypothesis-4.webp)

### The math

Let $Z$ be a bounding manifold of dimension $d$ with $T \subseteq Z \subseteq \mathbb{R}^{\ell}$, let $w$ be the context window and $m$ the number of output tokens collected. If

$$
m > \frac{2 w d}{\ell} \quad \text{and} \quad w \ge m ,
$$

then a generic set of transformers yields a topological embedding of $T^{w}$ into the LLM's output. The proof just rearranges ${2}wd \lt m\ell \le w\ell$. Such an $m$ exists whenever ${2}d \lt \ell$, that is, whenever the bounding manifold is less than half the width of the latent space.

### Worked example: re-deriving Table 3

Plug in the medians from Table 2. For GPT2 at small radius, $d$ = 389, $w$ = 1024, $\ell$ = 768: ${2} \cdot 1024 \cdot 389 / 768 \approx 1037.3$, so $m$ must be at least 1038, which exceeds $w$. The conditions fail.

| Model | $\ell$ | $w$ | Min $m$ (small $d$) | Satisfied? | Min $m$ (large $d$) | Satisfied? |
| --- | --- | --- | --- | --- | --- | --- |
| GPT2 | 768 | 1024 | 1038 | No | 38 | Yes |
| Llemma7B | 4096 | 4096 | 8193 | No | 23 | Yes |
| Mistral7B | 4096 | 4096 | 97 | Yes | 13 | Yes |
| Pythia6.9B | 4096 | 4096 | 217 | Yes | 11 | Yes |

I recomputed each cell from the formula and they match the paper. With large-radius dimensions, all four models satisfy the theorem. With the stricter small-radius dimensions, only Mistral7B and Pythia6.9B do, so the authors' claim that singularities "will persist" is firmest for those two.

### When it fails

"Generic" is doing heavy lifting. It means "true for all transformers except a measure-zero set", justified informally by the randomness of training. The theorem says the singularity is present in the output space; it does not say how often a user's prompt lands near one, or how large the resulting instability is. The paper does not run that experiment.

## Critique

**What is strong.**

- It converts a vague worry ("embeddings might not be smooth") into a falsifiable, per-token test with a clear null, a multiple-testing correction, and synthetic controls with known answers.
- The fiber bundle null is a smart middle ground. It forgives the multi-scale structure that real data almost always has, so its rejections are harder to dismiss as "just noise at small scales".
- The method is cheap and deterministic: one distance matrix, no training, no GPU. Anyone with open weights can rerun it.

**What is weak or unresolved.**

- **Knob selection.** $v_{\min}$, $v_{\max}$ and the reach estimate were chosen by eyeballing a random sample of tokens, partly "with the aim of identifying where changes in dimension were likely to occur". That is close to tuning the test toward rejection. A sensitivity sweep over these knobs is the missing ablation.
- **Reach is assumed, not measured.** The limitations section admits reach is hard to estimate. If the true reach is smaller than assumed, a smooth but sharply curved manifold could produce the same kinks. The authors frame this as "either singular or high curvature, both bad", which is fair for stability but weakens the literal title.
- **Llemma7B's 4096.** A median small-radius dimension equal to the full embedding width looks more like the estimator saturating (points nearly equidistant) than a measurement. The paper does not discuss it.
- **No behavioral link.** The abstract's claim that prompts with implicated tokens will "exhibit less stability" is argued from theory and prior literature, not measured here.
- **Only input embeddings.** Tied or untied output embeddings and intermediate layers are untested.

**What would change my mind.** If a sweep over $v_{\min}$, $v_{\max}$ and $W$ showed the same tokens rejecting across most settings, I would treat the singular-token lists as robust features of the models. If, instead, a prompt-perturbation study found that swapping a rejected token for a synonym changes outputs no more than swapping a non-rejected one, I would read the result as true geometry with little practical bite.

## What this means if you work with embeddings

This is my extrapolation, not the paper's. Three practical takeaways:

- **Treat local dimension as a per-token property.** Global intrinsic-dimension numbers average over neighborhoods that differ by two orders of magnitude.
- **Be careful with geodesic or UMAP-style reasoning on raw token embeddings.** The paper's discussion notes that geodesic distance may not track semantic distance near singular tokens.
- **The test is general.** The authors say their main contribution is a method for testing embedded points, not an LLM fact. It applies to any embedding table or activation dataset you can fit a distance matrix for.

If you want the complementary view, where concepts *are* well-behaved linear directions, see the [linear representation hypothesis explainer](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis) and the analysis of [categorical and hierarchical concept geometry](/blog/paper-reading/ai-interpretability/the-geometry-of-categorical-and-hierarchical-concepts-in-large-language-models). The two views are not in conflict: a space can be mostly linear in meaningful directions and still have a few hundred badly behaved points. For the tokenizer side of the story, which explains why "aunder" is a token at all, see [the BPE tokenizer deep dive](/blog/machine-learning/large-language-model/bpe-tokenizer).

## References

1. Michael Robinson, Sourya Dey, Tony Chiang. [Token Embeddings Violate the Manifold Hypothesis](https://arxiv.org/abs/2504.01002). NeurIPS 2025, arXiv:2504.01002.
2. Robinson, Dey, Sweet. The structure of the token space for large language models (2024). The source of the GPT2 local dimension histogram (Figure 1).
3. Alfred Gray. The volume of a small geodesic ball of a Riemannian manifold (1974). The volume expansion behind Equation (1).
4. Related posts on this site: [Linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis), [The geometry of categorical and hierarchical concepts in LLMs](/blog/paper-reading/ai-interpretability/the-geometry-of-categorical-and-hierarchical-concepts-in-large-language-models), [BPE tokenizer](/blog/machine-learning/large-language-model/bpe-tokenizer).

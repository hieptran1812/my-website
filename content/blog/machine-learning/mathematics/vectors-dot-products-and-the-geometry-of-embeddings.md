---
title: "Vectors, Dot Products and the Geometry of Embeddings: Reading CLIP's Logits"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "Build vector spaces, span, independence and bases from pictures, see the dot product as a shadow and cosine as pure direction, compute why random high-dimensional vectors are nearly orthogonal, and decode the one line that defines CLIP's contrastive logits."
tags: ["mathematics", "math-for-ai", "linear-algebra", "vectors", "dot-product", "cosine-similarity", "embeddings", "clip", "contrastive-learning", "high-dimensional-geometry"]
category: "machine-learning"
subcategory: "Mathematics"
author: "Hiep Tran"
featured: true
readTime: 18
---

> **Level:** Foundation · **Needs:** [How to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper), [Sets, functions and maps](/blog/machine-learning/mathematics/sets-functions-and-maps-the-grammar-of-ml-math), [Sums, products, einsum and log-sum-exp](/blog/machine-learning/mathematics/sums-products-einsum-and-log-sum-exp) · **Unlocks:** CLIP and contrastive losses, attention scores, embedding retrieval

Figure 3 of the CLIP paper (Radford et al., 2021) is not a diagram but NumPy-like pseudocode, and its heart is one line: `logits = np.dot(I_e, T_e.T) * np.exp(t)`, commented "scaled pairwise cosine similarities". Just above it, each embedding goes through `l2_normalize`. That line trained a model on 400 million image and caption pairs, and every piece of it is linear algebra.

By the end you will read that line symbol by symbol: why the embeddings are normalised, what the transpose does, and why a temperature is needed at all. The series bet holds here too: paper math is a small, closed vocabulary, and each word learned once, with a picture, a derivation, a proof and a numerical check, stops the appendix being a wall.

![Two vectors a = (2, 1) and b = (1, 2) at angle 36.9 degrees, with the shadow of b on a of length 1.79 drawn in green](/imgs/blogs/vectors-dot-products-and-the-geometry-of-embeddings-1.webp)

Two arrows, the angle between them, and the shadow one casts on the other. The rest of the post turns this picture into definitions and shows it still works in 768 dimensions, with one surprise.

## Vectors: arrows and lists are the same thing

**What it is for.** A vector is a way to store "how much of each ingredient" so that adding and scaling make sense.

Two pictures of one object help. An arrow is an instruction: walk 2 steps east and 1 north. A list is the numbers $(2, 1)$. The arrow is for intuition, the list is what your GPU stores; the arrow breaks past three dimensions, the list never does.

Two operations make lists into vectors. Take $a = (2, 1)$ and $b = (1, 2)$, the arrows in the opening figure:

$$
a + b = (2 + 1,\; 1 + 2) = (3, 3), \qquad 2a = (2 \cdot 2,\; 2 \cdot 1) = (4, 2).
$$

Addition walks one arrow then the other; scaling stretches an arrow without turning it. Both work entry by entry.

Formally, a **vector space** over the real numbers is a set $V$ with an addition and a scalar multiplication such that sums and multiples stay inside $V$, and the usual arithmetic rules hold (a zero vector $\mathbf{0}$, negatives, distributivity). The workhorse is $\mathbb{R}^d$, all lists of $d$ reals with entry-wise operations. When a paper writes $x \in \mathbb{R}^{d}$ it means: $x$ is a list of $d$ reals you may add and scale.

In practice $d$ is large: BERT-base embeddings live in $\mathbb{R}^{768}$, and CLIP's ViT-B/32 maps images and captions into a shared $\mathbb{R}^{512}$. The definition does not care.

## Linear combinations and span: everything you can reach by mixing

**What it is for.** A linear combination is a recipe: a little of this vector plus a little of that one. The span is the set of all dishes you can cook from a given pantry.

Paint mixing is the analogy: blue and yellow in different amounts give a family of greens. It breaks on one point: vectors allow negative amounts, so the reachable set is a whole line or plane through the origin, never a wedge.

![Three panels: the span of u is a line, the span of u and 2u is the same line, the span of u and v is the whole plane](/imgs/blogs/vectors-dot-products-and-the-geometry-of-embeddings-2.webp)

Here $u = (1, 0.5)$ and $v = (-0.5, 1)$. Multiples of $u$ sit on one line. Adding ${2}u = (2, 1)$ gains nothing, since it lies on that line. Adding $v$, a genuinely new direction, reaches the whole plane; the right panel marks integer recipes $c_1 u + c_2 v$.

A worked example. Which recipe of $u$ and $v$ gives $(0.5, 1.5)$? Try one of each:

$$
1 \cdot (1, 0.5) + 1 \cdot (-0.5, 1) = (1 - 0.5,\; 0.5 + 1) = (0.5, 1.5).
$$

So coefficients $(1, 1)$ reach it, and every point of the plane has some real pair of coefficients.

Formally, given vectors $v_1, \dots, v_k \in V$, a **linear combination** is $c_1 v_1 + \dots + c_k v_k$ with real **coefficients** $c_i$, and the **span** is the set of all of them:

$$
\operatorname{span}\{v_1, \dots, v_k\} = \Big\{ \sum_{i=1}^{k} c_i v_i \;:\; c_1, \dots, c_k \in \mathbb{R} \Big\}.
$$

In words: the span is everything you can build from the $v_i$ by adding and scaling. It always contains $\mathbf{0}$.

The most quoted linear combination in ML is from word2vec. Mikolov et al. (2013), [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781), report that the vector closest to $\mathrm{vec}(\text{King}) - \mathrm{vec}(\text{Man}) + \mathrm{vec}(\text{Woman})$ is $\mathrm{vec}(\text{Queen})$. Coefficients $(1, -1, 1)$, then a cosine nearest-neighbour search.

## Linear independence: no redundant directions

**What it is for.** Independence tells you whether every vector in your set adds a new direction, or whether one of them is a recipe made from the others.

A taxi analogy: "go north", "go east" and "go northeast" are three instructions, but the third mixes the first two and adds no new destination. The analogy breaks in 768 dimensions, where redundancy cannot be seen, only computed.

![u = (2, 1) and v = (-1, 1) placed tip to tail land exactly on w = (1, 2), so 1u + 1v - 1w = 0](/imgs/blogs/vectors-dot-products-and-the-geometry-of-embeddings-3.webp)

A worked example. Take $u = (2, 1)$, $v = (-1, 1)$ and $w = (1, 2)$. Walk $u$, then walk $v$:

$$
u + v = (2 - 1,\; 1 + 1) = (1, 2) = w, \quad\text{so}\quad 1 \cdot u + 1 \cdot v - 1 \cdot w = (0, 0).
$$

Coefficients $(1, 1, -1)$, not all zero, combine the three into the zero vector. That is dependence.

Formally, vectors $v_1, \dots, v_k$ are **linearly dependent** if there are coefficients $c_1, \dots, c_k$, not all zero, with $\sum_i c_i v_i = \mathbf{0}$. Otherwise they are **linearly independent**. Dependence is equivalent to "some $v_j$ is a combination of the others": pick any $j$ with $c_j \ne 0$ and divide by it.

The figure illustrates a fact proved below: any 3 vectors in $\mathbb{R}^{2}$ are dependent, and more generally any $d + 1$ vectors in $\mathbb{R}^d$ are. An embedding table with 50,000 rows in $\mathbb{R}^{768}$ is massively dependent, which is fine: you want independence in a basis, not a vocabulary.

## Basis and coordinates: a recipe depends on the pantry

**What it is for.** A basis is a smallest pantry that still reaches everything, and coordinates are the amounts in the recipe for a given vector.

The analogy is street directions in two cities, one with a square grid and one with diagonal avenues. The same place gets different directions ("2 east, 1 north" versus "1.5 along the avenue, 0.5 across"). Coordinates belong to the pair (vector, basis), never to the vector alone.

![The vector v = (2, 1) reached by 2e1 + 1e2 in the standard basis and by 1.5b1 + 0.5b2 in the basis b1 = (1, 1), b2 = (1, -1)](/imgs/blogs/vectors-dot-products-and-the-geometry-of-embeddings-4.webp)

A worked example with $v = (2, 1)$. In the **standard basis** $e_1 = (1, 0)$, $e_2 = (0, 1)$, the recipe is $v = 2e_1 + 1e_2$, so the coordinates are the entries themselves. Now take $b_1 = (1, 1)$ and $b_2 = (1, -1)$ and solve $c_1 b_1 + c_2 b_2 = (2, 1)$:

$$
\begin{aligned}
c_1 + c_2 &= 2, \\
c_1 - c_2 &= 1.
\end{aligned}
$$

Adding the two equations gives $2c_1 = 3$, so $c_1 = 1.5$; subtracting gives $2c_2 = 1$, so $c_2 = 0.5$. Check: $1.5 \cdot (1, 1) + 0.5 \cdot (1, -1) = (1.5 + 0.5,\; 1.5 - 0.5) = (2, 1)$. Same arrow, coordinates $(1.5, 0.5)$.

Formally, a **basis** of $V$ is a list $b_1, \dots, b_d$ that is linearly independent and spans $V$. Spanning guarantees a recipe exists; independence makes it unique (subtracting two recipes would give a nontrivial combination equal to zero). The unique numbers $c_i$ with $v = \sum_i c_i b_i$ are the **coordinates** of $v$ in that basis. Every basis of a given space has the same length, called its **dimension**; for $\mathbb{R}^d$ it is $d$.

An embedding's 768 entries are coordinates in whatever basis the network learned; nothing forces axis 17 to mean anything.

## The dot product: shadow and similarity

**What it is for.** The dot product measures how much two vectors point the same way, scaled by how long they are.

The picture is a shadow. Shine a light perpendicular to the line through $a$; $b$ casts a shadow on it. The dot product is the length of $a$ times the signed length of that shadow. The analogy is exact; its only limit is that no lamp shines in 768 dimensions, so there we use the formula.

<figure class="blog-anim">
<svg viewBox="0 0 720 340" role="img" aria-label="Vector b rotates a full turn around the origin while vector a stays fixed; the shadow of b on a grows, shrinks to zero at 90 degrees and turns negative past it, which is the dot product's sign" style="width:100%;height:auto;max-width:760px">
<style>
.v7-ax{stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:5 5}
.v7-circ{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:3 6}
.v7-a{stroke:var(--text-primary,#1f2937);stroke-width:5;stroke-linecap:round}
.v7-b{stroke:var(--accent,#6366f1);stroke-width:5;stroke-linecap:round}
.v7-dr{stroke:var(--text-secondary,#6b7280);stroke-width:2;stroke-dasharray:4 4}
.v7-lbl{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.v7-sm{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.v7-rot{transform-origin:0 0;animation:v7-rot 16s linear infinite}
.v7-bar{transform-origin:0 0;animation:v7-bar 16s linear infinite,v7-col 16s linear infinite}
.v7-foot{animation:v7-foot 16s linear infinite}
.v7-drop{transform-origin:0 0;animation:v7-drop 16s linear infinite}
@keyframes v7-rot{0%{transform:rotate(0deg)}100%{transform:rotate(-360deg)}}
@keyframes v7-bar{0.000%{transform:scaleX(1.0000)}4.167%{transform:scaleX(0.9659)}8.333%{transform:scaleX(0.8660)}12.500%{transform:scaleX(0.7071)}16.667%{transform:scaleX(0.5000)}20.833%{transform:scaleX(0.2588)}25.000%{transform:scaleX(0.0000)}29.167%{transform:scaleX(-0.2588)}33.333%{transform:scaleX(-0.5000)}37.500%{transform:scaleX(-0.7071)}41.667%{transform:scaleX(-0.8660)}45.833%{transform:scaleX(-0.9659)}50.000%{transform:scaleX(-1.0000)}54.167%{transform:scaleX(-0.9659)}58.333%{transform:scaleX(-0.8660)}62.500%{transform:scaleX(-0.7071)}66.667%{transform:scaleX(-0.5000)}70.833%{transform:scaleX(-0.2588)}75.000%{transform:scaleX(-0.0000)}79.167%{transform:scaleX(0.2588)}83.333%{transform:scaleX(0.5000)}87.500%{transform:scaleX(0.7071)}91.667%{transform:scaleX(0.8660)}95.833%{transform:scaleX(0.9659)}100.000%{transform:scaleX(1.0000)}}
@keyframes v7-foot{0.000%{transform:translateX(120.00px)}4.167%{transform:translateX(115.91px)}8.333%{transform:translateX(103.92px)}12.500%{transform:translateX(84.85px)}16.667%{transform:translateX(60.00px)}20.833%{transform:translateX(31.06px)}25.000%{transform:translateX(0.00px)}29.167%{transform:translateX(-31.06px)}33.333%{transform:translateX(-60.00px)}37.500%{transform:translateX(-84.85px)}41.667%{transform:translateX(-103.92px)}45.833%{transform:translateX(-115.91px)}50.000%{transform:translateX(-120.00px)}54.167%{transform:translateX(-115.91px)}58.333%{transform:translateX(-103.92px)}62.500%{transform:translateX(-84.85px)}66.667%{transform:translateX(-60.00px)}70.833%{transform:translateX(-31.06px)}75.000%{transform:translateX(-0.00px)}79.167%{transform:translateX(31.06px)}83.333%{transform:translateX(60.00px)}87.500%{transform:translateX(84.85px)}91.667%{transform:translateX(103.92px)}95.833%{transform:translateX(115.91px)}100.000%{transform:translateX(120.00px)}}
@keyframes v7-drop{0.000%{transform:scaleY(0.0000)}4.167%{transform:scaleY(0.2588)}8.333%{transform:scaleY(0.5000)}12.500%{transform:scaleY(0.7071)}16.667%{transform:scaleY(0.8660)}20.833%{transform:scaleY(0.9659)}25.000%{transform:scaleY(1.0000)}29.167%{transform:scaleY(0.9659)}33.333%{transform:scaleY(0.8660)}37.500%{transform:scaleY(0.7071)}41.667%{transform:scaleY(0.5000)}45.833%{transform:scaleY(0.2588)}50.000%{transform:scaleY(0.0000)}54.167%{transform:scaleY(-0.2588)}58.333%{transform:scaleY(-0.5000)}62.500%{transform:scaleY(-0.7071)}66.667%{transform:scaleY(-0.8660)}70.833%{transform:scaleY(-0.9659)}75.000%{transform:scaleY(-1.0000)}79.167%{transform:scaleY(-0.9659)}83.333%{transform:scaleY(-0.8660)}87.500%{transform:scaleY(-0.7071)}91.667%{transform:scaleY(-0.5000)}95.833%{transform:scaleY(-0.2588)}100.000%{transform:scaleY(-0.0000)}}
@keyframes v7-col{0%,24%{fill:#1971c2}26%,74%{fill:#f08c00}76%,100%{fill:#1971c2}}
@media (prefers-reduced-motion:reduce){.v7-rot,.v7-bar,.v7-foot,.v7-drop{animation:none}}
</style>
<text class="v7-lbl" x="40" y="30">b turns; its shadow on a is the dot product (up to the factor ‖a‖)</text>
<g transform="translate(260 190)">
<line class="v7-ax" x1="-200" y1="0" x2="230" y2="0"/>
<circle class="v7-circ" cx="0" cy="0" r="120"/>
<rect class="v7-bar" x="0" y="-7" width="120" height="14"/>
<line class="v7-a" x1="0" y1="0" x2="180" y2="0"/>
<text class="v7-lbl" x="188" y="-8">a</text>
<g class="v7-foot"><line class="v7-drop" x1="0" y1="0" x2="0" y2="-120"/></g>
<g class="v7-rot"><line class="v7-b" x1="0" y1="0" x2="120" y2="0"/><text class="v7-lbl" x="126" y="-6">b</text></g>
<circle cx="0" cy="0" r="5" fill="var(--text-primary,#1f2937)"/>
</g>
<text class="v7-sm" x="480" y="110">blue bar to the right: a · b &gt; 0</text>
<text class="v7-sm" x="480" y="140">bar length zero: a · b = 0 (90°)</text>
<text class="v7-sm" x="480" y="170">amber bar to the left: a · b &lt; 0</text>
<text class="v7-sm" x="480" y="215">a · b = ‖a‖ ‖b‖ cos θ</text>
<text class="v7-sm" x="40" y="325">the dashed circle is every direction b can point at fixed length</text>
</svg>
<figcaption>Hold a fixed and spin b once around the origin. The bar is the shadow ‖b‖ cos θ that b casts on a's line: it is longest when b points along a, vanishes when b is perpendicular, and points backwards (amber) when the angle passes 90°. Multiply the bar by ‖a‖ and you have the dot product.</figcaption>
</figure>

Under 90° the shadow points along $a$ and the dot product is positive; at 90° it vanishes; past 90° it points backwards and goes negative.

A worked example with the opening figure's $a = (2, 1)$ and $b = (1, 2)$. Multiply matching entries and add:

$$
a \cdot b = 2 \cdot 1 + 1 \cdot 2 = 4.
$$

Geometrically, the lengths are $\lVert a \rVert = \sqrt{2^2 + 1^2} = \sqrt{5} \approx 2.236$ and the same for $\lVert b \rVert$. The shadow of $b$ on $a$ is the point $\frac{a \cdot b}{a \cdot a}\, a = \frac{4}{5}(2, 1) = (1.6, 0.8)$, whose length is $4 / \sqrt{5} \approx 1.789$. Then $\lVert a \rVert \times \text{shadow} = \sqrt{5} \cdot 4/\sqrt{5} = 4$. The recipes agree, and $\cos\theta = 4 / (\sqrt{5}\sqrt{5}) = 0.8$, so $\theta \approx 36.87°$.

Formally, the **dot product** (or standard inner product) of $a, b \in \mathbb{R}^d$ is

$$
a \cdot b = a^{\top} b = \sum_{i=1}^{d} a_i b_i, \qquad \lVert a \rVert = \sqrt{a \cdot a}.
$$

Here $a_i$ is the $i$-th entry, $a^{\top} b$ writes it as row times column, and $\lVert a \rVert$ is the Euclidean length. The geometric form is

$$
a \cdot b = \lVert a \rVert \, \lVert b \rVert \cos\theta,
$$

where $\theta$ is the angle between $a$ and $b$. In the plane this is a theorem (from the law of cosines; the numerical check verifies it). In $\mathbb{R}^d$ for $d \gt 3$ it defines the angle: $\cos\theta = a \cdot b / (\lVert a \rVert \lVert b \rVert)$. That definition is legal because the Cauchy–Schwarz inequality from [the inequality toolkit](/blog/machine-learning/mathematics/the-inequality-toolkit-every-appendix-uses) guarantees the ratio lies in $[-1, 1]$.

The sign reads directly: $a \cdot b \gt 0$ is acute, $a \cdot b = 0$ means **orthogonal**, $a \cdot b \lt 0$ is obtuse. [Norms, inner products and distances](/blog/machine-learning/mathematics/norms-inner-products-and-distances) generalises this; [orthogonality, projections and QR](/blog/machine-learning/mathematics/orthogonality-projections-and-qr) makes the shadow a matrix.

### Cosine similarity versus the raw dot product

**What it is for.** Cosine similarity keeps only the direction part of the dot product, so a long vector cannot win a comparison just by being long.

An analogy: the dot product hears volume, cosine hears content. A key with a large norm scores high against almost any non-perpendicular query, which is a feature only if norm carries meaning in your model.

![Query q = (0.9, 1.2) with keys k1 = (2.5, 0) and k2 = (0.5, 0.5): the dot product ranks k1 first, cosine ranks k2 first](/imgs/blogs/vectors-dot-products-and-the-geometry-of-embeddings-5.webp)

A worked example. The query is $q = (0.9, 1.2)$, with $\lVert q \rVert = \sqrt{0.81 + 1.44} = 1.5$. Key $k_1 = (2.5, 0)$ is long and off to the side; key $k_2 = (0.5, 0.5)$ is short and almost aligned:

$$
\begin{aligned}
q \cdot k_1 &= 0.9 \cdot 2.5 + 1.2 \cdot 0 = 2.25, & \cos_1 &= \tfrac{2.25}{1.5 \cdot 2.5} = 0.60,\\
q \cdot k_2 &= 0.9 \cdot 0.5 + 1.2 \cdot 0.5 = 1.05, & \cos_2 &= \tfrac{1.05}{1.5 \cdot 0.7071} = 0.99.
\end{aligned}
$$

The dot product prefers $k_1$; cosine prefers $k_2$ (an angle of 8.1° against 53.1°). Same vectors, opposite rankings.

Formally, **cosine similarity** is

$$
\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert} = \hat a \cdot \hat b, \qquad \hat a = \frac{a}{\lVert a \rVert}.
$$

In words: rescale both to length one, then take the dot product; $\hat a$ is the unit vector along $a$. That is all CLIP's `l2_normalize` does: afterwards the dot product is the cosine.

Transformer attention uses raw $q \cdot k$, so a key's norm acts as learned loudness; retrieval systems usually normalise so no document dominates through length. When reading a paper, check which one it uses.

### Random vectors in high dimensions are nearly orthogonal

**What it is for.** This fact tells you what a cosine of 0.2 means in 768 dimensions: compared with random noise, it is a strong signal.

Low-dimensional intuition misleads. Two random arrows in the plane can meet at any angle; in 768 dimensions they land within a few degrees of 90° almost every time. The analogy: each coordinate adds a coin-flip plus or minus to the dot product, and many independent signs mostly cancel.

![Densities of the cosine between two random unit vectors for d = 3 (flat), d = 10 and d = 100 (sharp peak at 0 with standard deviation 0.1)](/imgs/blogs/vectors-dot-products-and-the-geometry-of-embeddings-6.webp)

The computation: let $x, y \in \mathbb{R}^d$ have independent entries with mean 0 and variance 1. Each product $x_i y_i$ has mean $\mathbb{E}[x_i]\mathbb{E}[y_i] = 0$ and variance $\mathbb{E}[x_i^2]\,\mathbb{E}[y_i^2] = 1 \cdot 1 = 1$. Summing $d$ independent terms,

$$
\mathbb{E}[x \cdot y] = 0, \qquad \operatorname{Var}(x \cdot y) = \sum_{i=1}^{d} \operatorname{Var}(x_i y_i) = d.
$$

So the dot product has typical size $\sqrt{d}$, while each length is about $\sqrt{d}$ (since $\lVert x \rVert^2$ sums $d$ terms of mean 1). The cosine has typical size $\sqrt{d} / (\sqrt{d}\sqrt{d}) = 1/\sqrt{d}$.

For $d = 100$ that is 0.1; for $d = 768$ it is about 0.036, an angle within about 2° of 90°. For uniformly random directions it is exact: the cosine has mean 0 and variance ${1}/d$, with the density plotted above (stated without proof; it is the standard marginal of the uniform distribution on the sphere). For $d = 3$ the density is flat, so low dimensions are the special case.

Two consequences. A cosine of 0.3 between 768-dimensional embeddings is about eight standard deviations above chance: real structure. And a $d$-dimensional space holds far more than $d$ nearly orthogonal directions, one reason a model can represent more features than dimensions.

## CLIP's logits, decoded

### Paper sentence, decoded

**Source.** Radford et al., 2021, [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), Figure 3 ("Numpy-like pseudocode for the core of an implementation of CLIP"). The logits are given as code, not a numbered equation, so the display below is that line in matrix notation. Section 2.5 adds that the temperature $\tau$ is learned, initialised to the equivalent of 0.07, and clipped so logits are never scaled by more than 100.

The key lines are `I_e = l2_normalize(np.dot(I_f, W_i), axis=1)`, the same for `T_e`, and the logits line. With $n$ pairs in the batch:

$$
L = e^{t}\, \hat I \, \hat T^{\top} \in \mathbb{R}^{n \times n}, \qquad L_{jk} = e^{t}\; \hat I_j \cdot \hat T_k = \frac{\cos(I_j, T_k)}{\tau}.
$$

In words: entry $(j, k)$ of the logit matrix is the cosine similarity between image $j$ and caption $k$, divided by a temperature.

| Symbol | What it is | Built in this post |
| --- | --- | --- |
| $I_f W_i$, $T_f W_t$ | features projected into the shared $\mathbb{R}^{d_e}$ | vectors |
| $\hat I_j$, $\hat T_k$ | image $j$ and caption $k$, normalised to length one | unit vectors |
| $\hat I \hat T^{\top}$ | all pairwise dot products; the transpose turns text rows into columns | row times column |
| $\hat I_j \cdot \hat T_k$ | cosine between image $j$ and caption $k$ | cosine |
| $e^{t} = 1/\tau$ | learned scale, stored as a log so it stays positive | temperature |
| diagonal $L_{jj}$ | matching pairs, pushed up by the loss | labels |

The loss applies cross-entropy along rows (each image picks its caption) and columns (each caption picks its image) and averages them; its log-sum-exp is covered in the [einsum and log-sum-exp](/blog/machine-learning/mathematics/sums-products-einsum-and-log-sum-exp) post.

![A 3 by 3 heatmap of cosine similarities with the diagonal outlined, scaled by 14.29 and softmaxed row by row into probabilities 0.994, 0.686 and 0.908 on the diagonal](/imgs/blogs/vectors-dot-products-and-the-geometry-of-embeddings-7.webp)

A hand-checkable example with $n = 3$ and $d_e = 2$. The image embeddings are $\hat I_1 = (1, 0)$, $\hat I_2 = (0.6, 0.8)$, $\hat I_3 = (0, 1)$, and the caption embeddings are $\hat T_1 = (0.96, 0.28)$, $\hat T_2 = (0.6, 0.8)$, $\hat T_3 = (0.28, 0.96)$. All have length one (e.g. $0.96^2 + 0.28^2 = 0.9216 + 0.0784 = 1$). Row 2 of the cosine matrix is

$$
\hat I_2 \cdot \hat T_1 = 0.576 + 0.224 = 0.8, \quad \hat I_2 \cdot \hat T_2 = 0.36 + 0.64 = 1, \quad \hat I_2 \cdot \hat T_3 = 0.168 + 0.768 = 0.936.
$$

Multiply by the initial scale $1/0.07 \approx 14.29$ to get logits $(11.43, 14.29, 13.37)$, and softmax: the correct caption gets 0.686 and the near-miss caption 3 still gets 0.275. Now the temperature's job is visible: cosines differ by at most 2, so a raw softmax over CLIP's 32,768-caption batches would be nearly uniform. Scaling stretches the gaps. At the clipped scale of 100 the correct caption's probability rises to 0.998.

## Proof, read line by line

The independence section leaned on a counting fact: $\mathbb{R}^d$ holds at most $d$ independent directions. A complete proof. It is the standard elimination argument found in the independence-and-dimension chapter of any linear algebra text (for example Strang, *Introduction to Linear Algebra*); Axler, *Linear Algebra Done Right* (3rd ed.), Result 2.23, proves the general version (an independent list is never longer than a spanning list) by a different exchange argument.

**Statement.** Let $d \ge 1$ and let $v_1, \dots, v_{d+1}$ be vectors, under

- (A1) every $v_i$ lies in $\mathbb{R}^d$, so it has exactly $d$ real entries;
- (A2) there are $d + 1$ of them, one more than the dimension.

Then $v_1, \dots, v_{d+1}$ are linearly dependent. Consequently any list of more than $d$ vectors in $\mathbb{R}^d$ is dependent.

**The idea in one sentence.** Use one vector to cancel everyone's last coordinate, dropping one dimension, then induct.

**Steps.** We induct on the dimension $d$.

1. Base case $d = 1$: the vectors are two numbers $v_1, v_2$. *(A1), (A2).*
2. If $v_1 = 0$, then $1 \cdot v_1 + 0 \cdot v_2 = 0$ is nontrivial. *Coefficient 1 is not zero.*
3. If $v_1 \ne 0$, then $v_2 \cdot v_1 + (-v_1) \cdot v_2 = 0$, with coefficient $-v_1 \ne 0$. *Real multiplication commutes.* So the base case holds.
4. Inductive hypothesis: for some $d \ge 2$, any $d$ vectors in $\mathbb{R}^{d-1}$ are dependent. *The claim one dimension down.*
5. Take $d + 1$ vectors in $\mathbb{R}^d$ and look at their last entries $v_i[d]$. *(A1).*
6. Case 1: every last entry is zero, so the first $d$ vectors are in effect $d$ vectors in $\mathbb{R}^{d-1}$. *Drop the zero coordinate.*
7. By step 4 they satisfy $\sum_{i=1}^{d} \lambda_i v_i = 0$ with some $\lambda_i \ne 0$; give $v_{d+1}$ coefficient zero. *Padding with zero coefficients keeps it nontrivial.*
8. Case 2: some last entry is nonzero; reorder so that $c = v_{d+1}[d] \ne 0$. *Relabelling is harmless.*
9. For $i = 1, \dots, d$ define $w_i = v_i - \frac{v_i[d]}{c}\, v_{d+1}$. *Division is legal: $c \ne 0$ and entries are real (A1).*
10. Each $w_i$ has last entry $v_i[d] - \frac{v_i[d]}{c} \cdot c = 0$. *Arithmetic.*
11. So $w_1, \dots, w_d$ are $d$ vectors living in a copy of $\mathbb{R}^{d-1}$, and by step 4 there are $\lambda_1, \dots, \lambda_d$, not all zero, with $\sum_{i=1}^{d} \lambda_i w_i = 0$. *Step 4; (A2) leaves exactly $d$ vectors once $v_{d+1}$ is set aside.*
12. Substitute the definition of $w_i$: $\sum_{i=1}^{d} \lambda_i v_i - \Big(\sum_{i=1}^{d} \frac{\lambda_i v_i[d]}{c}\Big) v_{d+1} = 0$. *Linearity: expand and collect the $v_{d+1}$ terms.*
13. The coefficients of $v_1, \dots, v_d$ are the $\lambda_i$, not all zero: nontrivial. *Step 11.* This completes the induction; for more than $d + 1$ vectors, pad with zero coefficients as in step 7.

**Where each assumption was used.**

| Assumption | Steps |
| --- | --- |
| (A1) real vectors in $\mathbb{R}^d$ | 1, 5, 9 |
| (A2) $d + 1$ vectors | 1, 11 |

**What breaks without it.** With exactly $d$ vectors (dropping A2) the claim is false: $e_1, \dots, e_d$ are independent, since $\sum_i c_i e_i = (c_1, \dots, c_d)$ is zero only when every $c_i$ is. The bound is tight. Let the vectors live in $\mathbb{R}^{D}$ with $D \gt d$ (dropping A1) and $d + 1$ standard basis vectors are independent again.

**Proof moves used:** `induction (layers / steps)` · `add-and-subtract`.

Step 9 is one step of Gaussian elimination, so the proof is also how libraries compute rank. The sibling post on [rank, null space and low-rank structure](/blog/machine-learning/mathematics/rank-null-space-and-low-rank-structure) picks the thread up from here.

## Check it numerically

The script checks four claims: the dot product equals $\lVert a \rVert \lVert b \rVert \cos\theta$ with the angle from the law of cosines, random cosines have standard deviation $1/\sqrt{d}$, five vectors in $\mathbb{R}^{4}$ have rank 4 at most, and the toy CLIP probabilities.

```python
import numpy as np
rng = np.random.default_rng(0)

# 1. a.b = |a||b|cos(theta), with cos from the law of cosines
a, b = rng.normal(size=5), rng.normal(size=5)
na, nb, nc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(a - b)
cos_lc = (na**2 + nb**2 - nc**2) / (2 * na * nb)
print(f"a.b = {a @ b:.6f}   |a||b|cos = {na * nb * cos_lc:.6f}")

# 2. cosine of random pairs: mean 0, std 1/sqrt(d)
for d in [3, 10, 100, 768, 4096]:
    x = rng.normal(size=(20000, d)); y = rng.normal(size=(20000, d))
    c = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
    print(f"d={d:5d}  mean={c.mean():+.4f}  std={c.std():.4f}  1/sqrt(d)={d**-0.5:.4f}")

# 3. d+1 vectors in R^d are dependent: rank never exceeds d
V = rng.normal(size=(5, 4))
print("rank of 5 random vectors in R^4:", np.linalg.matrix_rank(V))

# 4. CLIP-style logits on the toy embeddings
I = np.array([[1, 0], [0.6, 0.8], [0, 1]])
T = np.array([[0.96, 0.28], [0.6, 0.8], [0.28, 0.96]])
logits = (I @ T.T) * (1 / 0.07)
p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
print("row softmax diagonal:", np.round(np.diag(p), 4))
```

Output:

```
a.b = 1.083756   |a||b|cos = 1.083756
d=    3  mean=+0.0042  std=0.5740  1/sqrt(d)=0.5774
d=   10  mean=-0.0036  std=0.3154  1/sqrt(d)=0.3162
d=  100  mean=-0.0006  std=0.0997  1/sqrt(d)=0.1000
d=  768  mean=-0.0001  std=0.0362  1/sqrt(d)=0.0361
d= 4096  mean=-0.0001  std=0.0157  1/sqrt(d)=0.0156
rank of 5 random vectors in R^4: 4
row softmax diagonal: [0.9941 0.6858 0.9076]
```

The identity matches exactly because it is algebra; the standard deviations miss $1/\sqrt{d}$ by at most 0.0034, the expected sampling error from 20,000 pairs ($\sigma / \sqrt{2N} \approx 0.003$ for $d = 3$).

## Where this shows up in practice

**Vector search with FAISS.** FAISS, Meta's similarity-search library, offers an inner-product index. Its [documentation on metric types](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances) says this is not cosine similarity by itself, unless the vectors are normalised, and says to normalise both database vectors and queries. That is $\cos(a, b) = \hat a \cdot \hat b$ as an indexing rule; skip it and you get the length bias of the $k_1$, $k_2$ example.

**The $1/\sqrt{d_k}$ in attention.** Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Section 3.2.1 and its footnote, justify dividing scores by $\sqrt{d_k}$ with our near-orthogonality computation: unit-variance independent entries give $q \cdot k$ variance $d_k$, which saturates the softmax. CLIP's temperature is the same knob turned the other way, because cosines are too small.

**Contrastive training in OpenCLIP.** The open-source [OpenCLIP](https://github.com/mlfoundations/open_clip) reimplementation follows Figure 3: normalised features, a learned `logit_scale` stored as a log, and symmetric cross-entropy over the cosine matrix.

## What to remember

> - A basis is a smallest set of directions that reaches everything; coordinates are the recipe in that basis.
> - More than $d$ vectors in $\mathbb{R}^d$ are always linearly dependent; the proof is elimination plus induction.
> - The dot product is $\lVert a \rVert$ times the shadow of $b$ on $a$; cosine similarity is the dot product of the normalised vectors
> - Two random directions in $\mathbb{R}^d$ have cosine with standard deviation $1/\sqrt{d}$, so small cosines can still be strong signals.
> - CLIP's logits are an $n \times n$ table of cosines divided by a learned temperature.

## Where next

The series starts with [how to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper), and today's induction is catalogued in [proof techniques for ML researchers](/blog/machine-learning/mathematics/proof-techniques-for-ml-researchers). Next in this track, [matrices as linear maps](/blog/machine-learning/mathematics/matrices-as-linear-maps-what-a-layer-does) shows what a layer does, [rank, null space and low-rank structure](/blog/machine-learning/mathematics/rank-null-space-and-low-rank-structure) explains why LoRA's tiny updates work, and [determinants, volume and invertibility](/blog/machine-learning/mathematics/determinants-volume-and-invertibility) measures how a map stretches space.

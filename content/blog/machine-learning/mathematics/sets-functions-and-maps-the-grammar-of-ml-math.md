---
title: "Sets, Functions and Maps: The Grammar of ML Math"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "Read set-builder notation, domains, images, injective and bijective maps and composition fluently, and decode the ResNet residual map as one line of function grammar."
tags: ["mathematics", "math-for-ai", "sets", "functions", "composition", "bijection", "resnet", "neural-networks", "proofs"]
category: "machine-learning"
subcategory: "Mathematics"
author: "Hiep Tran"
featured: true
readTime: 18
---

> **Level:** Foundation · **Needs:** [How to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper) · **Unlocks:** linear maps as matrices, normalizing-flow and change-of-variables appendices, invertible-network papers, every "let $f_\theta: \mathcal{X} \to \mathcal{Y}$" opening line

The most cited sentence in deep learning is a function definition. Section 3.2 of the ResNet paper writes a building block as $\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$, and the introduction says the authors "hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping." Two words carry the whole idea: *mapping* and *residual*. Read as "a layer plus a skip connection", you have the engineering but not the math, and the math is what says when a residual network can be run backwards.

Every ML paper uses one grammar: sets are the nouns, functions are the verbs, composition builds the sentences. The vocabulary is small and closed; learn each piece once, with a picture, an example, a proof and a numerical check, and a theory section stops reading like a wall. By the end you will decode the ResNet equation symbol by symbol and read a short proof line by line.

![The series ladder with rung A lit: sets and maps are the grammar used by residual networks, normalizing flows and every layer written as a map from R^d to R^k.](/imgs/blogs/sets-functions-and-maps-the-grammar-of-ml-math-1.webp)

The ladder above maps the series. This post is the bottom rung, and the arrows show why it comes first: residual networks, normalizing flows and every layer definition are written in this language.

## 1. Sets: the nouns of a paper

A set is how a paper says "the things we are talking about". When an appendix says "for all $x \in \mathcal{X}$", it promises the statement for every member of one collection and nothing outside it.

A good physical picture is a guest list at a door. Order and duplicates do not matter; someone is on it or not. That yes-or-no test is called **membership**, written $x \in A$ ("$x$ is in $A$") or $x \notin A$. The analogy breaks because most ML sets are infinite, so they are defined by a rule rather than a list.

### Set-builder notation

A rule-defined set is written in **set-builder notation**: $\{ x \in S : P(x) \}$, read "all $x$ in $S$ such that $P(x)$ holds." The part before the colon says where to look, the colon reads "such that", and $P(x)$ is the test. Some authors write a vertical bar instead of the colon; it means the same thing.

![The closed unit disk drawn on the plane as the set of x in R^2 with norm at most 1; the boundary point (0.6, 0.8) and the inside point pass the test, (1.5, 1) fails.](/imgs/blogs/sets-functions-and-maps-the-grammar-of-ml-math-2.webp)

The figure shows the set $\{ x \in \mathbb{R}^2 : \lVert x \rVert \le 1 \}$, the closed unit disk. Check the three points by hand. For $(0.6, 0.8)$ the length is $\sqrt{0.36 + 0.64} = \sqrt{1} = 1$, which satisfies "at most 1", so it is in. For $(1.5, 1)$ the length is $\sqrt{2.25 + 1} = \sqrt{3.25} \approx 1.80$, so it is out. The point $(0.5, -0.5)$ has length about ${0.71}$ and sits inside.

### The handful of sets you will meet everywhere

With small finite sets, everything can be checked on paper. Take $A = \{1, 2, 3\}$ and $B = \{2, 3, 4\}$:

| Notation | Read as | Result for this A, B |
| --- | --- | --- |
| $A \cup B$ | union: in either | $\{1, 2, 3, 4\}$ |
| $A \cap B$ | intersection: in both | $\{2, 3\}$ |
| $A \setminus B$ | difference: in A, not in B | $\{1\}$ |
| $A \subseteq B$ | subset: every member of A is in B | false, since $1 \notin B$ |
| $A \times B$ | Cartesian product: all ordered pairs | 9 pairs, $(1,2), (1,3), \ldots, (3,4)$ |

The Cartesian product is the one ML leans on hardest. The set $\mathbb{R}^d$ is the product of $d$ copies of the real line $\mathbb{R}$: an element is an ordered list of $d$ real numbers, which is exactly a vector or an embedding. A dataset of input-label pairs lives in $\mathcal{X} \times \mathcal{Y}$.

Set-builder notation lets papers carve out the sets they actually need. The probability simplex, the set of all distributions over $k$ outcomes, is

$$
\Delta^{k-1} = \Big\{ p \in \mathbb{R}^k : p_i \ge 0 \text{ for all } i, \ \textstyle\sum_{i=1}^{k} p_i = 1 \Big\}.
$$

In words: lists of $k$ non-negative numbers that add up to one. The vector $(0.2, 0.3, 0.5)$ is in $\Delta^2$, while $(0.6, 0.6, -0.2)$ sums to one but fails non-negativity, so it is out.

## 2. Functions: domain, codomain, image

A function takes every allowed input to exactly one output: a layer, a loss, a sampler, a whole model.

The physical picture is a vending machine. The **domain** is the set of buttons you are allowed to press. The **codomain** is the set of things the machine is declared to be able to drop: everything in its catalogue. The **image** is what it actually drops when you press every button. Unlike a real machine, a function never jams: every input produces exactly one output.

The notation is $f: A \to B$, read "$f$ maps $A$ into $B$", with $A$ the domain and $B$ the codomain. The barred arrow $x \mapsto f(x)$ says where one element goes. The **image** is $f(A) = \{ f(x) : x \in A \}$, a subset of $B$ that may be strictly smaller.

![The map f(x) = x squared from A = {-2, -1, 0, 1, 2} to B = {0, 1, 2, 3, 4}: arrows land only on 0, 1 and 4, so 2 and 3 are never hit.](/imgs/blogs/sets-functions-and-maps-the-grammar-of-ml-math-3.webp)

Work the example in the figure. Take $A = \{-2, -1, 0, 1, 2\}$, $B = \{0, 1, 2, 3, 4\}$, and $f(x) = x^2$. Then $f(-2) = 4$, $f(-1) = 1$, $f(0) = 0$, $f(1) = 1$, $f(2) = 4$. The image is $f(A) = \{0, 1, 4\}$. The codomain still contains 2 and 3; nothing is wrong, they are just never reached.

### Preimage: running the question backwards

The **preimage** of a set $C \subseteq B$ asks which inputs land in $C$: $f^{-1}(C) = \{ x \in A : f(x) \in C \}$. In the example, $f^{-1}(\{4\}) = \{-2, 2\}$ and $f^{-1}(\{3\}) = \varnothing$, the empty set. The symbol $f^{-1}$ here does not claim that $f$ has an inverse; it is defined for every function. Applied to a set, it means preimage.

![ReLU on the real line: the image is the half-line from 0 upward, and every point of the half-line from minus infinity to 0 collapses onto the output 0.](/imgs/blogs/sets-functions-and-maps-the-grammar-of-ml-math-7.webp)

ReLU is the ML example worth memorizing. As $\operatorname{ReLU}: \mathbb{R} \to \mathbb{R}$, $x \mapsto \max(0, x)$, its codomain is the whole real line but its image is $[0, \infty)$. The preimage of the single output 0 is the whole half-line $(-\infty, 0]$: the inputs $-2$ and $-0.5$ in the chart both come out as 0. A "dead ReLU" in a training log is this set-theory fact. If a unit's pre-activation sits in that preimage for every input, its output is the constant 0 and its gradient vanishes.

In high dimensions, softmax is a map $\mathbb{R}^k \to \mathbb{R}^k$ whose image is not even the full simplex: every output entry is $e^{z_i} / \sum_j e^{z_j}$, which is strictly positive, so the image is the simplex with its boundary removed. That is why cross-entropy against a hard one-hot label can approach zero but never reach it. The exact one-hot vector is in the codomain, not in the image.

## 3. Injective, surjective, bijective

Two yes-or-no questions classify every map and decide whether it can be undone: can two inputs collide on one output, and is every output in the codomain hit?

![A two by two grid of small mapping diagrams: neither, injective but not surjective, surjective but not injective, and bijective.](/imgs/blogs/sets-functions-and-maps-the-grammar-of-ml-math-4.webp)

A map with no collisions is **injective** (one-to-one): $f(x) = f(x')$ forces $x = x'$. A map that hits every element of its codomain is **surjective** (onto): for every $y \in B$ there is some $x \in A$ with $f(x) = y$. A map that is both is **bijective**, and only then does an inverse function $f^{-1}: B \to A$ exist, with $f^{-1}(f(x)) = x$ for every $x$. In the grid, amber marks a collision and a hollow dot an output nobody reaches.

The answers depend on the declared domain and codomain, not just the formula:

| Map | Injective? | Surjective? | Why |
| --- | --- | --- | --- |
| $x \mapsto x^2$, $\mathbb{R} \to \mathbb{R}$ | no | no | $f(-2) = f(2) = 4$; nothing maps to $-1$ |
| $x \mapsto x^2$, $\mathbb{R} \to [0, \infty)$ | no | yes | same collision; every $y \ge 0$ has $\sqrt{y}$ |
| $x \mapsto e^x$, $\mathbb{R} \to \mathbb{R}$ | yes | no | strictly increasing; never negative |
| $x \mapsto e^x$, $\mathbb{R} \to (0, \infty)$ | yes | yes | inverse is $\log$ |
| ReLU, $\mathbb{R} \to \mathbb{R}$ | no | no | all of $(-\infty, 0]$ goes to 0 |
| softmax, $\mathbb{R}^k \to \mathbb{R}^k$ | no | no | adding a constant to every logit changes nothing |

The softmax row deserves a small computation. Take logits $z = (1, 2, 3)$ and $z' = (0, 1, 2) = z - 1$. Each entry of the softmax is $e^{z_i}/\sum_j e^{z_j}$, and subtracting 1 from every logit multiplies numerator and denominator by the same $e^{-1}$, so both give $(0.090, 0.245, 0.665)$. Two inputs, one output: softmax is not injective. Stable softmax code exploits this collision by subtracting the max logit.

### The dimension squeeze

A linear map from a bigger space to a smaller one can never be injective. Take $W = [1 \ \ 1]$, a 1 by 2 matrix, as a map $\mathbb{R}^2 \to \mathbb{R}$, $x \mapsto Wx$. Then $W(1, 0)^\top = 1$ and $W(0, 1)^\top = 1$: two different inputs, one output. In general a linear map $\mathbb{R}^d \to \mathbb{R}^k$ with $k \lt d$ always sends some nonzero vector to zero; we state this without proof here, and a later post on rank and null space proves it. So a bottleneck from 4096 to 512 dimensions must lose information, and invertible flow models keep every layer the same width.

## 4. Composition and inverses

Composition builds a big map from small ones. The notation is $(g \circ f)(x) = g(f(x))$, read "$g$ after $f$". The order trips everyone at first: the map written on the right acts first, because it sits closest to $x$.

Composition has one hard rule: $g \circ f$ is defined only when the outputs of $f$ are valid inputs for $g$. If $f: A \to B$ and $g: B \to C$, then $g \circ f: A \to C$. In code this rule is the shape check: a matmul shape mismatch is a composition the math never allowed.

Composition is associative, $h \circ (g \circ f) = (h \circ g) \circ f$, so long chains need no brackets. It is not commutative. With $f(x) = x + 1$ and $g(x) = 2x$ on the real line, $g(f(3)) = g(4) = 8$ while $f(g(3)) = f(6) = 7$.

<figure class="blog-anim">
<svg viewBox="0 0 760 340" role="img" aria-label="Two panels. The unit square is rotated by 90 degrees then stretched horizontally on the left, and stretched then rotated on the right; the two orders land on different rectangles." style="width:100%;height:auto;max-width:760px">
<style>
.c1-axis{stroke:var(--text-secondary,#6b7280);stroke-width:1.5}
.c1-target{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:2;stroke-dasharray:6 5}
.c1-sq{fill:var(--accent,#6366f1);fill-opacity:.35;stroke:var(--accent,#6366f1);stroke-width:2.5;transform-box:view-box;transform-origin:0 0}
.c1-ttl{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.c1-lbl{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
@keyframes c1-gf{0%,10%{transform:scale(1,1) rotate(0deg)}40%,50%{transform:scale(1,1) rotate(90deg)}80%,100%{transform:scale(2,1) rotate(90deg)}}
@keyframes c1-fg{0%,10%{transform:rotate(0deg) scale(1,1)}40%,50%{transform:rotate(0deg) scale(2,1)}80%,100%{transform:rotate(90deg) scale(2,1)}}
.c1-left{animation:c1-gf 10s ease-in-out infinite alternate}
.c1-right{animation:c1-fg 10s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){.c1-left,.c1-right{animation:none}}
</style>
<text class="c1-ttl" x="200" y="40">g ∘ f: rotate 90°, then stretch x by 2</text>
<text class="c1-ttl" x="570" y="40">f ∘ g: stretch x by 2, then rotate 90°</text>
<g transform="translate(200,270) scale(60,-60)">
<line class="c1-axis" vector-effect="non-scaling-stroke" x1="-2.6" y1="0" x2="1.6" y2="0"/>
<line class="c1-axis" vector-effect="non-scaling-stroke" x1="0" y1="-0.3" x2="0" y2="2.6"/>
<polygon class="c1-target" vector-effect="non-scaling-stroke" points="-2,0 0,0 0,1 -2,1"/>
<polygon class="c1-sq c1-left" vector-effect="non-scaling-stroke" points="0,0 1,0 1,1 0,1"/>
</g>
<g transform="translate(570,270) scale(60,-60)">
<line class="c1-axis" vector-effect="non-scaling-stroke" x1="-2.6" y1="0" x2="1.6" y2="0"/>
<line class="c1-axis" vector-effect="non-scaling-stroke" x1="0" y1="-0.3" x2="0" y2="2.6"/>
<polygon class="c1-target" vector-effect="non-scaling-stroke" points="-1,0 0,0 0,2 -1,2"/>
<polygon class="c1-sq c1-right" vector-effect="non-scaling-stroke" points="0,0 1,0 1,1 0,1"/>
</g>
<text class="c1-lbl" x="200" y="305">corner (1, 1) lands at (−2, 1): a wide box</text>
<text class="c1-lbl" x="570" y="305">corner (1, 1) lands at (−1, 2): a tall box</text>
</svg>
<figcaption>Same two maps, opposite order. The unit square starts at the origin; the dashed outline is where each composition ends. Order changes the answer.</figcaption>
</figure>

Let $f$ rotate the plane by 90 degrees, $(x, y) \mapsto (-y, x)$, and let $g$ stretch the horizontal axis, $(x, y) \mapsto (2x, y)$. Follow the corner $(1, 1)$. Under $g \circ f$: first $f(1, 1) = (-1, 1)$, then $g(-1, 1) = (-2, 1)$. Under $f \circ g$: first $g(1, 1) = (2, 1)$, then $f(2, 1) = (-1, 2)$.

### Inverses, and the socks-and-shoes rule

The **identity map** $\mathrm{id}_A: A \to A$, $x \mapsto x$, does nothing, and it plays the role of 1 in multiplication. A map $f: A \to B$ is invertible when some $g: B \to A$ satisfies $g \circ f = \mathrm{id}_A$ and $f \circ g = \mathrm{id}_B$; that $g$ is written $f^{-1}$. A map has an inverse exactly when it is bijective, and the inverse is unique.

Undoing a composition reverses the order: $(g \circ f)^{-1} = f^{-1} \circ g^{-1}$. You put on socks, then shoes; you take off shoes first. Check it on the animation: undo the stretch first, $(-2, 1) \mapsto (-1, 1)$, then undo the rotation, $(-1, 1) \mapsto (1, 1)$, and the corner is home.

## 5. A neural network is a composed map

The payoff: a feedforward network is a composition of simple maps, and its shapes are domain-codomain bookkeeping.

![A network drawn as columns of dots, one dot per dimension: R^4 to R^3 to R^3 to R^2, with each arrow labelled by its map and weight shape.](/imgs/blogs/sets-functions-and-maps-the-grammar-of-ml-math-5.webp)

One layer is a map $f_\ell: \mathbb{R}^{d_{\ell-1}} \to \mathbb{R}^{d_\ell}$, $h \mapsto \sigma(W_\ell h + b_\ell)$, where $W_\ell$ is a $d_\ell \times d_{\ell-1}$ weight matrix, $b_\ell$ is a bias vector in $\mathbb{R}^{d_\ell}$, and $\sigma$ is an activation applied entry by entry. The network is

$$
f = f_L \circ f_{L-1} \circ \cdots \circ f_1 : \mathbb{R}^{d_0} \to \mathbb{R}^{d_L}.
$$

In words: the input passes through $f_1$ first and $f_L$ last, and the whole thing is one map from input space to output space. In the figure each arrow must leave from the column the previous arrow arrived at.

A worked example with numbers you can check. Let $f_1: \mathbb{R}^2 \to \mathbb{R}^2$ use

$$
W_1 = \begin{pmatrix} 1 & -1 \\ 2 & 0 \end{pmatrix}, \quad b_1 = \begin{pmatrix} 0 \\ -1 \end{pmatrix}, \quad \sigma = \operatorname{ReLU},
$$

and let $f_2: \mathbb{R}^2 \to \mathbb{R}$ be $h \mapsto 3h_1 + 2h_2 + 0.5$. For the input $x = (1, 2)$: $W_1 x = (1 - 2, \ 2) = (-1, 2)$; adding $b_1$ gives $(-1, 1)$; ReLU gives $(0, 1)$; then $f_2$ gives ${0 + 2 + 0.5 = 2.5}$. So $(f_2 \circ f_1)(1, 2) = 2.5$.

Now feed it $x' = (0, 1)$: $W_1 x' = (-1, 0)$, plus $b_1$ is $(-1, -1)$, ReLU is $(0, 0)$, and the output is 0.5. Feed it $x'' = (0, 2)$: $W_1 x'' = (-2, 0)$, plus $b_1$ is $(-2, -1)$, ReLU is again $(0, 0)$, output 0.5. The ReLU layer collided two inputs, and section 7 proves one such layer makes the whole network non-injective.

### Where the parameters live

For a fixed parameter vector $\theta \in \Theta$, $f_\theta: \mathcal{X} \to \mathcal{Y}$ is one map. Training treats it as a map of two arguments, $f: \Theta \times \mathcal{X} \to \mathcal{Y}$, on a Cartesian product, and $\nabla_\theta$ differentiates along the $\Theta$ factor. "The model class" is the set $\{ f_\theta : \theta \in \Theta \}$, whose elements are functions.

## 6. The residual map: identity plus a small correction

A residual block is the identity map with a learned correction added on top. The ResNet authors call the function a block should represent $\mathcal{H}(\mathbf{x})$, and instead of learning it directly they learn the residual $\mathcal{F}(\mathbf{x}) := \mathcal{H}(\mathbf{x}) - \mathbf{x}$, so the block outputs $\mathcal{F}(\mathbf{x}) + \mathbf{x}$. In function grammar, the block is the map $\mathrm{id} + \mathcal{F}$.

![A vector field on the plane: from each integer grid point, an arrow to p + F(p) with F(x, y) = 0.4 (tanh y, -tanh x); every arrow is short.](/imgs/blogs/sets-functions-and-maps-the-grammar-of-ml-math-6.webp)

The figure draws one such block on the plane with $\mathcal{F}(x, y) = 0.4\,(\tanh y, -\tanh x)$. Each arrow starts at a grid point $p$ and ends at $p + \mathcal{F}(p)$. Since $\lvert \tanh \rvert \lt 1$, no arrow is longer than $0.4\sqrt{2} \approx 0.57$, and the longest drawn is 0.56. Check the amber point: $\mathcal{F}(2, 1) = 0.4\,(0.7616, -0.9640) = (0.3046, -0.3856)$, so $(2, 1)$ moves to $(2.3046, 0.6144)$. If training sets $\mathcal{F}$ to zero, the block is exactly the identity, which is the paper's argument that a deeper residual network should train no worse than a shallower one.

Across depth, the follow-up paper, "Identity Mappings in Deep Residual Networks" (He et al., 2016, [arXiv 1603.05027](https://arxiv.org/abs/1603.05027)), removes the activation after the addition so each block is exactly $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l)$. Unrolling the composition from block $l$ to block $L$ gives their Eq. 4:

$$
\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i).
$$

In words: a deep stack of identity-plus-correction maps is the input plus a sum of corrections, so a signal from any shallow block reaches any deep block through a plain addition.

### Paper sentence, decoded

**Source:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep Residual Learning for Image Recognition", 2015, [arXiv 1512.03385](https://arxiv.org/abs/1512.03385), Section 3.2, Eq. 1:

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}.
$$

| Symbol | What it is | Built in this post |
| --- | --- | --- |
| $\mathbf{x}$ | the block's input, an element of $\mathbb{R}^d$ (a flattened feature map) | section 1, $\mathbb{R}^d$ as a product set |
| $\mathbf{y}$ | the block's output, in the same $\mathbb{R}^d$ | section 2, codomain |
| $\{W_i\}$ | the set of weight matrices inside the block, the parameters | section 1 (a set), section 5 (where parameters live) |
| $\mathcal{F}(\cdot, \{W_i\})$ | the residual map $\mathbb{R}^d \to \mathbb{R}^d$; for two layers the paper writes $\mathcal{F} = W_2 \sigma(W_1 \mathbf{x})$, biases omitted | section 5, a layer as a map, composition |
| $+\,\mathbf{x}$ | the identity map, added entry by entry | section 4, $\mathrm{id}$; section 6 |

The plus sign is only defined when $\mathcal{F}(\mathbf{x})$ and $\mathbf{x}$ live in the same space, so the equation silently requires $\mathcal{F}: \mathbb{R}^d \to \mathbb{R}^d$. When a block changes dimension, the paper's Eq. 2 replaces $\mathbf{x}$ by $W_s \mathbf{x}$, a linear projection whose job is to make the codomains match. Read whole, Eq. 1 says: the block is the identity map plus a learned map from the space to itself.

## 7. Proof, read line by line

Track A teaches proof reading, so the proof is short and every move is visible. It explains section 5: injective layers compose to an injective network.

**Statement.** Let $f: A \to B$ and $g: B \to C$ be maps. Assume

- (A1) $f$ is injective;
- (A2) $g$ is injective;
- (A3) the codomain of $f$ is the domain of $g$, so $g \circ f: A \to C$ is defined.

Then $g \circ f$ is injective. **Corollary:** if $f_1, \ldots, f_L$ are injective and composable, $f_L \circ \cdots \circ f_1$ is injective.

**The idea in one sentence.** Start from two inputs with the same final output and peel the maps off from the outside in, using each map's injectivity once.

**Step 1.** Take any $x, x' \in A$ with $(g \circ f)(x) = (g \circ f)(x')$. *This is the definition of injective: we must show this forces $x = x'$. The composition is defined by (A3).*

**Step 2.** Rewrite as $g(f(x)) = g(f(x'))$. *Definition of composition.*

**Step 3.** The two points $f(x)$ and $f(x')$ are both in $B$, the domain of $g$, and $g$ sends them to the same output. *Again (A3): it is what makes $g$ applicable to them.*

**Step 4.** Therefore $f(x) = f(x')$. *Injectivity of $g$, (A2), applied to the two points $f(x), f(x')$.*

**Step 5.** Therefore $x = x'$. *Injectivity of $f$, (A1).* This ends the proof.

**Step 6 (corollary).** For $L = 1$ the claim is (A1) itself. If $f_{L-1} \circ \cdots \circ f_1$ is injective, the theorem with $f = f_{L-1} \circ \cdots \circ f_1$ and $g = f_L$ shows $f_L \circ \cdots \circ f_1$ is injective. *Induction on the number of layers.*

| Assumption | Used in step |
| --- | --- |
| (A1) $f$ injective | 5 |
| (A2) $g$ injective | 4 |
| (A3) composable | 1, 3 |

**What breaks without (A2).** Let $f = \mathrm{id}_{\mathbb{R}}$, which is injective, and let $g = \operatorname{ReLU}$, which violates (A2). Then $(g \circ f)(-2) = (g \circ f)(-1) = 0$, so the composition collides. That is the section 5 network in miniature.

The proof peels the outer map first, the socks-and-shoes order; longer deep-network proofs share this shape, with a bound in place of each equality.

**Proof moves used:** `induction (layers / steps)`. (Source: a standard fact; see Halmos, *Naive Set Theory*, Section 10, "Inverses and composites".)

## 8. Check it numerically

The script checks that order changes a composition, that a residual map with small $\mathcal{F}$ never collides two points and inverts by iterating $x \leftarrow y - \mathcal{F}(x)$, and that shapes must chain.

```python
import numpy as np
rng = np.random.default_rng(0)

# 1. Composition order: f = rotate 90 degrees, g = stretch x by 2
R = np.array([[0., -1.], [1., 0.]])
S = np.array([[2., 0.], [0., 1.]])
corner = np.array([1., 1.])
print("g(f(1,1)) =", S @ (R @ corner), "  f(g(1,1)) =", R @ (S @ corner))

# 2. Residual map x -> x + F(x), F(x,y) = 0.4 (tanh y, -tanh x), Lip(F) <= 0.4
F = lambda p: 0.4 * np.stack([np.tanh(p[..., 1]), -np.tanh(p[..., 0])], -1)
h = lambda p: p + F(p)
a, b = rng.normal(size=(100_000, 2)) * 2, rng.normal(size=(100_000, 2)) * 2
ratio = np.linalg.norm(h(a) - h(b), axis=1) / np.linalg.norm(a - b, axis=1)
print(f"min ||h(a)-h(b)|| / ||a-b|| over 1e5 pairs = {ratio.min():.4f}  (bound 1-0.4 = 0.6)")

# 3. Invert h by fixed-point iteration x <- y - F(x) (contraction, rate 0.4)
x_true = np.array([2., 1.])
y = h(x_true)
x = y.copy()
for k in range(1, 31):
    x = y - F(x)
    if k in (1, 5, 10, 30):
        print(f"iter {k:2d}: x = {x.round(6)}, error = {np.linalg.norm(x - x_true):.2e}")

# 4. A network as a composed map R^4 -> R^3 -> R^3 -> R^2
W1, W2, W3 = rng.normal(size=(3, 4)), rng.normal(size=(3, 3)), rng.normal(size=(2, 3))
relu = lambda z: np.maximum(z, 0)
f = lambda v: W3 @ relu(W2 @ relu(W1 @ v))
print("f(x).shape =", f(rng.normal(size=4)).shape)
try:
    W3 @ rng.normal(size=4)  # f3 expects R^3, gets R^4
except ValueError as e:
    print("f3 on R^4:", str(e)[:52])
```

```console
g(f(1,1)) = [-2.  1.]   f(g(1,1)) = [-1.  2.]
min ||h(a)-h(b)|| / ||a-b|| over 1e5 pairs = 0.8234  (bound 1-0.4 = 0.6)
iter  1: x = [2.085754 1.0065  ], error = 8.60e-02
iter  5: x = [2.000002 1.      ], error = 1.78e-06
iter 10: x = [2. 1.], error = 1.26e-12
iter 30: x = [2. 1.], error = 0.00e+00
f(x).shape = (2,)
f3 on R^4: matmul: Input operand 1 has a mismatch in its core d
```

The corners match the hand computation exactly. The smallest ratio, 0.82, clears the guaranteed floor of 0.6 because random pairs rarely align with the direction where $\mathcal{F}$ pulls hardest, and the inversion error falls faster than the worst-case factor 0.4 per step because $\tanh$ is flat near $(2, 1)$.

## 9. Where this shows up in practice

**Invertible residual networks.** Behrmann et al., "Invertible Residual Networks" (2019, [arXiv 1811.00995](https://arxiv.org/abs/1811.00995)), make a standard ResNet invertible by keeping every residual branch's Lipschitz constant below 1, enforced with spectral normalization during training. The injectivity half is a three-line argument in this post's vocabulary. If $\lVert \mathcal{F}(x) - \mathcal{F}(x') \rVert \le c \lVert x - x' \rVert$ with $c \lt 1$, then by the triangle inequality $\lVert (x + \mathcal{F}(x)) - (x' + \mathcal{F}(x')) \rVert \ge (1 - c) \lVert x - x' \rVert$, which is positive whenever $x \ne x'$. The inverse is computed by the same fixed-point iteration as part 3 of the script, the method the paper uses.

**Normalizing flows.** Dinh, Sohl-Dickstein and Bengio, "Density estimation using Real NVP" (2016, [arXiv 1605.08803](https://arxiv.org/abs/1605.08803)), build a generative model as a composition of bijections $\mathbb{R}^d \to \mathbb{R}^d$, each designed so its inverse is cheap. Bijectivity maps a sample back to its latent, the dimension squeeze forces equal widths, and likelihoods pull back through the stack in socks-and-shoes order. The model side is in [Normalizing Flows and the Change of Variables](/blog/machine-learning/image-generation/normalizing-flows-and-change-of-variables).

**Shape checking in every framework.** PyTorch's [`torch.nn.Sequential`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) is function composition as a data structure: the docs describe it as passing the output of each module as the input to the next. The "mat1 and mat2 shapes cannot be multiplied" error is the composition rule $f: A \to B$, $g: B \to C$ being violated at runtime.

## What to remember

> - A set is a membership test; set-builder notation $\{ x \in S : P(x) \}$ says where to look and what to test.
> - $f: A \to B$ declares a domain and a codomain; the image $f(A)$ can be much smaller than the codomain, as with ReLU and softmax.
> - Injective means no collisions, surjective means every output hit, bijective means both, and only bijections have inverses. A linear map to fewer dimensions is never injective.
> - $g \circ f$ means $f$ first. Shapes must chain, order matters, and $(g \circ f)^{-1} = f^{-1} \circ g^{-1}$.
> - A network is a composition of layer maps, and a residual block is $\mathrm{id} + \mathcal{F}$: the identity with a small correction.

## Where next

Unfamiliar symbols are covered by the notation decoder in [How to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper). The proof above used one move; [Proof techniques for ML researchers](/blog/machine-learning/mathematics/proof-techniques-for-ml-researchers) adds contrapositive, contradiction and construction, and [The inequality toolkit every appendix uses](/blog/machine-learning/mathematics/the-inequality-toolkit-every-appendix-uses) covers the triangle inequality that powered the invertible-ResNet argument. For the sums inside a softmax, read [Sums, products, einsum and log-sum-exp](/blog/machine-learning/mathematics/sums-products-einsum-and-log-sum-exp).

Next, a later post on vectors and linear maps turns these maps into matrices, and a post on rank and null space proves the dimension squeeze.

---
title: "Norms, Inner Products and Distances: Which Ruler Is This Bound Using?"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "Learn the norm axioms, the l1, l2 and l-infinity unit balls, matrix norms, dual norms and norm-equivalence constants, then decode spectral normalization and prove that operator norms are submultiplicative."
tags: ["mathematics", "math-for-ai", "linear-algebra", "norms", "inner-products", "spectral-norm", "dual-norms", "spectral-normalization", "power-iteration", "proofs"]
category: "machine-learning"
subcategory: "Mathematics"
author: "Hiep Tran"
featured: true
readTime: 18
---

> **Level:** Core · **Needs:** [Vectors, dot products and the geometry of embeddings](/blog/machine-learning/mathematics/vectors-dot-products-and-the-geometry-of-embeddings), [Matrices as linear maps](/blog/machine-learning/mathematics/matrices-as-linear-maps-what-a-layer-does), [The inequality toolkit](/blog/machine-learning/mathematics/the-inequality-toolkit-every-appendix-uses) · **Unlocks:** Lipschitz bounds, spectral normalization, adversarial robustness, gradient clipping, generalization bounds

Section 2.1 of Miyato et al.'s *Spectral Normalization for Generative Adversarial Networks* (ICLR 2018) defines its method in one line, $\bar{W}_{\mathrm{SN}}(W) := W / \sigma(W)$, just after writing $\sigma(A) := \max_{\lVert h \rVert_2 \leq 1} \lVert A h \rVert_2$ and calling it "the spectral norm of the matrix $A$ (L2 matrix norm of $A$)". If you cannot say what that maximum is and why dividing by it tames a GAN discriminator, the paper is a wall.

Every one of those symbols is a ruler, and a bound is only as meaningful as the ruler it was written with. Paper math is a small, closed vocabulary; learn each ruler once, with a picture, an example, a proof and a numerical check, and the double bars stop being decoration. By the end you will read Miyato's definition, estimate $\sigma(W)$ the way the paper does, and prove the inequality that makes it work.

![Three unit balls on a 2D plane: the l1 diamond, the l2 circle and the l-infinity square, with the point x = (0.6, 0.8) at l1 norm 1.4, l2 norm 1 and l-infinity norm 0.8](/imgs/blogs/norms-inner-products-and-distances-1.webp)

The picture above is the mental model: a norm is fully described by its **unit ball**, the set of points at size 1. The point $x = (0.6, 0.8)$ sits on the circle, outside the diamond and inside the square, so it has three sizes.

## 1. What a norm is, and the three axioms

**What it is for:** a norm turns a vector into one number, its size, so that "small error" and "bounded weights" mean something precise.

A tape measure, a city taxi meter and a chessboard king's move count are three rulers for the same street grid. The tape takes the straight line, the taxi adds blocks east plus blocks north, the king counts the larger of the two. All three are honest, and they disagree.

Take $x = (3, -4)$. The tape gives $\sqrt{3^2 + 4^2} = \sqrt{25} = 5$. The taxi meter gives $\lvert 3 \rvert + \lvert -4 \rvert = 7$. The king gives $\max(3, 4) = 4$. These are the three norms you will meet most:

$$
\lVert x \rVert_1 = \sum_{i=1}^{d} \lvert x_i \rvert, \qquad \lVert x \rVert_2 = \Bigl(\sum_{i=1}^{d} x_i^2\Bigr)^{1/2}, \qquad \lVert x \rVert_\infty = \max_{i} \lvert x_i \rvert .
$$

For $x \in \mathbb{R}^d$: the **ℓ1 norm** sums absolute entries, the **ℓ2 norm** is Euclidean length, and the **ℓ∞ norm** keeps only the largest entry. All three belong to the ℓp family, with ℓ∞ as the limit $p \to \infty$.

A **norm** is any function $\lVert \cdot \rVert$ obeying three rules, for all vectors $x, y$ and scalars $\alpha$:

$$
\begin{aligned}
&\text{(N1) } \lVert x \rVert \geq 0, \text{ and } \lVert x \rVert = 0 \iff x = 0 \\
&\text{(N2) } \lVert \alpha x \rVert = \lvert \alpha \rvert \, \lVert x \rVert \\
&\text{(N3) } \lVert x + y \rVert \leq \lVert x \rVert + \lVert y \rVert
\end{aligned}
$$

In words: only zero has size zero, scaling a vector scales its size, and a detour is never shorter than the direct route. (N3), the triangle inequality, is the axiom every proof leans on.

Two popular objects fail the axioms. The "ℓ0 norm", the count of nonzero entries, fails (N2): doubling $x$ does not double its count. The ℓ½ formula fails (N3): with $x = (1, 0)$ and $y = (0, 1)$, $\lVert x + y \rVert_{1/2} = 4 \gt 2$. Its unit ball is a pinched, non-convex star. When a paper calls such a thing a "norm", read it as a penalty and check that no step uses the triangle inequality.

### Reading the unit ball

The ball tells you which vectors a norm considers cheap. The ℓ1 diamond has corners on the axes, so an ℓ1 penalty favours putting everything into few coordinates, which is why Lasso gives sparse solutions. The ℓ∞ square charges nothing for spreading mass as long as no entry is large. The ℓ2 circle has no preferred direction, the default whenever a paper wants rotation invariance. The dashed outlines in the figure answer "how much must I scale the ball to touch $x$?": by 1.4 for the diamond, 0.8 for the square.

## 2. Inner products and the norm they induce

**What it is for:** an inner product measures alignment between two vectors, and gives a length and an angle for free.

The dot product $\langle x, y \rangle = \sum_i x_i y_i$ is the familiar case. Formally an **inner product** is a function $\langle \cdot, \cdot \rangle$ that is symmetric, linear in each argument, and positive: $\langle x, x \rangle \gt 0$ for every $x \neq 0$. Every inner product **induces** a norm:

$$
\lVert x \rVert = \sqrt{\langle x, x \rangle}.
$$

In words: length is the square root of a vector's alignment with itself; the dot product induces ℓ2.

Not every norm comes from an inner product. An induced norm must satisfy the **parallelogram law**, $\lVert x + y \rVert^2 + \lVert x - y \rVert^2 = 2\lVert x \rVert^2 + 2\lVert y \rVert^2$. With $x = (1, 0)$ and $y = (0, 1)$, ℓ2 gives ${2 + 2}$ on the left and 4 on the right. ℓ1 gives $2^2 + 2^2 = 8$ against 4, so no inner product sits behind ℓ1, and papers that need angles or projections stay in ℓ2.

Matrices get the **trace inner product** $\langle A, B \rangle = \operatorname{tr}(A^\top B) = \sum_{i,j} A_{ij} B_{ij}$, which treats a matrix as a long vector.

### From norm to distance

A norm gives a **distance** $d(x, y) = \lVert x - y \rVert$, and (N1) to (N3) become the metric axioms. Embedding papers often use cosine distance $1 - \cos\theta$ instead. For unit vectors $\lVert x - y \rVert_2^2 = 2 - 2\langle x, y \rangle = 2(1 - \cos\theta)$, so ranking by cosine and by Euclidean distance agree after normalization, although cosine distance itself breaks the triangle inequality.

## 3. Matrix norms: three ways to measure a layer

**What it is for:** a matrix norm measures how big a layer is: its largest stretch, the energy in its entries, or its total stretch over all directions.

A matrix is a rubber sheet being pulled, as in the [matrices post](/blog/machine-learning/mathematics/matrices-as-linear-maps-what-a-layer-does). Pull the unit circle through it and you get an ellipse. Its semi-axes are the **singular values** $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$, and every matrix norm in ML papers is some function of them.

![The matrix A = [[2, 1], [0, 1]] maps the unit circle to an ellipse with semi-axes 2.29 and 0.87; side panel lists spectral norm 2.29, Frobenius norm √6 = 2.45 and nuclear norm √10 = 3.16](/imgs/blogs/norms-inner-products-and-distances-2.webp)

Work through $A = \begin{pmatrix} 2 & 1 \\ 0 & 1 \end{pmatrix}$ by hand. The squared singular values are the eigenvalues of

$$
A^\top A = \begin{pmatrix} 4 & 2 \\ 2 & 2 \end{pmatrix}, \qquad \lambda^2 - 6\lambda + 4 = 0, \qquad \lambda = 3 \pm \sqrt{5}.
$$

The trace is 6 and the determinant is 4. So $\sigma_1 = \sqrt{3 + \sqrt{5}} = \sqrt{5.236} \approx 2.288$ and $\sigma_2 = \sqrt{3 - \sqrt{5}} = \sqrt{0.764} \approx 0.874$.

Now the three norms, each read off the ellipse:

$$
\lVert A \rVert_2 = \sigma_1, \qquad \lVert A \rVert_F = \Bigl(\sum_{i,j} A_{ij}^2\Bigr)^{1/2} = \Bigl(\sum_k \sigma_k^2\Bigr)^{1/2}, \qquad \lVert A \rVert_* = \sum_k \sigma_k .
$$

In words: the **spectral norm** is the longest semi-axis, the **Frobenius norm** is the ℓ2 norm of the entries, and the **nuclear norm** is the sum of the semi-axes. For our matrix: $\lVert A \rVert_2 \approx 2.288$, $\lVert A \rVert_F = \sqrt{4 + 1 + 1} \approx 2.449$, and since $\sigma_1 \sigma_2 = \lvert \det A \rvert = 2$, $(\sigma_1 + \sigma_2)^2 = 6 + 4 = 10$, so $\lVert A \rVert_* \approx 3.162$.

### The spectral norm is an operator norm

Miyato uses a second definition. Given vector norms on input and output, the **operator norm** of $A$ is

$$
\lVert A \rVert_{\mathrm{op}} = \sup_{x \neq 0} \frac{\lVert A x \rVert}{\lVert x \rVert} = \max_{\lVert x \rVert \leq 1} \lVert A x \rVert .
$$

In words: the largest stretch of any input. With ℓ2 on both sides the most-stretched input is the top right-singular vector $v_1 = (0.851, 0.526)$, landing on the long semi-axis, so the factor is $\sigma_1$. "Spectral norm", "operator 2-norm" and "$\sigma_{\max}(A)$" are three names for one number.

Change the vector norms and the operator norm changes: with ℓ1 on both sides it is the largest absolute column sum (2 for our $A$), with ℓ∞ the largest absolute row sum (3). Section 6 returns to this trap.

For rank $r$, $\lVert A \rVert_2 \leq \lVert A \rVert_F \leq \sqrt{r}\,\lVert A \rVert_2$ and $\lVert A \rVert_F \leq \lVert A \rVert_* \leq \sqrt{r}\,\lVert A \rVert_F$. On $A$: $2.288 \leq 2.449 \leq 3.236$ and $2.449 \leq 3.162 \leq 3.464$. A flat spectrum spreads the three by $\sqrt{r}$, a factor of 64 for a 4096-wide layer.

## 4. Dual norms: the best-case inner product

**What it is for:** the dual norm answers "over all inputs of size at most 1, how large can the inner product with $y$ get?" Every robustness bound and steepest-descent step asks it.

Picture a fence shaped like the unit ball and wind blowing along $y$: the dual norm is how far downwind you can stand inside the fence. In the ℓ1 diamond you end at a corner, since a linear function on a polygon peaks at a vertex; in the ℓ2 circle you stand where the wind points.

![Worked example for x = (3, -4): l1 norm 7, l2 norm 5, l-infinity norm 4; duality pairs l1 with l-infinity, l2 with l2 and spectral with nuclear; tight Hölder examples with y = (1, -1) and y = (0.6, -0.8)](/imgs/blogs/norms-inner-products-and-distances-4.webp)

The definition makes the picture precise:

$$
\lVert y \rVert_* = \sup_{\lVert x \rVert \leq 1} \langle x, y \rangle .
$$

In words: the largest inner product $y$ can have with a vector in the unit ball. Work it out for ℓ1 with $y = (3, -4)$. The diamond's corners are $(\pm 1, 0)$ and $(0, \pm 1)$, giving inner products $\pm 3$ and $\pm 4$. The best is 4, at the corner $(0, -1)$, and $4 = \lVert y \rVert_\infty$. In general ℓ1 and ℓ∞ are dual to each other, ℓ2 is its own dual, and ℓp pairs with ℓq when ${1/p + 1/q = 1}$.

The definition hands you **Hölder's inequality** for free. Scale any nonzero $x$ to $x / \lVert x \rVert$, which lies in the unit ball, so $\langle x / \lVert x \rVert, y \rangle \leq \lVert y \rVert_*$ and therefore

$$
\langle x, y \rangle \leq \lVert x \rVert \, \lVert y \rVert_* .
$$

In words: size times dual size bounds the inner product. With $x = (3, -4)$ and $y = (1, -1)$ this is $3 + 4 = 7 \leq 7 \cdot 1$, tight. With $y = (0.6, -0.8)$ in ℓ2 it is $1.8 + 3.2 = 5 \leq 5 \cdot 1$, Cauchy–Schwarz and tight again. Under the trace inner product, spectral and nuclear norms are dual.

The ML reading: the best first-order step of size $\varepsilon$ gains $\varepsilon \lVert \nabla f \rVert_*$. Under ℓ2 the step is $\varepsilon \nabla f / \lVert \nabla f \rVert_2$; under ℓ∞ it is $\varepsilon \operatorname{sign}(\nabla f)$, a corner of the cube. Section 10 cashes this in.

## 5. Norm equivalence: every norm agrees, up to a constant that grows with d

**What it is for:** equivalence constants convert a bound from one norm to another and price the conversion.

In finite dimensions all norms are equivalent, each within constant factors of the others, but the constants grow with the dimension $d$.

![Ratio of l1 to l2 norm against dimension d from 1 to 100: upper bound √d reaches 10, lower bound stays at 1, and the mean for Gaussian vectors tracks √(2d/π), reaching 8.00 at d = 100](/imgs/blogs/norms-inner-products-and-distances-3.webp)

The chain you will use most is

$$
\lVert x \rVert_\infty \leq \lVert x \rVert_2 \leq \lVert x \rVert_1 \leq \sqrt{d}\,\lVert x \rVert_2 \leq d\,\lVert x \rVert_\infty .
$$

In words: the rulers are ordered, and each conversion costs at most $\sqrt{d}$. On $x = (3, -4)$ with $d = 2$: $4 \leq 5 \leq 7 \leq \sqrt{2} \cdot 5 \approx 7.07 \leq 8$.

For $\lVert x \rVert_2 \leq \lVert x \rVert_1$: squaring the right side gives $\sum_i x_i^2$ plus non-negative cross terms $\lvert x_i \rvert \lvert x_j \rvert$; equality needs at most one nonzero entry.

For $\lVert x \rVert_1 \leq \sqrt{d}\,\lVert x \rVert_2$: $\lVert x \rVert_1$ is the inner product of $\lvert x \rvert$ with the all-ones vector $\mathbf{1}$, and Cauchy–Schwarz bounds it by $\lVert x \rVert_2 \lVert \mathbf{1} \rVert_2 = \sqrt{d}\,\lVert x \rVert_2$, with equality when all entries have equal size.

The figure shows a typical vector sits near the worst case: for a Gaussian vector the ratio is about $\sqrt{2d/\pi}$, roughly 80% of $\sqrt{d}$. At $d = 4096$, ℓ1 and ℓ2 bounds differ by about 51 on an ordinary vector; when a paper says "by equivalence", look for where that factor went.

## 6. How to tell which norm a paper's bound uses

**What it is for:** a correct proof in the wrong norm is a wrong claim, so identify the ruler first.

![Decision tree: a norm appears in a bound; if a subscript is written read it literally; with no subscript on a vector usually l2; with no subscript on a matrix usually spectral, Frobenius if it sums over entries; if the bound is a sup over a ball, find the dual norm](/imgs/blogs/norms-inner-products-and-distances-6.webp)

The habit I use on every new paper:

| Clue in the paper | Most likely norm | Confirm by |
| --- | --- | --- |
| Subscript written: $\lVert \cdot \rVert_1$, $\lVert \cdot \rVert_F$, $\lVert \cdot \rVert_*$ | Exactly that | Notation section |
| Bare $\lVert x \rVert$ on a vector | ℓ2 | A Cauchy–Schwarz step on it |
| Bare $\lVert W \rVert$ on a matrix in a theory paper | Spectral (operator 2-norm) | A step using $\lVert W x \rVert \leq \lVert W \rVert \lVert x \rVert$ |
| $\lVert W \rVert$ in a regularizer | Frobenius | Gradient ${2W}$ for the square |
| $\lVert A \rVert_\infty$ on a matrix | Ambiguous: max row sum or max entry | The paper's definition; never guess |
| A sup over a ball, $\sup_{\lVert \delta \rVert \leq \varepsilon}$ | The dual of that ball's norm appears | Hölder on the inner product |

Internalize the third row: if a proof writes $\lVert W x \rVert \leq \lVert W \rVert \, \lVert x \rVert$ on ℓ2 vectors, the matrix norm is spectral, or Frobenius at a cost of up to $\sqrt{r}$.

## 7. Paper sentence, decoded

### Spectral normalization (Miyato et al., 2018)

**Paper:** Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida, *Spectral Normalization for Generative Adversarial Networks*, ICLR 2018, [arXiv:1802.05957](https://arxiv.org/abs/1802.05957).

Section 2.1, Eqs. 6, 7 and 8 (Eq. 6 also gives the ratio form of the max):

$$
\sigma(A) := \max_{\lVert h \rVert_2 \leq 1} \lVert A h \rVert_2, \qquad \lVert f \rVert_{\mathrm{Lip}} \leq \prod_{l=1}^{L+1} \sigma(W^l), \qquad \bar{W}_{\mathrm{SN}}(W) := \frac{W}{\sigma(W)} .
$$

![Anatomy of the spectral normalization equation: W is the layer weight, sigma of W is the largest singular value, W bar SN is the rescaled weight with spectral norm exactly 1, and sigma is estimated as u transpose W v by one power iteration per step](/imgs/blogs/norms-inner-products-and-distances-5.webp)

| Symbol | Meaning | Section |
| --- | --- | --- |
| $h$ | layer input, in ℓ2 | 1 |
| $\sigma(A)$ | operator 2-norm, the largest singular value | 3 |
| $W^l$ | weight of layer $l$ of the discriminator $f$ | 3 |
| $\lVert f \rVert_{\mathrm{Lip}}$ | largest stretch of distances by $f$ | 3 |
| $\prod_l \sigma(W^l)$ | product of layer spectral norms (ReLU adds a factor of 1) | 8 |
| $\bar{W}_{\mathrm{SN}}$ | rescaled weight, spectral norm exactly 1 | 1, axiom (N2) |

In words: dividing each weight by its spectral norm makes every layer stretch by at most 1, so the discriminator is 1-Lipschitz. By (N2), $\sigma(W / c) = \sigma(W) / c$, so $\sigma(\bar{W}_{\mathrm{SN}}) = 1$; the product bound in Eq. 7 is the submultiplicativity we prove next.

A full SVD every step is expensive, so Appendix A uses **power iteration** (Eqs. 18 and 19): $\tilde{v} \leftarrow W^\top \tilde{u} / \lVert W^\top \tilde{u} \rVert_2$, then $\tilde{u} \leftarrow W \tilde{v} / \lVert W \tilde{v} \rVert_2$, and $\sigma(W) \approx \tilde{u}^\top W \tilde{v}$. Each round multiplies by $W^\top W$, amplifying the top direction by $\sigma_1^2$ against $\sigma_k^2$. The paper reuses $\tilde{u}$ across SGD steps and found one round per step enough.

<figure class="blog-anim">
<svg viewBox="0 0 680 400" role="img" aria-label="Power iteration on the matrix A with rows 2 1 and 0 1: a unit vector starting straight up rotates in steps toward the top right singular vector while the estimate of the largest singular value grows from 1.414 to 2.288" style="width:100%;height:auto;max-width:820px">
<style>
.p1-ax{stroke:var(--border,#d1d5db);stroke-width:1.5}
.p1-circ{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:5 5}
.p1-ell{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:2}
.p1-v{stroke:var(--accent,#6366f1);stroke-width:4;stroke-linecap:round}
.p1-w{stroke:var(--text-primary,#1f2937);stroke-width:3;stroke-linecap:round}
.p1-ref{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.p1-bar{fill:var(--accent,#6366f1)}
.p1-hl{fill:var(--accent,#6366f1);opacity:.15}
.p1-t{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.p1-s{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
@keyframes p1-vk{0.00%,12.50%{transform:rotate(-90.00deg)}16.67%,29.17%{transform:rotate(-45.00deg)}33.33%,45.83%{transform:rotate(-33.69deg)}50.00%,62.50%{transform:rotate(-32.01deg)}66.67%,79.17%{transform:rotate(-31.76deg)}83.33%,95.83%{transform:rotate(-31.72deg)}100%{transform:rotate(-31.72deg)}}
@keyframes p1-wk{0.00%,12.50%{transform:rotate(-45.00deg) scaleX(1.4142)}16.67%,29.17%{transform:rotate(-18.43deg) scaleX(2.2361)}33.33%,45.83%{transform:rotate(-14.04deg) scaleX(2.2871)}50.00%,62.50%{transform:rotate(-13.39deg) scaleX(2.2882)}66.67%,79.17%{transform:rotate(-13.30deg) scaleX(2.2882)}83.33%,95.83%{transform:rotate(-13.28deg) scaleX(2.2882)}100%{transform:rotate(-13.28deg) scaleX(2.2882)}}
@keyframes p1-bk{0.00%,12.50%{transform:scaleX(0.5657)}16.67%,29.17%{transform:scaleX(0.8944)}33.33%,45.83%{transform:scaleX(0.9148)}50.00%,62.50%{transform:scaleX(0.9153)}66.67%,79.17%{transform:scaleX(0.9153)}83.33%,95.83%{transform:scaleX(0.9153)}100%{transform:scaleX(0.9153)}}
@keyframes p1-rk{0.00%,12.50%{transform:translateY(0px)}16.67%,29.17%{transform:translateY(28px)}33.33%,45.83%{transform:translateY(56px)}50.00%,62.50%{transform:translateY(84px)}66.67%,79.17%{transform:translateY(112px)}83.33%,95.83%{transform:translateY(140px)}100%{transform:translateY(140px)}}
.p1-vm{transform-box:view-box;transform-origin:170px 200px;animation:p1-vk 15s ease-in-out infinite}
.p1-wm{transform-box:view-box;transform-origin:170px 200px;animation:p1-wk 15s ease-in-out infinite}
.p1-bm{transform-box:view-box;transform-origin:400px 0;animation:p1-bk 15s ease-in-out infinite}
.p1-rm{animation:p1-rk 15s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.p1-vm,.p1-wm,.p1-bm,.p1-rm{animation:none}}
</style>
<line class="p1-ax" x1="-20" y1="200" x2="360" y2="200"/>
<line class="p1-ax" x1="170" y1="50" x2="170" y2="350"/>
<circle class="p1-circ" cx="170" cy="200" r="70"/>
<polyline class="p1-ell" points="310.0,200.0 315.6,193.9 320.0,187.8 323.3,181.9 325.5,176.1 326.5,170.4 326.2,165.0 324.8,159.8 322.2,155.0 318.5,150.5 313.6,146.4 307.6,142.7 300.6,139.4 292.6,136.6 283.7,134.2 273.8,132.4 263.2,131.1 251.9,130.3 240.0,130.0 227.5,130.3 214.6,131.1 201.4,132.4 187.9,134.2 174.3,136.6 160.6,139.4 147.0,142.7 133.6,146.4 120.5,150.5 107.7,155.0 95.5,159.8 83.8,165.0 72.7,170.4 62.4,176.1 52.9,181.9 44.3,187.8 36.6,193.9 30.0,200.0 24.4,206.1 20.0,212.2 16.7,218.1 14.5,223.9 13.5,229.6 13.8,235.0 15.2,240.2 17.8,245.0 21.5,249.5 26.4,253.6 32.4,257.3 39.4,260.6 47.4,263.4 56.3,265.8 66.2,267.6 76.8,268.9 88.1,269.7 100.0,270.0 112.5,269.7 125.4,268.9 138.6,267.6 152.1,265.8 165.7,263.4 179.4,260.6 193.0,257.3 206.4,253.6 219.5,249.5 232.3,245.0 244.5,240.2 256.2,235.0 267.3,229.6 277.6,223.9 287.1,218.1 295.7,212.2 303.4,206.1 310.0,200.0"/>
<line class="p1-w p1-wm" x1="170" y1="200" x2="240" y2="200"/>
<line class="p1-v p1-vm" x1="170" y1="200" x2="240" y2="200"/>
<text class="p1-s" x="20" y="380">dashed: unit circle · grey: its image under A</text>
<text class="p1-t" x="20" y="30">accent: v_k on the circle · dark: A v_k</text>
<text class="p1-t" x="400" y="30">estimate σ̂ = ‖A v_k‖₂</text>
<rect class="p1-ref" x="400" y="44" width="183.1" height="22" rx="4"/>
<rect class="p1-bar p1-bm" x="400" y="44" width="200" height="22" rx="4"/>
<text class="p1-s" x="400" y="88">box = true σ₁ = 2.288</text>
<rect class="p1-hl p1-rm" x="392" y="106" width="280" height="24" rx="4"/>
<text class="p1-s" x="400" y="123">k = 0: v = (0.000, 1.000), σ̂ = 1.414</text>
<text class="p1-s" x="400" y="151">k = 1: v = (0.707, 0.707), σ̂ = 2.236</text>
<text class="p1-s" x="400" y="179">k = 2: v = (0.832, 0.555), σ̂ = 2.287</text>
<text class="p1-s" x="400" y="207">k = 3: v = (0.848, 0.530), σ̂ = 2.288</text>
<text class="p1-s" x="400" y="235">k = 4: v = (0.850, 0.526), σ̂ = 2.288</text>
<text class="p1-s" x="400" y="263">k = 5: v = (0.851, 0.526), σ̂ = 2.288</text>
<text class="p1-s" x="400" y="308">v_(k+1) = Aᵀ A v_k / ‖Aᵀ A v_k‖₂</text>
</svg>
<figcaption>Power iteration in motion on A = [[2, 1], [0, 1]]: the unit vector v_k (accent) starts straight up and swings toward the top right singular vector (0.851, 0.526) in a few steps, while A v_k (dark) grows to the longest semi-axis of the ellipse and the estimate σ̂ fills the box at σ₁ = 2.288.</figcaption>
</figure>

The accent vector lands within a degree of $v_1$ after three rounds because this matrix has a wide gap, $\sigma_1^2 / \sigma_2^2 \approx 6.9$. Random matrices have a much smaller gap; the numerical check shows the cost.

## 8. Proof, read line by line

**Statement.** Let $B : \mathbb{R}^p \to \mathbb{R}^n$ and $A : \mathbb{R}^n \to \mathbb{R}^m$ be matrices, give each space a norm, and define each operator norm as $\lVert M \rVert = \sup_{x \neq 0} \lVert M x \rVert / \lVert x \rVert$ using the norms of its own input and output spaces. Assume:

- (A1) $A$ and $B$ are linear.
- (A2) The output of $B$ and the input of $A$ are measured with the **same** norm on $\mathbb{R}^n$.
- (A3) The spaces are finite-dimensional, so $\lVert A \rVert$ and $\lVert B \rVert$ are finite.

Then $\lVert A B \rVert \leq \lVert A \rVert \, \lVert B \rVert$. This is standard; see Horn and Johnson, *Matrix Analysis*, 2nd ed., Section 5.6.

**The idea in one sentence:** apply "stretches by at most $\lVert M \rVert$" once per matrix, then take the supremum.

1. Claim: $\lVert M y \rVert \leq \lVert M \rVert \, \lVert y \rVert$ for every $y$. If $y = 0$, then $M y = 0$ by linearity (A1) and both sides are 0 by (N1).
2. If $y \neq 0$, then $\lVert M y \rVert / \lVert y \rVert \leq \lVert M \rVert$, because a supremum bounds every element of its set.
3. Multiply by $\lVert y \rVert \gt 0$; positivity (N1) keeps the direction. The claim follows.
4. Fix $x \neq 0$ in $\mathbb{R}^p$ and set $y = B x$. Composition of linear maps is the product (A1), so $A B x = A y$.
5. $\lVert A B x \rVert = \lVert A y \rVert \leq \lVert A \rVert \, \lVert y \rVert$, by step 1 for $A$, with $y$ in the input norm of $A$.
6. $\lVert y \rVert = \lVert B x \rVert \leq \lVert B \rVert \, \lVert x \rVert$, by step 1 for $B$, with $y$ in the output norm of $B$, the same norm by (A2).
7. $\lVert A B x \rVert \leq \lVert A \rVert \, \lVert B \rVert \, \lVert x \rVert$: chain 5 and 6, multiplying 6 by the finite (A3), non-negative $\lVert A \rVert$.
8. Divide by $\lVert x \rVert \gt 0$: the ratio $\lVert A B x \rVert / \lVert x \rVert$ is at most $\lVert A \rVert \, \lVert B \rVert$ for every $x \neq 0$.
9. So $\lVert A \rVert \lVert B \rVert$ is an upper bound, and the supremum, the least upper bound, gives $\lVert A B \rVert \leq \lVert A \rVert \, \lVert B \rVert$. ∎

Induction over layers gives Miyato's Eq. 7, with each 1-Lipschitz activation contributing a factor of at most 1.

**Where each assumption was used:**

| Assumption | Step |
| --- | --- |
| (A1) linearity | 1, 4 |
| (A2) matching middle norm | 5 and 6, which chain in step 7 |
| (A3) finiteness | 7 |

**What breaks without (A2).** Take $B = (1, 1)^\top$ from $\mathbb{R}$ to $\mathbb{R}^2$ with ℓ2 on its output, so $\lVert B \rVert = \sqrt{2}$. Take $A = (1 \;\; 1)$ with ℓ1 on its input, so $\lVert A \rVert = 1$, the largest absolute column entry. Then $A B = 2 \gt 1 \cdot \sqrt{2}$. The middle vector $(1, 1)$ was measured as $\sqrt{2}$ by one ruler and 2 by the other, and the bound fell through the gap.

**Proof moves used:** `induction (layers / steps)`, plus the supremum-as-least-upper-bound step that every operator-norm proof ends with.

## 9. Check it numerically

Miyato's Eqs. 18 and 19 on a random 256×128 weight, against NumPy's largest singular value:

```python
import numpy as np

rng = np.random.default_rng(0)
W = rng.standard_normal((256, 128))          # a layer's weight matrix

def power_iteration(W, steps, u=None):
    u = rng.standard_normal(W.shape[0]) if u is None else u
    for _ in range(steps):
        v = W.T @ u; v /= np.linalg.norm(v)  # Eq. 18, right vector
        u = W @ v;   u /= np.linalg.norm(u)  # Eq. 18, left vector
    return u @ W @ v, u                      # Eq. 19: sigma ~ u^T W v

sigma_true = np.linalg.norm(W, 2)            # largest singular value
for k in [1, 2, 5, 20, 100]:
    est, _ = power_iteration(W, k)
    print(f"steps={k:4d}  sigma_hat={est:.6f}  rel_err={(sigma_true-est)/sigma_true:.2e}")
print(f"numpy ord=2      sigma    ={sigma_true:.6f}")

# after normalization the spectral norm is 1, Frobenius and nuclear are not
est, _ = power_iteration(W, 100)
W_sn = W / est
s = np.linalg.svd(W_sn, compute_uv=False)
print(f"W_sn: spectral={s[0]:.6f}  frobenius={np.sqrt((s**2).sum()):.4f}  nuclear={s.sum():.4f}")

# submultiplicativity on random pairs: ||AB||_2 <= ||A||_2 ||B||_2
ratios = []
for _ in range(1000):
    A, B = rng.standard_normal((64, 32)), rng.standard_normal((32, 48))
    ratios.append(np.linalg.norm(A @ B, 2) / (np.linalg.norm(A, 2) * np.linalg.norm(B, 2)))
print(f"max ||AB||/(||A|| ||B||) over 1000 pairs = {max(ratios):.4f}")
```

```console
steps=   1  sigma_hat=19.024168  rel_err=3.01e-01
steps=   2  sigma_hat=21.972200  rel_err=1.92e-01
steps=   5  sigma_hat=25.659913  rel_err=5.68e-02
steps=  20  sigma_hat=27.090893  rel_err=4.22e-03
steps= 100  sigma_hat=27.205643  rel_err=8.76e-07
numpy ord=2      sigma    =27.205667
W_sn: spectral=1.000000  frobenius=6.6389  nuclear=70.0232
max ||AB||/(||A|| ||B||) over 1000 pairs = 0.8466
```

The estimate approaches from below, since $\tilde{u}^\top W \tilde{v} \leq \sigma_1$ for unit vectors, and agrees to six digits after 100 rounds. One cold-start round is 30% low because a Gaussian 256×128 matrix has a crowded top spectrum (its largest singular value sits near $\sqrt{256} + \sqrt{128} \approx 27.3$ with neighbours close behind); that is why Miyato recycles $\tilde{u}$ across steps. Normalization pins the spectral norm at 1 but leaves Frobenius at 6.64 and nuclear at 70, and the submultiplicativity ratio never exceeds 1, as the proof requires.

## 10. Where this shows up in practice

**Spectral normalization in PyTorch.** `torch.nn.utils.parametrizations.spectral_norm` divides a layer's weight by a power-iteration estimate of $\sigma(W)$ on every forward pass, with `n_power_iterations=1` by default and $u$ kept between calls, the recycle trick of Algorithm 1 ([PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.spectral_norm.html)). It uses Section 3's operator 2-norm, the (N2) scaling argument and the proof of Section 8.

**Gradient clipping by global norm.** `torch.nn.utils.clip_grad_norm_` treats all gradients as one vector, measures its norm (ℓ2 by default), and rescales it if it exceeds `max_norm` ([PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)). The ruler is the design: global ℓ2 clipping preserves the gradient's direction, while clipping each entry separately (an ℓ∞ projection) does not, and (N2) is why rescaling bounds the step.

**Adversarial examples and the dual norm.** Goodfellow, Shlens and Szegedy's *Explaining and Harnessing Adversarial Examples* (2014, [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)) introduced the fast gradient sign method: perturb the input by $\varepsilon \operatorname{sign}(\nabla_x J)$. That is Section 4's dual-norm computation: inside an ℓ∞ ball of radius $\varepsilon$ the best first-order move is that corner, and the loss rises by about $\varepsilon \lVert \nabla_x J \rVert_1$. Since $\lVert g \rVert_1$ can reach $\sqrt{d}\,\lVert g \rVert_2$ (Section 5), a tiny per-pixel budget on a large image moves the loss a lot.

## What to remember

> - A norm is three axioms; its unit ball is the whole story: diamond (ℓ1), circle (ℓ2), square (ℓ∞).
> - Spectral, Frobenius and nuclear norms are the max, the ℓ2 and the sum of the singular values.
> - The dual norm is the best inner product over the unit ball, and Hölder follows from it.
> - Norm-equivalence constants grow like $\sqrt{d}$: 64 in 4096 dimensions.
> - Operator norms are submultiplicative when the middle norm matches; that is spectral normalization's guarantee.

## Where next

If the notation still feels foreign, start with [how to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper). The same-wave posts on [orthogonality, projections and QR](/blog/machine-learning/mathematics/orthogonality-projections-and-qr) and on [rank, null space and low-rank structure](/blog/machine-learning/mathematics/rank-null-space-and-low-rank-structure) build on the ℓ2 geometry here, and [proof techniques for ML researchers](/blog/machine-learning/mathematics/proof-techniques-for-ml-researchers) covers the induction move.

Two later posts finish what this one starts. "SVD: the decomposition deep learning runs on" proves the ellipse picture. "Spectral norm, Lipschitz constants and stability" takes Miyato's Eq. 7 through attention and residual connections, where the product bound gets loose.

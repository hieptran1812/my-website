---
title: "Determinants, Volume and Invertibility: Why Flows Need Triangular Jacobians"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "Read the determinant as the factor by which a matrix scales volume, prove the triangular rule line by line, and decode the RealNVP coupling-layer log-determinant."
tags: ["mathematics", "math-for-ai", "linear-algebra", "determinant", "jacobian", "normalizing-flows", "realnvp", "log-det", "change-of-variables"]
category: "machine-learning"
subcategory: "Mathematics"
author: "Hiep Tran"
featured: true
readTime: 18
---

> **Level:** Core · **Needs:** [Matrices as linear maps](/blog/machine-learning/mathematics/matrices-as-linear-maps-what-a-layer-does), [Rank, null space and low-rank structure](/blog/machine-learning/mathematics/rank-null-space-and-low-rank-structure), [Sums, products, einsum and log-sum-exp](/blog/machine-learning/mathematics/sums-products-einsum-and-log-sum-exp) · **Unlocks:** RealNVP, Glow and NICE, change-of-variables likelihoods, Gaussian log-likelihoods, flow and diffusion likelihood appendices

Section 3.3 of *Density estimation using Real NVP* (Dinh, Sohl-Dickstein and Bengio, ICLR 2017) writes down a 2 by 2 block matrix of derivatives, says "given the observation that this Jacobian is triangular", and computes its determinant in one line as $\exp\bigl[\sum_j s(x_{1:d})_j\bigr]$. Without that line, every training step on a CIFAR image would need the determinant of a 3072 by 3072 matrix.

Paper math is a small, closed vocabulary: learn each idea once, with a picture, a derivation, a proof and a numerical check, and the appendix stops being a wall. By the end of this post you will read Eq. 6 of RealNVP symbol by symbol, prove the rule it relies on, and check the result with ten lines of NumPy.

![A grey dashed unit square of area 1 and its image under A = [2 −1; 1 1], a blue parallelogram on the columns (2, 1) and (−1, 1) with area 3 = det A](/imgs/blogs/determinants-volume-and-invertibility-1.webp)

The picture above is the whole idea. A matrix moves the unit square to a parallelogram, and the area of that parallelogram is the determinant.

## The determinant is a volume scale factor

**What it is for:** the determinant tells you how much a linear map stretches or shrinks space, as one number.

Take a rubber sheet with a grid drawn on it, and pull it with a linear map. Every grid square becomes the same parallelogram, just shifted. So the area of one image square tells you how every region changed size. The analogy breaks in one place: a matrix can flip the sheet over, which needs a sign.

Now the small numbers. The matrix in the figure is

$$
A = \begin{pmatrix} 2 & -1 \\ 1 & 1 \end{pmatrix}.
$$

Its columns are where the two basis vectors land: $Ae_1 = (2, 1)$ and $Ae_2 = (-1, 1)$. The unit square with corners $(0,0)$, $(1,0)$, $(1,1)$, $(0,1)$ lands on the parallelogram with corners $(0,0)$, $(2,1)$, $(1,2)$, $(-1,1)$.

By hand: the box $x \in [-1, 2]$, $y \in [0, 2]$ has area 6. Removing the parallelogram leaves four right triangles, two with legs 2 and 1 (area 1 each) and two with legs 1 and 1 (area one half each), total 3. So the parallelogram has area 6 minus 3, which is 3.

### The 2 by 2 formula

**Definition.** For a 2 by 2 matrix the **determinant** is

$$
\det \begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc.
$$

In words: main diagonal product minus the other diagonal product. Here $\det A = 2 \cdot 1 - (-1) \cdot 1 = 2 + 1 = 3$, the same area the triangles gave.

The formula holds for every layout of the columns once we allow a sign, which is the section after next.

### Three properties that pin it down

The determinant is the unique function on $n$ by $n$ matrices with three properties, each a statement about volume:

1. **Normalisation:** $\det I = 1$. The identity leaves the unit cube alone.
2. **Linearity in each column:** doubling one column doubles the volume.
3. **Alternating:** if two columns are equal, the determinant is zero. A box with two equal edges is flat.

Together they force a column swap to flip the sign. We state uniqueness without proof (Strang, *Introduction to Linear Algebra*, Chapter 5).

In a 4096-dimensional embedding space, a linear layer $W$ sends the unit cube to a slanted box and $\lvert \det W \rvert$ is that box's volume.

## The sign is orientation

**What it is for:** the sign of the determinant tells you whether the map kept space the same way round or flipped it like a mirror.

Turning from $e_1$ to $e_2$ the short way is counterclockwise. If turning from $Ae_1$ to $Ae_2$ is still counterclockwise, the determinant is positive; if clockwise, the map flipped the sheet and the determinant is negative. A mirror turns a left hand into a right one.

![Three planes: a shear keeps the counterclockwise turn (det +1), a mirror reverses it (det −1), and a singular map flattens the square onto the line y = x, where inputs (1, 0) and (0, 2) both land on (1, 1) (det 0, no inverse)](/imgs/blogs/determinants-volume-and-invertibility-2.webp)

In the left two panels, the shear $S = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ has $\det S = 1 \cdot 1 - 1 \cdot 0 = 1$: a slant, same area, same turn. The mirror $F = \begin{pmatrix} -1 & 0 \\ 0 & 1 \end{pmatrix}$ has $\det F = (-1) \cdot 1 - 0 \cdot 0 = -1$: same area, but the turn from $Fe_1$ to $Fe_2$ is clockwise.

In the animation the first column stays at $(1, 0)$ while the second column $(0.5, h)$ slides its height $h$ from 1 down to $-1$ and back, so the signed area is exactly $h$.

<figure class="blog-anim">
<svg viewBox="0 0 640 340" role="img" aria-label="A parallelogram with first column (1, 0) and second column (0.5, h); as h slides from 1 to minus 1 the shape flattens to a line at h = 0 and then flips below the axis, while a signed-area bar shrinks through zero and grows on the negative side" style="width:100%;height:auto;max-width:760px">
<style>
.d1-ax{stroke:var(--border,#d1d5db);stroke-width:1.5}
.d1-sq{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:1.5;stroke-dasharray:5 5}
.d1-c1{stroke:var(--text-primary,#1f2937);stroke-width:3.5;stroke-linecap:round}
.d1-c2{stroke:var(--text-primary,#1f2937);stroke-width:3.5;stroke-linecap:round}
.d1-par{stroke:var(--text-primary,#1f2937);stroke-width:1.5;fill-opacity:.45}
.d1-bar{fill-opacity:.8}
.d1-t{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.d1-s{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
@keyframes d1-sy{0%,12%{transform:scaleY(1)}88%,100%{transform:scaleY(-1)}}
@keyframes d1-sx{0%,12%{transform:scaleX(1)}88%,100%{transform:scaleX(-1)}}
@keyframes d1-col{0%,49%{fill:var(--accent,#6366f1)}51%,100%{fill:#f08c00}}
@keyframes d1-pos{0%,40%{opacity:1}46%,100%{opacity:0}}
@keyframes d1-zero{0%,42%{opacity:0}50%{opacity:1}58%,100%{opacity:0}}
@keyframes d1-neg{0%,54%{opacity:0}60%,100%{opacity:1}}
.d1-g{transform-box:view-box;transform-origin:150px 190px;animation:d1-sy 12s ease-in-out infinite alternate}
.d1-b{transform-box:view-box;transform-origin:470px 190px;animation:d1-sx 12s ease-in-out infinite alternate}
.d1-fill{fill:var(--accent,#6366f1);animation:d1-col 12s ease-in-out infinite alternate}
.d1-lp{animation:d1-pos 12s ease-in-out infinite alternate}
.d1-lz{opacity:0;animation:d1-zero 12s ease-in-out infinite alternate}
.d1-ln{opacity:0;animation:d1-neg 12s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){.d1-g,.d1-b,.d1-fill,.d1-lp,.d1-lz,.d1-ln{animation:none}}
</style>
<text class="d1-t" x="20" y="36">columns (1, 0) and (0.5, h): h slides from 1 to −1</text>
<line class="d1-ax" x1="40" y1="190" x2="320" y2="190"/>
<line class="d1-ax" x1="150" y1="70" x2="150" y2="310"/>
<rect class="d1-sq" x="150" y="100" width="90" height="90"/>
<g class="d1-g">
<polygon class="d1-par d1-fill" points="150,190 240,190 285,100 195,100"/>
<line class="d1-c2" x1="150" y1="190" x2="195" y2="100"/>
</g>
<line class="d1-c1" x1="150" y1="190" x2="240" y2="190"/>
<text class="d1-s" x="40" y="330">dashed: the unit square, area 1</text>
<text class="d1-t" x="360" y="140">signed area = det = h</text>
<line class="d1-ax" x1="350" y1="190" x2="590" y2="190"/>
<line class="d1-ax" x1="470" y1="160" x2="470" y2="220"/>
<rect class="d1-bar d1-fill d1-b" x="470" y="175" width="120" height="30" rx="3"/>
<text class="d1-s" x="340" y="245">−1</text>
<text class="d1-s" x="466" y="245">0</text>
<text class="d1-s" x="580" y="245">+1</text>
<text class="d1-t d1-lp" x="360" y="290">det &gt; 0: same turning direction</text>
<text class="d1-t d1-lz" x="360" y="290">det = 0: flat, no inverse</text>
<text class="d1-t d1-ln" x="360" y="290">det &lt; 0: orientation flipped</text>
</svg>
<figcaption>Signed area in motion: as the second column's height h passes through zero, the parallelogram flattens onto the axis (det = 0, no inverse) and then reappears flipped below it. The area bar tracks det = h, changing colour when the sign changes.</figcaption>
</figure>

So the determinant carries two pieces of information. Its absolute value $\lvert \det A \rvert$ is the volume scale factor. Its sign is orientation. Density formulas need only the first, hence $\lvert \det \cdot \rvert$ in papers.

## Zero means collapse, and collapse means no inverse

**What it is for:** a zero determinant is the cheapest test there is for "this map throws information away".

Press the rubber sheet flat onto a ruler: many points share each spot, so from where a point landed you cannot say where it started. No inverse, and zero area.

The right panel of the figure shows it with $C = \begin{pmatrix} 1 & 0.5 \\ 1 & 0.5 \end{pmatrix}$, whose determinant is $1 \cdot 0.5 - 0.5 \cdot 1 = 0$. Check two inputs by hand: $C(1, 0) = (1, 1)$ and $C(0, 2) = (0 + 1, 0 + 1) = (1, 1)$. Two different inputs, one output, so no function can undo $C$.

**Theorem (invertibility).** For a square matrix $A$, the following are equivalent:

- $\det A \neq 0$;
- $A$ is invertible;
- the columns of $A$ are linearly independent (full rank);
- $Ax = 0$ only for $x = 0$ (trivial null space).

In words: four descriptions of one thing. The link to rank is in the [rank and null space post](/blog/machine-learning/mathematics/rank-null-space-and-low-rank-structure): a nonzero $v$ with $Av = 0$ means a whole line is crushed to a point, and the unit cube flattens to zero volume.

In floating point a determinant is almost never exactly zero; a later post on the condition number makes "almost crushed" precise.

## Scale factors multiply

**What it is for:** this rule turns the determinant of a deep stack of layers into a sum of per-layer terms, which is the engine of every normalizing flow.

Enlarge a photo by 2 on a copier, then enlarge the copy by 1.5. The final area is 3 times the original, not 3.5.

![Three panels on one scale: the unit square (area 1) becomes a 2 by 1 rectangle under B = diag(2, 1) (area 2), then a parallelogram under A = [1 0.5; 0 1.5] (area 3), so det(AB) = 1.5 × 2 = 3](/imgs/blogs/determinants-volume-and-invertibility-3.webp)

With the figure's $B = \operatorname{diag}(2, 1)$ and $A = \begin{pmatrix} 1 & 0.5 \\ 0 & 1.5 \end{pmatrix}$, $\det B = 2$, $\det A = 1.5$, and multiplying out gives $AB = \begin{pmatrix} 2 & 0.5 \\ 0 & 1.5 \end{pmatrix}$ with $\det(AB) = 2 \cdot 1.5 - 0.5 \cdot 0 = 3$.

**Theorem (product rule).** For square matrices $A$ and $B$ of the same size,

$$
\det(AB) = \det A \cdot \det B.
$$

In words: doing $B$ then $A$ scales volume by the product of the two factors. We state it without proof (Strang, Chapter 5). Since $A A^{-1} = I$, it gives $\det(A^{-1}) = 1 / \det A$.

Take logs of the product rule, and a chain of $K$ layers gives

$$
\log \lvert \det(A_K \cdots A_2 A_1) \rvert = \sum_{k=1}^{K} \log \lvert \det A_k \rvert.
$$

In words: log-volume changes of stacked layers add, which is how a flow paper computes its likelihood.

## Triangular matrices are the cheap case

**What it is for:** if a matrix is triangular, its determinant costs $n$ multiplications instead of a full factorisation.

A general $n$ by $n$ determinant costs about $\tfrac{2}{3}n^{3}$ floating-point operations through LU factorisation. For a 3072-dimensional image that is about ${2}{\times}10^{10}$ per sample per step. A **triangular** matrix has all zeros above the diagonal (lower triangular) or below it (upper triangular).

**The rule:** the determinant of a triangular matrix is the product of its diagonal entries. Check it on a 3 by 3 example:

$$
L = \begin{pmatrix} 2 & 0 & 0 \\ 5 & 3 & 0 \\ -1 & 4 & 0.5 \end{pmatrix}, \qquad \det L = 2 \cdot 3 \cdot 0.5 = 3.
$$

Expanding along the first row (see the proof), only the corner 2 contributes, times the lower-right block's determinant $3 \cdot 0.5 - 0 \cdot 4 = 1.5$, so $\det L = 3$. The entries below the diagonal never mattered.

The geometric reason: column $j$ of a lower-triangular matrix reaches exactly $L_{jj}$ along its own axis $e_j$, plus a tilt toward later axes only. Each such tilt is a shear, and shears do not change volume, as the shear panel above showed. Straighten every tilt and you are left with an axis-aligned box with sides $L_{11}, \dots, L_{nn}$.

## Log-determinants: the form papers actually use

**What it is for:** working with $\log \lvert \det \rvert$ instead of $\det$ keeps the number in a range a computer can store.

A determinant is a product of many factors, and long products explode or vanish. A log scale is the fix, as with earthquake magnitudes: it turns multiplying into adding.

![Log-determinant against D for diagonal entries of 0.5: the sum of logs falls in a straight line to −207.94 at D = 300, while the float32 product matches it to −103.28 at D = 149 and then drops to −∞ at D = 150](/imgs/blogs/determinants-volume-and-invertibility-6.webp)

Real NumPy numbers: a diagonal matrix with $D$ entries all equal to 0.5 has $\log \det = D \log 0.5 = -0.6931\,D$. In float32, the product of the diagonal is ${2}^{-149} \approx {1.4}{\times}10^{-45}$ at $D = 149$, the smallest positive float32; at $D = 150$ it rounds to 0 and its log is $-\infty$.

At CIFAR-10 size, 3072 dimensions, with every diagonal entry 0.9 the true log-determinant is $3072 \log 0.9 = -323.67$, a determinant of about ${10}^{-140.6}$, but the float32 product returns the denormal ${6}{\times}10^{-45}$, wrong by about 96 orders of magnitude. With every entry equal to 1.1 the true log-determinant is 292.79 and the float32 product is `inf`.

**The identity:**

$$
\log \lvert \det L \rvert = \sum_{i=1}^{n} \log \lvert L_{ii} \rvert \quad \text{for triangular } L.
$$

In words: for a triangular matrix, the log-determinant is the sum of the logs of the diagonal; the sum stays representable. For a general matrix, `numpy.linalg.slogdet` and `torch.linalg.slogdet` factor $A = PLU$, apply this identity to $U$, and return the sign separately.

## Why flows care: the change-of-variables formula, previewed

**What it is for:** the determinant is the correction that keeps a probability density summing to one after you move space around.

Stretch a strip of evenly spread sand to twice its length and the sand is half as thick: density drops by the factor volume grew.

The smallest example: $X$ uniform on $[0, 1]$ has density 1 there. The variable $Y = 2X$ is uniform on $[0, 2]$, and since its total probability is still 1, its density is one half.

In $D$ dimensions the stretching factor near a point is the absolute determinant of the Jacobian (the matrix of partials $\partial f_i / \partial x_j$), since a smooth map looks linear up close. If $f$ is a bijection from data space to a latent space with a simple density $p_Z$, then

$$
p_X(x) = p_Z\bigl(f(x)\bigr) \left\lvert \det \frac{\partial f(x)}{\partial x^{\top}} \right\rvert .
$$

In words: the density of a data point is the latent density where it lands, times how much $f$ stretches volume there. This is Eq. 2 of RealNVP. A later post proves it; the [normalizing flows post](/blog/machine-learning/image-generation/normalizing-flows-and-change-of-variables) covers the models built on it.

Maximum likelihood needs $\log \lvert \det \partial f / \partial x^{\top} \rvert$ for every sample at every step, for a dense $D$ by $D$ matrix. RealNVP's answer: design $f$ so its Jacobian is triangular.

## Decoding the RealNVP coupling layer

![Affine coupling layer: x₁ is copied to y₁ and also feeds a network that outputs s and t; x₂ is multiplied by exp(s) and shifted by t to give y₂; notes give the log-determinant as the sum of s and the closed-form inverse](/imgs/blogs/determinants-volume-and-invertibility-5.webp)

The **affine coupling layer** splits $x \in \mathbb{R}^{D}$ at an index $d$:

$$
\begin{aligned}
y_{1:d} &= x_{1:d}, \\
y_{d+1:D} &= x_{d+1:D} \odot \exp\bigl(s(x_{1:d})\bigr) + t(x_{1:d}).
\end{aligned}
$$

These are Eqs. 4 and 5 of the paper. In words: copy the first half; scale and shift the second by amounts computed from the first.

A hand-checkable case with $D = 2$, $d = 1$: let $s(x_1) = 0.5\,x_1$ and $t(x_1) = x_1^{2}$, and evaluate at $x = (2, 1)$. Then $y_1 = 2$ and $y_2 = 1 \cdot e^{1} + 4 \approx 6.718$. The four partial derivatives are $\partial y_1 / \partial x_1 = 1$, $\partial y_1 / \partial x_2 = 0$, $\partial y_2 / \partial x_2 = e^{1} \approx 2.718$, and

$$
\frac{\partial y_2}{\partial x_1} = x_2 \, e^{0.5 x_1} \cdot 0.5 + 2 x_1 = 0.5\,e + 4 \approx 5.359 .
$$

So the Jacobian is $\begin{pmatrix} 1 & 0 \\ 5.359 & 2.718 \end{pmatrix}$, its determinant is $1 \cdot 2.718 - 0 \cdot 5.359 = 2.718 = e^{1}$, and the log-determinant is exactly $s(x_1) = 1$. The messy entry 5.359 did not matter.

![A 4 by 4 heatmap of a coupling-layer Jacobian: a zero block above the diagonal, an amber block below it that is never needed, and the diagonal 1, 1, 0.905, 2.460 multiplying to det J = 2.226, with log det J = −0.1 + 0.9 = 0.8](/imgs/blogs/determinants-volume-and-invertibility-4.webp)

For $D = 4$, $d = 2$ and $s = (-0.1, 0.9)$, the diagonal is $(1, 1, e^{-0.1}, e^{0.9}) = (1, 1, 0.905, 2.460)$, its product is 2.226, and $\log 2.226 = 0.8 = -0.1 + 0.9$.

### Paper sentence, decoded

**Source:** Laurent Dinh, Jascha Sohl-Dickstein and Samy Bengio, *Density estimation using Real NVP*, ICLR 2017, [arXiv:1605.08803](https://arxiv.org/abs/1605.08803), Section 3.3, Eq. 6 and the sentence after it.

$$
\frac{\partial y}{\partial x^{\top}} = \begin{bmatrix} \mathbb{I}_d & 0 \\ \dfrac{\partial y_{d+1:D}}{\partial x_{1:d}^{\top}} & \operatorname{diag}\bigl(\exp\left[s(x_{1:d})\right]\bigr) \end{bmatrix}
$$

followed by: "given the observation that this Jacobian is triangular, we can efficiently compute its determinant as $\exp\bigl[\sum_j s(x_{1:d})_j\bigr]$."

| Symbol | What it is | Built in this post |
| --- | --- | --- |
| $\partial y / \partial x^{\top}$ | the $D$ by $D$ Jacobian, entry $(i, j)$ is $\partial y_i / \partial x_j$ | change-of-variables section |
| $\mathbb{I}_d$ | identity: the copied half has slope 1 on itself | 2 by 2 example, entry $\partial y_1 / \partial x_1 = 1$ |
| ${0}$ (top right) | the copied half does not depend on $x_{d+1:D}$ | triangular section |
| $\partial y_{d+1:D} / \partial x_{1:d}^{\top}$ | how the scaled half responds to the copied half; involves $s'$ and $t'$ | the 5.359 entry |
| $\operatorname{diag}(\exp[s])$ | each scaled entry sees only its own input, slope $e^{s_j}$ | diagonal of the heatmap |
| $\exp[\sum_j s_j]$ | product of the diagonal, $\prod_j e^{s_j}$ | triangular rule, then exp of a sum |

In words: the Jacobian is lower triangular with a positive diagonal, so its determinant is the product $\prod_j e^{s_j} = e^{\sum_j s_j}$, and the log-determinant the loss needs is just $\sum_j s_j$. It "does not involve computing the Jacobian of s or t", so those can be deep convolutional networks.

The inverse is closed form too, $x_{d+1:D} = \bigl(y_{d+1:D} - t(y_{1:d})\bigr) \odot \exp\bigl(-s(y_{1:d})\bigr)$, since $y_{1:d} = x_{1:d}$. RealNVP alternates which half is copied, and by the product rule the stacked log-determinants add.

## Proof, read line by line

**Statement.** Let $L$ be an $n$ by $n$ matrix with entries $L_{ij}$. Assume

- (A1) $L$ is square, $n \geq 1$;
- (A2) $L$ is lower triangular: $L_{ij} = 0$ whenever $j \gt i$.

Then $\det L = \prod_{i=1}^{n} L_{ii}$. Moreover, for the affine coupling map of Eqs. 4 and 5, assume

- (A3) $s$ and $t$ are differentiable functions of $x_{1:d}$ only.

Then its Jacobian satisfies (A2), and $\log \lvert \det \partial y / \partial x^{\top} \rvert = \sum_{j=1}^{D-d} s_j(x_{1:d})$.

**The idea in one sentence:** expand along the first row, where (A2) leaves only one nonzero entry, and the remaining block is again lower triangular, so induction on $n$ peels off one diagonal entry at a time.

We use the **cofactor (Laplace) expansion** along the first row as the working formula for the determinant:

$$
\det L = \sum_{j=1}^{n} (-1)^{1+j} L_{1j} \det M_{1j},
$$

where $M_{1j}$ is the $(n-1)$ by $(n-1)$ matrix left after deleting row 1 and column $j$. It follows from the three properties (Strang, Section 5.2); for $n = 2$ it gives $ad - bc$.

**Part 1: the triangular rule, by induction on $n$.**

1. Base case $n = 1$: $\det L = L_{11} = \prod_{i=1}^{1} L_{ii}$. *By definition of a 1 by 1 determinant; uses (A1).*
2. Induction hypothesis: the rule holds for every lower-triangular matrix of size $n - 1$. *Standard induction setup.*
3. Expand $\det L$ along row 1: $\det L = \sum_{j=1}^{n} (-1)^{1+j} L_{1j} \det M_{1j}$. *Cofactor expansion.*
4. For $j \geq 2$ we have $j \gt 1$, so $L_{1j} = 0$. *By (A2) with $i = 1$.*
5. Hence $\det L = (-1)^{2} L_{11} \det M_{11} = L_{11} \det M_{11}$. *Every other term is zero by step 4.*
6. $M_{11}$ is $L$ with row 1 and column 1 removed, so its entry in row $a$, column $b$ is $L_{a+1,\,b+1}$. *Reindexing.*
7. If $b \gt a$ then $b + 1 \gt a + 1$, so $L_{a+1,\,b+1} = 0$: $M_{11}$ is lower triangular, with diagonal $L_{22}, \dots, L_{nn}$. *By (A2) applied to the shifted indices.*
8. By the hypothesis, $\det M_{11} = \prod_{i=2}^{n} L_{ii}$. *Step 2 applied to $M_{11}$, which has size $n - 1$.*
9. Combining steps 5 and 8: $\det L = L_{11} \prod_{i=2}^{n} L_{ii} = \prod_{i=1}^{n} L_{ii}$. *Algebra; this closes the induction.*

For upper-triangular matrices, expand along the first column instead.

**Part 2: the coupling Jacobian is lower triangular.** Write $J_{ij} = \partial y_i / \partial x_j$.

10. For $i \leq d$: $y_i = x_i$, so $J_{ij} = 1$ if $j = i$ and ${0}$ otherwise. *Differentiate Eq. 4. Rows 1 to $d$ are zero above the diagonal.*
11. For $i \gt d$ and $j \gt d$: $y_i = x_i \exp\bigl(s_{i-d}(x_{1:d})\bigr) + t_{i-d}(x_{1:d})$, and by (A3) neither $s$ nor $t$ depends on $x_j$. *Read off Eq. 5.*
12. So $J_{ij} = \exp(s_{i-d})$ if $j = i$ and ${0}$ if $j \neq i$, for $i, j \gt d$. *Only the explicit factor $x_i$ depends on $x_j$, and only when $j = i$.*
13. Entries with $j \gt i$ fall in step 10 (for $i \leq d$) or step 12 with $j \neq i$ (for $i \gt d$), so they are all zero: (A2) holds. *Case split on $i$.*
14. By Part 1, $\det J = \prod_{i=1}^{d} 1 \cdot \prod_{j=1}^{D-d} e^{s_j} = \exp\bigl(\sum_j s_j\bigr)$, which is positive, so $\log \lvert \det J \rvert = \sum_{j=1}^{D-d} s_j$. *Product of exponentials is the exponential of the sum.*

**Where each assumption was used:**

| Assumption | Used in step |
| --- | --- |
| (A1) square matrix | 1, 3 (cofactor expansion needs a square matrix) |
| (A2) zeros above the diagonal | 4, 7, and 14 via Part 1 |
| (A3) $s, t$ see only $x_{1:d}$ | 11, 12, 13 |

**What breaks without (A2).** The $\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$ has diagonal product 1 but determinant 1 minus 1, which is 0. In the coupling layer, (A3) guards this: if $s$ read $x_{d+1:D}$ too, the lower-right block would gain entries above its diagonal and the shortcut would silently be wrong.

**Proof moves used:** `induction (layers / steps)` · `change of variables` (as the motivation for the Jacobian).

## Check it numerically

The script builds a coupling layer with a random network for $s$ and $t$, takes its full 6 by 6 Jacobian by central differences, and compares $\sum_j s_j$ against `slogdet`.

```python
import numpy as np

rng = np.random.default_rng(0)
D, d, H = 6, 3, 16                      # input dim, pass-through dim, hidden width
W1, W2 = rng.normal(size=(H, d)), rng.normal(size=(2 * (D - d), H)) * 0.5

def coupling(x):                         # RealNVP affine coupling, Eq. 4-5
    h = np.tanh(W1 @ x[:d])
    s, t = np.split(W2 @ h, 2)          # scale and translation, each R^(D-d)
    y = x.copy()
    y[d:] = x[d:] * np.exp(s) + t
    return y, s

x = rng.normal(size=D)
y, s = coupling(x)

eps, J = 1e-6, np.zeros((D, D))          # full Jacobian by central differences
for j in range(D):
    e = np.zeros(D); e[j] = eps
    J[:, j] = (coupling(x + e)[0] - coupling(x - e)[0]) / (2 * eps)

sign, logdet_full = np.linalg.slogdet(J)
print("upper triangle max |J_ij|:", np.abs(np.triu(J, 1)).max())
print("diag of J            :", np.round(np.diag(J), 4))
print("log-det, sum of s    :", s.sum())
print("log-det, slogdet(J)  :", logdet_full, "sign", sign)
print("abs difference       :", abs(s.sum() - logdet_full))
```

```console
upper triangle max |J_ij|: 0.0
diag of J            : [1.     1.     1.     1.4143 1.0639 0.2023]
log-det, sum of s    : -1.1895741846942598
log-det, slogdet(J)  : -1.1895741849476824 sign 1.0
abs difference       : 2.534226162254072e-10
```

The upper triangle is exactly zero, as Part 2 of the proof predicts, and the two log-determinants agree to about ${2.5}{\times}10^{-10}$. That residual is finite-difference round-off, machine epsilon over the step ${10}^{-6}$, about ${10}^{-10}$. When I let the network read all of `x`, the upper triangle filled in and the shortcut gave $-0.478$ against `slogdet`'s 0.235: simply wrong.

## Where this shows up in practice

**Glow's invertible 1 by 1 convolution.** Glow (Kingma and Dhariwal, 2018, [arXiv:1807.03039](https://arxiv.org/abs/1807.03039)), Section 3.2, mixes channels with a learned $c$ by $c$ matrix $W$ at every pixel, with log-determinant $h \cdot w \cdot \log \lvert \det W \rvert$ (Eq. 9): the product rule over $h \cdot w$ identical blocks. To avoid an $O(c^{3})$ determinant it parameterises $W = PL(U + \operatorname{diag}(s))$ with $P$ a permutation, $L$ unit lower triangular and $U$ strictly upper triangular (Eq. 10). The two rules then give $\log \lvert \det W \rvert = \sum \log \lvert s \rvert$ (Eq. 11), an $O(c)$ cost.

**NICE's volume-preserving coupling.** RealNVP's predecessor, NICE (Dinh, Krueger and Bengio, 2014, [arXiv:1410.8516](https://arxiv.org/abs/1410.8516)), uses additive coupling, $y_{d+1:D} = x_{d+1:D} + m(x_{1:d})$: a lower-triangular Jacobian with ones on the diagonal, determinant exactly 1. It is the shear panel of the second figure in thousands of dimensions.

**Gaussian log-likelihoods via Cholesky.** A multivariate Gaussian's log-density contains $-\tfrac{1}{2}\log \det \Sigma$. With the Cholesky factorisation $\Sigma = LL^{\top}$, $L$ lower triangular, the product rule gives $\det \Sigma = (\det L)^{2}$ and the triangular rule gives $\log \det \Sigma = 2 \sum_i \log L_{ii}$. PyTorch's [`torch.distributions.MultivariateNormal`](https://pytorch.org/docs/stable/distributions.html#multivariatenormal) stores this `scale_tril` factor and reads the log-determinant off its diagonal.

## What to remember

> - $\lvert \det A \rvert$ is the factor by which $A$ scales volume; the sign records whether $A$ flips orientation.
> - $\det A = 0$ exactly when $A$ collapses some direction, which is exactly when $A$ has no inverse.
> - $\det(AB) = \det A \det B$, so log-determinants of stacked layers add.
> - A triangular matrix's determinant is the product of its diagonal.
> - Papers write $\log \lvert \det \rvert$ because products of thousands of factors underflow or overflow; RealNVP designs a triangular Jacobian so the log-determinant is just $\sum_j s_j$.

## Where next

New to the series? Start with [how to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper). The induction proof above uses the moves catalogued in [proof techniques for ML researchers](/blog/machine-learning/mathematics/proof-techniques-for-ml-researchers), and the log-of-a-product trick is developed in [sums, products, einsum and log-sum-exp](/blog/machine-learning/mathematics/sums-products-einsum-and-log-sum-exp).

Within this wave, [matrices as linear maps](/blog/machine-learning/mathematics/matrices-as-linear-maps-what-a-layer-does) gives the column picture, [rank, null space and low-rank structure](/blog/machine-learning/mathematics/rank-null-space-and-low-rank-structure) expands the collapse story, and [orthogonality, projections and QR](/blog/machine-learning/mathematics/orthogonality-projections-and-qr) covers the maps with $\lvert \det Q \rvert = 1$. Later posts cover eigenvalues, where the determinant becomes their product, and the full change-of-variables theorem.

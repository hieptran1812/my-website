---
title: "Orthogonality, Projections and QR: The Geometry Behind Least Squares and Muon"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "Learn orthonormal bases, orthogonal matrices, projection matrices, Gram–Schmidt and QR with pictures and hand-checkable numbers, prove that the least-squares residual is orthogonal to the column space, and decode the orthogonalised update in Muon."
tags: ["mathematics", "math-for-ai", "linear-algebra", "orthogonality", "projections", "qr-decomposition", "least-squares", "gram-schmidt", "muon", "optimizers"]
category: "machine-learning"
subcategory: "Mathematics"
author: "Hiep Tran"
featured: true
readTime: 18
---

> **Level:** Core · **Needs:** [Vectors, dot products and the geometry of embeddings](/blog/machine-learning/mathematics/vectors-dot-products-and-the-geometry-of-embeddings), [Matrices as linear maps](/blog/machine-learning/mathematics/matrices-as-linear-maps-what-a-layer-does), [Norms, inner products and distances](/blog/machine-learning/mathematics/norms-inner-products-and-distances) · **Unlocks:** Muon and Shampoo-style optimizers, orthogonal initialisation, linear probes, randomized SVD, PCA

Section 2.1 of *Muon is Scalable for LLM Training* (Liu et al., 2025) describes an optimizer used to train a 16B-parameter mixture-of-experts model, and its core fits on one line: $O_t = \text{Newton-Schulz}(M_t)$. A few sentences later the paper explains what that is approximating: $(M_t M_t^\top)^{-1/2} M_t = U V^\top$. Why would anyone throw away the singular values of a momentum matrix, and why would that help?

By the end you will read both lines, because they are built from **orthogonality** and its working form, **projection**, the same ideas behind least squares, QR, orthogonal initialisation and randomized SVD. Paper math is a small, closed vocabulary; learn each idea once, with a picture, a derivation, a proof and a numerical check, and the appendix stops being a wall.

![Projection of b = (1, 2) onto the line spanned by a = (2, 1): the foot p = (1.6, 0.8) and the perpendicular residual e](/imgs/blogs/orthogonality-projections-and-qr-1.webp)

The picture above is the whole post in one frame: the closest point to $b$ on a line (later, a subspace) is found by dropping a perpendicular, and the leftover meets the line at a right angle. Everything below is that picture, scaled up.

## 1. Orthogonality: when two directions share nothing
**What it is for:** orthogonality tells you that two directions carry completely separate information, so you can measure along one without disturbing the other.

Picture a city grid. Walking three blocks east tells you nothing about how far north you went, because the streets cross at right angles. If they crossed at 60 degrees, every eastward step would also carry you a bit north, and the two coordinates would be tangled.

In symbols, two vectors $u$ and $v$ are **orthogonal** when their dot product is zero:

$$
u \perp v \iff u^\top v = \sum_{i=1}^{n} u_i v_i = 0.
$$

In words: the component-wise products cancel exactly. Since $u^\top v = \lVert u \rVert \lVert v \rVert \cos\theta$, a zero dot product between nonzero vectors means a right angle.

Check it on the opening figure: the residual $e = (-0.6, 1.2)$ and the direction $a = (2, 1)$ give $(-0.6)(2) + (1.2)(1) = -1.2 + 1.2 = 0$. The right-angle mark is that zero.

The most useful consequence is **Pythagoras in any dimension**. If $u \perp v$, then

$$
\lVert u + v \rVert^2 = \lVert u \rVert^2 + 2\,u^\top v + \lVert v \rVert^2 = \lVert u \rVert^2 + \lVert v \rVert^2.
$$

The cross term $2\,u^\top v$ is the only thing between a general sum and Pythagoras, and orthogonality kills it. The proof in section 6 is this identity plus one add-and-subtract.

## 2. Orthonormal bases: coordinates for free
**What it is for:** an orthonormal basis lets you read off the coordinates of any vector with dot products alone, no equation solving.

Vectors $q_1, \dots, q_k$ are **orthonormal** if each has length one and every pair is orthogonal: $q_i^\top q_j = 1$ when $i = j$ and 0 otherwise. Stack them as the columns of $Q$ and the condition becomes $Q^\top Q = I$.

The payoff: if $x = c_1 q_1 + c_2 q_2$, dotting both sides with $q_1$ kills every term but the first, so $c_1 = q_1^\top x$. With a skewed basis you would solve a linear system; with an orthonormal one, each coordinate is one dot product.

Take $q_1 = \tfrac{1}{\sqrt 5}(2, 1)$ and $q_2 = \tfrac{1}{\sqrt 5}(-1, 2)$: their dot product is $\tfrac{1}{5}(-2 + 2) = 0$ and each has length 1. For $x = (1, 2)$, $c_1 = \tfrac{4}{\sqrt 5} \approx 1.789$ and $c_2 = \tfrac{3}{\sqrt 5} \approx 1.342$. Rebuild: $\tfrac{4}{5}(2, 1) + \tfrac{3}{5}(-1, 2) = (1, 2)$. The 1.789 returns in Gram–Schmidt as "the shadow".

## 3. Orthogonal matrices: rigid motions
**What it is for:** an orthogonal matrix moves vectors without stretching anything, so it changes the frame, never the geometry.

A square matrix $Q$ is **orthogonal** when $Q^\top Q = I$, so $Q^{-1} = Q^\top$. Picture rotating or flipping a rigid sheet of paper: distances drawn on it do not change.

![The rotation Q = [0.6 −0.8; 0.8 0.6] turns x and y by 53.1 degrees and keeps the unit circle, every length and the angle between them](/imgs/blogs/orthogonality-projections-and-qr-2.webp)

The figure uses the rotation built from a 3-4-5 triangle ($\cos\theta = 0.6$, $\sin\theta = 0.8$, $\theta \approx 53.13^\circ$). By hand, for $x = (2, 0.5)$:

$$
Qx = \begin{pmatrix} 0.6 & -0.8 \\ 0.8 & 0.6 \end{pmatrix} \begin{pmatrix} 2 \\ 0.5 \end{pmatrix} = \begin{pmatrix} 1.2 - 0.4 \\ 1.6 + 0.3 \end{pmatrix} = \begin{pmatrix} 0.8 \\ 1.9 \end{pmatrix}.
$$

Lengths: $\lVert x \rVert^2 = 4 + 0.25 = 4.25$ and $\lVert Qx \rVert^2 = 0.64 + 3.61 = 4.25$. For $y = (-1, 1)$, $Qy = (-1.4, -0.2)$, and $x^\top y = -1.5 = (Qx)^\top (Qy) = -1.12 - 0.38$. That agreement is a one-line theorem:

$$
(Qx)^\top (Qy) = x^\top Q^\top Q\, y = x^\top y.
$$

In words: an orthogonal matrix preserves every dot product, hence every length and angle.

A rectangular $Q \in \mathbb{R}^{m \times k}$ with $m \gt k$ and $Q^\top Q = I_k$ is **semi-orthogonal**: $k$ orthonormal columns in a bigger space. It still preserves lengths, but $Q Q^\top$ is no longer $I$; section 4 shows it is a projection. Muon produces one for each $A \times B$ weight matrix. Here the rigid-sheet picture breaks: the map embeds a small sheet in a bigger room.

## 4. Projection: the closest point
**What it is for:** projection finds the best approximation of a vector using only directions you are allowed to use.

Standing at $b$, you reach the nearest point of a straight road by walking at a right angle to it; a slanted path is a hypotenuse, hence longer.

### Onto a line

Let the line be all multiples $t\,a$. The foot $p = \hat t\, a$ is where the residual is orthogonal to $a$:

$$
a^\top (b - \hat t\, a) = 0 \quad\Longrightarrow\quad \hat t = \frac{a^\top b}{a^\top a}, \qquad p = \frac{a^\top b}{a^\top a}\, a.
$$

In words: how much of $b$ points along $a$, divided by the squared length of $a$ to undo its scale.

With $a = (2, 1)$ and $b = (1, 2)$: $a^\top b = 4$, $a^\top a = 5$, so $\hat t = 0.8$, $p = (1.6, 0.8)$ and $e = (-0.6, 1.2)$, with length $\sqrt{1.8} \approx 1.342$. The tip of $a$, $(2, 1)$, is farther from $b$: $\sqrt 2 \approx 1.414$.

### Onto a subspace

Now allow all columns of $A \in \mathbb{R}^{m \times n}$ at once. The reachable set is the **column space** $\{Ax\}$, and the residual must be orthogonal to every column. Stacking those conditions:

$$
A^\top (b - A\hat x) = 0 \quad\Longrightarrow\quad A^\top A\, \hat x = A^\top b.
$$

These are the **normal equations** ("normal" meaning perpendicular). With independent columns $A^\top A$ is invertible (step 2 of section 6), so

$$
\hat x = (A^\top A)^{-1} A^\top b, \qquad p = A (A^\top A)^{-1} A^\top b = P b.
$$

$P = A (A^\top A)^{-1} A^\top$ is the **projection matrix** onto the column space. It does not depend on $b$.

![Anatomy of P = A(AᵀA)⁻¹Aᵀ: Aᵀ measures, the inverse Gram matrix corrects for overlap, A rebuilds](/imgs/blogs/orthogonality-projections-and-qr-5.webp)

Read $P$ right to left. $A^\top b$ dots $b$ with every column: raw coordinates, already the answer if the columns were orthonormal. They usually overlap, so $(A^\top A)^{-1}$, the inverse **Gram matrix**, corrects for that. Then $A$ adds up the columns with the corrected weights.

Two properties fall out: $P^2 = P$ (projecting twice changes nothing) and $P^\top = P$. Also $I - P$ projects onto the **orthogonal complement**. With orthonormal columns $A^\top A = I$ and $P = Q Q^\top$: the correction step vanishes, which is why we orthonormalise.

## 5. Least squares is a projection

**What it is for:** least squares fits a model that cannot pass through all the data, and the geometry says the best fit is a projection.

Fit a line $y = c + d\,t$ to the points $(0, 6)$, $(1, 0)$, $(2, 0)$. No line passes through all three, so minimise the sum of squared vertical errors. The three equations $c + d\,t_i = y_i$ read $Ax \approx b$ with

$$
A = \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix}, \quad x = \begin{pmatrix} c \\ d \end{pmatrix}, \quad b = \begin{pmatrix} 6 \\ 0 \\ 0 \end{pmatrix}.
$$

Minimising $\lVert b - Ax \rVert^2$ is finding the closest point to $b$ in the column space, so the normal equations apply. Here $A^\top A$ has entries ${1+1+1} = 3$, ${0+1+2} = 3$ and ${0+1+4} = 5$, and $A^\top b = (6, 0)$. The determinant is ${15 - 9 = 6}$, so

$$
\hat x = \frac{1}{6} \begin{pmatrix} 5 & -3 \\ -3 & 3 \end{pmatrix} \begin{pmatrix} 6 \\ 0 \end{pmatrix} = \begin{pmatrix} 5 \\ -3 \end{pmatrix}.
$$

The best line is $y = 5 - 3t$, with fitted values $p = (5, 2, -1)$ and residual $e = (1, -2, 1)$.

![Least-squares line y = 5 − 3t through (0,6), (1,0), (2,0) with residuals +1, −2, +1 drawn as vertical dashed segments](/imgs/blogs/orthogonality-projections-and-qr-3.webp)

The residual is orthogonal to the first column, ${1 - 2 + 1 = 0}$, so the errors sum to zero. It is orthogonal to the second, ${0 - 2 + 2 = 0}$, so the errors are uncorrelated with $t$. Leftover correlation would let you tilt the line and do better.

For this $A$,

$$
P = \frac{1}{6} \begin{pmatrix} 5 & 2 & -1 \\ 2 & 2 & 2 \\ -1 & 2 & 5 \end{pmatrix}, \qquad Pb = \tfrac{1}{6}(30, 12, -6) = (5, 2, -1).
$$

In 4096 dimensions nothing changes: a linear probe on frozen embeddings projects the target onto the span of the features.

## 6. Proof, read line by line

Simplified from Strang, *Introduction to Linear Algebra* (5th ed.), Sections 4.2 and 4.3; no calculus needed.

**Statement.** Let $A \in \mathbb{R}^{m \times n}$ and $b \in \mathbb{R}^m$, with

- (A1) the columns of $A$ are linearly independent;
- (A2) distance is the Euclidean norm, $\lVert v \rVert^2 = v^\top v$.

Then $\hat x = (A^\top A)^{-1} A^\top b$ is the unique minimiser of $\lVert b - Ax \rVert^2$, its residual $e = b - A\hat x$ is orthogonal to every column of $A$, and $P = A(A^\top A)^{-1}A^\top$ satisfies $P^2 = P$ and $P^\top = P$.

**The idea in one sentence:** split any candidate's error into the residual plus a vector inside the column space, and let Pythagoras show the second piece can only add length.

1. Suppose $A^\top A v = 0$ for some $v$. Goal: $A^\top A$ is invertible.
2. Then $0 = v^\top A^\top A v = (Av)^\top (Av) = \lVert Av \rVert^2$, so $Av = 0$, so $v = 0$. Why: multiply by $v^\top$; a squared length is zero only for the zero vector; then (A1). So $A^\top A$ is invertible and $\hat x$ is well defined.
3. $A^\top e = A^\top b - A^\top A (A^\top A)^{-1} A^\top b = A^\top b - A^\top b = 0.$ Why: substitute $\hat x$; a matrix times its inverse is $I$. So $e$ is orthogonal to every column.
4. For any $x$: $b - Ax = (b - A\hat x) + A(\hat x - x) = e + A(\hat x - x).$ Why: add and subtract $A\hat x$.
5. $\lVert b - Ax \rVert^2 = \lVert e \rVert^2 + 2\, e^\top A(\hat x - x) + \lVert A(\hat x - x) \rVert^2.$ Why: expand the squared Euclidean norm of a sum, using (A2).
6. $e^\top A(\hat x - x) = (A^\top e)^\top (\hat x - x) = 0.$ Why: transpose rule $(A^\top e)^\top = e^\top A$, then step 3.
7. $\lVert b - Ax \rVert^2 = \lVert e \rVert^2 + \lVert A(\hat x - x) \rVert^2 \geq \lVert e \rVert^2.$ Why: steps 5 and 6; a squared norm is nonnegative.
8. Equality holds iff $A(\hat x - x) = 0$, iff $x = \hat x$. Why: by (A1), as in step 2. So $\hat x$ is the unique minimiser.
9. $P^2 = A(A^\top A)^{-1}\,(A^\top A)\,(A^\top A)^{-1}A^\top = A(A^\top A)^{-1}A^\top = P.$ Why: the middle pair cancels to $I$.
10. $P^\top = A\,\big((A^\top A)^{-1}\big)^\top A^\top = A (A^\top A)^{-1} A^\top = P.$ Why: $(XYZ)^\top = Z^\top Y^\top X^\top$; $A^\top A$ is symmetric, and the inverse of a symmetric matrix is symmetric. ∎

**Where each assumption was used:**

| Assumption | Used in |
| --- | --- |
| (A1) independent columns | step 2 (invertibility), step 8 (uniqueness) |
| (A2) Euclidean norm | step 5 (expanding the square into a cross term) |

**What breaks without it.** Drop (A2) and measure distance with the $\ell_1$ norm instead. For $b = (1, 2)$ and the line through $a = (2, 1)$, the $\ell_1$ distance from $b$ to $t\,a$ is $\lvert 1 - 2t \rvert + \lvert 2 - t \rvert$. At the perpendicular foot $t = 0.8$ it equals ${0.6 + 1.2 = 1.8}$, but at $t = 0.5$ it equals ${0 + 1.5 = 1.5}$. The closest point is no longer the perpendicular foot: only the norm built from the dot product inherits orthogonality. Drop (A1) instead (two identical columns) and $A^\top A$ is singular; $p$ still exists, but needs the pseudoinverse.

**The converse.** Any symmetric idempotent $P$ is the orthogonal projection onto its column space: split $b = Pb + (I - P)b$, and the pieces are orthogonal because $(Pb)^\top (I - P) b = b^\top (P - P^2) b = 0$. So the two properties characterise orthogonal projection.

**Proof moves used:** `add-and-subtract` · `convexity (first-order condition)` (the calculus route: the gradient of $\lVert b - Ax \rVert^2$ is $2A^\top(Ax - b)$, and setting it to zero gives the same normal equations).

## 7. Gram–Schmidt: manufacturing an orthonormal basis
**What it is for:** Gram–Schmidt straightens any independent directions into an orthonormal set spanning the same space, so the cheap formulas of section 2 apply.

Think of crooked streets you want to replace by a right-angled grid over the same neighbourhood. Keep the first street. From the second, remove the part running along the first. From the third, remove the parts along both. Rescale each to unit length.

![Gram–Schmidt in three panels: start with a₁ and a₂, normalise a₁ and find a₂'s shadow, subtract the shadow and normalise to get q₂](/imgs/blogs/orthogonality-projections-and-qr-4.webp)

The filmstrip runs it on $a_1 = (2, 1)$, $a_2 = (1, 2)$:

1. $q_1 = a_1 / \sqrt 5 \approx (0.894, 0.447)$.
2. Shadow of $a_2$: $q_1^\top a_2 = \tfrac{4}{\sqrt 5} \approx 1.789$, so the shadow is $(1.6, 0.8)$, the projection $p$ of section 4.
3. $v_2 = a_2 - (1.6, 0.8) = (-0.6, 1.2)$, length $\approx 1.342$, so $q_2 \approx (-0.447, 0.894)$.

The general step:

$$
v_j = a_j - \sum_{i \lt j} (q_i^\top a_j)\, q_i, \qquad q_j = \frac{v_j}{\lVert v_j \rVert}.
$$

In words: subtract the projection of $a_j$ onto the earlier $q$'s, then rescale. A zero $v_j$ means (A1) failed.

On the least-squares columns $a_1 = (1,1,1)$, $a_2 = (0,1,2)$: $q_1 = \tfrac{1}{\sqrt 3}(1,1,1)$; the shadow coefficient is $q_1^\top a_2 = \sqrt 3$, so the shadow is $(1,1,1)$; $v_2 = (-1, 0, 1)$ and $q_2 = \tfrac{1}{\sqrt 2}(-1, 0, 1)$.

## 8. QR: Gram–Schmidt written as a matrix product

**What it is for:** QR stores the result of Gram–Schmidt for reuse, and turns least squares into a triangular solve.

Gram–Schmidt's orthonormal vectors go into $Q$; its shadow coefficients and lengths go into an upper-triangular $R$, because column $j$ of $A$ only used $q_1, \dots, q_j$. That is the **QR decomposition**, $A = QR$.

![QR worked example: A = [1 0; 1 1; 1 2] equals Q with orthonormal columns times upper-triangular R = [√3 √3; 0 √2]](/imgs/blogs/orthogonality-projections-and-qr-6.webp)

For our matrix:

$$
\underbrace{\begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix}}_{A} = \underbrace{\begin{pmatrix} 1/\sqrt 3 & -1/\sqrt 2 \\ 1/\sqrt 3 & 0 \\ 1/\sqrt 3 & 1/\sqrt 2 \end{pmatrix}}_{Q} \underbrace{\begin{pmatrix} \sqrt 3 & \sqrt 3 \\ 0 & \sqrt 2 \end{pmatrix}}_{R}.
$$

Check the second column: $\sqrt 3\, q_1 + \sqrt 2\, q_2 = (1,1,1) + (-1, 0, 1) = (0, 1, 2)$. $R_{12} = \sqrt 3$ is the shadow removed; $R_{22} = \sqrt 2$ the length left.

Substitute $A = QR$ into the normal equations; $Q^\top Q = I$ and invertible $R$ leave

$$
R\, \hat x = Q^\top b.
$$

In words: rotate $b$ into the $Q$ frame, then back-substitute. Here $Q^\top b = (2\sqrt 3,\ -3\sqrt 2)$. The bottom row gives $d = -3$; the top row, $\sqrt 3\, c - 3\sqrt 3 = 2\sqrt 3$, gives $c = 5$. Same line, and we never formed $A^\top A$.

Forming $A^\top A$ squares the condition number, and plain Gram–Schmidt loses orthogonality on nearly dependent columns, so libraries use Householder QR or the SVD (stated without proof; see Trefethen and Bau, *Numerical Linear Algebra* (1997), Lectures 8, 10 and 11).

## 9. Decoding Muon's update

### Paper sentence, decoded

**Paper:** Jingyuan Liu, Jianlin Su, Xingcheng Yao et al., *Muon is Scalable for LLM Training*, 2025. [arXiv:2502.16982](https://arxiv.org/abs/2502.16982). Section 2.1 gives the update as Eq. 1:

$$
\begin{aligned}
M_t &= \mu M_{t-1} + \nabla \mathcal{L}_t(W_{t-1}) \\
O_t &= \text{Newton-Schulz}(M_t) \\
W_t &= W_{t-1} - \eta_t\, O_t
\end{aligned}
$$

and the text after it states the target of Newton–Schulz (unnumbered in the paper):

$$
(M_t M_t^\top)^{-1/2} M_t = U V^\top, \qquad M_t = U \Sigma V^\top.
$$

| Symbol | What it is | Built where |
| --- | --- | --- |
| $W_t$ | an $A \times B$ weight matrix after step $t$ | matrices as maps (sibling post) |
| $\nabla \mathcal{L}_t(W_{t-1})$ | the gradient, also $A \times B$ | a later calculus post |
| $\mu$, $M_t$ | momentum coefficient, momentum matrix | running average of gradients |
| $U \Sigma V^\top$ | SVD: $U$, $V$ semi-orthogonal, $\Sigma$ diagonal | section 3; the SVD post in Track C |
| $U V^\top$ | $M_t$ with every singular value set to 1 | section 3 |
| Newton-Schulz | matmul-only approximation of $U V^\top$ (Eq. 2) | below |
| $\eta_t$ | learning rate | standard |

**In words:** Muon keeps the directions of the momentum matrix and discards how hard it pushes along each, so every singular direction gets the same step size.

Why is $UV^\top$ the nearest semi-orthogonal matrix? For semi-orthogonal $O$, $\lVert O - M \rVert_F^2 = \lVert O \rVert_F^2 + \lVert M \rVert_F^2 - 2\operatorname{tr}(O^\top M)$, and the first term is the same for every such $O$. So we maximise $\operatorname{tr}(O^\top U \Sigma V^\top) = \sum_i \sigma_i (U^\top O V)_{ii}$; each diagonal entry is at most 1, and $O = UV^\top$ makes all of them 1. This is the best-approximation property of the polar decomposition (Higham, *Functions of Matrices*, 2008, Chapter 8).

By hand: $M = \begin{pmatrix} 3 & 0 \\ 4 & 5 \end{pmatrix}$ has $M^\top M = \begin{pmatrix} 25 & 20 \\ 20 & 25 \end{pmatrix}$ with eigenvalues 45 and 5, so singular values $\sqrt{45} \approx 6.708$ and $\sqrt 5 \approx 2.236$: one direction pushed three times harder. Its orthogonal factor (confirmed with `numpy.linalg.svd`) is

$$
UV^\top = \frac{1}{\sqrt 5}\begin{pmatrix} 2 & -1 \\ 1 & 2 \end{pmatrix} \approx \begin{pmatrix} 0.894 & -0.447 \\ 0.447 & 0.894 \end{pmatrix},
$$

a pure rotation by about 26.6 degrees. The imbalance is gone; the rotation is $M$'s direction.

An SVD per step is too slow at scale, so Eq. 2 starts from $X_0 = M_t / \lVert M_t \rVert_F$ (every singular value now at most 1) and iterates

$$
X_k = a X_{k-1} + b (X_{k-1} X_{k-1}^\top) X_{k-1} + c (X_{k-1} X_{k-1}^\top)^2 X_{k-1}
$$

with $a = 3.4445$, $b = -4.7750$, $c = 2.0315$. Each step keeps $U$ and $V$ and applies $s \mapsto a s + b s^3 + c s^5$ to every singular value.

<figure class="blog-anim">
<svg viewBox="0 0 640 360" role="img" aria-label="Newton-Schulz iterations reshape the ellipse image of the unit circle under the momentum matrix toward the unit circle, while bars show both singular values moving toward 1" style="width:100%;height:auto;max-width:760px">
<style>
.a1-grid{stroke:var(--border,#d1d5db);stroke-width:1;stroke-dasharray:4 4;fill:none}
.a1-target{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:2;stroke-dasharray:6 5}
.a1-ell{fill:var(--accent,#6366f1);fill-opacity:.18;stroke:var(--accent,#6366f1);stroke-width:3;transform-box:fill-box;transform-origin:center;animation:a1-e 12s ease-in-out infinite}
.a1-bar{fill:var(--accent,#6366f1);transform-box:fill-box;transform-origin:bottom}
.a1-b1{animation:a1-s1 12s ease-in-out infinite}
.a1-b2{animation:a1-s2 12s ease-in-out infinite}
.a1-one{stroke:var(--text-primary,#1f2937);stroke-width:2;stroke-dasharray:6 5}
.a1-t{font:500 15px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.a1-h{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.a1-k{font:600 18px ui-monospace,monospace;fill:var(--accent,#6366f1);opacity:0;animation:a1-kf 12s linear infinite}
.a1-k0{animation-delay:-0s}.a1-k1{animation-delay:-10s}.a1-k2{animation-delay:-8s}.a1-k3{animation-delay:-6s}.a1-k4{animation-delay:-4s}.a1-k5{animation-delay:-2s}
@keyframes a1-e{0.00%{transform:scale(0.9487,0.3162)}16.58%{transform:scale(0.9487,0.3162)}22.50%{transform:scale(0.9447,0.7518)}33.25%{transform:scale(0.9447,0.7518)}39.17%{transform:scale(1.0484,0.7568)}49.92%{transform:scale(1.0484,0.7568)}55.83%{transform:scale(1.0414,0.6819)}66.58%{transform:scale(1.0414,0.6819)}72.50%{transform:scale(1.1343,0.6825)}83.25%{transform:scale(1.1343,0.6825)}89.17%{transform:scale(1.1337,0.753)}99.92%{transform:scale(1.1337,0.753)}100%{transform:scale(0.9487,0.3162)}}
@keyframes a1-s1{0.00%{transform:scaleY(0.9487)}16.58%{transform:scaleY(0.9487)}22.50%{transform:scaleY(0.9447)}33.25%{transform:scaleY(0.9447)}39.17%{transform:scaleY(1.0484)}49.92%{transform:scaleY(1.0484)}55.83%{transform:scaleY(1.0414)}66.58%{transform:scaleY(1.0414)}72.50%{transform:scaleY(1.1343)}83.25%{transform:scaleY(1.1343)}89.17%{transform:scaleY(1.1337)}99.92%{transform:scaleY(1.1337)}100%{transform:scaleY(0.9487)}}
@keyframes a1-s2{0.00%{transform:scaleY(0.3162)}16.58%{transform:scaleY(0.3162)}22.50%{transform:scaleY(0.7518)}33.25%{transform:scaleY(0.7518)}39.17%{transform:scaleY(0.7568)}49.92%{transform:scaleY(0.7568)}55.83%{transform:scaleY(0.6819)}66.58%{transform:scaleY(0.6819)}72.50%{transform:scaleY(0.6825)}83.25%{transform:scaleY(0.6825)}89.17%{transform:scaleY(0.753)}99.92%{transform:scaleY(0.753)}100%{transform:scaleY(0.3162)}}
@keyframes a1-kf{0%,15.5%{opacity:1}16.67%,100%{opacity:0}}
@media (prefers-reduced-motion:reduce){.a1-ell{animation:none;transform:scale(1.1337,0.753)}.a1-b1{animation:none;transform:scaleY(1.1337)}.a1-b2{animation:none;transform:scaleY(0.753)}.a1-k{animation:none;opacity:0}.a1-k5{opacity:1}}
</style>
<line class="a1-grid" x1="20" y1="200" x2="360" y2="200"/>
<line class="a1-grid" x1="190" y1="30" x2="190" y2="370"/>
<circle class="a1-target" cx="190" cy="200" r="130"/>
<g transform="rotate(-71.57 190 200)">
<circle class="a1-ell" cx="190" cy="200" r="130"/>
</g>
<text class="a1-t" x="20" y="30">image of the unit circle under Xₖ</text>
<text class="a1-t" x="20" y="350">dashed circle = exact UVᵀ (all σ = 1)</text>
<text class="a1-h" x="410" y="100">singular values of Xₖ</text>
<line class="a1-one" x1="410" y1="170" x2="610" y2="170"/>
<text class="a1-t" x="560" y="162">σ = 1</text>
<rect class="a1-bar a1-b1" x="440" y="170" width="50" height="130" rx="4"/>
<rect class="a1-bar a1-b2" x="530" y="170" width="50" height="130" rx="4"/>
<line class="a1-grid" x1="410" y1="300" x2="610" y2="300" style="stroke-dasharray:none"/>
<text class="a1-t" x="455" y="325">σ₁</text>
<text class="a1-t" x="545" y="325">σ₂</text>
<text class="a1-k a1-k0" x="470" y="58">step k = 0</text>
<text class="a1-k a1-k1" x="470" y="58">step k = 1</text>
<text class="a1-k a1-k2" x="470" y="58">step k = 2</text>
<text class="a1-k a1-k3" x="470" y="58">step k = 3</text>
<text class="a1-k a1-k4" x="470" y="58">step k = 4</text>
<text class="a1-k a1-k5" x="470" y="58">step k = 5</text>
</svg>
<figcaption>Five Newton–Schulz steps on M = [3 0; 4 5]. The ellipse starts thin (σ₁ = 0.95, σ₂ = 0.32 after dividing by ‖M‖<sub>F</sub>) and swells toward the dashed unit circle; after step 5 the singular values sit at 1.13 and 0.75, near 1 but not equal, which is the trade Muon's tuned coefficients make for speed.</figcaption>
</figure>

After five steps the singular values sit at 1.13 and 0.75: near 1, not equal. The coefficients inflate small singular values fast rather than converge exactly, and a rough $U V^\top$ is a fine update direction. The paper then adds weight decay (Eq. 3) and rescales by $0.2 \sqrt{\max(A, B)}$ (Eq. 4) to match AdamW's update size.

![One Muon update from gradient through momentum, Frobenius normalisation, five Newton–Schulz steps and the rescaled weight-decayed step](/imgs/blogs/orthogonality-projections-and-qr-7.webp)

The strip writes Eq. 4 as $W_t = W_{t-1} - \eta_t \Delta_t$ with $\Delta_t = 0.2 \sqrt{\max(A,B)}\, O_t + \lambda W_{t-1}$. Its bars show singular values 4, 1.5 and 0.5 becoming 0.93, 0.35, 0.12 after normalising and 0.73, 1.08, 0.74 after five steps: a spread of 8 to 1 shrinks to about 1.5 to 1.

## 10. Check it numerically

Gram–Schmidt against `numpy.linalg.qr` on a random $50 \times 4$ matrix, then `lstsq` against the projection $Pb$, plus idempotence, symmetry, residual orthogonality and the QR solve.

```python
import numpy as np
rng = np.random.default_rng(0)

def gram_schmidt(A):
    Q = np.zeros_like(A); R = np.zeros((A.shape[1], A.shape[1]))
    for j in range(A.shape[1]):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]      # coordinate along q_i
            v -= R[i, j] * Q[:, i]           # remove that shadow
        R[j, j] = np.linalg.norm(v); Q[:, j] = v / R[j, j]
    return Q, R

A = rng.standard_normal((50, 4)); b = rng.standard_normal(50)
Q, R = gram_schmidt(A)
Qn, Rn = np.linalg.qr(A)
S = np.sign(np.diag(Rn))                     # numpy may flip signs
print("max|Q - Q_np| =", np.abs(Q - Qn * S).max())
print("max|Q^T Q - I| =", np.abs(Q.T @ Q - np.eye(4)).max())

P = A @ np.linalg.solve(A.T @ A, A.T)       # projection matrix
x_ls = np.linalg.lstsq(A, b, rcond=None)[0]
print("max|Pb - A x_ls| =", np.abs(P @ b - A @ x_ls).max())
print("max|P@P - P| =", np.abs(P @ P - P).max(), " max|P - P^T| =", np.abs(P - P.T).max())
print("max|A^T (b - Pb)| =", np.abs(A.T @ (b - P @ b)).max())
print("QR solve vs lstsq:", np.abs(np.linalg.solve(R, Q.T @ b) - x_ls).max())
```

```
max|Q - Q_np| = 1.5959455978986625e-16
max|Q^T Q - I| = 2.220446049250313e-16
max|Pb - A x_ls| = 4.440892098500626e-16
max|P@P - P| = 5.551115123125783e-17  max|P - P^T| = 4.163336342344337e-17
max|A^T (b - Pb)| = 2.485305685599144e-15
QR solve vs lstsq: 2.498001805406602e-16
```

Every residual sits at double-precision round-off (machine epsilon is about $2.2 \times 10^{-16}$), as exact identities should. The sign line matters: QR is unique only up to the sign of each column of $Q$, and LAPACK may return $-q_j$ with a negative $R_{jj}$, so we flip signs before comparing. The orthogonality residual is ten times larger only because it sums 50 products of order-one numbers.

## 11. Where this shows up in practice
**Muon and Moonlight (2025).** The paper decoded above trains Moonlight, a 3B/16B-parameter mixture-of-experts model, on 5.7T tokens with Muon, and reports about twice the computational efficiency of AdamW under compute-optimal training ([arXiv:2502.16982](https://arxiv.org/abs/2502.16982)). It uses section 3 and the decode: each matrix update becomes its nearest semi-orthogonal matrix, so no singular direction of the momentum dominates.

**Orthogonal initialisation in PyTorch.** `torch.nn.init.orthogonal_` cites Saxe, McClelland and Ganguli, *Exact solutions to the nonlinear dynamics of learning in deep linear neural networks* ([arXiv:1312.6120](https://arxiv.org/abs/1312.6120), 2013), which reports depth-independent learning times from random orthogonal initial conditions. The code draws a Gaussian matrix, calls `torch.linalg.qr`, and multiplies $Q$ by the signs of the diagonal of $R$, the same sign fix as our check. An orthogonal layer preserves norms, so signals neither explode nor vanish through depth at initialisation.

**Randomized SVD.** `torch.svd_lowrank` follows Algorithm 4.4 of Halko, Martinsson and Tropp, *Finding structure with randomness* ([arXiv:0909.4061](https://arxiv.org/abs/0909.4061), 2009): multiply by a thin random matrix, orthonormalise with QR, and work with $Q Q^\top A$. That is section 4's $P = QQ^\top$ applied to every column of $A$.

## What to remember

> - Orthogonal means a zero dot product, and its payoff is Pythagoras: the cross term vanishes.
> - With an orthonormal basis, coordinates are dot products; an orthogonal matrix ($Q^\top Q = I$) preserves lengths and angles.
> - Projection onto the column space is $P = A(A^\top A)^{-1}A^\top$: symmetric, idempotent, and $QQ^\top$ for orthonormal columns.
> - Least squares is projection: the residual is orthogonal to every column, and QR solves it as $R\hat x = Q^\top b$.
> - Muon swaps the momentum matrix for its nearest semi-orthogonal matrix $UV^\top$, via a few Newton–Schulz steps.

## Where next

The series starts with [how to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper), and the step-by-step proof style is taught in [proof techniques for ML researchers](/blog/machine-learning/mathematics/proof-techniques-for-ml-researchers). In this track, [rank, null space and low-rank structure](/blog/machine-learning/mathematics/rank-null-space-and-low-rank-structure) explains what (A1) excludes, and [determinants, volume and invertibility](/blog/machine-learning/mathematics/determinants-volume-and-invertibility) shows why an orthogonal matrix has determinant $\pm 1$. Next come eigenvalues and the SVD, and a later post on matrix functions, the polar decomposition and Newton–Schulz explains why the iteration works and why an orthogonalised update is steepest descent under the spectral norm.

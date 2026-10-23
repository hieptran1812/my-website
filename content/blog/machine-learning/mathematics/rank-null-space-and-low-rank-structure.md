---
title: "Rank, Null Space and Low-Rank Structure: The Math Under LoRA"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "Learn column space, null space, rank, rank-nullity and low-rank factorisation with pictures and hand-checkable examples, prove rank-nullity line by line, then read LoRA's update equation and count its parameters."
tags: ["mathematics", "math-for-ai", "linear-algebra", "rank", "null-space", "low-rank", "lora", "fine-tuning", "singular-values", "proofs"]
category: "machine-learning"
subcategory: "Mathematics"
author: "Hiep Tran"
featured: true
readTime: 17
---

> **Level:** Foundation · **Needs:** [How to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper), [Matrices as linear maps: what a layer does](/blog/machine-learning/mathematics/matrices-as-linear-maps-what-a-layer-does) · **Unlocks:** LoRA, QLoRA and adapter papers, low-rank attention (DeepSeek-V2 MLA), the SVD and Eckart–Young appendices

Section 4.1 of the LoRA paper (Hu et al., 2021) changes a frozen pretrained layer with one line: $h = W_0 x + \Delta W x = W_0 x + BAx$. The sentence before it says $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, "and the rank $r \ll \min(d, k)$". The paragraph before that rests the whole method on a hypothesis: the weight update during adaptation has a low "intrinsic rank".

If rank and low-rank decomposition are words you nod along to, this post is for you. Paper math is a small, closed vocabulary; learn each idea once, with a picture, a derivation, a proof and a numerical check, and the appendix stops being a wall. By the end you will read that equation symbol by symbol, prove the theorem behind it, and count LoRA's parameters for a real GPT-3 layer.

![A rank-1 matrix maps the plane onto the line y = x/2 and sends the red null-space line to the origin](/imgs/blogs/rank-null-space-and-low-rank-structure-1.webp)

The picture above holds the entire post. A matrix takes the input plane (left) to the output plane (right). This one lands everything on a single blue line and sends a whole red line of inputs to zero. The blue line is the **column space**, the red line is the **null space**, and the fact that two dimensions split as one surviving plus one destroyed is the **rank–nullity theorem**.

## What a matrix can reach: the column space

**What it is for:** the column space is every output a layer can possibly produce, whatever the input.

A slide projector is the physical picture. The film can hold any image, but the light only ever lands on the wall. The analogy breaks in one place: a matrix can map a big space into a much thinner one, like a whole plane onto a wire.

Take the matrix from the opening figure:

$$
A = \begin{pmatrix} 1 & 2 \\ 0.5 & 1 \end{pmatrix}, \qquad a_1 = \begin{pmatrix} 1 \\ 0.5 \end{pmatrix}, \quad a_2 = \begin{pmatrix} 2 \\ 1 \end{pmatrix}.
$$

Here $a_1$ and $a_2$ are its columns. In words: a matrix is fully described by where it sends the basis directions, and those destinations are its columns.

Multiplying by $x = (x_1, x_2)$ gives $Ax = x_1 a_1 + x_2 a_2$, a mix of the columns. Try $x = (1, 0)$: the output is $(1, 0.5)$. Try $x = (3, -1)$: the output is $3 a_1 - a_2 = (3 - 2, 1.5 - 1) = (1, 0.5)$ again.

Every output sits on the line $y = x/2$. The second column is twice the first, $a_2 = 2 a_1$, so every mix $x_1 a_1 + x_2 a_2 = (x_1 + 2x_2)\, a_1$ is a multiple of $a_1$. No input reaches a point off that line.

Formally, for $A \in \mathbb{R}^{m \times n}$ the **column space** is

$$
\operatorname{col}(A) = \{\, Ax : x \in \mathbb{R}^n \,\} = \operatorname{span}(a_1, \dots, a_n) \subseteq \mathbb{R}^m .
$$

Here $m$ is the output dimension, $n$ the input dimension, and $\operatorname{span}$ means all linear combinations. In words: reachable outputs are exactly the mixes of the columns, a flat subspace through the origin. A 4096×4096 projection in a transformer has the same structure; if some of its columns are mixes of others, the region it can write into is thinner than the space around it.

## What a matrix destroys: the null space

**What it is for:** the null space is the set of input changes a layer cannot see at all.

Slide your hand along a lamp's light ray and its shadow on the wall does not move. That direction is invisible to the shadow, and a matrix can have invisible directions too.

For our $A$, test $x = (2, -1)$:

$$
A \begin{pmatrix} 2 \\ -1 \end{pmatrix} = \begin{pmatrix} 1 \cdot 2 + 2 \cdot (-1) \\ 0.5 \cdot 2 + 1 \cdot (-1) \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}.
$$

In words: a nonzero input produces zero output, and so does every multiple of it; together they form the red line in the opening figure. The animation runs the same map on sixteen points of the unit circle: each slides straight to its image, all land on the blue line, and the two red points on the null-space line collapse to the origin.

<figure class="blog-anim">
<svg viewBox="0 0 720 380" role="img" aria-label="Sixteen points on the unit circle slide in straight lines to where the rank-1 matrix A sends them: all of them land on the line y = x/2, and the two red points on the null-space line slide to the origin" style="width:100%;height:auto;max-width:820px">
<style>
.r1-ax{stroke:var(--border,#d1d5db);stroke-width:1.5}
.r1-circ{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:5 5}
.r1-col{stroke:var(--accent,#6366f1);stroke-width:2.5;opacity:.55}
.r1-nul{stroke:#e03131;stroke-width:2;stroke-dasharray:7 5;opacity:.7}
.r1-d{fill:var(--accent,#6366f1)}
.r1-z{fill:#e03131}
.r1-t{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.r1-s{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.r1-c{font:600 14px ui-sans-serif,system-ui;fill:var(--accent,#6366f1)}
.r1-n{font:600 14px ui-sans-serif,system-ui;fill:#e03131}
@keyframes r1-m0{0%,15%{transform:translate(0px,0px)}85%,100%{transform:translate(-53.67px,-26.83px)}}
.r1-a0{animation:r1-m0 12s ease-in-out infinite alternate}
@keyframes r1-m1{0%,15%{transform:translate(0px,0px)}85%,100%{transform:translate(-8.51px,-29.92px)}}
.r1-a1{animation:r1-m1 12s ease-in-out infinite alternate}
@keyframes r1-m2{0%,15%{transform:translate(0px,0px)}85%,100%{transform:translate(37.95px,-28.46px)}}
.r1-a2{animation:r1-m2 12s ease-in-out infinite alternate}
@keyframes r1-m3{0%,15%{transform:translate(0px,0px)}85%,100%{transform:translate(78.62px,-22.66px)}}
.r1-a3{animation:r1-m3 12s ease-in-out infinite alternate}
@keyframes r1-m4{0%,15%{transform:translate(0px,0px)}85%,100%{transform:translate(107.33px,-13.42px)}}
.r1-a4{animation:r1-m4 12s ease-in-out infinite alternate}
@keyframes r1-m5{0%,15%{transform:translate(0px,0px)}85%,100%{transform:translate(119.70px,-2.13px)}}
.r1-a5{animation:r1-m5 12s ease-in-out infinite alternate}
@keyframes r1-m6{0%,15%{transform:translate(0px,0px)}85%,100%{transform:translate(113.84px,9.49px)}}
.r1-a6{animation:r1-m6 12s ease-in-out infinite alternate}
@keyframes r1-m7{0%,15%{transform:translate(0px,0px)}85%,100%{transform:translate(90.65px,19.66px)}}
.r1-a7{animation:r1-m7 12s ease-in-out infinite alternate}
@keyframes r1-m8{0%,15%{transform:translate(0px,0px)}85%,100%{transform:translate(53.67px,26.83px)}}
.r1-a8{animation:r1-m8 12s ease-in-out infinite alternate}
@keyframes r1-m9{0%,15%{transform:translate(0px,0px)}85%,100%{transform:translate(8.51px,29.92px)}}
.r1-a9{animation:r1-m9 12s ease-in-out infinite alternate}
@keyframes r1-m10{0%,15%{transform:translate(0px,0px)}85%,100%{transform:translate(-37.95px,28.46px)}}
.r1-a10{animation:r1-m10 12s ease-in-out infinite alternate}
@keyframes r1-m11{0%,15%{transform:translate(0px,0px)}85%,100%{transform:translate(-78.62px,22.66px)}}
.r1-a11{animation:r1-m11 12s ease-in-out infinite alternate}
@keyframes r1-m12{0%,15%{transform:translate(0px,0px)}85%,100%{transform:translate(-107.33px,13.42px)}}
.r1-a12{animation:r1-m12 12s ease-in-out infinite alternate}
@keyframes r1-m13{0%,15%{transform:translate(0px,0px)}85%,100%{transform:translate(-119.70px,2.13px)}}
.r1-a13{animation:r1-m13 12s ease-in-out infinite alternate}
@keyframes r1-m14{0%,15%{transform:translate(0px,0px)}85%,100%{transform:translate(-113.84px,-9.49px)}}
.r1-a14{animation:r1-m14 12s ease-in-out infinite alternate}
@keyframes r1-m15{0%,15%{transform:translate(0px,0px)}85%,100%{transform:translate(-90.65px,-19.66px)}}
.r1-a15{animation:r1-m15 12s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){.r1-a0,.r1-a1,.r1-a2,.r1-a3,.r1-a4,.r1-a5,.r1-a6,.r1-a7,.r1-a8,.r1-a9,.r1-a10,.r1-a11,.r1-a12,.r1-a13,.r1-a14,.r1-a15{animation:none}}
</style>
<line class="r1-ax" x1="24.0" y1="190.0" x2="456.0" y2="190.0"/>
<line class="r1-ax" x1="240.0" y1="358.0" x2="240.0" y2="22.0"/>
<circle class="r1-circ" cx="240" cy="190" r="60"/>
<line class="r1-col" x1="24.0" y1="298.0" x2="456.0" y2="82.0"/>
<line class="r1-nul" x1="48.0" y1="94.0" x2="432.0" y2="286.0"/>
<circle class="r1-z r1-a0" cx="293.7" cy="216.8" r="5"/>
<circle class="r1-d r1-a1" cx="299.8" cy="194.3" r="5"/>
<circle class="r1-d r1-a2" cx="296.9" cy="171.0" r="5"/>
<circle class="r1-d r1-a3" cx="285.3" cy="150.7" r="5"/>
<circle class="r1-d r1-a4" cx="266.8" cy="136.3" r="5"/>
<circle class="r1-d r1-a5" cx="244.3" cy="130.2" r="5"/>
<circle class="r1-d r1-a6" cx="221.0" cy="133.1" r="5"/>
<circle class="r1-d r1-a7" cx="200.7" cy="144.7" r="5"/>
<circle class="r1-z r1-a8" cx="186.3" cy="163.2" r="5"/>
<circle class="r1-d r1-a9" cx="180.2" cy="185.7" r="5"/>
<circle class="r1-d r1-a10" cx="183.1" cy="209.0" r="5"/>
<circle class="r1-d r1-a11" cx="194.7" cy="229.3" r="5"/>
<circle class="r1-d r1-a12" cx="213.2" cy="243.7" r="5"/>
<circle class="r1-d r1-a13" cx="235.7" cy="249.8" r="5"/>
<circle class="r1-d r1-a14" cx="259.0" cy="246.9" r="5"/>
<circle class="r1-d r1-a15" cx="279.3" cy="235.3" r="5"/>
<text class="r1-c" x="378" y="97">column space</text>
<text class="r1-n" x="24" y="67">null space</text>
<text class="r1-s" x="247" y="205">0</text>
<text class="r1-t" x="490" y="110">A = [1 2; 0.5 1], rank 1</text>
<text class="r1-s" x="490" y="150">each dot x slides straight</text>
<text class="r1-s" x="490" y="176">to Ax, then back</text>
<text class="r1-c" x="490" y="212">every Ax lands on y = x/2</text>
<text class="r1-n" x="490" y="248">red dots: ±(2, −1)/√5,</text>
<text class="r1-n" x="490" y="274">A sends them to 0</text>
<text class="r1-s" x="490" y="310">circle becomes a segment</text>
<text class="r1-s" x="490" y="336">of half-length √5 ≈ 2.24</text>
</svg>
<figcaption>The rank-1 map in motion: each point of the unit circle moves in a straight line to its image under A. The whole circle flattens onto the column space y = x/2, and the two red points on the null-space line collapse to the origin.</figcaption>
</figure>

Moving along the null space changes nothing downstream: if $An = 0$ then $A(x + n) = Ax$. Two different inputs give the same output, so the output cannot tell them apart. That is the precise sense in which a rank-deficient layer loses information.

Formally, the **null space** of $A \in \mathbb{R}^{m \times n}$ is

$$
\operatorname{null}(A) = \{\, x \in \mathbb{R}^n : Ax = 0 \,\} \subseteq \mathbb{R}^n .
$$

In words: every input sent to zero. In the language of [sets, functions and maps](/blog/machine-learning/mathematics/sets-functions-and-maps-the-grammar-of-ml-math) it is the preimage of ${0}$, often called the **kernel**, $\ker A$. Note where each set lives: the column space in the output space $\mathbb{R}^m$, the null space in the input space $\mathbb{R}^n$. Mixing them up is the most common slip when reading a proof.

## Rank: how many independent directions survive

**What it is for:** rank is one number that says how many genuinely different directions a matrix can output.

Picture a team sending status reports. If one person's report is always the sum of two others, the manager learns nothing new from it; rank counts the reports that carry new information.

Formally, the **rank** of $A$ is $\operatorname{rank}(A) = \dim \operatorname{col}(A)$, the size of the largest set of linearly independent columns. A standard fact, stated here without proof (Axler, *Linear Algebra Done Right*, 3rd ed., §3.F), is that the largest number of independent rows is the same, so $\operatorname{rank}(A) \leq \min(m, n)$.

![A 3×3 matrix whose third column is the sum of the first two, its null vector (1, 1, −1), and a bar splitting 3 columns into rank 2 plus nullity 1](/imgs/blogs/rank-null-space-and-low-rank-structure-2.webp)

The figure works through a 3×3 example:

$$
A = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{pmatrix}.
$$

In words: three columns, $a_1 = (1, 0, 1)$, $a_2 = (0, 1, 1)$ and $a_3 = (1, 1, 2)$, but the third is not new.

Check by hand. The first two are independent: $c_1 a_1 + c_2 a_2 = (c_1, c_2, c_1 + c_2)$ is zero only when $c_1 = c_2 = 0$. The third is their sum, $(1 + 0, 0 + 1, 1 + 1) = (1, 1, 2)$. So the rank is 2.

That redundancy hands us a null vector. Rearranging $a_1 + a_2 - a_3 = 0$ says $Ax = 0$ for $x = (1, 1, -1)$; row by row, ${1 + 0 - 1 = 0}$, ${0 + 1 - 1 = 0}$, ${1 + 1 - 2 = 0}$. Every dependency among the columns is a null vector, and every null vector is a dependency. A square matrix loses rank exactly when $\det A = 0$, the link the [determinants post](/blog/machine-learning/mathematics/determinants-volume-and-invertibility) develops.

## Rank–nullity: the dimension budget

**What it is for:** bookkeeping. Every input direction is either kept or destroyed.

The bar in the second figure is the accounting: 3 input dimensions, 2 survive, 1 dies, and ${2 + 1 = 3}$. The opening 2×2 matrix keeps 1 and destroys 1. The **rank–nullity theorem** says the books always balance. For $A \in \mathbb{R}^{m \times n}$,

$$
\operatorname{rank}(A) + \dim \operatorname{null}(A) = n .
$$

Here $\dim \operatorname{null}(A)$ is the **nullity**. In words: surviving directions plus destroyed directions equals the number of columns. We prove it below.

At transformer size: a 4096×4096 matrix of rank 8 keeps 8 directions and has a null space of dimension ${4096 - 8 = 4088}$. Almost every input direction is invisible to it. That is the kind of matrix LoRA adds to a frozen layer.

## Rank-1 matrices are outer products

**What it is for:** an outer product is the smallest building block of a matrix. It reads one number from the input and writes it along one direction.

A multiplication table is the picture: labels down the side, labels across the top, each cell the product. Every row is a scaled copy of the top row, so twelve cells carry only seven numbers.

![Heatmap of the outer product of u = (1, 2, 3) and v = (2, −1, 0, 1): every row is a multiple of v](/imgs/blogs/rank-null-space-and-low-rank-structure-3.webp)

With $u = (1, 2, 3)$ and $v = (2, -1, 0, 1)$, the **outer product** has entries $u_i v_j$:

$$
u v^\top = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix} \begin{pmatrix} 2 & -1 & 0 & 1 \end{pmatrix} = \begin{pmatrix} 2 & -1 & 0 & 1 \\ 4 & -2 & 0 & 2 \\ 6 & -3 & 0 & 3 \end{pmatrix}.
$$

In words: row $i$ is $u_i$ times $v^\top$, and every column is a multiple of $u$, so the rank is 1. In the notation of the [einsum post](/blog/machine-learning/mathematics/sums-products-einsum-and-log-sum-exp) it is `einsum("i,j->ij", u, v)`.

Applied to an input, associativity gives

$$
(u v^\top)\, x = u\, (v^\top x) = (v \cdot x)\, u .
$$

In words: one dot product with $v$, then write that single number along $u$. For $x = (1, 1, 1, 1)$, $v \cdot x = 2 - 1 + 0 + 1 = 2$, so the output is ${2u = (2, 4, 6)}$, which matches the row sums of the table.

Now the link to LoRA. Any product $BA$ with $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ is a sum of $r$ outer products:

$$
BA = \sum_{i=1}^{r} b_i\, \alpha_i^\top ,
$$

where $b_i$ is column $i$ of $B$ and $\alpha_i^\top$ is row $i$ of $A$. In words: read $r$ numbers from the input, one per row of $A$, and write each along its own column of $B$. Both sides have $(j, l)$ entry $\sum_i B_{ji} A_{il}$.

## Low-rank factorisation: W = BA and its parameter count

**What it is for:** a thin product $BA$ stores and trains a matrix with far fewer numbers, at the price of a ceiling on its rank.

Two facts make this exact. First, a product never has more rank than its thinnest factor:

$$
\operatorname{rank}(BA) \leq \min\bigl(\operatorname{rank}(B), \operatorname{rank}(A)\bigr) \leq r .
$$

In words: the inner dimension $r$ is a bottleneck. We derive it from rank–nullity after the proof.

Second, every rank-$r$ matrix $W$ factors as $BA$ with inner dimension $r$. Put $r$ independent columns of $W$ into $B$; each column of $W$ is a mix of them, and its mixing weights form the matching column of $A$. For the 3×3 example, $B = (a_1\ a_2)$ and $A = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{pmatrix}$.

So "rank at most $r$" and "factors through $r$ numbers" are the same statement. A full $W \in \mathbb{R}^{d \times k}$ has $dk$ entries; the pair $B, A$ has $dr + rk = r(d + k)$.

![LoRA anatomy: a frozen 4096 × 4096 W₀ beside thin trainable factors A (8 × 4096) and B (4096 × 8), whose outputs add to give h](/imgs/blogs/rank-null-space-and-low-rank-structure-4.webp)

For a 4096×4096 projection and $r = 8$: the full matrix has $4096 \times 4096 = 16{,}777{,}216$ entries, the factors $8 \times (4096 + 4096) = 65{,}536$, a ratio of 0.39%. Compute shrinks too, if you multiply in the right order: $B(Ax)$ costs $r(d + k)$ multiply-adds, while forming $BA$ first costs $dk$. For a square $d \times d$ matrix the saving needs $2rd \lt d^2$, that is $r \lt d/2$; the [Big-O post](/blog/machine-learning/mathematics/limits-big-o-and-asymptotics-in-ml-papers) explains why papers quote $O(r(d + k))$ against $O(dk)$.

## Approximate rank: what the singular values say

**What it is for:** real matrices are never exactly low rank, so you need a way to say "rank 4, up to noise".

The tool is the **singular value decomposition**, which a later post builds properly. One fact suffices here, stated without proof (Strang, *Introduction to Linear Algebra*, SVD chapter): every $W \in \mathbb{R}^{d \times k}$ has singular values $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$, and the number of nonzero ones equals the rank. In words: each singular value measures how strongly the matrix stretches one independent direction.

![Singular values of a 64 × 48 rank-4 matrix on a log scale: four large values, then a cliff to 10⁻¹⁴, or a gap to 10⁻² with noise](/imgs/blogs/rank-null-space-and-low-rank-structure-5.webp)

The chart shows a 64×48 matrix built as $BA$ with $r = 4$. The first four singular values are about 73, 64, 45 and 39; the fifth falls to about ${10}^{-14}$, float64 round-off. Add noise of size ${10}^{-3}$ (amber) and the tail rises to near ${10}^{-2}$: technically full rank, but with a gap of a factor of about 2,763 after $\sigma_4$. That gap is what "approximately rank 4" means, and it is why `np.linalg.matrix_rank` counts singular values above a tolerance. The best rank-$r$ approximation keeps the top $r$ singular directions (Eckart and Young, 1936, stated without proof); the [norms post](/blog/machine-learning/mathematics/norms-inner-products-and-distances) covers the norms it is measured in.

## The evidence that fine-tuning updates are low rank

**What it is for:** "updates are low rank" is an empirical claim, so it needs evidence, not a theorem.

The first evidence predates LoRA. Aghajanyan, Zettlemoyer and Gupta, [*Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*](https://arxiv.org/abs/2012.13255) (2020), trained a small vector projected into the full weight space by a fixed random matrix. Their abstract reports that 200 such parameters tune RoBERTa to 90% of full fine-tuning performance on MRPC: the update lies in the column space of a 200-column matrix.

LoRA turns this into a hypothesis about each weight matrix, then tests it by sweeping $r$.

![Grouped bars of LoRA WikiSQL accuracy on GPT-3 175B for ranks 1 to 64, from the paper's Table 6; the y axis starts at 60](/imgs/blogs/rank-null-space-and-low-rank-structure-6.webp)

The chart reproduces the WikiSQL rows of the paper's Table 6. Adapting the query and value projections, rank 1 scores 73.4 and rank 64 scores 73.5, and the whole sweep stays within 0.5 points. Adapting the query projection alone spreads over 1.7 points and benefits more from rank. Section 7.2 adds that the top singular directions learned with $r = 8$ and $r = 64$ overlap strongly, and Table 7 that $\Delta W$ amplifies directions already present in $W$ but not emphasised by it.

The authors state the limits: footnote 6 says they do not expect small $r$ to work for every task, for example one in a different language from pretraining. Low rank is a measured property of particular models and tasks, not a law.

### Paper sentence, decoded

**Paper:** Edward J. Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685), 2021. Section 4.1, Eq. 3:

$$
h = W_0 x + \Delta W x = W_0 x + BAx
$$

| Symbol | What it is | Built in |
| --- | --- | --- |
| $x \in \mathbb{R}^k$ | input to the layer | column space |
| $W_0 \in \mathbb{R}^{d \times k}$ | pretrained weight, frozen | column space |
| $\Delta W$ | the change fine-tuning makes | evidence |
| $B \in \mathbb{R}^{d \times r}$ | tall factor; its columns span the update's column space; zero at init | factorisation |
| $A \in \mathbb{R}^{r \times k}$ | wide factor; its rows are the directions read from $x$; Gaussian at init | outer products |
| $r \ll \min(d, k)$ | the rank budget | rank, corollary |
| $h \in \mathbb{R}^d$ | layer output | figure 4 |

In words: keep the old output and add a correction that reads $r$ numbers from the input and writes each along a learned direction, so the update has rank at most $r$ and costs $r(d + k)$ parameters. Because $B$ starts at zero, training starts exactly at the pretrained model. The paper also scales $\Delta W x$ by $\alpha / r$, which leaves the rank untouched.

## Proof, read line by line

**Statement (rank–nullity).** Let $A \in \mathbb{R}^{m \times n}$. Then $\operatorname{rank}(A) + \dim \operatorname{null}(A) = n$.

- (A1) The input space $\mathbb{R}^n$ is finite-dimensional, with dimension $n$.
- (A2) The map $x \mapsto Ax$ is linear: $A(cx + y) = cAx + Ay$.

**The idea in one sentence:** take a basis of the null space, extend it to a basis of the whole input space, and show that the images of the added vectors form a basis of the column space.

1. $\operatorname{null}(A)$ is a subspace (closed under sums and scaling by (A2)), so it has a basis $v_1, \dots, v_p$ with $p = \dim \operatorname{null}(A) \leq n$, finite by (A1).
2. Extend it to a basis $v_1, \dots, v_p, w_1, \dots, w_q$ of $\mathbb{R}^n$, so $p + q = n$. Any independent list in a finite-dimensional space extends to a basis, by (A1).
3. Claim: $Aw_1, \dots, Aw_q$ is a basis of $\operatorname{col}(A)$. Then $\operatorname{rank}(A) = q = n - p$. Steps 4 to 11 prove the claim.
4. Spanning. Take any $y \in \operatorname{col}(A)$; by definition $y = Ax$ for some $x$.
5. Expand $x$ in the basis of step 2: $x = \sum_{i=1}^{p} a_i v_i + \sum_{j=1}^{q} b_j w_j$.
6. Apply $A$: $y = \sum_i a_i\, Av_i + \sum_j b_j\, Aw_j$, by linearity (A2).
7. Each $Av_i = 0$, since $v_i$ is in the null space, so $y = \sum_j b_j\, Aw_j$. The $Aw_j$ span.
8. Independence. Suppose $\sum_j c_j\, Aw_j = 0$.
9. By linearity (A2), $A\bigl(\sum_j c_j w_j\bigr) = 0$, so $\sum_j c_j w_j \in \operatorname{null}(A)$.
10. The $v_i$ span the null space, so $\sum_j c_j w_j = \sum_i d_i v_i$ for some scalars $d_i$.
11. Then $\sum_i d_i v_i - \sum_j c_j w_j = 0$. The list from step 2 is a basis, hence independent, so every $c_j$ is zero.
12. Spanning plus independent means basis, so $\operatorname{rank}(A) = q = n - p$. ∎

**Where each assumption was used:**

| Assumption | Steps |
| --- | --- |
| (A1) finite dimension | 1 (finite basis), 2 (basis extension, $p + q = n$) |
| (A2) linearity | 1 (null space is a subspace), 6, 9 |

**What breaks without (A1).** Take all polynomials, of every degree, and the map $p(t) \mapsto t\, p(t)$. Its nullity is 0, yet no output is a nonzero constant, so it is not onto. In finite dimensions rank–nullity forbids this (a square map with nullity 0 has full rank); without (A1) there is no $n$ to balance and that corollary fails.

**Proof moves used:** none from the analytic list; this is the basis-extension argument that later eigenvalue and SVD proofs reuse. Source: Axler, *Linear Algebra Done Right*, 3rd ed., Theorem 3.22 (Fundamental Theorem of Linear Maps), simplified from linear maps to matrices.

### Corollary: why LoRA's update has rank at most r

Let $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$.

1. Every output of $BA$ is $B(Ax)$, an output of $B$, so $\operatorname{col}(BA) \subseteq \operatorname{col}(B)$ and $\operatorname{rank}(BA) \leq \operatorname{rank}(B) \leq r$.
2. If $Ax = 0$ then $BAx = 0$, so $\operatorname{null}(A) \subseteq \operatorname{null}(BA)$.
3. Both have $k$ columns, so by rank–nullity $\operatorname{rank}(BA) = k - \dim \operatorname{null}(BA) \leq k - \dim \operatorname{null}(A) = \operatorname{rank}(A)$.

Whatever the optimizer does, LoRA's update never uses more than $r$ directions.

## Check it numerically

The script builds a rank-4 matrix from thin random factors, reads its singular values, checks rank plus nullity, adds noise, and counts LoRA parameters for two real layer shapes.

```python
import numpy as np

rng = np.random.default_rng(0)
d, k, r = 64, 48, 4

# A rank-r matrix: product of a d x r and an r x k factor (W = B A)
B = rng.standard_normal((d, r))
A = rng.standard_normal((r, k))
W = B @ A
s = np.linalg.svd(W, compute_uv=False)
print("first 7 singular values:", np.array2string(s[:7], precision=3))
print("rank by SVD threshold:  ", np.linalg.matrix_rank(W))
print("rank + nullity =", np.linalg.matrix_rank(W), "+", k - np.linalg.matrix_rank(W), "=", k)

# Add small noise: the cliff becomes a gap, not a zero
Wn = W + 1e-3 * rng.standard_normal((d, k))
sn = np.linalg.svd(Wn, compute_uv=False)
print("noisy, s[3] / s[4]:      %.0f" % (sn[3] / sn[4]))

# LoRA parameter count, h = W0 x + B A x
def lora(d, k, r):
    return r * (d + k)
full = 4096 * 4096
print("4096x4096, r=8: %d vs %d (%.2f%%)" % (lora(4096, 4096, 8), full,
      100 * lora(4096, 4096, 8) / full))
gpt3 = 96 * 2 * lora(12288, 12288, 4)   # Wq and Wv, 96 layers, r=4
print("GPT-3 Wq+Wv, r=4: %d params, %.1f MB in fp16" % (gpt3, gpt3 * 2 / 1e6))
```

```
first 7 singular values: [7.308e+01 6.352e+01 4.468e+01 3.945e+01 9.281e-15 7.984e-15 6.793e-15]
rank by SVD threshold:   4
rank + nullity = 4 + 44 = 48
noisy, s[3] / s[4]:      2763
4096x4096, r=8: 65536 vs 16777216 (0.39%)
GPT-3 Wq+Wv, r=4: 18874368 params, 37.7 MB in fp16
```

Exactly four singular values are nonzero; the fifth sits near ${10}^{-14}$, the float64 round-off level for a largest singular value of about 73. Rank plus nullity is 48, the number of columns.

The last line reproduces the paper. GPT-3 175B has 96 layers of width 12,288; adapting query and value with $r = 4$ costs $96 \times 2 \times 4 \times (12{,}288 + 12{,}288) = 18{,}874{,}368$ parameters. Section 7.1 quotes this budget as "18M (roughly 35MB if stored in FP16)"; 37.7 million bytes is 36.0 MiB, so they agree up to rounding of units.

## Where this shows up in practice

**LoRA in Hugging Face PEFT.** The [PEFT LoRA guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) exposes `r` as "the rank of the update matrices" and scales by `lora_alpha / r`. Its `merge_and_unload()` folds the adapter into the base weight, which works because $W_0 + BA$ is just another $d \times k$ matrix, so merged inference costs what the original layer cost. [Debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft) covers what goes wrong in practice.

**QLoRA.** Dettmers et al., [*QLoRA: Efficient Finetuning of Quantized LLMs*](https://arxiv.org/abs/2305.14314) (2023), stores the frozen $W_0$ in 4-bit NormalFloat and backpropagates only into the LoRA factors, enough to fine-tune a 65B model on one 48GB GPU. The split in Eq. 3 makes it possible: the frozen path needs no gradients or optimizer state, while the $r(d + k)$ trainable numbers stay in high precision.

**Low-rank attention in DeepSeek-V2.** [*DeepSeek-V2*](https://arxiv.org/abs/2405.04434) (DeepSeek-AI, 2024) has a Section 2.1.2 titled "Low-Rank Key-Value Joint Compression". Each hidden state is projected down to a latent $c_t^{KV} = W^{DKV} h_t$ of dimension $d_c = 512$, and keys and values are projected back up from it, so the key and value maps factor through a 512-dimensional bottleneck like $BA$. Caching only the latent gives the abstract's 93.3% KV-cache reduction.

## What to remember

> - The **column space** is every reachable output; the **null space** is every input sent to zero. One lives in the output space, the other in the input space.
> - **Rank** counts independent columns (or rows, or nonzero singular values); rank plus nullity equals the number of columns.
> - $BA$ is a sum of $r$ outer products, so $\operatorname{rank}(BA) \leq r$, and it costs $r(d + k)$ parameters instead of $dk$.
> - "Fine-tuning updates are low rank" is an empirical finding, strong for many tasks and explicitly not promised for all.

## Where next

If the notation felt fast, start with [how to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper), and reread the proof with [proof techniques for ML researchers](/blog/machine-learning/mathematics/proof-techniques-for-ml-researchers) beside it. In this wave, [matrices as linear maps](/blog/machine-learning/mathematics/matrices-as-linear-maps-what-a-layer-does) shows why columns are images of basis vectors, and [orthogonality, projections and QR](/blog/machine-learning/mathematics/orthogonality-projections-and-qr) builds an orthonormal basis for a column space. Later posts on eigenvalues and on the singular value decomposition turn "count the nonzero singular values" into a theorem you can prove.

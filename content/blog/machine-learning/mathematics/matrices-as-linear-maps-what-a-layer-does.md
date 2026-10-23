---
title: "Matrices as Linear Maps: What a Layer Actually Does"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "See a weight matrix as a map that moves basis vectors, read matrix products as composition, and prove why two linear layers without an activation collapse into one, ending with the Transformer feed-forward layer decoded symbol by symbol."
tags: ["mathematics", "math-for-ai", "linear-algebra", "matrices", "linear-maps", "transformer", "feed-forward-network", "proofs"]
category: "machine-learning"
subcategory: "Mathematics"
author: "Hiep Tran"
featured: true
readTime: 18
---

> **Level:** Foundation · **Needs:** [How to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper), [Sets, functions and maps](/blog/machine-learning/mathematics/sets-functions-and-maps-the-grammar-of-ml-math), [Sums, products and einsum](/blog/machine-learning/mathematics/sums-products-einsum-and-log-sum-exp) · **Unlocks:** Transformer FFN and attention projections, LoRA, tensor parallelism, every "$W x + b$" in every appendix

Section 3.3 of "Attention Is All You Need" (Vaswani et al., 2017) defines the layer that holds about two thirds of each Transformer layer's parameters in a single line: $\mathrm{FFN}(x)=\max(0,\,xW_1+b_1)\,W_2+b_2$. It looks innocent. It hides four ideas a reader has to own before the line means anything: what a matrix does to a vector, why the order of $W_1$ and $W_2$ matters, what the bias $b_1$ is for, and why that small $\max(0,\cdot)$ is the reason the layer has two matrices instead of one.

By the end of this post you will read that line as a sentence, know the shape of every symbol in it, and have proved, line by line, that deleting the $\max$ would make $W_1$ and $W_2$ worth exactly one matrix. The series bet applies here as everywhere: paper math is a small, closed vocabulary, and each word, learned once with a picture, a derivation, a proof and a numerical check, stops being a wall.

![A 2x2 matrix sends e1 and e2 to its two columns and the unit square to a parallelogram of area 3](/imgs/blogs/matrices-as-linear-maps-what-a-layer-does-1.webp)

The figure above is the mental model for the whole post. A matrix is not a table of numbers that happens to multiply. It is a map of the plane, and its columns record exactly where the two basis arrows land. Everything else follows from that one picture.

## Linearity: the one promise a weight matrix makes

**What it is for.** Linearity is the promise that a map treats sums and scalings fairly: transform two inputs and add the results, or add first and transform once, and you get the same answer.

The physical picture is a rubber sheet with a grid drawn on it, pinned at one point. You may stretch it, shear it, rotate it or flip it, but three rules hold: grid lines stay straight, parallel lines stay parallel, and equally spaced lines stay equally spaced. The pin, the origin, never moves. The analogy breaks in one place: a real sheet can be stretched more in one corner than another, and a linear map cannot. It does the same thing everywhere.

A worked example with one number. The map $f(x)=3x$ is linear: $f(2+5)=21$ and $f(2)+f(5)=6+15=21$, and $f(4\cdot 2)=24=4\,f(2)$. The map $g(x)=3x+1$ is not: $g(2+5)=22$ but $g(2)+g(5)=7+16=23$. The extra ${1}$ was counted twice. Hold on to that failure; it is the bias, and it gets its own section.

Formally, a map $f:\mathbb{R}^{n}\to\mathbb{R}^{m}$ is **linear** if for all vectors $u,v\in\mathbb{R}^{n}$ and all scalars $a,b\in\mathbb{R}$,

$$
f(a\,u+b\,v)=a\,f(u)+b\,f(v).
$$

In words: the image of a weighted sum is the same weighted sum of the images. Setting $a=b=0$ gives $f(0)=0$, which is the pinned origin from the sheet picture. The arrow notation $f:\mathbb{R}^{n}\to\mathbb{R}^{m}$ is the domain-and-codomain grammar from the [sets and functions post](/blog/machine-learning/mathematics/sets-functions-and-maps-the-grammar-of-ml-math): inputs have $n$ coordinates, outputs have $m$.

## A matrix is a list of where the basis vectors go

**What it is for.** A matrix is a compact way to store a linear map: you only need to write down what happens to a handful of special inputs, and linearity fills in everything else.

The special inputs are the **standard basis vectors** $e_1=(1,0)$ and $e_2=(0,1)$ in the plane, and $e_1,\dots,e_n$ in $\mathbb{R}^{n}$, where $e_j$ has a ${1}$ in slot $j$ and zeros elsewhere. Every vector is a recipe in these ingredients: $x=(x_1,x_2)=x_1e_1+x_2e_2$. Linearity then says

$$
f(x)=f(x_1e_1+x_2e_2)=x_1\,f(e_1)+x_2\,f(e_2).
$$

In words: once you know where $e_1$ and $e_2$ land, you know where everything lands. So we write those two landing points side by side as columns, and call the result the matrix of $f$.

Take the matrix from the opening figure,

$$
A=\begin{pmatrix}2 & -1\\ 1 & 1\end{pmatrix}.
$$

Its first column says $Ae_1=(2,1)$ and its second says $Ae_2=(-1,1)$. Where does $x=(1,1)$ go? Use the recipe: $Ax=1\cdot(2,1)+1\cdot(-1,1)=(1,2)$. That is the top corner of the blue parallelogram in the figure, because $(1,1)$ is the top corner of the unit square. The unit square, area ${1}$, became a parallelogram of area $2\cdot 1-(-1)\cdot 1=3$; why that number is the determinant is the subject of the sibling post on [determinants, volume and invertibility](/blog/machine-learning/mathematics/determinants-volume-and-invertibility).

The general statement: for a linear $f:\mathbb{R}^{n}\to\mathbb{R}^{m}$, its **matrix** $A\in\mathbb{R}^{m\times n}$ has column $j$ equal to $f(e_j)$, and $f(x)=Ax=\sum_{j=1}^{n}x_j\,A_{:,j}$, where $A_{:,j}$ denotes column $j$. Read the shape as "$m$ rows because outputs have $m$ coordinates, $n$ columns because there are $n$ basis vectors to track". A layer mapping a 4096-dimensional embedding to 4096 dimensions is a list of 4096 columns, and column $j$ is the output vector that feature $j$ alone would produce. That reading is how interpretability papers talk about "the direction a neuron writes to".

### Two ways to read the product $Ax$

There are two equally correct ways to compute $Ax$, and papers switch between them without warning.

![Row view computes each output as a dot product; column view mixes the columns; both give (5, 4)](/imgs/blogs/matrices-as-linear-maps-what-a-layer-does-2.webp)

Take $x=(3,1)$ with the same $A$. The **row view** computes each output coordinate as a dot product of one row with $x$: row 1 gives $2\cdot 3+(-1)\cdot 1=5$ and row 2 gives $1\cdot 3+1\cdot 1=4$. The **column view** mixes the columns with weights taken from $x$: $3\cdot(2,1)+1\cdot(-1,1)=(6,3)+(-1,1)=(5,4)$. Same answer, $(5,4)$, as the figure shows in green.

The row view is the one to use when you ask "how similar is $x$ to each row?", which is how attention scores and classifier logits are read; the [vectors and dot products post](/blog/machine-learning/mathematics/vectors-dot-products-and-the-geometry-of-embeddings) builds that geometry. The column view is the one to use when you ask "which directions can the output ever point in?", and its answer, the span of the columns, is where [rank and null space](/blog/machine-learning/mathematics/rank-null-space-and-low-rank-structure) begin. In index form both are the same sum, $(Ax)_i=\sum_{j}A_{ij}x_j$; the row view loops over $i$ outside, the column view loops over $j$ outside.

## Matrix times matrix means "do one map, then the other"

**What it is for.** Multiplying two matrices produces the single matrix of the map you get by applying one and then the other, so a pipeline of linear steps can be stored as one table.

Picture two stations on an assembly line. Station $B$ bends a part, station $C$ then bends whatever arrives. The product $CB$ is a single station that does both bends at once. The analogy is honest about order: bending then twisting is not the same as twisting then bending. It breaks only in that real machines take time; a product is instant.

Work it out with two famous plane maps. Let $R$ rotate by 90 degrees and $S$ shear horizontally:

$$
R=\begin{pmatrix}0 & -1\\ 1 & 0\end{pmatrix},\qquad S=\begin{pmatrix}1 & 1\\ 0 & 1\end{pmatrix}.
$$

To find $RS$, push each column of $S$ through $R$. Column 1 of $S$ is $(1,0)$, and $R(1,0)=(0,1)$. Column 2 of $S$ is $(1,1)$, and $R(1,1)=1\cdot(0,1)+1\cdot(-1,0)=(-1,1)$. So $RS$ has columns $(0,1)$ and $(-1,1)$. Doing the same for $SR$: $S(0,1)=(1,1)$ and $S(-1,0)=(-1,0)$.

$$
RS=\begin{pmatrix}0 & -1\\ 1 & 1\end{pmatrix},\qquad SR=\begin{pmatrix}1 & -1\\ 1 & 0\end{pmatrix}.
$$

In words: the two products have different columns, so they are different maps, and the figure below shows the unit square landing in two different places.

![Shear then rotate versus rotate then shear send the unit square to different parallelograms](/imgs/blogs/matrices-as-linear-maps-what-a-layer-does-3.webp)

The rule that falls out of the worked example is the definition. For $C\in\mathbb{R}^{p\times m}$ and $B\in\mathbb{R}^{m\times n}$, the **product** $CB\in\mathbb{R}^{p\times n}$ has entries

$$
(CB)_{ij}=\sum_{k=1}^{m}C_{ik}\,B_{kj},
$$

which says: column $j$ of $CB$ is $C$ applied to column $j$ of $B$. The shared index $k$ is summed away, exactly the contraction that `einsum("ik,kj->ij", C, B)` spells out in the [einsum post](/blog/machine-learning/mathematics/sums-products-einsum-and-log-sum-exp). The inner sizes must match ($m$ and $m$), and they vanish from the result.

### The trap in reading order

With column vectors, $CBx=C(Bx)$, so $B$ acts first: **read products right to left**, like nested function calls $C(B(x))$. Most deep-learning papers, including the Transformer paper, put the vector on the left as a row: $xW_1W_2$ means apply $W_1$ first, then $W_2$, left to right. Neither convention is wrong. What goes wrong is reading one with the habits of the other.

| Convention | Written | Applied first | Weight shape for $n\to m$ | Where you see it |
| --- | --- | --- | --- | --- |
| Column vector | $W x + b$ | rightmost matrix | $m\times n$ | textbooks, most theory appendices |
| Row vector | $x W + b$ | leftmost matrix | $n\times m$ | Transformer paper, most code |
| PyTorch `nn.Linear` | $x A^{T} + b$ | n/a | $A$ stored as $m\times n$ | `weight` has shape `(out, in)` |

The two conventions are transposes of each other: $(Wx)^{T}=x^{T}W^{T}$. When a derivation looks off by a transpose, check the convention before you check the algebra.

## Affine maps: the bias slides the whole sheet

**What it is for.** A bias lets a layer shift its output away from the origin, so a zero input does not have to produce a zero output.

Back to the rubber sheet. A linear map stretches it with the pin fixed. An **affine map** stretches it and then picks up the pin and slides the whole sheet. The animation below shows the same $A$ as the opening figure, followed by a slide.

<figure class="blog-anim">
<svg viewBox="0 0 640 400" role="img" aria-label="A grid and the unit square morph under the matrix A with rows (2, -1) and (1, 1), then slide by the bias b = (0.5, -1)" style="width:100%;height:auto;max-width:760px">
<style>
.a8-grid{stroke:var(--border,#d1d5db);stroke-width:1;vector-effect:non-scaling-stroke}
.a8-axis{stroke:var(--text-secondary,#6b7280);stroke-width:1.5;vector-effect:non-scaling-stroke}
.a8-sq{fill:var(--accent,#6366f1);fill-opacity:.25;stroke:var(--accent,#6366f1);stroke-width:2.5;vector-effect:non-scaling-stroke}
.a8-e{stroke:var(--accent,#6366f1);stroke-width:3;vector-effect:non-scaling-stroke}
.a8-o{fill:var(--text-primary,#1f2937)}
.a8-lbl{font:600 17px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.a8-sub{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
@keyframes a8-map{0%,15%{transform:matrix(1,0,0,1,0,0)}35%,50%{transform:matrix(2,1,-1,1,0,0)}70%,85%{transform:matrix(2,1,-1,1,0.5,-1)}100%{transform:matrix(1,0,0,1,0,0)}}
@keyframes a8-p1{0%,18%{opacity:1}25%,93%{opacity:0}100%{opacity:1}}
@keyframes a8-p2{0%,25%{opacity:0}33%,52%{opacity:1}60%,100%{opacity:0}}
@keyframes a8-p3{0%,60%{opacity:0}68%,87%{opacity:1}95%,100%{opacity:0}}
.a8-m{transform-origin:0 0;animation:a8-map 12s ease-in-out infinite}
.a8-t1{animation:a8-p1 12s ease-in-out infinite}
.a8-t2{opacity:0;animation:a8-p2 12s ease-in-out infinite}
.a8-t3{opacity:0;animation:a8-p3 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.a8-m{animation:none;transform:matrix(2,1,-1,1,0,0)}.a8-t1,.a8-t3{animation:none;opacity:0}.a8-t2{animation:none;opacity:1}}
</style>
<g transform="translate(320,250) scale(55,-55)">
<line class="a8-axis" x1="-5.5" y1="0" x2="5.5" y2="0"/>
<line class="a8-axis" x1="0" y1="-2.6" x2="0" y2="4.2"/>
<g class="a8-m">
<line class="a8-grid" x1="-4" y1="-3" x2="-4" y2="4"/>
<line class="a8-grid" x1="-3" y1="-3" x2="-3" y2="4"/>
<line class="a8-grid" x1="-2" y1="-3" x2="-2" y2="4"/>
<line class="a8-grid" x1="-1" y1="-3" x2="-1" y2="4"/>
<line class="a8-grid" x1="0" y1="-3" x2="0" y2="4"/>
<line class="a8-grid" x1="1" y1="-3" x2="1" y2="4"/>
<line class="a8-grid" x1="2" y1="-3" x2="2" y2="4"/>
<line class="a8-grid" x1="3" y1="-3" x2="3" y2="4"/>
<line class="a8-grid" x1="4" y1="-3" x2="4" y2="4"/>
<line class="a8-grid" x1="-4" y1="-3" x2="4" y2="-3"/>
<line class="a8-grid" x1="-4" y1="-2" x2="4" y2="-2"/>
<line class="a8-grid" x1="-4" y1="-1" x2="4" y2="-1"/>
<line class="a8-grid" x1="-4" y1="0" x2="4" y2="0"/>
<line class="a8-grid" x1="-4" y1="1" x2="4" y2="1"/>
<line class="a8-grid" x1="-4" y1="2" x2="4" y2="2"/>
<line class="a8-grid" x1="-4" y1="3" x2="4" y2="3"/>
<line class="a8-grid" x1="-4" y1="4" x2="4" y2="4"/>
<polygon class="a8-sq" points="0,0 1,0 1,1 0,1"/>
<line class="a8-e" x1="0" y1="0" x2="1" y2="0"/>
<line class="a8-e" x1="0" y1="0" x2="0" y2="1"/>
<circle class="a8-o" cx="0" cy="0" r="0.09"/>
</g>
</g>
<text class="a8-lbl a8-t1" x="20" y="30">Identity: e₁ = (1, 0), e₂ = (0, 1), area 1</text>
<text class="a8-sub a8-t1" x="20" y="52">Nothing has happened yet.</text>
<text class="a8-lbl a8-t2" x="20" y="30">Linear: x ↦ A x, columns (2, 1) and (−1, 1), area 3</text>
<text class="a8-sub a8-t2" x="20" y="52">Grid lines stay straight, parallel, evenly spaced; the origin stays put.</text>
<text class="a8-lbl a8-t3" x="20" y="30">Affine: x ↦ A x + b, b = (0.5, −1)</text>
<text class="a8-sub a8-t3" x="20" y="52">Same shape, slid as a whole: the origin dot moves to b.</text>
</svg>
<figcaption>The same matrix as the first figure, in motion. The linear map stretches and shears the grid while keeping the origin fixed. Adding the bias then slides the whole sheet. That slide is the only thing an affine layer adds to a linear one.</figcaption>
</figure>

A worked example with the bias $b=(0.5,-1)$. The affine map is $x\mapsto Ax+b$. The origin goes to $A\cdot 0+b=(0.5,-1)$, so the pin has moved. The tip of $e_1$ goes to $(2,1)+(0.5,-1)=(2.5,0)$. The top corner $(1,1)$ goes to $(1,2)+(0.5,-1)=(1.5,1)$. Every point moves by the same $b$ after the stretch, so the parallelogram keeps its shape and area ${3}$.

Formally, an **affine map** is $f(x)=Ax+b$ with $A\in\mathbb{R}^{m\times n}$ and a **bias** (translation) $b\in\mathbb{R}^{m}$. It is linear exactly when $b=0$, because linearity forces $f(0)=0$ and here $f(0)=b$. That is the failure of $g(x)=3x+1$ from earlier: the ${1}$ is a bias.

There is a standard trick that turns affine back into linear, and you will meet it in graphics code and in proofs. Append a constant ${1}$ to the input:

$$
\begin{pmatrix}Ax+b\\ 1\end{pmatrix}=\begin{pmatrix}A & b\\ 0 & 1\end{pmatrix}\begin{pmatrix}x\\ 1\end{pmatrix}.
$$

In words: an affine map in $n$ dimensions is a linear map in $n+1$ dimensions that never touches the extra coordinate. The numerical check later uses the same trick, appending a column of ones so that least squares can fit a bias.

## Stacking linear layers without a nonlinearity buys nothing

**What it is for.** This fact explains why every neural network puts an activation function between its matrices: without one, depth is an illusion.

Start in one dimension. Layer one computes $h=3x+1$ and layer two computes $y=2h+5$. Substitute: $y=2(3x+1)+5=6x+7$. Two layers with four parameters collapsed into one layer with two. The stack can only ever draw a straight line, and so can a stack of a thousand such layers.

The matrix version is the same substitution. Two affine layers in column convention give

$$
W_2(W_1x+b_1)+b_2=(W_2W_1)\,x+(W_2b_1+b_2),
$$

which is one affine layer with weight $W_2W_1$ and bias $W_2b_1+b_2$. In words: the composition of two affine maps is affine, and the product matrix is the whole story. The proof section below does this carefully, and the numerical check confirms it to machine precision on Transformer-sized matrices.

Now watch the nonlinearity break the collapse. Take one input $x$, three hidden units with pre-activations $h_1=x+1$, $h_2=x$, $h_3=x-1$, and output weights $(1,-2,2)$. Without an activation, the output is $h_1-2h_2+2h_3=(x+1)-2x+2(x-1)=x-1$: a line. With ReLU, $\mathrm{relu}(t)=\max(0,t)$, applied to each unit, check $x=0.5$: the pre-activations are $(1.5,\,0.5,\,-0.5)$, the ReLU outputs are $(1.5,\,0.5,\,0)$, and the output is ${1.5-1+0=0.5}$. The line would have given ${0.5-1=-0.5}$.

![Same weights with and without ReLU: a straight line versus a curve with kinks at -1, 0 and 1](/imgs/blogs/matrices-as-linear-maps-what-a-layer-does-4.webp)

The figure plots both over $[-2,2]$. The ReLU version is ${0}$ until $x=-1$, rises to ${1}$ at $x=0$, falls to ${0}$ at $x=1$, then climbs. Each hidden unit contributes one kink, at the point where its pre-activation crosses zero. For $x\geq 1$ all three units are active, so the ReLU curve sits exactly on the line $x-1$; the nonlinearity matters only where some unit is switched off. That is the whole mechanism of a ReLU network: a piecewise linear function, with one possible kink per hidden unit, glued from linear pieces.

There is a second consequence that matters more in practice than in textbooks. The product $W_2W_1$ cannot have more rank than the narrowest layer in the stack. A $512\to 2048\to 512$ linear stack gains nothing over one $512\times 512$ matrix. A $512\to 64\to 512$ stack is worse: it can only represent maps of rank at most ${64}$. That restriction is a bug in a network and a feature in LoRA, which we meet below.

## Shapes and batching: one matrix, many tokens

**What it is for.** Writing inputs as rows lets one matrix multiplication process a whole batch of tokens at once, which is why GPUs like these layers.

In the row convention a token is a row vector $x\in\mathbb{R}^{1\times 512}$. Stack $n$ tokens as rows of $X\in\mathbb{R}^{n\times 512}$, and $XW_1$ applies the same $W_1$ to every row in one product, because row $i$ of $XW_1$ is (row $i$ of $X$) times $W_1$. A batch of $B$ sequences adds a leading axis, $(B,n,512)$, and frameworks broadcast the same product over it. The bias $b_1$, a single row, is added to every row by broadcasting.

![Proportional blocks: X (4x512) times W1 (512x2048) gives H (4x2048), then H times W2 gives Y (4x512)](/imgs/blogs/matrices-as-linear-maps-what-a-layer-does-5.webp)

The shape rule is $(m\times k)(k\times n)=(m\times n)$: the inner sizes must match and they disappear, as the amber blocks in the figure show. With the Transformer base sizes, $d_{\text{model}}=512$ and $d_{ff}=2048$, four tokens give $(4\times 512)(512\times 2048)=(4\times 2048)$, then $(4\times 2048)(2048\times 512)=(4\times 512)$. The output has the same shape as the input, which is what lets the paper add it back with a residual connection.

The numbers are worth one pass by hand. $W_1$ holds $512\times 2048=1{,}048{,}576$ weights and $b_1$ holds ${2048}$; $W_2$ holds another $1{,}048{,}576$ and $b_2$ holds ${512}$. The layer totals $2{,}099{,}712$ parameters. For four tokens the first product costs $4\times 512\times 2048=4{,}194{,}304$ multiply-adds, and the second costs the same. Cost grows linearly in the number of tokens because each token is handled independently: that is the meaning of "position-wise" in the paper's section title.

## Reading the Transformer's feed-forward layer

Everything is now in place to read the line from the hook.

### Paper sentence, decoded

**Source:** Ashish Vaswani et al., "Attention Is All You Need", NeurIPS 2017, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762), Section 3.3, Eq. 2:

$$
\mathrm{FFN}(x)=\max(0,\,xW_1+b_1)\,W_2+b_2
$$

![Each symbol in the FFN equation labelled with its shape and its job](/imgs/blogs/matrices-as-linear-maps-what-a-layer-does-6.webp)

| Symbol | What it is | Built in this post |
| --- | --- | --- |
| $x$ | one token's vector, a row of size ${512}$ | shapes and batching |
| $W_1$ | weight matrix, $512\times 2048$, expands the token | a matrix is a list of images; row convention |
| $b_1$ | bias row of size ${2048}$, shifts each hidden unit's zero point | affine maps |
| $xW_1+b_1$ | an affine map into 2048 dimensions | affine maps |
| $\max(0,\cdot)$ | ReLU, applied to each of the 2048 entries separately | the collapse section |
| $W_2$ | weight matrix, $2048\times 512$, projects back | composition, left to right |
| $b_2$ | bias row of size ${512}$ | affine maps |

In words: expand each token with an affine map into a space four times wider, zero out every negative coordinate, then map back down with a second affine map. The paper adds that the same $W_1,W_2$ serve every position while each layer has its own, and that the layer can equally be described as two convolutions with kernel size ${1}$. Both remarks are restatements of the row picture: one matrix, applied to every row.

Now the hook's question has an answer. Delete the $\max$ and the proof below shows the layer becomes $x(W_1W_2)+(b_1W_2+b_2)$, a single $512\times 512$ affine map whose two million parameters do the work of about 263 thousand ($512\times 512+512=262{,}656$). The ReLU is the only reason the expansion to 2048 buys anything: it gives the layer up to 2048 kinks with which to bend.

Later models kept the two matrices and swapped the activation. BERT and GPT-2 use GELU, a smooth relative of ReLU. Shazeer's "GLU Variants Improve Transformer" ([arXiv:2002.05202](https://arxiv.org/abs/2002.05202), 2020) writes the ReLU, GELU and gated variants side by side, including the SwiGLU form $(\mathrm{Swish}(xW)\otimes xV)\,W_2$ with a third matrix $V$ and an elementwise product $\otimes$, and reports that several gated variants improve quality over ReLU and GELU. LLaMA-style models adopted SwiGLU. Every variant keeps the structure decoded here: affine, elementwise nonlinearity, affine.

## Proof, read line by line

**Statement.** Let $f:\mathbb{R}^{n}\to\mathbb{R}^{m}$ and $g:\mathbb{R}^{m}\to\mathbb{R}^{p}$ with

- (A1) $f$ is linear, with matrix $B\in\mathbb{R}^{m\times n}$, so $f(x)=Bx$;
- (A2) $g$ is linear, with matrix $C\in\mathbb{R}^{p\times m}$, so $g(y)=Cy$;
- (A3) the spaces are finite-dimensional, so every linear map has a matrix (the columns lemma above).

Then $g\circ f$ is linear and its matrix is the product $CB$. Corollary: two stacked affine layers equal one affine layer.

**The idea in one sentence.** Linearity passes through each map in turn, and the columns lemma reduces "same matrix" to "same images of $e_1,\dots,e_n$", which the product definition delivers column by column.

1. Fix $u,v\in\mathbb{R}^{n}$ and scalars $a,b$. Then $(g\circ f)(au+bv)=g\big(f(au+bv)\big)$. *Definition of composition.*
2. $\quad=g\big(a\,f(u)+b\,f(v)\big)$. *By (A1), linearity of $f$.*
3. $\quad=a\,g\big(f(u)\big)+b\,g\big(f(v)\big)$. *By (A2), linearity of $g$ applied to the two vectors $f(u),f(v)\in\mathbb{R}^{m}$.*
4. $\quad=a\,(g\circ f)(u)+b\,(g\circ f)(v)$. *Definition of composition again.* So $g\circ f$ is linear.
5. Since $g\circ f$ is linear between finite-dimensional spaces, it has a matrix $M\in\mathbb{R}^{p\times n}$ whose column $j$ is $(g\circ f)(e_j)$. *Columns lemma, needs step 4 and (A3).*
6. $(g\circ f)(e_j)=g(Be_j)=g(B_{:,j})$. *By (A1), and $Be_j$ picks out column $j$ of $B$.*
7. $g(B_{:,j})=C\,B_{:,j}$. *By (A2).*
8. Entry $i$ of $C\,B_{:,j}$ is $\sum_{k}C_{ik}B_{kj}=(CB)_{ij}$. *Definition of the matrix product.*
9. So column $j$ of $M$ equals column $j$ of $CB$ for every $j$, hence $M=CB$. *Two matrices with equal columns are equal.* $\blacksquare$
10. Corollary. Let $f(x)=Bx+c$ and $g(y)=Cy+d$ be affine. Then $g(f(x))=C(Bx+c)+d$. *Substitute.*
11. $\quad=CBx+Cc+d$. *Linearity of $y\mapsto Cy$, which is (A2).*
12. So $g\circ f$ is affine with weight $CB$ and bias $Cc+d$. In the paper's row convention the same steps read $(xW_1+b_1)W_2+b_2=x(W_1W_2)+(b_1W_2+b_2)$. *Transpose of steps 10 and 11.*
13. For $L$ layers, apply step 12 to layers ${1}$ to $L-1$ (already one affine map by the induction hypothesis) and layer $L$. *Induction on depth.*

**Where each assumption was used.**

| Assumption | Used in step |
| --- | --- |
| (A1) $f$ linear | 2, 6 |
| (A2) $g$ linear | 3, 7, 11 |
| (A3) finite dimensions | 5 |

**What breaks without (A1).** Let $f$ be ReLU on $\mathbb{R}$ and $g$ the identity. Step 2 needs $f(1\cdot 1+1\cdot(-1))=f(1)+f(-1)$, but the left side is $f(0)=0$ and the right side is ${1+0=1}$. Step 2 fails, the composition is not linear, and the kinked curve in the ReLU figure is the visible result.

**Proof moves used:** `induction (layers / steps)`. Source: the standard result that the matrix of a composition is the product of the matrices, Sheldon Axler, *Linear Algebra Done Right*, Chapter 3, section 3.C (matrices of linear maps); the affine corollary is simplified from the same argument.

## Check it numerically

The script builds a Transformer-base FFN with random weights, checks the collapse identity from steps 10 to 12, measures how far the ReLU version is from the best possible single affine map, and recomputes the two products from the composition section.

```python
import numpy as np
rng = np.random.default_rng(0)
d_model, d_ff, n = 512, 2048, 4

X  = rng.standard_normal((n, d_model))           # 4 token rows
W1 = rng.standard_normal((d_model, d_ff)) / d_model**0.5
b1 = rng.standard_normal(d_ff)
W2 = rng.standard_normal((d_ff, d_model)) / d_ff**0.5
b2 = rng.standard_normal(d_model)

# 1. Two stacked affine layers, no activation
two_layers = (X @ W1 + b1) @ W2 + b2
W, b = W1 @ W2, b1 @ W2 + b2                     # the collapsed single layer
print("collapse max error:", np.abs(two_layers - (X @ W + b)).max())
print("collapsed W shape:", W.shape, "rank:", np.linalg.matrix_rank(W))

# 2. Insert ReLU: try to fit the output with the best single affine map
Xb  = rng.standard_normal((4096, d_model))       # many inputs this time
Y   = np.maximum(0, Xb @ W1 + b1) @ W2 + b2
A   = np.c_[Xb, np.ones(len(Xb))]                # augment with a 1 for the bias
coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
rel = np.linalg.norm(A @ coef - Y) / np.linalg.norm(Y - Y.mean(0))
print("ReLU FFN, best affine fit, relative residual:", round(rel, 3))

# 3. Order matters
R = np.array([[0, -1], [1, 0]]); S = np.array([[1, 1], [0, 1]])
print("RS =", (R @ S).tolist(), " SR =", (S @ R).tolist())
```

```console
collapse max error: 8.215650382226158e-15
collapsed W shape: (512, 512) rank: 512
ReLU FFN, best affine fit, relative residual: 0.362
RS = [[0, -1], [1, 1]]  SR = [[1, -1], [1, 0]]
```

Without the activation the two-layer stack and the single collapsed layer agree to about $10^{-14}$, which is float64 rounding accumulated over sums of 2048 terms, and the collapsed weight is a plain $512\times 512$ matrix. With ReLU, even the best affine map fitted by least squares over 4096 inputs leaves 36% of the output's variation unexplained, so the nonlinearity is doing real work. The two $2\times 2$ products match the hand computation exactly.

## Where this shows up in practice

**LoRA adapters (Hu et al., 2021).** "LoRA: Low-Rank Adaptation of Large Language Models" ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)) freezes a pretrained weight $W_0$ and trains a low-rank update $BA$, with $B$ tall and $A$ wide, so the adapted layer computes $W_0x+BAx$. Two pieces of this post carry it. The stack $x\mapsto BAx$ is the deliberate bottleneck from the collapse section: a product of a $d\times r$ and an $r\times d$ matrix has rank at most $r$, which is why it is cheap. And because $W_0x+BAx=(W_0+BA)x$ by linearity, the update can be merged into a single matrix after training, which is why the abstract claims no additional inference latency, unlike adapters.

**Tensor parallelism in Megatron-LM (Shoeybi et al., 2019).** "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" ([arXiv:1909.08053](https://arxiv.org/abs/1909.08053)) splits the FFN across GPUs using the column view of the product. Split $W_1=[W_1^{(1)},W_1^{(2)}]$ by columns and $XW_1=[XW_1^{(1)},XW_1^{(2)}]$: each GPU computes its own block of hidden units. Because the activation (GeLU in their model) acts on each entry separately, it can be applied to each block locally. Then $W_2$ is split by rows, each GPU produces a partial product, and one all-reduce sums them. The whole scheme is the index sum $\sum_k H_{ik}(W_2)_{kj}$ cut into two ranges of $k$.

**PyTorch `nn.Linear` (official docs).** The [`torch.nn.Linear` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) states that the module applies $y=xA^{T}+b$, stores `weight` with shape `(out_features, in_features)`, and accepts inputs of shape `(*, in_features)` with any number of leading dimensions. That is the convention table above, made concrete: the stored matrix is the column-convention $W$, the computation is the row-convention product, and the leading `*` is the batching rule. When a weight you load from a paper's checkpoint looks transposed, this line is usually why.

## What to remember

> - A matrix is a linear map written down by its columns: column $j$ is where $e_j$ lands, and $Ax$ mixes the columns with weights $x$.
> - Matrix products are composition. With column vectors read right to left; with the row vectors most papers use, read left to right. $RS\neq SR$ in general.
> - A bias makes a map affine: it slides the output and moves the origin to $b$. Affine maps compose to affine maps.
> - Two linear or affine layers with nothing between them equal one layer, with weight $W_2W_1$ and rank no larger than the narrowest layer. The activation is what makes depth count.
> - Shapes: $(m\times k)(k\times n)=(m\times n)$; one matrix serves every token row, which is what "position-wise" means.

## Where next

If any notation here felt slippery, the series opener, [How to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper), has the reading protocol, and the [proof techniques post](/blog/machine-learning/mathematics/proof-techniques-for-ml-researchers) covers the induction step used in step 13. Within this track, [rank, null space and low-rank structure](/blog/machine-learning/mathematics/rank-null-space-and-low-rank-structure) takes the column view to its conclusion and explains the LoRA bottleneck properly, [orthogonality, projections and QR](/blog/machine-learning/mathematics/orthogonality-projections-and-qr) studies the maps that preserve lengths, and [determinants, volume and invertibility](/blog/machine-learning/mathematics/determinants-volume-and-invertibility) explains the area ${3}$ in the opening figure. The upcoming Track C posts on eigenvectors and the singular value decomposition then answer the question this post leaves open: what is the simplest possible description of what a given matrix does to space?

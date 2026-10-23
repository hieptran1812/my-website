---
title: "Sums, Products, Einsum and Log-Sum-Exp: The Index Arithmetic Behind Attention"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "Manipulate sigma and pi sums with confidence, read any einsum string from its indices, and see why softmax overflows and how log-sum-exp fixes it, ending with attention written as two einsums."
tags: ["mathematics", "math-for-ai", "einsum", "softmax", "log-sum-exp", "attention", "numerical-stability", "summation"]
category: "machine-learning"
subcategory: "Mathematics"
author: "Hiep Tran"
featured: true
readTime: 18
---

> **Level:** Foundation · **Needs:** [How to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper), [Sets, functions and maps](/blog/machine-learning/mathematics/sets-functions-and-maps-the-grammar-of-ml-math) · **Unlocks:** Transformer attention, language-model likelihoods, cross-entropy and InfoNCE losses, FlashAttention's tiling math

The single most famous equation in modern machine learning is one line long. Section 3.2.1 of "Attention Is All You Need" (Vaswani et al., 2017) writes it as $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^{T}/\sqrt{d_k})\,V$. It hides three things: a sum over a shared index inside $QK^{T}$, an exponential that overflows if coded as written, and a normalisation over an axis the notation never names.

Language-model papers hide the same machinery in a second line, the log-likelihood $\log p(x)=\sum_{t}\log p(x_t \mid x_{\lt t})$, where a product of probabilities silently became a sum of logs. By the end you will read both lines, rewrite attention as two `einsum` calls, and know why the naive softmax returns `nan` and the one-line fix. The series bet holds here too: paper math is a small, closed vocabulary, and each word learned once with a picture, a proof and a check stops the appendix being a wall.

![Series ladder with rung A lit and arrows to attention, likelihood and softmax-loss papers](/imgs/blogs/sums-products-einsum-and-log-sum-exp-1.webp)

## Sigma and Pi are for-loops with a name

**What they are for.** A sum sign is a compact way to say "add up a list of terms that all follow the same pattern", and a product sign says the same for multiplication.

The cleanest analogy is a for-loop. The symbol $\sum_{i=1}^{n} a_i$ is `total = 0; for i in 1..n: total += a[i]`. The letter under the sign is the loop variable, the numbers are the loop bounds, and the expression to the right is the loop body. The analogy is honest for finite sums, which is all we need today; it breaks for infinite series, where "run the loop forever" needs a limit and the order of terms can change the answer (a later post on limits covers that).

A worked example with small numbers. Take $a = (3, 1, 4)$:

$$
\sum_{i=1}^{3} a_i = 3 + 1 + 4 = 8, \qquad \prod_{i=1}^{3} a_i = 3 \cdot 1 \cdot 4 = 12.
$$

The sum adds the three entries and the product multiplies them. By convention a sum over nothing is ${0}$ and a product over nothing is ${1}$: a loop that never runs leaves the accumulator untouched.

Formally, for a finite index set $I$ and numbers $a_i$, the **sum** $\sum_{i \in I} a_i$ is the total of all $a_i$ with $i$ in $I$, and the **product** $\prod_{i \in I} a_i$ is their product. The index $i$ is a **bound (dummy) variable**: renaming it to $j$ changes nothing, exactly as renaming a loop counter changes nothing.

### The four moves every derivation uses

Almost every line of sum algebra in an ML appendix is one of four moves.

**Move 1, linearity.** Constants pull out and sums of terms split:

$$
\sum_{i} (c\,a_i + b_i) = c \sum_{i} a_i + \sum_{i} b_i .
$$

In words: scaling every term scales the total, and adding two lists termwise adds their totals. Check it with $a = (3,1,4)$, $b = (1,1,1)$, $c = 2$: the left side is ${7 + 3 + 9 = 19}$ and the right side is $2 \cdot 8 + 3 = 19$. It is why the gradient of an average loss is the average of per-example gradients.

**Move 2, splitting the range.** A sum over a set can be cut into sums over disjoint pieces:

$$
\sum_{i=1}^{n} a_i = \sum_{i=1}^{k} a_i + \sum_{i=k+1}^{n} a_i .
$$

In words: add the first block, add the second block, add the two totals. FlashAttention uses it to split a row of scores into tiles.

**Move 3, shifting the index.** Substituting $j = i + 1$ moves the bounds by one in the other direction:

$$
\sum_{i=0}^{n-1} a_{i+1} = \sum_{j=1}^{n} a_j .
$$

In words: the same terms, counted with a different label. Shifting is how telescoping sums collapse.

**Move 4, swapping a finite double sum.** For a grid of numbers $a_{ij}$, summing rows first or columns first gives the same total:

$$
\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij} = \sum_{j=1}^{n} \sum_{i=1}^{m} a_{ij} .
$$

In words: a finite sum does not care about the order you visit its terms. The picture makes this concrete.

![A 3 by 4 grid whose row sums and column sums both total 24](/imgs/blogs/sums-products-einsum-and-log-sum-exp-2.webp)

Read the figure with a finger. The rows are $(1,2,0,3)$, $(4,1,2,1)$ and $(0,3,5,2)$, whose sums are 6, 8 and 10, so row-first gives 24. The columns sum to 5, 6, 7 and 6, so column-first also gives 24. Swapping is free here because the sum is finite; for infinite sums or integrals papers need a condition and flag it with "by Fubini".

### Logs turn products into sums

**What it is for.** The logarithm converts a product, which is numerically fragile, into a sum, which is not. The rule is

$$
\log \prod_{i=1}^{n} a_i = \sum_{i=1}^{n} \log a_i \qquad (a_i \gt 0).
$$

In words: the log of a product is the sum of the logs, as long as every factor is positive. A quick check: $\log(3 \cdot 1 \cdot 4) = \log 12 \approx 2.4849$, and $\log 3 + \log 1 + \log 4 \approx 1.0986 + 0 + 1.3863 = 2.4849$.

The analogy is a slide rule, or the decibel scale: multiplying loudness ratios becomes adding decibels. The reason ML cares is underflow. An autoregressive model assigns a sequence the probability $p(x) = \prod_{t} p(x_t \mid x_{\lt t})$. If each of 400 tokens has probability ${0.1}$, the product is ${10}^{-400}$, below the smallest positive float64 (about ${4.9 \times 10^{-324}}$), so the computer stores exactly zero and every comparison between sequences becomes a tie. The log version is $400 \cdot \log 0.1 \approx -921.03$, an ordinary number.

That is why the likelihood from the hook is a sum of logs, why "negative log-likelihood" is the standard loss, and why a product of independent probabilities becomes an average of log-probabilities you can minibatch.

## Einstein notation: let the indices do the bookkeeping

**What it is for.** Einstein notation describes a tensor operation by naming the axes of its inputs and its output; any axis that disappears is summed over. One string replaces a paragraph of transposes, reshapes and loops.

The physical picture is a patch panel. Each tensor is a device with labelled sockets, one per axis. Plugging two sockets with the same label together means "these axes run in lockstep and get summed away", and the labels left unplugged are what the output keeps. It breaks in one place: a batch index appears in both inputs and the output, shared but not consumed.

![Anatomy of the einsum bij,bjk->bik: the shared index j is summed away](/imgs/blogs/sums-products-einsum-and-log-sum-exp-3.webp)

The figure reads `bij,bjk->bik` the way NumPy and PyTorch do. Left of the arrow: one index list per input. Right: the output's indices. The letter `j` appears in both inputs and not in the output, so it is the summed (contracted) axis. Written out in full:

$$
C_{bik} = \sum_{j} A_{bij}\, B_{bjk}.
$$

In words: for each batch $b$, each output row $i$ and each output column $k$, multiply along the shared axis $j$ and add. That is batched matrix multiplication, and with $A$ of shape $(2,3,4)$ and $B$ of shape $(2,4,5)$ the output has shape $(2,3,5)$; the $j$ axis of size 4 is consumed.

### A 2 by 2 multiplication, by hand

Drop the batch axis and take `ij,jk->ik` with

$$
A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \qquad B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}.
$$

The rule $C_{ik} = \sum_j A_{ij} B_{jk}$ gives four sums of two terms each: $C_{11} = 1 \cdot 5 + 2 \cdot 7 = 19$, $C_{12} = 1 \cdot 6 + 2 \cdot 8 = 22$, $C_{21} = 3 \cdot 5 + 4 \cdot 7 = 43$, and $C_{22} = 3 \cdot 6 + 4 \cdot 8 = 50$. That is the ordinary matrix product: `ij,jk->ik` is its definition written as indices.

### The rules, stated once

An einsum string $s_1, s_2, \dots \to s_{\text{out}}$ means:

1. Every index letter names an axis; the same letter in two operands must have the same size.
2. The result has one axis per letter of $s_{\text{out}}$, in that order.
3. Each entry of the result is the product of the matching input entries, **summed over every letter that does not appear in $s_{\text{out}}$**.

That third rule is Move 4 in disguise: the summed indices can be visited in any order, which is why `opt_einsum` may reorder a contraction for speed without changing the answer.

The physics convention that gave the notation its name is even terser: Einstein wrote $C_{ik} = A_{ij} B_{jk}$ and let a repeated index imply the sum. Equivariant-network and tensor papers still use it: an index repeated in one term with no sigma means a sum.

A small phrasebook of the strings you will meet most:

| String | Operation | Formula |
| --- | --- | --- |
| `i,i->` | dot product | $\sum_i x_i y_i$ |
| `i,j->ij` | outer product | $x_i y_j$ |
| `ii->` | trace | $\sum_i A_{ii}$ |
| `bij,bjk->bik` | batched matmul | $\sum_j A_{bij} B_{bjk}$ |

In a Transformer the query tensor has four axes (batch, heads, tokens, head dimension), and the job is the same as in the 2 by 2 case: find the letters that vanish and you have found the sums.

## Log-sum-exp and a softmax that does not explode

**What it is for.** The **softmax** turns a vector of arbitrary real scores (logits) into a probability distribution, and the **log-sum-exp** is its normaliser written on the log scale. Both are smooth stand-ins for "pick the biggest".

The analogy is a crowded room. Each logit is how loud one voice is, measured in decibels. The room's total loudness is dominated by the loudest voice, slightly more than it when two voices are close, and almost exactly it when one voice shouts. That total is the log-sum-exp; the share of the room each voice owns is the softmax. Where it breaks: real loudness has a ceiling, logits do not.

Formally, for $x \in \mathbb{R}^n$,

$$
\mathrm{LSE}(x) = \log \sum_{j=1}^{n} e^{x_j}, \qquad \mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}} = e^{\,x_i - \mathrm{LSE}(x)} .
$$

In words: exponentiate every score to make it positive, normalise by the total so the shares add to one, and note that the normaliser's log is the LSE. The last equality gives the identity every cross-entropy implementation uses: $\log \mathrm{softmax}(x)_i = x_i - \mathrm{LSE}(x)$.

A small example: $x = (1, 2, 3)$. The exponentials are $e \approx 2.7183$, $e^{2} \approx 7.3891$ and $e^{3} \approx 20.0855$, totalling 30.1929. The softmax is $(0.0900, 0.2447, 0.6652)$, and $\mathrm{LSE}(x) = \log 30.1929 \approx 3.4076$, a little above the max of ${3}$, exactly as the loud-room picture predicts.

The gradient of LSE is the softmax:

$$
\frac{\partial}{\partial x_i} \log \sum_{j} e^{x_j} = \frac{e^{x_i}}{\sum_{j} e^{x_j}} = \mathrm{softmax}(x)_i .
$$

In words: nudging one logit raises the log-normaliser in proportion to that logit's share of the room. (Chain rule on $\log$, then linearity.) It is why the cross-entropy gradient with respect to logits is "predicted probabilities minus the one-hot target".

### Why the naive version overflows

The definition is fine mathematically and broken numerically. A floating-point format has a largest finite number, and $e^{x}$ crosses it quickly.

![log10 of exp(x) crossing the float16 and float32 maxima at x of 11.09 and 88.72](/imgs/blogs/sums-products-einsum-and-log-sum-exp-5.webp)

The chart plots the number of decimal digits in $e^{x}$, which is $x \log_{10} e \approx 0.4343\,x$, a straight line. The largest float16 is ${65504}$, so $e^{x}$ overflows to infinity once $x$ exceeds $\ln 65504 \approx 11.09$. Float32 tops out near ${3.4 \times 10^{38}}$, crossing at $x \approx 88.72$. Even float64 overflows past $x \approx 709.78$. Logits of 12 in a half-precision kernel are not exotic, which is part of why the paper divides by $\sqrt{d_k}$.

Once one exponential is infinite, the softmax computes $\infty / \infty$, which IEEE arithmetic defines as `nan`, which then spreads through every later layer. Very negative scores fail too: every $e^{x_j}$ underflows, giving ${0}/{0}$.

### The max-subtraction trick

The fix uses one identity. For any constant $c$,

$$
\mathrm{softmax}(x + c\mathbf{1}) = \mathrm{softmax}(x), \qquad \mathrm{LSE}(x + c\mathbf{1}) = \mathrm{LSE}(x) + c ,
$$

where $\mathbf{1}$ is the all-ones vector. In words: shifting every logit by the same amount leaves the probabilities alone and shifts the log-normaliser by exactly that amount. Choose $c = -m$ with $m = \max_j x_j$. Then the largest shifted logit is ${0}$, every exponential lies in $(0, 1]$, nothing can overflow, and the denominator is at least ${1}$ because the max term contributes $e^{0} = 1$, so nothing divides by zero either.

<figure class="blog-anim">
<svg viewBox="0 0 720 300" role="img" aria-label="Three logit bars slide up and down together by a constant c while the three softmax probability bars on the right stay exactly the same height" style="width:100%;height:auto;max-width:760px">
<style>
.s4-axis{stroke:var(--text-secondary,#6b7280);stroke-width:1.5}
.s4-zero{stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:5 5}
.s4-logit{fill:var(--accent,#6366f1);opacity:.85}
.s4-prob{fill:var(--surface,#f3f4f6);stroke:var(--text-primary,#1f2937);stroke-width:1.5}
.s4-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.s4-sm{font:13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
@keyframes s4-shift{0%{transform:translateY(0)}25%{transform:translateY(-50px)}50%{transform:translateY(0)}75%{transform:translateY(50px)}100%{transform:translateY(0)}}
.s4-mv{animation:s4-shift 10s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.s4-mv{animation:none}}
</style>
<text class="s4-lbl" x="180" y="28">logits x + c (all move together)</text>
<text class="s4-lbl" x="540" y="28">softmax(x + c)</text>
<line class="s4-zero" x1="40" y1="200" x2="320" y2="200"/>
<text class="s4-sm" x="22" y="204">0</text>
<g class="s4-mv">
<rect class="s4-logit" x="80" y="160" width="50" height="40"/>
<rect class="s4-logit" x="155" y="120" width="50" height="80"/>
<rect class="s4-logit" x="230" y="80" width="50" height="120"/>
<text class="s4-sm" x="105" y="152">x1 + c</text>
<text class="s4-sm" x="180" y="112">x2 + c</text>
<text class="s4-sm" x="255" y="72">x3 + c</text>
</g>
<line class="s4-axis" x1="400" y1="250" x2="680" y2="250"/>
<rect class="s4-prob" x="440" y="231.1" width="50" height="18.9"/>
<rect class="s4-prob" x="515" y="198.6" width="50" height="51.4"/>
<rect class="s4-prob" x="590" y="110.3" width="50" height="139.7"/>
<text class="s4-sm" x="465" y="223">0.090</text>
<text class="s4-sm" x="540" y="190">0.245</text>
<text class="s4-sm" x="615" y="102">0.665</text>
<text class="s4-sm" x="540" y="275">heights never change: the shift cancels in the ratio</text>
</svg>
<figcaption>Adding the same constant c to every logit slides the whole left panel up and down, yet the softmax on the right is frozen at (0.090, 0.245, 0.665). That invariance is what makes subtracting the max legal.</figcaption>
</figure>

The animation shows the identity as motion: the logit bars drift together, and the probability bars do not move, because $e^{x_i + c} / \sum_j e^{x_j + c}$ has a factor $e^{c}$ on top and bottom that cancels. Now run the trick on logits that break the naive recipe.

![Naive softmax on 1000, 1001, 1002 gives nan; subtracting the max gives 0.0900, 0.2447, 0.6652](/imgs/blogs/sums-products-einsum-and-log-sum-exp-6.webp)

Follow the right panel step by step. With $x = (1000, 1001, 1002)$ the max is $m = 1002$, so the shifted logits are $(-2, -1, 0)$. Their exponentials are $e^{-2} \approx 0.1353$, $e^{-1} \approx 0.3679$ and $e^{0} = 1$, summing to 1.5032. Dividing gives $(0.0900, 0.2447, 0.6652)$, the same answer as for $x = (1, 2, 3)$, which is no accident: these logits are $(1, 2, 3)$ shifted by ${999}$. The log-normaliser is recovered by adding the shift back: $\mathrm{LSE}(x) = 1002 + \log 1.5032 \approx 1002.4076$, which is ${999}$ more than the ${3.4076}$ we computed earlier. The naive panel never gets that far: $e^{1000}$ is infinite even in float64.

In code the stable version is three lines: subtract the row max, exponentiate, divide by the row sum. On the log scale, $\log \mathrm{softmax}(x)_i = (x_i - m) - \log \sum_j e^{x_j - m}$, which never even forms the tiny probabilities that would underflow. This is why frameworks expose a fused `log_softmax` and why you should feed a cross-entropy loss raw logits rather than probabilities.

## Attention, decoded

### Paper sentence, decoded

**Source:** Ashish Vaswani et al., "Attention Is All You Need", 2017, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762), Section 3.2.1, Eq. (1):

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right) V .
$$

| Symbol | What it is | Built in this post |
| --- | --- | --- |
| $Q$ | queries, one row per position, width $d_k$ | einsum operands |
| $K$ | keys, one row per position, width $d_k$ | einsum operands |
| $V$ | values, one row per key, width $d_v$ | einsum operands |
| $QK^{T}$ | all query-key dot products, $S_{ij} = \sum_d Q_{id} K_{jd}$ | the summed index $d$ |
| $\sqrt{d_k}$ | keeps score spread independent of width | overflow chart |
| $\mathrm{softmax}$ | over the key index $j$, per row | stable softmax |
| $\cdot\,V$ | weighted average of value rows, $\sum_j P_{ij} V_{jv}$ | the summed index $j$ |

The scores are $S_{ij} = \frac{1}{\sqrt{d_k}} \sum_d Q_{id} K_{jd}$, the einsum `id,jd->ij`: the head dimension $d$ appears in both inputs and not in the output, so it is contracted. The transpose disappeared: naming axes makes it unnecessary. The weights are $P_{ij} = \mathrm{softmax}_j(S_{i\cdot})$, normalised along $j$ only. The output is $O_{iv} = \sum_j P_{ij} V_{jv}$, the einsum `ij,jv->iv`, contracting the key index.

![Attention as cell grids: Q and K give scores, row softmax, then times V](/imgs/blogs/sums-products-einsum-and-log-sum-exp-4.webp)

Follow the blue row in the figure. Query 2 is $(2, 1, 0)$; its dot product with the last key $(2, 1, 1)$ is $2 \cdot 2 + 1 \cdot 1 + 0 \cdot 1 = 5$, the largest entry in its score row, so after scaling and softmax that key gets the biggest weight and its value row dominates the query's output.

In real code the tensors carry batch and head axes too, and the two strings become `bhid,bhjd->bhij` and `bhij,bhjv->bhiv`; $b$ and $h$ appear everywhere and are never summed. The paper's footnote 4 explains the $\sqrt{d_k}$: if the components of a query and a key are independent with mean ${0}$ and variance ${1}$, their dot product has variance $d_k$, so dividing by $\sqrt{d_k}$ restores unit variance and keeps the softmax away from the saturated, overflow-prone regime we just plotted.

**In words:** for each query, score every key by a scaled dot product, turn the scores into weights that sum to one, and return the weighted average of the values.

## Proof, read line by line

Track A posts teach proof reading itself, so here is a short, real proof decoded step by step: the two facts that make the stable softmax correct. The bound $\max_i x_i \le \mathrm{LSE}(x) \le \max_i x_i + \log n$ appears in Boyd and Vandenberghe, *Convex Optimization*, Section 3.1.5; the proof below is the standard one, written out in full.

**Statement.** Let $x \in \mathbb{R}^n$ and $c \in \mathbb{R}$, under

- (A1) $n$ is finite and $n \ge 1$;
- (A2) every $x_i$ is a finite real number.

Then (i) $\mathrm{softmax}(x + c\mathbf{1}) = \mathrm{softmax}(x)$, and (ii) with $m = \max_i x_i$, $\; m \le \mathrm{LSE}(x) \le m + \log n$.

**The idea in one sentence.** Pull the largest exponential out of the sum as a common factor, then bound what is left between one and $n$.

**Steps.**

1. For (i), write the shifted softmax entry: $\dfrac{e^{x_i + c}}{\sum_j e^{x_j + c}}$. *Definition of softmax applied to $x + c\mathbf{1}$.*
2. Factor: $e^{x_j + c} = e^{c} e^{x_j}$ for every $j$. *The exponential turns sums into products.*
3. Pull $e^{c}$ out of the denominator's sum: $\sum_j e^{c} e^{x_j} = e^{c} \sum_j e^{x_j}$. *Linearity, Move 1; the sum is finite by (A1).*
4. Cancel $e^{c}$ top and bottom, which is legal because $e^{c} \gt 0$. *The result is $\mathrm{softmax}(x)_i$, proving (i).*
5. For (ii), let $m = \max_i x_i$, which exists because the list is finite and non-empty. *Uses (A1).*
6. Write each exponent as $x_j = m + (x_j - m)$. *Add and subtract $m$.*
7. Then $\sum_j e^{x_j} = e^{m} \sum_j e^{x_j - m}$, so $\mathrm{LSE}(x) = m + \log \sum_j e^{x_j - m}$. *Steps 2 and 3 with $c = -m$, then log of a product is a sum of logs.*
8. Every term satisfies $0 \lt e^{x_j - m} \le 1$, since $x_j - m \le 0$. *Exponential is positive and increasing; (A2) keeps every term a real number.*
9. At least one term, the maximiser's, equals $e^{0} = 1$, so the sum is at least ${1}$. *Definition of $m$.*
10. There are $n$ terms, each at most ${1}$, so the sum is at most $n$. *Bound each term by its maximum, add.*
11. Hence $1 \le \sum_j e^{x_j - m} \le n$; take logs, which preserves order: $0 \le \log \sum_j e^{x_j - m} \le \log n$. *Log is increasing.*
12. Add $m$ and use step 7: $m \le \mathrm{LSE}(x) \le m + \log n$. *This proves (ii).*

**Where each assumption was used.**

| Assumption | Steps |
| --- | --- |
| (A1) finite, non-empty | 3, 5, 10 |
| (A2) finite entries | 8, 9 |

**What breaks without it.** Drop (A2) and allow $-\infty$, which is exactly what an attention mask writes into the scores. If every entry of a row is masked, then $m = -\infty$, step 6 subtracts infinity from infinity, the sum in step 9 is zero rather than at least one, and the softmax is ${0}/{0}$. That is the real-world "fully masked row returns `nan`" bug that bites padded batches; the post on [attention and masking bugs](/blog/machine-learning/debugging-training/attention-and-masking-bugs) walks through it. Dropping (A1) breaks the upper bound instead: with $n$ zeros, $\mathrm{LSE} = \log n$, which grows without limit, so no bound independent of $n$ can hold.

**Proof moves used:** `add-and-subtract`.

Note that step 7 is the stable algorithm itself: in numerical ML the proof and the code are often the same object.

## Check it numerically

The snippet runs both softmaxes in float64 and float16, checks the LSE bounds, and confirms that attention as einsum matches the matmul form and ignores a shift.

```python
import numpy as np
np.random.seed(0)

def softmax_naive(x):
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def softmax_stable(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)

def lse(x):
    m = x.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True))).squeeze(-1)

x = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    print("naive  float64:", softmax_naive(x))
print("stable float64:", softmax_stable(x).round(6))
print("LSE:", lse(x).round(6), " max:", x.max(), " max+log n:", (x.max() + np.log(3)).round(6))

h = np.array([10.0, 11.0, 12.0], dtype=np.float16)  # fp16 overflows at x > 11.09
with np.errstate(over="ignore", invalid="ignore"):
    print("naive  float16:", softmax_naive(h))
print("stable float16:", softmax_stable(h))

# attention as einsum vs matmul; softmax invariance to a constant shift
Q, K, V = (np.random.randn(2, 4, 8, 16) for _ in range(3))  # (batch, heads, n, d)
S = np.einsum("bhid,bhjd->bhij", Q, K) / np.sqrt(16)
out = np.einsum("bhij,bhjv->bhiv", softmax_stable(S), V)
ref = softmax_stable(Q @ K.swapaxes(-1, -2) / 4.0) @ V
print("einsum vs matmul max |diff|:", np.abs(out - ref).max())
print("shift by 7 max |diff|:", np.abs(softmax_stable(S + 7) - softmax_stable(S)).max())
```

Running it with `python3` and NumPy prints:

```
naive  float64: [nan nan nan]
stable float64: [0.090031 0.244728 0.665241]
LSE: 1002.407606  max: 1002.0  max+log n: 1003.098612
naive  float16: [ 0.  0. nan]
stable float16: [0.0901 0.2448 0.6655]
einsum vs matmul max |diff|: 6.661338147750939e-16
shift by 7 max |diff|: 3.3306690738754696e-16
```

The float16 line is the instructive failure. $e^{10} \approx 22026$ and $e^{11} \approx 59874$ both fit under ${65504}$, but $e^{12}$ does not, so the sum is infinite: the first two entries become finite over infinity, which is ${0}$, and the last becomes infinity over infinity, which is `nan`. The stable float16 answer matches float64 to about three digits, all a 10-bit mantissa holds. The LSE of ${1002.4076}$ sits between the max of ${1002}$ and max plus $\log 3$, as proved. The two agreement checks land near ${10}^{-16}$, a few units of float64 rounding: einsum and matmul add the same products in different orders, which changes rounding only.

## Where this shows up in practice

**FlashAttention's tiled softmax.** Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135), 2022), never materialises the full score matrix. Section 3.1 splits each score row into blocks (Move 2), keeps for each block its running max $m$ and its shifted sum $\ell = \sum_j e^{x_j - m}$, and merges two blocks with $m = \max(m_a, m_b)$ and $\ell = e^{m_a - m}\ell_a + e^{m_b - m}\ell_b$. That merge is step 7 of our proof applied twice: rescale each partial sum to the common max, then add. The post on [kernel fusion and FlashAttention](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) covers the memory side.

**Online softmax in one pass.** Milakov and Gimelshein, "Online normalizer calculation for softmax" ([arXiv:1805.02867](https://arxiv.org/abs/1805.02867), NVIDIA, 2018), observed that the max and the normaliser can be computed in a single sweep by updating the running sum whenever a new max appears, rescaling the old sum by $e^{m_{\text{old}} - m_{\text{new}}}$. The math is again the max-subtraction identity, applied one element at a time.

**Framework primitives.** PyTorch exposes [`torch.logsumexp`](https://pytorch.org/docs/stable/generated/torch.logsumexp.html), documented as numerically stabilised, and [`torch.einsum`](https://pytorch.org/docs/stable/generated/torch.einsum.html), which accepts exactly the strings in the phrasebook above. Its cross-entropy loss takes raw logits and applies `log_softmax` internally, which is the $x_i - \mathrm{LSE}(x)$ identity in production. In a custom loss, an InfoNCE objective or a mixture-of-experts router, reach for these primitives instead of `exp` followed by `log`.

## What to remember

> - Sigma and Pi are for-loops; the index is a dummy variable you may rename.
> - Four moves cover most derivations: linearity, splitting, index shifting, and swapping finite double sums.
> - Log turns a product of probabilities into a sum, which is why likelihoods are always written as sums of logs.
> - In an einsum string, any index that appears in the inputs but not the output is summed. Attention is `id,jd->ij`, a scale, a row softmax, then `ij,jv->iv`.
> - Softmax is invariant to adding a constant, LSE shifts by it, and $\max \le \mathrm{LSE} \le \max + \log n$; subtract the max before you exponentiate.

## Where next

If any notation here felt unfamiliar, [How to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper) is the decoder this post builds on, and [Sets, functions and maps](/blog/machine-learning/mathematics/sets-functions-and-maps-the-grammar-of-ml-math) explains the "softmax is a map from scores to the simplex" view. The proof above used one move; [Proof techniques for ML researchers](/blog/machine-learning/mathematics/proof-techniques-for-ml-researchers) shows the full set, and [The inequality toolkit every appendix uses](/blog/machine-learning/mathematics/the-inequality-toolkit-every-appendix-uses) turns bounds like step 10 into reflexes. [Limits, big-O and asymptotics in ML papers](/blog/machine-learning/mathematics/limits-big-o-and-asymptotics-in-ml-papers) decodes cost claims about attention. A later post on vectors and matrices gives einsum its geometric meaning.

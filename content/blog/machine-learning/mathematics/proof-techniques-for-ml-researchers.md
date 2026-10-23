---
title: "Proof Techniques for ML Researchers: Reading the Arguments Behind the Theorems"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "Learn the six proof techniques, the quantifier order and the stock phrases that ML papers use, so you can read the universal approximation theorem and follow an appendix proof line by line."
tags: ["mathematics", "math-for-ai", "proofs", "induction", "lipschitz", "universal-approximation", "logic", "quantifiers"]
category: "machine-learning"
subcategory: "Mathematics"
author: "Hiep Tran"
featured: true
readTime: 18
---

> **Level:** Foundation · **Needs:** [How to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper), [Sets, functions and maps](/blog/machine-learning/mathematics/sets-functions-and-maps-the-grammar-of-ml-math) · **Unlocks:** every appendix proof in the series, Lipschitz and robustness arguments, generalization bounds, the rigor track

In 1989 George Cybenko published a theorem every deep learning course still quotes. His Theorem 2 says, nearly word for word: let $\sigma$ be any continuous sigmoidal function; then finite sums of the form $G(x)=\sum_{j=1}^{N}\alpha_j\,\sigma(y_j^\top x+\theta_j)$ are dense in $C(I_n)$; in other words, given any $f \in C(I_n)$ and $\varepsilon \gt 0$, there is a sum $G$ of that form with $\lvert G(x)-f(x)\rvert \lt \varepsilon$ for all $x \in I_n$. Most engineers read that as "neural networks can learn anything." That reading is wrong in three ways, and each hides in the order of "given any", "there is" and "for all".

This post teaches six proof techniques, the quantifier order that decides what a theorem promises, and the stock phrases papers use to compress an argument into four words. By the end you will read Cybenko's statement as a precise game, and you will have followed one appendix-style proof line by line.

## The picture first: where proofs sit

![Ladder map with rung A, the language of papers, lit in blue at the bottom, and arrows from it to appendix proofs, theory sections and the universal approximation theorem](/imgs/blogs/proof-techniques-for-ml-researchers-1.webp)

The diagram above is the mental model. Rung A is the language of papers: notation, sets and functions, and the logic of proofs. Every rung above it states its results as theorems, so a reader who cannot follow a proof can only take the paper's word for it.

The thesis of this series: paper math is a small, closed vocabulary. Proofs are the clearest case. Almost every appendix argument is built from the six techniques below plus a short list of named moves, and once you can name the move, the wall of symbols becomes a sequence of small, checkable steps.

## What a proof is for

A **proof** is an argument that a statement follows from its assumptions, each step justified by a definition, an earlier result, or a rule of logic. Its job is not to make the claim plausible; experiments do that. Its job is to say exactly which assumptions the claim needs, so you know when it stops being true.

A useful analogy is a type checker. A theorem is a type signature for a fact: the hypotheses are the input types, the conclusion is the output type, and the proof is the evidence that the body compiles. The analogy breaks in one place: a type checker runs for you, while with a proof you are the checker.

Most theorems in ML papers have one of two shapes. The first is an implication, "if $P$ then $Q$", written $P \Rightarrow Q$. Here $P$ is the **hypothesis** (the assumptions) and $Q$ is the **conclusion**. The second shape adds quantifiers: "for every $x$" written $\forall x$, and "there exists an $x$" written $\exists x$. The rest of this post is about attacking each shape.

![Decision map: read the shape of the claim, then six branches lead to direct proof, contrapositive, contradiction, induction, construction and counterexample](/imgs/blogs/proof-techniques-for-ml-researchers-2.webp)

The map above is the toolkit at a glance. When a proof feels opaque, first ask which of these six branches the author took.

## Direct proof: start from the hypothesis and compute

A direct proof walks forward: assume $P$, unfold the definitions, push symbols until $Q$ falls out. It is the default because you see exactly how the hypothesis produces the conclusion.

Let us prove a fact that appears in every robustness paper. A function $f$ is **$L$-Lipschitz** if $\lvert f(u)-f(v)\rvert \le L\,\lvert u-v\rvert$ for all inputs $u, v$, where the constant $L \ge 0$ caps how fast the output can change. The claim: every Lipschitz function is continuous.

Continuity means: for every tolerance $\varepsilon \gt 0$ there is a radius $\delta \gt 0$ such that moving the input by less than $\delta$ moves the output by less than $\varepsilon$. Small numbers first. Take $L = 3$ and a target tolerance of ${0.3}$. Choose $\delta = {0.3}/3 = {0.1}$. If $\lvert u - v\rvert \lt {0.1}$, then $\lvert f(u) - f(v)\rvert \le 3 \times \lvert u-v\rvert \lt 3 \times {0.1} = {0.3}$. Done for that tolerance.

The general proof is the same arithmetic with letters. Given $\varepsilon \gt 0$, set $\delta = \varepsilon / L$ (if $L = 0$ the function is constant and any $\delta$ works). Then

$$
\lvert u-v\rvert \lt \delta \quad\Longrightarrow\quad \lvert f(u)-f(v)\rvert \le L\,\lvert u-v\rvert \lt L\delta = \varepsilon .
$$

In words: the Lipschitz constant turns an input budget into an output budget, so we pick the input budget by division. Notice the shape. The claim started with "for every $\varepsilon$", so the proof let the reader hand us an arbitrary $\varepsilon$ and then built a $\delta$ that depends on it. That dependence matters enormously in Cybenko's statement.

## Contrapositive: prove the equivalent statement that is easier

The **contrapositive** of $P \Rightarrow Q$ is $\neg Q \Rightarrow \neg P$, read "if not $Q$, then not $P$". It is the same statement in disguise. You reach for it when $\neg Q$ gives you something concrete to work with and $P$ does not.

![Truth table with columns P, Q, P implies Q, not Q implies not P, and the converse Q implies P; the implication and contrapositive columns match on every row while the converse differs in rows 2 and 3](/imgs/blogs/proof-techniques-for-ml-researchers-3.webp)

The table above is the whole justification: the $P \Rightarrow Q$ and $\neg Q \Rightarrow \neg P$ columns agree on all four rows, so they are the same statement. The **converse** $Q \Rightarrow P$ disagrees in rows 2 and 3. Confusing the converse with the original is the most common reading error in theory sections. Differentiable implies continuous, but $\lvert x\rvert$ is continuous and not differentiable at zero, so the converse fails.

An ML example. Claim: if a deterministic classifier reaches zero training error, the dataset has no two examples with the same input and different labels. A direct proof is awkward, because "zero training error" hands you no particular pair to look at. The contrapositive does.

Contrapositive: if the dataset contains $(x, 0)$ and $(x, 1)$, no classifier reaches zero training error. Proof: a classifier $h$ is a function, so $h(x)$ is one value, ${0}$ or ${1}$, and one of the two examples is misclassified. No network size removes that floor.

## Contradiction: assume the opposite and break something

A **proof by contradiction** assumes the claim is false and derives something impossible, such as $0 \lt 0$, so the assumption was wrong. It suits "no such object exists" or "there is exactly one", where the negation hands you an object to poke.

Claim: a strictly convex function has at most one minimizer. A function $f$ is **strictly convex** if for any two different points $a \ne b$ and any weight $t$ strictly between 0 and 1, $f(ta + (1-t)b) \lt t f(a) + (1-t) f(b)$: the graph lies strictly below every chord.

Small numbers first. Take $f(w) = (w-1)^2$ and pretend both $a = 0$ and $b = 2$ were minimizers. Both give $f = 1$. The midpoint $w = 1$ gives $f(1) = 0$, which is smaller than the supposed minimum value ${1}$. The pretend minimizers were not minimizers at all.

The general proof. Suppose $a \ne b$ are both minimizers with minimum value $m$, so $f(a) = f(b) = m$. Strict convexity at $t = \tfrac{1}{2}$ gives

$$
f\!\left(\tfrac{a+b}{2}\right) \lt \tfrac{1}{2} f(a) + \tfrac{1}{2} f(b) = m ,
$$

so the midpoint has a value below the minimum. That contradicts $m$ being the minimum, so two distinct minimizers cannot exist. This is why ridge regression has a unique solution: with penalty $\lambda \gt 0$ its objective is strictly convex.

## Induction: prove it for one layer, then for the next

**Induction** proves a statement $P(k)$ for every whole number $k$ in two moves. The **base case** shows $P(0)$ (or $P(1)$). The **inductive step** shows that $P(k-1)$ implies $P(k)$ for every $k$. Together they knock over the whole chain, like dominoes: the first falls, and each one knocks over the next.

ML is full of indexed objects: layers, gradient steps, tokens, rounds of an online algorithm. When a statement carries an index that counts something sequential, expect induction in the appendix.

### Induction over time steps

Run gradient descent on $f(w) = \tfrac{\lambda}{2} w^2$ with curvature $\lambda$ and step size $\eta$. The gradient is $\lambda w$, so the update is $w_{t+1} = w_t - \eta\lambda w_t = (1-\eta\lambda)\, w_t$. Claim: $w_t = (1-\eta\lambda)^t\, w_0$ for every step $t$.

Numbers first: $\lambda = 2$, $\eta = {0.1}$, $w_0 = 5$. The factor is $1 - {0.2} = {0.8}$. The updates give $w_1 = 4$, $w_2 = {3.2}$, $w_3 = {2.56}$, and the formula predicts ${0.8}^{3} \times 5 = {0.512} \times 5 = {2.56}$. They agree.

Proof. Base case: $t = 0$ gives $(1-\eta\lambda)^0 w_0 = w_0$. Inductive step: assume $w_{t} = (1-\eta\lambda)^{t} w_0$. Then $w_{t+1} = (1-\eta\lambda)\, w_t = (1-\eta\lambda)^{t+1} w_0$. That is the whole proof, and it already explains a training fact: the iterates shrink exactly when $\lvert 1-\eta\lambda\rvert \lt 1$, the "step size below ${2}/\lambda$" rule.

### Induction over layers

The induction ML theory uses most runs over layers. A network is a composition $F = f_K \circ \cdots \circ f_1$, we know something about each layer, and induction carries it through the stack.

![Four panels showing two points x and x prime after each of three layers; dashed circles of radius 1, 2, 1 and 3 bound the distance, and the realised distances 1, 1.6, 0.7 and 1.9 stay inside](/imgs/blogs/proof-techniques-for-ml-researchers-4.webp)

The figure above follows two inputs at distance 1 through three layers with Lipschitz constants 2, ${0.5}$ and 3. After the first layer the distance can be at most $2 \times 1 = 2$; after the second, at most ${0.5} \times 2 = 1$; after the third, at most $3 \times 1 = 3$. Dashed circles are the bounds; solid segments are actual distances, always inside. Each layer multiplies the bound by its constant: that is the inductive step, proved in full below.

## Construction and counterexample: the two halves of quantifiers

"There exists" statements are proved by **construction**: exhibit the object and check it. The $\delta = \varepsilon / L$ above was one. Another: "some ReLU network computes $\lvert x\rvert$." Construction: $\lvert x\rvert = \operatorname{ReLU}(x) + \operatorname{ReLU}(-x)$. Check: for $x = 3$ it gives ${3} + 0 = 3$, for $x = -2$ it gives ${0} + 2 = 2$.

Statements that begin with "for all" are disproved by a **counterexample**: one case where the claim fails. You never need more than one.

![A tiny ReLU network computing ReLU of one minus ReLU of x minus ReLU of minus x, next to its tent-shaped graph; the chord from minus one to one sits at zero while the function is one at zero, so convexity fails](/imgs/blogs/proof-techniques-for-ml-researchers-5.webp)

A false claim I have heard in a design review: "ReLU is convex, so a ReLU network is convex in its input." The figure above kills it. Take $h(x) = \operatorname{ReLU}(1 - \operatorname{ReLU}(x) - \operatorname{ReLU}(-x)) = \operatorname{ReLU}(1-\lvert x\rvert)$, a tent. Convexity would require the value at the midpoint to be at most the chord:

$$
h(0) \le \tfrac{1}{2} h(-1) + \tfrac{1}{2} h(1) .
$$

The left side is $h(0) = 1$ and the right side is $\tfrac{1}{2} \cdot 0 + \tfrac{1}{2} \cdot 0 = 0$, so the inequality fails and the claim is dead. The diagnosis is instructive too: a composition $g \circ f$ of convex functions is convex when the outer $g$ is also nondecreasing, and the subtraction of $\operatorname{ReLU}(x)$ and $\operatorname{ReLU}(-x)$ breaks that condition. Input-convex neural networks exist precisely because they restrict the weights to keep it.

## Quantifier order: the part that changes the meaning

The order of $\forall$ and $\exists$ is not style. Swapping them produces a different statement, often a false one. Small example over real numbers:

| Statement | Reads as | True? |
| --- | --- | --- |
| $\forall x\ \exists y:\ y \gt x$ | every number has a bigger number | Yes, take $y = x + 1$ |
| $\exists y\ \forall x:\ y \gt x$ | one number is bigger than all numbers | No, $x = y$ breaks it |

The rule: an $\exists$ that comes after a $\forall$ may depend on it. In the first row $y$ is chosen after $x$, so it can be $x+1$; in the second, $y$ comes first and must beat every $x$ at once.

A good tool is the **adversary game**. Read the quantifiers left to right. Each $\forall$ is a move by an adversary who wants the statement to fail; each $\exists$ is your move, made after seeing every earlier move. The statement is true when you have a winning strategy. Continuity is "$\forall \varepsilon\ \exists \delta\ \forall u, v$": the adversary names a tolerance, you name a radius, the adversary picks two close inputs.

<figure class="blog-anim">
<svg viewBox="0 0 640 380" role="img" aria-label="The epsilon game of the universal approximation theorem: the tube around f shrinks from 0.2 to 0.1 to 0.05, and each time a sum with more sigmoids is chosen that stays inside it" style="width:100%;height:auto;max-width:760px">
<style>
.u1-ax{stroke:var(--text-secondary,#6b7280);stroke-width:1.5;fill:none}
.u1-f{stroke:var(--text-primary,#1f2937);stroke-width:2.5;fill:none}
.u1-tube{stroke:var(--text-secondary,#6b7280);stroke-width:1.5;stroke-dasharray:6 5;fill:none}
.u1-g{stroke:var(--accent,#6366f1);stroke-width:3;fill:none}
.u1-t{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.u1-s{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.u1-a{font:600 15px ui-sans-serif,system-ui;fill:var(--accent,#6366f1)}
@keyframes u1-p1{0%,28%{opacity:1}33%,95%{opacity:0}100%{opacity:1}}
@keyframes u1-p2{0%,28%{opacity:0}33%,61%{opacity:1}66%,100%{opacity:0}}
@keyframes u1-p3{0%,61%{opacity:0}66%,95%{opacity:1}100%{opacity:0}}
.u1-1{animation:u1-p1 15s ease-in-out infinite}
.u1-2{animation:u1-p2 15s ease-in-out infinite}
.u1-3{animation:u1-p3 15s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.u1-1,.u1-3{animation:none;opacity:0}.u1-2{animation:none;opacity:1}}
</style>
<path class="u1-ax" d="M60,40 L60,320 L600,320"/>
<text class="u1-s" x="52" y="340">0</text>
<text class="u1-s" x="30" y="325">0</text>
<text class="u1-s" x="322" y="340">0.5</text>
<text class="u1-s" x="30" y="185">0.5</text>
<text class="u1-s" x="592" y="340">1</text>
<text class="u1-s" x="30" y="45">1</text>
<text class="u1-s" x="606" y="324">x</text>
<g class="u1-1">
<path class="u1-tube" d="M60.0,124.0 L64.5,118.9 L69.0,113.8 L73.5,108.7 L78.0,103.6 L82.5,98.6 L87.0,93.7 L91.5,88.9 L96.0,84.1 L100.5,79.5 L105.0,75.0 L109.5,70.6 L114.0,66.4 L118.5,62.3 L123.0,58.4 L127.5,54.7 L132.0,51.2 L136.5,47.8 L141.0,44.7 L145.5,41.8 L150.0,39.1 L154.5,36.7 L159.0,34.5 L163.5,32.5 L168.0,30.8 L172.5,29.3 L177.0,28.1 L181.5,27.2 L186.0,26.5 L190.5,26.1 L195.0,26.0 L199.5,26.1 L204.0,26.5 L208.5,27.2 L213.0,28.1 L217.5,29.3 L222.0,30.8 L226.5,32.5 L231.0,34.5 L235.5,36.7 L240.0,39.1 L244.5,41.8 L249.0,44.7 L253.5,47.8 L258.0,51.2 L262.5,54.7 L267.0,58.4 L271.5,62.3 L276.0,66.4 L280.5,70.6 L285.0,75.0 L289.5,79.5 L294.0,84.1 L298.5,88.9 L303.0,93.7 L307.5,98.6 L312.0,103.6 L316.5,108.7 L321.0,113.8 L325.5,118.9 L330.0,124.0 L334.5,129.1 L339.0,134.2 L343.5,139.3 L348.0,144.4 L352.5,149.4 L357.0,154.3 L361.5,159.1 L366.0,163.9 L370.5,168.5 L375.0,173.0 L379.5,177.4 L384.0,181.6 L388.5,185.7 L393.0,189.6 L397.5,193.3 L402.0,196.8 L406.5,200.2 L411.0,203.3 L415.5,206.2 L420.0,208.9 L424.5,211.3 L429.0,213.5 L433.5,215.5 L438.0,217.2 L442.5,218.7 L447.0,219.9 L451.5,220.8 L456.0,221.5 L460.5,221.9 L465.0,222.0 L469.5,221.9 L474.0,221.5 L478.5,220.8 L483.0,219.9 L487.5,218.7 L492.0,217.2 L496.5,215.5 L501.0,213.5 L505.5,211.3 L510.0,208.9 L514.5,206.2 L519.0,203.3 L523.5,200.2 L528.0,196.8 L532.5,193.3 L537.0,189.6 L541.5,185.7 L546.0,181.6 L550.5,177.4 L555.0,173.0 L559.5,168.5 L564.0,163.9 L568.5,159.1 L573.0,154.3 L577.5,149.4 L582.0,144.4 L586.5,139.3 L591.0,134.2 L595.5,129.1 L600.0,124.0"/>
<path class="u1-tube" d="M60.0,236.0 L64.5,230.9 L69.0,225.8 L73.5,220.7 L78.0,215.6 L82.5,210.6 L87.0,205.7 L91.5,200.9 L96.0,196.1 L100.5,191.5 L105.0,187.0 L109.5,182.6 L114.0,178.4 L118.5,174.3 L123.0,170.4 L127.5,166.7 L132.0,163.2 L136.5,159.8 L141.0,156.7 L145.5,153.8 L150.0,151.1 L154.5,148.7 L159.0,146.5 L163.5,144.5 L168.0,142.8 L172.5,141.3 L177.0,140.1 L181.5,139.2 L186.0,138.5 L190.5,138.1 L195.0,138.0 L199.5,138.1 L204.0,138.5 L208.5,139.2 L213.0,140.1 L217.5,141.3 L222.0,142.8 L226.5,144.5 L231.0,146.5 L235.5,148.7 L240.0,151.1 L244.5,153.8 L249.0,156.7 L253.5,159.8 L258.0,163.2 L262.5,166.7 L267.0,170.4 L271.5,174.3 L276.0,178.4 L280.5,182.6 L285.0,187.0 L289.5,191.5 L294.0,196.1 L298.5,200.9 L303.0,205.7 L307.5,210.6 L312.0,215.6 L316.5,220.7 L321.0,225.8 L325.5,230.9 L330.0,236.0 L334.5,241.1 L339.0,246.2 L343.5,251.3 L348.0,256.4 L352.5,261.4 L357.0,266.3 L361.5,271.1 L366.0,275.9 L370.5,280.5 L375.0,285.0 L379.5,289.4 L384.0,293.6 L388.5,297.7 L393.0,301.6 L397.5,305.3 L402.0,308.8 L406.5,312.2 L411.0,315.3 L415.5,318.2 L420.0,320.9 L424.5,323.3 L429.0,325.5 L433.5,327.5 L438.0,329.2 L442.5,330.7 L447.0,331.9 L451.5,332.8 L456.0,333.5 L460.5,333.9 L465.0,334.0 L469.5,333.9 L474.0,333.5 L478.5,332.8 L483.0,331.9 L487.5,330.7 L492.0,329.2 L496.5,327.5 L501.0,325.5 L505.5,323.3 L510.0,320.9 L514.5,318.2 L519.0,315.3 L523.5,312.2 L528.0,308.8 L532.5,305.3 L537.0,301.6 L541.5,297.7 L546.0,293.6 L550.5,289.4 L555.0,285.0 L559.5,280.5 L564.0,275.9 L568.5,271.1 L573.0,266.3 L577.5,261.4 L582.0,256.4 L586.5,251.3 L591.0,246.2 L595.5,241.1 L600.0,236.0"/>
<path class="u1-g" d="M60.0,179.4 L64.5,179.3 L69.0,179.1 L73.5,178.8 L78.0,178.5 L82.5,178.1 L87.0,177.5 L91.5,176.8 L96.0,176.0 L100.5,174.9 L105.0,173.6 L109.5,171.9 L114.0,169.9 L118.5,167.4 L123.0,164.5 L127.5,161.1 L132.0,157.2 L136.5,152.8 L141.0,148.0 L145.5,142.8 L150.0,137.6 L154.5,132.3 L159.0,127.2 L163.5,122.4 L168.0,118.0 L172.5,114.1 L177.0,110.6 L181.5,107.7 L186.0,105.3 L190.5,103.3 L195.0,101.7 L199.5,100.3 L204.0,99.3 L208.5,98.5 L213.0,97.9 L217.5,97.4 L222.0,97.1 L226.5,96.9 L231.0,96.8 L235.5,96.7 L240.0,96.8 L244.5,97.0 L249.0,97.3 L253.5,97.8 L258.0,98.4 L262.5,99.2 L267.0,100.2 L271.5,101.6 L276.0,103.3 L280.5,105.4 L285.0,108.1 L289.5,111.4 L294.0,115.4 L298.5,120.3 L303.0,126.1 L307.5,132.9 L312.0,140.8 L316.5,149.6 L321.0,159.2 L325.5,169.4 L330.0,180.0 L334.5,190.6 L339.0,200.8 L343.5,210.4 L348.0,219.2 L352.5,227.1 L357.0,233.9 L361.5,239.7 L366.0,244.6 L370.5,248.6 L375.0,251.9 L379.5,254.6 L384.0,256.7 L388.5,258.4 L393.0,259.8 L397.5,260.8 L402.0,261.6 L406.5,262.2 L411.0,262.7 L415.5,263.0 L420.0,263.2 L424.5,263.3 L429.0,263.2 L433.5,263.1 L438.0,262.9 L442.5,262.6 L447.0,262.1 L451.5,261.5 L456.0,260.7 L460.5,259.7 L465.0,258.3 L469.5,256.7 L474.0,254.7 L478.5,252.3 L483.0,249.4 L487.5,245.9 L492.0,242.0 L496.5,237.6 L501.0,232.8 L505.5,227.7 L510.0,222.4 L514.5,217.2 L519.0,212.0 L523.5,207.2 L528.0,202.8 L532.5,198.9 L537.0,195.5 L541.5,192.6 L546.0,190.1 L550.5,188.1 L555.0,186.4 L559.5,185.1 L564.0,184.0 L568.5,183.2 L573.0,182.5 L577.5,181.9 L582.0,181.5 L586.5,181.2 L591.0,180.9 L595.5,180.7 L600.0,180.6"/>
<text class="u1-t" x="330" y="64">1. Adversary picks ε = 0.2</text>
<text class="u1-a" x="330" y="88">2. You pick N = 3 sigmoids</text>
<text class="u1-s" x="330" y="110">max error 0.180 &lt; 0.2</text>
</g>
<g class="u1-2">
<path class="u1-tube" d="M60.0,152.0 L64.5,146.9 L69.0,141.8 L73.5,136.7 L78.0,131.6 L82.5,126.6 L87.0,121.7 L91.5,116.9 L96.0,112.1 L100.5,107.5 L105.0,103.0 L109.5,98.6 L114.0,94.4 L118.5,90.3 L123.0,86.4 L127.5,82.7 L132.0,79.2 L136.5,75.8 L141.0,72.7 L145.5,69.8 L150.0,67.1 L154.5,64.7 L159.0,62.5 L163.5,60.5 L168.0,58.8 L172.5,57.3 L177.0,56.1 L181.5,55.2 L186.0,54.5 L190.5,54.1 L195.0,54.0 L199.5,54.1 L204.0,54.5 L208.5,55.2 L213.0,56.1 L217.5,57.3 L222.0,58.8 L226.5,60.5 L231.0,62.5 L235.5,64.7 L240.0,67.1 L244.5,69.8 L249.0,72.7 L253.5,75.8 L258.0,79.2 L262.5,82.7 L267.0,86.4 L271.5,90.3 L276.0,94.4 L280.5,98.6 L285.0,103.0 L289.5,107.5 L294.0,112.1 L298.5,116.9 L303.0,121.7 L307.5,126.6 L312.0,131.6 L316.5,136.7 L321.0,141.8 L325.5,146.9 L330.0,152.0 L334.5,157.1 L339.0,162.2 L343.5,167.3 L348.0,172.4 L352.5,177.4 L357.0,182.3 L361.5,187.1 L366.0,191.9 L370.5,196.5 L375.0,201.0 L379.5,205.4 L384.0,209.6 L388.5,213.7 L393.0,217.6 L397.5,221.3 L402.0,224.8 L406.5,228.2 L411.0,231.3 L415.5,234.2 L420.0,236.9 L424.5,239.3 L429.0,241.5 L433.5,243.5 L438.0,245.2 L442.5,246.7 L447.0,247.9 L451.5,248.8 L456.0,249.5 L460.5,249.9 L465.0,250.0 L469.5,249.9 L474.0,249.5 L478.5,248.8 L483.0,247.9 L487.5,246.7 L492.0,245.2 L496.5,243.5 L501.0,241.5 L505.5,239.3 L510.0,236.9 L514.5,234.2 L519.0,231.3 L523.5,228.2 L528.0,224.8 L532.5,221.3 L537.0,217.6 L541.5,213.7 L546.0,209.6 L550.5,205.4 L555.0,201.0 L559.5,196.5 L564.0,191.9 L568.5,187.1 L573.0,182.3 L577.5,177.4 L582.0,172.4 L586.5,167.3 L591.0,162.2 L595.5,157.1 L600.0,152.0"/>
<path class="u1-tube" d="M60.0,208.0 L64.5,202.9 L69.0,197.8 L73.5,192.7 L78.0,187.6 L82.5,182.6 L87.0,177.7 L91.5,172.9 L96.0,168.1 L100.5,163.5 L105.0,159.0 L109.5,154.6 L114.0,150.4 L118.5,146.3 L123.0,142.4 L127.5,138.7 L132.0,135.2 L136.5,131.8 L141.0,128.7 L145.5,125.8 L150.0,123.1 L154.5,120.7 L159.0,118.5 L163.5,116.5 L168.0,114.8 L172.5,113.3 L177.0,112.1 L181.5,111.2 L186.0,110.5 L190.5,110.1 L195.0,110.0 L199.5,110.1 L204.0,110.5 L208.5,111.2 L213.0,112.1 L217.5,113.3 L222.0,114.8 L226.5,116.5 L231.0,118.5 L235.5,120.7 L240.0,123.1 L244.5,125.8 L249.0,128.7 L253.5,131.8 L258.0,135.2 L262.5,138.7 L267.0,142.4 L271.5,146.3 L276.0,150.4 L280.5,154.6 L285.0,159.0 L289.5,163.5 L294.0,168.1 L298.5,172.9 L303.0,177.7 L307.5,182.6 L312.0,187.6 L316.5,192.7 L321.0,197.8 L325.5,202.9 L330.0,208.0 L334.5,213.1 L339.0,218.2 L343.5,223.3 L348.0,228.4 L352.5,233.4 L357.0,238.3 L361.5,243.1 L366.0,247.9 L370.5,252.5 L375.0,257.0 L379.5,261.4 L384.0,265.6 L388.5,269.7 L393.0,273.6 L397.5,277.3 L402.0,280.8 L406.5,284.2 L411.0,287.3 L415.5,290.2 L420.0,292.9 L424.5,295.3 L429.0,297.5 L433.5,299.5 L438.0,301.2 L442.5,302.7 L447.0,303.9 L451.5,304.8 L456.0,305.5 L460.5,305.9 L465.0,306.0 L469.5,305.9 L474.0,305.5 L478.5,304.8 L483.0,303.9 L487.5,302.7 L492.0,301.2 L496.5,299.5 L501.0,297.5 L505.5,295.3 L510.0,292.9 L514.5,290.2 L519.0,287.3 L523.5,284.2 L528.0,280.8 L532.5,277.3 L537.0,273.6 L541.5,269.7 L546.0,265.6 L550.5,261.4 L555.0,257.0 L559.5,252.5 L564.0,247.9 L568.5,243.1 L573.0,238.3 L577.5,233.4 L582.0,228.4 L586.5,223.3 L591.0,218.2 L595.5,213.1 L600.0,208.0"/>
<path class="u1-g" d="M60.0,179.4 L64.5,179.1 L69.0,178.5 L73.5,177.5 L78.0,176.0 L82.5,173.6 L87.0,169.9 L91.5,164.5 L96.0,157.2 L100.5,148.0 L105.0,137.6 L109.5,127.2 L114.0,118.0 L118.5,110.6 L123.0,105.2 L127.5,101.6 L132.0,99.2 L136.5,97.6 L141.0,96.7 L145.5,96.1 L150.0,95.7 L154.5,95.5 L159.0,95.3 L163.5,95.3 L168.0,95.2 L172.5,95.2 L177.0,95.2 L181.5,95.1 L186.0,95.1 L190.5,95.1 L195.0,95.1 L199.5,95.1 L204.0,95.1 L208.5,95.1 L213.0,95.2 L217.5,95.2 L222.0,95.2 L226.5,95.3 L231.0,95.3 L235.5,95.5 L240.0,95.7 L244.5,96.1 L249.0,96.7 L253.5,97.6 L258.0,99.2 L262.5,101.6 L267.0,105.2 L271.5,110.6 L276.0,118.0 L280.5,127.2 L285.0,137.6 L289.5,148.0 L294.0,157.2 L298.5,164.5 L303.0,169.9 L307.5,173.6 L312.0,176.1 L316.5,177.6 L321.0,178.7 L325.5,179.4 L330.0,180.0 L334.5,180.6 L339.0,181.3 L343.5,182.4 L348.0,183.9 L352.5,186.4 L357.0,190.1 L361.5,195.5 L366.0,202.8 L370.5,212.0 L375.0,222.4 L379.5,232.8 L384.0,242.0 L388.5,249.4 L393.0,254.8 L397.5,258.4 L402.0,260.8 L406.5,262.4 L411.0,263.3 L415.5,263.9 L420.0,264.3 L424.5,264.5 L429.0,264.7 L433.5,264.7 L438.0,264.8 L442.5,264.8 L447.0,264.8 L451.5,264.9 L456.0,264.9 L460.5,264.9 L465.0,264.9 L469.5,264.9 L474.0,264.9 L478.5,264.9 L483.0,264.8 L487.5,264.8 L492.0,264.8 L496.5,264.7 L501.0,264.7 L505.5,264.5 L510.0,264.3 L514.5,263.9 L519.0,263.3 L523.5,262.4 L528.0,260.8 L532.5,258.4 L537.0,254.8 L541.5,249.4 L546.0,242.0 L550.5,232.8 L555.0,222.4 L559.5,212.0 L564.0,202.8 L568.5,195.5 L573.0,190.1 L577.5,186.4 L582.0,184.0 L586.5,182.5 L591.0,181.5 L595.5,180.9 L600.0,180.6"/>
<text class="u1-t" x="330" y="64">1. Adversary picks ε = 0.1</text>
<text class="u1-a" x="330" y="88">2. You pick N = 6 sigmoids</text>
<text class="u1-s" x="330" y="110">max error 0.072 &lt; 0.1</text>
</g>
<g class="u1-3">
<path class="u1-tube" d="M60.0,166.0 L64.5,160.9 L69.0,155.8 L73.5,150.7 L78.0,145.6 L82.5,140.6 L87.0,135.7 L91.5,130.9 L96.0,126.1 L100.5,121.5 L105.0,117.0 L109.5,112.6 L114.0,108.4 L118.5,104.3 L123.0,100.4 L127.5,96.7 L132.0,93.2 L136.5,89.8 L141.0,86.7 L145.5,83.8 L150.0,81.1 L154.5,78.7 L159.0,76.5 L163.5,74.5 L168.0,72.8 L172.5,71.3 L177.0,70.1 L181.5,69.2 L186.0,68.5 L190.5,68.1 L195.0,68.0 L199.5,68.1 L204.0,68.5 L208.5,69.2 L213.0,70.1 L217.5,71.3 L222.0,72.8 L226.5,74.5 L231.0,76.5 L235.5,78.7 L240.0,81.1 L244.5,83.8 L249.0,86.7 L253.5,89.8 L258.0,93.2 L262.5,96.7 L267.0,100.4 L271.5,104.3 L276.0,108.4 L280.5,112.6 L285.0,117.0 L289.5,121.5 L294.0,126.1 L298.5,130.9 L303.0,135.7 L307.5,140.6 L312.0,145.6 L316.5,150.7 L321.0,155.8 L325.5,160.9 L330.0,166.0 L334.5,171.1 L339.0,176.2 L343.5,181.3 L348.0,186.4 L352.5,191.4 L357.0,196.3 L361.5,201.1 L366.0,205.9 L370.5,210.5 L375.0,215.0 L379.5,219.4 L384.0,223.6 L388.5,227.7 L393.0,231.6 L397.5,235.3 L402.0,238.8 L406.5,242.2 L411.0,245.3 L415.5,248.2 L420.0,250.9 L424.5,253.3 L429.0,255.5 L433.5,257.5 L438.0,259.2 L442.5,260.7 L447.0,261.9 L451.5,262.8 L456.0,263.5 L460.5,263.9 L465.0,264.0 L469.5,263.9 L474.0,263.5 L478.5,262.8 L483.0,261.9 L487.5,260.7 L492.0,259.2 L496.5,257.5 L501.0,255.5 L505.5,253.3 L510.0,250.9 L514.5,248.2 L519.0,245.3 L523.5,242.2 L528.0,238.8 L532.5,235.3 L537.0,231.6 L541.5,227.7 L546.0,223.6 L550.5,219.4 L555.0,215.0 L559.5,210.5 L564.0,205.9 L568.5,201.1 L573.0,196.3 L577.5,191.4 L582.0,186.4 L586.5,181.3 L591.0,176.2 L595.5,171.1 L600.0,166.0"/>
<path class="u1-tube" d="M60.0,194.0 L64.5,188.9 L69.0,183.8 L73.5,178.7 L78.0,173.6 L82.5,168.6 L87.0,163.7 L91.5,158.9 L96.0,154.1 L100.5,149.5 L105.0,145.0 L109.5,140.6 L114.0,136.4 L118.5,132.3 L123.0,128.4 L127.5,124.7 L132.0,121.2 L136.5,117.8 L141.0,114.7 L145.5,111.8 L150.0,109.1 L154.5,106.7 L159.0,104.5 L163.5,102.5 L168.0,100.8 L172.5,99.3 L177.0,98.1 L181.5,97.2 L186.0,96.5 L190.5,96.1 L195.0,96.0 L199.5,96.1 L204.0,96.5 L208.5,97.2 L213.0,98.1 L217.5,99.3 L222.0,100.8 L226.5,102.5 L231.0,104.5 L235.5,106.7 L240.0,109.1 L244.5,111.8 L249.0,114.7 L253.5,117.8 L258.0,121.2 L262.5,124.7 L267.0,128.4 L271.5,132.3 L276.0,136.4 L280.5,140.6 L285.0,145.0 L289.5,149.5 L294.0,154.1 L298.5,158.9 L303.0,163.7 L307.5,168.6 L312.0,173.6 L316.5,178.7 L321.0,183.8 L325.5,188.9 L330.0,194.0 L334.5,199.1 L339.0,204.2 L343.5,209.3 L348.0,214.4 L352.5,219.4 L357.0,224.3 L361.5,229.1 L366.0,233.9 L370.5,238.5 L375.0,243.0 L379.5,247.4 L384.0,251.6 L388.5,255.7 L393.0,259.6 L397.5,263.3 L402.0,266.8 L406.5,270.2 L411.0,273.3 L415.5,276.2 L420.0,278.9 L424.5,281.3 L429.0,283.5 L433.5,285.5 L438.0,287.2 L442.5,288.7 L447.0,289.9 L451.5,290.8 L456.0,291.5 L460.5,291.9 L465.0,292.0 L469.5,291.9 L474.0,291.5 L478.5,290.8 L483.0,289.9 L487.5,288.7 L492.0,287.2 L496.5,285.5 L501.0,283.5 L505.5,281.3 L510.0,278.9 L514.5,276.2 L519.0,273.3 L523.5,270.2 L528.0,266.8 L532.5,263.3 L537.0,259.6 L541.5,255.7 L546.0,251.6 L550.5,247.4 L555.0,243.0 L559.5,238.5 L564.0,233.9 L568.5,229.1 L573.0,224.3 L577.5,219.4 L582.0,214.4 L586.5,209.3 L591.0,204.2 L595.5,199.1 L600.0,194.0"/>
<path class="u1-g" d="M60.0,179.7 L64.5,179.1 L69.0,177.7 L73.5,174.2 L78.0,166.8 L82.5,155.5 L87.0,144.2 L91.5,136.8 L96.0,133.3 L100.5,131.8 L105.0,131.1 L109.5,130.5 L114.0,129.3 L118.5,126.7 L123.0,121.4 L127.5,113.1 L132.0,104.8 L136.5,99.4 L141.0,96.8 L145.5,95.7 L150.0,95.3 L154.5,95.0 L159.0,94.5 L163.5,93.6 L168.0,91.6 L172.5,88.6 L177.0,85.5 L181.5,83.6 L186.0,82.6 L190.5,82.3 L195.0,82.2 L199.5,82.3 L204.0,82.6 L208.5,83.6 L213.0,85.5 L217.5,88.6 L222.0,91.6 L226.5,93.6 L231.0,94.5 L235.5,95.0 L240.0,95.3 L244.5,95.7 L249.0,96.8 L253.5,99.4 L258.0,104.8 L262.5,113.1 L267.0,121.4 L271.5,126.7 L276.0,129.3 L280.5,130.5 L285.0,131.1 L289.5,131.8 L294.0,133.3 L298.5,136.8 L303.0,144.2 L307.5,155.5 L312.0,166.8 L316.5,174.2 L321.0,177.7 L325.5,179.2 L330.0,180.0 L334.5,180.8 L339.0,182.3 L343.5,185.8 L348.0,193.2 L352.5,204.5 L357.0,215.8 L361.5,223.2 L366.0,226.7 L370.5,228.2 L375.0,228.9 L379.5,229.5 L384.0,230.7 L388.5,233.3 L393.0,238.6 L397.5,246.9 L402.0,255.2 L406.5,260.6 L411.0,263.2 L415.5,264.3 L420.0,264.7 L424.5,265.0 L429.0,265.5 L433.5,266.4 L438.0,268.4 L442.5,271.4 L447.0,274.5 L451.5,276.4 L456.0,277.4 L460.5,277.7 L465.0,277.8 L469.5,277.7 L474.0,277.4 L478.5,276.4 L483.0,274.5 L487.5,271.4 L492.0,268.4 L496.5,266.4 L501.0,265.5 L505.5,265.0 L510.0,264.7 L514.5,264.3 L519.0,263.2 L523.5,260.6 L528.0,255.2 L532.5,246.9 L537.0,238.6 L541.5,233.3 L546.0,230.7 L550.5,229.5 L555.0,228.9 L559.5,228.2 L564.0,226.7 L568.5,223.2 L573.0,215.8 L577.5,204.5 L582.0,193.2 L586.5,185.8 L591.0,182.3 L595.5,180.9 L600.0,180.3"/>
<text class="u1-t" x="330" y="64">1. Adversary picks ε = 0.05</text>
<text class="u1-a" x="330" y="88">2. You pick N = 12 sigmoids</text>
<text class="u1-s" x="330" y="110">max error 0.034 &lt; 0.05</text>
</g>
<path class="u1-f" d="M60.0,180.0 L64.5,174.9 L69.0,169.8 L73.5,164.7 L78.0,159.6 L82.5,154.6 L87.0,149.7 L91.5,144.9 L96.0,140.1 L100.5,135.5 L105.0,131.0 L109.5,126.6 L114.0,122.4 L118.5,118.3 L123.0,114.4 L127.5,110.7 L132.0,107.2 L136.5,103.8 L141.0,100.7 L145.5,97.8 L150.0,95.1 L154.5,92.7 L159.0,90.5 L163.5,88.5 L168.0,86.8 L172.5,85.3 L177.0,84.1 L181.5,83.2 L186.0,82.5 L190.5,82.1 L195.0,82.0 L199.5,82.1 L204.0,82.5 L208.5,83.2 L213.0,84.1 L217.5,85.3 L222.0,86.8 L226.5,88.5 L231.0,90.5 L235.5,92.7 L240.0,95.1 L244.5,97.8 L249.0,100.7 L253.5,103.8 L258.0,107.2 L262.5,110.7 L267.0,114.4 L271.5,118.3 L276.0,122.4 L280.5,126.6 L285.0,131.0 L289.5,135.5 L294.0,140.1 L298.5,144.9 L303.0,149.7 L307.5,154.6 L312.0,159.6 L316.5,164.7 L321.0,169.8 L325.5,174.9 L330.0,180.0 L334.5,185.1 L339.0,190.2 L343.5,195.3 L348.0,200.4 L352.5,205.4 L357.0,210.3 L361.5,215.1 L366.0,219.9 L370.5,224.5 L375.0,229.0 L379.5,233.4 L384.0,237.6 L388.5,241.7 L393.0,245.6 L397.5,249.3 L402.0,252.8 L406.5,256.2 L411.0,259.3 L415.5,262.2 L420.0,264.9 L424.5,267.3 L429.0,269.5 L433.5,271.5 L438.0,273.2 L442.5,274.7 L447.0,275.9 L451.5,276.8 L456.0,277.5 L460.5,277.9 L465.0,278.0 L469.5,277.9 L474.0,277.5 L478.5,276.8 L483.0,275.9 L487.5,274.7 L492.0,273.2 L496.5,271.5 L501.0,269.5 L505.5,267.3 L510.0,264.9 L514.5,262.2 L519.0,259.3 L523.5,256.2 L528.0,252.8 L532.5,249.3 L537.0,245.6 L541.5,241.7 L546.0,237.6 L550.5,233.4 L555.0,229.0 L559.5,224.5 L564.0,219.9 L568.5,215.1 L573.0,210.3 L577.5,205.4 L582.0,200.4 L586.5,195.3 L591.0,190.2 L595.5,185.1 L600.0,180.0"/>
<text class="u1-s" x="60" y="370">dashed: f ± ε, dark: target f, accent: G built after ε is known</text>
</svg>
<figcaption>The order of the quantifiers, played as a game. The adversary names the tolerance ε first; only then do you choose how many sigmoids N to use. Smaller ε forces larger N (3, then 6, then 12), and the theorem promises you can always answer, never that one fixed N answers every ε.</figcaption>
</figure>

The animation above plays this game for the theorem we are about to decode. The adversary names $\varepsilon = {0.2}$ and we answer with 3 sigmoids; the adversary tightens to ${0.1}$ and we need 6; at ${0.05}$ we need 12. The numbers come from the staircase construction in the next section. The theorem promises an answer to every challenge, never one answer to all of them.

## Idioms papers use, and what they hide

Papers compress routine reasoning into stock phrases. Each is a promise you may check.

| Phrase | What it means | What to check |
| --- | --- | --- |
| Without loss of generality (w.l.o.g.) | One case covers the others by symmetry or relabelling | The symmetry: "assume $a \le b$" needs a claim symmetric in $a, b$ |
| It suffices to show $X$ | $X$ implies the goal, so proving $X$ is enough | That $X$ really implies the goal and is not false |
| By a standard argument | A well-known technique applies; details omitted | Which one: often induction or a union bound |
| With high probability | Holds with probability at least $1-\delta$ for a small $\delta$ | How the bound depends on $\delta$; a $\log(1/\delta)$ term often hides |
| For sufficiently large $n$ | There exists a threshold $n_0$ beyond which it holds | How large $n_0$ is; often unstated |
| If and only if (iff) | Both $P \Rightarrow Q$ and $Q \Rightarrow P$ | That both directions are proved |

"It suffices to show" is where a proof most often quietly proves something easier than the claim, so read that reduction slowly.

## Reading Cybenko's theorem

### Paper sentence, decoded

**Source:** G. Cybenko, "Approximation by superpositions of a sigmoidal function," *Mathematics of Control, Signals, and Systems* 2(4), 303–314, 1989. [doi:10.1007/BF02551274](https://doi.org/10.1007/BF02551274). Theorem 2, which we paraphrase from the paper's wording (it has no arXiv version):

$$
\forall f \in C(I_n)\ \ \forall \varepsilon \gt 0\ \ \exists N, \{\alpha_j, y_j, \theta_j\}_{j=1}^{N}\ \ \forall x \in I_n:\quad \Bigl\lvert \sum_{j=1}^{N} \alpha_j\, \sigma(y_j^\top x + \theta_j) - f(x) \Bigr\rvert \lt \varepsilon .
$$

The paper states it in words; the display is the same sentence with the quantifiers made explicit.

| Symbol | What it is | Built where |
| --- | --- | --- |
| $I_n$ | the unit cube $[0,1]^n$, the input domain | [Sets, functions and maps](/blog/machine-learning/mathematics/sets-functions-and-maps-the-grammar-of-ml-math) |
| $C(I_n)$ | all continuous functions from $I_n$ to the reals | direct-proof section (continuity) |
| $\sigma$ | a continuous sigmoidal function: $\sigma(t) \to 1$ as $t \to \infty$ and $\sigma(t) \to 0$ as $t \to -\infty$ | this section |
| $N$ | number of hidden units | quantifier section |
| $y_j, \theta_j$ | weight vector and bias of hidden unit $j$ | [How to read the math](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper) |
| $\alpha_j$ | output weight of hidden unit $j$ | this section |
| $\forall \varepsilon\ \exists N$ | $N$ may depend on $f$ and on $\varepsilon$ | adversary game |
| $\forall x$ after $\exists$ | one network must work on the whole cube at once | quantifier section |

In words: for any continuous target and any tolerance you name, some one-hidden-layer sigmoid network stays within that tolerance everywhere on the cube.

![Target f equal to 0.5 plus 0.35 sine of 2 pi x on the unit interval, a dashed epsilon tube of width 0.1 on each side, and a six-sigmoid approximant G that stays inside, with maximum error 0.072 near x equals 0.55](/imgs/blogs/proof-techniques-for-ml-researchers-6.webp)

The figure shows one move of the game in one dimension. The target is $f(x) = {0.5} + {0.35}\sin(2\pi x)$ and the tolerance is ${0.1}$. Our answer is a staircase construction: split $[0,1]$ into $N$ equal pieces with grid points $t_j = j/N$, put one steep sigmoid at the middle $s_j$ of each piece, give it height $\alpha_j = f(t_j) - f(t_{j-1})$, and add the constant $f(0) = {0.5}$, which is itself of the allowed form because $1 \cdot \sigma(0 \cdot x + 0) = {0.5}$ for the logistic sigmoid. With $N = 6$ and slope 60 the maximum error is ${0.072}$, inside the tube.

The three misreadings from the opening are each a quantifier mistake:

1. **"One network fits everything."** The order is $\forall f\ \forall \varepsilon\ \exists N$, so $N$ and the weights are chosen after $f$ and $\varepsilon$. Swapping to $\exists N\ \forall f$ claims one network fits every function, which is false.
2. **"Networks are small."** The theorem says nothing about the size of $N$. In our staircase, halving $\varepsilon$ doubled $N$. Existence is not efficiency.
3. **"Networks can learn anything."** The statement is "there exist weights". It says nothing about whether gradient descent finds them or whether they generalize. Approximation, optimization and generalization are separate theorems; Cybenko proved the first.

Cybenko's proof uses the Hahn–Banach and Riesz representation theorems, a later rung; we state the theorem without proof.

## Proof, read line by line

We decode one short, complete proof in the format every later post uses. It is the standard composition argument; Szegedy et al. use it in Section 4.3 of [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199) (2013) to bound how much a network can amplify a small input change.

**Statement.** Let $f_k : \mathbb{R}^{d_{k-1}} \to \mathbb{R}^{d_k}$ for $k = 1, \dots, K$, and let $F_K = f_K \circ \cdots \circ f_1$.

- (A1) Each layer is Lipschitz: $\lVert f_k(u) - f_k(v) \rVert \le L_k \lVert u - v \rVert$ for all $u, v$, with $L_k \ge 0$.
- (A2) The norm that measures the output of $f_k$ is the same norm that measures the input of $f_{k+1}$.

Then $\lVert F_K(x) - F_K(x') \rVert \le \bigl(\prod_{k=1}^{K} L_k\bigr) \lVert x - x' \rVert$ for all $x, x'$.

**The idea in one sentence:** peel off the last layer, use its Lipschitz constant once, and hand the rest to the induction hypothesis.

**Steps.**

1. Define $F_0 = \mathrm{id}$ and $F_k = f_k \circ F_{k-1}$. Let $P(k)$ be the claim $\lVert F_k(x) - F_k(x') \rVert \le \bigl(\prod_{i=1}^{k} L_i\bigr)\lVert x - x'\rVert$. *Set up the induction over layers.*
2. $P(0)$ holds: $F_0$ is the identity and the empty product is 1, so both sides equal $\lVert x - x'\rVert$. *Base case.*
3. Assume $P(k-1)$ for some $k \ge 1$. *Inductive hypothesis.*
4. $\lVert F_k(x) - F_k(x')\rVert = \lVert f_k(u) - f_k(v)\rVert$ with $u = F_{k-1}(x)$, $v = F_{k-1}(x')$. *Definition of $F_k$; name the intermediate activations.*
5. $\le L_k \lVert u - v \rVert$. *By (A1) for layer $k$, which holds for all inputs, so in particular for these.*
6. $\le L_k \bigl(\prod_{i=1}^{k-1} L_i\bigr) \lVert x - x' \rVert$. *By $P(k-1)$, measured in the same norm thanks to (A2); multiplying an inequality by $L_k \ge 0$ keeps its direction.*
7. The right side equals $\bigl(\prod_{i=1}^{k} L_i\bigr)\lVert x - x'\rVert$, which is $P(k)$. By induction, $P(K)$ holds. *Conclude.*

Corollary: a linear layer $u \mapsto W u$ is $\lVert W \rVert_2$-Lipschitz in the Euclidean norm, where $\lVert W\rVert_2$ is the largest singular value, and ReLU is 1-Lipschitz because $\lvert \operatorname{ReLU}(a) - \operatorname{ReLU}(b)\rvert \le \lvert a - b\rvert$ in each coordinate. So a ReLU network is at most $\prod_k \lVert W_k\rVert_2$-Lipschitz.

**Where each assumption was used.**

| Assumption | Used in step |
| --- | --- |
| (A1) layer $k$ is $L_k$-Lipschitz, $L_k \ge 0$ | 5, and the sign in 6 |
| (A2) matching norms between layers | 6 |

**What breaks without (A2).** Let both layers be the identity on $\mathbb{R}^4$. Layer 1 is 1-Lipschitz from the max-norm $\lVert\cdot\rVert_\infty$ to itself; layer 2 is 1-Lipschitz from the Euclidean norm to itself. Chaining them blindly claims $\lVert x - x'\rVert_2 \le 1 \cdot \lVert x - x'\rVert_\infty$. Take $x - x' = (1, 1, 1, 1)$: the left side is $\sqrt{4} = 2$ and the right side is 1. The chain broke at the seam between two norms, and the missing factor is $\sqrt{d}$.

**Proof moves used:** `induction (layers / steps)`.

## Check it numerically

The snippet compares the product-of-norms bound for a random three-layer ReLU network with the largest amplification found over 20,000 random nearby input pairs, then re-runs the tent counterexample.

```python
import numpy as np
rng = np.random.default_rng(0)
relu = lambda z: np.maximum(z, 0)

# A 3-layer ReLU network f = W3 . relu . W2 . relu . W1
Ws = [rng.normal(size=(64, 16)), rng.normal(size=(64, 64)) / 8, rng.normal(size=(1, 64))]
def f(x):
    for W in Ws[:-1]:
        x = relu(W @ x)
    return Ws[-1] @ x

# Induction bound: product of spectral norms (ReLU is 1-Lipschitz)
bound = np.prod([np.linalg.norm(W, 2) for W in Ws])

# Largest observed ratio |f(x) - f(x')| / ||x - x'|| over random nearby pairs
ratios = []
for _ in range(20000):
    x = rng.normal(size=16)
    d = rng.normal(size=16) * 1e-3
    ratios.append(abs((f(x + d) - f(x)).item()) / np.linalg.norm(d))
print(f"product-of-norms bound : {bound:8.2f}")
print(f"largest observed ratio : {max(ratios):8.2f}")
print(f"bound holds            : {max(ratios) <= bound}")

# Counterexample: the tent network relu(1 - relu(x) - relu(-x))
tent = lambda x: relu(1 - relu(x) - relu(-x))
lhs, rhs = tent(0.0), 0.5 * tent(-1.0) + 0.5 * tent(1.0)
print(f"tent(0) = {lhs}, chord = {rhs}, convex? {lhs <= rhs}")
```

```console
product-of-norms bound :   190.31
largest observed ratio :    21.88
bound holds            : True
tent(0) = 1.0, chord = 0.0, convex? False
```

The bound holds, as the proof guarantees, but it is about nine times larger than anything observed. That gap is expected: the product assumes each layer stretches along its worst direction and that those directions line up, which random inputs rarely hit. A proof gives a guarantee, not a tight estimate.

## Where this shows up in practice

**Spectral normalization for GAN discriminators.** Miyato et al., [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957) (ICLR 2018), divide each weight matrix by an estimate of its largest singular value, so every linear layer is 1-Lipschitz. With 1-Lipschitz activations, the induction-over-layers proof caps the whole discriminator's Lipschitz constant at 1. PyTorch ships it as `torch.nn.utils.spectral_norm`; our [GAN post](/blog/machine-learning/image-generation/generative-adversarial-networks-and-why-they-lost) covers why it mattered for training.

**Bounding adversarial sensitivity.** Szegedy et al., [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199) (2013), the paper that introduced adversarial examples, use the composition bound to cap how much each layer can amplify a perturbation, and note that a large bound does not by itself prove an adversarial example exists: the guarantee-versus-estimate gap our check measured.

**Parseval networks.** Cisse et al., [Parseval Networks: Improving Robustness to Adversarial Examples](https://arxiv.org/abs/1704.08847) (2017), keep weight matrices close to orthonormal so each layer's Lipschitz constant stays at or below 1. The robustness argument is the same induction: control every factor and the product is controlled.

## What to remember

> - A theorem is a contract: read the assumptions first and find where each is used.
> - The claim's shape picks the technique: direct, contrapositive, contradiction, induction, construction, counterexample.
> - Quantifier order is meaning: an $\exists$ after a $\forall$ may depend on it, so Cybenko's $N$ depends on $f$ and $\varepsilon$.
> - Universal approximation is about existence only, not size, trainability or generalization.
> - Lipschitz bounds multiply across layers by induction; they are guarantees, often loose.

## Where next

If any notation felt unfamiliar, start with [how to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper). The inequalities behind step 5 get the same line-by-line treatment in [the inequality toolkit every appendix uses](/blog/machine-learning/mathematics/the-inequality-toolkit-every-appendix-uses). The products and sums in this post are the subject of [sums, products, einsum and log-sum-exp](/blog/machine-learning/mathematics/sums-products-einsum-and-log-sum-exp), and "for sufficiently large $n$" is unpacked in [limits, big-O and asymptotics in ML papers](/blog/machine-learning/mathematics/limits-big-o-and-asymptotics-in-ml-papers).

Later in the series, the rigor track gives precise definitions of continuity, compactness and convergence that appendix proofs invoke by name, and a post on concentration inequalities explains where the $\delta$ in "with high probability" comes from.

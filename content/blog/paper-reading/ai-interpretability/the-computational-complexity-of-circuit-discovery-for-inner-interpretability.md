---
title: "The Computational Complexity of Circuit Discovery for Inner Interpretability"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "What it means that finding the smallest faithful circuit in a neural network is NP-hard, why global faithfulness is harder still, and which relaxed circuit queries you can actually compute fast, built up from toy circuits instead of proofs."
tags: ["paper-reading", "circuit-discovery", "computational-complexity", "mechanistic-interpretability", "inner-interpretability", "parameterized-complexity", "quasi-minimal-circuits", "iclr-2025-spotlight"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 19
paper:
  title: "The Computational Complexity of Circuit Discovery for Inner Interpretability"
  authors: "Federico Adolfi, Martina G. Vilas, Todd Wareham"
  venue: "ICLR 2025 (spotlight)"
  url: "https://arxiv.org/abs/2410.08025"
---

> [!tldr]
> - **Claim.** Most circuit questions we ask of a network (which neurons suffice, which are necessary, which ablation breaks it, which patch steers it) are intractable in the worst case, even for plain MLPs.
> - **Why it matters.** It explains why circuit discovery heuristics give discrepant, unfaithful or non-minimal answers: unless P = NP, they cannot be solving the problem we think they solve.
> - **Most surprising result.** A circuit faithful on *all* inputs is harder in kind, not degree: $\Sigma_2^p$-complete, one level above the NP-complete test-set version.
> - **The way out.** A "quasi-minimal" circuit (sufficient, with one known load-bearing neuron) takes polynomial time via binary search. That is roughly what greedy methods like ACDC already find.
> - **Where it falls short.** All results are worst case, on MLPs with neuron-level circuits. Trained networks might have friendly structure; the paper cannot rule that in or out.

Circuit discovery rests on a hopeful picture: somewhere inside a large network, a small subnetwork does the job, and a good search will find it. Methods like ACDC and attribution patching do produce circuits. But those circuits disagree with each other, fail off-distribution, or keep neurons that do nothing.

Adolfi, Vilas and Wareham ask the question that comes before any heuristic: how hard are these search problems in principle? This post walks through their answers the way an engineer needs them: what problem, what hardness class, and what that means for your tool. Toy circuits do the explaining; proofs stay in the appendix.

## A five-minute complexity toolkit

You need six ideas to read the paper's results table. Each one answers the same question: how does the cost of solving the problem grow as the network grows?

| Class | Plain meaning | What it predicts for a circuit tool |
| --- | --- | --- |
| PTIME | Solvable in time polynomial in network size | Exact answers at scale are realistic |
| NP-complete | Guessed answers are easy to check; finding one is believed to need exponential time | No exact fast algorithm, but SAT and ILP solvers are a legitimate route |
| $\Sigma_2^p$-complete | "Exists C such that for all x": checking a guess is itself hard | Not expected to reduce to SAT; harder than NP |
| FPT | Exponential only in a parameter you can keep small | Fast if that parameter really is small in practice |
| W[1]-hard | Not FPT for the chosen parameters | Keeping those parameters small does not help |
| #P-complete | Counting solutions is hard | You cannot cheaply tell how many good circuits exist |

The FPT row is the paper's main tool for asking "which knob causes the hardness". A parameterized problem is fixed-parameter tractable relative to a parameter set $P$ if some algorithm solves every instance of size $n$ in time

$$
T(n, P) = f(P)\cdot n^{c},
$$

where $f$ is any computable function (it may be exponential) and $c$ is a constant that does not depend on $P$. The exponential blowup is quarantined inside $f(P)$. If depth were such a parameter, a depth-4 network would be cheap no matter how wide it grew. W[1]-hardness says no such quarantine exists, assuming the standard conjecture FPT $\neq$ W[1].

Two more terms. **Inapproximable** means none of the paper's five schemes works in polynomial time: additive (within a constant $c$ of optimal), multiplicative (a PTAS), and three probabilistic ones (rarely wrong, wrong with low probability, or usually fast). A **reduction** is a fast translation from a known-hard problem into yours, so a fast solver for yours would solve that one too.

## The setting: MLPs, circuits, and a query template

The model is a multi-layer perceptron $M$ with ReLU hidden units and a binary step on the output, following Barceló et al. (2020). Each layer computes $h_i = f_i(h_{i-1}W_i + b_i)$, where $h_0 = x$ is the input vector, $W_i$ is the weight matrix of layer $i$, $b_i$ its bias vector, and $f_i$ is ReLU except for the final step. A **circuit** $C$ is a subset of neurons that induces a subgraph of $M$; its size $|C|$ is a neuron count.

Every query varies along three axes, which the paper generates from one template (its Table 2):

- **Coverage.** *Local*: the circuit must match the model on given inputs. *Global*: on every possible input.
- **Size.** *Bounded*: at most $u$ neurons. *Unbounded*: any size. *Optimal*: the minimum size.
- **Minimality.** A circuit is *minimal* if no proper subset has the same property. Minimal is not minimum: a minimal circuit can be far bigger than the smallest.

![Figure 1 from Adolfi, Vilas and Wareham (2025): how sufficient, necessary and quasi-minimal circuits relate.](/imgs/blogs/the-computational-complexity-of-circuit-discovery-for-inner-interpretability-fig1.webp)

Figure 1 is the map for the rest of the post. The outer region is the network, which trivially reproduces itself. Inside it sit many minimal sufficient circuits, possibly overlapping. Their shared core is the necessary circuit. The quasi-minimal circuit is the pragmatic newcomer: sufficient, with one neuron known to matter, and possibly carrying passengers.

Why care about these particular objects? Table 1 of the paper ties each circuit type to an *affordance*, a thing it lets you describe, predict or control.

![Table 1 from Adolfi, Vilas and Wareham (2025): circuit types and their affordances.](/imgs/blogs/the-computational-complexity-of-circuit-discovery-for-inner-interpretability-fig2.webp)

Read the table as a shopping list. A minimal sufficient circuit gives a short description plus a control guarantee: ablating any of its neurons breaks the behavior. A necessary circuit names the subcomputation every implementation shares. The paper's question is what each item costs.

## Result 1: finding a small sufficient circuit is NP-complete

### The problem

The headline query is Bounded Local Sufficient Circuit (BLSC): given a network $M$, an input $x$ and a budget $u \le |M|$, find a circuit of at most $u$ neurons that, running alone, gives the same output as the full network. This is the formal version of "zero-ablate everything that does not matter and keep the rest".

$$
\text{BLSC:}\quad \exists\, C \subseteq M \ \text{ such that } \ |C| \le u \ \text{ and } \ C(x) = M(x).
$$

Here $C(x)$ is the output when every neuron outside $C$ is removed, which is what zero-ablation does.

### The intuition

Think of packing a toolbox under a weight limit. Checking a proposed toolbox is easy: try the job once. Finding the lightest one that works means reasoning about tools that only work in combination. A network with $n$ neurons has ${2^n}$ candidate circuits; the question is whether structure lets you avoid looking at most of them. For BLSC, in the worst case, it does not.

### The mechanism: a reduction from CLIQUE

The proof translates the classic CLIQUE problem (does a graph contain $k$ vertices that are all pairwise connected?) into a circuit question. The construction is small enough to run by hand, and it is the best way to see *why* circuit search is hard.

![Figure 2 from the appendix of Adolfi, Vilas and Wareham (2025): the reduction from CLIQUE to sufficient circuit.](/imgs/blogs/the-computational-complexity-of-circuit-discovery-for-inner-interpretability-fig4.webp)

Given a graph $G = (V, E)$ and a target $k$, build a four-layer MLP with $|V| + |E| + 2$ neurons:

1. One input neuron, fed the input ${1}$.
2. One **vertex neuron** per vertex, bias ${0}$, weight ${1}$ from the input.
3. One **edge neuron** per edge, bias $-1$, weight ${1}$ from each of its two endpoint vertices. It outputs $\mathrm{ReLU}(1 + 1 - 1) = 1$ when both endpoints are present and $\mathrm{ReLU}(1 - 1) = 0$ when only one is.
4. One output neuron with weight ${1}$ from every edge neuron and bias $-\big(\tfrac{k(k-1)}{2} - 1\big)$, followed by a step that outputs ${1}$ when its argument is positive.

The output therefore computes

$$
\text{out} = \mathrm{step}\Big(\sum_{e \in C} a_e \;-\; \big(\tfrac{k(k-1)}{2} - 1\big)\Big),
$$

where $a_e \in \{0, 1\}$ is the activation of edge neuron $e$ inside the circuit. It fires exactly when at least $k(k-1)/2$ edge neurons are alive. Set the neuron budget to $u = k(k-1)/2 + k + 2$.

### Worked toy circuit

Take four vertices A, B, C, D with edges A-B, A-C, B-C and C-D, and ask for a triangle, so $k = 3$. The output bias is $-(3 - 1) = -2$ and the budget is ${3 + 3 + 2 = 8}$ neurons.

![Toy version of the CLIQUE reduction: the only 8-neuron circuit that keeps the output at 1 is the triangle A, B, C plus its edge neurons.](/imgs/blogs/the-computational-complexity-of-circuit-discovery-for-inner-interpretability-2.webp)

The full network has all four edge neurons on, so the output is $\mathrm{step}(4 - 2) = 1$. Now look for an 8-neuron circuit with output 1:

- The input and output neurons must stay, leaving 6 slots for vertex and edge neurons.
- The output needs at least 3 live edge neurons, since $\mathrm{step}(2 - 2) = 0$.
- Each live edge neuron needs both endpoints kept, and 3 edges need at least 3 vertices. With exactly 3 vertices and 3 edges filling 6 slots, those 3 vertices must be pairwise connected.

The circuit {in, A, B, C, AB, AC, BC, out} gives $\mathrm{step}(3 - 2) = 1$. It exists if and only if the graph has a triangle. So a fast sufficient-circuit finder would be a fast clique finder, and CLIQUE is NP-complete. Checking a proposed circuit takes one forward pass, which puts BLSC in NP; together, NP-complete.

### What it means, and what it does not

The toy network has depth 4, one input, one output, and unit weights. None of that saved it. That is the intuition behind the parameterized result: sufficient circuit is W[1]-hard relative to model depth, input and output counts, maximum weight and bias, plus every feature of the requested circuit. In the paper's words, hardness "is not mitigated by keeping models shallow."

Approximation does not rescue it either. The hard variants are inapproximable under all five schemes, so no polynomial algorithm can promise a circuit within a fixed number of neurons of the minimum, or within a constant factor.

The good news hides in NP-completeness itself. NP-complete problems translate efficiently into Boolean satisfiability, so mature SAT solvers become a defensible tool. The next result shows this route is not open to every variant.

## Result 2: global faithfulness is a different class of hard

### The problem

A local circuit matches the model on the inputs you tested. What you usually want is a circuit that *is* the mechanism, matching the model everywhere. That is the global variant:

$$
\text{Global SC:}\quad \exists\, C \subseteq M \ \ \forall x:\ \ C(x) = M(x), \quad |C| \le u.
$$

The only change is the universal quantifier $\forall x$ over all inputs. The paper proves this variant is $\Sigma_2^p$-complete, one level up the polynomial hierarchy.

![Local versus global sufficient circuit: one quantifier turns a forward-pass check into a hard search.](/imgs/blogs/the-computational-complexity-of-circuit-discovery-for-inner-interpretability-4.webp)

### The intuition

For the local version, a friend can hand you a candidate circuit and you verify it with one forward pass. For the global version, verifying the candidate requires showing that *no* input makes it disagree with the model. That verification is itself a coNP-hard search over inputs. You are now nesting one hard problem inside another, which is exactly what the second level of the hierarchy captures.

### Worked toy circuit

Take a network with two binary inputs computing $x_1 \text{ AND } x_2$ through a few hidden ReLUs, with an output bias that is negative. Test it only on the inputs $(0,0)$ and $(0,1)$, where the output is ${0}$. A circuit that drops every hidden neuron and keeps only the input and output neurons also outputs ${0}$, because the negative bias alone decides the step. It matches on both test points, so it is locally sufficient, and it is globally wrong, because it fails on $(1,1)$.

Two inputs give four cases to enumerate; 512 inputs give ${2^{512}}$, and the global check becomes a search. (This toy example is mine, not the paper's.)

### What it means

This is the paper's formal account of **interpretability illusions**: circuits that pass local tests but fail to generalize. Two consequences follow. A circuit validated on an eval set carries only a *local* certificate. And the SAT route from Result 1 does not extend; you would need solvers for quantified formulas.

## Result 3: you cannot even count the good circuits

If good circuits were abundant, heuristic search would stumble on one often. So can you at least count them? Counting sufficient (or minimal sufficient) circuits is #P-complete locally, #P-hard globally, and #W[1]-hard even with depth as a parameter. You cannot cheaply tell whether the search space is rich or sparse; "circuits are plentiful" stays an empirical bet.

## Result 4: quasi-minimal circuits are cheap

### The problem

Every exact sufficient-circuit query so far is hard. What survives if you give up minimality? The authors' answer is **quasi-minimality**: a circuit that is sufficient and contains at least one known breaking point, a neuron whose removal breaks the behavior.

$$
\text{UQLSC:}\quad \text{find } C \subseteq M,\ v \in C \ \text{ with } \ C(x) = M(x) \ \text{ and } \ [C \setminus \{v\}](x) \neq M(x).
$$

Here $[C \setminus \{v\}](x)$ is the circuit's output after also removing neuron $v$. In a minimal circuit, *every* neuron is a breaking point. In a quasi-minimal one, you are only guaranteed one.

### The intuition

Picture removing furniture one piece at a time until the floor collapses. You do not know which pieces were structural, but the last one removed certainly was. That is the breaking point, and finding it needs no combinatorial search.

### The mechanism: binary search over removals

The paper's algorithm (Theorem 27 in the appendix) is short:

1. Order the neurons by any heuristic, for example gradient magnitude.
2. Form a sequence $M_0, M_1, \dots, M_n$, where $M_i$ removes the first $i$ neurons. $M_0$ is the full network and matches (label it YES); $M_n$ removes everything and fails (label it NO).
3. Evaluate the midpoint between the current YES and NO. If it matches, it becomes the new YES; otherwise it becomes the new NO.
4. Stop at an adjacent YES and NO pair. Return the YES circuit and the one neuron that differs.

![The quasi-minimal binary search: halve the removal sequence until an adjacent YES and NO pair exposes the breaking point.](/imgs/blogs/the-computational-complexity-of-circuit-discovery-for-inner-interpretability-3.webp)

An adjacent pair always exists because the sequence starts at YES and ends at NO, so somewhere it must flip. Notice that the sequence does not need to be monotone: binary search still lands on *a* flip, even if the behavior flickers. The paper puts the cost at roughly $O(n \log n)$.

### Worked toy circuit

Take an 8-neuron network and a removal order from your heuristic. $M_0$ is YES and $M_8$ is NO.

| Probe | Network | Neurons kept | Output matches? | New range |
| --- | --- | --- | --- | --- |
| 1 | $M_4$ | 4 | yes | $M_4$ to $M_8$ |
| 2 | $M_6$ | 2 | no | $M_4$ to $M_6$ |
| 3 | $M_5$ | 3 | yes | $M_5$ to $M_6$ |

After three probes, about $\log_2 8$, we return the 3 neurons kept in $M_5$ as the circuit and the sixth neuron in the removal order as the breaking point.

### What it means, and when it fails

The same trick extends to patching (Result 5), giving a second polynomial-time query. And the authors make a pointed observation about ACDC (Conmy et al., 2023), which removes one component at a time when the effect on behavior is small. Because it runs in polynomial time, it is unlikely to solve minimal sufficient circuit. What it is well-equipped to find is a quasi-minimal circuit.

That reframing explains three observed failures: circuits that are not globally faithful (the search is local), not minimal (passengers ride along), and highly variable across tasks (the result depends on the removal order). The guarantee itself is the limit: one neuron matters, the rest are unknown.

## Result 5: ablation, clamping and patching

Maybe sufficiency is too strict. The paper tries three intervention-style queries that ask less:

- **Circuit ablation (CA).** Find at most $u$ neurons whose removal from the *whole network* changes its output. A possibly non-minimal breaking set, so weaker than a minimal sufficient circuit.
- **Circuit clamping (CC).** Find at most $u$ neurons that, when fixed to a value $r$, change the output.
- **Circuit patching (CP).** Find at most $k$ neurons such that patching in their activations from input $y$, while every other neuron keeps its activations from $x$, makes the network output $M(y)$ for every $x$ in a set $X$. This formalizes activation and path patching.

All three are still NP-hard, local and global, and inapproximable under all five schemes. CA and CC are W[1]-hard, and CP is W[2]-hard, relative to parameter sets that include model depth and the circuit size. The local versions are NP-complete, so the SAT route stays open for them.

Patching gets the same rescue as sufficiency. **Quasi-minimal local circuit patching** asks for a patched set that produces the target behavior on every $x \in X$, plus one neuron whose unpatching breaks it for some $x$. It is in PTIME, by the same flavor of algorithm.

## Result 6: necessary circuits, robustness and gnostic neurons

### Necessary circuits are hard

A necessary circuit is a set $S$ of at most $k$ neurons that intersects *every* sufficient circuit. Intervening on it must have an effect even when the network is full of redundant backups. It is also NP-hard in both versions (and in $\Sigma_2^p$), W[1]-hard relative to depth, input and output counts, weights and circuit size, and inapproximable. The paper's verdict: tractable versions "are unlikely unless substantial restrictions or relaxations are introduced."

### Robustness is the first FPT result

Circuit robustness asks whether a behavior survives every small ablation within a chosen set $H$ of neurons:

$$
\text{BLCR:}\quad \forall\, H' \subseteq H \ \text{ with } \ |H'| \le k:\ \ M(x) = [M \setminus H'](x).
$$

Here $H$ is the neuron set you suspect holds redundant copies, and $k$ is how many you knock out at once. Local robustness is coNP-complete, the mirror image of NP: a NO answer has a short certificate (the breaking set), a YES answer does not. Parameterizing by model features does not help (coW[1]-hard).

But parameterizing by $|H|$ does. If the set you probe is small, you can just try its subsets. Table 4 reports local robustness as FPT in $|H|$, and both local and global as FPT in $|H|$ plus the number of input units. Concretely (my arithmetic, not the paper's): with $|H| = 10$ and $k = 3$, the subsets to test number ${1 + 10 + 45 + 120 = 176}$, each one forward pass, no matter how large the rest of the network is.

### Gnostic neurons are easy

Gnostic or "concept" neurons, which fire above threshold $t$ for every input in a set $X$ and below it for every input in $Y$, are in PTIME. You run the inputs, record activations, and filter. The hardness elsewhere comes from choosing *subsets of interacting neurons*; a per-neuron query never does.

### A bonus result for explainability

The same proof techniques tighten a known result about **sufficient reasons**, the smallest set of input features that fixes a model's decision. Earlier work showed hardness for general MLPs; the paper proves NP-hardness, W[1]-hardness and inapproximability under the probabilistic schemes even when the model has only one hidden layer.

## The whole landscape on one page

![Table 4 from Adolfi, Vilas and Wareham (2025): complexity of every circuit query variant. Question marks are open problems.](/imgs/blogs/the-computational-complexity-of-circuit-discovery-for-inner-interpretability-fig3.webp)

Table 4 is dense, so here is the shape of it:

| Query | Local | Global | Escape hatch |
| --- | --- | --- | --- |
| Sufficient circuit | NP-complete | $\Sigma_2^p$-complete | Quasi-minimal (PTIME) |
| Count sufficient circuits | #P-complete | #P-hard | None found |
| Circuit ablation / clamping | NP-complete | NP-hard, in $\Sigma_2^p$ | SAT for local |
| Circuit patching | NP-complete | NP-hard, in $\Sigma_2^p$ | Quasi-minimal (PTIME) |
| Necessary circuit | NP-hard, in $\Sigma_2^p$ | NP-hard, in $\Sigma_2^p$ | None found |
| Circuit robustness | coNP-complete | coNP-hard, in $\Pi_2^p$ | FPT in the probed set size |
| Gnostic neurons | PTIME | open | Already easy |

Several cells are question marks, including whether *unbounded* minimal sufficient circuit is hard. The authors flag these as "potentially fruitful open problems."

![The complexity ladder of circuit queries: from PTIME relaxations up to the Sigma-2-p-complete global sufficient circuit.](/imgs/blogs/the-computational-complexity-of-circuit-discovery-for-inner-interpretability-1.webp)

## What this means for your circuit tool

The paper's closing section is a set of research avenues, and they translate into a practical checklist. The decision graph below is my synthesis of that section, not a figure from the paper.

![A decision path for a hard circuit query: exact if PTIME, FPT if a parameter stays small, SAT if NP-complete, otherwise relax.](/imgs/blogs/the-computational-complexity-of-circuit-discovery-for-inner-interpretability-5.webp)

- **Name the problem your heuristic solves.** A greedy, polynomial method is solving a quasi-minimal problem. Do not call its output "the minimal circuit".
- **Separate local from global claims.** A circuit verified on a dataset is a local object; no cheap check certifies global faithfulness.
- **Hunt for small parameters.** Robustness shows the pattern: a hard query becomes cheap when the piece you search over is bounded.
- **Use solvers where the theory allows.** The NP-complete local queries can be encoded for SAT or ILP solvers on small instances, giving exact baselines for heuristics.

Feature-level methods such as [Sparse Feature Circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models) and [transcoder circuits](/blog/paper-reading/ai-interpretability/transcoders-find-interpretable-llm-feature-circuits) rely on gradient approximations and thresholds, which this paper suggests reading as fast answers to relaxed questions. For why neurons are an awkward unit in the first place, see [language model circuits in the neuron basis](/blog/machine-learning/ai-interpretability/language-model-circuits-neuron-basis).

## Critique

**What is strong.** The formalization is the contribution. Turning "faithfulness", "minimality" and "patching" into precise problems with one template makes the field's vocabulary checkable. The local versus global separation is a clean, memorable explanation of interpretability illusions. And the paper does not stop at bad news: quasi-minimality is a genuinely useful idea that names what greedy methods produce.

**What is weak or limited.**

- **Worst case only.** The hard instances are constructed, like the CLIQUE network, not trained. Trained networks could be systematically easier; the authors argue otherwise from empirical results, but that is an argument, not a measurement.
- **MLPs at neuron granularity.** The results bear on neuron-level discovery in transformers, but not directly on attention heads or SAE latents, where most circuit work happens. My reading: hardness should carry over, tractability would not automatically.
- **Binary input and output.** Several constructions use Boolean inputs and a step output. The appendix argues continuous output does not necessarily change the picture.
- **No experiments.** Nobody measures here how far real heuristic circuits are from minimal. The ACDC explanation is persuasive but untested.

**What would change my mind.** If trained networks were shown to have a structural property (heavy redundancy, or few interacting neurons per behavior) under which minimal sufficient circuit becomes FPT, and ACDC-style circuits on them came out minimal and globally faithful, I would read these results as true but practically loose.

## References

- Adolfi, F., Vilas, M. G., Wareham, T. *The Computational Complexity of Circuit Discovery for Inner Interpretability.* ICLR 2025 (spotlight). [arXiv:2410.08025](https://arxiv.org/abs/2410.08025)
- Conmy, A. et al. *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS 2023 (ACDC, discussed in the paper's Section 5).
- Barceló, P. et al. *Model Interpretability through the Lens of Computational Complexity.* NeurIPS 2020 (the MLP model and ReLU gadgets the proofs build on).
- On this site: [Sparse Feature Circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models), [Transcoders Find Interpretable LLM Feature Circuits](/blog/paper-reading/ai-interpretability/transcoders-find-interpretable-llm-feature-circuits), [Language model circuits in the neuron basis](/blog/machine-learning/ai-interpretability/language-model-circuits-neuron-basis).

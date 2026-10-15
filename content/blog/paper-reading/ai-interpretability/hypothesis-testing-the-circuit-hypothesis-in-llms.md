---
title: "Hypothesis Testing the Circuit Hypothesis in LLMs: Five Statistical Tests for Whether a Circuit Is Real"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How Shi et al. turn the vague claim that an LLM runs a task through a small circuit into five concrete hypothesis tests, and what those tests say about six famous circuits."
tags: ["paper-reading", "mechanistic-interpretability", "circuit-hypothesis", "hypothesis-testing", "circuits", "activation-patching", "large-language-model", "neurips-2024-poster"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 18
paper:
  title: "Hypothesis Testing the Circuit Hypothesis in LLMs"
  authors: "Claudia Shi, Nicolas Beltran-Velez, Achille Nazaret, Carolina Zheng, Adrià Garriga-Alonso, Andrew Jesson, Maggie Makar, David M. Blei"
  venue: "NeurIPS 2024"
  url: "https://arxiv.org/abs/2410.13032"
---

> [!tldr]
> **Claim.** "This task is done by that circuit" is a scientific hypothesis, so it deserves a hypothesis test. The paper names three ideal properties of a circuit (it preserves the behavior, localizes it, and is minimal) and builds five statistical tests around them.
>
> **Why it matters.** Circuit papers usually report one faithfulness number. These tests add a reference distribution and a p-value, so you can say how much better than chance a circuit is.
>
> **Most surprising result.** None of the four circuits found in real Transformers passes every idealized test. The IOI circuit in GPT-2 small fails all three, yet beats random circuits 90% the size of the whole model.
>
> **Where it breaks.** For two tests, *retaining* the null counts as passing, and the paper never measures their power. The partial-necessity results also depend on the ablation scheme.

Mechanistic interpretability has a favorite kind of result: a small subgraph inside a language model with a label like "this is the circuit that identifies indirect objects." The claim behind it is the **circuit hypothesis**: an LLM implements a capability through a small subnetwork, and the rest of the model is mostly irrelevant to it.

That claim says three things. The circuit reproduces the behavior, the rest of the model carries no task information, and the circuit contains nothing redundant. Yet the usual evidence is a single faithfulness score of the form "the circuit recovers X% of the model's logit difference." This paper, from David Blei's group at Columbia with collaborators at Michigan and FAR AI, asks the statistician's question: X% compared to what?

The answer is five hypothesis tests, a software package called `circuitry`, and a report card for six circuits from the literature. If you read our walkthrough of [sparse feature circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models), think of this paper as the auditor that checks such results afterwards.

## The objects: graphs, tasks, and circuits

The paper treats the model as a function $M : \mathcal{X} \to \mathcal{O}$ from token sequences to sequences of logit vectors. It is computed by a **computational graph** $G = (V, E)$: a directed acyclic graph whose nodes are operations and whose edges say "the output of $u$ feeds the input of $v$."

Following Elhage et al., nodes are attention heads and MLP layers, plus input nodes for token embeddings and an output node for the logits. Edges are residual-stream connections. Each edge into an attention head splits into three (query, key, value), which is why edge counts grow fast.

![Figure 1 from Shi et al. (2024): a simplified two-layer, two-head attention-only Transformer as a computational graph, with an arbitrary circuit highlighted.](/imgs/blogs/hypothesis-testing-the-circuit-hypothesis-in-llms-fig1.webp)

A **circuit** is a subgraph $C = (V_C, E_C)$ that keeps the input and output nodes and a subset of edges between them. Its **complement** $\bar{C}$ is every edge not in $C$. Running $\bar{C}$ is called **knocking out** $C$.

A **task** is a pair $\tau = (\mathcal{D}, s)$: a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ and a score function $s(o, y) \in \mathbb{R}$ where higher is better. For greater-than (Hanna et al.), the prompt is "The war lasted from the year 1973 to the year 19" and the score is the probability on years after 73 minus the probability on the rest.

| Task | Model | Score $s$ | Ablation used |
| --- | --- | --- | --- |
| IOI (indirect object) | GPT-2 small | logit(" Mary") minus logit(" John") | STR patching |
| Greater-than (G-T) | GPT-2 small | prob. of valid minus invalid years | STR patching |
| Induction | 2-layer, 8-head attention-only | log prob. of the correct token | zero ablation |
| Docstring (DS) | 4-layer attention-only | logit diff vs best wrong argument | STR patching |
| Tracr-R, Tracr-P | compiled RASP programs | sum of token-level $\ell_2$ distances | zero ablation |

The Tracr models are compiled from RASP programs, so their circuits are ground truth by construction. They are the sanity check: a good test should pass them.

## Running a circuit means patching everything else

A subgraph is not a runnable model. If node $v$ normally reads from $u$ and the edge $(u, v)$ is not in the circuit, something still has to arrive at that input. **Activation patching** replaces the missing input $a_u$ with a substitute $a_u^*$ for every edge outside the circuit.

**Zero ablation** sets $a_u^* = 0$. **Symmetric token replacement (STR)** builds a corrupted prompt $x^c$ with the key tokens swapped, runs the model on it, caches its activations, and feeds those into the missing edges while the circuit runs on $x$. For greater-than, "1973" becomes "1901": the sentence still reads naturally, but the "after 73" constraint is gone.

![Running a circuit with activation patching: edges inside the circuit carry clean activations, edges outside receive cached activations from a corrupted prompt (STR) or zeros (zero ablation).](/imgs/blogs/hypothesis-testing-the-circuit-hypothesis-in-llms-4.webp)

The ablation scheme hides inside the notation $C(x)$, but it is not neutral. Later, one test gives opposite pictures under STR and zero ablation.

## Faithfulness: the number everyone reports

Faithfulness measures how closely a circuit reproduces the model. The paper averages a per-example metric $F$ over the task data:

$$
F_\tau(M, C) := \mathbb{E}_{(X, Y) \sim \mathcal{D}} \big[ F(M, C, X, Y) \big]
$$

Here $M$ is the full model, $C$ the circuit, and $(X, Y)$ a random task example. All experiments use the squared score gap:

$$
F(M, C, x, y) = \big| s(M(x), y) - s(C(x), y) \big|^2
$$

Because it is a distance, **lower means more faithful**. A tiny example: on three IOI prompts the model's logit differences are 3.0, 2.5 and 4.0, and the circuit's are 2.8, 2.0 and 4.1. The squared gaps are 0.04, 0.25 and 0.01, so $F_\tau = 0.30 / 3 = 0.10$. Is 0.10 good? On its own, nobody can say.

## A hypothesis test in five parts

Every test in the paper fills the same five slots.

1. A **variable of interest** $Z^*$, such as the candidate circuit's faithfulness.
2. A **reference distribution** $P_Z$ with $n$ samples, such as the faithfulness of $n$ random circuits.
3. A **null hypothesis** $H_0$ relating $Z^*$ to $P_Z$.
4. A **test statistic** $t$ whose distribution under $H_0$ is known, often a binomial count.
5. A **rejection region** $R_\alpha$ with $P(t \in R_\alpha) \le \alpha$ under $H_0$. Landing in it lets us reject $H_0$ with confidence ${1-\alpha}$.

The p-value is the smallest $\alpha$ at which the observed statistic would be rejected. If the vocabulary is rusty, our [plain-language guide to p-values](/blog/trading/math-for-quants/hypothesis-testing-pvalues-math-for-quants) builds it from scratch.

![The five-part hypothesis-testing template. Every test in the paper fills the same slots; only the variable of interest and the reference distribution change.](/imgs/blogs/hypothesis-testing-the-circuit-hypothesis-in-llms-1.webp)

Think of a drug trial. The treated group is the variable of interest, the placebo group is the reference distribution, and the null is "the drug does nothing." The craft in this paper is choosing the placebo: a random circuit, a random edge, or the full model, depending on the property under test.

## Three ideal properties, five tests

The paper posits three properties of an ideal circuit.

- **Mechanism preservation.** The circuit alone performs the task as well as the model.
- **Mechanism localization.** Knocking the circuit out destroys the behavior.
- **Minimality.** The circuit has no redundant edges.

Preservation and localization each get a stringent **idealized** test, which compares against the model itself, and a **flexible** test, which compares against random circuits from a distribution you choose. Minimality gets one test. That makes five.

![How three ideal properties map onto five tests. Idealized tests compare against the model; flexible tests compare against random circuits.](/imgs/blogs/hypothesis-testing-the-circuit-hypothesis-in-llms-2.webp)

One subtlety governs every result table. For equivalence and independence, the null sits in the *ideal* direction, so **retaining the null is the pass**. For sufficiency and partial necessity, **rejecting the null is the pass**. For minimality, rejecting the null for any edge means the circuit has a redundant edge.

## Test 1: Equivalence

**Problem.** If a circuit really is the mechanism, it should do the task as well as the full model, not just "pretty well."

**Intuition.** Race the circuit against the model on each example. If they are equivalent, wins should split near 50/50, like a fair coin. A circuit that loses 97 times out of 100 is not the same machine.

**Math.** Define the per-example difference

$$
\Delta(x, y) = s(C^*(x); y) - s(M(x); y)
$$

where $C^*$ is the candidate circuit. The null says the circuit wins about half the time, within tolerance $\epsilon$ (the paper uses $\epsilon = 0.1$):

$$
H_0 : \left| P_{(X,Y) \sim \mathcal{D}}\big(\Delta(X, Y) > 0\big) - \tfrac{1}{2} \right| \lt \epsilon
$$

The statistic is the distance of the empirical win rate from one half, $t = \left| \frac{1}{n}\sum_i \mathbb{1}\{\Delta(x_i, y_i) > 0\} - \frac{1}{2} \right|$. The null covers a whole interval of win probabilities $\theta$, and the appendix shows the worst case is its edge, $\theta = \tfrac{1}{2} + \epsilon$. The p-value is the binomial probability, at that $\theta$, of a win rate at least as far from one half as observed.

**Worked example.** Appendix Table 3 says the IOI circuit beats GPT-2 small on 24% of examples. Suppose (my illustration; the paper does not give $n$ here) that is 24 wins out of 100. Then $t = 0.26$, and for $K \sim \text{Bin}(100, 0.6)$, the chance of $K \ge 76$ or $K \le 24$ is about 0.00056. We reject $H_0$: **not equivalent**. With 53 wins the p-value is about 0.94 and the null stands.

**When it fails.** A pass is the *absence* of evidence against equivalence, and with few examples a mediocre circuit can pass on low power alone, which the paper concedes it did not study. The test also breaks when the circuit equals the model, as with Tracr: every $\Delta$ is zero and the sign test cannot run. The paper counts Tracr as passing by construction.

## Test 2: Independence

**Problem.** Equivalence checks the circuit alone. Localization asks the mirror question: after knocking the circuit out, is anything task-relevant left?

**Intuition.** If the circuit is the only place the task lives, the knocked-out model's score on a prompt should tell you nothing about how the full model does on that prompt. The two score columns should look unrelated.

**Math.** The null is

$$
H_0 : s(\bar{C}^*(X); Y) \perp\!\!\!\perp s(M(X); Y)
$$

where $\bar{C}^*$ is the complement (the main text writes $C^*$, but Algorithm 2 makes clear the complement is meant). Dependence is measured with the **Hilbert-Schmidt Independence Criterion** (HSIC, Gretton et al. 2007), which compares the two variables in a kernel feature space. The standard empirical form, which I give because the paper's definition is compressed, is

$$
\widehat{\text{HSIC}} = \frac{1}{n^2} \operatorname{tr}(K H L H)
$$

where $K$ and $L$ are $n \times n$ kernel matrices (for example RBF with bandwidth $\rho$) over the two score vectors, and $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$ centers them. In the population HSIC is zero exactly under independence, and it catches nonlinear dependence that correlation misses. The p-value comes from a **permutation test**: shuffle the model's scores $B$ times, recompute HSIC, and count how often the shuffled value beats the observed one.

**Worked example.** Appendix Table 4 has the real numbers. Induction: HSIC 0.001, p-value 0.669, so the null is retained and the circuit passes. G-T has p-value 0.000, and IOI and Docstring 0.01: some task information survives knockout. The raw HSIC values are all tiny; the permutation test is what separates "tiny" from "zero."

**When it fails.** Passing means failing to reject, so an underpowered test passes bad circuits. The kernel bandwidth $\rho$ also shifts sensitivity, and no sweep is reported.

## Test 3: Minimality

**Problem.** A circuit padded with useless edges overstates how much of the model is involved. We want to flag removable edges.

**Intuition.** "Removing this edge barely changes the score" means nothing without a yardstick. The paper builds one by adding junk on purpose: bolt a random path onto the circuit, remove an edge *on that path*, and record the change. That is what an unnecessary edge looks like. Each real edge is compared against it.

**Math.** The effect of removing edge $e$ from circuit $C$ is

$$
\delta(e, C) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \big| s(C(x), y) - s(C_{-e}(x), y) \big|
$$

where $C_{-e}$ is $C$ without $e$. Let $(C^I, e^I) \sim R_I$ be a random inflated circuit and a random edge drawn from its newly added edges. The null says the candidate edge beats junk with high probability:

$$
H_0 : P_{(C^I, e^I) \sim R_I}\big(\delta(e^*, C^*) > \delta(e^I, C^I)\big) > q^*
$$

with $q^* = 0.9$. Rejecting $H_0$ for an edge means it is no more important than junk, so the circuit is **not minimal**. Testing all $m$ edges is $m$ tests, so a **Bonferroni correction** tests each at level $\alpha / m$.

![The minimality test. Removing candidate edge e* is compared against removing an edge e^I from a randomly added path; the junk-path knockouts form the reference distribution.](/imgs/blogs/hypothesis-testing-the-circuit-hypothesis-in-llms-3.webp)

**Worked example.** With $\alpha = 0.05$ and an IOI circuit of roughly 963 edges (the axis range of the paper's Figure 3), each edge is tested at ${0.05 / 963 \approx 5.2 \times 10^{-5}}$. That is a high bar for calling any edge redundant, which makes the finding that IOI *has* redundant edges more convincing. The reference uses 10,000 random edges for G-T and IOI and 1,000 for the rest.

![Figure 4 from Shi et al. (2024): one step of the minimality test on the Docstring circuit. The candidate edge is orange, the randomly added path blue, and the knocked-out reference edge red.](/imgs/blogs/hypothesis-testing-the-circuit-hypothesis-in-llms-fig4.webp)

**When it fails.** Everything rests on the assumption that a random path is useless. In a model with redundant, distributed computation, a random path may carry real signal, inflating the reference and making true edges look unnecessary. A useful side effect: ACDC sweeps its edge-importance threshold, and this test supplies a principled one.

## Test 4: Sufficiency (flexible)

**Problem.** Every discovered circuit fails equivalence. We still want to know *how good* a circuit is.

**Intuition.** Race the circuit against random circuits instead of the model. If it beats almost all random circuits of its size, it is no accident. Then grow the opponents until the circuit stops winning. That breakpoint is a graded quality score.

**Math.** With $C^r$ drawn from a reference distribution $\mathcal{R}$, the null is

$$
H_0 : P_{C^r \sim \mathcal{R}}\big(F_\tau(M, C^*) \lt F_\tau(M, C^r)\big) \le q^*
$$

Lower $F_\tau$ is more faithful, so the event inside reads "the candidate beats the random circuit." The statistic is the fraction of wins over $n$ random circuits; under $H_0$ the win count is binomial with success probability $q^*$, and a one-sided binomial test gives the p-value.

**Worked example.** With the paper's $n = 100$ and $q^* = 0.9$, suppose the candidate wins 97 times. The probability of 97 or more wins under $\text{Bin}(100, 0.9)$ is about 0.008, below $\alpha = 0.05$: sufficient. With 95 wins it is about 0.058, and the claim fails. Two wins out of a hundred separate the verdicts.

![Figure 2 from Shi et al. (2024): left, the probability the candidate is more faithful than a random circuit as random circuits grow; right, the probability a random circuit contains the canonical circuit.](/imgs/blogs/hypothesis-testing-the-circuit-hypothesis-in-llms-fig2.webp)

The IOI and G-T circuits stay significantly more faithful than random circuits up to **90% of the model's size**. Docstring and Induction hold up to **30%**. Whatever else is wrong with them, these circuits are far from random.

The right panel is the clever part. Under a perfect circuit hypothesis, a random circuit matches the candidate only when it *contains* it, so the left curve should mirror the right, and roughly it does. Induction and Docstring look alike on the left but differ on the right, which the authors read as Induction being closer to ideal.

**When it fails.** Everything depends on the sampler: random walks from embeddings to logits, added until the circuit has at least $k$ edges. A sampler biased toward late layers would move every curve.

## Test 5: Partial necessity (flexible)

**Problem.** Most real circuits fail independence. A weaker but meaningful claim is that the candidate matters *more* than other parts of the model.

**Intuition.** Knock out the candidate and measure the damage, then do the same for random circuits of equal size. If removing the candidate hurts more than almost any random removal, it is load-bearing, even if not the sole carrier.

**Math.** The null is

$$
H_0 : P_{C^r \sim \mathcal{R}}\big(C^* \text{ is worse to knock out than } C^r\big) \le q^*
$$

where "worse to knock out" is defined by the faithfulness comparison $F_\tau(M, C^*) > F_\tau(M, C^r)$, with the same binomial test. The lever is $\mathcal{R}$: drawing from the **complement** of $C^*$ is easier, drawing from the **whole model** is harder.

**Worked example.** The paper's Table 2, restated:

| Reference circuit drawn from | Induction | IOI | G-T | DS |
| --- | --- | --- | --- | --- |
| Complement of $C^*$ | pass | pass | pass | pass |
| Whole model $M$ | pass | pass | pass | fail |

All four candidates hurt more to remove than same-sized chunks of the rest of the model, but Docstring's knockout is not reliably worse than removing a random same-size subgraph of the full model.

**When it fails.** Growing the random circuits exposes an artifact. For G-T and IOI, **knocking out the complete model hurts less than knocking out the candidate**, and only under STR patching. Under zero ablation, the candidate's relative significance falls steadily as random circuits grow, as expected; under STR it stays flat. The authors conclude the knockdown metric alone cannot judge a circuit.

My reading (an extrapolation, not the paper's claim): when every edge is patched from a prompt that differs by two digits, "knock out the whole model" is close to "run the model on a near-identical prompt," which keeps much of the task structure.

## Results: the report card

All experiments reuse the ACDC configuration (Conmy et al. 2023), run ablations through TransformerLens, and set $q^* = 0.9$ and $\alpha = 0.05$. The `circuitry` package evaluates hundreds of circuits in a few minutes on one A5000 GPU. Table 1, the idealized tests, with "pass" meaning the circuit meets the ideal:

| Test | Tracr-P | Tracr-R | Induction | IOI | G-T | DS |
| --- | --- | --- | --- | --- | --- | --- |
| Equivalence ($\epsilon = 0.1$) | pass | pass | fail | fail | fail | fail |
| Independence | pass | pass | pass | fail | fail | fail |
| Minimality | pass | pass | pass | fail | fail | pass |

Tracr passes everything, so the tests do not reject ground truth. Induction passes two of three; Docstring passes minimality; IOI and G-T fail all three. Behind the equivalence row, the candidate beats the model on only 3% of examples for Induction, 7% for G-T, 10% for Docstring and 24% for IOI.

A note on notation. The Table 1 caption says a check mark means "the null is rejected," yet Tracr carries check marks for equivalence and independence, where rejection is failure, and Appendix E.2 says retaining counts as passing. The table only makes sense if a check means "passes," which is how I wrote it.

![Figure 3 from Shi et al. (2024): faithfulness of the G-T, Docstring and IOI circuits as edges are removed in order of their minimality score.](/imgs/blogs/hypothesis-testing-the-circuit-hypothesis-in-llms-fig3.webp)

Figure 3 removes edges starting from the least important. For G-T you can drop **around 50%** of the edges with no loss in faithfulness; for IOI, **about 20%**. The IOI curve is not monotonic, which the authors attribute to components like negative mover heads. Even Docstring tolerates losing a few edges, because its reference edges are all near zero.

![Figure 5 from Shi et al. (2024): the change in score from knocking out each edge of each circuit, sorted, with the reference distribution as a side histogram and a shaded region for non-significant edges.](/imgs/blogs/hypothesis-testing-the-circuit-hypothesis-in-llms-fig5.webp)

Figure 5's axes also give circuit sizes: about 5 edges for Tracr-P, 11 for Tracr-R, 6 for Induction, 24 for Docstring, 235 for G-T and 963 for IOI. The two real circuits that do best are the two smallest, both found in toy attention-only models.

### What is load-bearing in this setup

- **The ablation scheme.** Induction and Tracr use zero ablation; the rest use STR. Cross-circuit comparisons therefore mix two kinds of experiment.
- **The random-circuit sampler.** The flexible tests are defined relative to it, and random walks are a reasonable default, not a canonical one.
- **The original setup.** Each circuit is tested where it was discovered, and the authors note circuits are not robust to setup changes.

## Critique

**What is strong.** The framing is the contribution. Turning "this circuit explains the task" into nulls, reference distributions and p-values makes a field more honest, and none of the tests require searching the space of circuits, unlike the completeness checks in the original IOI paper. The flexible tests give a graded answer (beats random circuits up to 30% or 90% of model size) instead of a binary one. The Tracr controls are exactly the right sanity check.

**What is weak.**

- **No power analysis where retaining is passing.** The authors say so directly: Type II error for equivalence and independence was not studied. Until it is, a pass is weak evidence.
- **Inconsistent reporting.** Besides the Table 1 caption, Appendix E.3 says "only the Induction and Tracr circuits pass the minimality test," while Table 1 and Section 4.2 say Docstring passes too. In a paper about rigor, that matters.
- **Six circuits, small models.** Four real circuits, two in toy models, is a thin base. The paper wisely frames its verdict as about "existing circuits."
- **Ablation confound.** Mixing zero ablation and STR, then finding an STR artifact, means some differences may be about the ablation rather than the circuit.

**The ablation I would want.** Re-run all five tests for every real circuit under both ablation schemes, plus a power study: plant a broken circuit (Tracr with one edge removed) and report how often each test catches it at these sample sizes.

**What would change my mind.** If a power study showed equivalence and independence reliably reject a circuit missing a single edge, I would read Induction's independence pass as strong evidence of localization rather than absence of evidence. If IOI or G-T passed independence under zero ablation, I would read their failures as facts about STR patching, not about the circuits.

For what "faithful" means at the neuron level, see our post on [language model circuits in the neuron basis](/blog/machine-learning/ai-interpretability/language-model-circuits-neuron-basis).

## References

- Shi, C., Beltran-Velez, N., Nazaret, A., Zheng, C., Garriga-Alonso, A., Jesson, A., Makar, M., Blei, D. M. (2024). *Hypothesis Testing the Circuit Hypothesis in LLMs*. NeurIPS 2024. [arXiv:2410.13032](https://arxiv.org/abs/2410.13032)
- Code: [github.com/blei-lab/circuitry](https://github.com/blei-lab/circuitry)
- Conmy, A. et al. (2023). *Towards Automated Circuit Discovery for Mechanistic Interpretability* (ACDC).
- Wang, K. et al. (2023). *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*.
- Gretton, A. et al. (2007). *A Kernel Statistical Test of Independence*.
- On this site: [Sparse feature circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models), [Language model circuits in the neuron basis](/blog/machine-learning/ai-interpretability/language-model-circuits-neuron-basis), [Hypothesis testing and p-values, honestly](/blog/trading/math-for-quants/hypothesis-testing-pvalues-math-for-quants)

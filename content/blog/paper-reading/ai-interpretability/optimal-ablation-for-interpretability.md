---
title: "Optimal Ablation for Interpretability: Measuring What a Component Does by Replacing It With the Best Constant"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "Why zero, mean and resample ablation overstate how much a model component matters, how optimal ablation fixes it with one learned constant, and what that changes for circuit discovery, causal tracing and latent prediction."
tags: ["paper-reading", "optimal-ablation", "ablation", "activation-patching", "mechanistic-interpretability", "circuit-discovery", "causal-tracing", "tuned-lens", "interpretability", "neurips-2024-spotlight"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 21
paper:
  title: "Optimal Ablation for Interpretability"
  authors: "Maximilian Li, Lucas Janson"
  venue: "NeurIPS 2024 (spotlight)"
  url: "https://arxiv.org/abs/2409.09951"
---

> [!tldr]
> **The claim.** When we "ablate" a model component to see how much it matters, the replacement value we pick (zero, the mean, a value from another input) does damage of its own. Li and Janson propose **optimal ablation (OA)**: replace the component with the single constant that minimizes the ablated model's loss.
> **Why it matters.** Among all ablations that remove every bit of input information from the component, OA provably gives the smallest loss gap. So any extra damage reported by other methods is not "importance", it is the model being confused by a misleading value.
> **Most surprising result.** On the IOI task, for the median attention head or MLP, the OA loss gap is only **11.1%** of the zero-ablation gap and **17.7%** of the resample gap. Most of what earlier ablations measured was noise of their own making.
> **Where it fails.** OA needs a gradient-descent run per component (or per circuit), it cannot remove the spoofing a component causes when every possible value carries information, and its extra freedom could, in principle, let a circuit use mechanisms the model does not normally use.

Every interpretability study eventually asks the same question about some neuron, head or layer: *does this thing matter for the behavior we care about?* The standard answer is an experiment. Knock the component out, rerun the model, and see how much worse it gets.

The trouble is the word "out". A transformer has no "off" switch for an attention head. Its output is a vector that the rest of the network reads, and you must put *some* vector there. Zero? The average? The value it produced on a different prompt? Each choice is a different experiment, and as this paper shows, they give wildly different answers.

The paper turns "which replacement value?" into an optimization problem with a provably best answer, then shows that answer improves three tools: circuit discovery, factual-recall localization, and latent prediction.

## The question an ablation is secretly asking

**The problem.** Prior work uses incompatible ablations: Meng et al. (2022) add Gaussian noise, Geva et al. (2023) use zeros, Ghorbani and Zou (2020) use training means. Nobody agrees which number is "the" importance.

**The analogy.** To measure one relay runner's contribution, you cannot just delete them; someone must carry the baton. A substitute who sprints the wrong way loses the race, but that says more about the substitute than the runner. The fair test: how well could the best substitute do *knowing nothing about this particular race*?

**The mechanism.** The paper writes the importance of a component $A$ as a gap between an ablated model and the full model. Here $M$ is the trained model, $A$ is any internal component (a neuron, a direction, an attention head, a whole layer), and $A(x)$ is its value on input $x$.

$$
\Delta(M, A) := \mathcal{P}(M_{\setminus A}) - \mathcal{P}(M)
$$

where $M_{\setminus A}$ is the model with $A$ ablated and $\mathcal{P}$ is a performance metric. The paper pins down $\mathcal{P}$ as expected loss *against the full model's own predictions* on a subtask distribution $\mathcal{D}$ (a curated set of prompts, like IOI):

$$
\mathcal{P}(\tilde M) = \mathbb{E}_{X \sim \mathcal{D}}\, \mathcal{L}\big(\tilde M(X),\, M(X)\big)
$$

Since the full model scores zero against itself, $\Delta$ is just the ablated model's expected loss: with KL as $\mathcal{L}$, how far its outputs drift from the original. The paper calls this the **ablation loss gap**.

From the motivating question ("what is the best performance on $\mathcal{D}$ that $M$ could have achieved without $A$?") the paper derives three requirements:

| Phrase in the question | Requirement on $M_{\setminus A}$ |
| --- | --- |
| "the model $M$ could have achieved" | Change only the value of $A$; hold every weight and every other component fixed. Written $M_A(x, a)$: run $M$ on $x$ but set $A$ to $a$. |
| "without component $A$" | The value placed in $A$ must carry **no information about $x$**. |
| "best" | Among all ways of satisfying the first two, take the one with the **lowest** $\Delta$. |

The second row gets a name. A **total ablation** sets $M_{\setminus A}(X) = M_A(X, \mathcal{A})$ for some random value $\mathcal{A}$ that is statistically independent of $X$. Zero, mean and resample ablation are all total. Adding Gaussian noise to $A(x)$ is *partial*: the noisy value still depends on $x$. So is counterfactual ablation, discussed below.

## Deletion versus spoofing: two ways a replacement hurts

The "best" requirement rests on a distinction the paper makes explicit. Replacing $A(x)$ with $a$ can hurt in two ways.

1. **Deletion.** $A(x)$ carried information about $x$ that downstream computation used. Replacing it removes that information. This is the effect we actually want to measure.
2. **Spoofing.** The replacement value *inserts* information. Either it makes the model treat $x$ like some other input $x'$, or it contradicts what other components say about $x$ and pushes later activations into a regime never seen in training, where outputs are unregulated.

Every total ablation achieves maximal deletion, since $\mathcal{A}$ knows nothing about $X$. They differ only in how much spoofing they add. So the *least damaging* total ablation is the one closest to pure deletion.

![Deletion versus spoofing on the paper's two-input toy classifier: patching in the other input's value makes the model confidently wrong, while the best constant makes it hedge](/imgs/blogs/optimal-ablation-for-interpretability-1.webp)

The figure above is the paper's own toy example from Appendix D.1, redrawn. A classifier sees one of two inputs, $x$ or $x'$, and relies entirely on component $A$. Its outputs are $M(x) = (1, 0)$ and $M(x') = (0, 1)$. If we honestly remove $A$'s information, the best the model can do is hedge at $({0.5}, {0.5})$, a KL loss of $\log 2 \approx {0.69}$. If instead we patch in $A(x')$ while the input is $x$, the model puts all its probability on the wrong label and the KL loss is infinite. Same component, same "removal", infinitely different verdict.

## Optimal ablation: the definition

**The mechanism.** Pick the one constant that does the least damage across the whole subtask, and use it as the replacement for every input.

$$
M^{\setminus A}_{(\text{opt})}(x) := M_A(x, a^*), \qquad a^* := \arg\min_{a}\; \mathbb{E}_{X \sim \mathcal{D}}\, \mathcal{L}\big(M_A(X, a),\, M(X)\big)
$$

Here $a^*$ has the same shape as $A(x)$, for example a $d_\text{model}$-dimensional vector for an attention head's output. Plugging this model into the gap gives $\Delta_\text{opt}(M, A)$. Because $a^*$ is a constant, it is trivially independent of $X$, so OA is a total ablation.

**The guarantee (Proposition 2.3).** For any total ablation, $\Delta_\text{opt}(M, A) \le \Delta(M, A)$. The proof is three lines and worth reading because it shows exactly why the result holds:

$$
\Delta(M, A) = \mathbb{E}_{\mathcal{A}}\Big[\, \underbrace{\mathbb{E}_{X}\, \mathcal{L}\big(M_A(X, \mathcal{A}), M(X)\big) \,\big|\, \mathcal{A}}_{g(\mathcal{A})} \Big] \;\ge\; \min_a g(a) = \Delta_\text{opt}(M, A)
$$

Because $\mathcal{A}$ is independent of $X$, conditioning on $\mathcal{A} = a$ just gives $g(a)$, the loss of the fixed constant $a$. Any random replacement is an average of constants, and an average is never below the minimum. Zero ablation is the case where $\mathcal{A}$ is always ${0}$; resample ablation draws $\mathcal{A} = A(X')$ from an independent prompt.

The consequence is the paper's key interpretive move. For any method that replaces $A(x)$ with an input-independent value, **the excess gap $\Delta - \Delta_\text{opt}$ is entirely spoofing**. The constant $a^*$ achieves lower loss without revealing anything more about $x$.

### A worked micro-example

Take the toy classifier again, with the two inputs equally likely. Suppose a constant $a$ makes the model output $(q, 1-q)$ on both inputs. The expected KL to the full model is

$$
g(q) = \tfrac{1}{2}\big(-\log q\big) + \tfrac{1}{2}\big(-\log(1-q)\big)
$$

which is minimized at $q = {0.5}$ with $g = \log 2 \approx {0.693}$. That is $\Delta_\text{opt}$.

Now compare. Suppose (my assumption, for illustration) zero ablation happens to push the output to $({0.9}, {0.1})$. Then $\Delta_\text{zero} = \tfrac{1}{2}({0.105} + {2.303}) \approx {1.204}$. Resample ablation draws the other input's value half the time, so half the time the loss is infinite and $\Delta_\text{resample} = \infty$.

| Method | $\Delta$ | Spoofing share, $\Delta - \Delta_\text{opt}$ |
| --- | --- | --- |
| Optimal | 0.693 | 0 |
| Zero (assumed output) | 1.204 | 0.511 |
| Resample | infinite | infinite |

The component's real importance is the same in every row. Only the substitute changed.

### How $a^*$ is computed

There is no closed form, so the paper uses mini-batch SGD on $a$, initialized at the subtask mean $\mathbb{E}_{X \sim \mathcal{D}}[A(X)]$. In the circuit experiments, a single constant per component is shared across all token positions (except the position-0 BOS token, which is never ablated because it does not depend on the input). The authors call this a conservative choice: other methods get position-specific means, and OA still wins.

### Why the mean is a special case

Appendix D.2 shows that if the loss were squared distance and the model *linear* in $A$, the optimal constant would be exactly $\mathbb{E}[A(X)]$. Mean ablation is the right answer for a linear world. Transformers are not linear, so OA replaces "the average value" with "the value whose *effect* is best on average".

**When it fails.** The paper is upfront that OA does not remove all spoofing. If $A$ always carries strong, exact information, there may be no value that hedges between inputs, and every constant still lies to the model. It also costs an optimization per component, versus one forward pass for zero or mean.

## How the existing ablations compare

![The five ablation methods side by side: what each puts in place of A(x), whether it is total or partial, whether it needs a hand-built counterfactual, and the median ratio of the OA gap to its gap on IOI](/imgs/blogs/optimal-ablation-for-interpretability-2.webp)

The menu above has one method that is not a total ablation. **Counterfactual ablation (CF)**, from Hanna et al. (2023) and the IOI work of Wang et al. (2022), replaces $A(x)$ with $A(\pi(x))$, where $\pi(x)$ is a hand-built "neutral" twin of $x$. In IOI, $\pi$ swaps the names for random distinct names. CF is the most popular method in the field because $\pi(x)$ shares almost every token with $x$, so it avoids the "contradicts other components" flavor of spoofing.

CF has two costs. It needs someone to engineer a good $\pi$ for each subtask, which is hard beyond toy prompts. And since $A(\pi(x))$ depends on $x$, it is partial and can leak task information.

The paper tests all of these on the Indirect Object Identification (IOI) task with GPT-2 small. A typical prompt is "When Mary and John went to the store, Mary gave the apple to ___", and the model should answer "John". They measure $\Delta$ for every attention head and MLP block.

![Table 1 from Li and Janson (2024): rank correlation of each method's single-component loss gap with CF, and the median ratio of the OA gap to each method's gap on IOI](/imgs/blogs/optimal-ablation-for-interpretability-fig5.webp)

Two numbers stand out in Table 1:

- **The gap is mostly spoofing.** For the median component, $\Delta_\text{opt}$ is 11.1% of $\Delta_\text{zero}$, 33.0% of $\Delta_\text{mean}$ and 17.7% of $\Delta_\text{resample}$. Read that as: roughly 89% of the zero-ablation number is damage caused by the zero itself.
- **OA agrees with the careful method.** Among all the methods, $\Delta_\text{opt}$ has the highest rank correlation with $\Delta_\text{CF}$: 0.907, against 0.590 for zero and about 0.83 for mean, resample and CF-mean. OA reaches CF's ranking without needing anyone to hand-craft counterfactual prompts.

## Application 1: circuit discovery

**The problem.** Circuit discovery looks for a small set of edges in the model's computation graph that is sufficient to reproduce a behavior. The graph has vertices (heads, MLPs) and edges $e_k = (A_j, A_i, z)$ meaning "the output of $A_j$ is the $z$-th input to $A_i$". Ablating an edge replaces that input with some value $a$.

The objective is to pick a kept-edge set $\tilde E$ that trades loss against size:

$$
\tilde E^* = \arg\min_{\tilde E \subseteq E}\; \Big[\, \Delta(M, E \setminus \tilde E) + R(\tilde E) \,\Big]
$$

where $E$ is the set of all edges, $E \setminus \tilde E$ is the set of ablated edges, and $R$ is a sparsity penalty. The ablation method sits inside $\Delta$, so it decides which circuits look good.

With OA, each vertex $A_j$ gets one learned constant $a^*_j$, passed to all of its ablated out-edges. The catch is that the best constants depend on which edges are kept. That rules out ACDC (Conmy et al., 2023) and Edge Attribution Patching, which would need a fresh OA optimization per candidate circuit. So the authors needed a search method that can learn the constants and the circuit together.

### Uniform Gradient Sampling (UGS)

**The analogy.** Give every edge a biased coin, flip them all, run the model, and nudge each bias by whether keeping that edge helped. Over many flips the coins converge to "always keep" or "always drop".

**The mechanism.** Each edge holds a logit $\tilde\theta_k$, and $\theta_k = \sigma(\tilde\theta_k)$ is the estimated probability that $e_k$ belongs to the circuit. An edge can also be *partially* ablated with a coefficient $\alpha_k \in [0, 1]$: the child receives $\alpha_k A_j(x) + (1 - \alpha_k) a_j$. Here $\alpha_k = 1$ keeps the edge and $\alpha_k = 0$ fully ablates it.

![How UGS samples an ablation coefficient per edge, and why only the uniform draws carry gradient](/imgs/blogs/optimal-ablation-for-interpretability-3.webp)

With a window $w(\theta_k) = \theta_k(1 - \theta_k)$, each forward pass samples, per edge and per input:

- $\alpha_k = 1$ with probability $\theta_k - \tfrac{1}{2}w(\theta_k)$
- $\alpha_k \sim \text{Unif}(0, 1)$ with probability $w(\theta_k)$
- $\alpha_k = 0$ with probability ${1 - \theta_k - \tfrac{1}{2}w(\theta_k)}$

The gradient for $\tilde\theta_k$ uses only the inputs where $\alpha_k$ landed strictly inside $(0, 1)$. With a batch of $b$ inputs and $N_k$ the count of such inputs,

$$
\nabla_{\theta_k} \mathcal{L}_\text{UGS} = \nabla_{\theta_k} R(\vec\theta) + \frac{1}{N_k} \sum_{j=1}^{b} \mathbb{1}\big(\alpha^{(j)}_k \in (0,1)\big)\, \nabla_{\alpha^{(j)}_k} \mathcal{L}\big(M_{\vec\alpha}(X^{(j)}),\, M(X^{(j)})\big)
$$

where $R(\vec\theta)$ is a continuous relaxation of the sparsity penalty and $M_{\vec\alpha}$ is the model with every edge partially ablated by its own $\alpha$.

**Why the uniform draws?** Appendix F.5 gives the reason. The gradient we want for an edge is its *marginal* ablation gap: loss with the edge dropped minus loss with it kept, averaged over the other edges' coins. The average of $\partial \mathcal{L} / \partial \alpha_k$ over $\alpha_k \sim \text{Unif}(0, 1)$ is exactly $\mathcal{L}(\alpha_k{=}1) - \mathcal{L}(\alpha_k{=}0)$, by the fundamental theorem of calculus. So the uniform samples are an unbiased finite-difference estimator, computed for every edge at once in one backward pass. A naive importance-sampling estimator was tried and gave poor estimates, because most of its variance comes from the other edges' coin flips.

**Worked numbers.** An undecided edge with $\theta_k = {0.5}$ has $w = {0.25}$: it is kept 37.5% of the time, gets a uniform draw 25% of the time, and is ablated 37.5% of the time. A confident edge with $\theta_k = {0.9}$ has $w = {0.09}$: kept 85.5%, uniform 9%, ablated 5.5%. The window shrinks as the edge decides, so settled edges stop consuming gradient samples.

The ablation constants $\hat a_j$ are trained by SGD at the same time as the $\tilde\theta_k$, at a lower learning rate (0.002). Training costs about 1–2 hours on one A100, comparable to ACDC.

### What the circuits look like

![Figure 1 from Li and Janson (2024): left, the IOI Pareto frontier of loss gap against edge count under counterfactual ablation for ACDC, EAP, HCGS and UGS; right, UGS circuits under each ablation method, with the manual circuit marked](/imgs/blogs/optimal-ablation-for-interpretability-fig1.webp)

Two findings, one per panel.

**UGS is a better search algorithm, regardless of ablation.** On the left, with CF ablation on IOI, UGS finds a 385-edge circuit with $\Delta_\text{CF} = {0.220}$. That is 52% fewer edges than the smallest ACDC circuit with comparable loss, and 48% lower loss than the best ACDC circuit of comparable size. The appendix reports UGS Pareto-dominating the other searches for every ablation type on both IOI and Greater-Than.

**OA reveals that small circuits explain more than we thought.** On the right, at 1,000 edges on IOI, OA-ablated circuits reach 32% lower $\Delta$ than CF, 62% lower than mean and 88% lower than resample. The gap widens at smaller circuit sizes.

The manual circuits tell the most interesting story:

| Ablation | Optimized circuit (no more edges than the manual one) vs. manual circuit |
| --- | --- |
| Optimal | 29% lower $\Delta$ on IOI, 42% lower on Greater-Than |
| Mean, resample, CF | 84–85% lower on IOI, 70–84% lower on Greater-Than |

Under older ablations, hand-built circuits look far from optimal; under OA they look nearly optimal for their size. The authors read this as more evidence that spoofing inflates the older $\Delta$ values.

A natural worry is that a learned constant per vertex gives the optimizer enough freedom to make *any* edge set look good. Table 3 in the appendix checks this against random circuits. On IOI, a random circuit under OA has loss 2.723 while the UGS circuit has 0.176, a z-score of $-100.57$. OA still separates real mechanisms from arbitrary subgraphs, and by a wider margin than the other methods.

## Application 2: localizing factual recall

**The problem.** Causal tracing (Meng et al., 2022) asks which layers store a fact like "The Eiffel Tower is located in the city of Paris". It corrupts the subject tokens, then restores one component's clean value and measures how much of the correct probability comes back. The corruption is Gaussian noise $Z \sim \mathcal{N}(0, 9\Sigma)$ added to the subject's token embeddings, where $\Sigma$ is a diagonal matrix of per-neuron embedding variances.

Noise is a partial ablation and a heavy spoofer: it pushes embeddings off the training manifold. The paper swaps it for OA: replace the subject token embeddings with a learned constant $a_A$ (**OA tracing**, OAT, against Gaussian-noise tracing, GNT).

The score is the **average indirect effect**, the fraction of the corruption-induced probability drop that restoring $A$ recovers:

$$
\text{AIE}(A) := \min\!\left(0,\; 1 - \frac{\mathbb{E}\,\max\!\big(0,\, [M(X)]_Y - [M_A(\xi(X), A(X))]_Y\big)}{\mathbb{E}[M(X)]_Y - \mathbb{E}[M(\xi(X))]_Y}\right)
$$

Here $[M(x)]_y$ is the probability the model assigns to the correct answer $y$, $\xi$ is the corruption, and $M_A(\xi(X), A(X))$ runs the corrupted input but restores $A$'s clean value. (The equation is printed with "min" in the paper; since the text describes AIE as a nonnegative proportion recovered, I read that as a typo for max.) The numerator is itself an ablation loss gap with a particular loss, which is why OA plugs in naturally.

![Figure 2 from Li and Janson (2024): AIE of attention and MLP layers in GPT-2 XL under Gaussian-noise tracing versus optimal-ablation tracing, with sliding windows (top) and single layers (bottom)](/imgs/blogs/optimal-ablation-for-interpretability-fig2.webp)

On GPT-2 XL, trained on 60% of the prompts and evaluated on 40%, the full model puts 30.6% on the right answer, GNT drops it to 12.3% and OAT to 8.7%. OAT removes more subject information with a constant than GNT does with noise.

The localization becomes sharper:

| Measurement (last-token attention, window of 5) | GNT | OAT |
| --- | --- | --- |
| Peak AIE (window over layers 30–34) | 20.2% | 42.6% |
| Average AIE, windows centered on layers 15–23 | 7.0% | 1.7% |

GNT spreads a small positive effect over most layers. OAT shows a few layers with large effects and the rest near zero. It can also see single layers: attention layers 30, 32 and 34 at the last token each exceed 8% AIE alone, where Meng et al. needed sliding windows because single-layer GNT effects were too small to read.

**When it fails.** The authors note that it is still unproven whether sharper localization helps downstream uses such as surgical model edits.

## Application 3: OCA lens for latent prediction

**The problem.** Logit lens applies the unembedding to an intermediate residual, which equals zero-ablating every later layer: a poor substitute, since early residuals do not speak the final layer's "language". Tuned lens (Belrose et al., 2023) learns a per-layer affine map $f_i(\ell_i) = W_i \ell_i + b_i$ that translates the last-token residual $\ell_i$ after layer $i$ into a guess for the final residual $\ell_N$. If that guess produces low loss, then $\ell_i$ already contains what the model needs.

**The mechanism.** **Optimal Constant Attention (OCA) lens** asks the same question differently. Run layers $i+1$ to $N$ normally, but replace each attention layer's output at the last token position with a learned constant $\hat a_k$. MLPs still run. Since attention is the only way information moves between token positions, this cuts off any new context arriving after layer $i$, while letting the model's own MLPs do the translation.

![Tuned lens learns an affine translation from layer i to the final layer; OCA lens keeps the model's own later MLPs and only replaces later attention outputs at the last token with learned constants](/imgs/blogs/optimal-ablation-for-interpretability-4.webp)

The constants are trained to match the full model:

$$
\hat a = (\hat a_{i+1}, \dots, \hat a_N) = \arg\min_{a}\; \mathbb{E}_X\, \mathcal{L}\big(M_\text{OCA}(X),\, M(X)\big)
$$

Each $\hat a_k \in \mathbb{R}^{d_\text{model}}$, so OCA lens has $O(N d_\text{model})$ parameters against tuned lens's $O(d_\text{model}^2)$. For GPT-2 XL that is 48 layers times 1,600 dimensions at most, about 77k numbers, against roughly 2.6 million for one ${1600 \times 1600}$ matrix. (The arithmetic is mine from the public GPT-2 XL dimensions; the asymptotic comparison is the paper's.)

![Figure 3 from Li and Janson (2024): left, prediction loss by layer for tuned lens and ablation-based lenses on GPT-2 XL; middle and right, magnitude correlation and direction similarity under basis-aligned projections](/imgs/blogs/optimal-ablation-for-interpretability-fig3.webp)

Three results, in order of the panels:

- **Prediction.** OCA lens reaches a predictive loss below 0.01 around layer 35 of GPT-2 XL. Tuned lens never gets there, even at the last layer. The panel also plots mean- and resample-ablation versions of the same lens as baselines, so the comparison isolates the choice of replacement value.
- **Causal faithfulness.** A good lens should react to an intervention on $\ell_i$ the way the real model does. Using a projection that removes one direction of $\ell_i$, OCA lens scores higher on both magnitude correlation and direction similarity at every layer, and the appendix shows the same for other interventions.
- **Elicitation.** Halawi et al. (2024) found that intermediate layers can be *more* accurate than the output when the prompt contains false demonstrations. OCA lens increases the best-layer accuracy boost on 12 of 15 datasets with true demonstrations and 11 of 15 with false ones.

![Figure 4 from Li and Janson (2024): calibrated elicitation accuracy by layer for the 15-dataset average, DBPedia and MRPC, comparing OCA lens (dashed) and tuned lens (solid) under true and false demonstrations](/imgs/blogs/optimal-ablation-for-interpretability-fig4.webp)

The headline case is DBPedia topic classification, in the middle panel above. The elicitation boost rises from 2.9% to 18.0% with true demonstrations and from 19.2% to 28.8% with false ones. Training takes about 3–5 hours, similar to tuned lens.

## Results at a glance

| Application | Model | Baseline | OA-based result |
| --- | --- | --- | --- |
| Single-component importance (IOI) | GPT-2 small | Zero / mean / resample | OA gap is 11.1% / 33.0% / 17.7% of theirs (median); rank corr. with CF 0.907 |
| Circuit discovery (IOI, 1,000 edges) | GPT-2 small | CF / mean / resample | 32% / 62% / 88% lower $\Delta$ |
| Circuit search (IOI, CF) | GPT-2 small | ACDC | 385 edges, $\Delta_\text{CF} = {0.220}$; 52% fewer edges at equal loss |
| Factual recall | GPT-2 XL | Gaussian-noise tracing | Peak AIE 42.6% vs 20.2%; unimportant layers 1.7% vs 7.0% |
| Latent prediction | GPT-2 family | Tuned lens | Loss under 0.01 by layer 35; better faithfulness at all layers |
| Elicitation | GPT-2 XL | Tuned lens | Boost improved on 12/15 (true) and 11/15 (false) datasets |

**What is load-bearing and might not transfer.** Every experiment uses GPT-2 models and well-studied synthetic subtasks. The circuit results depend on IOI and Greater-Than, where a manual circuit exists to compare against. The 1,000-edge comparison uses KL to the full model as the loss. It is my extrapolation, not the paper's claim, that the ordering will hold on modern models with gated MLPs and many more heads; the proof will, but the size of the spoofing share is an empirical question.

## Critique

**What is strong.** A real theorem, short enough to check in your head. The deletion/spoofing split explains *why* CF works (parallel tokens limit spoofing) and why it cannot generalize. OA improves three genuinely different tools. And one position-agnostic constant per component works on prompts of any shape, which CF and resample ablation cannot.

**What is weak or under-tested.**

- **OA can recruit dormant mechanisms.** The authors admit that the extra freedom of a learned constant may let a circuit use pathways the model does not normally use for the task. The random-circuit z-scores argue against arbitrary success, but they do not rule out a *plausible-looking but unused* mechanism.
- **"Importance" is still undefined.** The paper says so in its limitations: there is no proof that the minimal gap is *the* right aggregation of causal effects, only that it is the one free of excess spoofing.
- **Optimization noise is invisible.** $a^*$ is found by SGD with no closed form. The paper does not report how much $\Delta_\text{opt}$ varies across seeds or initializations, which matters when ranking hundreds of components.

**The missing ablation.** I would want OA compared against learned replacements that leak a controlled number of bits about $x$. That would show whether OA's lower gap comes from removing spoofing or partly from optimizing harder than the baselines.

**What would change my mind.** If OA-discovered circuits, tested on held-out prompts or with a downstream edit, turned out to rely on edges that CF-discovered circuits never use and that do not matter under natural inputs, I would read OA's lower loss as the optimizer exploiting the model rather than measuring it.

For more on how circuit-discovery results are judged, see [Edge Pruning for transformer circuits](/blog/paper-reading/ai-interpretability/finding-transformer-circuits-with-edge-pruning), the concurrent work the paper benchmarks against in its appendix, and [Hypothesis testing the circuit hypothesis](/blog/paper-reading/ai-interpretability/hypothesis-testing-the-circuit-hypothesis-in-llms), which asks what a circuit should satisfy. For background on what circuits are, see [language model circuits in the neuron basis](/blog/machine-learning/ai-interpretability/language-model-circuits-neuron-basis).

## References

1. Li, M. and Janson, L. (2024). *Optimal Ablation for Interpretability.* NeurIPS 2024 (spotlight). [arXiv:2409.09951](https://arxiv.org/abs/2409.09951)
2. Wang, K. et al. (2022). *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small.* (IOI task and counterfactual mean ablation.)
3. Hanna, M., Liu, O. and Variengien, A. (2023). *How does GPT-2 compute greater-than?* (Greater-Than task and counterfactual ablation.)
4. Conmy, A. et al. (2023). *Towards Automated Circuit Discovery for Mechanistic Interpretability.* (ACDC.)
5. Meng, K. et al. (2022). *Locating and Editing Factual Associations in GPT.* (Causal tracing.)
6. Belrose, N. et al. (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens.*
7. Halawi, D., Denain, J.-S. and Steinhardt, J. (2024). *Overthinking the Truth: Understanding how Language Models Process False Demonstrations.*
8. On this site: [Finding transformer circuits with Edge Pruning](/blog/paper-reading/ai-interpretability/finding-transformer-circuits-with-edge-pruning), [Hypothesis testing the circuit hypothesis in LLMs](/blog/paper-reading/ai-interpretability/hypothesis-testing-the-circuit-hypothesis-in-llms).

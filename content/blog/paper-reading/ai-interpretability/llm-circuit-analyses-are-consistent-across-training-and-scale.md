---
title: "LLM Circuit Analyses Are Consistent Across Training and Scale"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How Tigges et al. tracked four circuits across 154 checkpoints and six Pythia sizes, and why the algorithm survives even when the attention heads running it are replaced."
category: "paper-reading"
subcategory: "AI Interpretability"
tags:
  - paper-reading
  - large-language-model
  - circuit-analysis
  - mechanistic-interpretability
  - training-dynamics
  - scaling
  - pythia
  - eap-ig
  - neurips-2024-poster
  - neurips-2024
author: "Hiep Tran"
featured: true
readTime: 21
image: "/imgs/blogs/llm-circuit-analyses-are-consistent-across-training-and-scale-1.webp"
excerpt: "Most circuit papers study one frozen checkpoint. Tigges, Hanna, Yu and Biderman re-find four circuits at every one of Pythia's 154 checkpoints and show that the algorithm stays put while the heads that implement it come and go."
paper:
  title: "LLM Circuit Analyses Are Consistent Across Training and Scale"
  authors: "Curt Tigges, Michael Hanna, Qinan Yu, Stella Biderman"
  venue: "NeurIPS 2024"
  url: "https://arxiv.org/abs/2407.10827"
---

> [!tldr]
> **Claim.** A circuit found in a small model at the end of pre-training is a reasonable guide to the same task in a bigger model, and in the same model earlier or later in training.
>
> **Why it matters.** Almost every circuit paper studies one frozen snapshot, while deployed models keep training and come in many sizes. If circuits did not transfer, every result would need redoing per checkpoint.
>
> **Most surprising result.** Individual attention heads gain and lose their jobs late in training (a Pythia-160m name-mover head quits at about 30B tokens), yet the IOI algorithm's step-level metrics stay flat and task performance never drops.
>
> **Where it is weak.** Four easy tasks, one model family, a detailed algorithm check on only one of the four circuits, and no models above 2.8B in the circuit analysis.

Mechanistic interpretability has a quiet sampling problem. Whether the circuit is built from [features in superposition](/blog/machine-learning/ai-interpretability/what-is-superposition) or [raw neurons](/blog/machine-learning/ai-interpretability/language-model-circuits-neuron-basis), we reverse-engineer a circuit in GPT-2 small at its final checkpoint, publish it, and implicitly assume the finding means something for other models and other moments in training. Nobody had checked that assumption on a real decoder-only language model over a real pre-training run.

Tigges, Hanna, Yu and Biderman check it. They take the Pythia suite, which publishes 154 checkpoints for each model size, and re-run circuit discovery at every checkpoint for four classic tasks. The answer is mostly reassuring, with a twist: the *algorithm* is stable, but the *components* implementing it are not.

## The measurement pipeline

The paper is one recipe applied thousands of times. For every model and checkpoint, it scores every edge in the computational graph, cuts the smallest subgraph that still does the task, and then measures that subgraph three different ways.

![The study's pipeline: score edges with EAP-IG, binary-search a minimal faithful circuit, then measure components, algorithm and graph](/imgs/blogs/llm-circuit-analyses-are-consistent-across-training-and-scale-1.webp)

The diagram above is the mental model. The left half is the expensive, automated part; the right half is three questions, one per results section.

| Question | Measurement | Paper section |
| --- | --- | --- |
| When do task abilities and the heads behind them appear? | Behavioral metric + component scores per checkpoint | Section 3 |
| Does the algorithm change when heads change? | Path-patching ratios on the IOI circuit | Section 4 |
| How much does the circuit's node set churn, and how big is it? | EWMA Jaccard similarity, node counts | Section 5 |

### The models and the clock

Pythia is a family of autoregressive models trained on the same data in the same order. Each size has 154 checkpoints: steps 0, 1, 2, 4, up to 512 (11 log-spaced early checkpoints), then every 1,000 steps from 1,000 to 143,000. With a batch of 2.1M tokens, 143,000 steps is about 300B tokens.

The circuit analysis covers Pythia-70m, 160m, 410m, 1.4b and 2.8b. Behavioral curves add 6.9b and 12b, which were dropped from circuit finding for compute reasons.

## The four tasks and their metrics

**The problem.** A circuit is defined relative to a task, and a task needs a continuous number that says how well the model is doing. Accuracy is too coarse to see gradual learning, so the paper uses a contrast between the right and wrong answer.

**The intuition.** Think of a judge comparing two candidates. We do not care how much the model likes "Mary" in absolute terms; we care how much more it likes "Mary" than the distractor "John".

### Logit difference

For the indirect object identification (IOI) task, the prompt is "When John and Mary went to the store, John gave a drink to", and the right continuation is Mary. The metric is

$$
\mathrm{LD}(x) = \ell_{\text{IO}}(x) - \ell_{\text{S}}(x)
$$

where $x$ is the prompt, $\ell_{\text{IO}}(x)$ is the pre-softmax logit the model assigns to the indirect object (Mary), and $\ell_{\text{S}}(x)$ is the logit for the repeated subject (John). A positive value means the model prefers the correct name. The Gendered-Pronoun task uses the same shape: the logit of "he" minus "she" (or the reverse) after a sentence like "So Paul is such a good cook, isn't".

### Probability difference

Greater-Than and subject-verb agreement (SVA) have many valid answers, so a single logit pair does not fit. For a Greater-Than prompt $s$ = "The war lasted from the year 1732 to the year 17", the metric is

$$
\mathrm{PD}(s) = \sum_{y=33}^{99} p(y \mid s) \;-\; \sum_{y=00}^{32} p(y \mid s)
$$

where $y$ ranges over two-digit year completions and $p(y \mid s)$ is the model's next-token probability. It rewards mass on years after 32 and penalizes mass on years at or before it.

**Tiny example.** Suppose the model puts 0.70 total probability on 33 to 99 and 0.20 on 00 to 32 (the remaining 0.10 goes to other tokens). Then $\mathrm{PD} = 0.70 - 0.20 = 0.50$. A model that guessed uniformly over the 100 two-digit years would score ${0.67 - 0.33 = 0.34}$, so the metric has a non-zero chance baseline worth remembering.

**When it fails.** The authors prefer logit differences because they expose negative components better; probability differences saturate near 1 and flatten late-training changes.

| Task | Clean example | Corrupted example | Metric | Size |
| --- | --- | --- | --- | --- |
| IOI | "...John gave a drink to" (want Mary) | "...Mary gave a drink to" | logit diff | 70 prompts |
| Gendered-Pronoun | "So Paul is such a good cook, isn't" (want he) | name swapped to a female name | logit diff | 70 prompts |
| Greater-Than | "from the year 1732 to the year 17" | start year ends in 01 | prob diff | 200 prompts |
| SVA | "The keys on the cabinet" (want are) | "The key on the cabinet" | prob diff | 200 prompts |

## What a circuit is, and how to test one

**The problem.** A model has thousands of attention heads and MLPs, connected by many more edges. We want the small part that actually does the task, and we want proof that it is the real part, not a story we told ourselves.

**The analogy.** Imagine a factory floor where you suspect only a few machines build a given product. You keep those machines running on the real order, and you feed every other machine the inputs of a different order. If the right product still comes out, your few machines are the ones doing the work.

**The mechanism.** A circuit here is a set of edges between nodes, where nodes are attention heads and MLPs. To test it, run the model on the clean prompt, but for every edge *outside* the circuit, substitute the activation it would have carried on the corrupted prompt. If the output still favors Mary, the circuit is faithful.

![The faithfulness test: the circuit keeps its clean activations while every other edge carries the corrupted prompt's activations](/imgs/blogs/llm-circuit-analyses-are-consistent-across-training-and-scale-2.webp)

This is a demanding test. Circuits are often under 5% of the model's edges, so the test corrupts the overwhelming majority of the network and asks the remaining sliver to carry the task alone.

### The faithfulness threshold

The paper cuts the minimal circuit that keeps at least 80% of the full model's performance. Written as a ratio,

$$
F(C) = \frac{m(C)}{m(M)} \;\ge\; 0.8
$$

where $C$ is the candidate edge set, $M$ is the full model, and $m(\cdot)$ is the task metric (logit or probability difference) measured with everything outside the given set corrupted. The paper states the 80% criterion but does not spell out whether $m$ is baseline-subtracted against the fully corrupted run; the ratio above is my simplified reading.

## Scoring edges with EAP-IG

**The problem.** Testing every edge by actually patching it costs one forward pass per edge, and the number of edges grows with model size. Doing that at 154 checkpoints for five models is out of reach. The paper needs a method whose cost does not depend on model size.

**The analogy.** Instead of unplugging every wire in a building to see which lights go out, you measure the current in every wire once and estimate the effect of unplugging each from that single reading.

**The mechanism.** Edge attribution patching (EAP) uses a first-order Taylor approximation: the change in the metric from corrupting an edge is about the activation change times the gradient. EAP with integrated gradients (EAP-IG) averages the gradient along a straight path from corrupted to clean activations, which tames the error when the metric is far from linear. The authors pick EAP-IG because it finds more faithful circuits with fewer edges than plain EAP.

The paper itself does not write the formula; the following is the standard EAP-IG score from Hanna et al.'s earlier work, which this paper cites as its method:

$$
s(u \to v) = \big(z_u - z'_u\big)^{\top} \, \frac{1}{K} \sum_{k=1}^{K} \frac{\partial\, L\!\left(z' + \tfrac{k}{K}(z - z')\right)}{\partial z_v}
$$

Here $u \to v$ is an edge from upstream node $u$ to downstream node $v$; $z_u \in \mathbb{R}^{d}$ is $u$'s output on the clean prompt and $z'_u$ on the corrupted one, with $d$ the residual stream width; $L$ is the task loss; $z'$ and $z$ are the corrupted and clean inputs being interpolated; and $K$ is the number of interpolation steps. The dot product turns a $d$-dimensional activation change into one scalar importance.

**Tiny example.** Take $d = 2$, $z_u - z'_u = (1.0, -0.5)$, and suppose the averaged gradient at $v$ is $(0.4, 0.2)$. The score is ${1.0 \times 0.4 + (-0.5) \times 0.2 = 0.3}$. An edge whose activation does not change between clean and corrupted prompts scores zero no matter how large the gradient is.

### Cutting the circuit by binary search

With every edge scored, the circuit is the top-$n$ edges by absolute score, and the paper binary-searches $n$ between 1 edge and 5% of all edges for the smallest $n$ with $F \ge 0.8$.

```python
def minimal_circuit(edges_sorted, faithfulness, lo=1, hi=None):
    """edges_sorted: edges ranked by |EAP-IG score|, highest first.
    faithfulness(edge_set) -> m(C) / m(M), measured with the rest corrupted."""
    hi = hi or int(0.05 * len(edges_sorted))
    best = hi
    while lo <= hi:
        n = (lo + hi) // 2
        if faithfulness(edges_sorted[:n]) >= 0.8:
            best, hi = n, n - 1   # faithful: try smaller
        else:
            lo = n + 1            # not faithful: need more edges
    return edges_sorted[:best]

# With 20,000 edges the search space is 1..1,000, so about 10 faithfulness
# evaluations (each two forward passes) replace 1,000 one-at-a-time checks.
```

**When it fails.** Binary search assumes faithfulness grows roughly monotonically with $n$; a negative-effect edge entering the top set can break that. The authors also flag the deeper issue: 80% faithfulness says the circuit is *sufficient* for most of the behavior, not that it is *complete*. A redundant backup path left outside the circuit is invisible to this test.

## Result 1: abilities arrive at the same token count

The first experiment is purely behavioral: plot each task's metric for every model across training.

![Figure 1 from Tigges et al. (2024): task metrics for Pythia-70m to 12b over tokens seen](/imgs/blogs/llm-circuit-analyses-are-consistent-across-training-and-scale-fig1.webp)

Three patterns show up across all four tasks.

- **Same endpoint.** Every model except Pythia-70m ends training at roughly the same performance. These tasks are simple enough that scale does not buy more accuracy.
- **No unlearning.** Once learned, performance is essentially non-decreasing. The one clear drop (Pythia-70m late in SVA) is the weakest model.
- **A speed ceiling.** Past a certain size, bigger models do not learn the task sooner. Gendered-Pronoun shows the same rate for every model from 70M up, Greater-Than from 160M up. On IOI, 410M to 2.8B learn fastest, while 6.9B and 12B have curves closer to 160m.

The ceiling runs against the usual "bigger learns faster" expectation (compare the debate over [emergent abilities](/blog/machine-learning/scaling-laws/emergent-abilities-scaling)). Appendix E finds the same asymptote on SciQ, PIQA, WinoGrande and ARC Easy, using accuracy data logged during the original Pythia runs.

## Result 2: the components arrive with the ability

**The problem.** A behavior curve says *when* a model learns a task, not *what* it built to learn it. To connect the two, the paper scores each in-circuit attention head for four known functional roles at every checkpoint.

| Component | What it does | Supports |
| --- | --- | --- |
| Induction head | On $[A][B] \dots [A]$, attends to and boosts $[B]$ | IOI, Greater-Than |
| Successor head | Sees "11" or "Thursday", boosts "12" or "Friday" | Greater-Than |
| Copy suppression head | Lowers the logit of a repeated token already predicted | IOI |
| Name-mover head | Attends to the correct name and copies it to the output | IOI |

### Scoring a head's role: the succession score

Each role has its own metric from prior work. The succession score shows the general pattern: simulate what the head would write if it attended perfectly to a token, then read that through the unembedding.

$$
M = W_U \, W_{OV} \, \mathrm{MLP}_0\!\left(W_E\right)
$$

Here $W_E$ is the token embedding matrix, $\mathrm{MLP}_0$ is the first MLP layer (applied to every embedding), $W_{OV}$ is the head's combined output-value matrix, and $W_U$ is the unembedding. $M$ is a $|V| \times |V|$ matrix over the vocabulary $V$: entry $M[x][y]$ says how much the head boosts output token $y$ when it attends to input token $x$. A pair such as (Tuesday, Wednesday) counts as a hit if $M[x][y]$ beats $M[x][y']$ for every other candidate $y'$ in the dataset, and the score is the hit rate.

**Tiny example.** Restrict to days of the week. If the row for "Tuesday" gives Wednesday 2.1, Thursday 1.4, Monday 0.9, the head passes that pair. If 5 of 7 day pairs pass, that head's succession score on days is about 0.71.

### What the scores show

For each model and checkpoint, the paper sums each role's score over all in-circuit heads and normalizes across checkpoints.

![Figure 2 from Tigges et al. (2024): emergence of successor, induction, copy suppression and name-mover behavior across model sizes](/imgs/blogs/llm-circuit-analyses-are-consistent-across-training-and-scale-fig2.webp)

Induction heads appear just after about ${2 \times 10^9}$ tokens in most models, replicating Olsson et al., and Greater-Than performance follows immediately. Successor heads emerge in the same window. Name movers appear between ${2}$ and ${8 \times 10^9}$ tokens across sizes, during or just before IOI behavior appears; copy suppression heads come in on a similar timescale but at varied speeds.

The authors suggest that how fast these heads can form sets how fast the task can be learned. That is correlational; no experiment intervenes on formation speed.

One oddity sets up the next section. Successor behavior in the circuit *decreases* late in training, while Greater-Than performance does not.

## Result 3: heads come and go

To see what is happening inside, the paper zooms into Pythia-160m and plots the top five heads per role over the full 300B tokens.

![Figure 3 from Tigges et al. (2024): per-head role scores in Pythia-160m; heads lose their function and others take over](/imgs/blogs/llm-circuit-analyses-are-consistent-across-training-and-scale-fig3.webp)

The identity of the heads is not constant. Head 4.6 (layer 4, head 6) acquires name-mover behavior around ${4 \times 10^9}$ tokens and abruptly loses it around ${3 \times 10^{10}}$. The main successor head 5.9 fades toward the end of training, at exactly the time head 11.0 shows more successor behavior.

![Roles persist while the heads filling them rotate, with task performance flat throughout](/imgs/blogs/llm-circuit-analyses-are-consistent-across-training-and-scale-3.webp)

Think of the role as a job title and the head as the employee: employees leave and get replaced, the job description stays, and the output of the department does not dip. Earlier studies, trained on at most about 50M tokens, missed this.

The authors suggest self-repair (heads that compensate when another is ablated) may drive the hand-off, but they leave the actual "load-balancing" mechanism open. That gap matters: we see the swap, we do not see why it happens.

## Result 4: the algorithm survives the swaps

**The problem.** If heads change, maybe the whole algorithm changes too, and the circuit we wrote down at step 143,000 describes a different computation from the one at step 20,000. The paper tests this on IOI, the best-characterized of the four circuits.

**The analogy.** A relay race can swap runners without changing the order of the legs. The question is whether the legs (the algorithmic steps) still carry the same share of the race.

### The three-step IOI algorithm

First, the authors reverse-engineer Pythia-160m's IOI circuit at the end of training, following Wang et al.'s original procedure.

![Figure 4 from Tigges et al. (2024): A is Pythia-160m's final IOI circuit; B to D track each step's share of the effect across training and model size](/imgs/blogs/llm-circuit-analyses-are-consistent-across-training-and-scale-fig4.webp)

Panel A has three logical steps, read backwards from the logits.

1. **Name movers and copy suppression heads** write directly to the output: name movers copy the indirect object; copy suppression heads push down the repeated name.
2. **S-inhibition heads** send token and position signals that tell the name movers which name *not* to attend to.
3. **Induction and duplicate-token heads** detect that "John" appeared twice, feeding the S-inhibition heads.

![The three IOI steps as layers, each paired with the stability metric that tracks it](/imgs/blogs/llm-circuit-analyses-are-consistent-across-training-and-scale-4.webp)

One Pythia-specific twist: in the original GPT-2 IOI circuit, copy suppression heads *hurt* the task by suppressing the correct name. In Pythia they help, because both names are already highly predicted by the time the signal reaches them, and they suppress the more repeated one, which is the wrong answer.

### The path-patching ratio

Each step gets a score built from path patching. You change a sender set $H$ to its corrupted activation, let that change flow only through a receiver set $R$, and measure the effect on the logit difference.

$$
r(H \to R) = \frac{\sum_{h \in H} \left|\Delta \mathrm{LD}_{h \to R}\right|}{\sum_{h \in \mathcal{U}(R)} \left|\Delta \mathrm{LD}_{h \to R}\right|}
$$

where $H$ is the set of heads classified into the target role at that checkpoint, $R$ is the downstream receiver set (the logits, the name movers, or the S-inhibition heads), $\mathcal{U}(R)$ is every head upstream of $R$, and $\Delta \mathrm{LD}_{h \to R}$ is the change in logit difference when only head $h$'s path into $R$ is corrupted. A ratio near 1 says the known role accounts for nearly all the effect at that step.

| Panel | Senders $H$ | Receivers $R$ | What a drop would mean |
| --- | --- | --- | --- |
| B | Name movers + copy suppression | Logits (direct effect) | New, unclassified heads now write the answer |
| C | S-inhibition heads | Name movers | Name movers get their routing signal elsewhere |
| D | Induction + duplicate-token heads | S-inhibition heads | Duplicate detection moved to unknown heads |

A head joins $H$ only if it scores above 10% on its role test and has a negative effect on logit difference when ablated. S-inhibition heads also have to pass Wang et al.'s position-signal test.

**Tiny example.** Say five heads upstream of the name movers have absolute effects 0.9, 0.6, 0.3, 0.1, 0.1, and only the first two pass the S-inhibition test. Then $r = (0.9 + 0.6) / 2.0 = 0.75$. If a later checkpoint swaps in a new S-inhibition head that also passes the test, $r$ stays near 0.75 even though the head IDs changed: the metric tracks the *role*, not the *occupant*.

### What it shows

Panels B to D are flat once the circuit has formed, for every model from 160m to 2.8b. No model shows a dramatic shift in any metric when components turn over. Scores are generally above 50%, and the direct-effect step (name movers plus copy suppression) stays above 70%.

There is one scale difference. The Pythia-160m circuit does not use duplicate-token heads, while the larger models do. So the big picture transfers across sizes, but small details do not.

**When it fails.** The ratio is only as good as the role tests. A head doing something new but scoring 11% on an old metric would be counted as the old role, which pushes the ratio toward stability. And the check runs on one circuit; Greater-Than, SVA and Gendered-Pronoun get component tracking but no step-level algorithm audit.

## Result 5: the subgraph churns, and grows with scale

The last section steps back from roles and asks how much the raw node set changes from checkpoint to checkpoint.

### EWMA Jaccard similarity

**The problem.** Raw node overlap between adjacent checkpoints is noisy; a single swapped head drops it sharply. The authors want the long-run trend.

**The math.** Jaccard similarity between two node sets $A$ and $B$ is

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

and the per-checkpoint similarity $x_t$ is smoothed with an exponentially weighted moving average

$$
\hat{x}_t = \alpha\, \hat{x}_{t-1} + (1 - \alpha)\, x_t, \qquad \alpha = 0.5
$$

where $t$ indexes checkpoints, $x_t$ is the Jaccard similarity between the circuit at checkpoint $t$ and earlier circuits, and $\alpha$ is the decay factor. With $\alpha = 0.5$, a value from four checkpoints back carries weight ${0.5^4} \approx 0.06$, so the smoother remembers only a handful of steps.

**Tiny example.** Circuit A has nodes {a, b, c, d}, circuit B has {b, c, d, e, f}. The intersection has 3 nodes, the union 6, so $J = 0.5$. If the running average was 0.8 and today's value is 0.5, the new average is ${0.5 \times 0.8 + 0.5 \times 0.5 = 0.65}$.

![Figure 5 from Tigges et al. (2024): EWMA Jaccard similarity of circuit node sets over training, per task and model](/imgs/blogs/llm-circuit-analyses-are-consistent-across-training-and-scale-fig5.webp)

Larger models have higher average similarity and fewer sharp drops; Pythia-70m and 160m are the most volatile. Gendered-Pronoun shows significant changes even late in training. In the appendix, similarity to the *final* circuit trends upward over training while still fluctuating at every checkpoint: the circuit drifts toward its endpoint while parts keep swapping.

### Circuit size tracks model size

Averaging node counts over checkpoints and correlating with model size gives a Pearson $r$ of 0.72 for IOI and SVA, 0.9 for Greater-Than, and 0.6 for Gendered-Pronoun. Larger models do not solve the task with more specialized heads; they spread the same role across more heads.

**When it fails.** Five model sizes is a small sample for a correlation, and the paper reports no confidence intervals. An $r$ of 0.6 on five points is weak evidence on its own. The paper also notes circuit size can fluctuate a lot within a model, with no clear pattern by model or task.

## Results at a glance

| Finding | Evidence | Models | Strength |
| --- | --- | --- | --- |
| Tasks are learned at similar token counts across sizes | Behavior curves, 4 tasks + 4 benchmarks | 70m to 12b | Strong, direct |
| Learning speed hits a ceiling with scale | Same curves; 6.9b and 12b no faster on IOI | 70m to 12b | Strong, direct |
| Induction heads at about ${2 \times 10^9}$ tokens, name movers at ${2}$ to ${8 \times 10^9}$ | Component scores over in-circuit heads | 70m to 2.8b | Strong |
| Heads lose and gain roles late in training | Per-head scores | 160m only | Clear, one model |
| IOI algorithm stable after formation | Path-patching ratios, B above 70% | 160m to 2.8b | Good, one circuit |
| Bigger models, more circuit nodes | Pearson $r$ from 0.6 to 0.9 | 70m to 2.8b | Suggestive, 5 points |

### What is load-bearing and might not transfer

Pythia is unusually controlled: every size sees the same data in the same order with the same recipe. That is exactly what makes "same token count across sizes" testable, and exactly what might not hold across model families with different data mixes or learning-rate schedules. The authors tie the timing result to the quantization model of neural scaling (models learn sub-tasks in order of how often they occur in the data), which predicts identical timing only when the data distribution is shared.

The tasks are also easy enough that a 160M model solves them. For hard tasks with many possible algorithms, the authors themselves expect less consistency.

## Critique

**What is strong.** The sweep is the contribution: circuits re-found at every checkpoint over 300B tokens, made feasible by a size-independent discovery method (EAP-IG). Separating "which heads" from "which algorithm" is the conceptual move that keeps the head-churn result from being read as instability.

**What is weak.**

- The algorithm-stability claim rests on one circuit (IOI) and on role tests with a 10% threshold. Thresholded role labels bias the metric toward stability, as noted above.
- The faithfulness criterion (80%) guarantees sufficiency, not completeness. Backup heads outside the circuit are exactly what self-repair would use, and they are invisible here.
- Circuit finding stops at 2.8B. The "consistent across scale" claim for circuits spans a 40× range of small models; the 6.9B and 12B evidence is behavioral only.
- The mechanism for the head hand-off is named (self-repair, load balancing) but not tested. An ablation at the moment of a swap would have been the natural experiment.

**What would change my mind.** If someone repeated this on a second suite with intermediate checkpoints and a different data order (for example OLMo, which the authors excluded for missing checkpoints) and found the same tasks emerging at clearly different token counts across sizes, I would read the timing result as a property of Pythia's shared data order rather than of language models. Likewise, finding a task whose circuit switches algorithm mid-training without a behavior dip would break the main claim.

My own extrapolation, not the paper's: this is a good argument for doing interpretability on small, fully trained proxies, but a weak argument for skipping verification on fine-tuned models, since fine-tuning applies a very different data distribution than continued pre-training does.

## References

1. Curt Tigges, Michael Hanna, Qinan Yu, Stella Biderman. "LLM Circuit Analyses Are Consistent Across Training and Scale." NeurIPS 2024. [arXiv:2407.10827](https://arxiv.org/abs/2407.10827)
2. Biderman et al. "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling." ICML 2023. [arXiv:2304.01373](https://arxiv.org/abs/2304.01373)
3. Wang et al. "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small." ICLR 2023. [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
4. Hanna, Pezzelle, Belinkov. "Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms." COLM 2024. [arXiv:2403.17806](https://arxiv.org/abs/2403.17806)
5. Olsson et al. "In-context Learning and Induction Heads." Transformer Circuits Thread, 2022. [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)

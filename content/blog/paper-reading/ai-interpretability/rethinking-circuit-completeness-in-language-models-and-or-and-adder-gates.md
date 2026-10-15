---
title: "Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "Why circuit discovery keeps missing backup paths, how splitting a circuit into AND, OR and ADDER gates explains it, and how running noising and denoising together recovers a circuit that is both faithful and complete."
tags: ["paper-reading", "circuit-discovery", "circuit-completeness", "logic-gates", "activation-patching", "mechanistic-interpretability", "gpt-2", "neurips-2025-poster"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 22
paper:
  title: "Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates"
  authors: "Hang Chen, Jiaying Zhu, Xinyu Yang, Wenya Wang"
  venue: "NeurIPS 2025"
  url: "https://arxiv.org/abs/2505.10039"
---

> [!tldr]
> - **The claim.** Circuits found by today's discovery methods are incomplete because they only partially detect **OR gates**: places where any one of several upstream paths is enough to drive a downstream component.
> - **The mechanism.** Every receiver node behaves like an **AND**, **OR**, or **ADDER** gate over its incoming edges. Noising-based intervention (Ns) sees AND and ADDER gates; denoising-based intervention (Dn) sees OR and ADDER gates. Run both and the set difference of the two circuits sorts every edge into its gate type.
> - **Why it matters.** The fix plugs into greedy search (ACDC), linear estimation (EAP) and differentiable masks (Edge Pruning) at most doubling their cost, and it makes circuits both more complete and more stable across random seeds.
> - **Most surprising result.** On GPT-2 small, EAP under plain noising finds **zero** OR edges by construction. Removing an Ns+Dn circuit of 5,000 edges drops IOI accuracy to 0.09, versus 0.29 when removing the Ns-only circuit of the same size.
> - **Where it is thin.** One small model, three toy-scale tasks, binary gate definitions laid over a continuous network, and a "same number of edges in both runs" alignment rule that is fitted rather than derived.

## The problem: circuits that change every time you look

Circuit discovery treats a transformer's forward pass as a graph: nodes are attention heads and MLPs, and an edge $i \to j$ means component $i$ feeds component $j$. You prune edges until a small subgraph still does the task, and hope that subgraph is the mechanism.

The trouble is that two runs of the same method can return two different circuits. Change ACDC's edge order or Edge Pruning's mask initialization and edges swap in and out. A real mechanism should not depend on the seed.

The paper's diagnosis: suppose two identical, disjoint paths $C_1$ and $C_2$ are combined by an OR. Either path alone reproduces the output, so a method chasing faithfulness and sparsity keeps one and drops the other, and which one survives is an accident. The circuit is faithful but incomplete, and the incompleteness shows up as randomness.

Earlier fixes exist and each has a cost:

| Approach | What it buys | What it costs |
| --- | --- | --- |
| Try every combination of interventions | Finds every OR branch in principle | Exponential: ${2^N}$ subsets for $N$ nodes |
| Denoising-based patching | Restores OR gates quickly | Loses faithfulness, cannot label which edges are OR |
| Add a completeness term $D(G \setminus C \,\|\, G)$ to the objective | Directly rewards completeness | Not usable by greedy search, still no gate labels |

The gap this paper fills: a cheap way to recover *all* the gates and to say which edge belongs to which gate. If you have not met circuit discovery before, the post on [sparse feature circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models) covers the same patching machinery at the feature level.

## Contributions

1. Three logic gates (AND, OR, ADDER) into which any circuit decomposes, and a corollary giving the minimal edge sets for optimal faithfulness and for optimal completeness.
2. An account of which gates noising and denoising can see, and how greedy search, linear estimation and differentiable masks recover each.
3. The Ns+Dn framework: run both interventions, split the circuit into gates with set operations, at most doubling the baseline's cost.
4. A misalignment score that checks the two circuits were compared at matching granularity.
5. A look at the gates inside GPT-2 small's IOI circuit: what they connect, how much they contribute, how common each is.

## Method

### Setup: two ways to intervene

**The problem.** To decide whether an edge matters, you have to break it and watch the output. But "break" can start from two different places, and the paper's whole argument rests on that choice.

**The analogy.** A room is lit by several lamps. Noising starts with every lamp on and switches one off; denoising starts with every lamp off and switches one on. If two lamps are redundant, the first test never notices either. If two switches must both be on, the second test never notices either.

**The mechanism.** Each sample is a pair: a clean text $x$ and a corrupted text $\tilde{x}$ minimally changed to flip the answer (in IOI, the second "John" becomes "Alice"). Running both gives a clean activation $x_i$ and a corrupted activation $\tilde{x}_i$ at every component $i$. A kept edge carries its clean value; a pruned edge carries the corrupted one.

**The math.** Noising (Ns) starts from the clean run and asks which edges, when corrupted, push the output away from the model's:

$$
\arg\min_{C}\; \mathbb{E}_{(x,\tilde{x})\in T}\Big[D\big(p_G(y \mid x)\,\|\,p_C(y \mid x,\tilde{x})\big)\Big] \quad \text{s.t.}\quad 1 - \frac{|C|}{|G|} \ge s
$$

Here $G$ is the full computational graph, $C \subset G$ the candidate circuit, $T$ the task, $p_G(y \mid x)$ the model's output distribution on the clean text, and $p_C(y \mid x, \tilde{x})$ the output when every edge outside $C$ is replaced by its corrupted activation. $D$ is a distance between distributions (the paper uses KL divergence and task accuracy), $|C|/|G|$ is the fraction of edges kept, and $s$ is the required sparsity.

Denoising (Dn) mirrors it. Start from the corrupted run, patch clean activations back in, and keep the edges that restore the clean behaviour:

$$
\arg\min_{C}\; \mathbb{E}_{(x,\tilde{x})\in T}\Big[D\big(p_G(\tilde{y} \mid \tilde{x})\,\|\,p_C(\tilde{y} \mid \tilde{x},x)\big)\Big] \quad \text{s.t.}\quad 1 - \frac{|C|}{|G|} \ge s
$$

where $\tilde{y}$ is the output on the corrupted run. Almost every published method uses only the first objective.

Two evaluation metrics follow. **Faithfulness** is $D(G \,\|\, C)$: how close the circuit alone gets to the model, lower is better. **Completeness** is $D(G \setminus C \,\|\, G)$: how badly the model breaks when you delete the circuit, higher is better.

![Noising versus denoising: Ns starts all-on and corrupts one edge, Dn starts all-off and restores one edge, so each intervention is blind to a different gate type](/imgs/blogs/rethinking-circuit-completeness-in-language-models-and-or-and-adder-gates-2.webp)

### The three gates

**The problem.** Whether a single edge matters depends on what its siblings are doing, so a method that tests edges one at a time sees a different picture depending on their state.

**The mechanism.** Take a receiver node $B$ with senders $A_1, A_2, \ldots$. Write $A_i = 1$ when the sender is kept and $A_i = 0$ when it is ablated. The paper defines three patterns of $B$'s effect on the output:

$$
\text{AND: } B = A_1 \wedge A_2 \wedge \cdots \qquad \text{OR: } B = A_1 \vee A_2 \vee \cdots \qquad \text{ADDER: } B = A_1 + A_2 + \cdots
$$

- **AND.** $B$ matters only if every sender is kept. Lose one and $B$'s effect nearly vanishes.
- **OR.** $B$ matters as long as at least one sender is kept. Only losing all of them kills it.
- **ADDER.** Each sender contributes its own share. Lose one and the effect drops but survives; lose all and it is zero. $B$ takes values ${0, 1, 2, \ldots}$ rather than just on or off.

**A worked example from the paper's toy model.** Appendix D builds a one-layer transformer with model dimension 1, two heads $A_1, A_2$ whose biases are the only signal, and an activation function $m$ that implements each gate. With both biases equal to 1:

| Gate | $m(x)$ | Both kept | Only $A_1$ | Only $A_2$ | Both ablated |
| --- | --- | --- | --- | --- | --- |
| AND | $\mathrm{ReLU}(x - 1)$ | 1 | 0 | 0 | 0 |
| OR | ${1 - \mathrm{ReLU}(1 - x)}$ | 1 | 1 | 1 | 0 |
| ADDER (bias of $A_2$ set to 1.5) | $\mathrm{ReLU}(x)$ | 2.5 | 1 | 1.5 | 0 |

Trace the AND row once: both heads on gives $x = 2$, so $\mathrm{ReLU}(2 - 1) = 1$. Drop either head and $x = 1$, so $\mathrm{ReLU}(0) = 0$. One missing sender is as bad as both.

The same signature appears in the real model. The authors took each gate type found in GPT-2 small's IOI circuit, ablated one or two sender edges on the same receiver 30 times, and measured the change in KL. The averages are below.

![The three gates under ablation: AND collapses when one sender goes, OR barely moves, ADDER loses about half, using the measured change in KL from the paper's Table 5](/imgs/blogs/rethinking-circuit-completeness-in-language-models-and-or-and-adder-gates-1.webp)

A quick additivity check of my own on those numbers: the two single-edge ADDER effects are 0.32 and 0.36, which sum to 0.68, close to the 0.65 with both edges kept. That is what "each sender contributes its share" should look like. The OR row is the striking one: 0.28 with both edges, 0.27 with either alone.

**Why Ns and Dn each go blind.** Under Ns every gate starts in state 1. Flipping one AND sender to 0 kills the gate, and flipping one ADDER sender shrinks it, so both get detected. Flipping one OR sender changes nothing, because the other sender still holds it up. Under Dn every gate starts in state 0. Restoring one OR sender lights it, restoring one ADDER sender lights part of it, but restoring one AND sender does nothing because its partner is still off. Ns and Dn are blind to exactly opposite gates, and both see ADDER.

### What faithfulness and completeness each require

**The problem.** Faithfulness and completeness are usually discussed as if they pull in the same direction. With gates in hand, the paper shows they want different minimal edge sets.

**The mechanism.** Faithfulness asks the circuit alone to reproduce the output. An AND gate missing a sender produces nothing, and an ADDER gate missing a sender produces too little, so both must be complete. An OR gate reaches full effect with one sender, so sparsity trims the rest. Completeness asks the *removed* model to break. Delete one AND sender and the AND gate in the leftover model is already dead, so one edge suffices. Delete one OR sender and the leftover OR still fires, so every OR edge must go.

That is the paper's **Corollary 1**:

| Target | AND gates | OR gates | ADDER gates |
| --- | --- | --- | --- |
| Optimal faithfulness + sparsity | all edges | any one edge | all edges |
| Optimal completeness + sparsity | any one edge | all edges | all edges |

![Figure 2 from Chen et al. (2025): a toy circuit where B1 is an AND of A1 and A2, B2 is an OR of A3 and A4, and C adds B1 and B2. Panel (b) keeps one OR edge for faithfulness, panel (c) keeps one AND edge for completeness](/imgs/blogs/rethinking-circuit-completeness-in-language-models-and-or-and-adder-gates-fig1.webp)

Read the paper's toy in Figure 2 with the table beside it. Panel (a) is the full circuit: four sender edges ($A_1, A_2 \to B_1$ and $A_3, A_4 \to B_2$) plus the two ADDER edges into $C$. Panel (b), the faithfulness optimum, drops one of $A_3 \to B_2$ or $A_4 \to B_2$. Panel (c), the completeness optimum, drops one of $A_1 \to B_1$ or $A_2 \to B_1$. A circuit that wants both properties must keep every edge.

This also explains why the field settled on Ns: complete AND and ADDER gates plus partial OR gates is exactly the faithfulness optimum, which is what the field was measuring.

### Why existing methods miss OR edges

**The problem.** Corollary 1 says Ns-based methods *may* drop OR edges. How many depends on the search strategy.

**The mechanism, per family:**

- **Greedy search (ACDC).** It visits edges in a fixed order and prunes each one whose removal changes the output less than a threshold $\tau$, *keeping the pruned state* for later steps. Testing $A_1$ first: $A_2$ still holds up the OR, so $A_1$ is pruned. Testing $A_2$ next: now $A_1$ is gone, so $A_2$ looks critical and is kept. Exactly one OR edge survives, and the search order picks which.
- **Differentiable mask (Edge Pruning).** Faithfulness requires at least one OR edge and sparsity punishes the second, so the optimizer keeps one. The mask initialization picks which.
- **Linear estimation (EAP).** It scores every edge from one gradient, with *all other edges still clean*. Each OR sender is scored while its sibling holds the gate up, so both score near zero. EAP keeps neither.

![How the three method families handle one OR gate under noising: greedy search keeps one edge by search order, masks keep one by initialization, linear estimation keeps none](/imgs/blogs/rethinking-circuit-completeness-in-language-models-and-or-and-adder-gates-4.webp)

The paper verifies this on the toy model and summarizes it in Table 1. Under Dn the pattern flips: all three recover full OR and ADDER gates, greedy search and masks recover one AND edge, and EAP recovers none.

| Strategy | Method family | AND | OR | ADDER | Faithful | Complete |
| --- | --- | --- | --- | --- | --- | --- |
| Ns | greedy search | full | partial | full | yes | no |
| Ns | linear estimation | full | none | full | no | no |
| Ns | differentiable mask | full | partial | full | yes | no |
| Dn | greedy search | partial | full | full | no | yes |
| Dn | linear estimation | none | full | full | no | no |
| Dn | differentiable mask | partial | full | full | no | yes |

Note the second row. EAP under Ns is not even faithful by Corollary 1, because faithfulness needs at least one edge per OR gate.

### The Ns+Dn framework

**The problem.** We want every gate, complete, without the exponential cost of trying every combination of ablations.

**The analogy.** Go back to the lamps. Do the "all on, switch one off" survey, then the "all off, switch one on" survey. A lamp that matters in the first survey only is part of an AND. One that matters in the second only is part of an OR. One that matters in both is an ADDER.

**The mechanism.** Run the baseline method twice, once under Ns and once under Dn, at the same edge budget. Call the outputs $C_{\text{Ns}}$ and $C_{\text{Dn}}$. Then:

$$
C_{\text{AND}} = C_{\text{Ns}} \setminus C_{\text{Dn}}, \qquad C_{\text{OR}} = C_{\text{Dn}} \setminus C_{\text{Ns}}, \qquad C_{\text{ADDER}} = C_{\text{Ns}} \cap C_{\text{Dn}}
$$

The "logically sound" circuit optimizes both objectives at once:

$$
\arg\min_{C}\; \mathbb{E}_{(x,\tilde{x})\in T}\Big[D\big(p_G(y \mid x)\,\|\,p_C(y \mid x,\tilde{x})\big) + D\big(p_G(\tilde{y} \mid \tilde{x})\,\|\,p_C(\tilde{y} \mid \tilde{x},x)\big)\Big] \quad \text{s.t.}\quad 1 - \frac{|C|}{|G|} \ge s
$$

![The Ns+Dn pipeline: two runs at an equal edge budget, set operations that split the result into AND, OR and ADDER gates, and the union as the logically sound circuit](/imgs/blogs/rethinking-circuit-completeness-in-language-models-and-or-and-adder-gates-3.webp)

**How each baseline is modified** (Appendix E.3):

- **ACDC.** The pruning test $D_{\text{KL}}(G \,\|\, H_{\text{new}}) - D_{\text{KL}}(G \,\|\, H) \lt \tau$, where $H$ is the current subgraph and $H_{\text{new}}$ is $H$ with one candidate edge removed, becomes a sum of the clean-run and corrupted-run versions of the same difference.
- **EAP.** The first-order estimate of an edge's effect gets a second term computed on the corrupted run:

$$
\Delta_i \approx (\tilde{x}_i - x_i)^{\top} \frac{\partial L(x)}{\partial x_i} + (\tilde{x}_i - x_i)^{\top} \frac{\partial L(\tilde{x})}{\partial \tilde{x}_i}
$$

where $x_i, \tilde{x}_i \in \mathbb{R}^{d}$ are the clean and corrupted activations at edge $i$ (with $d$ the residual width) and $L$ is the task metric. The paper writes the Dn term with the same $(\tilde{x}_i - x_i)$ sign; since EAP ranks edges by magnitude, the sign convention is not load-bearing.
- **Edge Pruning.** Optimizing the summed objective directly caused gradient interference, so the authors train the Ns and Dn masks separately and average them.

**A worked example.** Here is the whole idea on the toy gates from the earlier table. This is my own sketch of Appendix D, with single-edge tests scored from a fixed baseline the way linear estimation does it.

```python
import numpy as np

relu = lambda v: np.maximum(v, 0.0)

# activation function and sender biases for each toy gate (paper, Appendix D)
GATES = {
    "AND":   (lambda s: relu(s - 1.0),       (1.0, 1.0)),
    "OR":    (lambda s: 1.0 - relu(1.0 - s), (1.0, 1.0)),
    "ADDER": (lambda s: relu(s),             (1.0, 1.5)),
}

def out(gate, kept):
    """Receiver output when kept[i] says whether sender i carries its clean value."""
    m, bias = GATES[gate]
    return float(m(sum(b for b, k in zip(bias, kept) if k)))  # zero ablation

def ns_edges(gate, tau=0.5):
    base = out(gate, (1, 1))                           # clean run, everything on
    return {i for i in (0, 1)
            if abs(base - out(gate, tuple(int(j != i) for j in (0, 1)))) > tau}

def dn_edges(gate, tau=0.5):
    base = out(gate, (0, 0))                           # corrupted run, everything off
    return {i for i in (0, 1)
            if abs(out(gate, tuple(int(j == i) for j in (0, 1))) - base) > tau}

for g in GATES:
    ns, dn = ns_edges(g), dn_edges(g)
    print(f"{g:5s} Ns={sorted(ns)} Dn={sorted(dn)} "
          f"AND={sorted(ns - dn)} OR={sorted(dn - ns)} ADDER={sorted(ns & dn)}")
# AND   Ns=[0, 1] Dn=[]     AND=[0, 1] OR=[]     ADDER=[]
# OR    Ns=[]     Dn=[0, 1] AND=[]     OR=[0, 1] ADDER=[]
# ADDER Ns=[0, 1] Dn=[0, 1] AND=[]     OR=[]     ADDER=[0, 1]
```

Every gate lands in its own bucket, and Ns alone would have returned an empty set for the OR gate, exactly the EAP failure from the previous section.

**Cost.** The paper argues the overhead is "reapplying the Ns algorithm once more under the strategy of Dn", so at most ${2}\times$ the baseline. Compare that with ${2^N}$ for brute-force combinations. (The paper calls the brute-force route "an NP problem" and concedes in Appendix E.1 that it cannot prove this formally; "exponential" is the accurate word.)

**When it fails.** The set operations assume the two circuits were cut at comparable granularity, which is the next section. They also assume every gate is one of the three. An XOR, where $B$ fires when exactly one sender is on, would be invisible to both surveys' simple signatures, and the authors list it as a limitation.

### Granularity alignment and the misalignment score

**The problem.** Set difference is only meaningful if $C_{\text{Ns}}$ and $C_{\text{Dn}}$ are the same size, roughly. If the Ns circuit has many more edges, its extra edges are mostly weak ADDER edges the Dn run never reached, and they get mislabelled as AND.

![Figure 3 from Chen et al. (2025): Venn diagrams of the Ns and Dn circuits. With aligned granularity the regions are exactly AND, ADDER and OR; with misaligned granularity, ADDER edges leak into the AND or OR regions](/imgs/blogs/rethinking-circuit-completeness-in-language-models-and-or-and-adder-gates-fig2.webp)

**The mechanism.** The test reuses the gate signatures. True AND edges show nearly the same effect whether you ablate one or two of them; ADDER edges do not. So the AND misalignment score compares one-edge versus two-edge ablation in the AND set against the same quantity in a random subset of it:

$$
S_{\text{AND}} = \mathbb{E}_{i,j}\big[D(C_{\text{AND}} \setminus i \,\|\, C_{\text{AND}} \setminus \{i,j\})\big] - \mathbb{E}_{i^*,j^*}\big[D(C^*_{\text{AND}} \setminus i^* \,\|\, C^*_{\text{AND}} \setminus \{i^*,j^*\})\big]
$$

where $i, j$ are two edges into the same receiver in $C_{\text{AND}}$, and $C^*_{\text{AND}} = C_{\text{AND}} \setminus K_{\text{AND}}$ is the AND set with a random subset $K_{\text{AND}}$ removed. A higher score means more ADDER edges are hiding in the AND set. The OR score is analogous, uses a "difference in differences" because OR and ADDER effects both get tiny in large circuits, and adds a constant $m = 1.5$ to keep it positive.

**What they found.** Holding a 100-edge Ns circuit on IOI and sweeping the Dn circuit from 60 to 140 edges, the summed score bottoms out near equal sizes. The fitted optimal Ns:Dn ratios (Table 6) range from 1.03:1 to 1.12:1, which the authors round to 1:1 everywhere.

## Experiments and results

All experiments use GPT-2 small as the graph and three standard tasks:

| Task | Clean prompt (corrupted word) | Target | Corrupted target |
| --- | --- | --- | --- |
| IOI (indirect object) | When Mary and John went to the store, John (Alice) gave a drink to | Mary | other names |
| GT (greater than) | The war lasted from 1517 (1501) to 15 | 18 or 19 or ... 99 | other digits |
| SA (syntactic agreement) | Many girls (girl) insulted | themselves | herself |

Each method (ACDC, EAP, Edge Pruning) is compared under Ns and Ns+Dn at six circuit sizes: 100, 200, 500, 1,000, 2,000 and 5,000 edges.

### Completeness

![Figure 4 from Chen et al. (2025): completeness of Ns and Ns+Dn circuits on IOI, GT and SA. After removing the circuit, KL rises and accuracy falls more for Ns+Dn at every edge count](/imgs/blogs/rethinking-circuit-completeness-in-language-models-and-or-and-adder-gates-fig3.webp)

Removing an Ns+Dn circuit hurts the model more than removing an Ns circuit of the same size, for every method, task and size. The full IOI numbers (Tables 7 and 8) at 1,000 and 5,000 edges:

| Method | Edges | KL after removal, Ns | Dn | Ns+Dn | Accuracy after removal, Ns | Dn | Ns+Dn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACDC | 1,000 | 0.81 | 0.96 | **1.03** | 0.40 | 0.34 | **0.33** |
| EAP | 1,000 | 0.74 | 0.96 | **1.02** | 0.39 | 0.29 | **0.26** |
| Edge Pruning | 1,000 | 1.08 | 1.11 | **1.13** | 0.32 | 0.27 | **0.26** |
| ACDC | 5,000 | 1.34 | 1.42 | **1.46** | 0.26 | 0.22 | **0.21** |
| EAP | 5,000 | 1.21 | 1.42 | **1.48** | 0.29 | 0.13 | **0.09** |
| Edge Pruning | 5,000 | 1.62 | 1.73 | **1.73** | 0.21 | 0.11 | **0.11** |

Higher KL and lower accuracy after removal both mean more complete. The largest gap is EAP, exactly the method that finds no OR edges under Ns: at 5,000 edges, the accuracy left after deletion falls from 0.29 to 0.09, a two-thirds drop (my arithmetic: $(0.29 - 0.09)/0.29 \approx 0.69$).

**Stability.** Across 30 seeds, the authors measure how much larger the pairwise Hamming distance between circuits is under Ns than under Ns+Dn (Table 3). On IOI at 5,000 edges the excess is 127 edges for ACDC, 14.5 for EAP and 509.5 for Edge Pruning, growing with circuit size. That fits the diagnosis: the randomness was the arbitrary choice of which OR edge to keep.

### Faithfulness

At equal edge budgets the paper reports $C_{\text{Ns+Dn}} \approx C_{\text{Ns}} > C_{\text{Dn}}$ in faithfulness (Appendix H). Dn circuits miss AND edges, as Corollary 1 predicts. For EAP, Ns+Dn is much more faithful than Ns, because Ns-EAP misses the "one edge per OR gate" minimum. So the framework buys completeness without a faithfulness tax.

### What the gates look like inside IOI

![Figure 6 from Chen et al. (2025): two-layer slices of the AND, OR and ADDER circuits ACDC finds on IOI, coloured by the component's role in the original IOI circuit](/imgs/blogs/rethinking-circuit-completeness-in-language-models-and-or-and-adder-gates-fig4.webp)

Mapping each component to its known IOI role gives the paper's most interpretable result:

| Gate | Typical senders | Example from IOI | Layer span |
| --- | --- | --- | --- |
| AND | Different functions that must cooperate | Induction heads need both duplicate-token heads and previous-token heads; name movers need S-inhibition and induction heads | Medium |
| OR | Copies of the same function (backups) | S-inhibition heads fed by several induction heads; backup name movers fed by several S-inhibition heads | Shortest, adjacent layers |
| ADDER | Shallow and deep functions combined, often MLPs | Name movers summing everything between duplicate-token and S-inhibition heads | Longest |

OR gates are where backup mechanisms live, which is why missing them makes a circuit look complete when it is not.

**Contribution and proportion.** ADDER gates carry the largest share of the output effect; AND and OR contribute similarly (Appendix J). ACDC finds roughly equal numbers of all three, while EAP and Edge Pruning find far fewer OR edges. Edge Pruning also finds few AND edges because it scores edges individually, so a two-edge AND gate looks like two half-strength edges (Appendix K).

### An unlearning toy

In Appendix L, the authors find a circuit for harmful responses on PKU-SafeRLHF and delete it. Starting from 92.74% accuracy, the remaining graph scores 29.17% after deleting the EAP circuit, 31.44% after Edge Pruning, and 13.21% after the Ns+Dn circuit. If you care about removing a capability rather than explaining it, the backup paths are exactly what you need to find.

### What is load-bearing and might not transfer

- **Model scale.** Only GPT-2 small is tested. Larger models plausibly carry more redundancy, and so more OR gates.
- **Corruption choice.** Every gate label depends on what "corrupted" means; a mean ablation instead of interchange could move an edge between AND and ADDER.
- **The 1:1 rule.** It is fitted on three tasks and one model, and could drift on a model with many more backup paths.

## Critique

**What is strong.** The gate vocabulary turns a vague complaint ("circuits are unstable") into a mechanism with a testable signature. The fix is cheap, method-agnostic and produces labels, not just a bigger circuit. The IOI gate anatomy agrees with what the original IOI work found about backup name movers, an independent sanity check.

**What is weak.**

- **Binary gates on a continuous network.** Real receivers are mixtures: $B$ might be $A_1 \wedge (A_2 \vee A_3)$, or an ADDER with saturation. The set-difference rule forces every edge into one of three bins, and the misalignment score is the only check that the binning is sane.
- **Dn alone does most of the completeness work.** In Tables 7 and 8 the Dn column is usually close to Ns+Dn (EAP at 1,000 edges: KL 0.96 versus 1.02). The framework's value is getting that completeness while keeping Ns-level faithfulness, which the paper shows mainly through appendix figures rather than a headline table.
- **Missing baseline.** I would want to see a size-matched control: take $C_{\text{Ns}}$ and add the next highest-scoring Ns edges until it matches $|C_{\text{Ns+Dn}}|$. If that closes most of the completeness gap, the gain is about budget allocation, not OR gates.
- **Loose complexity claim.** "NP problem" is asserted and then admitted to be unprovable; it is an exponential enumeration argument.
- **Small details.** The unlearning dataset citation is broken, and EAP, which is deterministic given its data, shows nonzero seed randomness that is never explained.

**What would change my mind.** If, on a larger model, the edges in $C_{\text{Ns}} \setminus C_{\text{Dn}}$ and $C_{\text{Dn}} \setminus C_{\text{Ns}}$ stopped showing the one-edge versus two-edge ablation signature of the Table 5 kind, the gate labels would be artefacts of the set operation rather than properties of the model. Equally, if the size-matched control above recovered the same completeness, I would stop crediting OR-gate recovery for the gain.

## What I'd build with this

These are my extrapolations, not the paper's claims.

1. **Gate-aware ablation for safety.** Unlearning should target OR gates first, since those are the backups a single-path ablation leaves standing. Appendix L is a hint, not yet a test.
2. **Gate typing for feature circuits.** The same Ns/Dn split applies to sparse-autoencoder feature graphs, where redundant features are common. It would be a natural extension of the [sparse feature circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models) pipeline.
3. **Skill composition as satisfiability.** The paper's own future direction: if task A needs edge $i \to j$ and task B needs it removed, knowing it is an OR edge in A resolves the conflict. Writing circuits as boolean formulas makes this a SAT check.

## References

1. Hang Chen, Jiaying Zhu, Xinyu Yang, Wenya Wang. *Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates.* NeurIPS 2025. [arXiv:2505.10039](https://arxiv.org/abs/2505.10039)
2. Conmy et al. *Towards Automated Circuit Discovery for Mechanistic Interpretability* (ACDC). NeurIPS 2023.
3. Syed, Rager, Conmy. *Attribution Patching Outperforms Automated Circuit Discovery* (EAP). BlackboxNLP 2024.
4. Bhaskar et al. *Finding Transformer Circuits with Edge Pruning.* NeurIPS 2024.
5. Wang et al. *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small.* ICLR 2023.
6. Heimersheim and Nanda. *How to Use and Interpret Activation Patching.* arXiv:2404.15255, 2024.
7. On this site: [Sparse feature circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models)

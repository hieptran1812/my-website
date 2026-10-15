---
title: "A Implies B: How LLMs Run a Four-Step Logic Circuit Inside Their Attention Heads"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How Hong et al. use causal mediation analysis to find a sparse, modular circuit of four attention-head families that Mistral-7B and Gemma-2 use to answer a small propositional logic problem, and what changes at 27B and at 3 layers."
tags: ["paper-reading", "mechanistic-interpretability", "circuit-analysis", "logical-reasoning", "propositional-logic", "activation-patching", "causal-mediation-analysis", "attention-heads", "neurips-2025-spotlight"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 19
paper:
  title: "A Implies B: Circuit Analysis in LLMs for Propositional Logical Reasoning"
  authors: "Guan Zhe Hong, Nishanth Dikkala, Enming Luo, Cyrus Rashtchian, Xin Wang, Rina Panigrahy"
  venue: "NeurIPS 2025 (spotlight)"
  url: "https://arxiv.org/abs/2411.04105"
---

> [!tldr]
> **Claim.** When Mistral-7B, Gemma-2-9B and Gemma-2-27B answer a tiny propositional logic question, the work is done by a sparse set of attention heads that split into four families: one locates the queried rule, one moves it, one processes the facts, one decides. Each step runs in its own layers, in order.
>
> **Why it matters.** It is evidence that a pre-trained LLM can solve a multi-step problem with modular, sequential parts rather than one smeared computation, and the same layout shows up in two independently trained model families.
>
> **Most surprising result.** The models are *lazy*. The circuit does almost nothing with the Rules and Facts until the Question names the query variable. A 3-layer transformer trained only on this task does the opposite: it precomputes partial answers and mixes the steps.
>
> **Where it breaks.** All analysis is on the first proof token, with 60-sample experiments, mostly at the level of attention patterns. The 27B analysis is partial.

Ask an LLM "A or B implies C. A is true. What is C?" and it answers. The interpretability question is *how*: does it run something like a proof, step by step, in different parts of the network, or pattern-match the prompt in one diffuse pass?

This paper, from Purdue and Google Research, answers that for one carefully chosen logic problem. It uses causal mediation analysis (the formal name for activation patching) to find the heads that matter, then targeted patching to pin down what each one does. For why "a circuit" is a hypothesis that needs testing, see [Hypothesis Testing the Circuit Hypothesis in LLMs](/blog/paper-reading/ai-interpretability/hypothesis-testing-the-circuit-hypothesis-in-llms).

## The problem: five variables, two rules, one query

Circuit analysis needs a task in a narrow band. Too simple (copy a token) teaches nothing about reasoning. Too complex (a word problem) and you cannot change one part of the input while holding the rest fixed, which patching requires.

The authors pick a propositional logic template with five boolean variables. There are two rules, three facts and one question:

| Part | Example | Role |
| --- | --- | --- |
| Rules | `A or B implies C. D implies E.` | One LogOp chain (with "and" or "or"), one linear chain |
| Facts | `A is true. B is false. D is true.` | Truth values of the three premise variables |
| Question | `what is the truth value of C?` | The QUERY variable picks which chain matters |
| Answer | `A is true. A or B implies C; C True.` | A minimal proof, then the value |

The model has to resolve which rule the query refers to, find the facts relevant to that rule while ignoring the distractor, and know that for "or" the fact "A is true" alone settles it. Every word and capital letter is a single token, so positions are easy to control. The models get 4 (Gemma) or 6 (Mistral) solved examples before the real question.

### Why only the first answer token

The authors analyze just the first token the model writes. That sounds narrow, but it is the right place to look. Before writing "A", the model must have executed the whole chain

$$
\text{QUERY} \;\rightarrow\; \text{relevant rule} \;\rightarrow\; \text{relevant fact(s)} \;\rightarrow\; \text{decision}
$$

with no chain of thought to lean on. Later tokens can copy from text already written. The paper also reports that on the disjunction problem, wrong answers are always wrong from the first token, so that is where errors originate.

### How hard is it for the models?

Not trivial. Even with few-shot examples, proof accuracy is 70% to 86%:

| Model | Problem version studied | Proof accuracy |
| --- | --- | --- |
| Mistral-7B | "or" only | ~70% (96% on linear-chain queries) |
| Gemma-2-9B | "or" only | ~83% |
| Gemma-2-27B | full ("and" and "or") | 86% |

One detail matters for the circuit story: in the test question, the linear chain's premise is always true, so the model cannot shortcut by spotting a false premise without resolving which rule was queried.

![Figure 1 from Hong et al. (2025): problem set-up (left), the necessity, sufficiency and fine-grained analysis steps (middle), and the shared four-step finding (right).](/imgs/blogs/a-implies-b-circuit-analysis-in-llms-for-propositional-logical-reasoning-fig1.webp)

## Technique 1: causal mediation analysis as a circuit search

**The problem.** Attention maps tell you where a head looks, not whether its output matters. You need a causal test: if this head's output changed, would the answer change?

**The analogy.** In a relay race, swap one runner's baton for one from a different race. If the team now finishes the other course, that runner was on the critical path.

**The mechanism.** Causal mediation analysis (Pearl, 2001) splits the total effect of an input $X$ on an output $Y$ into a direct effect and an indirect effect through a mediator $M$. In this paper:

1. Build a **normal prompt** whose first answer token is, say, B.
2. Build a **counterfactual prompt** that differs in one controlled way. The workhorse change is flipping the QUERY from C to E, which sends the model down an entirely different chain ("E, so D implies E, so D") with answer D.
3. Cache every head's activation on the counterfactual prompt.
4. Run the normal prompt, but overwrite one head's output with its cached counterfactual activation. Everything downstream recomputes.
5. Measure how far the model's preference moved from B toward D.

A head whose swap pushes the model strongly toward the counterfactual answer has a large **indirect effect** and belongs to the circuit.

**The math.** The outcome is a logit difference between the two candidate answer tokens. For a prompt $X$ with original answer $y_{\text{orig}}$ and counterfactual answer $y_{\text{alt}}$:

$$
\Delta(X) \;=\; \operatorname{logit}(X)[y_{\text{alt}}] \;-\; \operatorname{logit}(X)[y_{\text{orig}}]
$$

Here $\operatorname{logit}(X)[y]$ is the pre-softmax score the model assigns to token $y$ at the answer position. A negative $\Delta$ means the model prefers the original answer.

Raw logit differences are hard to compare across heads and models, so the paper calibrates them into a score that runs roughly from 0 to 1. For head $(\ell, h)$ (layer $\ell$, head index $h$) patched at token position $t$, averaged over $N$ prompt pairs:

$$
S_{(\ell,h,t)} \;=\; \frac{\dfrac{1}{N}\displaystyle\sum_{n=1}^{N} \Delta_{\text{orig}\to\text{alt},\,n;(\ell,h,t)} \;-\; \Delta^{\dagger}_{\text{orig}}}{\Delta_{\text{alt}} \;-\; \Delta^{\dagger}_{\text{orig}}}
$$

The symbols:

- $\Delta_{\text{orig}\to\text{alt},n;(\ell,h,t)}$ is the logit difference on the original prompt $n$ after patching that one head's activation from the counterfactual run.
- $\Delta^{\dagger}_{\text{orig}}$ is the average logit difference on the unpatched original prompts. This is the "nothing happened" baseline, and it is negative.
- $\Delta_{\text{alt}}$ is the average logit difference on the unpatched counterfactual prompts. This is the "fully switched" ceiling, and it is positive.

So $S = 0$ means the patch did nothing and $S = 1$ means the single patch moved the model all the way to the counterfactual answer. Each experiment uses 60 samples unless the paper says otherwise.

**Worked example (my illustrative numbers, not the paper's).** Say the unpatched model on original prompts has $\Delta^{\dagger}_{\text{orig}} = -4$ (it prefers B by 4 logits). On counterfactual prompts it has $\Delta_{\text{alt}} = +4$ (it prefers D by 4). Patching one head gives an average of $+2$. Then

$$
S = \frac{2 - (-4)}{4 - (-4)} = \frac{6}{8} = 0.75
$$

That one head moved the model three quarters of the way to the other answer. Most heads in a 9B model score near zero.

**When it fails.** A single-head patch can understate a head's importance when neighbors do the same job and compensate. The paper hits exactly this in Mistral-7B, which uses grouped-query attention: heads $(\ell, 4h)$ through $(\ell, 4h+3)$ share keys and values. Patching heads one at a time missed the group (9, 24 to 27), which showed up only when the whole group was patched together.

![Necessity patching swaps one head's activation and checks whether the answer flips; sufficiency verification freezes everything outside the circuit and checks whether the circuit alone carries the answer.](/imgs/blogs/a-implies-b-circuit-analysis-in-llms-for-propositional-logical-reasoning-1.webp)

### First result: heads matter, MLPs mostly do not

Patching every head and MLP in Gemma-2-9B gives a sparse picture. MLPs score mostly below 0.1, except MLP-0, which earlier work (Wang et al., 2023) describes as a nonlinear token embedding. The final circuit has just 13 heads.

## Technique 2: patching the query, key and value separately

**The problem.** Knowing that a head matters does not tell you *what* it does: where it looks, or what it carries.

**The analogy.** A librarian can be important because she knows which shelf to go to, or because she carries the book back to the desk. Swap her shelf list and see if she fetches the wrong book. Swap the book in her hands and see if the wrong book arrives.

**The mechanism.** An attention head at position $t$ computes

$$
\operatorname{head}(t) \;=\; \sum_{s \le t} \operatorname{softmax}_s\!\left(\frac{q_t \cdot k_s}{\sqrt{d_k}}\right) v_s
$$

where $q_t$ is the query vector at the current position, $k_s$ and $v_s$ are the key and value vectors at earlier position $s$, and $d_k$ is the head dimension. The attention weight on position $s$ is the softmax term. You can patch $q$, $k$ or $v$ independently:

- Patching **$q_t$** changes what the head is looking for.
- Patching **$k_s$** changes what positions advertise, so it moves where the head lands.
- Patching **$v_s$** leaves the attention pattern alone and changes what gets copied.

If query patching has a big effect but value patching does not, the head's job is deciding *where to look*. If value patching dominates and the pattern is fixed, the head is a mover that always reads the same spot and copies whatever is there.

**Worked example.** A head at the QUERY token puts weight 0.9 on position 3. Flip the QUERY from C to E. If the weight jumps to position 9, query patching matters: the head chose a different rule. If the weight stays on position 3 but $v_3$ now carries different content, value patching matters.

**When it fails.** Grouped-query attention couples keys and values across heads. In Gemma-2-9B, keys and values are shared by pairs of heads, so a key-patching result names a key group, and you have to map it back. The paper does this explicitly: strong keys (17,4), (19,5), (22,2) and (23,6) correspond to heads (17,9), (19,11), (22,5) and (23,12).

![Figure 4 from Hong et al. (2025): query (a) and value (c) patching scores for every Gemma-2-9B head, with typical attention patterns per family (b).](/imgs/blogs/a-implies-b-circuit-analysis-in-llms-for-propositional-logical-reasoning-fig3.webp)

## The four head families

Query and value patching, plus attention statistics, sort the important Gemma-2-9B heads into four families. Each family runs at a specific token position and in a specific band of layers.

| Family | Representative head | Which part matters | What it attends to | Statistic in the paper |
| --- | --- | --- | --- | --- |
| Queried-rule locating | (19,11) | Query activation | At QUERY, the conclusion variable of the queried rule | Above 85% attention on that variable |
| Queried-rule mover | (23,6) | Value activation only | At ":", the QUERY token | ~50% on QUERY, about 5x the next-largest position |
| Fact processing | (26,12) | Query activation | At ":", the Facts section | ~60% on the sentence stating the correct fact |
| Decision | (30,9) | Query activation | At ":", the correct answer variable | Above 80% on a single token |

The ":" is the last token of the prompt, right before the model writes its first answer token. The full 9B circuit lists the other members: locating heads (17,9), (21,7), (22,5), (23,12); movers (20,7), (24,15); fact processors (24,5), (25,7), (26,0); decision head (28,12).

### Reading the circuit as an algorithm

In layer order, the families form a four-step procedure:

1. **Locate (layers 17 to 23).** At QUERY, find the rule that concludes in the query variable and store it at the QUERY position.
2. **Move (layers 20 to 24).** At ":", read QUERY with a fixed pattern and copy the rule information over. Only the values depend on the query, exactly what a mover should look like.
3. **Process facts (layers 24 to 26).** At ":", read the Facts section for the relevant fact.
4. **Decide (layers 28 to 30).** At ":", put most weight on the correct answer token and push it to the output.

![Figure 3 from Hong et al. (2025): the Gemma-2-9B reasoning circuit for the first answer token and its four head families.](/imgs/blogs/a-implies-b-circuit-analysis-in-llms-for-propositional-logical-reasoning-fig2.webp)

The striking part is the ordering. Nothing told the model to factor the problem this way, and the authors' own prior conjecture (from small models) was that it would not.

![The Gemma-2-9B circuit as a pipeline: each head family runs at one token position in one band of layers and hands its result to the next.](/imgs/blogs/a-implies-b-circuit-analysis-in-llms-for-propositional-logical-reasoning-2.webp)

Mistral-7B has the same four families in layers 9 to 19. Its cleanest locating head, (12,9), puts 93.0 ± 9.4% of its relevant-context attention on the correct conclusion variable. Its mover (13,11) puts 54.2 ± 12.5% on QUERY, against 5.2 ± 1.1% for the runner-up position.

## Technique 3: verifying sufficiency by freezing everything else

**The problem.** Necessity says each circuit head matters, not that the circuit is *enough*. Other heads could do half the work in a way no single-head patch reveals.

**The analogy.** Silence the whole orchestra except the strings and check whether the melody is still there.

**The mechanism.** Run the model with every attention head outside the candidate circuit $C$ frozen to its counterfactual activation, and only the heads in $C$ allowed to compute normally on the real prompt. MLPs are not frozen: following Wang et al. (2023), the paper treats them as part of the direct path between heads, since they do not move information across positions. The question is whether the circuit alone carries the answer.

**The math.** The reported number is a ratio of logit differences:

$$
R(C) \;=\; \frac{\Delta^{C}_{\text{normal}}}{\Delta_{\text{normal}}}
$$

$\Delta_{\text{normal}}$ is the average logit difference of the untouched model on normal prompts. $\Delta^{C}_{\text{normal}}$ is the same quantity when only $C$ runs on the normal prompt and every other head sees the counterfactual. $R = 1$ means the circuit alone reproduces the model's behavior. $R = -1$ means the model behaves as if it saw the counterfactual prompt. The appendix describes the same test with the roles of the two prompts swapped; the logic is symmetric.

**Worked example.** If the full model prefers the right answer by 5 logits and the circuit-only run prefers it by 4.7, then $R = 4.7 / 5 = 0.94$, which is exactly Gemma-2-9B's result.

**The results.** The paper then removes one family at a time:

| Circuit $C^{\dagger}$ | Gemma-2-9B | Mistral-7B |
| --- | --- | --- |
| $C_{\text{null}}$ (no heads run normally) | -1.0 | -1.0 |
| Full $C$ | **0.94** | **0.98** |
| $C$ minus locating heads | -0.97 | -1.02 |
| $C$ minus mover heads | -0.40 | -0.99 |
| $C$ minus fact-processing heads | 0.17 | -0.25 |
| $C$ minus decision heads | -1.11 | -0.89 |

The full circuit recovers 94% (Gemma) and 98% (Mistral). Drop any one family and the result collapses.

![Sufficiency ratios for the full circuit and for the circuit with one head family removed, Gemma-2-9B and Mistral-7B, from Tables 1 and 2 of the paper.](/imgs/blogs/a-implies-b-circuit-analysis-in-llms-for-propositional-logical-reasoning-3.webp)

Two readings the paper states less loudly than I would. "Collapses" is not uniform: without fact-processing heads Gemma-2-9B still sits at 0.17, so something outside the named heads carries part of that step. And the locating heads are the most load-bearing in both models, which fits their role as the first link.

**When it fails.** The paper itself calls sufficiency verification "a major open problem". Lower-layer heads that do token-level processing are not part of the reasoning circuit, yet ablating them could break everything; the circuit is defined as the query-sensitive part, not the only part that matters. The Gemma-2-9B verification is also coarser, patching circuit heads at every position from QUERY to ":".

## Technique 4: targeted counterfactuals to confirm each role

**The problem.** Flipping the QUERY lights up every family. To confirm a head *locates rules*, you need a counterfactual that only one family should care about.

- **Swap rule order.** "D implies E. A or B implies C." has the same answer, but the queried rule moved. A head that looks the rule up should depend on its *keys* in the Rules section, so patch only those.
- **Swap fact values.** The fact-processing heads should now show the strongest effect, and they do.
- **Flip the logical operator** (27B only), covered below.

**Worked example.** Under the rule-order swap the answer stays "C is true". A head that finds the rule by matching content against keys will be misled when its Rules-section keys come from the swapped prompt, so it lands on the wrong rule. The paper finds exactly this: the keys with the strongest effect belong to the locating heads.

![Figure 5 from Hong et al. (2025): (a) key patching after swapping rule order, strongest on the locating heads; (b) attention statistics per head family in Gemma-2-9B.](/imgs/blogs/a-implies-b-circuit-analysis-in-llms-for-propositional-logical-reasoning-fig4.webp)

**When it fails.** Attention statistics are correlational: a head can look at the right token for an unrelated reason. The paper treats the bar charts as support and the targeted patches as the causal evidence, and flags where it only has the former.

## Lazy reasoning and circuit reuse

Two findings fall out of the circuit's layout.

**The models are lazy.** The core circuit does not preprocess the Rules and Facts as they stream in. It activates after the Question names the query variable, so the work for the first token happens at QUERY and at ":". That is a choice, and a small model trained on the task makes the opposite one (below).

**Sub-circuits get reused.** When Gemma-2-9B and Mistral-7B later cite the rule in their proof, they reuse the same locating heads at other positions, and the decision heads fire again. This matches Merullo et al.'s (2024) finding that circuit components are shared across tasks.

## What changes at 27B

Gemma-2-27B is studied on the full problem, with both "and" and "or" rules. The paper warns that this analysis is less exhaustive because of compute limits. Two differences stand out.

**Logical-operator heads.** Flipping "and" to "or" isolates heads that care about the operator. Heads (23,13) and (23,31) attend specifically to the word "and" or "or" in the queried rule and have the strongest direct effects on the decision heads, fed by (20,16) and (22,13). The 9B model has no comparable heads, though it was only studied on "or".

**More parallel wiring.** In Gemma-2-9B, fact-processing heads feed the decision heads and have weak direct effects on the logits. In Gemma-2-27B, several fact-processing heads write straight to the output and bypass the decision heads.

![Figure 25 from Hong et al. (2025): direct effects of heads on the logits in Gemma-2-9B (a) and Gemma-2-27B (b); more 27B heads write straight to the output.](/imgs/blogs/a-implies-b-circuit-analysis-in-llms-for-propositional-logical-reasoning-fig5.webp)

The same four roles exist, but the wiring changes and the bigger model gains components. That sits well next to [LLM Circuit Analyses Are Consistent Across Training and Scale](/blog/paper-reading/ai-interpretability/llm-circuit-analyses-are-consistent-across-training-and-scale), which found algorithm-level stability with component-level churn. Here the algorithm is stable from 9B to 27B and the plumbing is not.

## The contrast: a 3-layer transformer does it differently

To check whether modularity is inevitable, the authors train attention-only GPT-2-style transformers from scratch on a length-3 version of the problem. The smallest with 100% test accuracy has 3 layers and 3 heads, and its mechanism looks nothing like the LLMs'.

**A routing direction.** At the QUERY position, the layer-2 attention output contains a direction $h_{\text{route}}$ whose sign says which chain was queried:

$$
\text{attn}_2(\text{QUERY}) \;\ni\; \alpha(X)\, h_{\text{route}}, \qquad \alpha(X) > 0 \text{ for linear},\;\; \alpha(X) \lt 0 \text{ for LogOp}
$$

Here $\alpha(X)$ is a sample-dependent coefficient. Layer 3 reads the sign and switches mode: over 90% of its attention goes to QUERY for the linear chain, under 5% for the LogOp chain. The causal check is clean. They estimate $\hat h_{\text{route}}$ by averaging layer-2 output over 1,000 linear-chain samples. Subtracting it on 500 linear-chain test prompts makes the model output the correct first token *of the LogOp chain*. Adding it to 500 LogOp prompts sends over 95% of layer-3 attention to QUERY.

**Eager, not lazy.** A linear probe on the layer-2 residual at the QUERY position predicts the linear chain's answer with over 97% test accuracy on 5,000 samples. The same probe for the LogOp chain's answer gets about 27% on test against 94% on train, so that answer is not yet there. For the LogOp chain, layers 1 and 2 pass forward a partial answer (the false nodes for "and", the true nodes for "or") and layer 3 finishes it.

![Pre-trained LLMs wait for the query and then run four separate steps; the 3-layer transformer trained on the task precomputes partial answers and routes on a single direction, with steps intermingled.](/imgs/blogs/a-implies-b-circuit-analysis-in-llms-for-propositional-logical-reasoning-4.webp)

This is the paper's best argument that the LLM result is not a tautology: a network can solve the problem perfectly without modular, lazy structure, yet the LLMs arrive at the modular solution anyway. My guess (not the paper's claim) is that the modular structure is reused machinery from general in-context retrieval, which pre-training builds and a 3-layer task-specific model never needs.

## What is load-bearing in the setup

What the experiments lean on:

- **Single-token variables.** "The conclusion variable" is one position. Multi-token names would blur the clean attention patterns.
- **One analyzed token.** The rest of the proof is barely examined beyond the reuse finding.
- **Fixed template.** What makes patching possible is also why this is not yet evidence about free-form reasoning.
- **Model subsets.** The 7B and 9B models only see "or", so cross-model comparisons mix problem difficulty with size.

## Critique

**What is strong.** The experimental design is the real contribution: QUERY flips for discovery, then rule, fact and operator swaps to attribute a *role* rather than an importance score. The same four-family layout in Mistral and Gemma, trained by different teams, is the result most likely to last. The 3-layer contrast turns "LLMs are modular" from a description into a finding.

**What is weak.** The sufficiency test leaves MLPs running, so "13 heads suffice" means "13 heads plus all MLPs plus the layers below." Experiments use 60 samples, with no error bars on the sufficiency ratios. There are small inconsistencies too: Table 1's caption describes $C_{\text{null}}$ as "no intervention," while the Mistral table and the appendix describe it as every head frozen to the counterfactual (the second reading matches the -1.0). The appendix calls Gemma-2-27B accuracy "nearly perfect" while the main text says 86%.

**Missing ablations.** Longer chains (listed as future work), multi-token variable names, and errors: when the model writes the wrong first token, which family failed? That would test whether the circuit explains failures as well as successes.

**What would change my mind.** If longer reasoning chains (three or four hops) were solved by heads that do not fall into these families, or if the same heads were needed for unrelated retrieval tasks with no logic at all, I would read this circuit as general in-context lookup machinery rather than a reasoning circuit. Conversely, if error analysis showed most wrong answers trace to the locating heads picking the wrong rule, that would make the four-step story much stronger.

For a sense of how circuit claims can be stress-tested at the edge level, see [Finding Transformer Circuits with Edge Pruning](/blog/paper-reading/ai-interpretability/finding-transformer-circuits-with-edge-pruning), which optimizes a circuit directly instead of patching heads one at a time.

## References

1. Hong, G. Z., Dikkala, N., Luo, E., Rashtchian, C., Wang, X., and Panigrahy, R. [A Implies B: Circuit Analysis in LLMs for Propositional Logical Reasoning](https://arxiv.org/abs/2411.04105). NeurIPS 2025 (spotlight). arXiv:2411.04105.
2. Wang, K., Variengien, A., Conmy, A., Shlegeris, B., and Steinhardt, J. Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small. ICLR 2023.
3. Pearl, J. Direct and Indirect Effects. UAI 2001.
4. Related on this site: [Hypothesis Testing the Circuit Hypothesis in LLMs](/blog/paper-reading/ai-interpretability/hypothesis-testing-the-circuit-hypothesis-in-llms), [LLM Circuit Analyses Are Consistent Across Training and Scale](/blog/paper-reading/ai-interpretability/llm-circuit-analyses-are-consistent-across-training-and-scale), [Finding Transformer Circuits with Edge Pruning](/blog/paper-reading/ai-interpretability/finding-transformer-circuits-with-edge-pruning).

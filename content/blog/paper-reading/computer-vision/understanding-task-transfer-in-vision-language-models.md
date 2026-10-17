---
title: "Understanding Task Transfer in Vision-Language Models: A Map of What One Finetune Does to Everything Else"
publishDate: "2026-10-17"
date: "2026-10-17"
description: "How finetuning a VLM on one perception task moves its zero-shot accuracy on twelve others, how the Perfection Gap Factor makes those moves comparable, and how to turn the resulting transfer graph into a finetuning plan."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["interpretability-analysis", "cvpr-2026", "cvpr-2026-oral", "vision-language-models", "task-transfer", "perfection-gap-factor", "lora", "blink-benchmark", "qwen2-5-vl", "data-selection"]
author: "Hiep Tran"
featured: true
readTime: 18
image: "/imgs/blogs/understanding-task-transfer-in-vision-language-models-fig1.webp"
excerpt: "Finetune Qwen-2.5-VL on one BLINK perception task and twelve other tasks move, some up and some down. This CVPR 2026 oral builds the map, a normalized metric for it, and a first recipe for using it to pick training data."
paper:
  title: "Understanding Task Transfer in Vision-Language Models"
  authors: "Bhuvan Sachdeva, Karan Uppal, Abhinav Java, Vineeth N. Balasubramanian"
  venue: "CVPR 2026 (Oral)"
  url: "https://arxiv.org/abs/2511.18787"
---

> [!tldr]
> - **What it does.** Sachdeva et al. (Microsoft Research India) finetune Qwen-2.5-VL 3B, 7B and 32B with LoRA on each of 13 BLINK perception tasks, one at a time, and measure what happens to zero-shot accuracy on all 13. That gives a 13 by 13 transfer matrix per model, over four seeds.
> - **The key tool.** The *Perfection Gap Factor* (PGF) scores a change as the fraction of remaining headroom it closes, so a +3 point gain near the ceiling outranks a +10 point gain on a task stuck at 40%.
> - **What they find.** Positive transfer grows with model size; low-level tasks (relative depth, reflectance, visual correspondence) are both the best donors and the most improvable targets; and at 32B a clique of nine tasks all lift each other.
> - **Most surprising.** On two target tasks (Jigsaw and Object Localization), a training mix chosen from *other* tasks by PGF beat finetuning on the target task's own data.
> - **Where it is thin.** One model family carries almost all of it, everything is multiple choice, and the data-selection result is explicitly preliminary.

![Figure 1 from Sachdeva et al. (2026): finetuning Qwen-2.5-VL 32B on each perception task and evaluating on all others yields a transfer graph and a PGF heatmap](/imgs/blogs/understanding-task-transfer-in-vision-language-models-fig1.webp)

The figure above is the whole study at a glance: finetune on one task, evaluate on every task, keep the strongest edges, and read the structure. The rest of this post unpacks the metric that makes the heatmap meaningful, the two aggregates built on it, and what the map says about how to finetune a VLM without breaking it.

## The problem: finetuning one perception skill is a gamble

Vision-language models are good at document QA and chart reading, and surprisingly bad at perception that humans find trivial. On the BLINK benchmark, the paper notes, GPT-4o averages 60.04% and GPT-4V 51.14%, against roughly 95% for humans. BLINK asks which marked point is closer to the camera, which crop fills a jigsaw hole, or how many objects a scene holds.

The practitioner's fix is obvious: collect data for the weak task and run a LoRA finetune (the mechanics are in [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft)). The catch is that nobody knows what that finetune does to the other twelve skills. A relative-depth finetune might quietly sharpen counting, or it might wreck forensic detection. Without a map, every task-specific finetune is a bet on the skills you did not measure.

Prior work on task relationships does not answer this. Taskonomy (Zamir et al.) measured transfer by pretraining an encoder on a source task and training a new decoder on the target, in the CNN era. That is *transfer learning*: the target always gets its own supervised training. Here the question is *zero-shot* transfer in a foundation model: you train on task A only, and ask what A's training did to B, which the model never trained on.

Raw accuracy deltas cannot answer it either: BLINK tasks sit at very different baselines, so "+2 points" means something different on each. That is what PGF is for.

## Contributions

1. **A systematic zero-shot transfer study** of 13 BLINK perception tasks across three Qwen-2.5-VL sizes (3B, 7B, 32B), four seeds each, plus a qualitative replication on LLaVA-1.5-13B.
2. **The Perfection Gap Factor**, a headroom-normalized change score, and two aggregates on top of it: *transferability* (how much a source task moves others) and *malleability* (how much a target task gets moved).
3. **Structural findings**: transfer strengthens with scale, depends on perceptual level and granularity, and organizes into cliques of mutually helpful or mutually harmful tasks, plus four "personas" (donors, pirates, sponges, sieves).
4. **A video check**: the image-trained checkpoints also move spatio-temporal tasks on VSI-Bench, with the same donor and pirate showing up.
5. **A data-selection demo**: picking source tasks by PGF to train for a target that has no data of its own.

## Method: three measurements built on one ratio

There is no architecture here: the techniques are the metric, two aggregates, and a graph.

### Perfection Gap Factor: score the headroom you closed

**The problem.** Raw accuracy change is not comparable across tasks. A model at 90% on a task with a 95% ceiling has 5 points of room; the same model at 40% has 55. A +3 on the first is most of what is left to win. A +10 on the second barely dents it.

**The analogy.** Think of a progress bar for "how far to perfect." PGF asks what fraction of the *empty* part of the bar the finetune filled, rather than how many pixels of fill it added.

**The mechanism.** For a base model $M$, a source task $T_i$, and a target task $T_j$, take three numbers: the base model's accuracy on $T_j$, the accuracy after finetuning on $T_i$, and the ceiling of $T_j$. Then divide the change by the room that was left.

$$
\mu_{i\to j} \;=\; \frac{\mathrm{Acc}\big(M(T_i),\,T_j\big) \;-\; \mathrm{Acc}\big(M,\,T_j\big)}{U_j \;-\; \mathrm{Acc}\big(M,\,T_j\big) \;+\; \epsilon}
$$

Here $\mathrm{Acc}(M, T_j)$ is the zero-shot accuracy of the untouched model on target $T_j$; $M(T_i)$ is the model after a LoRA finetune on source task $T_i$; $U_j$ is the ceiling accuracy of $T_j$ (the paper uses $U_j = 100$ unless stated otherwise); and $\epsilon = 10^{-6}$ keeps the ratio finite when the baseline sits exactly at the ceiling. A value of 1 means the finetune closed the whole gap; 0 means nothing moved; a negative value means accuracy fell.

**The worked example.** The paper's Table 1 uses three toy targets, drawn to scale below.

![Three accuracy gauges: Task A gains 3 points and closes 60% of its headroom, Task B gains 10 points but closes only 18%, Task C loses 1 point and its PGF goes strongly negative](/imgs/blogs/understanding-task-transfer-in-vision-language-models-1.webp)

| Task | Baseline | After FT | Ceiling | Raw gain | PGF |
| --- | --- | --- | --- | --- | --- |
| A | 90 | 93 | 95 | +3 | ${3/5 = 0.60}$ |
| B | 40 | 50 | 95 | +10 | ${10/55 = 0.18}$ |
| C | 98 | 97 | 99 | −1 | see note |

Task B has the largest raw gain and the smallest normalized one. One detail worth catching: the paper's table prints Task C's PGF as −0.50, but with the stated ceiling of 99 the formula gives $-1/(99-98) = -1.0$. The −0.50 is what you get with a ceiling of 100, so I read it as a typo in the toy table, not a different formula. It does not affect any experiment, since all experiments use $U_j = 100$.

**Why it works, and when it breaks.** The paper's own Figure 9 analysis spells out the shape. For gains, PGF is capped at 1. For losses there is no symmetric cap: with $m$ multiple-choice questions, the highest baseline below perfect is $(m-1)/m$, and a collapse to zero from there gives

$$
\mathrm{PGF}_{\min} \;=\; \frac{0 - \tfrac{m-1}{m}}{1 - \tfrac{m-1}{m}} \;=\; -(m-1).
$$

With $m = 200$ questions that is −199. So PGF is asymmetric by construction, which is why the paper analyzes positive and negative transfer separately instead of averaging them. It is also ceiling-sensitive: near-perfect baselines turn small shifts into large values, rewarding late gains and punishing late regressions hard.

### Transferability and malleability: magnitude times breadth

**The problem.** A 13 by 13 PGF matrix is too much to reason about. You want one number per source task ("is finetuning on this good for the model in general?") and one per target task ("how easily does this skill move?"). A plain row mean fails: one huge spike on a single target can outrank a task that helps everything a little.

**The analogy.** Rate a donor not just by the size of their biggest gift but by how many people they helped. A breadth discount makes a one-off windfall score lower than a steady contribution to many.

**The mechanism.** For source task $T_i$, count $p$, the number of targets with $\mu_{i\to j} > 0$, and $n$, the number with $\mu_{i\to j} \lt 0$, out of $N$ targets. Positive and negative transferability are

$$
\Delta^{(i)}_{+} \;=\; \frac{1 - e^{-p/N}}{p}\sum_{j=1}^{N} \mu_{i\to j}\,\mathbb{1}\{\mu_{i\to j} > 0\},
\qquad
\Delta^{(i)}_{-} \;=\; \frac{1 - e^{-n/N}}{n}\sum_{j=1}^{N} \mu_{i\to j}\,\mathbb{1}\{\mu_{i\to j} \lt 0\}.
$$

Read it as two factors: the sum divided by $p$ is the *average* positive PGF over the targets that improved, and ${1 - e^{-p/N}}$ is a breadth weight that grows with the fraction of targets affected. Malleability $\Theta^{(j)}_{\pm}$ is the same formula run down a column instead of along a row: sum over source tasks $i$ for a fixed target $j$, with $p$ and $n$ now counting sources. The definitions accept any $\mu$; the paper uses PGF by default and repeats the analysis with relative gain in its supplementary.

**The worked example.** Take $N = 13$. Source X helps one target with PGF 0.60 and leaves the rest flat. Source Y helps eight targets with PGF 0.15 each.

![Two sources as 13-cell strips: X has one strong cell, Y has eight moderate cells; the breadth weight curve 1 minus exp of minus p over 13 lifts Y above X](/imgs/blogs/understanding-task-transfer-in-vision-language-models-2.webp)

- X: average positive PGF ${0.60}$, weight ${1 - e^{-1/13} \approx 0.074}$, so $\Delta_{+} \approx 0.044$.
- Y: average positive PGF ${0.15}$, weight ${1 - e^{-8/13} \approx 0.460}$, so $\Delta_{+} \approx 0.069$.

Y wins despite a peak four times smaller. That is exactly the behavior the paper wants: "a task with large but isolated transfer effects will be scored lower than one that provides consistent improvements across many targets."

**When it breaks.** Two quirks follow from the formula; neither is discussed in the paper, so treat them as my reading. First, the weight saturates at ${1 - e^{-1} \approx 0.632}$ even when every target improves, so the scale has no natural "1.0 = perfect donor" anchor. Second, the sign test is strict: a target that moved by a noise-sized +0.01 counts toward $p$ as fully as a +0.5. Averaging over four seeds softens this, but there is no significance threshold on individual cells.

### The transfer graph and task cliques

**The mechanism.** Each PGF matrix becomes a directed graph: tasks are nodes, and edge $i \to j$ carries $\mu_{i\to j}$. For the visualizations, the authors keep only the strongest 20% of positive and negative edges (an ablation in the supplementary shows the graph at 25th to 100th percentile cutoffs).

On that graph they define a *task clique*: a set $C$ of tasks such that for every ordered pair of distinct members $T_i, T_j \in C$, the transfer $\mu_{i\to j}$ has the same sign. All positive makes a positive clique (every member helps every other member); all negative makes a negative clique. They test clique stability across seeds with Wilcoxon tests.

The key word is *ordered*. A clique of size $k$ needs $k(k-1)$ directed edges to agree, so a size-9 clique is 72 same-signed transfers. That is a much stronger statement than "these tasks correlate."

## Experimental setup

| Setting | Value |
| --- | --- |
| Base models | Qwen-2.5-VL 3B, 7B, 32B (plus LLaVA-1.5-13B as a qualitative check) |
| Tasks | 13 BLINK perception tasks (IQ Test excluded: no training set) |
| Training data | Rebuilt from BLINK's original source datasets, same task format |
| Evaluation | BLINK validation splits; answers extracted with GPT-4.1 |
| Finetuning | LoRA rank 8, alpha 16, batch 16, cosine schedule, warmup ratio 0.03 |
| Hardware | 8x A100 40GB, DeepSpeed ZeRO-2 (3B, 7B) and ZeRO-3 (32B) |
| Seeds | 4 per configuration |
| Ceiling | $U_j = 100$ by default |

BLINK classifies each task by **perceptual level** (low, mid, high) and **granularity** (pixel, crop, image). The low-level tasks are relative depth, relative reflectance and visual correspondence, all pixel-level. Image-level tasks include art style, counting, forensic detection, multi-view reasoning, spatial reasoning and visual similarity. That two-axis split is what the category analysis aggregates over.

## Results

![Figure 2 from Sachdeva et al. (2026): PGF heatmaps for Qwen-2.5-VL 3B, 7B and 32B; rows are source tasks, columns are target tasks](/imgs/blogs/understanding-task-transfer-in-vision-language-models-fig2.webp)

Each row reads as "what finetuning on this task did to everything." The visual difference between the panels is the headline: at 32B the matrix is mostly warm, at 7B it is scarred with deep negative cells, and at 3B it is mild in both directions.

### Low-level and image-level tasks are the generous ones

Averaged by category, finetuning on **low-level** tasks gives the highest positive transferability at every model size, and low-level tasks are also the most improved targets (highest positive malleability). By granularity, **image-level** sources transfer best, while both image-level and pixel-level targets are the most malleable.

The authors' interpretation is that early-stage visual features are reusable, so training anything that sharpens them spreads benefit. It is a hypothesis drawn from aggregate scores, not a mechanistic test.

### Positive transfer grows with scale; negative transfer does not

![Figure 5 from Sachdeva et al. (2026): average positive transferability rises with model size, while negative transferability shows no consistent trend](/imgs/blogs/understanding-task-transfer-in-vision-language-models-fig3.webp)

The supplementary heatmaps make the scale effect concrete. On the paper's reported scale, the per-source positive transferability at 3B ranges from 1.81 to 5.95; at 32B every source lands between 10.06 and 15.00. At 32B, nine of the thirteen targets have a negative malleability of exactly zero, meaning no source task hurt them on average.

The 7B model is the odd one out on the negative side. Its worst sources reach negative transferability of −24.84 and −17.14 (object localization and functional correspondence rows), far beyond anything at 3B or 32B. The paper reports "no consistent trend" for negative transfer, and the 7B spike is why.

### Personas: donors, pirates, sponges and sieves

The persona labels are a simple thresholding of the aggregates, consistent across model sizes: a task whose transferability (or malleability) beats the model average at every size gets the label.

![A 6 by 6 schematic transfer matrix: the semantic correspondence row is mostly green (donor), functional correspondence mostly red (pirate), visual similarity and relative depth columns green (sponges), forensic detection column red (sieve)](/imgs/blogs/understanding-task-transfer-in-vision-language-models-3.webp)

| Persona | Definition | Task(s) found | Significance |
| --- | --- | --- | --- |
| Donor | Source with above-average positive transferability | Semantic correspondence | $p \lt 0.01$, all models |
| Pirate | Source with above-average negative transferability | Functional correspondence | $p \lt 0.05$, 3B and 7B |
| Sponge | Target with above-average positive malleability | Visual similarity, relative depth, relative reflectance | VS and RD $p \lt 0.001$ |
| Sieve | Target with above-average negative malleability | Forensic detection | $p \lt 0.005$, 3B and 32B |

The donor and pirate pair is the interesting one. Semantic correspondence (match the same part across two different objects) and functional correspondence (match parts by what they *do*) sound like siblings, yet one helps nearly everything and the other hurts. The paper does not explain why. My guess, flagged as speculation, is that functional correspondence's labels push the model toward affordance reasoning that conflicts with the appearance cues most other BLINK tasks rely on.

### Cliques: nine tasks that all help each other

![Figure 6 from Sachdeva et al. (2026): a positive clique of size 9 in Qwen-2.5-VL 32B](/imgs/blogs/understanding-task-transfer-in-vision-language-models-fig4.webp)

In 3B and 7B the largest cliques have 3 or 4 tasks. In 32B there is a positive clique of nine: art style, counting, functional correspondence, jigsaw, relative depth, relative reflectance, spatial relation, visual correspondence and visual similarity. The paper also shows a negative clique of four in 32B: forensic detection, semantic correspondence, multi-view reasoning and object localization hurt each other in every direction.

Two things stand out. Functional correspondence, the pirate at smaller sizes, sits inside the big positive clique at 32B, so personas are not fixed properties of a task. And semantic correspondence, the donor, appears in the negative clique, which means "helps the average target" and "hurts specific targets" coexist.

### Video and other checks

On VSI-Bench (over 5,000 questions from about 288 egocentric indoor videos), the image-trained 3B and 7B checkpoints still move video tasks. Relative reflectance acts as a donor and forensic detection as a pirate; object counting is a sponge, while object appearance order and object relative distance act as sieves. The magnitudes are small: the readable cells in Figure 7 stay within about ±0.15.

Three supplementary checks support the main picture. More finetuning steps raise transferability monotonically while preserving the pattern. LLaVA-1.5-13B shows the same visual similarity, art style and jigsaw clique and the same relative-depth sponge. And the cosine similarity of the final-layer LoRA output projections is highest among visual similarity, jigsaw and art style, which the authors attribute to all three being multi-image tasks.

### Data selection: training a target without its own data

![Figure 8 from Sachdeva et al. (2026): PGF-informed data mixtures versus random mixtures, the baseline, and direct supervision on the target task](/imgs/blogs/understanding-task-transfer-in-vision-language-models-fig5.webp)

The setup: you want to improve task $T$ but have no training data for it, only datasets for other tasks. Pick the sources whose PGF to $T$ clears a threshold, train on that mix, and compare against a random mix of the same sources. The untouched model and $M(T)$, the model finetuned directly on $T$'s data, are the lower and upper reference points.

On Qwen-2.5-VL 7B, PGF-informed mixes beat random mixes across the target tasks shown, and on Jigsaw and Object Localization they beat direct supervision. The authors call this a preliminary finding and say a full study is future work. The paper does not give the threshold or the per-target mixtures in the main text, so I would not treat the size of the gap as settled.

### What is load-bearing in this setup

- **Multiple-choice evaluation.** BLINK answers are a letter. That bounds the output space and, as the authors note, may hide failure modes that open-ended generation would show.
- **LoRA rank 8.** A low-rank update limits how far the model drifts. A full finetune or a higher rank could produce very different negative transfer.
- **The ceiling choice.** $U_j = 100$ ignores that a two-option task has a 50% chance floor. A supplementary ablation swaps $U_j$ for the best observed accuracy on each target, which turns the diagonal into a row of ones since direct supervision usually scores best.
- **One model family.** Three of four models are Qwen-2.5-VL. The LLaVA check is qualitative.

## Critique

**What is strong.** The question is practical and was genuinely unmeasured. Four seeds, reported standard deviations, and significance tests on the persona and clique claims put this above a typical "we ran it once" analysis paper. PGF is simple, its bounds are worked out honestly (including the ugly −199 floor), and the paper keeps positive and negative transfer separate instead of netting them.

**What is weak.**

- The headline structure rests on Qwen-2.5-VL. The 7B model's negative transfer is wildly out of line with 3B and 32B, and the paper does not explain it. The supplementary floats an architectural difference (35 layers at 3B, 27 wider layers at 7B) only for the LoRA cosine analysis.
- PGF ignores chance accuracy. A two-choice task at 55% has 45 points of nominal headroom but only 5 points above guessing. Rescaling to $(\mathrm{Acc} - c)/(U - c)$ with chance level $c$ would be a cheap fix.
- The breadth weight counts any positive cell, however tiny, and its maximum is 0.632, so absolute transferability values are hard to interpret outside a relative ranking.
- The data-selection experiment is one model, an unstated threshold, and no comparison against simpler heuristics like "train on the target's three nearest tasks by LoRA cosine similarity."

**Missing ablations.** Does transfer compose? The paper measures single-source finetunes; it never checks whether the PGF of a two-source mix is close to the sum of the two single-source PGFs. Every practical use of the graph (mixing, curricula) assumes something like additivity. A LoRA-rank sweep and one full-finetune run would also say whether the map belongs to the task or to the adapter.

**What would change my mind.** If a second VLM family at a similar scale (say InternVL or LLaVA-OneVision at 7B) produced a transfer matrix whose rank correlation with Qwen's was low, I would read these maps as model-specific fingerprints rather than structure of the tasks. Conversely, if a 50-pair additivity test showed mixture PGF tracking the sum of pairwise PGFs, I would trust the graph as a planning tool far more than the current single data-selection plot allows.

## Where this goes next

The authors' own closing points at data mixing. Here are concrete research directions that the transfer graph makes tractable now, each with a first experiment. These are my proposals, not the paper's.

**1. Curriculum ordering from the transfer graph.**
*Question:* does the order of multi-task finetuning matter when the tasks have directed transfer? *Why now:* the graph is directed, so it suggests an order: train donors first, then tasks downstream of them, and train sieves last or in isolation. *First experiment:* on Qwen-2.5-VL 7B, train the same total data in three schedules (random interleave, donor-first ordering derived from $\Delta^{(i)}_{+}$, reverse ordering) and report average PGF across all 13 tasks.

**2. A predictive model of mixture transfer.**
*Question:* can you predict the effect of training on a mix of tasks from the pairwise matrix alone? *Why now:* the paper gives the pairwise matrix with seed variance, which is the training set for such a predictor. *First experiment:* at 3B, train 20 random two-task mixes, measure their PGF row, and fit "sum of single-source rows" and "max of single-source rows" models. The residual tells you whether pairwise maps are enough to plan mixtures.

**3. Pirate-aware data mixing.**
*Question:* when a pirate task's data is still needed, can you keep its target gain while removing its collateral damage? *Why now:* the graph names the pirate and its victims, which gives you a concrete protected set. *First experiment:* train on functional correspondence with a replay buffer drawn only from its top-3 negative targets, versus a uniform replay buffer of the same size; compare negative transferability.

**4. Cheap proxies for the full matrix.**
*Question:* the full map costs 13 finetunes times 13 evaluations times 4 seeds per model; can a cheaper signal predict it? *Why now:* the supplementary already shows LoRA output-projection cosine similarity clustering the same multi-image tasks the PGF clique contains. *First experiment:* compute gradient cosine similarity between task losses at the base checkpoint (no finetuning), and report its Spearman correlation with the 32B PGF matrix.

**5. One adapter per clique, merged.**
*Question:* do positive cliques define the right units for adapter merging? *Why now:* a clique is a set of tasks with no internal interference, which is the condition model-merging methods hope for. *First experiment:* train one LoRA on the nine-task 32B clique and one per task, merge the per-task adapters with task arithmetic, and compare both against the clique adapter. The sibling post on [weight disentanglement in task arithmetic](/blog/paper-reading/computer-vision/understanding-and-enforcing-weight-disentanglement-in-task-arithmetic) covers why merging fails when task updates interfere.

## References

- Sachdeva, B., Uppal, K., Java, A., and Balasubramanian, V. N. *Understanding Task Transfer in Vision-Language Models.* CVPR 2026 (Oral). [arXiv:2511.18787](https://arxiv.org/abs/2511.18787). Project page: [aka.ms/task-transfer-vlms](https://aka.ms/task-transfer-vlms).
- Fu, X. et al. *BLINK: Multimodal Large Language Models Can See but Not Perceive.* ECCV 2024. The benchmark behind all 13 tasks.
- Zamir, A. R. et al. *Taskonomy: Disentangling Task Transfer Learning.* CVPR 2018. The supervised-transfer predecessor.
- Yang, J. et al. *Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces.* CVPR 2025. Source of VSI-Bench.
- On this site: [Qwen2-VL and any-resolution perception](/blog/machine-learning/frontier-models/qwen2-vl-enhancing-vision-language-models-perception-of-the-world-at-any-resolution), the model family this study finetunes; [weight disentanglement in task arithmetic](/blog/paper-reading/computer-vision/understanding-and-enforcing-weight-disentanglement-in-task-arithmetic); [data mixing and curriculum](/blog/machine-learning/training-data/data-mixing-domain-weighting-and-curriculum).

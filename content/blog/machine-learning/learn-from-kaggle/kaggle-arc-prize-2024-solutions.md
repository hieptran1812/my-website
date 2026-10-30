---
title: "ARC Prize 2024: how the ARChitects won by training on every test task"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "A synthesis of every published top-20 writeup from Kaggle's ARC Prize 2024: why an 8B model fine-tuned on each hidden task's own demonstrations reached 53.5%, how a 0.5B model and a hand-pruned solver library tied for second at 40, why the 2020 icecuber notebook sat under almost every entry, and what a public-equals-private leaderboard did to the game."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "arc-agi", "test-time-training", "abstract-reasoning", "data-augmentation", "lora", "program-synthesis", "test-time-compute"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 22
---

> **TL;DR.** ARC Prize 2024 (June to November 2024, 1,427 teams) asked an offline Kaggle notebook to solve 100 hidden ARC-AGI-1 puzzles in 12 hours: look at two to five input/output grid pairs, infer the rule, and draw the exact output grid for a new input, with two attempts. **The ARChitects** won with 53.5% by taking Mistral-NeMo-Minitron-8B, shrinking its vocabulary to 64 tokens so each grid cell is one token, pretraining on hundreds of thousands of synthetic tasks, then **fine-tuning a LoRA on the hidden tasks' own demonstrations at test time**, enumerating every answer above a probability cutoff and keeping the two that stayed most probable across eight rotated and flipped views of the task. Second and third tied at 40: a 0.5B model with the same test-time training idea, and a solver library with no ML at all, pruned against the leaderboard. **The lesson:** on genuinely new tasks, the model has to learn at inference time. A frozen model, however large, was not enough.

![The lever strip: ARC Prize 2024 was decided on test-time compute, specifically test-time training plus augmentation scoring, with synthetic data feeding the pretraining stage](/imgs/blogs/kaggle-arc-prize-2024-solutions-1.webp)

The strip above is this series' mental model, introduced in [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work): five levers, one lit per competition. In [AIMO Progress Prize 1](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-1-solutions) test-time compute meant sampling more and voting. In ARC Prize 2024 it meant something stronger: changing the model's weights for every task, inside the scoring notebook. That one idea separated first place from everyone else by 13.5 points.

It is also a thin top-20 synthesis: **only 6 of the top 20 published anything** (ranks 1 to 5 and 13). We go deep on those six, say plainly what is unknown about the other fourteen, and borrow context from the ARC Prize technical report and the paper-award winners.

## 1. The task in 60 seconds

ARC, the Abstraction and Reasoning Corpus, is François Chollet's benchmark for skill acquisition. Each task is a handful of **demonstration pairs**: an input grid and an output grid, up to 30 by 30 cells, each cell one of 10 colours. Somewhere in those pairs is a rule. You get one or more **test inputs** and must produce the output grid exactly: its height, its width and every cell. No partial credit.

![The notebook budget: 12 hours, internet off, two T4s or one P100, 100 hidden tasks with two attempts each, a leaderboard that is the whole hidden set, three submissions a day, about 7.2 minutes per task](/imgs/blogs/kaggle-arc-prize-2024-solutions-2.webp)

The budget card above is the box everyone worked in: at most 12 hours, no internet, public data and models allowed. On accelerators the sources differ: the technical report says "a single P100 GPU" while the ARChitects say they were "constrained by Kaggle's 2 Nvidia T4 GPUs". Both options existed, so this post reports whatever each team reports. 720 minutes over 100 tasks is about **7.2 minutes per task**, and the report puts each entry at roughly USD 10 of compute: intelligence per dollar, not per data centre.

The public data was 400 training tasks and 400 evaluation tasks, all with solutions. The hidden test set was 100 new tasks, and here is the design decision that shaped the whole competition: **the leaderboard was 100% of the hidden set.** The public score was the private score. Hosts cut daily submissions from 5 to 3 on 27 June 2024 to make probing that board harder.

Prizes: USD 25,000, 10,000, 5,000, 5,000 and 5,000 for places 1 to 5, a USD 75,000 paper-award track, and a USD 600,000 grand prize for 85% that nobody came near. One more rule matters later: only teams that open-sourced their solution could be ranked for prizes.

The data is not ours to reproduce, so here is an illustrative task in the same style.

![An illustrative ARC-style task: three demonstration pairs where a coloured shape is mirrored left-right and painted red, a test input, and a lower band showing the 3rd-place trick of solving the geometry on a colour-blind binary mask before searching for the colour](/imgs/blogs/kaggle-arc-prize-2024-solutions-3.webp)

The rule: mirror the shape left-right and paint it red. You inferred that from three examples; a model that has never seen this rule must do the same from about 100 tokens of evidence. The lower band previews a third-place idea: solve the geometry on a colour-blind mask, then search for the colour.

## 2. The metric: exact match, two attempts

For each test output $i$ out of $N$, you submit two attempts $\hat{y}_i^{(1)}$ and $\hat{y}_i^{(2)}$. The output counts only if one of them equals the truth $y_i$ cell for cell, size included:

$$
\text{score} = \frac{1}{N}\sum_{i=1}^{N} \max_{k \in \{1,2\}} \mathbb{1}\left[\hat{y}_i^{(k)} = y_i\right]
$$

where $\mathbb{1}[\cdot]$ is 1 when the grids are identical and 0 otherwise. With 100 hidden tasks, the leaderboard moves in steps of one task, which is why ties are everywhere in the top 20.

![The metric on four toy outputs: attempt 1 exact scores 1, attempt 2 exact scores 1, one wrong cell scores 0, and the right pattern at the wrong size scores 0, for a total of 2 of 4 or 50%](/imgs/blogs/kaggle-arc-prize-2024-solutions-4.webp)

The toy above has four outputs. Row 1 is solved by attempt 1, row 2 by attempt 2. Row 3's second attempt is wrong in exactly one cell and scores 0, the same as a blank grid. Row 4 draws the right pattern inside a 3 by 3 canvas when the answer is 2 by 2; size is part of the answer, so it scores 0 too. Total $\frac{2}{4} = 50\%$.

Two consequences follow. **The second attempt is real value**, so every top design ends in "pick the best two" from a ranked list. And **near-misses are worthless**, which favours enumeration and verification over plausible generation.

## 3. What made it hard

ARC is built so that memorisation fails: a model can learn ARC's vocabulary (objects, symmetry, counting, recolouring), but each hidden task combines it in a new way. Nikola Hu, 13th, put a number on the gap: his early from-scratch transformer scored 30 to 40% on the public evaluation set but about 4 on the hidden set. Accuracy on known distributions said almost nothing about novel ones.

Three constraints turned that into an engineering problem.

| Constraint | What it forced |
|---|---|
| 12 h, 2x T4 (16 GB each) or 1x P100, no internet | An 8B model fits only with 4-bit weights and LoRA |
| Exact match, two attempts | Candidates must be enumerated and ranked; sampling noise is fatal |
| Public = private, 3 submissions a day | The leaderboard leaked the hidden tasks; some used it, some refused |

The first constraint decided the winner. Test-time training means backward passes over every task's demonstrations inside the 12 hours; the ARChitects chose their model to fit 16 GB, one T4, for training and inference with at least 8,000 tokens of context. Barbadillo, 2nd, found 1.5B and 7B models more data-efficient than his 0.5B one but too slow to fine-tune 100 times in 12 hours.

## 4. The top-20 scorecard

Scores are % of the 100 hidden tasks. Public-rank move is "none" for every row because the public board was the private board; the only movements came from teams withdrawing or being removed at the end, covered in section 9.

| Rank | Team | Private | Public-rank move | Key idea | Source |
|---|---|---|---|---|---|
| (n/r) | MindsAI | 55.5 | none | Test-time fine-tuning, AIRV; not open-sourced, not ranked | report |
| 1 | the ARChitects | 53.5 | none | Minitron-8B, 64-token vocab, test-time LoRA, DFS, stability scoring | [paper](https://github.com/da-fr/arc-prize-2024/blob/main/the_architects.pdf) |
| 2 | Guillermo Barbadillo | 40.00 | none | Qwen2.5-0.5B, one LoRA per task, AIRV, plus 2020 DSL | [writeup](https://www.kaggle.com/competitions/arc-prize-2024/writeups/guillermo-barbadillo-2nd-place-solution-for-the-ar) |
| 3 | alijs | 40.00 | none | 26-task notebook plus hand-written functions pruned on the leaderboard; no ML | [writeup](https://www.kaggle.com/competitions/arc-prize-2024/writeups/alijs-3rd-place-solution) |
| 4 | William Wu | 37.00 | none | DSL search, decision trees, CNNs, vote; no LLM | [writeup](https://www.kaggle.com/competitions/arc-prize-2024/writeups/william-wu-4th-place-solution) |
| 5 | PoohAI | 37.00 | none | Solver ensemble, error filters, leaderboard probing | [writeup](https://www.kaggle.com/competitions/arc-prize-2024/writeups/poohai-5th-place-solution) |
| 6 | [Deleted account] | 37.00 | none | unknown | no writeup |
| 7 | vialactea | 36.00 | none | unknown | no writeup |
| 8 | Dawid Kopiczko | 36.00 | none | unknown | no writeup |
| 9 | GOA | 36.00 | none | unknown | no writeup |
| 10 | optima | 35.00 | none | unknown | no writeup |
| 11 | Giotto.ai | 35.00 | none | unknown (company team) | no writeup |
| 12 | Junhua Yang | 30.00 | none | unknown | no writeup |
| 13 | Nikola Hu | 30.00 | none | From-scratch 162M transformer, 2D positions, test-time training | [writeup](https://www.kaggle.com/competitions/arc-prize-2024/writeups/nikola-hu-sharing-my-arc-prize-2024-code-31-points) |
| 14 | OuterloopAI | 30.00 | none | unknown | no writeup |
| 15 | Ahmed El Fazouani | 29.00 | none | unknown | no writeup |
| 16 | h309 = o1411 | 29.00 | none | unknown | no writeup |
| 17 | LIGHT | 29.00 | none | unknown | no writeup |
| 18 | 暗黑AGI | 29.00 | none | unknown | no writeup |
| 19 | Benjamin Kovacs | 29.00 | none | unknown | no writeup |
| 20 | linear regression | 29.00 | none | unknown | no writeup |

![Score ladder of the top 20 plus MindsAI: 53.5 for 1st, a cliff to 40 for 2nd and 3rd, a slow slide to a cluster at 29, the public 26-task notebook as a floor line and the unclaimed 85% grand prize far above](/imgs/blogs/kaggle-arc-prize-2024-solutions-5.webp)

The gap from 1st (53.5) to 2nd (40) is 13.5 points; the whole span from 2nd to 20th (40 to 29) is 11. The dashed line at 26 is the public notebook every classical entry built on, and the six-team cluster at 29 sits just three tasks above it. The 85% grand prize line was never in reach.

## 5. The top-20 landscape

![Technique matrix for the top 20: rows are techniques, columns are private ranks; only ranks 1 to 5 and 13 have sources, every other column is unknown, and the right-hand column counts how many of the 6 sourced teams used each technique](/imgs/blogs/kaggle-arc-prize-2024-solutions-6.webp)

Counted from the matrix cells, across the 6 teams with a source:

- **5 of 6 used the 2020 DSL lineage** (icecuber and other 2020 winners): ranks 2, 3, 4, 5 and 13. Only the winner had no classical component.
- **3 of 6 used a neural transducer with test-time training** (1, 2, 13); all three also used augmentation voting or rescoring and synthetic data.
- **4 of 6 used geometric or colour augmentation** (1, 2, 4, 13); PoohAI's writeup does not say.
- **2 of 6 tuned or probed against the leaderboard** (3, 5); rank 2 used extra submissions only as compute.
- **2 of 6 shrank the vocabulary** to fit the grid: 64 tokens for rank 1, 19 for rank 13.

**Table stakes** was not a model but a public notebook, Gholami and Kazeminia's "3-arc24-developed-2020-winning-solutions": icecuber's 2020 C++ program search plus other 2020 solvers, 26 on the hidden set on CPU. Every team that wrote up, except the winner, ensembled with it or its components. The 29-point cluster is plausibly that notebook plus tweaks (Lyrialtus, 21st with 28, documented a recolouring preprocessor on top of icecuber), but for the 14 silent teams that is a guess.

**What separated rank 1 from 2 to 20** is one row: test-time training on an 8B pretrained model, with enumeration and stability scoring, and no DSL. Whether anything separates the top 5 from 6 to 20 is unknowable: ranks 3 to 11 span five tasks, and nobody from 6 to 12 wrote up.

**Contradictory findings.** Three disagreements run through the six sources:

1. **Use the leaderboard or not.** alijs (3rd) pruned against it and PoohAI (5th) probed it; Barbadillo (2nd) called probing "totally against the spirit of the competition".
2. **Do LLMs help a classical ensemble?** William Wu (4th) placed with zero LLMs yet concluded test-time training is the state of the art; PoohAI thought an LLM would need fine-tuning to help; Barbadillo gained by pairing his LLM's first attempt with the DSL's.
3. **Pretrained or from scratch?** Ranks 1 and 2 fine-tuned pretrained LLMs; Nikola Hu (13th) reached 30 with a 162M model trained from scratch in a month.

## 6. The winning solution: the ARChitects (53.5)

Daniel Franzen and Jan Disselhoff are the Kaggle roster; David Hartmann co-authored the paper, "The LLM ARChitect: Solving ARC-AGI Is A Matter of Perspective". On 100 public-evaluation tasks they held out (the other 300 went into training) the system solves 72.5. On the hidden set it scored 53.5, and 56.5 in a run that finished after the deadline.

![The ARChitects' pipeline in five stages: synthetic data, pretraining an 8B model with a 64-token vocabulary on one H100, test-time LoRA training on each hidden task's demonstrations across two T4s, depth-first generation of all answers above a probability cutoff, and selection by re-scoring each candidate under eight augmentations](/imgs/blogs/kaggle-arc-prize-2024-solutions-7.webp)

The title says it: every stage looks at the task from several symmetric views, and the model is both generator and judge.

### A tokenizer built for grids

A general tokenizer chops `0 0 8 8` into unpredictable pieces. The ARChitects shrank the tokenizer and embeddings to **64 symbols**, so every cell is exactly one token (see [task-specialised tokenizer design](/blog/machine-learning/large-language-model/task-specialized-tokenizer-design)). A 30 by 30 grid then costs 900 tokens plus row breaks, which is what makes 8,000 tokens enough for a whole task.

### Data: replace the training set with generators

The 400 public training tasks are too few. They replaced them with **Re-ARC**, Michael Hodel's procedural generators that produce unlimited fresh examples of each training rule, then added 75% of the public evaluation set, ConceptARC, and **ARC-Heavy** (BARC's roughly 200k LLM-generated tasks, covered in section 8). The best private model, "Nemo-heavy", trained on 531,318 examples.

Augmentation runs through every stage: the 8 symmetries of the square (the D8 group), colour permutations, and shuffled demonstration order. A rotated task is still a valid task with the rotated answer, so the model never sees the same sequence twice.

### Pretraining, then test-time training

Preliminary training used LoRA rank 256 on all layers including the input and output embeddings, learning rate 1e-4 (different for embeddings), 4-bit weights and gradient checkpointing via [unsloth's 4-bit path](/blog/machine-learning/open-source-library/unsloth-4bit-quantization-nf4), on one H100. The base was **Mistral-NeMo-Minitron-8B-Base**, chosen because it fit 16 GB with the context they needed.

On Kaggle, a **second LoRA of rank 64** is trained on the hidden tasks' demonstration pairs, augmented, time-boxed, with the loss computed on all outputs. This turns "a model that knows ARC" into "a model that has practised this task". A separate fine-tune per task helped slightly but cost too much runtime, so they trained on the test set's demonstrations jointly and split the test set across the two T4s, running the whole pipeline in parallel on each half. Each GPU then has 720 minutes for 50 tasks, about 14.4 minutes per task.

There is no leakage: the demonstrations are part of the task's input, and training on them is what a human does when studying examples. The [LoRA debugging post](/blog/machine-learning/debugging-training/debugging-lora-and-peft) lists what to check before trusting a per-task adapter.

### Generation: enumerate, do not sample

Greedy decoding returns one answer; sampling returns noisy duplicates. With exact match you want **every answer plausible enough to rank**. The ARChitects wrote a depth-first search that enumerates every completion whose cumulative probability stays above a cutoff (their figure uses 5%), reusing the KV cache along each branch, and runs it on the original and augmented versions of each task. It scored better and ran faster than sampling.

### Selection: the model as its own judge

Then each candidate is re-scored: transform the task and the candidate under the 8 augmentations, compute the model's probability of the transformed candidate in each view, and rank by the **product** of those probabilities (the sum of log-softmax). A true answer should stay probable from every perspective; an artefact of one view should not.

![Two candidates scored under eight views: candidate Ay1 has the higher probability on the original view but collapses on rotations, while candidate Ax1 stays steady and wins on the sum of log-probabilities, minus 17.49 against minus 21.47](/imgs/blogs/kaggle-arc-prize-2024-solutions-8.webp)

In the figure, Ay1 beats Ax1 on the original view (0.165 against 0.1375) but drops to 0.02 or 0.03 on several rotations; Ax1 never drops below 0.09. Summed log-probabilities flip the order. The paper reports this selection was about 25% better than their baseline selection for 10 to 17 extra minutes per 100 tasks on an H100, and the DFS log-probabilities could be reused in scoring for a further small gain.

Here is the whole generate-and-select idea as an illustrative toy. It is not the ARChitects' code: the "model" is a hard-coded table of next-token probabilities, and the eight view scores are invented to match the figure.

```python
# Illustrative toy, not the ARChitects' code: depth-first enumeration of every
# completion above a probability cutoff, then re-scoring under augmentations.
import math

# A toy "model": next-token distribution given the prefix of a 3-token answer.
NEXT = {
    (): {"A": 0.55, "B": 0.45},
    ("A",): {"x": 0.50, "y": 0.50},
    ("B",): {"x": 0.95, "y": 0.05},
    ("A", "x"): {"1": 0.50, "2": 0.50},
    ("A", "y"): {"1": 0.60, "2": 0.40},
    ("B", "x"): {"1": 0.90, "2": 0.10},
    ("B", "y"): {"1": 0.50, "2": 0.50},
}

def greedy():
    seq, p = (), 1.0
    while seq in NEXT:
        tok, q = max(NEXT[seq].items(), key=lambda kv: kv[1])
        seq, p = seq + (tok,), p * q
    return "".join(seq), round(p, 4)

def dfs(prefix=(), p=1.0, cutoff=0.05, out=None):
    out = [] if out is None else out
    if prefix not in NEXT:              # finished answer
        out.append(("".join(prefix), round(p, 4)))
        return out
    for tok, q in NEXT[prefix].items():
        if p * q >= cutoff:             # prune any branch already below the cutoff
            dfs(prefix + (tok,), p * q, cutoff, out)
    return out

print("greedy :", greedy())
cands = sorted(dfs(), key=lambda c: -c[1])
print("dfs>=5%:", cands)

# Re-score each candidate under 8 views of the task (invented probabilities).
VIEWS = {
    "Bx1": [0.3847, 0.31, 0.35, 0.30, 0.37, 0.33, 0.36, 0.32],
    "Ay1": [0.165, 0.02, 0.21, 0.03, 0.15, 0.02, 0.19, 0.04],
    "Ax1": [0.1375, 0.10, 0.12, 0.11, 0.13, 0.09, 0.12, 0.10],
    "Ax2": [0.1375, 0.05, 0.11, 0.04, 0.12, 0.06, 0.10, 0.05],
    "Ay2": [0.11, 0.03, 0.09, 0.02, 0.10, 0.03, 0.08, 0.02],
}
score = {c: sum(math.log(p) for p in VIEWS[c]) for c, _ in cands}
for c, s in sorted(score.items(), key=lambda kv: -kv[1]):
    print(f"{c}: sum log p = {s:7.2f}")
print("two attempts:", sorted(score, key=score.get, reverse=True)[:2])
```

```console
greedy : ('Ax1', 0.1375)
dfs>=5%: [('Bx1', 0.3847), ('Ay1', 0.165), ('Ax1', 0.1375), ('Ax2', 0.1375), ('Ay2', 0.11)]
Bx1: sum log p =   -8.64
Ax1: sum log p =  -17.49
Ax2: sum log p =  -20.64
Ay1: sum log p =  -21.47
Ay2: sum log p =  -24.28
two attempts: ['Bx1', 'Ax1']
```

Greedy commits to `A` (0.55 beats 0.45) and never sees `Bx1`, the most probable full answer at 0.3847; DFS finds all five answers above 5%. Stability scoring then keeps `Bx1` and promotes `Ax1` over `Ay1`, although `Ay1` was more probable on the original view.

## 7. Second and third place

### 2nd: Guillermo Barbadillo, Omni-ARC (40.00)

Barbadillo's thesis is that **representation is the hard part**, so one small model trains on many ARC-derived tasks: examples plus input to output, and generating a new valid input from the examples, which forces it to learn the input distribution.

Data: 1,420 unique tasks (ARC, RE-ARC, Mini-ARC and four small sets). Beyond rotations, flips, colour changes and train/test swaps he added **problem augmentation**: compose a task with a reversible transform (rotate, flip, pad, upscale, mirror) applied to inputs only or outputs only. Grids were written as Markdown with a `grid shape: HxW` line and numbered rows, not a custom vocabulary.

Training: **Qwen2.5-0.5B-Instruct**, LoRA rank 128, learning rate 5e-5 with warmup and linear decay, batch 16, 2e5 steps, max length 8196, on 2x A6000. A full fine-tune did no better.

The decisive part is **test-time fine-tuning, one model per test task**: 100 separate fine-tunes per submission. For a task with $n$ demonstrations he trains leave-one-out, using $n-1$ pairs as examples and the remaining one as the target, with augmentation, batch size 1, about 300 steps at learning rate 8e-5, continuing from the trained LoRA. That single change took one model **from 11 to 33 solved tasks**, a 3x gain, and it is the cleanest ablation of test-time training in any 2024 writeup.

Inference uses **AIRV** (augment, infer, reverse, vote), the scheme MindsAI described publicly: 96 predictions per task at temperature 0 in vLLM, each made on an augmented view and mapped back, then a vote. The final submission took the first attempt from his model and the first attempt from the public 2020 DSL solution running on CPU in parallel. Here is AIRV as a toy, with a stub "model" that knows the rule but garbles two views:

```python
# Illustrative toy of augment-infer-reverse-vote (AIRV). Not Barbadillo's code:
# the "model" is a stub that knows the rule but garbles some views.
from collections import Counter
import numpy as np

D8 = [  # (forward, inverse) pairs for the 8 symmetries of the square
    (lambda g: g, lambda g: g),
    (lambda g: np.rot90(g, 1), lambda g: np.rot90(g, -1)),
    (lambda g: np.rot90(g, 2), lambda g: np.rot90(g, -2)),
    (lambda g: np.rot90(g, 3), lambda g: np.rot90(g, -3)),
    (np.fliplr, np.fliplr),
    (np.flipud, np.flipud),
    (np.transpose, np.transpose),
    (lambda g: np.rot90(g, 2).T, lambda g: np.rot90(g.T, -2)),
]

def true_rule(g):            # the hidden task: paint every non-background cell red (2)
    return np.where(g > 0, 2, 0)

def stub_model(view, k):     # right on most views, wrong on views 3 and 6
    out = true_rule(view)
    if k in (3, 6):
        out = out.copy(); out[0, 0] = 5
    return out

test = np.array([[8, 0, 0], [8, 8, 8], [0, 0, 8]])
votes = Counter()
for k, (fwd, inv) in enumerate(D8):
    pred = inv(stub_model(fwd(test), k))      # predict in the view, map back
    votes[pred.tobytes()] += 1
ranked = votes.most_common()
print("distinct answers:", len(ranked), "| vote counts:", [c for _, c in ranked])
best = np.frombuffer(ranked[0][0], dtype=test.dtype).reshape(test.shape)
print("attempt 1 correct:", np.array_equal(best, true_rule(test)))
```

```console
distinct answers: 3 | vote counts: [6, 1, 1]
attempt 1 correct: True
```

The two errors land on different cells once mapped back, so they split their votes while the six correct views agree: correct answers are invariant across views, mistakes usually are not.

His negative results (section 10) are as useful as his method; the sharpest is that a trained verifier beat voting offline and gave nothing on the leaderboard, likely because test-time-trained models already vote well. One practical trick: he treated the three daily submissions as extra compute, which his writeup counts as 36 hours a week against the 30-hour weekly notebook GPU quota.

### 3rd: alijs (40.00), no machine learning at all

Agnis Liukis's notebook summary describes an ensemble of two parts: the best public notebook (the 26-point icecuber-based solver set), and his own code, which **added 14 points**. His part is "some kind of guided brute-force": a library of hand-written Python functions, each tried against the task's demonstration pairs; any function that reproduces at least one demonstration is applied to the test input.

The development loop was the leaderboard. Because public equalled private, he used leaderboard feedback to tune functions toward specific hidden tasks, and deleted every function or code path that did not raise the score. Deleting dead code cut runtime, which is what made room to run the 26-point solver in the same 12-hour notebook. He says plainly that the result is "extremely overfitted to the given test-set" and congratulates the teams that resisted doing the same.

He focused on symmetry and pattern-completion tasks and on recolouring by row or column rule. The idea that transfers is in figure 3: many tasks are **a known reasoning step plus one small extra step** (mirror the shape and also recolour it; draw the spiral and also colour it). So solve first in a **colour-independent binary view**, background versus not background, and only if the geometry fits, search separately for the colouring. A toy version, again not his code:

```python
# Illustrative toy of "guided brute force" in the spirit of the 3rd-place notes.
# Not alijs's code. Step 1: solve geometry on a binary mask. Step 2: search colours.
import numpy as np

GEOMETRY = {                      # a tiny library of hand-written candidate steps
    "identity": lambda m: m,
    "mirror_lr": np.fliplr,
    "mirror_ud": np.flipud,
    "transpose": np.transpose,
}

def solve(demos, test):
    for name, f in GEOMETRY.items():
        # colour-independent view: background (0) vs anything else (1)
        if not all(np.array_equal(f(i > 0), o > 0) for i, o in demos):
            continue
        for colour in range(1, 10):           # geometry fits; now find the paint
            paint = lambda g: np.where(f(g > 0), colour, 0)
            if all(np.array_equal(paint(i), o) for i, o in demos):
                return name, colour, paint(test)
    return None

demos = [
    (np.array([[1,0,0,0],[1,1,0,0],[1,0,0,0],[0,0,0,0]]),
     np.array([[0,0,0,2],[0,0,2,2],[0,0,0,2],[0,0,0,0]])),
    (np.array([[3,3,0,0],[0,3,0,0],[0,0,0,0],[0,0,0,0]]),
     np.array([[0,0,2,2],[0,0,2,0],[0,0,0,0],[0,0,0,0]])),
    (np.array([[0,4,0,0],[4,4,4,0],[0,0,0,0],[0,0,0,0]]),
     np.array([[0,0,2,0],[0,2,2,2],[0,0,0,0],[0,0,0,0]])),
]
test = np.array([[8,0,0,0],[8,8,8,0],[0,0,8,0],[0,0,0,0]])
name, colour, out = solve(demos, test)
print(name, "then paint colour", colour)
print(out)
```

```console
mirror_lr then paint colour 2
[[0 0 0 2]
 [0 2 2 2]
 [0 2 0 0]
 [0 0 0 0]]
```

Splitting the search turns a product (4 geometries times 9 colours) into a sum: geometry is checked once on the mask, and colour is searched only for geometries that pass. With a real library of hundreds of functions, that factorisation is the difference between fitting in 12 hours and not.

The ARC technical report credits this as the best single-submission brute-force or DSL score, 40%. But fourteen points were bought partly with leaderboard feedback on the very tasks being scored: legal as written, and one reason the 2025 edition split its leaderboard (see the [ARC Prize 2025 post](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2025-solutions)).

## 8. Ranks 4 to 20, by approach family

### Classical ensembles, no LLM: William Wu (4th) and PoohAI (5th), both 37.00

**William Wu** ranked his components: DSL search over a DAG of transformations, decision trees, CNNs, then augmentation (diagonal flips, colour switching, symmetry). Solvers came from 2020 winners (icecuber, golubev, szabo7zoltan, ilialar, mehrankazeminia, somayyehgholami) plus Hodel's arc-dsl. Why he beat 2020: the bigger box, 30 GB RAM and 12 hours instead of 9. Ensembling was a majority vote with icecuber prioritised and exempt, plus "probabilistic trials" that pick a model roughly in proportion to how many new problems it solves. He was 7th at the deadline, 4th after withdrawals.

**PoohAI** (four members) ensembled solvers, added filters for each solver's typical mistakes (the genetic-algorithm solver draws redundant lines; others produce the wrong shape), and **probed the leaderboard**: tasks ran in the same ID-sorted order every time, so within 100 submissions they identified which 26 tasks the public notebook solves, then located each new solve by binary search. Their lesson generalises: "removing wrong attempts is easier" than picking the right one.

### From-scratch transduction: Nikola Hu (13th, 30.00)

The only from-scratch model in the top 20: a 162M-parameter decoder, 18 layers, 4 heads, built in the final month on 3x RTX 4090. The vocabulary has 19 tokens (10 colours plus structural markers such as ROW_SEPARATOR), and each token carries a **2D positional encoding**: row, column, a topology index and a sample index from learned tables, concatenated, which he reports beat 1D encoding. Data was Re-ARC and BARC.

Test-time training ("active inference", adopted from a public MindsAI interview) was the "game-changer" when he was stuck at 4 points; he also arrived independently at AIRV-style reverse augmentation. His 31 in the writeup (30 on the board) is an ensemble with the public notebook, split unstated.

### No writeup: ranks 6 to 12 and 14 to 20

Fourteen teams, including a deleted account at 6th, the company team Giotto.ai at 11th and OuterloopAI at 14th, published nothing in the forum, in the writeup list or on arcprize.org. We do not know their methods. One weak hint: rank 20, yuanzhe zhou, asked in a forum thread whether C++ solvers were "the only valid way", noting that the best public notebook is icecuber-based, and a reply estimated that tree models solve only 3 to 5 hidden tasks. That is indicative of the field's mood, not of anyone's final submission.

### The contrasting entries: off Kaggle

The paper awards show what the Kaggle box hid. First prize went to **BARC** (Li et al.): 100 hand-written seed programs remixed by GPT-4o-mini into about 200k (ARC-Heavy) and 400k synthetic tasks, then Llama3.1-8B trained two ways. **Induction** (write a Python program, sample up to 20k, vote by execution) reached 38.0% of the 400 public evaluation tasks; **transduction** (predict the grid directly, with test-time training) 43.0%; the ensemble 56.75%. Induction wins on precise composition, transduction on fuzzy perceptual concepts. A scaled-down Kaggle version scored only 19% private.

Second prize, Akyürek et al., measured test-time training directly: up to 6x over fine-tuned baselines and 53.0% on public evaluation with an 8B model. On the off-Kaggle ARC-AGI-Pub board, with up to USD 10,000 of API compute, frontier models used as-is scored 18% (o1-preview) and 5% (GPT-4o) on the semi-private set.

## 9. What the forum taught everyone

- **"Daily Submission Limit Change"** (Greg Kamradt, [discussion/515322](https://www.kaggle.com/competitions/arc-prize-2024/discussion/515322)): 5 a day became 3, to discourage probing.
- **The public 26-task notebook** ([mehrankazeminia](https://www.kaggle.com/code/mehrankazeminia/3-arc24-developed-2020-winning-solutions)) became the classical floor.
- **"What happened to first place?"** ([discussion/550442](https://www.kaggle.com/competitions/arc-prize-2024/discussion/550442)): MindsAI's 55.5 was removed from the prize ranking for not open-sourcing.
- **The torino disappearance** ([discussion/545733](https://www.kaggle.com/competitions/arc-prize-2024/discussion/545733)): a top-3 team vanished the day before the deadline. CPMP suspected multiple accounts used to beat the 3-a-day cap; the host never confirmed a cause, so treat it as suspected only.
- **The shake-up that was not**: public and private ranks are identical for all 20. Only withdrawals moved anyone: William Wu went from 7th to 4th, PoohAI from about 8th to 5th.
- **Probing as ethics**: Barbadillo suggested an API that releases one task only after the previous answer, the design [AIMO 1](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-1-solutions) used.

## 10. What did not work

| Tried | Result | Reported by |
|---|---|---|
| Greedy or multinomial sampling for candidates | Worse score and slower than DFS with a probability cutoff | the ARChitects |
| Verifier/selector model | 81.0% top-2 offline vs 70.0% for voting, no leaderboard gain | Barbadillo (2nd) |
| Code generation from 285 hand-written DSL tasks | Solved 5% of evaluation | Barbadillo (2nd) |
| BARC's 400k synthetic tasks | No gain | Barbadillo (2nd) |
| Smaller models (135M, 0.3B) / larger (1.5B, 7B) | Worse / too slow for 12 h | Barbadillo (2nd) |
| Picking the right attempt among candidates | Harder than filtering out wrong ones | PoohAI (5th) |
| Full attention among input tokens ("mask hack") | No gain, disabled | Nikola Hu (13th) |
| High public-eval accuracy without test-time training | 30 to 40% public eval, about 4 hidden | Nikola Hu (13th) |
| Recolouring preprocessing for icecuber | Helps some tasks, "ruins some tasks" | Lyrialtus (21st) |
| Graph-based and neural-cellular-automata approaches | Under 1% | Lyrialtus (21st) |

The pattern: making a frozen model smarter (verifiers, more data, bigger models) lost to letting it briefly specialise on the task in front of it.

## 11. Lessons that transfer

1. **When the test distribution is new by construction, train at test time.** 11 to 33 tasks in Barbadillo's ablation; up to 6x in the TTT paper. It is [test-time compute](/blog/machine-learning/scaling-laws/test-time-compute-scaling) spent on gradients instead of samples.
2. **Exploit the task's symmetry group everywhere**, including scoring: rank candidates by stability across views (ARChitects; Barbadillo's AIRV).
3. **Fit the tokenizer to the domain.** One token per cell, 64 symbols (rank 1) or 19 (rank 13), makes grids dense and unambiguous.
4. **Enumerate, then rank, when the metric is exact match.** DFS above a cutoff found answers greedy decoding never reaches (ARChitects paper).
5. **Factor the search.** Solve geometry colour-blind, then search colours (alijs); induction and transduction solve different tasks, so ensemble them (BARC).
6. **If the hidden set is the leaderboard and order is fixed, the leaderboard leaks.** Decide your ethics before the deadline, and as a host, do not design it that way (PoohAI; discussion/515322).

### If you enter a competition like this

- Hold out real evaluation tasks; public-eval accuracy without test-time adaptation did not predict hidden scores (Nikola Hu).
- Budget the 12 hours per task first: test-time training steps, candidates per task, and scoring passes must fit.
- Write a custom tokenizer or grid format before choosing a model size.
- Make the second attempt count: always produce a ranked candidate list.
- Ensemble a symbolic solver's first attempt with your model's; the methods fail on different tasks.
- Keep a probe-free validation loop. The [Google Code Golf 2025](/blog/machine-learning/learn-from-kaggle/kaggle-google-code-golf-2025-solutions) and [NeuroGolf 2026](/blog/machine-learning/learn-from-kaggle/kaggle-neurogolf-2026-solutions) posts attack the same tasks from the program side; ARC Prize 2026 gets its own post after it closes.

## 12. Sources

- Competition: https://www.kaggle.com/competitions/arc-prize-2024 ; technical report, arXiv 2412.04604, https://arcprize.org/media/arc-prize-2024-technical-report.pdf
- 1st, the ARChitects: paper https://github.com/da-fr/arc-prize-2024/blob/main/the_architects.pdf ; code https://github.com/da-fr/arc-prize-2024 ; re-run notebook https://www.kaggle.com/code/gregkamradt/arc-prize-v8
- 2nd, Guillermo Barbadillo: https://www.kaggle.com/competitions/arc-prize-2024/writeups/guillermo-barbadillo-2nd-place-solution-for-the-ar ; https://ironbar.github.io/arc24/05_Solution_Summary/ ; https://github.com/ironbar/arc24 ; https://www.kaggle.com/code/ironbar/single-task-test-time-fine-tuning-for-arc24
- 3rd, alijs: https://www.kaggle.com/competitions/arc-prize-2024/writeups/alijs-3rd-place-solution ; https://www.kaggle.com/code/alijs1/arc-prize-2024-solution-3rd-place-score-40
- 4th, William Wu: https://www.kaggle.com/competitions/arc-prize-2024/writeups/william-wu-4th-place-solution ; https://www.kaggle.com/code/williamwu88/fork-of-small-sample-arc24-7d97ca
- 5th, PoohAI: https://www.kaggle.com/competitions/arc-prize-2024/writeups/poohai-5th-place-solution ; https://www.kaggle.com/code/gromml/arc-prize-2024-poohai-solution
- 13th, Nikola Hu: https://www.kaggle.com/competitions/arc-prize-2024/writeups/nikola-hu-sharing-my-arc-prize-2024-code-31-points ; https://www.kaggle.com/competitions/arc-prize-2024/discussion/545747 ; https://github.com/zoenguyenramirez/arc-prize-2024
- 21st, Lyrialtus: https://www.kaggle.com/competitions/arc-prize-2024/discussion/550209
- Threads: https://www.kaggle.com/competitions/arc-prize-2024/discussion/515322 ; https://www.kaggle.com/competitions/arc-prize-2024/discussion/534826 ; https://www.kaggle.com/competitions/arc-prize-2024/discussion/550442 ; https://www.kaggle.com/competitions/arc-prize-2024/discussion/545733 ; https://www.kaggle.com/competitions/arc-prize-2024/discussion/542428
- Public baseline: https://www.kaggle.com/code/mehrankazeminia/3-arc24-developed-2020-winning-solutions
- Papers: BARC, arXiv 2411.02272; Test-Time Training, arXiv 2411.07279

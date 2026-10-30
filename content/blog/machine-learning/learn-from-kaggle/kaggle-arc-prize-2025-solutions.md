---
title: "ARC Prize 2025: what ARC-AGI-2 changed, and the fork-of-a-fork leaderboard"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "ARC-AGI-2 cut the winning Kaggle score from 53.5% to 24.03%. Only the top 3 teams pretrained their own model; most of the rest of the top 20 ran lightly tuned forks of one 2024 notebook inside a two-task noise band. This is what every documented top-20 entry did, and what transfers."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "arc-prize", "arc-agi", "test-time-training", "synthetic-data", "diffusion-language-models", "abstract-reasoning", "evaluation-noise"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 23
---

> **TL;DR.** ARC Prize 2025 ran the 2024 format (predict an exact output grid, two attempts) on a new, harder benchmark, ARC-AGI-2, with twice the compute: four L4 GPUs for 12 hours. NVARC (Ivan Sorokin and Jean-François Puget of NVIDIA) won with 24.03% private, ahead of the ARChitects (16.53%) and MindsAI & Tufa Labs (12.64%). Those three were the only teams that pretrained their own model, and they were the only teams above 7%. Ranks 4 to 20 all scored between 5.00 and 6.67, a band two tasks wide, and every documented one of them is a lightly tuned fork of the ARChitects' 2024 public notebook. **Decisive lever: data, specifically off-Kaggle pretraining on synthetic puzzles.** **The one lesson:** when a benchmark gets harder, the recipe that won last year still runs, but it becomes the floor, and the rank order above that floor is mostly seed noise.

![The lever strip: ARC Prize 2025 was decided by data and off-Kaggle pretraining, with per-task test-time training as table stakes](/imgs/blogs/kaggle-arc-prize-2025-solutions-1.webp)

The strip above is the mental model for this post. In [ARC Prize 2024](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2024-solutions) the decisive lever was test-time compute: the ARChitects showed that fine-tuning a language model on each puzzle's own demonstrations, then searching and rescoring its answers, could reach 53.5% on ARC-AGI-1. In 2025 that idea became table stakes. What separated the podium from everyone else happened before the notebook started: weeks of H100 or TPU time building a model that already knew what ARC puzzles look like.

So this post has two stories: what ARC-AGI-2 changed and what the podium added to beat it, and a leaderboard where six of nine documented top-20 entries share one ancestor notebook and a seed change moved a team from 343rd public to 4th private. The series opener, [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work), explains the notebook format this post assumes.

## 1. The task in 60 seconds

![The 2024 and 2025 budget cards side by side: four L4s instead of two T4s, 120 private tasks instead of 100, and a winning score that fell from 53.5% to 24.03%](/imgs/blogs/kaggle-arc-prize-2025-solutions-2.webp)

An ARC task is a tiny visual puzzle: a few demonstration pairs of input and output grids (up to 30 by 30 cells, 10 colours) and one or more test inputs. You infer the rule and produce each test output exactly, right shape and every cell right. No text, no instructions, no partial credit.

ARC Prize 2025 ran on Kaggle from late March to 2025-11-03, with winners announced on 2025-12-05. The host changed four things from 2024. The benchmark moved from ARC-AGI-1 to ARC-AGI-2, a new set of human-calibrated tasks, each solved by at least two humans in testing; at launch the overview said humans had collectively scored 100% while "the best AI systems only score 4%". Compute doubled: Kaggle's L4x4 machines (four L4s, 96 GB in total, roughly USD 50 per submission) were made available only to this competition, with internet off and a 12-hour runtime. The leaderboard was split: about 240 hidden tasks, 120 scored on the public board and 120 on the private (Lonnie's writeup; the API reports a 50% public share). Winners had to open-source before final standings. The card compares both years: 1,427 teams in 2024 and 1,455 in 2025, one submission a day instead of three.

The USD 1,000,000 pool paid USD 25k, 10k, 5k, 5k and 5k for places 1 to 5, USD 50k, 20k and 5k in paper awards, and held a USD 700k Grand Prize for 85% that nobody came close to.

## 2. The metric, with a worked toy score

![An illustrative diagonal-extension toy task: attempt 1 misses one cell and scores 0, attempt 2 matches exactly and scores 1; four toy tasks total 2.5 of 4](/imgs/blogs/kaggle-arc-prize-2025-solutions-3.webp)

Each test output gets two attempts. If either attempt matches the true grid exactly, that test output counts as solved. A task's score is the fraction of its test outputs solved, and the competition score is the mean task score across the hidden tasks, reported as a percentage:

$$
\text{score} = \frac{100}{T}\sum_{t=1}^{T}\frac{1}{K_t}\sum_{k=1}^{K_t}\mathbb{1}\left[\hat{y}^{(1)}_{t,k} = y_{t,k} \;\lor\; \hat{y}^{(2)}_{t,k} = y_{t,k}\right]
$$

Here $T$ is the number of tasks, $K_t$ the number of test outputs in task $t$, $y_{t,k}$ the true grid and $\hat{y}^{(1)}, \hat{y}^{(2)}$ the two attempts. Equality means same shape and every cell equal.

The figure works the rule on an illustrative toy task (not an ARC-AGI-2 task): extend each coloured seed diagonally down-right to the edge. Attempt 1 stops one cell short, 24 of 25 cells right, and scores zero. Attempt 2 is exact. The toy below scores four such tasks:

```python
# Illustrative toy, not any team's code: the ARC Prize scoring rule.
# Each task has one or more test outputs; each test output gets 2 attempts.
# A test output counts only on an exact match of shape and every cell.

def exact(a, b):
    return a == b  # lists of lists: same shape and same cells

def task_score(attempts, truths):
    """attempts: one [attempt_1, attempt_2] pair per test output."""
    hits = [any(exact(t, y) for t in pair) for pair, y in zip(attempts, truths)]
    return sum(hits) / len(truths)

G = 3  # green
truth = [[0]*5 for _ in range(5)]
for r, c in [(1, 0), (2, 1), (3, 2), (4, 3)]:
    truth[r][c] = G
almost = [row[:] for row in truth]
almost[4][3] = 0                     # one cell short of the diagonal

scores = {
    "A": task_score([[almost, truth]], [truth]),                 # attempt 2 hits
    "B": task_score([[almost, almost]], [truth]),                # 24 of 25 cells is still 0
    "C": task_score([[truth, almost], [almost, almost]], [truth, truth]),  # 1 of 2 outputs
    "D": task_score([[truth, almost]], [truth]),                 # attempt 1 hits
}
print(scores)
print(f"total = {sum(scores.values())} / {len(scores)} = {100*sum(scores.values())/len(scores):.1f}%")
print(f"one task on a 120-task split = {100/120:.2f} points")
```

```console
{'A': 1.0, 'B': 0.0, 'C': 0.5, 'D': 1.0}
total = 2.5 / 4 = 62.5%
one task on a 120-task split = 0.83 points
```

The last line explains half of this post. On a 120-task split, one solved single-output task is worth ${0.83}$ points. Ranks 4 to 20 spanned 5.00 to 6.67, a difference of 1.67 points: exactly two tasks. No partial credit on a set that small turns the leaderboard into a coarse staircase, and one lucky sample is a whole step.

## 3. What made it hard: what ARC-AGI-2 changed

ARC-AGI-2 was built to defeat the 2024 recipe, and it did, unevenly:

| Place | 2024 (ARC-AGI-1) | 2025 (ARC-AGI-2) | Ratio |
|---|---|---|---|
| 1st | the ARChitects 53.5 | NVARC 24.03 | 2.2x |
| 2nd | Barbadillo 40.0 | the ARChitects 16.53 | 2.4x |
| 3rd | alijs 40.0 | MindsAI & Tufa Labs 12.64 | 3.2x |
| 4th | William Wu 37.0 | Lonnie 6.67 | 5.5x |
| 5th | PoohAI 37.0 | Barbadillo 6.53 | 5.7x |

The podium lost a factor of two to three; places 4 and 5 lost five to six. Teams that invested in new pretraining recovered part of the gap. Teams that carried the 2024 recipe over did not, even with twice the GPUs.

Why did the old recipe break? The writeups give three mechanisms.

**The tasks need global edits.** The ARChitects named their 2024 pipeline's weakness: tasks needing global restructuring, such as diagonals and simulations, defeat a model that writes one token at a time and cannot revise an emitted cell.

**Test-time training stopped compounding.** MindsAI, who pioneered test-time training (TTT) and augmentation voting for ARC in 2023, wrote that "ARC-AGI-2 appears partially adversarial to the current TTT/AIRV paradigm".

**Search and code generation got too expensive.** NVARC found reasoning models and test-time code generation too slow for 12 hours on four L4s (topic 652927). Barbadillo's replication of a published induction model reached about 22% on ARC-AGI-1 evaluation and 0.8% on ARC-AGI-2.

Off Kaggle, the host called 2025 "the Year of the Refinement Loop": Gemini 3 Pro with Poetiq's refinement reached 54% at about USD 30 per task, Claude Opus 4.5 (Thinking 64k) 37.6% at USD 2.20. NVARC spent roughly USD 0.20 per task, under a hundredth of the Poetiq run, so the reasoning had to be paid for somewhere else.

## 4. The top-20 scorecard

Private percentages from the Kaggle API (2026-10-28). Seven of the 20 published a writeup (rxe's lives in his notebook); ranks 18 and 20 left public notebooks with change logs.

| Rank | Team | Private | Public rank (score) | Key idea | Source |
|---|---|---|---|---|---|
| 1 | NVARC | 24.03 | 1 (27.64) | gpt-oss-120b synthetic puzzles, Qwen3-4B full SFT, ARChitects-style TTT, DFS, rescoring | [writeup](https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc) |
| 2 | the ARChitects | 16.53 | 2 (21.67) | Masked-diffusion LLaDA-8B, 2D RoPE, recursive soft-mask refinement, shape predictor | [writeup](https://www.kaggle.com/competitions/arc-prize-2025/writeups/the-architects-solution-2025) |
| 3 | MindsAI & Tufa Labs | 12.64 | 3 (15.42) | Pruned 660M CodeT5 trained on >100M examples, TTT + AIRV | [writeup](https://www.kaggle.com/competitions/arc-prize-2025/writeups/mindsai-and-tufa-labs-arc-prize-2025-solution) |
| 4 | Lonnie | 6.67 | 343 (4.17) | Fork of a fork of the ARChitects-2024 notebook; changed the seed | [writeup](https://www.kaggle.com/competitions/arc-prize-2025/writeups/arc-prize-2025-competition-writeup-5th-place) |
| 5 | Guillermo Barbadillo @ Veridas | 6.53 | 5 (11.94) | ARChitects-2024 with per-task TTT on 4 GPUs; research on search-and-learn | [writeup](https://www.kaggle.com/competitions/arc-prize-2025/writeups/exploring-the-combination-of-search-and-learn-for) |
| 6 | ippeiogawa | 6.25 | 7 (10.28) | ARChitects-2024, 24 TTT epochs, DFS top_k 4, picked consistent subs | [writeup](https://www.kaggle.com/competitions/arc-prize-2025/writeups/lb10-00-with-small-change-to-2024architects) |
| 7 | Ggmam1 | 6.25 | 55 (7.08) | unknown | no writeup |
| 8 | rxe | 6.11 | 6 (10.42) | ARChitects-2024 as master/worker per-task TTT, tweaked DFS | [notebook](https://www.kaggle.com/code/albartrose/10-4-notebook-arcitects-tweaked) |
| 9 | Abdelmoneim Desouki | 5.83 | 33 (7.50) | unknown | no writeup |
| 10 | Boiled-Sweet-Potato | 5.69 | 48 (7.08) | unknown | no writeup |
| 11 | MenghaoZhao | 5.42 | 18 (7.92) | unknown | no writeup |
| 12 | Ttkuma | 5.42 | 113 (6.25) | unknown | no writeup |
| 13 | yarkhom | 5.28 | 144 (5.83) | unknown | no writeup |
| 14 | Hira Norm | 5.28 | 38 (7.08) | unknown | no writeup |
| 15 | PoerMP | 5.28 | 31 (7.50) | unknown | no writeup |
| 16 | vldmrc | 5.00 | 132 (5.83) | unknown | no writeup |
| 17 | The Unforgiven7 | 5.00 | 100 (6.25) | unknown | no writeup |
| 18 | sanyul | 5.00 | 23 (7.50) | ARChitects-2024 fork with "active layer" freezing | no writeup; [notebook](https://www.kaggle.com/code/sanyul/lb-5-83-baseline-from-1st-place-of-2024) |
| 19 | Hestu | 5.00 | 54 (7.08) | unknown | no writeup |
| 20 | 暗黑AGI (boristown team) | 5.00 | 184 (5.42) | Root 4-GPU public baseline of 2025 | no writeup; [notebook](https://www.kaggle.com/code/boristown/baseline-from-1st-place-of-2024) |

Two source conflicts. MindsAI's writeup calls 15.42% its private score; the API and arcprize.org give 12.64 private, 15.42 public, used here. Lonnie's writeup says "5th place" and 344th public; the API says 4th and 343rd, used here.

![Public versus private for all 20 teams: the top 3 each lost 2.8 to 5.1 points, and ranks 4 to 20 sit inside a two-task band from 5.00 to 6.67](/imgs/blogs/kaggle-arc-prize-2025-solutions-5.webp)

The podium held its order but lost 3.61 (NVARC), 5.14 (the ARChitects) and 2.78 (MindsAI) points. 3rd to 4th is 5.97 points, over three times the whole spread of ranks 4 to 20. Inside that band public order barely predicts private: Lonnie climbed from 343rd while Barbadillo, rxe and ippeiogawa, the public leaders among the forks at 10 to 12, fell back to about 6.

One team is missing entirely. Giotto.ai led the public board "till the last minute" per the NVARC team (topic 614436); a mid-competition arcprize.org snapshot listed it at 27.08% (unverified). It is on neither final board in the API, and nothing we found explains why.

## 5. The top-20 landscape

![Top-20 technique matrix: all nine documented entries trained per task at test time; only ranks 1 to 3 pretrained their own model; eleven columns are unknown](/imgs/blogs/kaggle-arc-prize-2025-solutions-4.webp)

The matrix counts what nine documented scored entries did (ranks 1, 2, 3, 4, 5, 6, 8, 18 and 20). Eleven columns are question marks: ranks 7, 9 to 17 and 19 published nothing we could find, including rank 7's four-person team. Those columns stay unknown; nothing in this post assigns them a method.

**Table stakes.** Per-task test-time training: 9 of 9, each entry fine-tuning on every hidden task's own augmented demonstrations before predicting. Augmentation-based rescoring or voting: 8 of 9, with the ARChitects' candidate selection for their diffusion model not described in enough detail to count. LoRA at test time and depth-first search (DFS) decoding: 7 of 9 each, everyone in the 2024 lineage plus NVARC.

**The single line between the top 3 and ranks 4 to 20** is off-Kaggle pretraining on ARC-specific data: 3 of 9, and those 3 are exactly the only entries above 7%. NVARC used 32 H100s for 27 hours on 3.26 million samples. The ARChitects ran about 175k pretraining steps on eight H100s. MindsAI trained on Google TPUs over many months. Everyone else, as far as the writeups show, reused the 2024 ARChitects checkpoint.

**What nobody did:** program synthesis in a scored submission, 0 of 9. Barbadillo researched it all year; NVARC moved code generation out of the notebook and into its training-data pipeline.

**Contradictions worth knowing.**

| Question | One side | Other side |
|---|---|---|
| Pick the best public sub or the most stable? | ippeiogawa (6th) picked the two most consistent subs and gained a place | the ARChitects (2nd) picked the higher-public sub and left an unselected 19.17 private on the table |
| Is seed variance noise or a hyperparameter? | Lonnie (4th): the seed is a first-class hyperparameter at 120 tasks | sanyul (18th notebook): 4.17 to 5.83 across five identical runs, "reason remains unknown" |
| Reasoning and code at test time? | Barbadillo (5th): search-and-learn adds ~5 points on ARC-AGI-1 | NVARC (1st): too slow under Kaggle limits, belongs in pretraining data |

## 6. The winning solution, explained: NVARC

![NVARC: a gpt-oss-120b funnel narrows 266,593 mixed puzzle summaries to 103,253 verified puzzles; a Qwen3-4B trained on 3.26M samples then does per-puzzle LoRA TTT, batched DFS and 8-augmentation rescoring on Kaggle](/imgs/blogs/kaggle-arc-prize-2025-solutions-6.webp)

NVARC is Ivan Sorokin and Jean-François Puget (CPMP), both at NVIDIA. Their summary: "Kaggle compute limits forces us to focus on efficiency." Reasoning models were too slow, so "the heavy lifting (generative LLM for puzzle understanding + code gen)" moved out of the notebook and into synthetic data built before the competition clock mattered. The pipeline has three stages.

### Stage 1: synthetic puzzles that a program can vouch for

Asking an LLM for "new ARC puzzles" produces noise, because nothing checks that a puzzle has one consistent rule. NVARC built a funnel checked by code at each stage, using gpt-oss-120b through NVIDIA NeMo-Skills on an eight-H100 node at about 15k tokens per second.

1. **Descriptions.** From H-ARC (natural-language solution descriptions by more than 1,700 humans, "surprisingly underestimated") plus 160 BARC descriptions, the LLM described and summarised 716 ARC-AGI-2 training puzzles.
2. **Mixing.** Two summaries are mixed into a new, harder puzzle summary: 266,593 of them.
3. **Inputs.** The LLM writes an input-grid generator in Python plus a unit test. 126,901 survived.
4. **Outputs by agreement.** The LLM writes the transformation program several times independently; a puzzle survives only if they agree. 103,253 did.

Step 4 is the clever part. In the illustrative toy below (not NVARC's code), two independently written correct programs agree on every random input; an off-by-one version disagrees, so the puzzle is dropped rather than taught with a wrong answer.

```python
# Illustrative toy, not NVARC's code: the agreement filter behind synthetic puzzles.
# An LLM writes one input generator and several independent output programs
# for the same puzzle description. Keep the puzzle only if every output program
# returns the same grid on every generated input.
import random

def gen_input(rng, n=6):
    g = [[0]*n for _ in range(n)]
    r, c = rng.randrange(n), rng.randrange(n)
    g[r][c] = rng.randint(1, 9)
    return g

def diag_a(g):                       # correct reading of "extend diagonally down-right"
    n = len(g); out = [row[:] for row in g]
    for r in range(n):
        for c in range(n):
            if g[r][c]:
                k = 1
                while r+k < n and c+k < n:
                    out[r+k][c+k] = g[r][c]; k += 1
    return out

def diag_b(g):                       # a second, independently written correct version
    n = len(g); out = [row[:] for row in g]
    seeds = [(r, c, g[r][c]) for r in range(n) for c in range(n) if g[r][c]]
    for r, c, v in seeds:
        for k in range(n):
            if r+k < n and c+k < n:
                out[r+k][c+k] = v
    return out

def diag_buggy(g):                   # off by one: stops one cell early
    n = len(g); out = [row[:] for row in g]
    for r in range(n):
        for c in range(n):
            if g[r][c]:
                k = 1
                while r+k < n-1 and c+k < n-1:
                    out[r+k][c+k] = g[r][c]; k += 1
    return out

def keep(programs, n_inputs=20, seed=0):
    rng = random.Random(seed)
    for _ in range(n_inputs):
        x = gen_input(rng)
        outs = [p(x) for p in programs]
        if any(o != outs[0] for o in outs):
            return False
    return True

print("two correct programs agree   ->", keep([diag_a, diag_b]))
print("one buggy program disagrees  ->", keep([diag_a, diag_b, diag_buggy]))
```

```console
two correct programs agree   -> True
one buggy program disagrees  -> False
```

Agreement is not proof (two programs can share a misreading), but it cheaply removes the commonest failure, a wrong answer key. The [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation) post covers this verify-before-train pattern in general. In 2024 Barbadillo had found that more BARC puzzles did not help; quality, not volume, made the difference.

### Stage 2: compress the puzzles into a 4B model

The puzzles were expanded into 3,255,481 augmented training samples of up to 7 pairs each. NVARC's own puzzles dominate the mix: 47,337 puzzles at 24 augmentations each (34.8% of samples) and 55,886 at 32 each (54.9%, a set that also used evaluation-puzzle descriptions). The public sets (MINI-ARC, ConceptARC, RE-ARC and the ARC-AGI-2 training puzzles) each contributed 256 augmentations per puzzle.

The format is minimal. Grids are written row by row inside the Qwen3 chat template, input as the user turn and output as the assistant turn, so the model needs only 16 tokens: ten digits, newline, "user", "assistant", two special tokens and padding. Qwen3-4B was trained with full supervised fine-tuning using NVIDIA NeMo RL on the Megatron backend: 4 nodes of 8 H100s for 27 hours. Local validation was loss on the augmented 120 evaluation puzzles, which tracked the public board.

### Stage 3: on Kaggle, adapt and search per puzzle

At test time NVARC ran the ARChitects' 2024 playbook on its own model. Per puzzle: a LoRA adapter of rank 256 (alpha 32, bf16, no 4-bit, no gradient checkpointing, FlashAttention-2 through Unsloth) is fine-tuned on that puzzle's demonstrations. Batched DFS then enumerates candidate grids whose probability stays above a cutoff, which was their main optimisation. Every candidate is rescored under exactly the same 8 augmentations, so scores are comparable across candidates, and the top two become the attempts. Batch-invariant kernels from Thinking Machines fixed run-to-run nondeterminism but cost about 17% speed, so NVARC left them out. (If LoRA at this rank is new to you, the [LoRA and PEFT debugging](/blog/machine-learning/debugging-training/debugging-lora-and-peft) post covers the failure modes.)

### The TRM side bet, and the deadline that went wrong

In the last ten days NVARC also trained a Tiny Recursive Model (TRM, about 7M parameters, the 2025 paper-award winner) inside about two hours of Kaggle time. Tuned, it reached 7.5% on its own, and its 10 attempts were added to the Qwen candidate pool and rescored by Qwen. It lifted a Qwen3-2B run from 21.53 to 22.50 and a 10-hour Qwen3-4B run not at all (27.22 to 27.22). After the deadline, ten more days of TRM training reached 18.05% pass@2 on public evaluation, which CPMP estimated would have meant roughly 15% public and 8% private on Kaggle.

The final submission was not the one the team wanted: their preferred version was stuck 10 hours in the queue, and the selected one ran out of time on the last ~5% of tasks. A post-deadline late submission by Ivan scored 29.72 public. NVARC still won by 7.5 points.

## 7. 2nd and 3rd in depth, then the rest of the top 20

### 2nd: the ARChitects (16.53)

![Autoregressive versus masked diffusion on the diagonal toy: the left-to-right decoder locks an early wrong cell, while the diffusion decoder revises it on a later refinement step](/imgs/blogs/kaggle-arc-prize-2025-solutions-7.webp)

The 2024 winners (Daniel Franzen, Jan Disselhoff, David Hartmann) first upgraded their autoregressive pipeline: per-task TTT, speculative decoding plus prefix caching for faster DFS, a cutoff lowered from 17% to 7%, 32 scoring augmentations instead of 8. It peaked at 16.94% public in August. Then they replaced it.

The final system is a masked-diffusion language model, LLaDA-8B, which starts from a grid of masked positions and fills them over many steps. Two additions make it work for ARC. "Soft masking" adds the mask embedding to every token, so the model treats its own previous guesses as revisable, and recursive latent sampling feeds outputs back in until they stop changing. A 2D positional encoding (based on "Golden Gate RoPE") replaces the 1D one, so the model knows which cells are vertical neighbours. The figure shows why on a diagonal: the autoregressive decoder lives with an early wrong cell, the diffusion decoder fixes it later. The [diffusion language models](/blog/machine-learning/large-language-model/diffusion-language-models-architecture-training-finetuning) post explains the training objective in depth.

Pretraining took about 175k steps at batch 8 on eight H100s over ReARC, ARC-GEN-100K, ARC-AGI-1 and 2, ARC-Heavy and ConceptARC. At test time each task got 128 TTT steps on one L4. Because a diffusion model fills a fixed canvas, a second LLaDA predicted the output shape first, correctly about 85% of the time. With a known shape the model solved about 30.5% of held-out evaluation tasks after 102 inference steps, so the team expected about 26% combined. They got 21.67 public and 16.53 private. An unselected submission scored 19.17 on both boards; they passed on it because its public score was lower. That is 2.64 private points lost to the selection rule alone.

### 3rd: MindsAI & Tufa Labs (12.64)

Jack Cole's team (with Dries Smit, Isaiah Pressman, Mohamed Osman and Michael Hodel, who built RE-ARC) went the other way on model size. They pruned Salesforce's CodeT5-Large encoder-decoder to 660M parameters, keeping all 24 encoder layers and cutting the decoder to 16, and trained it on more than 100M examples (about 70M ARC-style) on TPUs over many months. Training used span corruption, prompt and answer reversals, and BPE dropout.

At test time: about 45k steps of TTT, then AIRV (augment, infer, reverse, vote) with 10k augmented inferences per task, two checkpoints ensembled, about 11 hours on the four L4s.

Their ablations are the most transferable numbers of the competition. TTT and AIRV compose almost perfectly: about 812% gain over zero-shot combined, against roughly 410 to 430% each. At equal compute, ensembling seeds beat doubling the TTT and AIRV samples by 6.2%. New mixup and combine augmentations added 6.3% top-2 on their ARC 1.5 set. Encoder depth mattered far more than decoder depth. A simplified recipe keeps 90 to 95% of the gain on one consumer GPU.

### Ranks 4 to 20, by approach family

![Six documented top-20 entries descend from one 2024 notebook; plotted on private score, the whole family fits inside the run-to-run noise of a single fork](/imgs/blogs/kaggle-arc-prize-2025-solutions-8.webp)

**Family A: the ARChitects-2024 lineage (ranks 4, 5, 6, 8, 18, 20).** The root is the ARChitects' 2024 public notebook (Mistral-NeMo-Minitron-8B, LoRA TTT, DFS, augmented scoring). Every documented team below the podium descends from it.

- **暗黑AGI, the boristown team (20th, 5.00).** Built the root public baseline of 2025: the 2024 notebook on four GPUs instead of two (twice the throughput), global seed 42, deterministic per-GPU seeding.
- **sanyul (18th, 5.00).** Forked boristown's notebook and added `num_active_layers`, training only the first N layers and freezing the rest, with LoRA settings tuned to fit 12 hours. The notebook records five runs on the semi-private set of 4.17, 5, 5, 5.42 and 5.83, and says "the reason for this significant discrepancy remains unknown."
- **Lonnie (4th, 6.67).** Forked sanyul's notebook. The model is the ARChitects' 2024 checkpoint in 4-bit, LoRA rank 32, only the first 32 layers trainable. His only change was the random seed, set to 19920627. He reports the same notebook scores from 3.33% to 6.67% across seeds, four tasks of 120, and argues the seed is a legitimate hyperparameter when 120 tasks decide ranks. He went from 343rd public to 4th private and USD 5k.
- **Guillermo Barbadillo (5th, 6.53).** The 2024 runner-up's scored submission was ARChitects-2024 with per-task TTT, efficient use of four GPUs, and tuning. It was the first entry over 10% (11.94 in April) and lost 5.41 points to the private split.
- **ippeiogawa (6th, 6.25).** Per-task TTT for 24 epochs and DFS top_k of 4 to cut search time. Not every task finished in 12 hours, but fewer epochs "dropped a lot". He chose his two final submissions for consistency across runs rather than best public score, and moved from 7th to 6th.
- **rxe, Richard Emslie (8th, 6.11).** Restructured the notebook as a master handing tasks to per-GPU workers that each run TTT and inference; LoRA rank 32, 10 epochs. A training collator that skipped examples and a DFS tweak that may skip one or two low-probability tokens on long grids moved him from 8 to 10.4 public.

Plotted on private score, the family occupies one band from 5.00 to 6.67. sanyul's five identical runs span 4.17 to 5.83, and Lonnie's seeds span 3.33 to 6.67. The noise inside a single fork is as wide as the differences between all of them.

**Family B: program synthesis as research, not as a score.** Barbadillo's year-long project was "search and learn": sample programs, relabel the failed ones with the outputs they actually produced (hindsight relabelling), fine-tune on those at test time, sample again. At a fixed 512-prediction budget on ARC-AGI-1 evaluation, pure search scored 23.3%, one relabel round 26.0%, and four rounds of 128 scored 28.3%, about +5 points. None of it solved a private ARC-AGI-2 task. His [solution site](https://ironbar.github.io/arc25/05_Solution_Summary/) is the best negative-results document of the competition.

**Family C: no writeup (ranks 7, 9 to 17, 19).** Ggmam1 (four people, 55th to 7th), Abdelmoneim Desouki, Boiled-Sweet-Potato, MenghaoZhao, Ttkuma, yarkhom, Hira Norm, PoerMP, vldmrc, The Unforgiven7 and Hestu published nothing we could find. Their scores sit in the same 5.00 to 6.25 band as the documented forks, so it is plausible they belong to the same lineage, but that is an inference, unverified, and the matrix leaves them blank.

### The best contrasting entry: the paper track

The paper awards went to Alexia Jolicoeur-Martineau's "Less is More: Recursive Reasoning with Tiny Networks" (TRM, about 7M parameters, 45% on ARC-AGI-1 and 8% on ARC-AGI-2 per the host), Pourcel, Colas and Oudeyer's SOAR (self-improving language models for evolutionary program synthesis) and Liao and Gu's "ARC-AGI Without Pretraining".

## 8. What the forum taught everyone

**The public notebook became the floor.** The ARChitects' 2024 notebook was the base of most 2025 entries (ippeiogawa and Lonnie both say so). boristown's four-GPU version scored 5.00 private, and every documented fork landed within two tasks of it.

**Seed variance is the size of the field.** The toy below is illustrative, not anyone's notebook: assume a fork that always solves 4 private tasks and solves each of 8 borderline tasks with probability 0.3.

```python
# Illustrative toy, not any team's notebook: how much can a seed move a score
# when 120 tasks decide the board? Assume a forked notebook that always solves
# 4 private tasks and solves each of 8 "borderline" tasks with probability 0.3.
import random

N_TASKS, SURE, BORDER, P = 120, 4, 8, 0.3

def run(seed):
    rng = random.Random(seed)
    solved = SURE + sum(rng.random() < P for _ in range(BORDER))
    return 100 * solved / N_TASKS

scores = sorted(run(s) for s in range(20))
print("20 seeds:", [round(s, 2) for s in scores])
print(f"min {scores[0]:.2f}  median {scores[10]:.2f}  max {scores[-1]:.2f}")
print(f"expected {100*(SURE + BORDER*P)/N_TASKS:.2f}")
```

```console
20 seeds: [4.17, 4.17, 4.17, 4.17, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 6.67, 7.5, 7.5]
min 4.17  median 5.00  max 7.50
expected 5.33
```

Twenty identical notebooks with expected score 5.33 spread from 4.17 to 7.50, wider than the whole 5.00 to 6.67 band of ranks 4 to 20; fourteen land inside it. The assumptions are invented, but the shape matches what sanyul and Lonnie measured, and the [AIMO 3 post](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-3-solutions) found the same on 50 problems: when the field shares a backbone, the order measures noise.

**Submission choice cost real points, in both directions.** The ARChitects picked the higher public score and left 2.64 private points behind. NVARC's preferred version sat in the queue and the fallback ran out of time on the last ~5% of tasks. ippeiogawa picked for consistency and gained a place. With one submission a day, pick on stability, not on a public score drawn from 120 tasks.

**The test set itself moved.** Greg Kamradt's "Test Set Edits" thread (discussion 573301) records task fixes after community reports; one corrected task is 0.83 points.

## 9. What did not work

| What | Result | Who reported it |
|---|---|---|
| Reasoning models and test-time code generation | Too slow for 12 h on 4x L4 | NVARC (1st) |
| Autoregressive decoding on global-edit tasks | Cannot revise earlier cells; replaced by diffusion | the ARChitects (2nd) |
| Refinement, DPO on beam pairs, targeted ARC-2 data | Not additive | MindsAI (3rd) |
| Own DSL with ~40 primitives and synthetic generators | No real ARC task solved | Barbadillo (5th) |
| BARC induction model on ARC-AGI-2 | 0.8% on evaluation (vs ~22% on ARC-AGI-1) | Barbadillo (5th) |
| Diversity prompting; execution-feedback refinement | Worse than plain prompting; no gain over independent samples | Barbadillo (5th) |
| RL for program synthesis | 22.3% to 27% at 480 predictions, but training repeatedly collapsed | Barbadillo (5th) |
| Lowering TTT epochs to fit the clock | Score "dropped a lot" | ippeiogawa (6th) |
| Full rewrite of the public notebook | 10.4 dropped to 6 to 7 public | rxe (8th) |
| Batch-invariant kernels for determinism | Worked, but ~17% slower | NVARC (1st) |
| Picking the higher-public submission | 2.64 private points left unselected | the ARChitects (2nd) |

Almost every negative result spends test-time compute on reasoning (code generation, refinement, RL-tuned search), and lost because it could not pay for itself inside a 12-hour box.

## 10. Lessons that transfer

1. **When the benchmark hardens, last year's winning recipe becomes the floor, not the ceiling.** The 2024 notebook still scored 5 to 6.67 on ARC-AGI-2, enough for 4th to 20th, and no documented fork got above that. (The 2025 scorecard; Lonnie, ippeiogawa, rxe.)
2. **Move expensive reasoning out of the notebook and into the data.** NVARC used a 120B model to write and cross-check programs offline, then compressed the result into a 4B model the Kaggle box could adapt per puzzle. (NVARC writeup; CPMP, topic 652927.)
3. **Verify synthetic data with independent agreement, not volume.** Keep a generated example only when several independently generated solutions agree. (NVARC's output-program filter; Barbadillo's 2024 finding that more BARC data did not help.)
4. **Match the decoder to the edit pattern of the task.** Tasks with global structure favour a decoder that can revise, and the ARChitects gained by switching to masked diffusion with 2D positions. (ARChitects writeup and report.)
5. **Test-time methods compose; seeds are cheap ensembles.** TTT plus voting was nearly additive, and seed ensembles beat doubling the sample budget. (MindsAI writeup.)
6. **On a small hidden split, treat rank differences under two tasks as noise.** Pick final submissions for stability. (Lonnie; sanyul's notebook; ippeiogawa; the ARChitects' 19.17.)

### If you enter a competition like this

- Compute the value of one item on the hidden split (here ${100/120 = 0.83}$ points) before reading any leaderboard gap.
- Run your best notebook at least three times with different seeds before believing any change smaller than the spread.
- Decide early whether you will pretrain off the platform; a fork will not reach a podium that needs it.
- If you generate training data with an LLM, have it write several independent solutions and keep only the ones that agree.
- Budget wall-clock per task with a margin for the slowest 5%; a timeout loses the tail tasks entirely.
- Submit final candidates days before the deadline; a queue can decide which version counts.

Sibling posts cover the other ARC-shaped competitions: [Google Code Golf 2025](/blog/machine-learning/learn-from-kaggle/kaggle-google-code-golf-2025-solutions) (shortest program per ARC task) and [NeuroGolf 2026](/blog/machine-learning/learn-from-kaggle/kaggle-neurogolf-2026-solutions) (smallest network). ARC Prize 2026 gets its own post once it closes.

## Sources

- Competition page: https://www.kaggle.com/competitions/arc-prize-2025
- Host results analysis: https://arcprize.org/blog/arc-prize-2025-results-analysis ; technical report arXiv 2601.10904
- 1st, NVARC: https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc ; code https://github.com/1ytic/NVARC ; forum topics https://www.kaggle.com/competitions/arc-prize-2025/discussion/614436 , https://www.kaggle.com/competitions/arc-prize-2025/discussion/615018 , https://www.kaggle.com/competitions/arc-prize-2025/discussion/652927
- 2nd, the ARChitects: https://www.kaggle.com/competitions/arc-prize-2025/writeups/the-architects-solution-2025 ; report https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/
- 3rd, MindsAI & Tufa Labs: https://www.kaggle.com/competitions/arc-prize-2025/writeups/mindsai-and-tufa-labs-arc-prize-2025-solution ; PDF https://github.com/jcole75/arc_2025_mindsai/blob/main/MindsAI_Tufa_Labs_2025_Solution.pdf ; dataset https://huggingface.co/datasets/mindware/arc-agi-mega ; arXiv 2506.14276
- 4th, Lonnie: https://www.kaggle.com/competitions/arc-prize-2025/writeups/arc-prize-2025-competition-writeup-5th-place ; notebook https://www.kaggle.com/code/lonnieqin/lb-5-83-baseline-from-1st-place-of-2024
- 5th, Guillermo Barbadillo: https://www.kaggle.com/competitions/arc-prize-2025/writeups/exploring-the-combination-of-search-and-learn-for ; https://ironbar.github.io/arc25/05_Solution_Summary/
- 6th, ippeiogawa: https://www.kaggle.com/competitions/arc-prize-2025/writeups/lb10-00-with-small-change-to-2024architects
- 8th, rxe: https://www.kaggle.com/code/albartrose/10-4-notebook-arcitects-tweaked
- 18th, sanyul (notebook): https://www.kaggle.com/code/sanyul/lb-5-83-baseline-from-1st-place-of-2024
- 20th, boristown team (notebook): https://www.kaggle.com/code/boristown/baseline-from-1st-place-of-2024
- Root public notebook, the ARChitects 2024: https://www.kaggle.com/code/dfranzen/arc-prize-2024-solution-by-the-architects
- Test set edits: https://www.kaggle.com/competitions/arc-prize-2025/discussion/573301 ; changelog https://github.com/arcprize/ARC-AGI-2/blob/main/changelog.md
- Ranks 7, 9 to 17 and 19: no writeup found.

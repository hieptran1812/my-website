---
title: "Kaggle Eedi Mining Misconceptions: how the top 20 teams ranked 2,500+ ways to be wrong"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "A synthesis of the top-20 writeups from Kaggle's 2024 Eedi competition: why every serious team built a Qwen2.5 retrieve-then-rerank stack, how the winner fed a 72B listwise reranker on two T4s, and why synthetic questions for never-seen misconceptions decided the leaderboard."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "retrieval", "reranking", "synthetic-data", "qwen", "lora", "quantization", "education"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 22
---

> **TL;DR.** Kaggle's Eedi competition (September to December 2024, 1,446 teams, a USD 55,000 prize pool) gave you a math multiple-choice question and one wrong answer, and asked you to rank a pool of about 2,587 named misconceptions so that the one which explains the wrong answer lands near the top. The metric was MAP@25. Raja Biswas, solo as team MTH 101, won with 0.63881 private, 0.034 clear of second, using a cascade of Qwen2.5 models (a 14B retriever, then 14B, 32B and 72B rerankers on shrinking lists) trained on 10.6k synthetic questions that Claude 3.5 Sonnet wrote for groups of easily confused misconceptions. Every one of the 15 top-20 teams that published a writeup ran a LoRA-tuned retrieve-then-rerank stack; 13 of them generated synthetic questions. **The lesson:** when most test labels never appear in training, the edge is manufacturing training rows for those labels and checking your validation actually contains them.

![Lever strip with Data lit for Eedi](/imgs/blogs/kaggle-eedi-mining-misconceptions-solutions-1.webp)

The strip above is the series frame: five levers, one lit per competition. Eedi is a data competition wearing a retrieval costume. Almost every team built the standard search funnel; what separated the podium from 15th was how each team created examples of misconceptions nobody had labelled, and how honestly it measured that. The [Science Exam post](/blog/machine-learning/learn-from-kaggle/kaggle-llm-science-exam-solutions) covers the retrieval half; this one is about what changes when you retrieve a label instead of a paragraph.

## The task in 60 seconds

Eedi runs Diagnostic Questions: four-option math questions where each of the three distractors was written to catch one specific misconception, so a wrong answer tells a teacher *why* the student is wrong. Experts tagged misconceptions by hand. The competition, hosted with Vanderbilt University and The Learning Agency Lab, asked for a model that does the tagging.

![Budget card: two 16 GB T4s, nine hours, internet off](/imgs/blogs/kaggle-eedi-mining-misconceptions-solutions-2.webp)

Submissions were offline Kaggle notebooks, nine hours, any public weights allowed. The GPU was Kaggle's pair of 16 GB T4s, the constraint that shaped every top pipeline: a 72B model fits only 4-bit quantised and split across both cards. Training data was about 1.8k questions; the data page puts the hidden test at about 1,000, forum probing at about 685. The public board scored only 28% of it. Leaderboard prizes were USD 12k, 8k, 5k and 5k, plus a CPU-only efficiency track at 12k, 8k and 5k.

![Illustrative question with each wrong answer mapped to a misconception](/imgs/blogs/kaggle-eedi-mining-misconceptions-solutions-3.webp)

The question above is illustrative, in the dataset's style. For ${3 + 2 \times 4}$, a student answering 20 worked left to right, 9 added everything, 24 multiplied everything. Each row is one (question, wrong answer) pair with construct and subject text; the target is one ID from `misconception_mapping.csv`. Text came from images via human-in-the-loop OCR, so LaTeX fragments were part of the input.

The decisive fact surfaced early in the forum: most misconceptions appear only once in train. Third place counted more than 900 of the roughly 2,587 that never appear at all, and the task asked for generalisation to unseen ones. That turned a ranking problem into a zero-shot labelling problem.

## The metric: MAP@25, worked on four rows

Each row has exactly one correct misconception, so mean average precision collapses to something simple. You submit 25 misconception IDs per row in order.

$$
\text{MAP@25} = \frac{1}{U}\sum_{u=1}^{U} \frac{\mathbb{1}[r_u \le 25]}{r_u}
$$

Here $U$ is the number of (question, wrong answer) rows and $r_u$ is the rank at which the correct misconception appears in row $u$'s list. A hit at rank 1 is worth 1, rank 2 is worth one half, rank 5 a fifth, and a miss outside the 25 is worth nothing.

![MAP@25 on four toy rows, mean 0.425](/imgs/blogs/kaggle-eedi-mining-misconceptions-solutions-4.webp)

Four rows with the truth at ranks 1, 2, 5 and absent score $(1 + 0.5 + 0.2 + 0)/4 = 0.425$. The shape of ${1/r}$ means almost all value sits in the top three slots, and slots 9 to 25 together are worth at most ${1/9}$ per row. That is why the winner filled them from a cheap model and spent the expensive one on the top five.

## What made it hard

Three things, stacked.

**Counterfactual reasoning.** The winner put it plainly: LLMs are good at solving math and bad at pretending to be a novice who makes a specific error, or at being a tutor who names that error. Solving ${3 + 2 \times 4}$ is easy; explaining which flawed rule yields 24 is a different skill, and it is the one the metric rewards.

**Near-duplicate labels.** The pool contains near-duplicate misconceptions. A contrastive retriever treats every other label in the batch as a negative, so two phrasings of one idea punish each other; 1st and 14th place both changed batch construction to avoid it.

**Unseen labels and a small public split.** A model trained only on train ranks the labels it has seen above the ones it has not. And any choice tuned on the public board was tuned on 28% of the test.

## The top-20 scorecard

All 20 teams from the private leaderboard. "Public" is the public rank and the move to private (positive means the team rose). Five teams, ranks 16 to 20, published no writeup; their methods are unknown and stay unknown here.

| Rank | Team | Private | Public (move) | Key idea | Source |
|---|---|---|---|---|---|
| 1 | MTH 101 (Raja Biswas) | 0.63881 | 6 (+5) | 14B retriever, 14B then 32B pointwise, 72B listwise top 5; Claude synthetic questions in confusable clusters; distilled CoT | [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution) |
| 2 | CQYR | 0.60480 | 8 (+6) | Linq-Mistral + 2x Qwen2.5-14B retrievers; sliding-window listwise rerank with 14B, then 72B + Llama-3.3-70B | [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/cqyr-2nd-place-solution) |
| 3 | Waseda Pochi | 0.60248 | 1 (-2) | Public 14B embedders, Qwen-32B-AWQ with 6 LoRAs, LB-probed boost for unseen misconceptions | [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/waseda-pochi-3rd-place-solution-with-magic-boost) |
| 4 | takoi & charmq & kami | 0.60044 | 3 (-1) | Unseen-only candidate slots, 100-shot synthetic prompts, three 32B pointwise rerankers, seen labels x0.40 | [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/takoi-charmq-kami-4th-place-solution) |
| 5 | Ebi & ktr | 0.59765 | 4 (-1) | Weak 1.5B retriever, Qwen2.5-32B listwise over 52 single-letter options x 2 windows; SubjectId folds | [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/ebi-ktr-5th-place-solution) |
| 6 | Nikita Babich | 0.58998 | 5 (-1) | Four retrievers, 32B 50-option reranker with self-distillation and rephrased-label TTA | [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/nikita-babich-6th-place-solution) |
| 7 | terekaerumasahmet | 0.58693 | 2 (-5) | Rank-vote of three retriever + 32B reranker pipelines; iterative unseen boost | [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/terekaerumasahmet-private-7th-public-2nd-place-sol) |
| 8 | Yannan Chen | 0.58160 | 13 (+5) | Four 14B fold retrievers, one 32B yes-logit reranker, wise-FT merge at alpha 0.8 | [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/yannan-chen-8th-place-solution) |
| 9 | Study Hard | 0.57114 | 7 (-2) | GPT-4o synthetic filtered by the retriever; 50-token rationales; 32B causal-LM reranker | [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/study-hard-private-9th-public-7th-place-solution) |
| 10 | sayoulala | 0.56616 | 19 (+9) | Author of the recipe thread; 14B + Coder-14B retrievers, 32B 25-option reranker, fold-LoRA fusion | [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/sayoulala-10st-place-solution-summary) |
| 11 | H&Y&L | 0.56496 | 9 (-2) | Three-stage curriculum on real external data instead of synthetic; 32B retriever and reranker | [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/h-y-l-private-11th-public-9th-place-solution-summa) |
| 12 | chizhu | 0.56429 | 18 (+6) | Concatenated embeddings of three 14B retrievers; rerankers on random 25-of-50 subsets | [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/chizhu-12th-place-solution) |
| 13 | qwenqwenqwen... | 0.56133 | 26 (+13) | Math-7B solution as retriever input; 32B listwise over 100, then 72B listwise over 25 in fp16 layer by layer | [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/qwenqwenqwenqwenqwenqwenqwenqw-13th-place-solution) |
| 14 | eeee (rsi) | 0.55874 | 16 (+2) | False-negative masking; 32B pointwise reranker that sees the whole list and reuses the KV cache | [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/eeee-rsi-14th-place-solution) |
| 15 | iws | 0.55392 | 10 (-5) | Progressive mining SFR to 14B to 32B; random rather than hardest negatives | [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/iws-15th-place-solution) |
| 16 | mathminds | 0.55306 | 17 (+1) | unknown | no writeup |
| 17 | B.Z.Z.Z.L. | 0.55085 | 11 (-6) | unknown | no writeup |
| 18 | Ilya Novoselskiy | 0.54993 | 23 (+5) | unknown | no writeup |
| 19 | LRKBY | 0.54805 | 12 (-7) | unknown | no writeup |
| 20 | JIN | 0.54603 | 29 (+9) | unknown | no writeup |

Read the private column as a cliff then a slope. MTH 101 is 0.034 ahead of 2nd; ranks 2 to 5 sit inside 0.008 of each other; the remaining fifteen descend gently to 0.546.

## The top-20 landscape

![Top-20 technique matrix](/imgs/blogs/kaggle-eedi-mining-misconceptions-solutions-5.webp)

The matrix counts only what the 15 writeups report; an empty cell means "not reported". I recomputed the counts from the cells.

**Table stakes.** All 15 used LoRA. 14 of 15 used a decoder LLM, nearly always Qwen2.5-14B, as the retriever (5th used a 1.5B Stella model). 13 of 15 had a Qwen2.5-32B in the final rerank, 13 of 15 ran 4-bit inference, and 13 of 15 generated synthetic questions, 10 with a frontier API model (Claude 3.5 Sonnet, GPT-4o or Gemini 1.5 Pro). Most started from sayoulala's public recipe for an LLM recall model (CV 0.490, LB 0.352). The [contrastive losses for retrieval post](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval) covers the loss.

**What separated 1 to 5 from 6 to 20.** Two things, both visible in the matrix.

- **Bigger final rerankers.** Only 1st, 2nd and 13th ran a 70B-class model; 4th ran three 32B rerankers and 5th reranked a deep top 104. Below 5th, one 32B was the norm.
- **Explicit work on unseen labels.** ConstructId folds (1st), SubjectId folds (5th), offline misconception explanations (2nd), an LB-probed boost (3rd), unseen-only slots and seen-label down-weighting (4th). Below 5th, most teams used QuestionId folds and no unseen handling (exceptions: 6th, 7th, 14th).

**Contradictory findings.** The writeups disagree in useful ways, and I would not pick a side without testing on your own data.

| Question | Said yes | Said no |
|---|---|---|
| Hard-negative mining helps the retriever? | 8th ("very helpful"), 14th (per-epoch refresh) | 1st (iterative mining did not work), 5th (relaxed negatives), 15th (random from top 200 on purpose) |
| Synthetic rows help the reranker? | 1st, 4th (+~0.01 private) | 6th ("did not work"), 3rd ("added little") |
| 72B reranker beats 32B? | 1st, 2nd | 6th ("the same results as for 32B") |
| QwQ-32B-Preview useful? | 2nd (as a retriever, 0.500 private) | 1st, 5th, 7th (as a reranker) |
| Reasoning text in the reranker? | 1st (+0.019) | 13th (step-by-step solutions hurt), 12th (its weakest member) |

My reading of the pattern: what helped depended on what the team's retriever already did well.

## The winning solution: MTH 101

Raja Biswas made two decisions most teams did not: spend the large-model budget on the top five candidates only, and generate synthetic data for *groups* of confusable labels. His code, notebook and synthetic dataset are public (Sources). For the general funnel shape, see [retrieval, ranking and reranking](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking).

![The 1st place funnel from 2,587 labels to a 72B over five](/imgs/blogs/kaggle-eedi-mining-misconceptions-solutions-6.webp)

### Stage 1: a retriever tuned for recall, not MAP

A Qwen2.5-14B encoder with multiple-negatives ranking loss, LoRA rank 64, alpha 128, batch 128, 12 epochs, temperature 0.01. He optimised **recall@32**, not MAP: the retriever only has to get the truth into the list. He kept the top 32 plus up to 32 more within a constant of the top score, so ambiguous rows get longer lists. Batches held **one example per misconception**, so no in-batch negative is secretly a positive. The 14B was first pretrained on all uncurated synthetic data. Alone it scored 0.524 public, 0.475 private. Iterative hard-negative mining, cross-device negatives, subject-grouped batches and NV-Embed-v2 style bidirectional attention did not help.

### Stage 2: synthetic questions for clusters of confusable misconceptions

This is the lit lever. Generating one question per misconception teaches each label in isolation; the ranker needs the *boundary* between labels that look alike. So:

1. Cluster misconceptions that co-occur in the retriever's and rankers' validation predictions. Labels that the model confuses end up together.
2. For each cluster, Claude 3.5 Sonnet writes 5 new questions, shown 4 to 5 reference questions from the same cluster. The prompt, built with Anthropic's metaprompt, asks for analysis and planning first, the exact misconception wording, and questions harder than the references.
3. GPT-4o, as judge, scores 0 to 10 how logically the misconception leads to the wrong answer; low scores are dropped.
4. Claude also invented new misconceptions while writing. Over 4,000 survived deduplication: string normalisation, then gte-base similarity of at least 0.995 counts as the same label, 0.95 to 0.995 is dropped as ambiguous, and anything below is kept as new.

The result: 1.8k competition plus 10.6k synthetic questions over 4,791 misconceptions. The [synthetic data generation post](/blog/machine-learning/training-data/synthetic-data-generation) covers the generic generate, judge, dedup loop; the Eedi move is choosing which labels to generate for together.

### Stage 3: distilled "reasoners"

Claude 3.5 Sonnet, shown the question, both answers, the true misconception and related ones, wrote the student's likely faulty reasoning in 5 to 6 sentences. Qwen2.5-Math-7B, Qwen2.5-14B and Qwen2.5-32B were fine-tuned to produce it **without** seeing the misconception, and run in vLLM at temperature 0.7, top-p 0.8, 256 tokens. A small model taught to narrate a novice's mistake hands the ranker evidence the raw question lacks.

### Stage 4: three rerankers on shrinking lists

Pointwise rankers score one (row, misconception) pair as logit("Yes") minus logit("No"), cross-entropy over 1 positive and the rest negatives, LoRA rank 64, lm_head frozen. The 14B keeps 8, the 32B keeps 5. The 72B is listwise: it sees all 5 candidates, up to 5 reference examples and three reasoning texts (7B, 14B, 32B), and orders them. All were AutoAWQ-quantised with task-specific calibration sets.

![The 1st place ablation ladder on private MAP@25](/imgs/blogs/kaggle-eedi-mining-misconceptions-solutions-7.webp)

The ablation puts a number on every idea. The largest step, +0.044, is pseudo-labelling the synthetic data with Qwen2.5-Math-72B and Qwen2.5-72B pointwise teachers (my reading: soft targets that say which wrong labels are *almost* right). Few-shot references add 0.036, reasoning text 0.019, the 32B 0.010 and the 72B 0.013. The big models are the last steps, not the first.

### A toy version of the cascade

An illustrative toy, not the team's code: MAP@25 as in the figure, and the winner's stitching of three rerankers into one list.

```python
# Illustrative toy, not any team's code: MAP@25 with one correct label,
# and the 1st-place way of stitching three rerankers into one 25-item list.

def map_at_k(ranked_lists, truths, k=25):
    total = 0.0
    for ranked, truth in zip(ranked_lists, truths):
        top = list(ranked[:k])
        total += 1.0 / (top.index(truth) + 1) if truth in top else 0.0
    return total / len(truths)

def cascade(candidates, s14, s32, s72, keep14=8, keep32=5):
    """candidates: retriever output. s14/s32/s72: id -> score per reranker."""
    by14 = sorted(candidates, key=lambda m: -s14[m])   # 14B scores everything
    by32 = sorted(by14[:keep14], key=lambda m: -s32[m])  # 32B sees the top 8
    by72 = sorted(by32[:keep32], key=lambda m: -s72[m])  # 72B orders the top 5
    return by72 + by32[keep32:] + by14[keep14:]          # slots 1-5, 6-8, 9+

# The four rows of the MAP@25 figure: truth at rank 1, 2, 5, and missing.
rows = [list(range(100, 125))] * 4
truths = [100, 101, 104, 999]
print("toy MAP@25:", round(map_at_k(rows, truths), 3))

# One row: the true misconception (id 7) sits 7th in retriever order.
cands = list(range(1, 33))
s14 = {m: 1.0 / m for m in cands}; s14[7] = 0.30   # 14B lifts it to 4th
s32 = {m: 0.0 for m in cands}; s32[7] = 1.0        # 32B puts it 1st
s72 = {m: 0.0 for m in cands}; s72[7] = 1.0        # 72B keeps it 1st
final = cascade(cands, s14, s32, s72)
print("retriever rank:", cands.index(7) + 1,
      "| after 14B:", sorted(cands, key=lambda m: -s14[m]).index(7) + 1,
      "| final:", final.index(7) + 1, "| list length:", len(final[:25]))
```

```console
toy MAP@25: 0.425
retriever rank: 7 | after 14B: 4 | final: 1 | list length: 25
```

Moving one truth from rank 7 to rank 1 lifts that row from about 0.14 to 1.0, and the expensive model looked at only five candidates.

### Validation: the split that matched the test

He grouped folds by **ConstructId**: QuestionId was too optimistic, SubjectId too pessimistic. Synthetic data sat in its own fold 99; final models refit on everything. Sixth public, 1st private is what a well-chosen split looks like from outside.

## Second and third place

### 2nd: CQYR, the listwise sliding window

Four members (copasta, wangqihanginthesky, yokuyama, honglihang), training code and inference notebook public. Three ideas stand out.

**Enrich the labels.** Llama-3.1-70B-Instruct and Qwen2.5-72B-Instruct wrote a detailed explanation of each misconception, lifting the retriever 2 to 4% at zero submission cost, since label embeddings are precomputed. They also pulled parent subjects from the NeurIPS 2022 Eedi challenge metadata, which shares SubjectIds.

**Three generations of synthetic questions.** An LLM writes a question for a misconception with few-shot examples; Qwen-Math solves it normally for the correct answer and then "under the misconception" for the wrong one; gpt-4o-mini rates quality and rows scoring 2 or less out of 5 are dropped.

**Sliding-window listwise reranking.** Qwen2.5-14B-Instruct reorders ranks 8 to 17, then Qwen2.5-72B-Instruct and Llama-3.3-70B-Instruct finalise ranks 1 to 10, with reversed-order TTA. QLoRA on random lists of 3 to 25 items; the 72B trained about 8 hours on an H100. They quantised with intel/auto-round to 4 bits and padded the 72B intermediate size from 29568 to 29696 so it splits across GPUs. vLLM [prefix caching](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) saved about 10%, and a logits processor (+100 on candidate tokens) replaced a classification head.

Their ablation (private / public): retrievers 0.513 / 0.530; plus Qwen 14B 0.568 / 0.583; plus Qwen 72B alone 0.593 / 0.582; all three rerankers 0.596 / 0.609; plus the 8 to 17 window 0.604 / 0.622. Compare the 72B alone with the 14B alone: public went *down* (0.583 to 0.582) while private went up 0.025. Their validation said keep it, and they did.

### 3rd: Waseda Pochi, the magic boost

Yi-Chia Chen, solo, led the public board at 0.67081. The base was modest: public Qwen-14B embedders, top 35, a Qwen-32B-Instruct-AWQ reranker with 6 LoRAs, scoring 0.590 public and 0.564 private.

The "magic" was label-prior estimation by leaderboard probing. Submitting one prediction per row restricted to seen misconceptions scored 0.154; restricted to unseen ones, 0.444. He read that as about one seen row per three unseen, and multiplied unseen probabilities by a constant until 75% of top-1 predictions were unseen. Public went from 0.590 to 0.658 and private from 0.564 to 0.602: 0.038 private from post-processing alone. A toy version shows both the gain and the risk:

```python
# Illustrative toy, not any team's code: a label-prior shift and the
# "boost unseen labels until 75% of top-1 picks are unseen" correction.
import numpy as np

rng = np.random.default_rng(0)
n_rows, n_labels = 2000, 300
unseen = np.zeros(n_labels, bool); unseen[100:] = True   # 200 of 300 unseen

# Test truth: 75% of rows have an unseen misconception (the 3rd-place guess).
truth = np.where(rng.random(n_rows) < 0.75,
                 rng.integers(100, 300, n_rows), rng.integers(0, 100, n_rows))

# A reranker trained on train data: signal on the truth, plus a bias toward
# labels it has seen (those are the only positives it ever trained on).
logits = rng.normal(0, 1, (n_rows, n_labels)) + np.where(unseen, 0.0, 1.0)
logits[np.arange(n_rows), truth] += 2.5
probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)

def map25(p):
    order = np.argsort(-p, 1)[:, :25]
    hit = order == truth[:, None]
    return (hit / np.arange(1, 26)).sum(1).mean()

def boost(p, target=0.75):
    c = 1.0
    while (unseen[np.argmax(p * np.where(unseen, c, 1.0), 1)]).mean() < target:
        c *= 1.05
    return p * np.where(unseen, c, 1.0), c

print(f"top-1 unseen share before: {unseen[probs.argmax(1)].mean():.2f}")
boosted, c = boost(probs)
print(f"boost factor c = {c:.2f}")
print(f"MAP@25 before {map25(probs):.3f}  after {map25(boosted):.3f}")

# Same boost, but the guess was wrong: only 25% of test rows are unseen.
truth = np.where(rng.random(n_rows) < 0.25,
                 rng.integers(100, 300, n_rows), rng.integers(0, 100, n_rows))
logits = rng.normal(0, 1, (n_rows, n_labels)) + np.where(unseen, 0.0, 1.0)
logits[np.arange(n_rows), truth] += 2.5
probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
print(f"wrong prior: MAP@25 before {map25(probs):.3f}  after {map25(boost(probs)[0]):.3f}")
```

```console
top-1 unseen share before: 0.20
boost factor c = 3.23
MAP@25 before 0.373  after 0.493
wrong prior: MAP@25 before 0.533  after 0.388
```

When the prior is right, a model biased toward seen labels picks an unseen top-1 only 20% of the time, and the boost lifts the toy's MAP@25 from 0.373 to 0.493. When the guessed prior is wrong, the same boost drops it from 0.533 to 0.388. The correction is only as good as the probe, and the probe was run on 28% of the test.

## Ranks 4 to 20, by approach family

### Unseen labels as a first-class design goal (4th, 6th, 7th)

**4th, takoi & charmq & kami** tuned every stage for unseen labels. Qwen2.5-72B-Instruct-AWQ wrote about 8,000 questions for misconceptions absent from train, with **100 train questions as few-shot examples** in a roughly 20k-token prompt ("it was crucial to include a large number of examples"). An untuned Qwen2.5-32B guessed a misconception per row for the retriever input. Candidates: top 25 overall plus top 15 unseen. Three Qwen2.5-32B-GPTQ pointwise rerankers, then seen misconceptions multiplied by 0.40.

**6th, Nikita Babich** (solo) weighted seen and unseen misconceptions equally in validation, arguing a QuestionId split (about 20% unseen) rewards overfitting to seen labels. His largest gain came from **excluding all test-only misconceptions from reranker training**. Self-distillation (soft logits from a second seed, KL plus cross-entropy) and rephrasing each misconception as "missing knowledge plus student belief" for TTA added about 0.01 each. A final stage rescored the top 2 in both orders, since an unsure reranker keeps the retriever's order.

**7th, terekaerumasahmet** (tereka, aerdem4, kaerunantoka, irrohas) was public 2nd. Three retriever plus Qwen2.5-32B-AWQ reranker pipelines, each a different format (40-option multiple choice run on two overlapping windows, yes/no pointwise, and a 9-option multiple choice trained with shuffled order and run with order TTA), merged by summing ${1/\text{rank}}$. Unseen misconceptions were boosted iteratively, unless a boost would make one rank 1 for some pair. This team is the one linked in the forum's "private dataset accidentally shared" thread (below); its writeup does not mention using that data.

### The listwise single-token reranker (5th, 13th)

**5th, Ebi & ktr** ran the most surprising top-five stack: a weak retriever (stella_en_1.5B_v5, CV 0.41) rescued by a deep listwise rerank. Qwen2.5-32B-Instruct sees 52 candidates labelled A to Z and a to z, emits one token, and the letter probabilities sort the list; two windows cover the top 104. A Qwen2.5-32B explanation of each mistake fed every model (+0.04 retriever CV), and **SubjectId** folds brought CV close to LB (0.626 vs 0.633).

**13th, qwenqwenqwen...** rose 13 places, the biggest jump in the top 20. Qwen2.5-Math-7B solved each question to feed a Qwen2.5-14B retriever (top 100); Qwen2.5-32B ranked all 100 listwise, then Qwen2.5-72B the top 25, both fp16 loaded **one layer at a time**: about 500 of the 540 minutes, no quantisation loss.

### Pointwise rerankers with training tricks (8th, 9th, 14th)

**8th, Yannan Chen** (solo) used four Qwen2.5-14B fold retrievers and one Qwen2.5-32B "Yes"-logit reranker over the top 40, merged with **wise-FT at alpha 0.8** before AWQ, in about 6 hours. Public ladder: synthetic data 0.3x to 0.49, ensembling 0.51, reranker 0.57, wise-FT 0.59. Gotcha: AutoAWQ dropped a small Qwen's lm_head and embedding weights (random letters out), fixed by copying the keys back.

**9th, Study Hard** generated GPT-4o questions and, for misconceptions already in train, kept only rows where the best retriever put the target in its top 5 to 10: the retriever acts as a quality filter. A Qwen2.5-32B causal-LM reranker reading the "yes" logit beat a binary classification head by 0.035 public (0.641 vs 0.606), and reranker training began only a week before the deadline.

**14th, eeee (rsi)** generated 13,921 GPT-4o rows (at least 7 per misconception), masked same-misconception in-batch negatives, and built a pointwise 32B that sees a shuffled list of about 30 candidates, reusing `past_key_values` for batches of 5. Post-mortem: one misconception per call cost far more time than listwise. Their out-of-distribution split tracked the leaderboard.

### Ensembles and merging (10th, 12th, 15th)

**10th, sayoulala**, whose recipe thread most of the field started from, pretrained Qwen2.5-14B and Qwen2.5-Coder-14B retrievers on 7k GPT-4o rows, then fine-tuned and fused five fold LoRAs; the reranker was a Qwen2.5-32B-AWQ choosing among the top 25 as a multiple-choice question. "LoRA fusion remains effective in any scenario," he wrote. **12th, chizhu** (solo) concatenated three 14B retrievers' embeddings and trained rerankers on 25 random candidates drawn from the top 50, so the reranker never learns the retriever's order. **15th, iws** (solo) mined progressively (SFR, then a 14B, then a 32B, each mining the next one's top 200) and sampled *random* negatives from that pool, reasoning that the target is recall@25, not top-1; averaging two pipelines' overlapping IDs added 0.02.

### Real data instead of synthetic (11th)

**11th, H&Y&L** is the contrast case: no generated questions. They ran a three-stage curriculum, an external set first, then MalAlgoQA (wrong-answer rationales treated as misconceptions), then train, for both a Qwen2.5-32B retriever and reranker. Staging beat mixing; external data was worth 0.02 to 0.03 at retrieval. Cross-device negatives took public from 0.567 to 0.574.

### No writeup (16th to 20th)

mathminds (16th), B.Z.Z.Z.L. (17th), Ilya Novoselskiy (18th), LRKBY (19th) and JIN (20th) published no writeup that a search of the leaderboard and 198 forum topics could find. Their methods are unknown.

### The contrasting entry

The cleanest contrast is between 3rd and 6th. Both believed the test was dominated by unseen labels. Waseda Pochi measured the prior on the public board and corrected the output, gaining 0.038 private. Nikita Babich changed the training set and the validation weighting so the model generalised without a correction, and finished 5th public and 6th private with almost no movement. One approach is a measurement, the other is a design; the measurement won on points here, and the design is the one you can ship.

## What the forum taught everyone

![Public to private rank slope chart for the top 20](/imgs/blogs/kaggle-eedi-mining-misconceptions-solutions-8.webp)

- **The recipe thread.** sayoulala's "Some tricks for training LLM Recall Model" (171 votes) set the field's default retriever; earlier baselines scored LB 0.189 to 0.246.
- **The long tail.** moth's "Most misconceptions appear only once in train" made synthetic data and unseen handling the centre of the competition.
- **Fitting 70B on 2x T4.** Jagat Kiran's thread on running 70B-class models on two T4s and Ahmet Erdem's logits-processor thread were plumbing 2nd place used directly. The [quantization for LLM serving post](/blog/machine-learning/model-serving/quantization-for-llm-serving) explains the 4-bit trade-off.
- **Test-size probing.** Forum members estimated about 685 test questions with many unseen subjects; 3rd place cited this.
- **The dataset thread.** "Private dataset accidentally shared by Eedi?" (63 votes) reported that a member of the eventual 7th-place team had requested a non-public Eedi misconception-labelled dataset used in a UMass NAACL paper, and asked whether it overlapped the test. No host ruling appears in the thread, and it is not known whether any team used that data.

**The shake-up was moderate.** The public top two finished 3rd and 7th; MTH 101 climbed from 6th to 1st with a 0.034 margin. Within the top 20, 13th rose 13 places and 10th and 20th rose 9; 19th fell 7 and 17th fell 6. Sixth place credited the host's balanced split of seen and unseen labels across public and private for keeping the shuffle mild. The lesson the winners shared is in CQYR's 72B row: a choice that looks flat on 28% of the test can be worth 0.025 on the other 72%.

## What did not work

Aggregated across the 15 writeups, with who reported it.

| Idea | Reported by | Note |
|---|---|---|
| Iterative hard-negative mining | 1st; 5th (hard negatives in general) | 8th and 14th found the opposite |
| QwQ-32B-Preview as reranker | 1st, 5th, 7th | 2nd used QwQ well as a retriever |
| 72B reranker over 32B | 6th | 1st and 2nd found it decisive |
| Ensembling several 72B models | 13th | |
| Synthetic data in reranker training | 6th; 3rd ("added little"); 13th (several variants) | 1st and 4th gained from it |
| LLM rewrites of positives for oversampling | 14th | |
| Rephrasing misconceptions for recall | 7th (recall "almost same") | 6th found it useful as TTA |
| Other answer options, few-shot examples or step-by-step solutions in the reranker prompt | 14th; 5th; 13th | |
| Two-step 7B then 32B listwise | 5th | |
| Other single-token label sets (letter pairs, digits, hiragana) | 5th | |
| Concatenating or averaging many retriever vectors | 7th | 4th and 12th concatenated successfully |
| Full-data reranker | 7th | |
| fp16 rerankers with all LoRAs, quantised afterwards | 6th | |
| Qwen2.5-Coder | 12th | 10th used a Coder-14B retriever |
| LightGBM or CatBoost on top of reranker outputs | 12th | |
| 32B retriever over 14B | 13th (kept, but not best) | |
| Distilling 72B outputs | 10th (offline, not submitted) | 1st's teacher pseudo-labels worked |
| LoRA-merge "continuous fine-tuning" | 1st | |
| LoRA pretraining the question generator | 6th (hurt reasoning) | |

Most rows have a team on each side: the shared stack was strong, and the remaining gains were small and data-dependent.

## Lessons that transfer

1. **Spend the big model on the top of the list.** Under ${1/r}$, slot 1 is worth nine times slot 9. The winner ran a 72B on five candidates and a 14B on the rest, and the ablation shows the 72B adding 0.013 on top of everything else (1st place writeup).
2. **Generate synthetic data for the boundaries, not the labels.** Clustering misconceptions the model already confuses, then writing questions for the cluster, beat one-label-at-a-time generation, and the biggest single step in the ablation was teacher pseudo-labels on that data (1st place writeup).
3. **Label-side compute is free.** Anything you do to the label texts, explanations, rephrasings, embeddings, happens offline and costs nothing in the nine hours (2nd place writeup, 6th place writeup).
4. **Choose the validation split by the axis the test will stress.** ConstructId (1st), SubjectId (5th), equal-weight seen/unseen (6th) and an out-of-distribution holdout (14th) all moved CV toward LB; QuestionId folds overstated performance on seen labels (1st, 5th and 6th place writeups).
5. **Prior correction is a measurement, so treat it like one.** The unseen boost was worth 0.038 private for 3rd place, and the toy shows the same boost costing more than it gains when the prior is wrong (3rd place writeup).
6. **A weak retriever can be rescued by a deep reranker.** 5th place reranked 104 candidates with a 1.5B retriever behind it and finished above teams with far stronger retrievers (5th place writeup).

For a deeper look at the reranker side, see [reranker models: training and fine-tuning](/blog/machine-learning/large-language-model/reranker-models-training-finetuning-case-studies); for the LoRA failure modes several teams hit, [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft).

### If you enter a competition like this

- Count labels with zero or one training row first. If many, the competition is about synthetic data.
- Validate on labels absent from training; report seen and unseen scores separately.
- Start from the public recipe and spend your own time on data and the final rerank.
- One example per label per batch, or mask same-label negatives.
- Precompute everything on the label side.
- Time 4-bit inference of your largest reranker in the real limit in week one.
- If you probe the leaderboard for a prior, decide in advance how far to trust a 28% sample.

The sibling post on [MAP: Charting Student Math Misunderstandings](/blog/machine-learning/learn-from-kaggle/kaggle-map-student-math-misunderstandings-solutions) covers the 2025 follow-up competition on student math misunderstandings, where the input is a student's written explanation rather than a chosen option. The [PII Data Detection post](/blog/machine-learning/learn-from-kaggle/kaggle-pii-data-detection-solutions) is the other Wave 2 competition decided by LLM-written training data, and the [LMSYS post](/blog/machine-learning/learn-from-kaggle/kaggle-lmsys-chatbot-arena-solutions) shows the same "fit a bigger model into the box" moves on a preference task.

## Sources

- Competition: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics
- 1st, MTH 101: [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution), [summary](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551402), [code](https://github.com/rbiswasfc/eedi-mining-misconceptions), [inference notebook](https://www.kaggle.com/code/conjuring92/eedi-a2-pipeline), [synthetic dataset](https://www.kaggle.com/datasets/conjuring92/eedi-mcq-dataset)
- 2nd, CQYR: [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/cqyr-2nd-place-solution), [training code](https://github.com/wangqihanginthesky/Eedi_kaggle), [inference](https://www.kaggle.com/code/honglihang/2nd-place-inference-code)
- 3rd, Waseda Pochi: [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/waseda-pochi-3rd-place-solution-with-magic-boost), [inference](https://www.kaggle.com/code/threerabbits/eedi-11-21-myq14b-q32b-rerank-mod-novel-local-suf)
- 4th, takoi & charmq & kami: [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/takoi-charmq-kami-4th-place-solution), [code (takoi)](https://github.com/TakoiHirokazu/kaggle-Eedi-4th-solution), [code (charmq)](https://github.com/charmq00/kaggle_eedi_public), [slides](https://speakerdeck.com/rist/4th-place-solution-eedi-mining-misconceptions-in-mathematics)
- 5th, Ebi & ktr: [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/ebi-ktr-5th-place-solution), [code](https://github.com/ebinan92/Eedi-5th-solution)
- 6th, Nikita Babich: [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/nikita-babich-6th-place-solution)
- 7th, terekaerumasahmet: [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/terekaerumasahmet-private-7th-public-2nd-place-sol), [Masaya's part](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551430)
- 8th, Yannan Chen: [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/yannan-chen-8th-place-solution), [training code](https://github.com/yannan90/kaggle-2024-Eedi-8th-place-training-code)
- 9th, Study Hard: [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/study-hard-private-9th-public-7th-place-solution)
- 10th, sayoulala: [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/sayoulala-10st-place-solution-summary), [recipe thread](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/543519)
- 11th, H&Y&L: [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/h-y-l-private-11th-public-9th-place-solution-summa), [code](https://github.com/MilchstraB/Eedi)
- 12th, chizhu: [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/chizhu-12th-place-solution)
- 13th, qwenqwenqwen...: [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/qwenqwenqwenqwenqwenqwenqwenqw-13th-place-solution)
- 14th, eeee (rsi): [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/eeee-rsi-14th-place-solution)
- 15th, iws: [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/iws-15th-place-solution)
- Efficiency track 1st, Dipam Chakraborty: [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/dipam-chakraborty-efficiency-track)
- Forum threads: [long tail](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/533920), [baselines](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/534317), [logits processors](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/546978), [70B on 2x T4](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/550223), [dataset question](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/550619), [test-size probing](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/550875)

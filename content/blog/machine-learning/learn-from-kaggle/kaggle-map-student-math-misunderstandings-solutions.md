---
title: "MAP Charting Student Math Misunderstandings on Kaggle: how the top 20 classified misconceptions when label noise was the ceiling"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "A synthesis of the top-20 MAP solutions: why the winner made one submission all competition, how a shared-prefix attention mask and INT8 kernels put two 32B models on a pair of T4s, what synthetic explanations and soft labels bought, and why 0.00171 separated 1st from 20th."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "text-classification", "label-noise", "map-at-k", "quantization", "lora", "knowledge-distillation", "ensembling", "education"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 22
---

> **TL;DR.** Winner: **tascj, solo, private MAP@3 0.94919**, 14th on the public leaderboard, with exactly **one submission** in three months. Decisive lever: **model fit.** He fully fine-tuned Qwen3-32B and GLM-Z1-32B as "suffix classifiers" that score only the 8 to 12 labels a question can take, then ran 32B models on two T4s with INT8 kernels and layer-by-layer weight streaming. The lesson: when labels are noisy and the top 20 sit inside 0.00171 of each other, the team with a validation signal it can trust beats the teams chasing the public leaderboard. Every top-20 writeup that named a backbone used a Qwen-family model; what separated the very top was scoring only the question's candidate labels and spending systems work to afford 32B.

![The five-lever strip with Model fit lit as decisive and Metric insight half-lit](/imgs/blogs/kaggle-map-student-math-misunderstandings-solutions-1.webp)

The strip above is the series' mental model: five places a Kaggle LLM competition can be won. MAP was won on **model fit**, with **metric insight** half-lit: the test set uses the same 15 questions as train, so half of each label is free and the rest is a short per-question list. Data, the lever that decided [DAIGT](/blog/machine-learning/learn-from-kaggle/kaggle-detect-ai-generated-text-solutions) and the sibling [Eedi misconception competition](/blog/machine-learning/learn-from-kaggle/kaggle-eedi-mining-misconceptions-solutions), helped the 2nd and 7th place teams but not the winner.

MAP ran from 2025-07-10 to 2025-10-15, hosted by Vanderbilt University and The Learning Agency on Eedi diagnostic questions: 1,857 teams, 2,175 competitors, 39,742 submissions, USD 55,000 across six places. The notebook format is explained in the series opener, [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work).

## 1. The task in 60 seconds

A student answers a multiple-choice maths question and writes a sentence or two explaining why. Annotators tag the explanation. Your model reproduces the tag. It sounds like a weekend text-classification project, and it turned into a three-month fight over the fourth decimal place.

![The notebook budget: two T4 GPUs, nine hours, internet off, about 16,000 hidden rows split half public and half private, 36,696 training rows over 15 questions](/imgs/blogs/kaggle-map-student-math-misunderstandings-solutions-2.webp)

The budget card sets the physics: **two NVIDIA T4s**, at most **9 hours**, **internet off**, about **16,000 hidden rows** scored **50% public, 50% private**. The T4 is the binding constraint. It has no bfloat16 support, and the 1st-place writeup measured an unstable **~20 TFLOPS** of FP16 in practice against **65** on the datasheet. Training happens anywhere; the winner used one A100 80GB or an RTX Pro 6000 Blackwell.

Training data is **36,696 rows** (13th place's count; the winner deduplicated to 35,960) over only **15 QuestionIds**, and Chris Deotte showed early that the test set uses the same 15. Each question has just 2 to 5 possible misconceptions.

![A triangle split into nine small triangles with six shaded, the question and four choices, two student cards with their labels, and the anatomy of the label False_Misconception:WNB](/imgs/blogs/kaggle-map-student-math-misunderstandings-solutions-3.webp)

The redrawn sample appears in several writeup prompts. "What fraction of the shape is not shaded?" A triangle is cut into 9 equal small triangles, 6 shaded; the answer is 1/3. A student who picks 1/3 and writes "There is 9 triangles and 3 aren't shaded" is tagged `True_Correct:NA`. A student who picks 3/6 and writes "3 of the triangles are not shaded and 6 are shaded so therefore it is 3/6" compares part to part instead of part to whole, the `WNB` misconception, so the label is `False_Misconception:WNB`.

| Label part | Values | Who decides it |
| --- | --- | --- |
| `True` / `False` | was the chosen answer right? | the answer key, so **derivable by rule** |
| `Correct` / `Misconception` / `Neither` | what the explanation shows | the annotator, the hard part |
| misconception name | `WNB`, `Incomplete`, ... or `NA` | the annotator, from a short per-question list |

That gives **65 combined labels**, or **37** once True/False is added back by rule. For the triangle question the candidate set seen in train is four: `Correct`, `Neither`, `Misconception:Incomplete`, `Misconception:WNB`. Nearly every top-20 team exploited that structure.

## 2. The metric, with a worked toy score

You submit up to three ranked guesses per row. A hit at guess $k$ scores $1/k$, a miss scores 0, averaged over rows:

$$
\mathrm{MAP@3} = \frac{1}{N}\sum_{i=1}^{N} \sum_{k=1}^{3} \frac{\mathbb{1}[\hat{y}_{i,k} = y_i]}{k}
$$

where $N$ is the number of rows, $y_i$ the true label of row $i$ and $\hat{y}_{i,k}$ your $k$-th guess.

![Four toy rows with the true label at guess 1, 2, 3 and missing, scoring 1, one half, one third and zero for a mean of 0.458, plus the 0.00091 gap between 1st and 6th translated into about 15 rows](/imgs/blogs/kaggle-map-student-math-misunderstandings-solutions-4.webp)

The toy scores four rows at 1, 0.5, 0.333 and 0, a mean of **0.458**. Now the real leaderboard through the same lens: the private set is about 8,000 rows, so moving one row's true label from guess 2 to guess 1 adds 0.5 / 8,000 = **0.0000625**. The gap between 1st (0.94919) and 6th (0.94828) is **0.00091**, about **15 rows** making that move. Six prize places were decided by the equivalent of 15 student explanations.

Guesses 2 and 3 are not wasted: `Correct` and `Neither` are plausible for almost any row, so an unsure model still banks half a point. The fight is over rare misconceptions ranked 4th or lower. The toy below (illustrative, not any team's code) shows the cheapest fix the top 20 used: zero every label never seen with this question.

```python
# Illustrative toy, NOT any team's code: MAP@3 plus the per-question label mask.
import numpy as np

LABELS = ["Correct:NA", "Neither:NA", "Misconception:Incomplete",
          "Misconception:WNB", "Misconception:Additive", "Misconception:Duplication"]
# Labels ever seen with this QuestionId in train (the 4th-place candidate set).
SEEN = {"q_triangle": {"Correct:NA", "Neither:NA",
                       "Misconception:Incomplete", "Misconception:WNB"}}

def map_at_3(truth, ranked):
    """Mean over rows of 1/position of the true label in the top 3, else 0."""
    score = 0.0
    for t, guesses in zip(truth, ranked):
        for pos, g in enumerate(guesses[:3], start=1):
            if g == t:
                score += 1.0 / pos
                break
    return score / len(truth)

def top3(probs, qid=None):
    p = probs.copy()
    if qid is not None:                       # zero labels never seen for this question
        p *= np.array([lab in SEEN[qid] for lab in LABELS], dtype=float)
    return [LABELS[i] for i in np.argsort(-p)[:3]]

truth = ["Correct:NA", "Misconception:WNB", "Neither:NA", "Misconception:Incomplete"]
probs = np.array([
    [0.70, 0.20, 0.05, 0.03, 0.01, 0.01],    # easy: right at guess 1
    [0.10, 0.30, 0.05, 0.25, 0.28, 0.02],    # WNB at guess 3, behind an impossible label
    [0.45, 0.30, 0.10, 0.05, 0.05, 0.05],    # Neither at guess 2
    [0.30, 0.25, 0.12, 0.08, 0.05, 0.20],    # Incomplete at 4th behind an impossible label
])
raw = [top3(p) for p in probs]
masked = [top3(p, "q_triangle") for p in probs]
print("raw    MAP@3 =", round(map_at_3(truth, raw), 4))   # 0.4583
print("masked MAP@3 =", round(map_at_3(truth, masked), 4)) # 0.5833
```

Masking promotes WNB from guess 3 to 2 and pulls Incomplete into the top 3, lifting the toy from 0.4583 to 0.5833. On real data the gain is far smaller, because a good model rarely ranks an impossible label high, but at a 0.00091 spread "far smaller" still decides medals.

## 3. What made it hard

None of the three hard parts was the model's reading ability.

**Label noise set a ceiling.** Deotte's forum thread asked whether public LB 0.950 was a "resistance level", and no team reached 0.955 public. The winner writes that validation scores "vary a lot using different seeds", mostly because of `Neither` labels; 13th place calls the labels "subjective with inconsistent annotation". There were outright errors too: 12 rows of QuestionId 31778 marked `True` for answer 9 when the answer is 6. Once a model agrees with the annotators as often as they agree with themselves, extra capacity learns noise. [Finding label noise](/blog/machine-learning/debugging-training/garbage-in-finding-label-noise) is the debugging companion.

**Small models were already close.** Deotte's threads showed a 1B encoder (Ettin) at public 0.940 to 0.942, near a Gemma2-9B baseline. The medal zone lived in the last 1% of the score range.

**The public leaderboard showed three decimals and was noisy anyway.** Seed noise of about ±0.002 (13th place's measurement) swamped most real improvements.

![Dumbbell chart of public rank versus private rank for the top 20, with tascj rising from 14 to 1, Kelvin from 99 to 20, and iws falling from 3 to 13](/imgs/blogs/kaggle-map-student-math-misunderstandings-solutions-5.webp)

Ranks moved a lot while scores barely moved. The public 1st finished 3rd; iws fell from 3rd to 13th, RAG from 5th to 17th; Kelvin rose from 99th to 20th. The winner climbed from 14th, and the leaderboard API confirms the folk tale: **tascj made 1 submission**, about 5.5 hours before the deadline, against 202 for 2nd place and 286 for 3rd.

## 4. The top-20 scorecard

Scores and ranks come from Kaggle's leaderboard API; ties (9/10, 11/12) keep API order. Ranks 1 to 13 took gold.

| Rank | Team | Private | Public rank | Key idea | Source |
| --- | --- | --- | --- | --- | --- |
| 1 | tascj | 0.94919 | 14 | Suffix classification, shared-prefix mask; full FT Qwen3-32B + GLM-Z1-32B; W8A8 + layer-wise T4 inference | [writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/1st-place-solution) |
| 2 | looking for remote job | 0.94890 | 2 | 80k LLM-written rows for rare labels; soft labels from 4 teachers | [writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/map2025-private-and-public-2nd) |
| 3 | monsaraida & Masaya | 0.94866 | 1 | Regularised Qwen3-14B classifier + Qwen2.5 CausalLM over the question's options | [writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/3rd-place-solution) |
| 4 | THLUO | 0.94835 | 13 | Pointwise Yes/No reranker with few-shot examples per candidate | [writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/4th-place-solution-single-model-pb-0-948-lb-0-951) |
| 5 | Naoism | 0.94830 | 24 | Diversity ensemble of 7B to 14B models, two from public notebooks | [writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/5th-place-solution) |
| 6 | Manan Jhaveri | 0.94828 | 22 | 37-class "Qwen-semble"; pseudo-labelled duplicates | [writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/mananjhaveri-map-6th-place-solution) |
| 7 | S T I T Y | 0.94820 | 10 | Label-token logits; ~50k synthetic rows; 32B soft labels | [writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/private-7th-public-10th-place-solution) |
| 8 | Takashi Someya | 0.94818 | 18 | Reconstructed a to d options; symbol labels; 50 models | [writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/8th-place-solution) |
| 9 | kanto-kaggler | 0.94801 | 28 | Two 32B CausalLMs x 5 folds, added label tokens, AWQ | [writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/9th-place-solution) |
| 10 | Liwen Zheng | 0.94801 | 17 | 15 classifiers; focal + smoothed CE; head served via vLLM embed | [writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/10th-place-solution) |
| 11 | sayoulala | 0.94797 | 4 | unknown | no writeup |
| 12 | Eisuke Okuda | 0.94797 | 19 | 14B students distilled from 72B teachers; rubric-based CV | [writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/12th-place-solution) |
| 13 | iws | 0.94792 | 3 | Classifier + listwise 32B/72B; chosen on synthetic holdout | [writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/map-competition-13th-place-gold-medal-solution) |
| 14 | AnnieGo & kyoumonemui | 0.94790 | 20 | unknown | no writeup |
| 15 | You've Already Won | 0.94786 | 12 | Qwen3 + Phi-4 family; shared Neither label | [writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/15th-place-solution-writeup) |
| 16 | Takoi | 0.94775 | 7 | unknown | no writeup |
| 17 | RAG | 0.94767 | 5 | unknown | no writeup |
| 18 | Chris Deotte | 0.94756 | 9 | "Pyramid" cascade; ~1,000 models trained | [writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/18th-place-pyramid-ensemble) |
| 19 | Hacker | 0.94752 | 70 | unknown | no writeup |
| 20 | Kelvin | 0.94748 | 99 | unknown | no writeup |

Fourteen of the twenty published a writeup. For the other six we know scores and nothing about methods.

## 5. The top-20 landscape

![Technique matrix: twenty rank columns with six shaded as no writeup, eleven technique rows, and per-row counts such as 13 of 13 known used a Qwen model and 2 of 14 used an uncertainty cascade](/imgs/blogs/kaggle-map-student-math-misunderstandings-solutions-6.webp)

The matrix recounts eleven techniques across the 14 writeups. A "?" means the writeup does not say and is excluded from the denominator: "8 of 9 known" means 8 used it, 1 did not, 5 were silent.

**Table stakes.** Every team that named a backbone used a **Qwen-family model** (13 of 13 known). **Quantised inference** was universal where stated (9 of 9), **vLLM** nearly so (9 of 11). Most fine-tuned with **LoRA or QLoRA** at learning rates of 1e-4 to 2e-4 for 1 to 4 epochs; only 4 of 13 known did full fine-tuning anywhere. And they exploited the shared questions: 9 of 13 known derived True/False by rule, and 8 of 9 scored only the labels the question can take.

**What separated the top 5 from 6 to 20.** With 0.00091 between 1st and 6th, read these as tendencies.

| Pattern | Top 5 | Ranks 6 to 20 |
| --- | --- | --- |
| Formulation | 1st suffix classification, 3rd per-question options, 4th pointwise reranker: **only the question's candidates** are scored | mostly 37- or 65-way classifiers or label-token CausalLMs |
| Model size | 1st to 4th all had a 32B+ member | 6th, 8th, 12th took gold at 14B or smaller |
| Noise handling | soft labels (2nd, 4th) or regularisation plus multi-seed validation (1st, 3rd) | ensembles of standard classifiers |
| Exception | 5th: nothing above 14B, a diversity ensemble picked on public LB | |

**Contradictions.** Showing the other answer choices in the prompt was worth +0.003 to +0.004 for Masaya (3rd) and appears in the 1st, 2nd, 7th, 8th, 13th and 18th place prompts, yet **lowered** the leaderboard for 15th and did nothing for 12th. LLM-written training data was central for 2nd and 7th, worth at most ~0.001 for 8th, and dropped by 12th, 13th and 18th. Rerankers were 4th place's whole solution and failed for 9th and 10th. Qwen2.5 beat Qwen3 for 9th and 13th; the reverse held for 15th. At this noise level, one team's ablation is one sample, not a finding.

## 6. The winning solution, explained

tascj's writeup is short; three ideas are worth rebuilding.

### 6.1 Suffix classification: score the candidates, not the vocabulary

You do not need the model to *generate* a label; you need it to *compare* 8 to 12 known candidates. Build one prompt, the "prefix": question, choices, correct answer, the question's common misconceptions, the student's answer and explanation. Append each candidate label as a "suffix" in submission format, such as `False_Correct:NA`. The last token's hidden state goes through `Linear(hidden, 1)`, giving one logit per candidate; softmax and cross-entropy over the candidates is the loss.

Run naively, that encodes the long prefix once per candidate. tascj packs one sequence, `prefix ++ suffix0 ++ suffix1 ++ ...`, with a FlexAttention mask so each suffix token sees the prefix and its own earlier tokens, never another suffix.

![A ten-by-ten attention mask: four prefix tokens lower-triangular, three two-token suffixes that each see the full prefix and themselves but not each other, with the last token of each suffix feeding a linear scorer](/imgs/blogs/kaggle-map-student-math-misunderstandings-solutions-7.webp)

A suffix row is filled across the prefix columns and its own diagonal block, empty over other suffixes. The writeup's rule is causal AND (same suffix OR key is prefix) AND same document. As a boolean matrix (illustrative toy, with an assumed 300-token prefix for the cost line):

```python
# Illustrative toy, NOT the 1st-place code: the shared-prefix suffix mask as a boolean matrix.
import numpy as np

def suffix_mask(prefix_len, suffix_lens):
    # suffix_id = -1 for prefix tokens, i for tokens of candidate suffix i
    suffix_id = np.array([-1] * prefix_len +
                         [i for i, n in enumerate(suffix_lens) for _ in range(n)])
    q = np.arange(len(suffix_id))[:, None]   # query positions (rows)
    k = np.arange(len(suffix_id))[None, :]   # key positions (columns)
    causal = q >= k
    key_is_prefix = suffix_id[None, :] == -1
    same_suffix = suffix_id[:, None] == suffix_id[None, :]
    return causal & (same_suffix | key_is_prefix)

prefix_len, suffix_lens = 4, [2, 2, 2]
m = suffix_mask(prefix_len, suffix_lens)
for row in m.astype(int):
    print("".join("#" if x else "." for x in row))

# Score positions: the last token of each suffix feeds Linear(hidden, 1).
ends = np.cumsum([prefix_len] + suffix_lens)[1:] - 1
print("score tokens:", ends.tolist())          # [5, 7, 9]

# Cost: packed sequence vs one full sequence per candidate (12 candidates, 300-token prefix).
P, S, C = 300, 6, 12
print("tokens packed:", P + C * S, " tokens unpacked:", C * (P + S))   # 372 vs 3672
```

The printed mask matches the figure, and the last line is the point: about 10x fewer tokens at 12 candidates under these assumed lengths. Two side effects matter as much as speed. The model can never put mass on a label the question cannot take, so the section 2 mask is built into the architecture. And every candidate is scored by the same head, so a rare misconception gets the same treatment as `Correct` instead of being one of 65 output rows the softmax barely trains.

### 6.2 Validation you can trust: seeds, loss, and one submission

The winner calls this "the most critical aspect of this competition". Single-seed validation was "highly unstable and misleading". He ran 5 folds x 5 seeds with Qwen3-8B, found a 3-seed ensemble stable, and from then on judged every idea on the **hardest fold** with a **3-seed ensemble**, trusting **loss** over MAP@3 when they disagreed.

![Dot chart of single-seed and 3-seed MAP@3 for seven models from 7B to 32B, with seed spread as wide as a model-size step and the two-family 32B ensemble at 0.9496](/imgs/blogs/kaggle-map-student-math-misunderstandings-solutions-8.webp)

DeepSeek-R1-Distill-Qwen-14B's three seeds span 0.9421 to 0.9450, a **0.0029** spread, larger than the 3-seed gap between Qwen3-8B (0.9455) and Qwen3-32B (0.9484). One seed can tell you a 14B is worse than a 7B; three seeds give a clean ladder, and mixing two 32B families (0.9496) beats either alone. Multi-seed ensembles were also "much better" than multi-fold ones, so the final models trained on full data with 3 seeds each. With that yardstick the public leaderboard had nothing to add, which is why one submission was enough. [Cross-validation done wrong](/blog/machine-learning/debugging-training/cross-validation-done-wrong) is the general version of this argument.

Training was full-parameter, `epoch=1, batch_size=32, learning_rate=1e-5`, using his own `offload_adam` (built during the [WSDM Cup](/blog/machine-learning/learn-from-kaggle/kaggle-wsdm-cup-multilingual-arena-solutions)) to fit a 32B fine-tune on one 80 GB GPU; [DeepSpeed ZeRO and offload](/blog/machine-learning/distributed-training/deepspeed-zero-and-offload) covers the idea at cluster scale. An auxiliary SFT loss on label justifications written by Qwen3-235B-A22B-Thinking helped R1-Distill-7B, not Qwen3-8B; he rates it marginal.

### 6.3 Two 32B models on two T4s

The plan was six 32B models (two families x three seeds) on hardware that cannot hold one in FP16.

- **Compute.** Short prefill-only workloads are dominated by `nn.Linear`. LMDeploy's W8A8 INT8 kernels sustained **40+ TFLOPS** on the T4 against ~20 for FP16, so every model was SmoothQuant-quantised (alpha 0.75) with ensemble validation "almost identical". [SmoothQuant and activation quantisation](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache) explains the alpha.
- **Memory.** Keep 2 transformer layers on the GPU and overlap computing layer k with loading layer k+1 from disk, feeding 640 samples per forward (40 micro-batches of 16).

About **65 minutes** for 16,000 rows; prefix caching, never implemented, could have halved it. The cautionary tale: Kaggle's `/tmp` is copy-on-write with a cap, so deleted files free no space. Layer checkpoints there crashed his runs, and only **4 of the 6** models made the final ensemble.

## 7. 2nd and 3rd in depth

### 2nd: looking for remote job (HZM and Baiph), denoise with soft labels

2nd on both leaderboards. Their framing: an easy classification scored by a ranking metric, so labels are ambiguous and the job is to **denoise**.

HZM's best private single model took four steps. Commercial LLMs (GPT-4, GPT-5, Claude Sonnet, Gemini, Seed, Doubao) wrote **80k** explanations for the 30-plus thin labels, each prompted with contrastive examples of the same misconception on the same and a different problem, under both `True_` and `False_`. Four teachers (Phi-4-reasoning-14B, Qwen3-32B, Mistral-12B, Qwen2.5-72B) trained on train.csv labelled that data, and their 5-fold out-of-fold averages became soft labels for the real rows. Qwen3-14B then trained with a modified TRL `SFTTrainer`: generation loss on the hard label letter plus cross-entropy between option-token logits and the soft label, over all 65 options lettered A to BM. Fold-0 CV 0.955, public 0.950, private 0.946. New to soft targets? Start with [knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals).

Baiph merged to 37 classes, had DeepSeek write a one-line description of every misconception as lettered options, and showed the question's four choices. A nuance the summaries miss: his table puts the three models trained on the 80k synthetic data in "submission 2" and his non-synthetic members in "submission 1 (best)"; the team writes that extra data plus distillation helped single models "but there is no benefit after fusion". The final fusion was four 37-class and four 65-class models at equal weight, all cast as multiple-choice generation so vLLM could serve them in time.

### 3rd: monsaraida & Masaya, the public leader

**monsaraida** trained a LoRA Qwen3-14B sequence classifier, chosen over ten bases including gemma-2-9b, three maths models, Mixtral-8x7B and a pruned gpt-oss. Showing **all choices plus the selected one** gained over +0.001; a per-question "common mistakes" hint list, +0.0003. A 65-class main head shared training with True/False, Category and 36-class misconception heads. Against noise: **R-Drop** (over +0.001 CV), **AWP** (slight), **EMA** (stability). On the T4, fp16 instead of bf16 was 2x faster and dropping max-length padding another 2x, so **4x**, spent on more ensemble members; a second stage re-inferred the least confident 50%.

**Masaya** used CausalLMs that answer with a letter from **only that question's options** (Correct, Neither, its 2 to 5 misconceptions), True/False added back by rule: +0.002 to +0.003 and about 3x faster. Listing the other answer choices added +0.003 to +0.004. LoRA r=16, alpha 32, all seven projections, 2 epochs, LR 1e-4.

| Model | CV (fold 0) | Public | Private | Inference |
| --- | --- | --- | --- | --- |
| Qwen2.5-14B-AWQ | 0.9469 | 0.943 | 0.942 | 30 min |
| Qwen2.5-32B-AWQ | 0.9498 | 0.949 | 0.944 | 60 min |
| Qwen2.5-72B-GPTQ | n/a | 0.950 | 0.946 | 190 min |
| Qwen2.5-72B-GPTQ with EMA | n/a | 0.950 | **0.948** | 190 min |
| Qwen2.5-32B-AWQ, no choices in prompt | 0.9474 | 0.945 | 0.941 | 60 min |

The 72B was merged, GPTQ-quantised and CPU-offloaded, and only low top-1-probability rows went to it, cutting its time to about an hour. The EMA 72B was the best private single model and was left out because its public score looked weaker: the public-LB trap the winner avoided.

## 8. Ranks 4 to 20, grouped by approach

### Family A: score each candidate separately (4th, 13th)

**4th, THLUO (solo).** A pointwise **reranker** borrowed from the Eedi 1st place: one candidate per prompt, "Does this diagnosis is correct? (Yes/No)", all 4 to 6 candidates of a row in one batch with cross-entropy over their Yes logits. The distinguishing idea was **prompt_v2**, up to 6 real explanations for *every* candidate inside the prompt: CV 0.9480 to 0.9493, but inference from 40 minutes to 4 hours. His best single model (Qwen3-Reranker-8B distilled from Qwen2.5-Math-72B, 2 epochs, full data) scored 0.95120 public and 0.94873 private, above his final blend (0.94835), because the Qwen3-32B-AWQ partner scored only 0.94655 private. See [reranker models](/blog/machine-learning/large-language-model/reranker-models-training-finetuning-case-studies) for pointwise versus listwise.

**13th, iws (solo, public 3rd).** Two Qwen3-14B-Base classifiers plus four **listwise** CausalLMs (Qwen3-32B and Qwen2.5-72B) that pick a numbered option from the question's shuffled targets; the 72B's `intermediate_size` was padded from 29568 to 29696 so vLLM tensor parallelism works. Models were chosen on **held-out synthetic data**, because on public "nearly any combination yielded similar scores"; relying on public "would likely have cost us the medal". He kept gold by 0.00002.

### Family B: label-token CausalLMs and synthetic data (7th, 8th, 9th)

**7th, S T I T Y (5 members).** Cross-entropy on label-token logits scores like a classifier head but serves at vLLM speed. Of ~200,000 GPT-4.1 and GPT-5-mini explanations, ~50,000 were used unfiltered, with a Qwen2.5-32B trained on real data supplying soft labels rather than trusting the synthetic ones. Gain: +0.002 to +0.005 LB on a Gemma2-9B baseline, mixed elsewhere.

**8th, Takashi Someya (solo).** Reconstructed the hidden a to d options from student explanations and rewrote questions as "Which number is the greatest? (a) 6 (b) 6.2 (c) 6.079 (d) 6.0001", so the model can see the distractors. Labels became single-token symbols; ~30,000 gpt-5 and o3 rows targeted cells with low out-of-fold MAP, gaining at most ~0.001. Final: 50 models.

**9th, kanto-kaggler (3 members).** Qwen2.5-32B-Instruct and R1-Distill-Qwen-32B, 5 folds each, QLoRA with 65 added label tokens. Moving inference from bitsandbytes to AWQ kept accuracy and cut time to about 1/6. Adding two 72B models to five 32B scored *below* ten 32B models on private.

### Family C: sequence-classification ensembles (5th, 6th, 10th, 12th, 15th)

**5th, Naoism (solo).** The thesis was **diversity**: the baseline gets about 94% right, errors live in rare `True_Misconception` rows and ambiguous explanations, and one model's quirks bias exactly those. Six 7B to 14B members mixed 37- and 65-class heads; two were **unmodified public notebooks**. Once single models reached ~0.947 public, CV and LB stopped correlating, so members were chosen on public LB with near-uniform weights. He calls it a gamble that paid.

**6th, Manan Jhaveri (solo).** A 37-label QLoRA ensemble of Qwen3-Embedding-8B, Qwen3-14B and Qwen2.5-14B. Conflicting duplicates were dropped, then **pseudo-labelled by the ensemble and added back**; GPT-4o wrote 100 to 120 rows for each of the 10 rarest labels. A 0.945 member trained on augmentation looked weak alone, yet removing it always hurt.

**10th, Liwen Zheng (solo).** Fifteen classifiers; focal loss mixed with smoothed cross-entropy, with **per-class weights phased in over ~33% of training**. The trick: split the merged 32B backbone from its classifier head, GPTQ the backbone, run vLLM in embed mode, apply the FP16 head outside. That serves a classification head at vLLM speed.

**12th, Eisuke Okuda (solo).** Five Qwen2.5-14B students **distilled from two 72B teachers**, the biggest single gain at +0.003 to +0.005. An LLM rubric score of explanation quality was used not as training data but to **rebuild the CV split**, because MAP@3 varied by rubric group.

**15th, You've Already Won (5 members).** Qwen3-14B and Phi-4 models on 37 classes, with JJ's **label sharing**: `Neither:NA` shared across questions, `Correct:NA` made question-specific, since "neither" language is generic and "correct" language is not. Over-tuned weights to public cost them: 15-plus unselected submissions scored 0.948 private or better, and they fell to silver.

### Family D: the contrasting entry, Chris Deotte's pyramid (18th)

<figure class="blog-anim">
<svg viewBox="0 0 760 410" role="img" aria-label="Chris Deotte's pyramid ensemble: small models score all test rows, then medium, large and extra-large models re-score the most uncertain 50%, 10% and 6% of rows with weights 2x, 4x and 8x" style="width:100%;height:auto;max-width:860px">
<style>
.mp-t{font:600 17px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.mp-s{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.mp-l{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.mp-bar{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.mp-tick{stroke:var(--border,#d1d5db);stroke-width:1}
.mp-hi{fill:var(--accent,#6366f1);opacity:.55;transform-box:fill-box;transform-origin:right center;animation:mp-shrink 16s ease-in-out infinite}
.mp-lad{fill:var(--accent,#6366f1);opacity:.75}
.mp-row{fill:var(--accent,#6366f1);opacity:0}
.mp-st{opacity:0}
.mp-static{opacity:0}
@keyframes mp-shrink{0%,22%{transform:scaleX(1)}25%,47%{transform:scaleX(.5)}50%,72%{transform:scaleX(.1)}75%,97%{transform:scaleX(.06)}100%{transform:scaleX(1)}}
@keyframes mp-w1{0%,22%{opacity:1}25%,97%{opacity:0}100%{opacity:1}}
@keyframes mp-w2{0%,22%{opacity:0}25%,47%{opacity:1}50%,100%{opacity:0}}
@keyframes mp-w3{0%,47%{opacity:0}50%,72%{opacity:1}75%,100%{opacity:0}}
@keyframes mp-w4{0%,72%{opacity:0}75%,97%{opacity:1}100%{opacity:0}}
@keyframes mp-r1{0%,22%{opacity:.14}25%,97%{opacity:0}100%{opacity:.14}}
@keyframes mp-r2{0%,22%{opacity:0}25%,47%{opacity:.14}50%,100%{opacity:0}}
@keyframes mp-r3{0%,47%{opacity:0}50%,72%{opacity:.14}75%,100%{opacity:0}}
@keyframes mp-r4{0%,72%{opacity:0}75%,97%{opacity:.14}100%{opacity:0}}
.mp-a1{animation:mp-w1 16s ease-in-out infinite}
.mp-a2{animation:mp-w2 16s ease-in-out infinite}
.mp-a3{animation:mp-w3 16s ease-in-out infinite}
.mp-a4{animation:mp-w4 16s ease-in-out infinite}
.mp-b1{animation:mp-r1 16s ease-in-out infinite}
.mp-b2{animation:mp-r2 16s ease-in-out infinite}
.mp-b3{animation:mp-r3 16s ease-in-out infinite}
.mp-b4{animation:mp-r4 16s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.mp-hi,.mp-st,.mp-row{animation:none}.mp-hi{transform:none}.mp-st,.mp-row{opacity:0}.mp-static{opacity:1}}
</style>
<text class="mp-t" x="40" y="30">Pyramid ensemble (18th place): bigger models only see the hardest rows</text>
<text class="mp-s" x="40" y="56">all test rows, sorted by current uncertainty (most uncertain on the right)</text>
<rect class="mp-bar" x="40" y="68" width="680" height="46" rx="6"/>
<rect class="mp-hi" x="40" y="68" width="680" height="46" rx="6"/>
<line class="mp-tick" x1="380" y1="64" x2="380" y2="118"/>
<line class="mp-tick" x1="652" y1="64" x2="652" y2="118"/>
<line class="mp-tick" x1="679" y1="64" x2="679" y2="118"/>
<text class="mp-s" x="40" y="134">0%</text>
<text class="mp-s" x="362" y="134">50%</text>
<text class="mp-s" x="612" y="134">10%</text>
<text class="mp-s" x="684" y="134">6%</text>
<text class="mp-l mp-st mp-a1" x="40" y="164">Stage 1: small models score 100% of rows, weight 1x</text>
<text class="mp-l mp-st mp-a2" x="40" y="164">Stage 2: medium models re-score the most uncertain 50%, weight 2x</text>
<text class="mp-l mp-st mp-a3" x="40" y="164">Stage 3: large models re-score the most uncertain 10%, weight 4x</text>
<text class="mp-l mp-st mp-a4" x="40" y="164">Stage 4: an extra-large model re-scores the top 6%, weight 8x</text>
<text class="mp-l mp-static" x="40" y="164">Four stages: 100% (1x), 50% (2x), 10% (4x), 6% (8x) of rows</text>
<rect class="mp-row mp-b1" x="30" y="180" width="700" height="34" rx="6"/>
<rect class="mp-row mp-b2" x="30" y="220" width="700" height="34" rx="6"/>
<rect class="mp-row mp-b3" x="30" y="260" width="700" height="34" rx="6"/>
<rect class="mp-row mp-b4" x="30" y="300" width="700" height="34" rx="6"/>
<text class="mp-l" x="40" y="202">1 small models, 1x</text>
<text class="mp-l" x="40" y="242">2 medium models, 2x</text>
<text class="mp-l" x="40" y="282">3 large models, 4x</text>
<text class="mp-l" x="40" y="322">4 extra-large model, 8x</text>
<rect class="mp-lad" x="260" y="186" width="460" height="22" rx="4"/>
<rect class="mp-lad" x="490" y="226" width="230" height="22" rx="4"/>
<rect class="mp-lad" x="674" y="266" width="46" height="22" rx="4"/>
<rect class="mp-lad" x="692" y="306" width="28" height="22" rx="4"/>
<text class="mp-s" x="40" y="360">uncertainty is recomputed after every stage, so each subset is re-picked</text>
<text class="mp-t" x="40" y="392">CV 0.951, public 0.952, private 0.947</text>
</svg>
<figcaption>Chris Deotte's 18th-place pyramid: each stage re-scores a smaller, harder slice of the test rows with a bigger model and a larger weight.</figcaption>
</figure>

Deotte trained about 1,000 models over three months on roughly four A100s, across sequence classification, single-token CausalLM, shuffled multiple choice with at most 6 options and a three-head model, on backbones from 1B to 72B. The final submission spends compute where uncertainty is: small models on all rows, medium on the most uncertain 50% at 2x weight, large on 10% at 4x, extra-large on 6% at 8x. CV 0.951, public 0.952, private 0.947. It is the opposite strategy to the winner (breadth and a cascade versus two validated 32B families), and his synthetic-data verdict is blunt: "mostly not" an improvement, so he used none.

### The six without writeups

**sayoulala (11th), AnnieGo & kyoumonemui (14th), Takoi (16th), RAG (17th), Hacker (19th) and Kelvin (20th)** published no solution the top-20 search could find (forum, web, GitHub, Zenn, Medium). Their methods are unknown and their matrix columns are shaded. RAG's conjuring92 won the Eedi competition whose reranker 4th place reused, but nothing MAP-specific is published. Kelvin's +79 and Hacker's +51 rank climbs, the largest in the top 20, stay unexplained.

## 9. What the forum taught everyone

- **"Does Test Data Have Questions Different Than Train Data?"** (Deotte, discussion 589400): same 15 questions, True/False derivable, the QuestionId 31778 errors. This thread made 37-class models and per-question masks standard.
- **"Is Public LB 0.950 a Resistance Level?"** (discussion 591054): the noise ceiling, named early.
- **Public notebooks became ensemble members.** Deotte's Gemma2-9B classifier, jaytonde's DeepSeek-Math-7B and sinchir0's full-FT Qwen3-8B notebook seeded at least 5th, 7th, 9th and 10th.
- **"The Rise of the 1-Shot Submit"** (hengck23, discussion 611998): the post-mortem on the winner's single submission.
- **"Why Are Gold Medal Teams Skipping Qwen3 for Qwen2.5?"** (discussion 612201): never settled.

## 10. What did not work

From the 14 writeups; "failed" means no leaderboard gain for that team, not a law.

| Idea | Reported failing | Note |
| --- | --- | --- |
| Chain of thought before the label | 4th (biggest time sink), 13th | |
| Listwise reranker | 4th, 10th | worked for 13th |
| Pointwise or pairwise rerank on a classifier | 9th, 10th; pairwise too slow for 3rd | |
| Retrieved similar explanations in the prompt | 4th | random per-label examples did work |
| 70B to 72B under the time limit | 9th, 13th | 3rd's EMA 72B was its best private single |
| Llama-3.3-70B, Gemma, Mistral, gpt-oss-20b | 3rd, 9th; gemma-2-9b unstable for 15th | Gemma2-9B worked for 5th, 7th |
| LLM synthetic training data | 12th, 13th, 18th | worked for 2nd, 7th |
| ~10k rule-based augmentations | 6th | ~50 per category helped |
| Two-stage or hierarchical Category then Misconception | 6th, 10th | |
| Focal, bootstrapping or smoothing losses | 6th; inconsistent for 15th | worked for 10th |
| Correct-vs-Neither swap post-processor | 15th | |
| Merging fold LoRAs; distilling into ~1B encoders | 9th; 12th | |
| Flipping True/False on QuestionId 31778 at inference | 3rd | 4th, 9th, 10th fixed train rows instead |

## 11. Lessons that transfer

1. **When labels are noisy, build the yardstick before the model.** The winner judged changes on a hardest-fold, 3-seed ensemble and on loss; 13th used a synthetic holdout. Both say public LB would have misled them. (1st, 13th.)
2. **Shrink the output space to what this input can be.** Rule-derived True/False and scoring only the question's candidates was faster and more accurate for 1st, 3rd and 4th. (1st, 3rd, 4th.)
3. **Score many candidates against one shared prefix.** A suffix mask or prefix caching encodes context once. (1st.)
4. **Old GPUs reward systems work.** INT8 kernels at 40+ TFLOPS versus ~20, layer streaming, fp16 not bf16, no padding turned 14B budgets into 32B budgets. (1st, 3rd.)
5. **Spend big-model compute only on rows that need it.** Masaya's 72B on low-confidence rows and Deotte's pyramid. (3rd, 18th.)
6. **Soft labels beat louder hard labels.** Teacher soft targets handled noise for 2nd, 4th, 7th and 12th. (2nd, 4th, 7th, 12th.)

### If you enter a competition like this

- Read the data thread first: what overlaps between train and test, and what does it make free?
- Measure seed variance on day one; if it exceeds the gains you hope for, validate on multi-seed ensembles.
- Trust validation loss over a noisy ranking metric.
- Restrict outputs per input, in the architecture if you can.
- Profile the inference GPU early: dtypes, real TFLOPS, disk (`/kaggle/working`, not `/tmp`).
- Use synthetic data for rare classes with soft labels, and expect it to help single models more than ensembles.
- Pick final submissions from your offline yardstick.

## Sources

- Competition: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings (scores from Kaggle's leaderboard API)
- 1st, tascj: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/1st-place-solution ; https://www.kaggle.com/code/tascj0/map-submit ; https://github.com/tascj/kaggle-map-charting-student-math-misunderstandings ; https://github.com/tascj/offload_adam
- 2nd: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/map2025-private-and-public-2nd
- 3rd: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/3rd-place-solution
- 4th: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/4th-place-solution-single-model-pb-0-948-lb-0-951
- 5th: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/5th-place-solution
- 6th: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/mananjhaveri-map-6th-place-solution ; https://github.com/mananjhaveri/MAP-Charting-Student-Math-Misunderstandings-6th-place-solution
- 7th: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/private-7th-public-10th-place-solution ; https://www.kaggle.com/datasets/isakatsuyoshi/map-synthetic-dataset
- 8th: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/8th-place-solution
- 9th: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/9th-place-solution ; https://zenn.dev/mkj/articles/1f374627377474
- 10th: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/10th-place-solution
- 12th: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/12th-place-solution
- 13th: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/map-competition-13th-place-gold-medal-solution
- 15th: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/15th-place-solution-writeup ; https://github.com/T3pp31/map-charting-student-math-misunderstandings
- 18th: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/18th-place-pyramid-ensemble
- Forum threads 589400, 591054, 611998, 612201: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion

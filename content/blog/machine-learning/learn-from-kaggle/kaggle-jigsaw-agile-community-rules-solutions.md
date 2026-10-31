---
title: "Jigsaw Agile Community Rules on Kaggle: how the top 20 turned the test set's own examples into a training set"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "A synthesis of the top-20 Jigsaw Agile Community Rules solutions: why every winner fine-tuned a LoRA inside the 12-hour notebook on the labelled examples shipped in test.csv, how one Yes/No logit became a classifier, why per-rule AUC rewards rank normalisation, and why 0.00317 separated 1st from 20th."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "text-classification", "content-moderation", "test-time-training", "lora", "unsloth", "auc", "ensembling", "pseudo-labeling"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 22
---

> **TL;DR.** Winner: **Guanshuo Xu, solo, private AUC 0.92930**, also 1st on the public leaderboard. Decisive lever: **test-time compute**, in its training form. Every row of the hidden test file carried two labelled comments that broke its rule and two that did not, so the test file was also a training set for rules nobody had seen. Every top solution fine-tuned 7B to 14B instruct models with LoRA *inside* the 12-hour submission notebook on those examples, then read the probability of a violation from one next-token logit. Xu's edge was craft: loss on the Yes/No token only, log-odds over answer-token variants, per-rule rank normalisation, six models. The lesson: when the test set ships labelled examples, the examples are the model. What separated the top 8 from ranks 9 to 20 was fitting a 13B to 14B model into two T4s.

![The five-lever strip with Test-time compute lit as decisive and Data and Metric insight half-lit](/imgs/blogs/kaggle-jigsaw-agile-community-rules-solutions-1.webp)

The strip is the series' mental model: five places a Kaggle LLM competition can be won. Jigsaw Agile Community Rules was won on **test-time compute**, though not the sampling-and-voting kind that decided the [AIMO competitions](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-1-solutions). The extra compute at test time was *training*: a LoRA fitted on each hidden rule's labelled examples during the submission run, the idea that also powered the [ARC Prize 2024](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2024-solutions). **Data** is half-lit because two cheap cleaning steps (drop the subreddit column, vote away conflicting labels) recur in the winning writeups. **Metric insight** is half-lit because the metric averages AUC per rule, which rewards normalising scores within a rule before blending.

The competition ran from 2025-07-23 to 2025-10-23, hosted by Jigsaw and Conversation AI, with 2,445 teams and USD 100,000 in prizes (35k, 25k, 15k, then 5k each for 4th to 8th). If the notebook format is new to you, start with [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work).

## 1. The task in 60 seconds

A Reddit moderator writes a rule ("no advertising"), and a comment arrives. Does the comment break the rule? Your model outputs a probability. The twist: training covers two rules, and the hidden test set asks about rules you have never seen.

![The notebook budget: two T4 GPUs with 16 GB each, twelve hours, internet off, about 55,000 hidden rows split 30% public and 70% private, two rules in train and six in test with four never seen](/imgs/blogs/kaggle-jigsaw-agile-community-rules-solutions-2.webp)

It is a code competition: your notebook runs on **two NVIDIA T4s** with 16 GB each, at most **12 hours**, **internet off**. Public external models were allowed, so you could bring Qwen3-14B weights as a dataset, but not call an API. The official pages give no test row count; 8th place described the public share as "29% of 55k rows", so about **55,000 rows**, split **30:70** public to private. The 1st-place writeup says the host confirmed the split was random, which is why most top teams trusted the public leaderboard.

The training file has only **2 rules**. The test file has **6**, 4 of them unseen. A model fitted on train.csv alone would judge "no medical advice" having only seen advertising and legal-advice violations.

![An illustrative test row: comment body, rule, a struck-through subreddit, two positive and two negative example comments, and arrows peeling the four examples into a stack of training rows](/imgs/blogs/kaggle-jigsaw-agile-community-rules-solutions-3.webp)

Each row has a `body`, a `rule`, a `subreddit`, and four more columns: `positive_example_1`, `positive_example_2`, `negative_example_1`, `negative_example_2`, labelled comments that do and do not violate *this rule*. The host meant them as few-shot context. The row in the figure is invented; the shape is real.

So every test row carries four labelled comments for its rule. Deduplicated, that is a training set per hidden rule, sitting in the file your notebook receives at submission time. Early on, neibyr posted a notebook, "LB 0.878 train on test data", that fine-tuned a 0.5B Qwen2.5 on exactly those examples, and asked in the forum whether that was a rule grey area. It stayed legal and became the foundation of every winning solution.

Labels were noisy. The same (body, rule) pair sometimes appears with both labels, and Psi's forum thread asked whether the labelling was synthetic, since labels in the 2-million-comment source corpus felt random.

## 2. The metric, with a worked toy score

The metric is **column-averaged AUC**: ROC AUC per rule, then the plain mean.

$$
\text{Score} = \frac{1}{R}\sum_{r=1}^{R} \text{AUC}_r, \qquad
\text{AUC}_r = \frac{1}{|P_r|\,|N_r|}\sum_{p \in P_r}\sum_{n \in N_r} \mathbb{1}[s_p > s_n] + \tfrac{1}{2}\,\mathbb{1}[s_p = s_n]
$$

$P_r$ and $N_r$ are the violating and clean comments under rule $r$, and $s$ is your score. AUC is the fraction of (violator, clean) pairs you order correctly. Two consequences follow. Only the *order* within a rule matters, so any monotone transform of one rule's scores leaves its AUC unchanged. And scores are never compared *across* rules.

![Two number lines, one per rule: advertising scores perfectly ordered for AUC 1.000, spoilers with one mis-ordered pair for AUC 0.750, mean 0.875, versus a pooled AUC of 0.688](/imgs/blogs/kaggle-jigsaw-agile-community-rules-solutions-4.webp)

The toy: eight comments, two rules, two violators and two clean comments each.

```python
# Illustrative toy, NOT any team's code: column-averaged (per-rule) AUC.
from itertools import product

def auc(labels, scores):
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in product(pos, neg))
    return wins / (len(pos) * len(neg))

rows = [  # (rule, label, score)
    ("advertising", 1, 0.95), ("advertising", 1, 0.80),
    ("advertising", 0, 0.70), ("advertising", 0, 0.40),
    ("spoilers",    1, 0.35), ("spoilers",    1, 0.10),
    ("spoilers",    0, 0.20), ("spoilers",    0, 0.05),
]
rules = sorted({r for r, _, _ in rows})
per_rule = {}
for r in rules:
    ys = [y for rr, y, _ in rows if rr == r]
    ss = [s for rr, _, s in rows if rr == r]
    per_rule[r] = auc(ys, ss)
    print(f"AUC[{r}] = {per_rule[r]:.3f}")
print(f"column-averaged AUC = {sum(per_rule.values()) / len(rules):.3f}")
print(f"pooled (global) AUC = {auc([y for _, y, _ in rows], [s for _, _, s in rows]):.3f}")
```

```console
AUC[advertising] = 1.000
AUC[spoilers] = 0.750
column-averaged AUC = 0.875
pooled (global) AUC = 0.688
```

Under "advertising" both violators (0.95, 0.80) sit above both clean comments (0.70, 0.40): 4 of 4. Under "spoilers" the violator at 0.10 sits below the clean comment at 0.20: 3 of 4. The mean is 0.875. Pooled into one AUC the same scores give 11 of 16 = 0.688, because every spoiler is scored below every advertising comment, and the competition metric does not care. That is why writeups talk about **per-rule rank normalisation**: when you average two models, one whose scores for a rule are squeezed into a narrow band barely moves that rule's ordering. Converting each model's scores to within-rule ranks scaled to [0, 1] gives every model an equal say in every rule. Whether that helped is one of the competition's disagreements (section 9).

The winner beat 2nd by **0.00007** AUC and 20th by **0.00317**: a handful of pair orderings out of millions.

## 3. What made it hard

**Unseen rules.** Offline fine-tuning can teach "judge a comment against a rule" in general, but the 4 hidden rules could only be learned from examples that existed only inside the submission run. Training had to happen in the notebook, within 12 hours, on two T4s. That turns a modelling competition into a systems competition: how big a model can you fine-tune *and* run over 55,000 rows before the clock runs out?

**The T4.** 16 GB and no bfloat16; 12th place reported public notebooks running bf16 on T4 about 6 times slower. A 14B model meant 4-bit weights and LoRA, in practice [Unsloth](/blog/machine-learning/open-source-library/unsloth-vram-budget-and-export). The top writeups read like ops runbooks: one model per GPU, length-sorted inference, memory headroom because identical reruns crashed.

**Noise.** Chris Deotte asked whether 0.931 public was a ceiling, since the leaderboard sat there for weeks. 4th place measured about **0.002 AUC** of seed noise. The whole top 20 sits inside 0.00317.

![Dumbbell chart of public versus private rank for the private top 20: Guanshuo Xu held 1st, team ^^ rose from 6 to 2, YF rose from 29 to 18, next gen VNG warriors fell from 8 to 14](/imgs/blogs/kaggle-jigsaw-agile-community-rules-solutions-5.webp)

Yet the shake-up was mild. Public ranks 1 to 5 finished private 1, 5, 4, 8 and 3. With a random 30:70 split of about 55,000 rows, the public leaderboard was a decent validation set. The biggest riser, YF from public 29th to private 18th, built its own leak-free local validation.

## 4. The top-20 scorecard

Private leaderboard order from Kaggle's leaderboard API. "Move" is public rank minus private rank (positive = rose). 15 of 20 published a writeup.

| Rank | Team | Private AUC | Public rank (move) | Key idea | Source |
|---|---|---|---|---|---|
| 1 | Guanshuo Xu | 0.92930 | 1 (0) | Six in-notebook LoRAs, answer-token loss, per-rule rank blend | [writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613305) |
| 2 | ^^ | 0.92923 | 6 (+4) | Majority-vote label cleaning, four 4-bit LoRAs | [writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/614279) |
| 3 | Sergio Manuel Papadakis | 0.92871 | 5 (+2) | Pretrained LoRA init, LLM embeddings into GBDT/SVC | [writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613324) |
| 4 | Dream Team | 0.92860 | 3 (-1) | SALSA, soft labels, uncertainty pseudo-labels | [writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613473) |
| 5 | Flash | 0.92855 | 2 (-3) | Paradigm-diverse stack: LLM, NLI, embedders | [writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613421) |
| 6 | ducnh279 | 0.92781 | 12 (+6) | Deep Mutual Learning of three Qwen3 peers | [writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613150) |
| 7 | ktr | 0.92760 | 9 (+2) | Safety-tuned backbones, ShieldGemma-9B best | [writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613215) |
| 8 | Reddit comment moderators | 0.92759 | 4 (-4) | Qwen3-14B x3 + Llama2-13B + bge, 4-seed configs | [writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613195) |
| 9 | lllleeeo + mango | 0.92747 | 7 (-2) | Small TTT LLMs + encoders + triplet retrieval | [writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613185) |
| 10 | Nikita Babych | 0.92745 | 13 (+3) | unknown | no writeup |
| 11 | takaito/Muj!rush!/yukko/miyaki | 0.92734 | 18 (+7) | Online BERT + SentenceBERT + 3B/4B LLMs, example TTA | [writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613506) |
| 12 | losingself | 0.92702 | 16 (+4) | Found the source corpus, distilled into DeBERTa | [writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613096) |
| 13 | ISE@JNU-1 | 0.92698 | 10 (-3) | Cross-rule negatives as augmentation | [writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613265) |
| 14 | next gen VNG warriors | 0.92678 | 8 (-6) | unknown | no writeup |
| 15 | hasibirok0 / kfsky / Kinosuke | 0.92657 | 15 (0) | Seven-model blend, synthetic rows, compliance prompt | [writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613611) |
| 16 | PPD | 0.92633 | 25 (+9) | unknown | no writeup |
| 17 | Ebi | 0.92625 | 11 (-6) | Qwen3-32B paraphrases inside the run | [writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613071) |
| 18 | YF | 0.92617 | 29 (+11) | Multiple-choice pre-task, leak-free local CV | [writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/617588) |
| 19 | Fidel Escalona | 0.92613 | 23 (+4) | unknown | no writeup |
| 20 | Artem Silevich | 0.92613 | 27 (+7) | unknown | no writeup |

Ebi's post is titled "18th Solution (public 11th)", but the API ranks the team 17th; this post uses the API rank. Ranks 19 and 20 tie.

## 5. The top-20 landscape

![Technique matrix: twenty rank columns with five shaded as no writeup, twelve technique rows, and per-row counts such as 14 of 15 trained on the test set's examples and 2 of 15 used a safety-tuned base](/imgs/blogs/kaggle-jigsaw-agile-community-rules-solutions-6.webp)

The matrix covers the 15 teams with writeups. Filled means in the final solution, empty means not used or dropped, "?" means not stated. Counts are "of 15", so every number is a floor.

**Table stakes:**

- **Train on the test set's labelled examples:** 14 of 15. Only 17th does not say where its training rows came from.
- **A Qwen3 model in the final blend:** 14 of 15; only 12th, built on a distilled DeBERTa, did without.
- **One next-token Yes/No readout:** 11 of 15. 3rd fed LLM embeddings to tree heads and 8th used a classification head.
- **Drop subreddit:** 9 of 15 said so; 15th kept it.
- **Vote-clean or deduplicate labels:** 9 of 15.

**What separated the top 8: model size, engineered into the budget.** Fine-tuning a 13B or larger LLM appears in 9 of 15 writeups: every one of ranks 1 to 8, then only 17th. Ranks 9, 11, 12, 13, 15 and 18 topped out at 4B to 8B and leaned on encoders; 11th said anything above 4B was too slow at inference. 5th is the asterisk: its largest *test-time-trained* models were 8B, and its Qwen2.5-32B was trained offline on the official data only.

On top of size, each top-8 team had one distinctive training signal: answer-token loss masking (1st), majority-vote labels (2nd), pretrained LoRAs as the starting point (3rd), soft labels and uncertainty pseudo-labels (4th), paradigm diversity (5th), mutual learning (6th), moderation-tuned backbones (7th), seed-controlled ensembles (8th).

## 6. The winning solution, explained

Guanshuo Xu worked solo. His writeup is short, and almost every choice in it is a small, measured improvement on the public train-on-test notebook.

### 6.1 Validation: the public leaderboard

His first decision: **validation = public leaderboard**. The host said the split is random, 30% of a large test set has low variance, and local CV on 2 rules cannot represent the 4 new ones. Every model was fine-tuned inside the submission run, so each experiment cost a full submission but measured exactly what would be scored.

### 6.2 Data: the test examples, minus the subreddit

From neibyr's notebook, each test row's four examples become (rule, comment, label) training rows alongside the train set. Two changes: each LLM's native chat template, and **drop the subreddit column before deduplication**, which he called a "significant" improvement. The mechanism: the same example comment appears on many rows. Keyed with subreddit, each copy survives deduplication and popular examples are counted many times; keyed on (body, rule), they collapse. The subreddit also invites a shortcut that does not transfer to unseen rules.

Test examples were upsampled once and everything trained for 1 epoch, so effectively one epoch on train and two on test examples. No LoRA rank, alpha or learning rate is stated.

### 6.3 Loss on one token

![Token strip for one training sequence with only the Yes answer token carrying loss, then the answer-position logits of five Yes-variants and five No-variants merged by logsumexp into P(yes), feeding a per-rule rank blend](/imgs/blogs/kaggle-jigsaw-agile-community-rules-solutions-7.webp)

A chat-formatted example is hundreds of tokens, and chat templates insert extra tokens around the answer, such as Qwen3's empty `<think>\n\n</think>\n\n` block. Standard fine-tuning puts loss on every assistant token. Xu **masked the loss to the single Yes/No token**: training got faster, and convergence became consistent across model families, so one hyperparameter set worked for all. When the only thing you read is one token, gradient spent on `</think>` is wasted.

He reached 0.931 public early with small models in plain transformers and PyTorch, then moved to **Unsloth** to fit 14B models into 16 GB with [4-bit weights](/blog/machine-learning/open-source-library/unsloth-4bit-quantization-nf4). If LoRA is unfamiliar, the [LoRA and PEFT debugging guide](/blog/machine-learning/debugging-training/debugging-lora-and-peft) covers how the adapter works and fails.

### 6.4 Read a classifier out of one logit

At inference he never calls `generate()`. One `forward()` pass gives the logits at the answer position. A model may put its "yes" mass on `Yes`, `yes` or `True`, so he takes the log-odds of a set of Yes-variant tokens against a set of No-variant tokens. The writeup says "sigmoid of log-odds of Yes_ids vs No_ids"; pooling each set with logsumexp, as below, is our reading:

$$
P(\text{yes}) = \sigma\!\Big(\log\!\sum_{t \in \mathcal{Y}} e^{z_t} \;-\; \log\!\sum_{t \in \mathcal{N}} e^{z_t}\Big)
$$

with $\mathcal{Y}$ = {Yes, YES, Y, yes, True} and $\mathcal{N}$ = {No, NO, N, no, False}, plus space-prefixed forms, and $z_t$ the logit of token $t$. Inputs are sorted by length to cut padding.

```python
# Illustrative toy, NOT the 1st-place code: Yes/No readout from one next-token logit vector.
import math

def lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))

# Pretend logits at the answer position for the variant tokens (invented numbers).
logits = {"Yes": 3.1, "YES": 0.4, "Y": -0.2, "yes": 1.0, "True": -1.5,
          "No": 2.2, "NO": -0.3, "N": -0.9, "no": 0.6, "False": -2.0}
yes = [logits[t] for t in ("Yes", "YES", "Y", "yes", "True")]
no = [logits[t] for t in ("No", "NO", "N", "no", "False")]
p_yes = 1 / (1 + math.exp(-(lse(yes) - lse(no))))
p_plain = 1 / (1 + math.exp(-(logits["Yes"] - logits["No"])))
print(f"LSE(yes)={lse(yes):.3f}  LSE(no)={lse(no):.3f}")
print(f"P(yes), variant sets   = {p_yes:.3f}")
print(f"P(yes), Yes vs No only = {p_plain:.3f}")
```

```console
LSE(yes)=3.312  LSE(no)=2.496
P(yes), variant sets   = 0.694
P(yes), Yes vs No only = 0.711
```

In this invented example the variant sets move the probability from 0.711 to 0.694 because "no" spread its mass over `no` and `NO`. After a fine-tune that trained only `Yes` and `No`, variants carry little mass and the readouts mostly agree; the sum is insurance against a template that prefers another spelling. What AUC needs is a continuous score, not a hard label from greedy decoding.

### 6.5 Six models, ranked per rule

![Dot chart of public and private AUC for the winner's six models and the weighted ensemble, from Ettin-400M at 0.8944 private to Qwen3-14B at 0.9239 and the ensemble at 0.9293](/imgs/blogs/kaggle-jigsaw-agile-community-rules-solutions-8.webp)

Each model's scores were ranked **within each rule**, normalised to [0, 1], then combined as a weighted average. Early on, per-rule ranking alone moved his ensemble from 0.929 to 0.931 public. Public / private AUC: Qwen3-14B 0.9297 / 0.9239; Qwen2.5-14B 0.9287 / 0.9232; Qwen3-8B 0.9272 / 0.9236; Qwen3-4B-Instruct-2507 0.9258 / 0.9198; Llama3.1-8B 0.9257 / 0.9202; the encoder Ettin-400M 0.8991 / 0.8944; **weighted ensemble 0.9344 / 0.9293**.

Larger was better, and Qwen3 beat the other families. The ensemble added 0.0047 public and 0.0054 private over the best single model, more than twice the 0.002 seed noise. Ettin sits 0.03 behind the LLMs and still earns a place: cheap diversity. The whole solution is a public baseline with five measured changes, each aimed at the actual metric and the actual hardware.

## 7. 2nd and 3rd in depth

### 2nd: team ^^, clean labels and keep the GPUs busy

Team ^^ (andyy12345, yaoqingjin, trudih, yfr999, hoatha) rose from public 6th to private 2nd, 0.00007 behind. They started from mks2192's public notebook showing a one-epoch Unsloth LoRA of Llama-3.1-8B fits in a Kaggle run, and unlike Xu built **local CV** (StratifiedKFold, 10% holdout, mean AUC over 3 seeds), because the public leaderboard swung and identical notebooks sometimes crashed.

Their central step was **label cleaning by majority vote**: key on `body || rule`, ignoring subreddit, take the majority label, then deduplicate. Fewer, cleaner rows suit a one-epoch LoRA. Subreddit was dropped as a shortcut, since the same behaviour is judged differently in different subreddits.

Models: phi-4 (14B), Qwen3-14B, Qwen3-8B and Qwen2.5-7B-Instruct, bitsandbytes 4-bit via Unsloth, LoRA with `trl.SFTTrainer`, one epoch, scored by the Yes/No logit difference and fused with weights. The engineering is worth copying: two model groups train on the two GPUs at once via `CUDA_VISIBLE_DEVICES`, 11 hours for the final submission, using only **80 to 90% of GPU memory** because reruns sometimes died with "Notebook Threw Exception". A run that crashes at hour 11 scores nothing.

### 3rd: Sergio Manuel Papadakis, a pretrained start and embeddings into trees

Papadakis, solo, finished 3rd from public 5th. Before submission he trained LoRAs for one epoch on all public rules, bodies and examples with a zero-shot prompt ending in "Violation:", so test-time training starts from a LoRA that already knows the task.

**The slow ensemble:** two each of Qwen2.5-14B, Qwen2.5-7B, Qwen3-14B and Qwen3-8B, one per T4. vLLM returns not a token but the **embedding of the final "Violation:" token**, and LightGBM, XGBoost and SVC heads are fitted on those embeddings. The LLM becomes a feature extractor. The two LoRAs of each 7B/8B model were merged to save time.

**The fast ensemble:** two bge-base-en-v1.5 models trained with triplet loss and the rule as anchor (one on *negated* rules), using distances to the positive and negative centroids as features; Qwen3-Embedding-0.6B with ArcFace and per-rule heads; DeBERTa-v3-small.

Partial ensembles, public / private: 2x Qwen3-14B 0.92571 / 0.92049; 2x Qwen2.5-7B 0.91024 / 0.90637; 2x bge 0.90952 / 0.89714; DeBERTa-v3-small 0.89546 / 0.88946. The final 0.92871 private is 0.0082 above his best group.

## 8. Ranks 4 to 20, grouped by approach

### Family A: one LLM stack, a different training signal (4th, 6th, 7th, 8th)

**4th, Dream Team (Ruslan Berdichevsky, solo): SALSA**, written up as a paper (arXiv 2510.22691). Constrain the answer to "0" or "1" and read the next-token logits as the classifier. Duplicates collapse into a **soft target**, the per-text violation ratio, trained with Generalized Cross-Entropy (q = 0.955), which barely beat plain cross-entropy on private (0.92313 vs 0.92309). Batches are grouped by rule so the rule prefix stays in the KV cache. Then two stages: an AWQ Qwen3-4B scores the test set, the ~8,000 *most uncertain* rows become soft pseudo-labels, and an AWQ Qwen3-14B trains on them; the blend puts 0.8 on Stage 1 and 0.2 on Stage 2. Final 0.9286 private versus 0.92469 for a plain 4B + 14B blend, about +0.004.

**6th, ducnh279 (solo): Deep Mutual Learning.** No room for a bigger teacher, offline distillation would cover only 2 rules, and the teacher's softmax was nearly one-hot. So peers trained jointly, each with cross-entropy plus a KL term toward the others ([distillation background](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals)). Qwen3-14B went from 0.925 / 0.921 alone to 0.929 / 0.925. The final cohort was Qwen3-14B, Qwen3-8B and Qwen3Guard-Gen-4B, three because two peers learned correlated errors. He rose from public 12th to 6th.

**7th, ktr (kcotton21, solo): start from a moderator.** +0.007 from ignoring subreddit, and local CV with no correlation to the leaderboard. His move was backbone choice: **ShieldGemma-9B was the best single model** (0.927 / 0.922), ahead of Qwen3-14B (0.926 / 0.919); Qwen3-8B-Guard beat vanilla Qwen3-8B by 0.003 public. Safety-tuned models already know what a policy violation looks like. He also patched vLLM 0.10.0 with sed so it would run Gemma2 in fp16.

**8th, Reddit comment moderators.** Three Qwen3-14B seeds, Llama2-13B and bge-base inside 12 hours, with a standard `AutoModelForSequenceClassification` head instead of a generated token (they could not say which was better). Every submission was one configuration over 4 seeds, so a leaderboard change could be told from noise. Length grouping doubled training speed, and length-bucketed inference freed time for one more 8B model.

### Family B: smaller LLMs plus encoders (9th, 11th, 13th, 15th)

**9th, lllleeeo + mango** blended three parts 0.7 / 0.25 / 0.05: generative TTT (Qwen3-4B-2507, Llama3.2-3B, Phi-3.5-mini, Qwen2.5-7B), encoders (DeBERTa-v3-base, E5, MPNet), and BGE triplet retrieval. Unsloth cut a run from 7.5 hours on 2 GPUs to 2.5 on 1.

**11th, takaito / Muj!rush! / yukko / miyaki** ran fully online: a gte-multilingual-base BERT, per-rule contrastive SentenceBERT (score = cosine of rule and comment), and Llama-3.2-3B plus Qwen3-4B with **TTA over different example combinations**. Hard pseudo-labels moved private from 0.92579 to 0.92734.

**13th, ISE@JNU-1: cross-rule negatives.** A comment that violates "no advertising" is almost always fine under "no spoilers", so they moved 30% of rows to a random other rule with label 0. Qwen3-8B GPTQ-int4 went from 0.922 to 0.927 public (0.923 private).

**15th, hasibirok0 / kfsky / Kinosuke** blended seven models, including 19,000 synthetic rows from gpt-oss-120b and two large Qwen3 models for DeBERTa pretraining (+0.003). Two contrarian choices: they **kept** subreddit, and asked Qwen3-4B whether the comment *complies*, reading P("No"), which lifted it from 0.918 to 0.920 for reasons they could not explain.

### Family C: change the data, not the model (12th, 17th, 18th)

**12th, losingself (solo): find the source.** Probing showed under 0.5% of test bodies are missing from a 2-million-comment unlabelled Reddit dataset he then published. He soft-labelled 800,000 rule-comment rows with the best public LLM ensemble and pretrained DeBERTa on them offline: DeBERTa-base went from 0.912 to about 0.924, and the final used DeBERTa-large in a submission under 6 hours. It is the only top-20 entry built around a distilled encoder.

**17th, Ebi (solo):** Qwen3-32B writes a label-preserving paraphrase of each training body with rule-specific instructions, inside the 12 hours, then QLoRA of phi-4, Qwen2.5-14B and Qwen3-14B. He fell from public 11th to 17th.

**18th, YF:** stage 1 retrieves the 3 most similar negatives per positive with Qwen3-Embedding-8B and asks a 4-option "which one violates the rule?" question; stage 2 continues to Yes/No. A leak-free local split (train bodies that never appear as examples) chose the recipe: one Qwen3-4B seed went from 0.6537 to 0.7448 local AUC with the pre-task (not comparable to leaderboard values). They rose 11 places, the most in the top 20.

### Family D: the contrasting entry, a diverse ensemble at 5th

**5th, Flash (ivanisaev, benbla, raki21, cnumber, kishanvavdara)** stacked *paradigms* rather than LLMs: TTT Llama-3.1-8B (LoRA r = 256, alpha 32) and Qwen3-8B; six embedders with per-rule logistic regression and kNN; a DeBERTa NLI model test-time-trained on biases, head and LayerNorms only; and an offline Qwen2.5-32B (GPTQ + LoRA) with **asymmetric example selection** (most similar positive, least similar negative), +0.006 over plain few-shot. Blending used per-rule rank normalisation. Their best single, Llama-3.1-8B, scored 0.92084 public and 0.92618 private, yet the blend slid from public 2nd to 5th.

### The five without writeups

Nikita Babych (10th), next gen VNG warriors (14th), PPD (16th), Fidel Escalona (19th) and Artem Silevich (20th) published no solution post that a forum scan or web search found. Their methods stay "?". bibanh of next gen VNG warriors wrote the public Qwen2.5-32B GPTQ notebook that 12th place used as its teacher; whether their own final built on it is not stated.

## 9. What the forum taught everyone

**Train on test.** neibyr's grey-area thread and 0.878 notebook defined the competition. Monnu's earlier "Qwen2.5 LoRA Finetune Baseline LB 0.872" used a 32B model trained offline; a 0.5B model trained on the test examples beat it. That gap is the TL;DR in two numbers.

**The hidden rules did not stay hidden.** c-number proposed encoding characters of the hidden rule text into the score, shlomoron organised a joint probing effort (one submission per character), and "The rules are revealed" posted all six: no advertising and no legal advice (the two seen in train), and no financial advice, no medical advice, no promotion of illegal activity, no spoilers. For hosts, a hidden prompt behind a scored leaderboard is not hidden.

**Is 0.931 a ceiling?** Chris Deotte's own entry finished 120th: retrieval of the most similar examples per row, 4x TTA, and Qwen2.5-32B reranking the 20% most uncertain rows. A sensible pipeline that did not train on the test examples deeply enough.

**Contradictions, with sources.**

- **Pseudo-labels:** helped 4th (~+0.004), 11th (+0.00155) and 12th; failed for 8th and 17th, and 9th's two-stage training did not work.
- **Per-rule versus global ranking:** 1st reports a consistent gain; 8th found global ranking better; 9th found rule-based weighting did not help; 7th argued per-rule ranking does not change the metric. For a single model, 7th is right, since AUC is rank-invariant within a rule; for a blend, ranking changes how much each model counts in each rule, and whether that helps depends on the models.
- **Subreddit:** +0.007 from dropping it for 7th; 15th kept it.
- **Synthetic data:** +0.003 for 15th's DeBERTa and in 17th's final; hurt 7th; helped 13th on one hidden rule and hurt the others.
- **Local CV:** 7th saw no correlation with the leaderboard; 2nd and 18th built local splits and relied on them.

## 10. What did not work

| Tried | Reported by |
|---|---|
| Pseudo-labelling | 8th, 17th (9th: two-stage training) |
| Synthetic training data | 7th; 13th (all but one hidden rule); 15th (for TTT) |
| Qwen3-32B trained on train data only | 7th |
| Models above 4B in a small-model stack | 11th; 15th (rejected for time) |
| Continued pretraining on about 38 GB of Reddit | 11th |
| Local CV that tracks the leaderboard | 7th; 8th |
| Semantic example matching, rule-specific prompts, label smoothing, layer freezing | 9th |
| Rule-based ensemble weighting, within-rule ranking | 9th; 8th |
| Weight optimisation, EMA, chain-of-thought prompts | 15th |
| Muon optimizer | 17th |
| ArcFace with per-rule classes | 3rd |
| bf16 on T4 (about 6x slower) | 12th |

Two patterns stand out. Chain-of-thought and longer prompts lost to a single answer token: the task needed a score, not reasoning. And almost every attempt to add *external* data failed, while every attempt to use the *test set's own* examples better worked. The one external-data success, 12th, found the exact corpus the test came from.

## 11. Lessons that transfer

1. **When the test set ships labelled examples, test-time training is the model.** A 0.5B model trained on them beat a 32B trained without, and 14 of 15 writeups built on it.
2. **Read a classifier from one token.** Mask the loss to the answer, read Yes-versus-No log-odds with `forward()` (1st, 4th). Faster to train, and a continuous score is what AUC needs.
3. **Delete shortcut features and vote out conflicting labels.** Subreddit removal (+0.007 for 7th, "significant" for 1st) and majority voting (2nd) cost nothing.
4. **Match the blend to the metric.** With per-rule AUC, test within-rule ranking against global ranking before averaging; 1st and 8th found opposite answers.
5. **Engineer for the box.** Two models in parallel, length-sorted inference, 4-bit weights, memory headroom. Fitting a 14B model was the line between the top 8 and the rest.
6. **Know your noise floor.** 4th measured 0.002 AUC of seed noise; 8th ran every config over 4 seeds.

### If you enter a competition like this

- Read the test file's columns before the train file's: what labelled signal ships with the test rows?
- Build the in-notebook training loop first, with a timer, and measure throughput per model size on the real hardware.
- Mask the loss to the answer, read logits, sort by length, keep memory headroom.
- Decide the deduplication key and the shortcut columns early.
- Decide whether the public leaderboard is valid validation (random split, large test) or build a leak-free local split.
- Run promising configs over several seeds before trusting a gain below the seed spread.
- Blend model families plus one non-LLM model.

Related: [LMSYS Chatbot Arena](/blog/machine-learning/learn-from-kaggle/kaggle-lmsys-chatbot-arena-solutions) (big models on the same two T4s), [MAP misconceptions](/blog/machine-learning/learn-from-kaggle/kaggle-map-student-math-misunderstandings-solutions) (noisy labels, a narrow leaderboard), and in this wave [Make Data Count](/blog/machine-learning/learn-from-kaggle/kaggle-make-data-count-solutions) and [Drawing with LLMs](/blog/machine-learning/learn-from-kaggle/kaggle-drawing-with-llms-solutions).

## Sources

Competition: https://www.kaggle.com/competitions/jigsaw-agile-community-rules (leaderboard from Kaggle's LeaderboardService API). Writeups, all under https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/:

- 1st: [1st-place-solution](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/1st-place-solution); code https://www.kaggle.com/code/wowfattie/notebookc4e4a9206d
- 2nd: [2nd-place-solution](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/2nd-place-solution)
- 3rd: [3rd-place-solution](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/3rd-place-solution); notebook https://www.kaggle.com/code/socom20/t73t74-ens-7b-9b-unsloth-qwen14bx4-qwen8bx1-qwen7
- 4th: [4th-place-solution](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/4th-place-solution); paper https://arxiv.org/abs/2510.22691
- 5th: [5th-place-solution-diverse-ensemble](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/5th-place-solution-diverse-ensemble)
- 6th: [6th-place-solution](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/6th-place-solution)
- 7th: [7th-place-solution](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/7th-place-solution)
- 8th: [8th-place-solution](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/8th-place-solution-qwen3-14b3-llama2-13b1-b)
- 9th: [9th-place-solution](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/9th-place-solution)
- 11th: [11th-place-solution](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/11th-place-solution)
- 12th: [12th-place-solution](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/12th-place-solution); code https://github.com/level14taken/jigsaw-comp
- 13th: [13th-place-solution](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/13th-place-solution-using-positive-examples-from)
- 15th: [15th-place-solution](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/15th-place-solution)
- 17th (titled 18th): [18th-solutionprivate-11th](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/18th-solutionprivate-11th)
- 18th: [18th-place-solution](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/18th-place-solution)
- No writeup found: 10th, 14th, 16th, 19th, 20th.

Forum threads (discussion IDs under https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/): neibyr train-on-test 598099; c-number probing 604092; shlomoron joint probing 607118; "The rules are revealed" 607941; monnu baseline 591445; Chris Deotte ceiling 610635 and 120th place 613168; Psi labelling 613100.

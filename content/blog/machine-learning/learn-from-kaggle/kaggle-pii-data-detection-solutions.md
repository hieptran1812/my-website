---
title: "PII Data Detection: how the top 20 teams used LLM-written essays to train a recall-hungry DeBERTa"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "A synthesis of the top 20 finishers in Kaggle's PII Data Detection competition: why LLM-generated essays with injected PII decided it, how an F5 metric turned every model into a recall machine, and which post-processing rules were worth a model."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "pii-detection", "named-entity-recognition", "token-classification", "deberta", "synthetic-data", "f-beta", "post-processing"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 23
---

> **TL;DR.** In The Learning Agency Lab's PII Data Detection competition (2024), 2,048 teams had to find seven kinds of personal information, token by token, in student essays. Every one of the 13 top-20 teams that published a writeup used DeBERTa-v3-large, and 12 of 13 trained on the same free dataset of Mixtral-written essays. The winners, Fold Zero (private F5 0.96988), stacked ten DeBERTa variants under a rule layer they called "the key". The decisive lever was **data**: an LLM used as a factory for labelled training essays, because the real training set had only 945 essays with any PII. The one lesson: when labels are scarce and you can write the labels yourself, a stronger generator beats a bigger classifier.

![The lever strip with Data lit: PII Data Detection was decided by LLM-written training essays, with the F5 metric shaping every threshold](/imgs/blogs/kaggle-pii-data-detection-solutions-1.webp)

The strip above is the frame for every post in this series, with the decisive lever lit. PII Data Detection is the odd one out in a series about LLM competitions: no winning team submitted an LLM. The LLMs lived upstream, writing thousands of fake student essays with fake names, emails and addresses planted in known positions, which made the labels free. The second force was the metric: F-beta with beta 5 makes a missed token cost as much as 25 false alarms, so every serious team bent its model toward recall.

This post synthesises the writeups of the top 20 private finishers: 13 published a full writeup, one posted a single trick, and six posted nothing, named as such below. If you are new to the series, start with [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work).

## 1. The task in 60 seconds

Vanderbilt University and The Learning Agency Lab, the same hosts as [LLM Detect AI Generated Text](/blog/machine-learning/learn-from-kaggle/kaggle-detect-ai-generated-text-solutions), named the goal in the competition's slug: detect personal information so it can be removed from educational data. The essays came from a massive open online course, all answering a single assignment prompt; the writeups show them discussing design-thinking tools such as Mind Mapping, Storytelling and Visualization. The competition ran from 2024-01-17 to 2024-04-23 and drew 2,048 teams, 2,510 competitors and 55,847 submissions.

![Notebook budget card and an illustrative essay fragment tagged token by token: a student name in BIO tags, an instructor name left as O, an email, and a street address with a newline inside it](/imgs/blogs/kaggle-pii-data-detection-solutions-2.webp)

Input: an essay already split into spaCy tokens. Output: a `(document, token, label)` triple for every token that is personal information, of seven types: `NAME_STUDENT`, `EMAIL`, `USERNAME`, `ID_NUM`, `PHONE_NUM`, `URL_PERSONAL` and `STREET_ADDRESS`, in BIO format (a `B-` tag starts a span, `I-` continues it), so 15 labels counting `O`, the "not PII" label you must never submit. One subtlety decided many points: `NAME_STUDENT` means students only; an instructor's name, or a cited author's, is `O`. The original PII had been replaced by surrogates of the same type.

About 22,000 essays existed, and 70% were held back for the hidden test. The training set had 6,807 essays, and only 945 of them, about 14%, contained any PII at all. The public leaderboard scored 25% of the test; the private 75% paid out. Notebooks ran on CPU or GPU for at most 9 hours with internet off.

Prize money was USD 60,000: USD 13k, 10k and 5k for the leaderboard top three, and USD 15k, 12k and 5k for a CPU-only **efficiency track** scored with the same speed-versus-accuracy formula as DAIGT. The efficiency winners were not checked for this post.

## 2. The metric, worked on a toy

The score is a micro-averaged F-beta over all positive predictions, with beta set to 5:

$$
F_\beta = \frac{(1+\beta^2)\,P\,R}{\beta^2 P + R}, \qquad P = \frac{TP}{TP+FP}, \quad R = \frac{TP}{TP+FN}
$$

A true positive (TP) is a predicted triple that is in the answer key, a false positive (FP) one that is not, a false negative (FN) a key triple you missed. With beta = 5 the weights show directly:

$$
F_5 = \frac{26\,TP}{26\,TP + 25\,FN + FP}
$$

A miss enters the denominator 25 times; a false alarm enters it once. The figure below works three models on an essay set with 10 true PII tokens.

![F5 on a toy with 10 true PII tokens: a balanced model scores 0.800, a trigger-happy model with 10 false alarms scores 0.963, and a cautious model with perfect precision scores 0.609](/imgs/blogs/kaggle-pii-data-detection-solutions-3.webp)

The balanced model finds 8 and raises 2 false alarms: F5 is 208 / 260 = 0.800. The trigger-happy model finds all 10 but raises 10 false alarms, so its precision is a coin flip, yet F5 is 260 / 270 = 0.963. The cautious model is never wrong and finds only 6: F5 is 156 / 256 = 0.609. The model that doubles its output to catch the last two names wins by a mile, which is why nearly every top team treated the "O" class as a nuisance.

Two metric events mattered. On 2024-03-12 the hosts fixed a hole: duplicate predictions for the same document-token pair, which could game the metric, are now all dropped before scoring ([483538](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/483538)). And Damien Mourot showed by probing that the public test contained no `PHONE_NUM` at all; only 4 of the 6,807 training essays had one ([480067](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/480067)). A whole class was invisible to the leaderboard everyone watched.

## 3. What made it hard

Three constraints decided the competition, and they compound.

**Scarce, lopsided labels.** About 86% of training essays carry no positive label, and inside the PII essays almost every token is `O`. `NAME_STUDENT` dominates the positives; `PHONE_NUM`, `URL_PERSONAL` and `STREET_ADDRESS` appear a handful of times. A classifier trained on this learns that "O" is nearly always right, exactly the instinct F5 punishes. Our post on [class imbalance and when accuracy lies](/blog/machine-learning/debugging-training/class-imbalance-and-when-accuracy-lies) covers the general pattern; here the cure was not reweighting alone but manufacturing more positives.

**Two tokenizations that disagree.** The key lives on spaCy word tokens; DeBERTa sees sub-word pieces. Every team had to map one to the other, and the map has a hole.

![The alignment trap: the newline inside a street address is a spaCy token with a label, but the DeBERTa tokenizer drops whitespace, so a naive mapping splits the address in two; fixes replace the whitespace with a placeholder or fill it by rule](/imgs/blogs/kaggle-pii-data-detection-solutions-4.webp)

A two-line street address contains a `\n` token labelled `I-STREET_ADDRESS`. DeBERTa's tokenizer discards whitespace, so the `\n` gets no prediction and falls back to `O`: a guaranteed miss. Joseph Zhou of 11th place traced this to the public notebooks everyone forked: they skipped whitespace tokens when mapping. Replacing whitespace tokens with a character absent from the data (he used a full-width yen sign), or filling `\n` between two address tokens by rule, was worth +0.003 on its own ([497184](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497184)). Eight of the 13 full writeups mention a fix. Nicholas Broad, 9th: "I can't tell you how much time I spent debugging alignment errors."

**Long essays and a short context.** Many essays exceed 512 tokens. Raja Biswas's thread on truncation ([473011](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/473011)) pushed teams toward long maximum lengths or sliding windows with stride, and the private leaderboard rewarded the long-context side, as 5th place's member scores show below.

## 4. Top-20 scorecard

Ranks and scores are from the Kaggle leaderboard API. "Public → private" is the move from public rank to private rank. Gold went to the top 14. Ranks 7 and 8 tied exactly at 0.96796.

| Rank | Team | Private F5 | Public → private | Key idea | Source |
|---|---|---|---|---|---|
| 1 | Fold Zero (5 members) | 0.96988 | 14 → 1 | 10 DeBERTa variants, Optuna-weighted vote, heavy rules | [497374](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374) |
| 2 | dott (solo) | 0.96970 | 7 → 2 | 6 DeBERTa, 7 classes without BIO, O probability x0.02 to 0.03 | [497352](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497352) |
| 3 | Rai (solo) | 0.96950 | 28 → 3 | One backbone, pre-trained on mpware's essays, then fine-tuned | [497482](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497482) |
| 4 | Luca Massaron and team (5) | 0.96929 | 17 → 4 | Llama-3-70B-written essays with Faker PII; focal loss | [497367](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497367) |
| 5 | R & T & M (Ryota, takai380, min fuka) | 0.96891 | 1 → 5 | Each member a different max length; member voting | [497306](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497306) |
| 6 | Yusuke Nakayama | 0.96808 | 13 → 6 | Fine-tuned Mistral generator; loss on scored tokens | [498304](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/498304) |
| 7 | Mike & Wasjaip | 0.96796 | 21 → 7 | 14 public checkpoints, runtime split by length | [497310](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497310) |
| 8 | Amed | 0.96796 | 84 → 8 | One model; external sets chosen by per-class error | [497323](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497323) |
| 9 | Nicholas Broad | 0.96755 | 159 → 9 | Author of the Mixtral set; ~30 labels hand-fixed | [497177](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497177) |
| 10 | DarioAr | 0.96705 | 80 → 10 | unknown | no writeup |
| 11 | I \| C \| S \| J | 0.96703 | 25 → 11 | Whitespace placeholder for `\n` in addresses | [497184](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497184) (trick only) |
| 12 | GoGo | 0.96670 | 68 → 12 | Own Qwen/Mistral essays; no post-processing | [498130](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/498130) |
| 13 | Yuki Okumura | 0.96643 | 226 → 13 | DeBERTa + RoBERTa at three lengths, focal loss | [497371](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497371) |
| 14 | M & Y & K | 0.96615 | 5 → 14 | 13-class and 8-class models mixed | [497968](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497968) |
| 15 | Team Espresso | 0.96609 | 18 → 15 | unknown | no writeup |
| 16 | Falcon&takaito | 0.96533 | 46 → 16 | unknown | no writeup |
| 17 | bird | 0.96525 | 6 → 17 | unknown | no writeup |
| 18 | Dileep Jayamal | 0.96523 | 42 → 18 | Drop 30% of O labels from the loss | [497266](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497266) |
| 19 | Gleb Anferov | 0.96514 | 154 → 19 | unknown | no writeup |
| 20 | JungleBeastDS | 0.96506 | 53 → 20 | unknown | no writeup |

The 4th and 9th team names on the board are emoji, so members are named instead. Dileep Jayamal's post says "19th" but the API places him 18th, and a "20th place" post by jarvisai7 ([497405](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497405)) matches no team on the final board (likely a team removed later; not confirmed). The spread is the story: 0.00482 from 1st to 20th, 0.00018 from 1st to 2nd. A handful of tokens reorders the gold zone.

## 5. The top-20 landscape

The matrix records which of 13 techniques each team's writeup reports: filled is used, empty is not used, a question mark is not stated. Counts are over the 13 full writeups (ranks 1 to 9, 12, 13, 14 and 18), recounted from the cells.

![Technique matrix for the top 20: 13 full writeups, DeBERTa-v3-large in all 13, nbroad's Mixtral essays in 12, a recall bias on the O class in 11, and question marks for the six teams without a writeup](/imgs/blogs/kaggle-pii-data-detection-solutions-5.webp)

**Table stakes.** Everyone who wrote a solution used DeBERTa-v3-large token classification (13 of 13). Twelve of 13 trained on Nicholas Broad's Mixtral-generated essays (7th trained nothing). Eleven of 13 biased the decision toward recall with an O weight or threshold; the other two do not say. Nine of 13 ensembled six or more models, and 8 of 13 repaired the newline inside addresses.

**What separated the top 5 from 6 to 20.** No single technique, but the top-5 writeups share three things that ranks 6 to 20 mostly have one or two of:

- **Stronger or staged synthetic data.** 1st generated about 2,000 essays of its own on top of the public sets; 4th generated about 4,600 with Llama-3-70B; 3rd and 5th pre-trained on external essays before fine-tuning on the competition's.
- **Long inference contexts.** 4th inferred at 4,000 tokens with stride 1,024; 5th's longest-context member scored the team's best private score.
- **Large, systematically tuned post-processing** built from out-of-fold (OOF) error analysis: 1st, 2nd and 3rd all describe a rule layer they iterated on.

**Contradictions.** The writeups disagree with each other on several points:

| Question | One side | Other side |
|---|---|---|
| More external data? | 1st used several sets; 13th merged mpware and nbroad without downsampling | 3rd: "more external data" did not help; 8th dropped two sets after error analysis; jarvisai7 capped external rows under 4,000 |
| mpware's essays | Useful as pre-training (3rd); best single model when used alone (jarvisai7, 0.967) | Harmful directly for some classes (3rd); labels disagree on ID_NUM and USERNAME (8th) |
| Post-processing | "The key" (1st); central to 2nd, 3rd, 14th | 12th used none; jarvisai7 found regex rules stopped helping as models improved |
| Faker PII injection | Core of 4th's and 6th's data recipes | 8th: replacing or inserting Faker PII did not help |
| Context length | Long won private for 5th | 9th used 512 throughout and still climbed 159 → 9 |
| BIO labels | 1st kept 13 labels | 2nd, 5th, 6th, 8th, 14th collapsed to 7 or 8 classes and rebuilt BIO by rule |

"External data" was never one thing: each public set had its own labelling conventions, and whether it helped depended on whether its idea of an ID number matched the hosts'.

## 6. The winning solution, explained

Fold Zero were Patrick Chan, Gaurav Rawat, YoungPaik, Geremie Yeo and Yevhenii Maslov, writeup by Geremie Yeo ([497374](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374)), with code at [bogoconic1/pii-detection-1st-place](https://github.com/bogoconic1/pii-detection-1st-place). They were 14th on the public board.

![First place pipeline: competition essays plus three generated sets feed 4-fold training with O weight 0.05, ten DeBERTa-v3-large variants in seven groups, an Optuna-tuned weighted vote, and a rule layer, to private F5 0.96988](/imgs/blogs/kaggle-pii-data-detection-solutions-6.webp)

### The intuition: the encoder is fine, the inputs and outputs are not

A DeBERTa-v3-large trained on the competition data already spots names well. Its failures are at the edges: too few addresses and phone numbers to learn them, a calibration where "O" is almost always right, and mistakes a human catches at a glance (a lowercase "name", a three-character ID, an email without `@`). So the winners worked on both ends of the model: more PII in more forms going in, and a rule layer coming out.

### The mechanism: data in

Training data was the competition's 6,807 essays plus Broad's Mixtral essays ("by far the most useful"), mpware's Mixtral-8x7B essays and about 2,000 of their own, with no downsampling of PII-free essays. Every such set works the same way: put the PII (typically from the Faker library) in the LLM's prompt yourself, and because you chose the values, the labels come for free. Below is an illustrative toy, not any team's code, with a template standing in for the LLM and a name list standing in for Faker.

```python
# Illustrative toy, not any team's code: labels for free, then a recall-biased decision.
import random

random.seed(0)
FIRST = ["Lindsey", "Omar", "Priya", "Tomas"]
LAST = ["Ortiz", "Haddad", "Rao", "Novak"]

def make_essay():
    """Insert sampled PII into a template; the labels come from construction."""
    first, last = random.choice(FIRST), random.choice(LAST)
    email = f"{first.lower()}.{last.lower()}@example.com"
    words = ["My", "name", "is", first, last, ".", "Dr.", "Kim", "asked", "us",
             "to", "try", "Mind", "Mapping", ".", "Write", "to", email, "."]
    labels = ["O"] * len(words)
    labels[3], labels[4] = "B-NAME_STUDENT", "I-NAME_STUDENT"
    labels[17] = "B-EMAIL"          # "Kim" stays O: an instructor, not a student
    return words, labels

def decide(probs, o_scale):
    """probs: {label: p} for one token. Scale O, then take the argmax."""
    scaled = {k: (v * o_scale if k == "O" else v) for k, v in probs.items()}
    return max(scaled, key=scaled.get)

def f_beta(tp, fp, fn, beta=5.0):
    b2 = beta * beta
    return (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp)

words, labels = make_essay()
print([(w, l) for w, l in zip(words, labels) if l != "O"])

tokens = {
    "Lindsey": {"O": 0.93, "NAME_STUDENT": 0.06, "EMAIL": 0.01},
    "the":     {"O": 0.995, "NAME_STUDENT": 0.004, "EMAIL": 0.001},
}
for tok, p in tokens.items():
    print(tok, decide(p, 1.0), "->", decide(p, 0.025))

for name, tp, fp, fn in [("balanced", 8, 2, 2), ("trigger-happy", 10, 10, 0), ("cautious", 6, 0, 4)]:
    print(f"{name:14s} F5={f_beta(tp, fp, fn):.3f}")

# [('Tomas', 'B-NAME_STUDENT'), ('Novak', 'I-NAME_STUDENT'), ('tomas.novak@example.com', 'B-EMAIL')]
# Lindsey O -> NAME_STUDENT
# the O -> O
# balanced       F5=0.800
# trigger-happy  F5=0.963
# cautious       F5=0.609
```

Real pipelines must also find each planted value in the LLM's output, which may paraphrase it; how carefully that was done is one reason 8th place found some public sets disagreeing with the hosts' labels. The [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation) post covers the general recipe and its traps.

### The mechanism: the models

Training: 4 folds by `document % 4`, then one or two full fits; 13 labels; length 1,600 to 2,048; 3 to 4 epochs at learning rate 1e-5; and an O-class loss weight of 0.05, a twentieth of a PII token's. Diversity came from four variants:

- **Multi-sample dropout**, for smoother training.
- **A BiLSTM layer before the classifier** (NaN losses fixed with careful initialisation and `save_safetensors=False`).
- **Knowledge distillation** from teachers trained on other datasets with the same folds, worth +0.005 to 0.01 on CV and leaderboard (see [knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals) for the mechanism).
- **First- and last-name swap augmentation**, so the model could not memorise the training names.

Ten models in seven groups were combined by a **weighted vote**, weights and threshold tuned by Optuna on out-of-fold predictions.

### The mechanism: the rule layer

The writeup calls post-processing "the key":

| Rule | What it catches |
|---|---|
| Per-label thresholds | Each class gets its own recall-precision trade |
| Drop `NAME_STUDENT` not title-cased, or with a digit or underscore | "the", "Mind", usernames mistaken for names |
| Propagate a predicted name to every occurrence in the essay | The model found "Lindsey" once and missed it twice |
| `PHONE_NUM` with 9 or more digits becomes `ID_NUM` | Student numbers read as phones |
| Repair `\n` inside `STREET_ADDRESS` | The alignment hole from section 3 |
| B-USERNAME, "-", B-USERNAME becomes one I-USERNAME span | Hyphenated usernames split in two |
| Drop `ID_NUM` under 4 or over 25 characters, `URL_PERSONAL` under 10, `EMAIL` without `@` | Short-fragment false positives |
| Span repair, then a regex fallback | Structured types the model missed outright |

Name propagation is the F5 logic in miniature: when a miss costs 25, copying a confident prediction to its other occurrences is almost free money.

## 7. Second and third in depth

### 2nd: dott, collapse the labels and scale the O

dott ([497352](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497352)), solo and 7th on public, made three choices. He tokenised the word list with `is_split_into_words=True` and averaged sub-token probabilities back to each word, ignoring whitespace tokens. He **dropped the B- and I- prefixes**: spans of the same type are always separated in this data, so BIO is rebuilt exactly by rule. And he multiplied the O probability by a factor tuned on public, consistently 0.02 to 0.03, before the argmax.

![Scaling the O probability by 0.025: a name token with P(O) 0.93 and P(NAME) 0.06 flips to NAME, while a function word with P(O) 0.995 and P(NAME) 0.004 stays O](/imgs/blogs/kaggle-pii-data-detection-solutions-7.webp)

At 0.025, a PII class wins whenever its probability exceeds 1/40 of O's: a 6% name becomes a name, a 0.4% one does not. That is the toy's `decide` function, the F5 metric turned into a decision.

Training: a bag of 6 deberta-v3-large models on full data, maximum lengths 512, 1,024 and 2,048 with stride 32, batch 8, learning rate 2e-5, 4 epochs with cosine schedule and one warm-up epoch, AdamW. Nicholas Broad's set was mixed in at sample weight 0.5. Post-processing propagated names within an essay, relabelled 1-character and non-title-case strings, and handled `\n` inside addresses.

### 3rd: Rai, one backbone and a two-stage diet

Rai's own writeup ([497482](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497482), code at [srai9/pii-data-detection-3rd-place-solution](https://github.com/srai9/pii-data-detection-3rd-place-solution)) argues that the recipe mattered more than the ensemble. One backbone, deberta-v3-large, trained in two stages: **pre-train on mpware's Mixtral essays**, then fine-tune on the competition data plus Nicholas Broad's set. Training on mpware's data directly hurt some classes; using it as a pre-training stage raised both CV and leaderboard. He then retrained on full data with several seeds.

He added `\n\n`, `\t\r` and `\n` as tokens, trained 2 epochs at learning rate 1e-5, and used stratified 5-fold CV on PII present or absent, well correlated with public. Because `PHONE_NUM` barely existed in train and not at all on public, he **generated his own Mistral essays and pulled in external sets purely to evaluate rare classes**, not to train on them. Synthetic data as a test set for classes your validation cannot see is a quietly important idea.

Post-processing came from OOF errors: an O threshold, then false-positive rules for very short predictions, common instructor names and titles (Mr, Mrs, Dr), and a fix for names labelled both B- and I- in the same essay. He moved from 28th on public to 3rd.

## 8. Ranks 4 to 20, by approach family

### Family A: a better generator (4, 6, 12)

**4th, Rashmi B, Luca Massaron, steubk, AK and Priyanshu Kumar** titled their writeup "Llama3 70B is all you need" ([497367](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497367)). Personas from Mistral-7B, Gemini Pro 1.0 and GPT-3.5-turbo, a sample design-thinking tool and Faker PII went into prompts; Llama-3-70B-Instruct and Mixtral-8x7B-Instruct, served with vLLM, wrote about 4,600 essays. Their sentence: "A single model trained with dataset generated from Llama3 70B Instruct model is better than our best ensemble." Models: 2 x 5-fold DeBERTa-v3-large, focal loss, BiLSTM plus GRU head, trained at length 1,280 and inferred at 4,000 with stride 1,024.

**6th, Yusuke Nakayama** ([498304](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/498304)) was the only top-20 team to **fine-tune the generator**. He fine-tuned Mistral-7B-Instruct-v0.2 to map (tool name, the essay's PII, the hosts' instruction) to the essay, then sampled tools and Faker PII to write 1,722 essays. His head maps DeBERTa probabilities to characters, then to spaCy tokens with a mean scatter, so the loss is computed on the scored tokens. Maximum length: 4,600 training, 6,300 inference. His blend scored 0.9765 CV, 0.9770 public and 0.9681 private, and he noted "not much correlation between CV and LB".

**12th, GoGo** ([498130](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/498130)) asked Qwen-7B, Mistral-7B and Qwen-14B to continue the opening paragraphs of real essays (names masked), weaving in PII sampled from a vocabulary built from the competition's own values. Single-model scores: Broad's mix 0.962, **Mistral-7B with 8k essays 0.966**, Qwen-14B with 8k 0.964. A weighted loss (PII 10, O 1) made the model insensitive to thresholds, and the team used **no pre- or post-processing at all**. An unselected 5-model notebook would have scored 0.96816, above 6th place's 0.96808.

### Family B: diversity by context length and label scheme (5, 13, 14)

**5th, R & T & M** ([497306](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497306)) were 1st on public. Each member owned a context length; members then voted:

| Member | Train / infer length | CV | Public | Private |
|---|---|---|---|---|
| Ryota | 128 / 128 | 0.979 | 0.973 | 0.960 |
| takai380 | 512 / 512 | 0.977 | 0.975 | 0.965 |
| min fuka | 1,536 / 4,096 | not given | 0.972 | **0.967** |

Short contexts won CV and public; the longest won private, and min fuka's part alone was gold level. takai380 dropped BIO tags because "these tags are rule-based".

**13th, Yuki Okumura** ([497371](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497371), code at [okumura2997/kaggle-pii-solution](https://github.com/okumura2997/kaggle-pii-solution)) made the biggest climb in the top 20, 226 to 13. He ensembled backbones, lengths (512, 1,024, 2,048) and external sets, choosing settings stable across all four folds rather than best on any one.

**14th, M & Y & K** ([497968](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497968)) mixed 13-class and 8-class models (DeBERTa-v3-large, DeBERTa-xlarge, Longformer), labelling each member "high recall" or "high precision"; dynamic padding halved inference time. They fell from 5th on public.

### Family C: one strong model and chosen data (8, 9, 18)

**8th, Amed** ([497323](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497323)) shipped one deberta-v3-large in 5 folds trained on competition data plus Broad's set only: 0.968 public, 0.967 private. His reusable idea: run your model on each external set and read the per-class F5. Broad's set scored 0.998 (easy, consistent), mpware's 0.883 (USERNAME 0.48, ID_NUM 0.40), tonyarobertson's 0.923 (ID_NUM 0.41). Low per-class scores there mean the set's labels disagree with the hosts', so he dropped those sets.

**9th, Nicholas Broad** ([497177](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497177), code at [nbroad1881/pii-data-detection](https://github.com/nbroad1881/pii-data-detection)) climbed from 159th. He improved his Mixtral set with famous-person quotes (non-PII names) and team-mate phrases ("With my teammates Joe and Sally" is PII), **fixed about 30 training labels by hand**, and trained deberta-v3-large variants plus deberta-v2-xlarge with LoRA (see [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft)), all at length 512.

**18th, Dileep Jayamal** ([497266](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497266)) trained deberta-v3-large with 6 frozen layers, each of 5 folds mixed with a different external set, and a custom loss that **drops 30% of O labels** and weights O at 0.01.

### Family D: train nothing (7)

**7th, Mike & Wasjaip** ([497310](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497310)) used **14 publicly shared DeBERTa checkpoints**. Their idea was runtime allocation: the best models predict every essay in the first ~3 hours; the rest predict only the shortest two-thirds in the next ~5.5 hours, because PII count does not grow with length but runtime does. Per-label thresholds and pattern rules finished it.

### One trick (11) and no writeup (10, 15, 16, 17, 19, 20)

11th place posted only the whitespace trick from section 3. Six teams posted nothing that could be found: DarioAr (10th), Team Espresso (15th), Falcon&takaito (16th), bird (17th), Gleb Anferov (19th) and JungleBeastDS (20th). Their methods are unknown. Team Espresso included Raja Biswas, author of the "Impact of External Datasets" thread, who with Nicholas Broad had been on DAIGT's winning team.

Outside the board, jarvisai7's "20th place" post reports a best single model of 0.967 trained on mpware's essays only and external data kept under 4,000 rows; treat it as a practitioner's report, not a ranked solution.

### The best contrasting entries

**7th trained nothing** and won gold by spending its 9 hours on other people's models. **12th used no post-processing** and argued that forced a robust model. Against Fold Zero's machinery, the margin bought was 0.00192 (1st versus 7th).

## 9. What the forum taught everyone

Raja Biswas's "Impact of External Datasets" ([473139](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/473139), 169 votes) showed that each shared synthetic set (pjmathematician's 2k, moth's 4.4k, Broad's 2.3k) gave a strong boost and predicted that external data would decide the outcome. Broad's "2355 essays generated by Mixtral 8x7b" ([472221](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/472221), 134 votes) lifted his public score from 0.854 to 0.888 and became the dataset 12 of 13 writeup teams trained on. moth's "+4400 external generated texts" ([469493](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/469493)) gave the 945-of-6,807 statistic. For the privacy side of the same problem in a production data pipeline, see [PII, safety and toxicity filtering](/blog/machine-learning/training-data/pii-safety-and-toxicity-filtering).

Then the board shook.

![Public rank on a log scale against private rank for the top 20: the public leader finished 5th, while teams from public ranks 84, 159 and 226 climbed into the top 13](/imgs/blogs/kaggle-pii-data-detection-solutions-8.webp)

Climbers mostly trusted cross-validation or ran simple, robust models: Okumura 226 → 13, Broad 159 → 9, Amed 84 → 8, Rai 28 → 3. Public leaders fell: R & T & M 1 → 5, M & Y & K 5 → 14, bird 6 → 17, and public 2nd to 26th. The public slice had no `PHONE_NUM` and few rare classes, so it rewarded tuning for names; and long contexts generalised better to private. With neighbours 0.0002 apart, some of the reshuffle is plain noise, and the 25th-place team named itself "THE SHAKING AGENCY LAB".

## 10. What did not work

Aggregated from every top-20 writeup that reported a negative result:

| Tried | Result | Reported by |
|---|---|---|
| Relabelling the training set | No gain | 3rd |
| More external data | No gain | 3rd; jarvisai7 capped it under 4,000 rows |
| Second-stage false-positive model | No gain (3rd); +0.0005 CV only, dropped (5th) | 3rd, 5th |
| AWP, label smoothing, GRU head, layer re-initialisation | No gain | 5th |
| Replacing or inserting Faker PII (100%, 50%, 10%) | No gain | 8th |
| mpware or tonyarobertson data used directly | Hurt some classes | 8th, 3rd |
| `document % 4` folds | High fold variance | 4th (13th used them anyway, picking settings stable across folds) |
| Augmentations in the final submission | No gain | 2nd |
| Softmax averaging, max voting; regex rules once models were strong | Worse than logit mean; no gain | jarvisai7 |

Read the Faker row next to 4th and 6th, whose data recipes depend on Faker. One plausible reading, ours rather than any writeup's, is that the difference is where the PII goes: 4th and 6th had the LLM write an essay around the planted values, while 8th swapped values into existing essays. An LLM that writes around a name produces the surrounding phrasing ("my teammate", "contact me at") a model needs; a swapped string does not.

## 11. Lessons that transfer

1. **An LLM is a data factory for a small model.** Planted PII gives free labels, and a stronger generator beat a bigger ensemble (4th, Llama-3-70B).
2. **Translate the metric into the decision rule.** Under F5, shrink the background class: O weight 0.05 (1st), O probability x0.02 to 0.03 (2nd), O weight 0.01 (18th).
3. **Post-processing from OOF error analysis is worth a model.** 1st called its rule layer "the key".
4. **Check the plumbing between tokenizers.** The `\n`-in-address bug sat in the public notebooks everyone forked and was worth +0.003 (11th).
5. **Audit external data per class** before using it (8th).
6. **Probe what the public board does not contain.** A missing class and short-context tuning turned public leaders into 5th, 14th and 17th.

### If you enter a competition like this

- Compute the cost of a miss versus a false alarm from the metric first, and build it into the loss or decision.
- Count positives per class; generate rare classes, and an evaluation set for them (3rd).
- Write and unit-test the token mapping on whitespace, newlines and sub-word splits before modelling.
- Keep one long-context model in the ensemble even if short contexts win CV.
- Select final submissions by CV stability; 12th's unsubmitted notebook would have finished 6th.

Related posts in this series: [Eedi: mining misconceptions](/blog/machine-learning/learn-from-kaggle/kaggle-eedi-mining-misconceptions-solutions) and [MAP: student math misunderstandings](/blog/machine-learning/learn-from-kaggle/kaggle-map-student-math-misunderstandings-solutions) cover two more education-data competitions where synthetic data mattered, and [LMSYS Chatbot Arena](/blog/machine-learning/learn-from-kaggle/kaggle-lmsys-chatbot-arena-solutions) shows the opposite regime, where fitting a bigger model into the notebook was the lever.

## Sources

- Competition page: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data
- 1st: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374; code https://github.com/bogoconic1/pii-detection-1st-place
- 2nd: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497352
- 3rd: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497482; code https://github.com/srai9/pii-data-detection-3rd-place-solution
- 4th: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497367
- 5th: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497306
- 6th: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/498304
- 7th: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497310
- 8th: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497323
- 9th: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497177; code https://github.com/nbroad1881/pii-data-detection
- 11th (trick only): https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497184
- 12th: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/498130
- 13th: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497371; code https://github.com/okumura2997/kaggle-pii-solution
- 14th: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497968
- 18th (titled "19th"): https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497266
- "20th" post by jarvisai7, not on the final board: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497405
- No writeup found: 10th DarioAr, 15th Team Espresso, 16th Falcon&takaito, 17th bird, 19th Gleb Anferov, 20th JungleBeastDS
- Forum threads: external datasets https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/473139; Mixtral essays https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/472221; 4,400 generated texts https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/469493; truncation https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/473011; no PHONE_NUM on public https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/480067; metric fix https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/483538

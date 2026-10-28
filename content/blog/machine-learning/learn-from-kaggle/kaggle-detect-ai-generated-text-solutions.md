---
title: "LLM Detect AI Generated Text: how the top 20 teams won by building the right data"
date: "2026-10-28"
publishDate: "2026-10-28"
description: "A synthesis of the top 20 DAIGT finishers on Kaggle: why a diverse synthetic corpus beat every clever model, how typo noise flipped from signal to attack, and why the public leaderboard leaders collapsed in the shake-up."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "ai-text-detection", "synthetic-data", "deberta", "tf-idf", "data-augmentation", "domain-adaptation", "roc-auc"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 23
---

> **TL;DR.** In LLM Detect AI Generated Text (DAIGT, 2023 to 2024), 4,358 teams had to tell student essays from LLM essays with almost no generated training data. The winners (Udbhav Bamba, Raja Biswas and Nicholas Broad, private AUC 0.987824) said it plainly: "the modeling approach was less important", because a 160k-row training mix built for diversity made several single models score above 0.970. The decisive lever was **data**. The one lesson: when the hidden test comes from generators you cannot see, the winning move is to cover the space of generators, decoding settings and noise, not to fit the leaderboard you can see.

![The lever strip with Data lit: DAIGT was decided by a training set that matched the hidden essays](/imgs/blogs/kaggle-detect-ai-generated-text-solutions-1.webp)

The strip above is the frame for every post in this series: five levers (data, retrieval, model fit, test-time compute, metric insight), with the decisive one lit. DAIGT is the cleanest example of the first. The winners' best single model was a 7B LLM, but 2nd place used only DeBERTa models, and 6th and 8th used no fine-tuned neural network at all. What they shared was a training set that looked more like the hidden test than anyone else's.

It also had one of the biggest shake-ups in Kaggle NLP. The public leader finished 9th, and a forensics researcher who spent four or five days on the problem jumped from public rank 2,995 to private rank 6. Understanding why is most of what this competition teaches.

This post synthesises the writeups of the top 20 private finishers: 14 of them published one, six did not, and the six are named as such below. If you are new to the series, start with [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work).

## 1. The task in 60 seconds

Vanderbilt University and The Learning Agency Lab asked: was this middle or high school essay written by a student or by an LLM? You output a probability that each essay is generated. From 2023-10-31 to 2024-01-22 it drew 4,358 teams, 5,264 competitors and 110,052 submissions.

![Notebook budget card and an illustrative obfuscated essay: two human-only train prompts, a hidden multi-prompt test with injected character noise](/imgs/blogs/kaggle-detect-ai-generated-text-solutions-2.webp)

The data is the trap. Roughly 10,000 essays were written in response to 7 prompts. The training set covered only 2 of them, and it was "nearly all" human: a handful of generated examples and a line telling you that "you may wish to generate more essays". The hidden test of about 9,000 essays came from the other prompts and from "multiple, unknown generative models". The public leaderboard scored 46% of the test; the private leaderboard, which paid out, scored the rest.

The budget was generous by later standards: a CPU or GPU notebook, up to 9 hours, internet off, external data allowed, and 3 final submissions you could select. Prize money totalled USD 110,000 across two tracks. The leaderboard track paid USD 20k, 10k, 8k and 7k for 1st to 4th and USD 5k each for 5th to 7th. A separate **efficiency track** for CPU-only notebooks paid USD 20k, 10k, 8k, 7k and 5k, scored by a formula that trades AUC against runtime:

$$
\text{Efficiency} = \frac{\text{AUC}}{\text{Benchmark} - \max\text{AUC}} + \frac{\text{RuntimeSeconds}}{32400}
$$

Lower is better (Benchmark is the sample submission's AUC, so the first term is negative). The leaderboard API lists Kelvin, rank 40 on the main board, as the efficiency winner; the rest of that ranking was not verified for this post.

Two early discoveries reshaped the problem. Nicholas Broad showed the training essays were exact matches for essays in the public **PERSUADE 2.0** corpus (also behind the Feedback Prize competitions) and listed its 15 prompts ([452172](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452172)); Carl McBride Ellis then identified which 7 the test used. The human side was public. The generated side, which models the hosts used and how they sampled them, was the unknown.

## 2. The metric, worked on a toy

The score was ROC AUC on your generated probability. AUC has a simple reading: pick one generated essay and one human essay at random, and AUC is the probability that your model scores the generated one higher.

$$
\text{AUC} = \frac{1}{|G|\,|H|} \sum_{g \in G} \sum_{h \in H} \mathbf{1}\left[s(g) > s(h)\right]
$$

Here $G$ is the set of generated essays, $H$ the set of human essays, $s(\cdot)$ your score, and ties count one half.

![ROC AUC on a toy: 4 generated and 4 human scores, 13 of 16 pairs correctly ordered, AUC 0.8125](/imgs/blogs/kaggle-detect-ai-generated-text-solutions-3.webp)

Take four generated essays scored 0.90, 0.80, 0.55, 0.30 and four human essays scored 0.60, 0.40, 0.20, 0.10. There are ${4 \times 4 = 16}$ pairs. The two top generated scores beat all four humans (8 pairs). The essay at 0.55 beats three humans but loses to the one at 0.60. The essay at 0.30 beats two and loses to two. That is ${8 + 3 + 2 = 13}$ correct pairs, so AUC $= 13/16 = 0.8125$.

Notice what is missing from that calculation: the actual values. Replace every score by its rank and nothing changes. Two consequences shaped the competition:

- **Calibration is irrelevant.** Output 0.51 for every generated essay and 0.49 for every human one, and you score a perfect 1.0.
- **Rank averaging is the natural ensemble.** Models with very different score scales (an SVM margin, a softmax, a log-probability statistic) blend cleanly once converted to ranks. The winners did exactly that.

The leaderboard was also extremely compressed at the top: the 1st place's 0.987824 and the 20th's 0.932606 are 0.055 apart, In pair terms, 0.988 means roughly 1 mis-ordered pair in 80.

## 3. What made it hard

Three things, each of which ended up defining a family of solutions.

**No labelled positives, and an unknown generator distribution.** Radek Osmulski's 500 shared essays were the first positives, and Darek Kłeczek's DAIGT V2 and V4 datasets became the base most teams (1st, 2nd and 4th among them) built on. But any dataset you generate encodes your choice of models, prompts and sampling, while the hidden test encoded the hosts'. Fit your own generator too well and you learn to detect *your* LLM, not *theirs*.

**No trustworthy validation.** Writeup after writeup says the same thing. 4th place: the models "all had perfect CV scores". 8th place: CV was "always close to 1", useless. 13th: CV was "practically none". 19th never found a correlation between CV and leaderboard. When your validation set is generated by the same pipeline as your training set, it measures nothing about the hosts' generators.

**Injected character noise.** The test essays contained artificial character-level obfuscations: inserted, deleted and replaced characters that do not appear in PERSUADE. Some characters present in PERSUADE never appear in `train.csv` at all. 4th place concluded the obfuscations were "completely random" character insertions and replacements; the hosts never published the exact process.

![Typo noise as signal, then as attack: human typos separate classes in train, injected noise erases that in test, and the fix injects typos into 30 percent of generated essays and lets a BPE tokenizer turn them into features](/imgs/blogs/kaggle-detect-ai-generated-text-solutions-4.webp)

Typos were the easiest signal in the data: students make them, LLMs do not. A "Find Typos" thread scored 0.714 in 14 lines of code by counting them. Then the obfuscation put typos into generated essays too, and any model that had learned "typo means human" was wrong in a systematic direction. The field split two ways: **remove** the noise (spell checkers, a T5 deobfuscator, Levenshtein correction) or **train through** it (inject typos into generated training essays so they stop carrying the label). Removing it was the riskier bet.

## 4. Top-20 scorecard

Ranks and scores are from the Kaggle leaderboard API. "Public → private" is the move from public rank to private rank.

| Rank | Team | Private AUC | Public → private | Key idea | Source |
|---|---|---|---|---|---|
| 1 | Udbhav Bamba, Raja Biswas, Nicholas Broad (4 emoji) | 0.987824 | 9 → 1 | 160k diverse "datamix"; five model types, rank-averaged | [473295](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/473295) |
| 2 | Guanshuo Xu | 0.983412 | 17 → 2 | DeBERTa pretrained on 500k SlimPajama pairs, adapted on student-mimicking essays | [470395](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470395) |
| 3 | nlp team | 0.974994 | 15 → 3 | DeBERTa on up to 1.2M continuations + pseudo-labelled TF-IDF | [470333](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470333) |
| 4 | Ertugrul & Chase | 0.973915 | 3 → 4 | BPE TF-IDF with typo augmentation + Mistral probability features + DeBERTa | [470179](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470179) |
| 5 | Linguistic Ninjas | 0.972147 | 22 → 5 | 1.7M rows; test-time distillation into short-context students | [470093](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093) |
| 6 | Davide Cozzolino | 0.969132 | 2,995 → 6 | Phi-2 entropy statistics, One-Class SVM on human essays only | [471831](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/471831) |
| 7 | Hao Mei | 0.965133 | 2,357 → 7 | Base-model generations; one DeBERTa-v3-large | [470643](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470643) |
| 8 | Abdullah Meda | 0.956501 | 1,031 → 8 | GPT-2 perplexity + GLTR features on 800k rows | [470224](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470224) |
| 9 | LLMLab | 0.947710 | 1 → 9 | ~60% TF-IDF + deobfuscation + DeBERTa/RoBERTa | [470255](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470255) |
| 10 | IC2 | 0.947353 | 1,880 → 10 | unknown | no writeup |
| 11 | Data Driven Poets | 0.944266 | 1,033 → 11 | unknown | no writeup |
| 12 | Uesugi Erii | 0.942678 | 168 → 12 | 820k RefinedWeb rows; TF-IDF 0.85 : DeBERTa 0.15 | [470396](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470396) |
| 13 | RB | 0.938988 | 41 → 13 | Transformers only, ~45k rows | [470593](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470593) |
| 14 | takaito | 0.936732 | 117 → 14 | TF-IDF + untuned OpenAI detector feature + pseudo-labels | [470193](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470193) |
| 15 | FakeDetectLab | 0.935624 | 13 → 15 | unknown | no writeup |
| 16 | 冲冲冲！ | 0.934437 | 2,635 → 16 | unknown | no writeup |
| 17 | Magus | 0.933942 | 7 → 17 | unknown | no writeup |
| 18 | ai galvatron | 0.933833 | 66 → 18 | Adversarial data selection; TF-IDF + 7B LLMs | [470061](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470061) |
| 19 | Darek Kłeczek | 0.932794 | 39 → 19 | Robustness-first blend by the DAIGT dataset author | [470181](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470181) |
| 20 | Shimataro | 0.932606 | 44 → 20 | unknown | no writeup |

Gold went to the top 18. Of the top 20, only ranks 1 to 5, 9, 15 and 17 started in the public top 25.

## 5. The top-20 landscape

![Technique matrix for the top 20: 14 teams with writeups, which techniques each used, unknowns shown as question marks for the 6 teams without a writeup](/imgs/blogs/kaggle-detect-ai-generated-text-solutions-5.webp)

The matrix counts each technique across the 14 teams with writeups; the 6 columns without one stay unknown rather than guessed. A few rows carry the story.

**Table stakes.** A DeBERTa-family classifier (10 of 14 writeups), DAIGT V2/V4 as the starting data, and no CV anyone trusted, so robustness decisions were made by reasoning. 4th set its final weights by "common sense", deliberately not by public leaderboard.

**What separated the top 5 from 6 to 20.** First, scale and diversity of *their own* generated data: 7 of 14 generated more than 100k rows, including four of the top five (4th instead merged about 700k public rows). The same 7 used out-of-domain web text (Pile, SlimPajama, RefinedWeb) as human/AI pairs, popularised by James Day's thread ["0.963 with BERT: Transformers love diverse data"](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/465882), which moved a DeBERTa from roughly 0.75 to 0.80 on the leaderboard to 0.958 to 0.963 with about 500k Pile continuations from 12 LLMs. Second, an explicit **domain-adaptation** step that pulled the general detector back toward student essays: student-mimicking generators (2nd), test-time distillation (5th), or confident pseudo-labels on the test set (1st, 3rd, 4th).

**A different road to gold.** Ranks 6 and 8 fine-tuned no neural model: they fed cheap statistics from a frozen language model (entropy, surprisal, perplexity, GLTR ranks) to a classical classifier, and 14th used an off-the-shelf detector the same way. Nothing test-specific to overfit, so they climbed.

**Where the leaderboard misled.** Test-vocabulary TF-IDF, a vectoriser whose vocabulary is fit on the test essays themselves, appears in 8 of 14 finals (7th only as a small tie-breaker), but only 3rd and 4th among the top five kept it, and both had large transformer components beside it. The teams that leaned hardest on TF-IDF (9th at about 60% weight, 12th at 85%) finished 9th and 12th, behind every top-8 team that relied on generated data or frozen-LM statistics.

**Contradictory findings.** The writeups disagree on five things:

| Question | Said yes | Said no | Context from the writeups |
|---|---|---|---|
| Fine-tune an LLM on PERSUADE to generate student-like essays? | 1st, 2nd | 12th (quality dropped), 13th | 13th's own diagnosis: she used Mistral-7B-*Instruct*; 2nd used base models |
| Fine-tune a 7B LLM as the classifier? | 1st (best single model, 0.984) | 12th (0.642 LB), 13th (0.69 private) | 1st had the 160k diverse mix; 12th trained on 82k at over 10 hours per epoch |
| Correct the typos before classifying? | 3rd (only above 15 errors), 9th, 19th | 8th, 12th (hurt) | 9th's David Austin: fix *systematic* errors, but random noise may be distributed differently in train and test |
| MLM pretraining on test text? | 1st (from-scratch DeBERTa-small) | 7th (0.689), 13th | 1st used it with Mistral pseudo-labels on a tiny model as one ensemble member |
| Longer context? | 4th trained at 1024 tokens | 7th, 13th: 512 beat 1024 | No writeup explains it; 5th pushed short context to 128 *characters* |

## 6. The winning solution, explained

The 1st-place team published two writeups: a short one by Raja Biswas ([470121](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470121)) and a comprehensive one by Nicholas Broad ([473295](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/473295)), with code at [rbiswasfc/llm-detect-ai](https://github.com/rbiswasfc/llm-detect-ai). Their thesis fits in one sentence from the writeup: "The modeling approach was less important, as we had multiple single models in the 0.970+ range due to the quality of the dataset." Each individual model "would have been in the gold medal range".

![First place: a 160k datamix of about 40k human and 120k generated essays feeds five models whose private AUCs range from 0.943 to 0.984, rank-averaged to 0.987824](/imgs/blogs/kaggle-detect-ai-generated-text-solutions-6.webp)

### The intuition: cover the space, then plug the holes

Think of the hosts' generators as an unknown region in "text space". You cannot sample from it, but you can surround it: with many model families, sampling settings and kinds of noise, wherever the hosts' region sits is probably close to something you trained on. The team built the mix incrementally, each round plugging the previous model's blind spots.

### The mechanism: the datamix

About 160k rows, about 40k of them human:

- **Human text:** all 15 PERSUADE prompts (not just the test's 7), plus ELLIPSE, NarrativeQA, Wikipedia, NLTK Brown, IMDB reviews and the human side of the GPT-2 output dataset.
- **Generated text, four sources:** proprietary LLMs (gpt-3.5, gpt-4, claude, cohere, gemini, palm); open LLMs (llama, falcon, mistral, mixtral); existing sets (DAIGT V2 subset, OUTFOX, Ghostbuster data and others); and **LLMs fine-tuned on PERSUADE** (mistral, llama, falcon, deci-lm, t5, pythia, BLOOM, GPT2), conditioned on essay metadata, trained with one prompt held out to force new styles, or generating one discourse span at a time.
- **Decoding diversity:** contrastive search, guidance scale, typical_p, high temperature, large top-k, prompts with and without source texts, rewrites and fill-in-the-blank.
- **Attack augmentations** on a random subset: character delete/insert/swap, spelling correction, synonyms, obfuscation, back-translation, random capitalisation, sentence swaps; plus a heavily normalised copy to resist character attacks.

The last bullet is the answer to section 3's dilemma. The team did not try to remove the hosts' noise; they made noise uninformative by putting it on both classes.

### The models

| Model | Private | Public | Note |
|---|---|---|---|
| Mistral-7B-v0.1 (Q)LoRA, sequence classification | 0.984 | 0.966 | `r=64`, `lora_alpha=16`, dropout 0.1, targets q/k/v/o; most ensemble weight |
| Ghostbuster re-implementation | 0.974 | 0.957 | Llama-7B + TinyLlama-1.1B token probabilities, 100 features, SVM + Random Forest on RAPIDS |
| Unsupervised (Ahmet Erdem's similarity baseline) | 0.971 | 0.957 | Mistral-confident weak labels, mean instead of min aggregation |
| DeBERTa-v3-large, classification or pairwise ranking loss | 0.963 | 0.961 | Continuous labels: essays from instruction-tuned LLMs get a label below 1 |
| DeBERTa-v3-small trained from scratch at inference | 0.943 | 0.942 | Tokenizer and MLM pretraining on train + test, then 1 epoch on datamix + Mistral pseudo-labels |

The **Ghostbuster** re-implementation swaps the paper's OpenAI models for Llama-7B and TinyLlama-1.1B (same tokenizer) plus Brown-trained unigram and trigram models, then takes 10 aggregations over 4 probability sequences and 6 ratios: 100 features. The **from-scratch DeBERTa-small** is transductive: its tokenizer and pretraining see the test essays, so the injected noise becomes part of its vocabulary.

### The ensemble

Each model's test scores were converted to ranks from 1 to $n$, the ranks averaged with light weights set by public leaderboard and intuition, most weight on Mistral. Here is an illustrative toy of the two ideas that generalise from this solution, typo augmentation on generated essays and rank averaging. This is not the team's code.

```python
# Illustrative toy, not any team's code: typo augmentation + rank-average ensemble.
import random
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

LETTERS = "abcdefghijklmnopqrstuvwxyz"

def add_typos(text: str, rate: float = 0.02, seed: int = 0) -> str:
    """Randomly insert, delete or replace characters, like the hosts' obfuscation."""
    rng = random.Random(seed)
    out = []
    for ch in text:
        r = rng.random()
        if r < rate / 3:            # delete
            continue
        elif r < 2 * rate / 3:      # replace
            out.append(rng.choice(LETTERS))
        elif r < rate:              # insert after
            out.extend([ch, rng.choice(LETTERS)])
        else:
            out.append(ch)
    return "".join(out)

def augment(texts, labels, frac=0.30, seed=0):
    """Put typos on a fraction of GENERATED essays so typos stop meaning 'human'."""
    rng = random.Random(seed)
    return [add_typos(t, seed=i) if y == 1 and rng.random() < frac else t
            for i, (t, y) in enumerate(zip(texts, labels))]

def rank_average(score_lists, weights):
    """AUC only sees order, so blend ranks, not raw scores."""
    ranks = [rankdata(s) / len(s) for s in score_lists]
    return np.average(np.vstack(ranks), axis=0, weights=weights)

y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
llm_logits = np.array([4.1, 2.3, 0.2, -0.5, 0.9, -1.0, -2.2, -3.0])  # unbounded
svm_margin = np.array([0.8, 0.9, 0.4, 0.7, 0.1, 0.5, -0.3, -0.9])    # different scale, different mistakes
blend = rank_average([llm_logits, svm_margin], weights=[0.5, 0.5])
print(roc_auc_score(y, llm_logits), roc_auc_score(y, svm_margin), roc_auc_score(y, blend))
# 0.875 0.9375 0.96875: each model misorders a different pair, so the rank blend beats both
```

The `frac=0.30` default mirrors 4th place's choice of injecting typos into 30% of generated essays; 1st place applied its attacks to an unstated "random subset".

## 7. Second and third in depth

### 2nd: Guanshuo Xu, pretrain broad, adapt narrow

Guanshuo Xu ([470395](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470395)) started late, solo. DeBERTa on the shared datasets gave "nearly perfect CV but much lower and unstable public LB". Following James Day, he generated about 500k human/AI pairs from SlimPajama with various LLMs and trained DeBERTa-v3-large: 0.916 public, 0.967 private. This **pretraining** step makes a general human-versus-AI classifier.

Then **domain adaptation**. The test prompts reference source articles he could not find, so generated essays could look very student-like. He fine-tuned LLMs on student essays with the no-code H2O LLM Studio so they mimicked students down to citations and typos, generated essays, and fine-tuned the classifier on them: about 0.94 public, 0.98 private. The final submission averaged 2 DeBERTa-v3-large and 1 DeBERTa-large, plus the same three fine-tuned on DAIGT V4: **0.967 public, 0.983 private**.

The telling detail: blending with a 0.961 public TF-IDF kernel gave his best public score and a lower private one. He did not select it.

### 3rd: nlp team, scale plus a careful blend

Yevhenii Maslov's writeup ([470333](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470333)) reports a failure honestly. The team began with **hard-example mining**: each iteration added the 500 human and 500 generated samples the previous model got most wrong, stopping when the leaderboard stopped improving. The best of these 11k-row models scored 0.927 public and **0.845 private**: the stopping rule had fit the public split.

What worked was the opposite: **continuations of Pile and SlimPajama text by about 35 open-source LLMs**, filtered and sampled under three temperature scenarios. Three DeBERTa-v3-large models trained on 500k, 1M and 1.2M rows; the best, at about 1M, scored **0.956 public, 0.967 private**.

The TF-IDF side added 1k confident test pseudo-labels ($p \lt 0.01$ or $p > 0.99$): public flat, private 0.893 to 0.927. The blend used TF-IDF only where transformers were uncertain (0.1 to 0.9), and a UMAP nearest-neighbour step rescaled scores within $[0.9, 1.1]$. A T5 deobfuscator ran only on texts with more than 15 errors.

## 8. Ranks 4 to 20, by approach family

### Family A: big diverse data plus a domain step (4, 5, 7, 12)

- **4th, Ertugrul & Chase** ([470179](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470179)) called it "combined arms": TF-IDF n-grams (3 to 7) over a [BPE tokenizer](/blog/machine-learning/large-language-model/bpe-tokenizer) trained on the data, with random typos injected into 30% of generated essays; Mistral-7B token-probability features (such as the 10th percentile of log-probabilities) re-trained on TF-IDF pseudo-labels; a Longformer predicting the *source dataset*; and DeBERTa-v3 on about 700k public rows for uncertain cases.
- **5th, Linguistic Ninjas** ([470093](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093), code [jday96314/AiContentDetection](https://github.com/jday96314/AiContentDetection)) trained on 1.7M examples, mostly Pile and SlimPajama completions with randomised sampling per request. Their **test-time domain adaptation** is the most original idea in the top 20: a teacher (1 DeBERTa + 2 Mamba) soft-labels the test set, then students reading only 128 and 256 *characters* are fine-tuned on test chunks to imitate it. The students without Mamba scored 0.977 private, enough for 3rd, but were not selected.
- **7th, Hao Mei** ([470643](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470643)), in his first competition, generated only with **base** Falcon-7B, Mistral-7B and Llama-2-7B, arguing the hosts' samples lacked chat formatting and RLHF causes mode collapse. His prompt was a fake exam sheet with a real PERSUADE essay and a new header to continue. One DeBERTa-v3-large at 512 tokens scored 0.965 private (0.957 at 1024).
- **12th, Uesugi Erii** ([470396](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470396)) built 820k RefinedWeb rows (410k pairs): a random prefix as the prompt, the true remainder as human, Mistral and Llama-2 continuations as generated, both truncated to the same random length so length could not leak the label. His blend leaned TF-IDF 0.85 : DeBERTa 0.15, deliberately using what he called "feature leakage".

### Family B: frozen language-model statistics, no fine-tuning (6, 8, 14)

- **6th, Davide Cozzolino** ([471831](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/471831)), a multimedia-forensics professor, spent "only four or five days". He reduced Phi-2 token entropy and surprisal to 5 statistics and fit an RBF **One-Class SVM on the human essays in `train.csv` only**; no generated text was used. His forensics prior: a model of authentic data generalises to forgeries it has never seen. The pipeline ran in about 20 minutes on a P100. By his own submission log, going from 3 features to 5 left public unchanged at 0.873 and lifted private from 0.902 to 0.968 (his document prints 0.968132; the API records 0.969132), a reminder of how little the public split could tell you.
- **8th, Abdullah Meda** ([470224](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470224)) used GPT-2 perplexity and GLTR rank counts (how many tokens fall in the model's top 10, 100, 1000) in a VotingClassifier, about 10 minutes per submission. His ablation is a lesson in its own right: GPT-2 small perplexity features scored 0.506 private on DAIGT V2 (44k rows) and 0.656 on an 800k-row dataset, about 18 times the data; adding GLTR and moving to GPT-2 large reached 0.956.
- **14th, takaito** ([470193](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470193)) added `roberta-large-openai-detector`, **untuned** (tuning overfit), as a feature beside public TF-IDF features in LightGBM and XGBoost, plus per-prompt pseudo-labels (top 10% most confident as generated, top 7.5% as human).

### Family C: TF-IDF-heavy blends (9, 18, 19)

- **9th, LLMLab** ([470255](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470255)) was public #1 with about 60% of the weight on TF-IDF and the rest on DeBERTa and RoBERTa over 200k to 700k rows. Their deobfuscation gave +0.006 public and -0.001 private. David Austin's nuance is worth keeping: remove *systematic* mistakes but *inject* random ones.
- **18th, ai galvatron** ([470061](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470061)) selected training data adversarially: train a small model on 10% of the pooled public sets, keep the hardest of the other 90%, mix 50/50 with PERSUADE. Final blend 0.4 TF-IDF, 0.6 Mistral/Llama-7B models. Exact models and LoRA ranks are not stated.
- **19th, Darek Kłeczek** ([470181](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470181)), author of the DAIGT datasets, built a robustness-first blend: five TF-IDF learners, a readability ranking, DeBERTa-v3-large and Mistral-7B QLoRA, plus last-day Levenshtein spelling correction.

### Family D: transformers only, small data (13)

- **13th, RB** ([470593](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470593)) trained DeBERTa-xlarge, DeBERTa-v3-large and RoBERTa-large at 512 tokens on about 45k rows (DAIGT V4 plus her own GPT essays prompted to sound like a grade 6 to 12 student). 0.957 public, 0.938 private; her TF-IDF blend scored 0.97 public but 0.927 private, and she chose the transformers.

### No writeup: 10, 11, 15, 16, 17, 20

IC2, Data Driven Poets, FakeDetectLab, 冲冲冲！, Magus and Shimataro published no solution (all 32 forum pages scanned); their methods are unknown. Three of them climbed from public 1,880, 1,033 and 2,635, like 6th to 8th, which is suggestive but proves nothing.

### The best contrasting entry

6th place is the contrast worth remembering. Everyone else asked "what does AI text look like?" and chased the hosts' generators. Cozzolino asked "what does *human* text look like?", which has a stable answer: the human side of this competition was a public corpus. His model never saw a generated essay, so it could not overfit one.

## 9. What the forum taught everyone

![Public rank against private rank on a log scale: TF-IDF leaders fell from the public top 10, while general detectors climbed from public ranks above 1,000 into the top 20](/imgs/blogs/kaggle-detect-ai-generated-text-solutions-7.webp)

Shared work moved the whole field: Radek's essays, Darek's DAIGT datasets, James Day's diverse continuations, Ertugrul Demir's custom-tokenizer thread, Ahmet Erdem's 0.91 unsupervised baseline (adopted by 1st), sorokin's T5 deobfuscator (used by 3rd). Alejandro Mosquera asked early whether TF-IDF's "unreasonable" strength was an artefact of this test set. The shake-up answered yes.

The chart above is that answer. Public leaders fell (LLMLab 1 to 9, Secret Sauce 2 to 21, CPMP 4 to 163, J.M 5 to 105) while 6th, 7th and 8th climbed from public 2,995, 2,357 and 1,031. CPMP's post-mortem, ["The curse of early success"](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470154), is the clearest diagnosis: 0.970 public with n-grams over a vocabulary fit on test and a LinearSVC, so the public score never asked him to build general data or models. A thread suspected "a leak or secret sauce" behind the climbers; the writeups show only generalisation.

Why did a 46% public split mislead? Test-vocabulary TF-IDF and leaderboard-driven choices fit the *specific* essays in the public portion, and every probe chosen by public score added more of that fit. 3rd place's hard-example miner (0.927 public, 0.845 private) is the controlled experiment.

![Training-data scale against private AUC: beyond about 100k rows more rows did not buy rank, and first place had the most diverse mix rather than the largest](/imgs/blogs/kaggle-detect-ai-generated-text-solutions-8.webp)

Scale mattered up to a point (8th's 44k-to-800k jump), then stopped: 5th trained on 1.7M rows, 1st on 160k. Past roughly 100k rows, what you generated beat how much.

## 10. What did not work

Aggregated across the 14 top-20 writeups, with who reported it:

| Idea | Reported by | Outcome |
|---|---|---|
| Hard-example mining with public-LB stopping | 3rd | 0.927 public, 0.845 private |
| Test-vocabulary tokenizer or TF-IDF tuned to public LB | 8th; CPMP (public 4, private 163) | Hurt; 12th left TF-IDF untouched to avoid local overfit |
| Spelling correction or deobfuscation (T5, PySpellChecker, distance- and NN-based) | 8th, 12th | Hurt (3rd, 9th, 19th used limited forms) |
| Fine-tuning RoBERTa or ChatGPT detectors on large data | 8th; 14th (tuning the OpenAI detector) | 8th: 0.674 private; 14th: overfit |
| Fine-tuned 7B LLM classifier | 12th, 13th | 0.642 LB; 0.69 private |
| LoRA on DeBERTa | 4th | "didn't go well" |
| Instruct LLMs fine-tuned to mimic PERSUADE | 12th, 13th | Quality dropped; use the base model |
| MLM pretraining on test or train | 7th (0.689), 13th | Failed |
| 1024-token training, sliding-window inference | 7th, 13th | 512 was better |
| Log-likelihood and readability features | 8th | Hurt (19th still used readability as one blend member) |
| Mistral reward model on pairs; Ghostbuster; pairwise ranking + Bradley-Terry; token-level NER detector | 19th | Failed (1st made Ghostbuster work) |

The pattern: whatever fit the public test or the training generator more tightly tended to fail; whatever made the detector more general held.

## 11. Lessons that transfer

1. **Build the training set; the model is secondary.** 1st had several single models above 0.970 "due to the quality of the dataset" ([473295](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/473295)). See [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation): coverage beats volume.
2. **Diversity of generators and decoding settings generalises.** Many families, samplers and student-mimicking fine-tunes ([473295](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/473295), [470395](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470395), [470093](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093)). Deduplicate against your eval data, as 12th and 13th did with embedding similarity (see [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale)).
3. **Pretrain broad, then adapt narrow.** 2nd went from SlimPajama pairs to student-like essays; 5th distilled a teacher into short-context students on the test set ([470395](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470395), [470093](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093)). [Knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals) covers the teacher-student half.
4. **Anticipate adversarial noise by training through it.** Put typos and obfuscation on generated training text so the noise stops carrying the label ([473295](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/473295), [470179](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470179)).
5. **For AUC, ensemble ranks.** The metric only sees order ([473295](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/473295)).
6. **A public leaderboard that rewards test-fitting is a trap.** Pick the submissions you would trust on prompts you have never seen ([470154](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470154)).

### If you enter a competition like this

- Find the source corpus of the human side first; it is often public.
- Generate from many base and instruct models and samplers; track each row's source.
- Put noise augmentations on the generated class so noise stops meaning human.
- Build at least one model that never sees your generated data (a one-class or frozen-LM-statistics model) as a robustness anchor.
- Treat CV as a smoke test; decide final weights by a written argument.
- Select one submission that you would defend on unseen prompts, even if it is not your public best.
- If you want a 7B classifier, see [LoRA and QLoRA](/blog/machine-learning/reinforcement-learning/lora-qlora-for-rlhf) for the adapter mechanics, but only after the data is right.

Other levers, same lens: [LLM Science Exam](/blog/machine-learning/learn-from-kaggle/kaggle-llm-science-exam-solutions) (retrieval), [LLM Prompt Recovery](/blog/machine-learning/learn-from-kaggle/kaggle-llm-prompt-recovery-solutions) (metric insight), [LMSYS Chatbot Arena](/blog/machine-learning/learn-from-kaggle/kaggle-lmsys-chatbot-arena-solutions) and the [WSDM Cup multilingual arena](/blog/machine-learning/learn-from-kaggle/kaggle-wsdm-cup-multilingual-arena-solutions) (model fit).

## Sources

- Competition page: https://www.kaggle.com/competitions/llm-detect-ai-generated-text
- 1st: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/473295 and https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470121; code https://github.com/rbiswasfc/llm-detect-ai
- 2nd: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470395
- 3rd: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470333
- 4th: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470179 and https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/471934
- 5th: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093; code https://github.com/jday96314/AiContentDetection
- 6th: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/471831; code https://github.com/davin11/entropy-based-text-detector
- 7th: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470643; code https://github.com/Tailen/Kaggle-Detect-AI-Generated-Text-7th-Solution
- 8th: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470224
- 9th: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470255
- 12th: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470396
- 13th: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470593
- 14th: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470193
- 18th: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470061
- 19th: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470181
- Forum threads: PERSUADE match https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452172; Radek's 500 essays https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452155; James Day https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/465882; custom tokenizers https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/458522; CPMP https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470154; Secret Sauce https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470148
- Ranks and scores: Kaggle leaderboard API, pulled 2026-10-28

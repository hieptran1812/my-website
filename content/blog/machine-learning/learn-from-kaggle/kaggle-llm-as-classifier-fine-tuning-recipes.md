---
title: "LLMs as classifiers on Kaggle: the fine-tuning recipes eight competitions actually used"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "How top Kaggle teams turned Gemma, Qwen, Llama and Mistral into classifiers across LMSYS, WSDM, MAP, Jigsaw, Eedi, Science Exam, DAIGT and PII: heads versus label-token logits, LoRA settings exactly as reported, truncation, swap TTA, pseudo-labels, CV and distillation."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "lora", "text-classification", "knowledge-distillation", "pseudo-labeling", "test-time-augmentation", "fine-tuning"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 13
---

> **TL;DR.** Across eight Kaggle text competitions, the top teams converged on one shape: a 7B to 32B decoder with LoRA, a readout that is either a new classification head or the logits of existing label tokens, TTA spent on uncertain rows, soft labels from 70B-class teachers, and 8-bit or 4-bit weights. The hyperparameters did not converge: reported LoRA ranks run from 8 to 1024 and learning rates from 4e-6 to 5e-4. This post tabulates what teams *said they used* and recommends no default nobody published.

Every competition here asks one question in a different costume: given some text, output a probability over a small set of labels. LMSYS wants "A wins, B wins, tie", Jigsaw wants "violates the rule or not", MAP wants one of 65 misconceptions. In each, the top of the board was held by a model built to *generate text*. The engineering is in that gap: making a next-token predictor emit a calibrated class distribution inside a fixed, internet-off notebook, usually two T4s for 9 to 12 hours.

The diagram above is the mental model: one backbone, three places for the label to come out. A new linear head on the last hidden state, the vocabulary logits restricted to a few label tokens, or one score per candidate with a softmax across candidates. Which panel won depended on the inference engine more than on accuracy.

![Three ways to read a class label out of a decoder LLM: a new classification head, restricted label-token logits, or per-candidate scoring](/imgs/blogs/kaggle-llm-as-classifier-fine-tuning-recipes-1.webp)

Every hyperparameter below is quoted as a team wrote it, every count is recounted from the per-competition technique matrices, and "?" means the writeups are silent. For the notebook contract itself, see [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work).

## 1. Three readouts, and why the engine picked one

**Readout (a): a classification head.** `AutoModelForSequenceClassification` bolts a random linear layer onto the last token's hidden state. It was unanimous in [LMSYS Chatbot Arena](/blog/machine-learning/learn-from-kaggle/kaggle-lmsys-chatbot-arena-solutions): all 10 top-20 writeups used a head, none used label-token logits. 1st (BlackPearl) used the stock head; 2nd (Rist) swapped in `Dropout, Linear(h, h/2), Dropout, GELU, Linear(h/2, 3)`; 16th (Chris Deotte) trained three heads (preference plus the identity of model A and model B), worth 0.891 to 0.890 in log loss.

**Readout (b): label-token logits.** Keep the language-model head. Put the options in the prompt, read the logits of `A` and `B` (or `Yes` and `No`) at the last position, and softmax over just those.

```python
# Illustrative toy, not any team's code.
import numpy as np

def softmax(z):
    z = np.asarray(z, dtype=float) - np.max(z)
    e = np.exp(z)
    return e / e.sum()

# The lm_head already scores every vocabulary token at the last position.
rng = np.random.default_rng(0)
vocab_logits = rng.normal(0.0, 1.0, size=32_000)   # stand-in for one forward pass
LABEL_IDS = {"A": 330, "B": 365}                     # toy token ids
vocab_logits[LABEL_IDS["A"]] = 4.1
vocab_logits[LABEL_IDS["B"]] = 3.2

# Keep only the label tokens and renormalise: a 2-way classifier for free.
pair = softmax([vocab_logits[i] for i in LABEL_IDS.values()])
print("label-token readout  P(A), P(B) =", pair.round(3))
# In training, the loss is cross-entropy over these two logits only
# (WSDM 4th: "between A and B logits only"), not the full vocabulary.
```

```console
label-token readout  P(A), P(B) = [0.711 0.289]
```

A two-way softmax of a 0.9 logit gap is sigmoid(0.9) = 0.711. No new parameters, no random head to warm up.

**Readout (c): one score per candidate.** When the label set is large or varies per question, score each candidate. [MAP](/blog/machine-learning/learn-from-kaggle/kaggle-map-student-math-misunderstandings-solutions) 1st place (tascj) took "last-token hidden state of prefix+suffix_i, Linear(hidden,1), logits over candidates, cross-entropy". [Eedi](/blog/machine-learning/learn-from-kaggle/kaggle-eedi-mining-misconceptions-solutions) 1st (MTH 101) scored each misconception as `logit("Yes") - logit("No")` with the `lm_head` frozen. [Science Exam](/blog/machine-learning/learn-from-kaggle/kaggle-llm-science-exam-solutions) 1st (H2O) used one binary sample per (question, option).

Why did the field drift from (a) to (b) in 2025? vLLM. [WSDM](/blog/machine-learning/learn-from-kaggle/kaggle-wsdm-cup-multilingual-arena-solutions) 3rd (PLaMo) chose "CausalLM, not SequenceClassification" to re-use the Eedi 1st-place pipeline under vLLM. MAP 2nd turned 65 classes into letters `A..BM` "so vLLM could serve it in time". MAP 7th is the cleanest statement: classification heads "scored similarly but logits allowed fast vLLM inference". MAP 10th found a third route, serving the backbone with vLLM `task="embed"` and applying the FP16 head outside it.

## 2. The cross-competition comparison

"Writeups" is how many of the top 20 published a method; matrix counts are out of that number.

| Competition | Metric | Writeups | Dominant readout | Typical backbone | What made it a classifier problem |
|---|---|---|---|---|---|
| [LMSYS](/blog/machine-learning/learn-from-kaggle/kaggle-lmsys-chatbot-arena-solutions) | 3-class log loss | 10 | Seq-cls head (10/10) | Gemma-2-9B (10/10) | Calibrated A/B/tie probabilities |
| [WSDM Cup](/blog/machine-learning/learn-from-kaggle/kaggle-wsdm-cup-multilingual-arena-solutions) | Accuracy | 8 | Split: head 4, A/B tokens 4 | Qwen2.5-14B (7/8) | Binary winner, multilingual, long inputs |
| [Science Exam](/blog/machine-learning/learn-from-kaggle/kaggle-llm-science-exam-solutions) | MAP@3 | 17 | DeBERTa multiple-choice (15/17) | DeBERTa-v3, 7B readers | Rank 5 options given retrieved context |
| [DAIGT](/blog/machine-learning/learn-from-kaggle/kaggle-detect-ai-generated-text-solutions) | ROC AUC | 14 | DeBERTa (10/14), 7B in 3/14 | Mistral-7B, DeBERTa-v3 | Human vs machine essay |
| [PII](/blog/machine-learning/learn-from-kaggle/kaggle-pii-data-detection-solutions) | Micro F5 | 13 | Token-classification head (13/13) | DeBERTa-v3-large | BIO tags, recall-weighted |
| [Eedi](/blog/machine-learning/learn-from-kaggle/kaggle-eedi-mining-misconceptions-solutions) | MAP@25 | 15 | Yes/No or option tokens (15/15) | Qwen2.5-32B reranker (13/15) | Retrieve then rerank about 2.5k labels |
| [MAP](/blog/machine-learning/learn-from-kaggle/kaggle-map-student-math-misunderstandings-solutions) | MAP@3 | 14 | Both: head 8, tokens 9 | Qwen3/Qwen2.5 14B to 32B | 65 labels, few per question |
| [Jigsaw Rules](/blog/machine-learning/learn-from-kaggle/kaggle-jigsaw-agile-community-rules-solutions) | Per-rule AUC | 15 | Yes/No token (11/15) | Qwen3 4B to 14B | Unseen rules, train on test examples |

![Cross-competition matrix: eight competitions against eight classifier recipe choices, with N of M counts](/imgs/blogs/kaggle-llm-as-classifier-fine-tuning-recipes-2.webp)

Read it by column. LoRA (LMSYS 8/10, Eedi 15/15, MAP 11 of 12 known) and low-bit inference past 14B (Eedi 12/15, MAP 9 of 9 known) are nearly solid; the readout columns flip by year. PII and DAIGT are the control group: in PII every writeup used DeBERTa-v3-large with a token head, and in DAIGT 7B classifiers worked for one team.

Two recounts differ from earlier sources (MAP full fine-tuning stays 4 of 13 known: 1st, 5th, 7th and Chris Deotte's 18th, who fully fine-tuned models up to 14B): Eedi distillation is 2 of 15, not 3, since 10th's "did not work"; Eedi 4-bit is 12/15, since 14th never stated its quantisation.

## 3. LoRA settings, exactly as reported

These are settings teams *published*. None is a recommendation; the spread is the point.

| Competition, rank, team | Method | r | alpha | lr | Epochs | Other stated detail |
|---|---|---|---|---|---|---|
| LMSYS 1st, BlackPearl | LoRA 9B, QLoRA 70B/72B | 64 | 128 | 5e-5 (distill), 1e-5 (post-pretrain) | 2 | All linear, global batch 64 |
| LMSYS 4th, CAST | LoRA bf16 | 64 | 16 | ? | ? | Dropout 0.05 |
| LMSYS 9th, Ebi | LoRA (Gemma) | 32 | 64 | 8e-5 | 1 | All linear, global batch 16 |
| LMSYS 13th, heartkilla | QLoRA 4-bit, stage 3 | 32 | 32 | ? | ? | q/k/v/o, rsLoRA "~0.02 gain" |
| LMSYS 18th, Shelton | LoRA | 64 | 32 | 1e-5 | 1 | No frozen layers (public notebook froze 16) |
| WSDM 1st, whitefebruary | Full fine-tune | n/a | n/a | 4e-6 peak | ? | ZeRO-3, 50 warm-up steps |
| WSDM 5th, Just doing | LoRA fp16 | 64 | 128 | 1e-5 (A), 5e-5 (B) | 1 | LoRA+ style split, batch 16 |
| WSDM 12th, monsaraida | QLoRA | 8 | ? | 2e-4 | 1 | |
| DAIGT 1st | (Q)LoRA, seq-cls | 64 | 16 | ? | ? | q/k/v/o, dropout 0.1 |
| Eedi 5th, Ebi & ktr | LoRA | 24 | 48 | 5e-5 | 1 | Global batch 8, Unsloth |
| Eedi 7th | LoRA | 8 | 16 | 1e-4 | 2 | |
| Eedi 8th, Yannan Chen | 4-bit + LoRA incl. lm_head | 32 | 64 | 5e-4 | 4 | wise-FT merge alpha 0.8 |
| MAP 1st, tascj | Full fine-tune | n/a | n/a | 1e-5 | 1 | Batch 32, offloaded Adam |
| MAP 8th, Takashi Someya | LoRA | 16 | 16 | 2e-4 | 3 | All 7 projections, batch 12 |
| MAP 9th, kanto-kaggler | LoRA | 32 | 64 | 1e-4 | 2 | Batch 8 x accum 2 |
| MAP 10th, Liwen Zheng | 4-bit NF4 LoRA | 64 | 128 | 1e-4 | 3 to 4 | Score head saved |
| Jigsaw 5th, Flash | LoRA | 256 | 32 | about 2e-4 | ? | paged_adamw_8bit, cosine |

![Reported LoRA ranks per team across competitions, clustered at 64 with outliers at 8 and 1024](/imgs/blogs/kaggle-llm-as-classifier-fine-tuning-recipes-3.webp)

The mode is r=64, but the figure is a census, not a curve: the lowest stated rank (8, WSDM 12th and Eedi 7th) and the highest (1024, LMSYS 16th, worth +0.002) both finished near gold. MAP 8th wrote that "r and alpha did not matter; LoRA beat full FT" in his runs. LMSYS 3rd (Raja) found rsLoRA, LoRA+, DoRA and full tuning "not clearly better". One setting does repeat: of the 14 rows that state an epoch count, 11 trained for one or two epochs. Target modules split between all linear layers (LMSYS 1st and 9th, MAP 8th and 9th, Jigsaw 5th) and attention only (LMSYS 13th, DAIGT 1st).

The real lesson is a coupling, measured by Chris Deotte in LMSYS 16th.

![LMSYS 16th place measured best LoRA alpha at three learning rates: higher lr, smaller best alpha](/imgs/blogs/kaggle-llm-as-classifier-fine-tuning-recipes-4.webp)

At lr 2e-4 the best alpha was 4 "at every rank"; at 6e-5 it was 64; at 2e-5 it was 192. His explanation: the effective backbone learning rate is alpha/r times the head learning rate, so alpha and lr are one knob, not two. So copy a row whole; mixing alpha from one row with lr from another breaks it. WSDM 3rd adds the failure mode at the edge: lr 1e-4 gave 0.7051 CV, and "lr 1.4e-4 collapses to 0.508", which is chance on a binary task. For the scaling factor's mechanics, see the [LoRA and QLoRA explainer](/blog/machine-learning/reinforcement-learning/lora-qlora-for-rlhf).

## 4. Max length and truncation: which tokens you throw away

Arena inputs are a prompt plus two long responses, often multi-turn. Every top team made an explicit truncation decision, and they disagreed.

![Three truncation strategies on a multi-turn conversation: left truncation, middle cut per field, and a per-field token budget](/imgs/blogs/kaggle-llm-as-classifier-fine-tuning-recipes-6.webp)

- **Keep the end.** 5 of 10 LMSYS writeups explicitly kept the end of the conversation. LMSYS 9th set `truncation_side="left"` at 1536 tokens in training and 1792 at inference; 16th's improvement ladder ended with "truncate left" as the step that reached 0.885 log loss.
- **Cut the middle of each field.** WSDM moved this way: 1st used "proportional truncation from the middle of each field", 2nd truncated "in the middle in proportion to length", and 12th inserted a `(snip 123 chars)` marker. WSDM 5th is the one team that measured the choice: mid-cut 0.6938 against left-cut 0.6907.
- **Budget per field.** LMSYS 4th trained at 2048 and inferred at 3072; when the last round overflowed, the remaining budget split 20% prompt, 40% A, 40% B, and the round was dropped if fewer than 80 tokens remained.

Two patterns repeat. **Infer longer than you train**: LMSYS 4th (2048 to 3072), LMSYS 9th (1536 to 1792), WSDM 5th (4096 to 8192 with CPU offload), Science Exam 13th (256 to 768), DAIGT 3rd (256 to 1512). And longer is not always better: WSDM 6th found "3072 gave nothing" over 2500; DAIGT 13th said "longer hurt LB". In PII, 5th found longer won on private while shorter won CV and public.

## 5. Swap TTA: averaging out position bias

A pairwise judge tends to prefer whichever answer it reads first. Scoring both orders and averaging cancels that bias.

```python
# Illustrative toy, not any team's code: a 3-way judge with a first-position bias.
import numpy as np

def softmax(z):
    z = np.asarray(z, dtype=float) - np.max(z)
    e = np.exp(z)
    return e / e.sum()

def predict(first, second):
    """Returns [first wins, second wins, tie]; favours whoever is shown first."""
    quality = {"A": 1.0, "B": 0.4}
    bias = 0.5
    return softmax([quality[first] + bias, quality[second], 0.2])

p_ab = predict("A", "B")              # [A wins, B wins, tie]
p_ba = predict("B", "A")              # [B wins, A wins, tie] in swapped order
p_ba_flipped = p_ba[[1, 0, 2]]        # map back to [A wins, B wins, tie]
p_tta = (p_ab + p_ba_flipped) / 2

print("original order       ", p_ab.round(3))
print("swapped, flipped back", p_ba_flipped.round(3))
print("swap-TTA average     ", p_tta.round(3))
print("position-bias gap on A:", round(float(p_ab[0] - p_ba_flipped[0]), 3))
```

```console
original order        [0.623 0.207 0.17 ]
swapped, flipped back [0.425 0.384 0.191]
swap-TTA average      [0.524 0.296 0.18 ]
position-bias gap on A: 0.198
```

The same pair gets P(A wins) of 0.623 or 0.425 depending only on order, a 0.198 swing.

![Swap TTA with toy numbers: score both orders, flip the swapped labels back, average; top WSDM teams ran the second pass only on uncertain rows](/imgs/blogs/kaggle-llm-as-classifier-fine-tuning-recipes-5.webp)

All 10 LMSYS writeups used A/B swapping in some form, but not all as TTA. LMSYS 2nd put both orders in one optimiser step ("full swap", worth 0.003); 5th and 9th trained a second model on swapped data instead; 3rd's flip TTA was worth about 0.007. WSDM teams rationed it: 1st ran the swapped pass on "the 33% most uncertain rows", 3rd on 25%, 12th on the "40% of rows closest to 0.5". WSDM 6th found "TTA (no effect for sequence classification)".

Multi-option tasks use order TTA instead (4 of 15 Eedi writeups, including 2nd's reversed candidate order). It is not free money. Science Exam 1st found option-order TTA "helped but was shaky, never beat binary", and MAP 10th lists TTA as not working.

## 6. Pseudo-labels and distillation from a bigger model

The most transferable recipe: train a model too big for the notebook, use it to label data, train one that fits on those soft labels.

![Out-of-fold teacher distillation: five 70B-class teachers make soft labels, a 9B or 14B student learns them, then is quantised to fit the notebook](/imgs/blogs/kaggle-llm-as-classifier-fine-tuning-recipes-7.webp)

LMSYS 1st trained Llama-3-70B and Qwen2-72B per fold, collected out-of-fold class distributions, and distilled them into Gemma-2-9B, averaged the five fold adapters into one and quantised to GPTQ 8-bit. WSDM 1st did the same with five Qwen2.5-72B teachers into Qwen2.5-14B, with a loss of "weighted CE + KL + cosine". WSDM 3rd found something cheaper: "self-distillation from the 14B's own logits matched 72B distillation", with temperature 5 and soft-loss weight 0.9. In MAP, 12th called distillation "the biggest single gain (+0.003 to +0.005)".

It also failed most often when copied: among 10 LMSYS writeups, 1 success and 3 failures (3rd "couldn't get it right"; 13th, distilling from Gemma-27B, "should have tried distilling from an ensemble of large models"; 19th found no working method). WSDM 6th found it helped Gemma-2-9B but not Qwen2.5-14B.

Pseudo-labels are the sibling: WSDM 6th's 560k from a Gemma-2-9B labeller alone reached above 0.693; Jigsaw 11th moved private 0.92579 to 0.92734 with hard test pseudo-labels; DAIGT 3rd took private 0.893 to 0.927 with about 1k confident test pseudo-labels. Yet LMSYS 13th, Jigsaw 8th and Jigsaw 17th list it as not working, and LMSYS 3rd found it "loosened CV-LB correlation". The theory behind the loss terms is in the [LLM distillation deep-dive](/blog/machine-learning/large-language-model/distillation-in-llm).

## 7. CV schemes: what the winners trusted

| Competition, team | Split | Stated reason or result |
|---|---|---|
| LMSYS 1st | 5 folds | CV and LB "very consistent" |
| LMSYS 2nd | StratifiedGroupKFold by prompt, 20% held out | Same prompt never in train and val |
| Eedi 1st | GroupKFold on ConstructId | QuestionId "too optimistic", SubjectId "too pessimistic" |
| Eedi 5th | GroupKFold(SubjectId), 5 folds | CV 0.626 vs LB 0.633 |
| MAP 1st | 5 folds x 5 seeds, hardest fold kept | "Trust loss over MAP@3" |
| MAP 5th | StratifiedGroupKFold by StudentExplanation | CV-LB correlation "almost completely disappeared" above about 0.947 |
| Jigsaw 1st | Public LB as validation | Test examples were the training data |
| DAIGT 4th, 8th, 13th | Effectively none | "they all had perfect CV scores" |

The group key matters most: Eedi 1st tried three granularities and picked the middle one, guarding against near-duplicates across the split (see [cross-validation done wrong](/blog/machine-learning/debugging-training/cross-validation-done-wrong)). Where no honest split existed, teams said so and leaned on seeds or the public board.

## 8. Case studies

**1. Eedi 9th: head versus yes-logit, confounded.** Study Hard ran a 14B binary classifier and a Qwen2.5-32B causal LM reading the "yes" token logit. The 32B yes-logit model scored 0.571 private against 0.542 for the classifier, and the 14B classifier was dropped for speed. That is not "token logits beat heads by 0.029": the models differ in size. The only clean same-size comparison in the set is MAP 7th, who found the two readouts "scored similarly".

**2. WSDM 4th: picking quantisation by CV.** KiMi Lab scored the same student both ways. GPTQ 8-bit: CV 0.713, public 0.703, private 0.714. AWQ 4-bit: CV 0.710, public 0.708, private 0.709. Public favoured AWQ by 0.005; they picked GPTQ on CV and it won private by 0.005. LMSYS 9th likewise chose 8-bit ("no score loss"). For the quantisation formats themselves, see [quantization in LLMs](/blog/machine-learning/large-language-model/quantization-in-llm).

**3. DAIGT: when the 7B classifier loses.** 12th trained a Mistral-7B classifier on 82k essays at more than 10 hours per epoch on a 3090 Ti and got LB 0.642; 13th's 7B fine-tune reached 0.69 private. 1st place's Mistral-7B LoRA worked (best single model 0.984 private) on a far larger and more diverse generated corpus. The classifier recipe was the same; the data was not.

## When to reach for an LLM classifier, and when not to

Reach for a decoder classifier when the label needs reasoning over long, varied text and a 70B-class teacher can label more data than you have. Use label-token logits if you serve with vLLM, a head if you serve with plain transformers. Copy settings rows whole, retune alpha with the learning rate, and spend TTA only on uncertain rows.

Skip it for span tagging (PII) or surface-statistics signals (DAIGT), where encoders won. The sibling posts *Fitting big models in small GPUs*, *The test-time compute playbook* and *Metric exploits and shake-ups* go deeper on quantisation, TTA and leaderboard risk.

## Sources

- **LMSYS Chatbot Arena** (`kaggle.com/competitions/lmsys-chatbot-arena/discussion/<id>`): 1st 527629, 2nd 527685, 3rd 527766, 4th 529067, 5th 527669, 9th 527704, 13th 540876, 16th 527596, 18th 527595, 19th 528288 and 527651.
- **WSDM Cup Multilingual Chatbot Arena** (`kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/<id>`): 1st 569902, 2nd 567948, 3rd 567584, 4th 568522, 5th 567856, 6th 567600, 7th 567589, 12th 567614.
- **Kaggle LLM Science Exam** (`kaggle.com/competitions/kaggle-llm-science-exam/discussion/<id>`): 1st 446422 and 446240, 2nd 448256, 5th 446293, 6th 447647, 7th 447155, 13th 446231.
- **LLM Detect AI Generated Text** (`kaggle.com/competitions/llm-detect-ai-generated-text/discussion/<id>`): 1st 473295 and 470121, 3rd 470333, 4th 470179, 12th 470396, 13th 470593.
- **PII Data Detection** (`kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/<id>`): 5th 497306.
- **Eedi Mining Misconceptions**: [1st](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution), [2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/cqyr-2nd-place-solution), [5th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/ebi-ktr-5th-place-solution), [6th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/nikita-babich-6th-place-solution), [7th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/terekaerumasahmet-private-7th-public-2nd-place-sol), [8th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/yannan-chen-8th-place-solution), [9th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/study-hard-private-9th-public-7th-place-solution), [10th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/sayoulala-10st-place-solution-summary), [14th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/eeee-rsi-14th-place-solution).
- **MAP Charting Student Math Misunderstandings**: [1st](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/1st-place-solution), [2nd](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/map2025-private-and-public-2nd), [5th](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/5th-place-solution), [7th](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/private-7th-public-10th-place-solution), [8th](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/8th-place-solution), [9th](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/9th-place-solution), [10th](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/10th-place-solution), [12th](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/12th-place-solution).
- **Jigsaw Agile Community Rules** (`kaggle.com/competitions/jigsaw-agile-community-rules/discussion/<id>`): 1st 613305, 5th 613421, 6th 613150, 8th 613195, 11th 613506, 15th 613611, 17th 613071.

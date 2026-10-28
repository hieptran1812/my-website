# LLM - Detect AI Generated Text (DAIGT, 2023-24): top-20 addendum

- **Slug:** `llm-detect-ai-generated-text` (competition id 61542, forumId 3863365, 4,358 teams on the API board)
- **Existing deep coverage:** `.claude/plans/kaggle-llm-writeup-bank-part1.md`, H2 `## 2. LLM - Detect AI Generated Text
  (2023-24) -- "DAIGT"` (lines ~248-431): ranks 1-3 in depth, 4th and 5th short, public threads (Radek, Broad PERSUADE
  match, Darek DAIGT V2/V4, James Day 465882, Ertugrul BPE tokenizer, CPMP "curse of early success"), shake-up overview.
  This file adds ranks 6-20 and the facts the bank lacks for 4, 5 and 9.
- Source of all ranks/scores: `competitions.LeaderboardService/GetLeaderboard` (pulled 2026-10-28). Public = 46% of test.
- Medal note from the bank: gold = top 18.

## Top-20 table (private LB, API)

| Priv | Team | Members (userName) | Private | Public rank | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | (4 emoji) | ubamba98, conjuring92, nbroad | 0.987824 | 9 | /discussion/473295 (+470121) | (bank) 160k diverse "datamix", Mistral-7B QLoRA + DeBERTa + Ghostbuster + unsupervised, rank ensemble |
| 2 | Guanshuo Xu | wowfattie | 0.983412 | 17 | /discussion/470395 | (bank) DeBERTa "pretrained" on 500k SlimPajama pairs, then essays from student-mimicking fine-tuned LLMs |
| 3 | nlp team | ivanisaev, evilpsycho42, muhammad4hmed, dmitriikhizbullin, evgeniimaslov2 | 0.974994 | 15 | /discussion/470333 | (bank) 12 DeBERTa-v3-large (some on 0.5-1.2M Pile/SlimPajama continuations from ~35 LLMs) + TF-IDF with pseudo-labels |
| 4 | Ertugrul & Chase | chasembowers, datafan07 | 0.973915 | 3 | /discussion/470179 (+471934) | (bank) BPE-TF-IDF + typo augmentation, Mistral-7B token-prob features + pseudo-labels, Longformer source clf, DeBERTa 700k |
| 5 | Linguistic Ninjas | tarobxl, poskok, hirokitanio, jsday96, domdejonge | 0.972147 | 22 | /discussion/470093 | (bank) DeBERTa + Mamba on 1.7M rows; test-time distillation into short-context students |
| 6 | Davide Cozzolino | davidecozzolino | 0.969132 | 2995 | /discussion/471831 | Phi-2 entropy/surprisal statistics (5 features) -> One-Class SVM trained on human essays only |
| 7 | Hao Mei | tailen | 0.965133 | 2357 | /discussion/470643 | data from non-instruction-tuned 7B base models (400k SlimPajama + 25k PERSUADE-style); one DeBERTa-v3-large |
| 8 | Abdullah Meda | abdullahmeda | 0.956501 | 1031 | /discussion/470224 | GPT-2 perplexity + GLTR rank features, VotingClassifier, 800k-row dataset; 10-minute run |
| 9 | LLMLab | darraghdog, tivfrvqhs5, christofhenkel | 0.947710 | 1 | /discussion/470255 | TF-IDF + reverse-engineered deobfuscation + 2 DeBERTa-v3-large + 1 RoBERTa on 250k-700k diverse rows |
| 10 | IC2 | jonathanchan | 0.947353 | 1880 | no writeup found | - |
| 11 | Data Driven Poets | teitlax, wngauv | 0.944266 | 1033 | no writeup found | - |
| 12 | Uesugi Erii | uesugierii | 0.942678 | 168 | /discussion/470396 | 820k falcon-refinedweb human/continuation pairs, DeBERTa-v3-large; 0.85 TF-IDF : 0.15 DeBERTa |
| 13 | RB | rashmibanthia | 0.938988 | 41 | /discussion/470593 | transformers only: DeBERTa-xlarge + v3-large + RoBERTa-large on ~45k (DAIGT V4 + own GPT essays) |
| 14 | takaito | takaito | 0.936732 | 117 | /discussion/470193 | public TF-IDF notebook + untuned roberta-large-openai-detector as a feature, LGBM/XGB, per-prompt pseudo-labels |
| 15 | FakeDetectLab | shaquanquan, susannahsu, catlion6, yangzhenze, zylee0 | 0.935624 | 13 | no writeup found | - |
| 16 | 冲冲冲！ | syuan0330, liucaiyi | 0.934437 | 2635 | no writeup found | - |
| 17 | Magus | gmhost | 0.933942 | 7 | no writeup found | - |
| 18 | ai galvatron | kevinehsani, peirenc, junglebeastds, lazysheep277 | 0.933833 | 66 | /discussion/470061 | adversarially selected hard examples from public sets; 0.4 TF-IDF n-grams + 0.6 Mistral/Llama-7B ensemble |
| 19 | Darek Kłeczek | thedrcat | 0.932794 | 39 | /discussion/470181 | diverse blend: Ridge/MNB/SGD/CatBoost/LGBM TF-IDF + readability ranking + DeBERTa + Mistral-7B QLoRA |
| 20 | Shimataro | shimataro | 0.932606 | 44 | no writeup found | - |

Discussion URLs are `https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/<id>`. 14 of 20 have a
writeup (1-9, 12-14, 18, 19).

## Per-team synthesis

### Ranks 4, 5, 9: facts missing from the bank

- **4th, Ertugrul & Chase** (/discussion/470179): no reliable CV existed ("they all had perfect CV scores"), so final
  weights were set manually by "common sense", deliberately not by public LB. DeBERTa-v3 component: ~700k public rows,
  1 epoch (0.999 CV early), 1024 tokens, no early stopping, first layers frozen (Ertugrul in comments: CLS-token
  classification, many frozen encoder layers, batch 4, no quantisation, "lora didn't go well", ~30 h on a consumer GPU).
  Chase's component (/discussion/471934, full text on a Notion page that did not render for WebFetch): Mistral softmax
  statistics, e.g. "Global probability 10%" = 10th percentile of the log softmax probabilities, scored by AUC on a DAIGT
  V2 sample, then refined with test pseudo-labels.
- **5th**: bank is complete.
- **9th, LLMLab** (/discussion/470255): public #1 -> private #9. Novel parts named by the team: datasets of ~200k-700k
  rows from open LLMs of 7-70B plus commercial APIs, wide temperature/top_p, essay generation + paraphrase + completion
  prompts; **statistical and reverse-engineered deobfuscation** (+0.006 public / -0.001 private); cluster post-process
  (pairs of highly similar texts are almost always both generated, ~+0.001). Best single BERT-type model 0.96 public /
  0.915 private. Weights (darraghdog in comments): 20% deberta-v3-large on 160k/110k LLM/human, 10% roberta-large on the
  same, 10% deberta-v3-large on that plus ~500k SlimPajama continuations; **the remaining ~60% on TF-IDF models**. David
  Austin on deobfuscation: remove *systematic* mistakes (e.g. pyspellchecker replacing one letter throughout) but
  *inject* random mistakes, because random noise may be distributed differently in train and test. Their 0.99 public
  came from TF-IDF + DeBERTa blending "and magic happened"; it did not hold on private.

### 6th - Davide Cozzolino (solo) - /discussion/471831; GitHub davin11/entropy-based-text-detector (+ Documentation.pdf)

Public 2,995 -> private 6, the largest climb in the gold zone. An assistant professor in multimedia forensics, he spent
"only four or five days". **One-class training**: an RBF One-Class SVM (`gamma='auto', nu=0.05` in `train.py`) fit
**only on the human essays in the competition's `train.csv`**; no generated text was used for training. His forensics
prior: training only on authentic data generalises better and resists overfitting (cites his Noiseprint camera work).
Features: token-level **entropy** (expected information) and **surprisal** (information content) vectors from
**microsoft/phi-2** (MIT licence; it beat the other LLMs he tried), summarised as statistics; 38 candidate features,
selected by single-feature AUC on DAIGT V2. Final 5 features: median entropy, median surprisal, mean characters per word,
std of surprisal, 5th percentile of entropy. Submission history from the doc (public / private): 7 feats 0.747 / 0.627;
2 feats 0.767 / 0.662; 3 feats 0.873 / 0.902; **5 feats 0.873 / 0.968**. Feature extraction 18 min for 1,378 essays on a
P100; whole pipeline ~20 min. The doc prints the best private as 0.968132 vs the API's 0.969132 (doc typo or different
submission; API value used above). Distinctive idea: anomaly detection on human text rather than a human-vs-AI classifier.

### 7th - Hao Mei (solo, first competition) - /discussion/470643; GitHub Tailen/Kaggle-Detect-AI-Generated-Text-7th-Solution

Public 2,357 -> private 7. Distrusted TF-IDF from the start. **"Magic sauce": generate only with non-instruction-tuned
models**, reasoning that the three host-provided generated essays lack the ChatGPT answer format and that RLHF causes
mode collapse (cites a LessWrong post). Data: ~400k non-essay continuations from SlimPajama plus 25k PERSUADE-style
essays, from vanilla **Falcon-7B, Mistral-7B, Llama-2-7B**, temperature 0.8-1.2, top_p 0.8-0.98, frequency penalty 0-0.3.
Essay prompt is a few-shot "fake exam sheet": "(100 points) [instruction] ... around 500 words", Name / Grade / Date /
title + one real PERSUADE essay, then a new header with a same-gender name, same grade, for the model to continue; 10
generations per PERSUADE essay, filtered by embedding similarity and Levenshtein distance to the seed, length and
repetition. Model: **one DeBERTa-v3-large**, train 512 tokens, infer 1024; ~8 h per run on a 3090. Training at 1024 was
worse. Final: predictions in the 40th-60th percentile replaced by a TF-IDF + SGD model (CV up slightly, LB flat).
Ablation (public / private): v3-base PERSUADE-only 0.869 / 0.875; v3-base + SlimPajama 0.921 / 0.920; v3-base 1024
0.910 / 0.901; **v3-large 512 0.942 / 0.965**; v3-large 1024 0.922 / 0.957. Failed: semantic-block bi-LSTM (0.763);
neighbour-difference LSTM (0.685); Robust Encodings against char attacks; sliding-window logit averaging (0.929);
MLM pretraining on test with deberta-v3-xsmall then fine-tune (0.689).

### 8th - Abdullah Meda (solo) - /discussion/470224; code kaggle.com/code/abdullahmeda/8th-lb-solution-ppl-gltr-features; HF Space abdullahmeda/detect-ai-text

Public 1,031 -> private 8. **Zero fine-tuning of neural models in the final**: classic detection-paper features
(arXiv 2301.07597, GLTR arXiv 1906.04043). Text- and sentence-level **perplexity** from GPT-2 small/medium/large (not XL)
plus **GLTR Test-2** counts of tokens in top-10 / top-100 / top-1000 / 1000+ ranks; sklearn VotingClassifier;
features/models persisted with joblib; ~10 min per submission. The decisive switch was data: DAIGT V2 44k gave < 0.7 AUC;
starblasters8's 800k dataset gave the jump. Ablation (public / private): 44k PPL GPT-2 small 0.674 / 0.506; 800k PPL
0.800 / 0.656; 800k GLTR 0.910 / 0.918; 800k PPL+GLTR small 0.923 / 0.926; medium 0.932 / 0.953; **large 0.938 / 0.956**.
CV "always close to 1", useless. Hurt: test-vocab GPT-2 tokenizer, T5 deobfuscator, log-likelihood features,
PySpellChecker, readability scores. Did not work: fine-tuning chatgpt-detector-roberta / roberta-large on 800k (CV ~0.98,
LB 0.793 / 0.674 private). Best *unselected* sub: 50/50 blend with TF-IDF = 0.964 private.

### 12th - Uesugi Erii (solo) - /discussion/470396

Kept PERSUADE-prompt generations only as local CV after they trained poorly; inspired by Stable Diffusion Image-to-Prompts,
went for breadth: **tiiuae/falcon-refinedweb**, deduplicated across chunks with `bge-base-en-v1.5` (add only if
similarity < 0.7), > 768 tokens -> ~600k texts. For each text, a random 128-384-token prefix is the prompt; the true
remainder is human (0) and LLM continuations are generated (1); mainly **Mistral and Llama-2** variants (others judged
low-quality), > 170-token filter -> 410k generated, 820k total at 1:1. Training truncates both texts to
`randint(128, min(len_human, len_llm))` so length cannot leak the label. DeBERTa-v3-large, 512 train/infer, 3 epochs,
mean pooling, LLRD. TF-IDF = batprem's "Analyse edge cases" notebook unmodified (he explicitly frames test-vocab TF-IDF as
feature leakage worth using). Ensemble **TF-IDF 0.85 : DeBERTa 0.15**. Abandoned: fine-tuning an LLM to mimic PERSUADE
(quality dropped), adversarial generate-select loop (compute), Mistral-7B classifier on 82k (>10 h/epoch on a 3090 Ti,
LB 0.642). Did not work: soft labels, spelling-error classification, spelling correction (distance- and NN-based),
prompt engineering for DeBERTa.

### 13th - RB / Rashmi B (solo) - /discussion/470593

**Transformers only**; 0.957 public / 0.938 private, best private was also selected (would have been ~2000th on public).
Data ~45k: Darek's DAIGT V4 ("the magic dataset") + own essays from a system prompt asking for an "average non native
english speaker grade 6-12 student", 450-650 words, "a minimal amount of typos", conditioned on PERSUADE demographic
fields, balanced across the 7 prompts; dedup vs PERSUADE/train with `bge-base-en-v1.5` cosine. Models, all max_len 512
(longer hurt LB), 5 folds: **DeBERTa-xlarge** (0.92 private, 50% weight), DeBERTa-v3-large (0.847, 25%), RoBERTa-large
(0.849, 25%). CV "practically none". Her TF-IDF+DeBERTa sub was 0.97 public / 0.927 private. Did not work: data from
Mistral-7B-Instruct-v0.2 fine-tuned on PERSUADE (in hindsight should have used base Mistral, per 2nd place); Electra and
other DeBERTa variants; MLM pretraining; LGBM on log probs (looked good on private, under-explored); 7B fine-tune
(0.69 private); no HPO possible without validation.

### 14th - takaito (solo) - /discussion/470193

Base: batprem's "LLM DAIGT Analyse edge cases" TF-IDF notebook. Two additions: (1) **`roberta-large-openai-detector`
used untuned** (0.813 public / 0.839 private alone; tuning overfit), scored on the first 512 and last 512 tokens, fed as
features with the word-frequency features into LightGBM and XGBoost, ensembled with the public notebook -> 0.965 / 0.930;
(2) **pseudo-labels per `prompt_id`**: top 10% most confident as generated and top 7.5% as human (BERT was better at
spotting generated text). Smaller tweaks: training weights, noisy-data filtering, feature reduction for speed.

### 18th - ai galvatron (4 members) - /discussion/470061

40% TF-IDF (average of 2-, 3-, 4-, 5-gram models), 60% LLM ensemble of mostly 7B Mistral/Llama models. **Adversarial
selection of training data**: train a small LM on 10% of the pooled public datasets, predict the remaining 90%, keep the
hardest examples, mix 50/50 with PERSUADE. Found late; final-day hastily trained models; they think a stronger selector
model and larger LoRA rank would help.

### 19th - Darek Kłeczek (solo) - /discussion/470181

Author of the DAIGT V2/V4 datasets everyone used. Phases: data generation (never found CV/LB correlation) -> single
models (only from January) -> last-week ensemble built for robustness -> last-day Levenshtein spelling correction
(piotrkoz's code). Final blend: Ridge on V2 + V4 sample with test tokenisation + spelling fix; MNB/SGD/CatBoost from top
public notebooks; LightGBM with source-weighted sampling from V3 and a two-step 10k-feature selection; a ranking from
Flesch-Kincaid grade / reading ease; DeBERTa-v3-large on a V4 subset; Mistral-instruct-7B QLoRA on V4 (hotchpotch
pipeline). Probed public LB little ("good in hindsight"). Failed: Mistral reward model on (original, generated) pairs;
Ghostbuster re-implementation with Llama probabilities (the 1st-place team made this work); quantised unigram/trigram
probability models; DeBERTa pairwise ranking + Bradley-Terry; token-level NER detector (interpretable, weak on LB); "ridge
forest".

### No writeup found: 10th IC2, 11th Data Driven Poets, 15th FakeDetectLab, 16th 冲冲冲！, 17th Magus, 20th Shimataro

All 32 forum pages (~630 topics) scanned; no solution topic and no comments by these members in the 14 solution threads
fetched. One web search found nothing. Notable: IC2 (1,880 -> 10), Data Driven Poets (1,033 -> 11) and 冲冲冲！ (2,635 ->
16) are huge climbers, a pattern shared by the LM-feature / general-data teams (6th, 7th, 8th) whose writeups exist;
their methods are unknown.

## Technique frequency matrix

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Own large-scale generation (>100k) | Y | Y | Y | N | Y | N | Y | N | Y | ? | ? | Y | N | N | ? | ? | ? | N | N | ? |
| Out-of-domain web text pairs (Pile/SlimPajama/RefinedWeb) | Y | Y | Y | N | Y | N | Y | N | Y | ? | ? | Y | N | N | ? | ? | ? | N | N | ? |
| Base (non-instruct) or student-mimicking generators | Y | Y | ? | N | ? | N | Y | N | ? | ? | ? | Y | N | N | ? | ? | ? | ? | N | ? |
| Typo / obfuscation augmentation or deobfuscation | Y | N | Y | Y | ? | N | N | N | Y | ? | ? | N | N | N | ? | ? | ? | ? | Y | ? |
| Test-vocabulary TF-IDF in final | N | N | Y | Y | N | N | Y(small) | N | Y | ? | ? | Y | N | Y | ? | ? | ? | Y | Y | ? |
| DeBERTa(-v3-large) classifier | Y | Y | Y | Y | Y | N | Y | N | Y | ? | ? | Y | Y | N | ? | ? | ? | ? | Y | ? |
| Fine-tuned 7B LLM classifier (LoRA) | Y | N | N | N | N | N | N | N | N | ? | ? | N | N | N | ? | ? | ? | Y | Y | ? |
| LM token-probability / perplexity features | Y | N | N | Y | N | Y | N | Y | N | ? | ? | N | N | Y(detector) | ? | ? | ? | N | N | ? |
| Pseudo-labels on test | Y | N | Y | Y | Y | N | N | N | N | ? | ? | N | N | Y | ? | ? | ? | N | N | ? |
| Test-time training / distillation | Y | N | N | N | Y | N | N | N | N | ? | ? | N | N | N | ? | ? | ? | N | N | ? |
| Rank-average ensemble | Y | N | N | N | N | N | N | N | N | ? | ? | N | N | N | ? | ? | ? | N | N | ? |
| No neural fine-tuning at all | N | N | N | N | N | Y | N | Y | N | ? | ? | N | N | Y | ? | ? | ? | N | N | ? |

Counted summary (14 teams with writeups):
- 8/14 generated their own data at > 100k scale (1, 2, 3, 5, 7, 9, 12 and 4th used ~700k public rows); **all of the top 5
  except 4th** did, and 4th used the largest public merge.
- 7/14 used out-of-domain web-text human/continuation pairs (1, 2, 3, 5, 7, 9, 12), inspired by James Day's /discussion/465882.
- 11/14 used a DeBERTa-family classifier; only 3 fine-tuned a 7B LLM classifier (1, 18, 19).
- 7/14 kept test-vocabulary TF-IDF in the final; of those only 3rd and 4th are in the top 5.
- 4/14 relied on LM probability features (1, 4, 6, 8); two of them (6, 8) used **no** fine-tuned neural model.
- 5/14 pseudo-labelled the test set (1, 3, 4, 5, 14).
- 1/14 rank-averaged (1st); AUC makes it natural, few did it.

## Consensus vs divergence

**Table stakes:** no trustworthy CV (nearly every writeup says so: 4, 8, 13, 19 explicitly), so robustness came from data
diversity and from not trusting public LB; DAIGT V2/V4 as the base; DeBERTa-v3-large as the default classifier.

**What separated the top 5 from 6-20:** scale and diversity of *own* generated data (hundreds of thousands to 1.7M rows,
many generator families and decoding settings) plus an explicit domain-adaptation step (student-mimicking generators
2nd, test-time distillation 5th, pseudo-labels 1st/3rd/4th). Ranks 6-8 instead reached gold with **cheap
general-purpose LM statistics** (entropy/perplexity/GLTR) and no test fitting; ranks 9-20 leaned more on
test-vocab TF-IDF and smaller data.

**Contradictions:**
- *Test-vocab TF-IDF.* 12th: "tantamount to leaking ... there's no reason not to employ it", final weight 0.85
  /discussion/470396; 9th put ~60% on TF-IDF /discussion/470255. Versus 8th: "suspected it was overfitting to the public
  test set" /discussion/470224, 7th distrusted it /discussion/470643, 13th: transformers alone beat TF-IDF and the blend
  /discussion/470593. Outcome: the TF-IDF-heavy teams fell (LLMLab 1 -> 9, Secret Sauce 2 -> 21, CPMP 4 -> 163 per bank).
- *Fixing typos.* 3rd (T5 deobfuscator on >15 errors, bank), 9th (reverse-engineered deobfuscation, +0.006 public /
  -0.001 private), 19th (Levenshtein correction) used it; 8th found T5 deobfuscator and PySpellChecker hurt
  /discussion/470224, 12th found every spelling fix failed /discussion/470396.
- *Fine-tuning a generator on PERSUADE.* 1st and 2nd made it work (bank; 2nd via H2O LLM Studio on base models); 12th
  (quality dropped) and 13th (Mistral-7B-Instruct) failed; 13th's own diagnosis: use the base model, not Instruct.
- *Fine-tuned 7B classifier.* 1st's best single model (Mistral-7B QLoRA 0.984 private, bank); 12th (0.642 LB) and 13th
  (0.69 private) failed with it.
- *MLM pretraining on test.* 1st used it for a from-scratch DeBERTa-small (bank); 7th (0.689) and 13th failed.
- *Context length.* 7th and 13th: 512 beat 1024 in training; 4th trained at 1024; 5th used 128/256-*character* students.

## Aggregated "what did not work"

- Test-vocab tokenizer / TF-IDF tuned to public LB - 8th (hurt), CPMP (bank); 12th left TF-IDF untouched to avoid local overfit.
- Spelling correction / deobfuscation (T5, PySpellChecker, distance- and NN-based) - 8th, 12th.
- Spelling-error-based classification - 12th.
- Fine-tuning roberta/chatgpt-detector on large data - 8th; tuning roberta-openai-detector - 14th (overfit).
- Fine-tuning a 7B LLM classifier - 12th, 13th; LoRA on DeBERTa - 4th ("didn't go well").
- Fine-tuned (instruct) LLM generators mimicking PERSUADE - 12th, 13th.
- MLM pretraining (on test or train) - 7th, 13th.
- Training at 1024 tokens / sliding-window inference - 7th; > 512 tokens - 13th.
- Bi-LSTM over semantic blocks; Robust Encodings - 7th.
- Soft labels; prompt engineering for DeBERTa - 12th.
- Log-likelihood features, readability scores - 8th (but 19th used readability as one blend member).
- Mistral reward model on pairs; Ghostbuster re-implementation; n-gram probability models; pairwise ranking +
  Bradley-Terry; token-level NER detector; ridge forest - 19th.
- Electra / other DeBERTa variants; HPO without validation - 13th.
- Hard-example mining against public LB - 3rd (0.927 public / 0.845 private, bank).

## Shake-up

From the API (public -> private), top 20:
- Massive climbs: Davide Cozzolino 2,995 -> 6, 冲冲冲！ 2,635 -> 16, Hao Mei 2,357 -> 7, IC2 1,880 -> 10, Data Driven Poets
  1,033 -> 11, Abdullah Meda 1,031 -> 8, Uesugi Erii 168 -> 12, takaito 117 -> 14. Explanations in writeups: 6th
  (one-class training on human text only, no generated data to overfit), 7th (refused TF-IDF, base-model generations), 8th
  (general PPL/GLTR features). A thread asked "There must be a leak or secret sauce" (/discussion/470063); the writeups
  show no leak, only generalisation.
- Falls: LLMLab 1 -> 9 (their best selected private sub used their own data, not public-script data: "Just as well we
  did that"), Magus 7 -> 17, FakeDetectLab 13 -> 15, Secret Sauce 2 -> 21 (outside top 20; 0.986 public), Psi 6 -> 22.
- Gap to note: 13th says her best private equalled her selected sub; 8th's best unselected (0.964) would have been ~7th.

## UNVERIFIED

- Methods of 10th, 11th, 15th, 16th, 17th, 20th: unknown.
- Chase Bowers' Notion write-up (4th, LLM component) could not be rendered; feature list beyond "10th percentile of log
  softmax probability" is not verified.
- Davide's doc lists best private 0.968132 while the API gives 0.969132.
- 18th's exact models, LoRA ranks and selector model are not stated.
- 9th's deobfuscation notebook ("will be published") was not located.
- Matrix cells for ranks 1-5 come from the bank summaries; "?" means not stated there.

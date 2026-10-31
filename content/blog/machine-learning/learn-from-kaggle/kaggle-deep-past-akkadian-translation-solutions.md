---
title: "Deep Past Challenge: LLMs as data engineers, ByT5 as the translator"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "Kaggle's Akkadian-to-English translation challenge was won by teams that used frontier LLMs to turn scanned scholarship into clean parallel text, then trained a small byte-level ByT5 on it. Here is what all 17 documented top-20 teams did, and what transfers."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "machine-translation", "byt5", "low-resource-nlp", "synthetic-data", "ocr", "minimum-bayes-risk", "data-engineering"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 22
---

> **TL;DR.** The Deep Past Challenge asked for English translations of Old Assyrian clay tablets, transliterated into Latin script, from a notebook with 9 hours of compute and no internet. DataTech Club won with 42.8957 private, ahead of KE WU (41.9311) and darraghdog with conjuring92 (41.5091). Every one of the 17 top-20 teams that published a writeup shipped **ByT5**, a small byte-level encoder-decoder, and at least 15 of those 17 used a **frontier LLM or vision-language model as the data engineer**: reading scanned books, splitting documents into sentence pairs, translating German and Turkish editions, and inventing synthetic drills. Only 2 of the 17 let a decoder LLM translate in the final submission. **Decisive lever: data.** **The one lesson:** when the labelled data is tiny but the unlabelled scholarship is large, spend your LLM budget building the training set, and spend your notebook budget on a small specialist that fits.

![The lever strip: Deep Past was decided by data built with LLMs, with candidate selection by MBR as the second lever](/imgs/blogs/kaggle-deep-past-akkadian-translation-solutions-1.webp)

Most competitions in this series are about getting a large model to reason inside a Kaggle box. Deep Past inverted that: the large models worked offline as OCR engines, aligners, translators and drill writers, and what entered the box was a 0.6B to 3.7B parameter ByT5 trained on what they produced. The series opener, [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work), explains the notebook format this post assumes.

## 1. The task in 60 seconds

![The budget card: 9 hours offline, about 4,000 hidden test sentences, roughly 1,500 translated documents to start from, and about 900 scholarly PDFs to mine](/imgs/blogs/kaggle-deep-past-akkadian-translation-solutions-2.webp)

Around four thousand years ago, merchants from Assur traded tin and textiles with Anatolia and wrote to each other on clay tablets, in Old Assyrian, a dialect of Akkadian. Tens of thousands survive. Most have been transliterated (each cuneiform sign written out as a Latin-script syllable), far fewer translated into English. The Deep Past Initiative ran this Kaggle competition from 2025-12-16 to 2026-03-23 to see whether machine translation could close that gap.

The input is one transliterated sentence. The output is its English translation. The host supplied a training file of document-level translations (about 1,500 rows per the Data page; 5th place counted 1,561), more than 8,000 transliterated texts with metadata but no translation, and the unprocessed text of roughly 900 scholarly publications, many of them scans of German and Turkish editions. A mid-competition data update changed the files, and 2nd place describes about 6,500 documents after it; which version each team trained on is unverified, so the numbers in this post are the teams' own.

It was a code competition: a notebook with internet off and a 9-hour limit, CPU or GPU; the writeups that mention hardware ran on two T4s. 14th place counts about 4,000 test sentences, 34% of which scored the public board. 2,674 teams made 68,346 submissions for USD 50,000 in prizes (15k, 10k, 8k, 7k, 5k, 5k for places 1 to 6, which sums to 50k).

![An illustrative tablet in Old Assyrian letter style: hyphenated syllables, a Sumerogram for silver, a sign index, diacritics and a gap marker, and the English that must preserve each of them](/imgs/blogs/kaggle-deep-past-akkadian-translation-solutions-3.webp)

The figure is a sample I constructed in the style of an Old Assyrian letter, not a row from the competition data. It shows what makes the input strange. Hyphens join the syllables of one word. Uppercase words such as KÙ.BABBAR are Sumerograms, signs borrowed from Sumerian and read as Akkadian words (here, silver). Numbers glued to syllables are sign indices that distinguish homophones. Diacritics carry meaning: š is not s. And `<gap>` marks where the clay is broken, which the translation must keep, because a scholar reading the output needs to know a number or a name is missing. Each test row is one sentence; 11th place measured test transliterations at 129 to 267 characters.

## 2. The metric, with a worked toy score

The score is the geometric mean of two corpus-level metrics computed with SacreBLEU, micro-averaged over the whole test set:

$$
\text{score} = \sqrt{\text{BLEU} \times \text{chrF++}}
$$

**BLEU** counts how many word 1-grams to 4-grams of the hypothesis appear in the reference, takes the geometric mean of those four precisions, and multiplies by a brevity penalty for outputs shorter than the reference. **chrF++** counts character n-grams (up to 6) plus word 1-grams and 2-grams, and combines precision and recall with an F-score that weights recall twice as much as precision. A word with one wrong accent still shares most of its character n-grams with the reference, so chrF++ gives partial credit where BLEU gives none. 

![The metric on a toy corpus of two sentences: dropping diacritics scores 73.9, swapping silver for tin scores 84.5, and a faithful paraphrase scores 37.0](/imgs/blogs/kaggle-deep-past-akkadian-translation-solutions-4.webp)

Here is the toy behind the figure. Two reference sentences, three systems:

```python
# Illustrative toy, not the host's scoring code: sqrt(BLEU x chrF++) with SacreBLEU.
import math
from sacrebleu.metrics import BLEU, CHRF

refs = ["Thus says Aššur-idī: say to Pūšu-kēn.",
        "I paid 1 mina of silver to your messenger."]

systems = {
    "A: drops diacritics": ["Thus says Assur-idi: say to Pusu-ken.",
                            "I paid 1 mina of silver to your messenger."],
    "B: wrong commodity":  ["Thus says Aššur-idī: say to Pūšu-kēn.",
                            "I paid 1 mina of tin to your messenger."],
    "C: paraphrase":       ["Aššur-idī says this: tell Pūšu-kēn.",
                            "I gave your messenger 1 mina of silver."],
}

bleu, chrf = BLEU(), CHRF(word_order=2)   # word_order=2 makes it chrF++
for name, hyps in systems.items():
    b = bleu.corpus_score(hyps, [refs]).score
    c = chrf.corpus_score(hyps, [refs]).score
    print(f"{name:22s} BLEU={b:5.1f}  chrF++={c:5.1f}  score={math.sqrt(b * c):5.1f}")
```

```console
A: drops diacritics    BLEU= 70.7  chrF++= 77.3  score= 73.9
B: wrong commodity     BLEU= 81.1  chrF++= 88.0  score= 84.5
C: paraphrase          BLEU= 23.3  chrF++= 58.8  score= 37.0
```

System B changed the meaning, a merchant paying in tin rather than silver, and it beat System A, which kept the meaning but dropped the accents on two names. System C is a perfectly good translation in a different word order and lost half its score. The metric rewards the reference's surface form: orthography, name spellings, phrasing. That explains most of what follows. Normalising transliteration, keeping diacritics byte-exact, and training on translations in the test references' house style were worth more than fluency.

## 3. What made it hard: 1,500 documents and a library of scans

Three constraints decided this competition.

**The labelled data was tiny and at the wrong granularity.** The training file paired whole documents with whole translations, while the test asked for single sentences. Splitting a document translation into sentence pairs is not mechanical: Akkadian puts the verb last and translations reorder clauses freely. 7th place aligned sentences by hand in a spreadsheet for hours before concluding the gain was small.

**The real data was locked in scans.** The roughly 900 publications held tens of thousands of translated tablets, but as page images and noisy OCR text, often with the transliteration on one page and the translation on the next, in German or Turkish, with superscripts, subscripts and editorial brackets. 1st place called the theme of its writeup "Data Quality Dictates Everything", and 2nd place wrote that "every point of improvement came from building a larger, cleaner training corpus".

**The box was small and offline.** Nine hours on two T4s, no API calls. A 70B translator was out of the question, and even the 32B and 72B Qwen2.5 models that Geremie Yeo post-trained finished 24th. Whatever you knew had to fit a model that could translate about 4,000 sentences, many times over, inside that budget.

The resolution most teams found: use the expensive, smart models offline to build the data, and ship a cheap, narrow model. ByT5 fit that shape. It reads raw UTF-8 bytes with no learned vocabulary, so š, í, subscript digits and editorial brackets never become odd subword pieces (see [how tokenizer choice shapes a model](/blog/machine-learning/large-language-model/designing-choosing-tokenizer-llm)). And it is an encoder-decoder from about 0.3B to 13B parameters, so a team could pick the largest size its budget allowed and still decode thousands of sentences on a T4.

## 4. The top-20 scorecard

Private and public scores are from the Kaggle API (2026-10-28). Seventeen of the 20 published a writeup. Two self-titled places are off by one from the API: vinpro's "20th" post is API rank 19 and Karl's "21th" is API rank 20, and this post uses the API ranks. A "17th Place Solution" by kirito1sasa matches no team in the API's top 60, so it is excluded.

| Rank | Team | Private | Public rank (score) | Move | Key idea | Source |
|---|---|---|---|---|---|---|
| 1 | DataTech Club | 42.8957 | 1 (41.6694) | = | Gemini-3.0-pro reads page images; 11 ByT5-XL, MBR | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684353) |
| 2 | KE WU | 41.9311 | 3 (41.0478) | up 1 | One ByT5-large; three LLM alignment pipelines | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684345) |
| 3 | How I Learned to Stop Worrying | 41.5091 | 2 (41.6451) | down 1 | Claude-written drills for pre-training; Qwen3-8B judge | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684425) |
| 4 | Wisdom and Vitaly | 40.8899 | 8 (39.9734) | up 4 | 14.5k Gemini pairs; mT5 + ByT5; chrF++ MBR | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/685925) |
| 5 | ebi & okumura | 40.8077 | 4 (40.4156) | down 1 | 4-phase VLM extractor; staged training; one ByT5-XL | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684847) |
| 6 | M & J & M | 40.7360 | 7 (40.0310) | up 1 | Dictionary-guided alignment; context sampling; 15-model MBR | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684231) |
| 7 | Yurnero | 40.4372 | 6 (40.2765) | down 1 | ~36k samples; single ByT5-XL | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684215) |
| 8 | Hrithik Reddy | 40.2797 | 5 (40.3472) | down 3 | Open-weight only; 350k noisy then 65k clean | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684329) |
| 9 | Eleftheria 14 | 40.1770 | 11 (39.5449) | up 2 | 30+ cleaning passes; gap augmentation; source-grounded MBR | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/685928) |
| 10 | Kelvin | 39.9717 | 19 (38.5589) | up 9 | Pre-train, SFT, pseudo-label; MADLAD in MBR | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684211) |
| 11 | kwa | 39.8929 | 12 (39.4983) | up 1 | Multilingual pre-training; metric-as-utility MBR | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684429) |
| 12 | Ha-ar-pì-im | 39.7225 | 15 (38.9239) | up 3 | NotebookLM extraction; template pairs; LoRA last | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684866) |
| 13 | DataBoom | 39.6175 | 22 (38.4482) | up 9 | Public corpora; test-time source repair | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/686956) |
| 14 | H&B&Z | 39.5518 | 21 (38.4536) | up 7 | Qwen3-14B; online learning on the test set | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684236) |
| 15 | It's showtime! | 39.2971 | 10 (39.6487) | down 5 | Kimi K2 extraction; Gemma teacher; ByT5 + Qwen3 | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684819) |
| 16 | John Doe #2 | 39.2743 | 14 (39.1694) | down 2 | no writeup | no writeup |
| 17 | lost in transliteration | 39.1371 | 16 (38.7083) | down 1 | no writeup | no writeup |
| 18 | JoJoImpact | 39.1357 | 20 (38.5569) | up 2 | no writeup | no writeup |
| 19 | vinpro | 38.9826 | 18 (38.6004) | down 1 | Denoising pre-training; soup; name fuzzy matching | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684381) |
| 20 | Karl | 38.9230 | 23 (38.3640) | up 3 | One ByT5-base; Gemini pseudo-labels +4.0 | [writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684267) |

![Public versus private rank for the top 20: the podium barely moved, while Kelvin, DataBoom and H&B&Z climbed 7 to 9 places from the middle of the board](/imgs/blogs/kaggle-deep-past-akkadian-translation-solutions-5.webp)

The board is tight: 1st beat 2nd by 0.96 points, and the top 20 spans 3.97 (42.8957 minus 38.9230). The shake-up was moderate: the podium held, and the big moves happened in the middle. claw fell from 13th public to 21st private and out of this table.

## 5. The top-20 landscape

![The top-20 technique matrix: all 17 documented teams shipped ByT5 and 15 used an LLM to build training data, but only 2 let an LLM translate](/imgs/blogs/kaggle-deep-past-akkadian-translation-solutions-6.webp)

Counts are over the 17 teams with a writeup; "?" cells make them floors. Read it top to bottom as the recipe:

- **ByT5 in the final submission: 17 of 17** (largest size: 8 XL, 4 large, 5 base).
- **An LLM or VLM built the training data: at least 15 of 17.** Gemini in 8 teams, plus Claude, GPT, Qwen3.5-Plus, Kimi K2, DeepSeek, GPT-OSS-120B and NotebookLM; 8th and 15th used open-weight models only.
- **A noisy pre-training stage before clean fine-tuning: 9 of 17.** Pseudo-labelled untranslated texts: 9. Synthetic data of some kind: 8.
- **Multi-model ensembles: 9 of 17. MBR selection: 8.** Weight averaging shipped by 5, while 2 others say soups failed.
- **A decoder LLM translating in the final ensemble: 2 of 17** (14th and 15th). One more (3rd) used an LLM only to pick between ByT5 outputs.
- **RL, DPO or minimum risk training tried: 5 of 17.** It helped one team, 14th, on a Qwen LLM with the method unstated.

**Table stakes.** Byte-level ByT5; a training set rebuilt or expanded far beyond the starting document pairs, using LLM or VLM extraction, sentence alignment, and German or Turkish to English translation; heavy normalisation of transliteration conventions (determinatives such as `{d}`, subscripts, gaps, fractions); and sentence-level training to match the test.

**What separated the top 5 from 6 to 20** was the scale and cleanliness of extracted data more than model size. 1st, 4th and 5th all read **page images** with a multimodal model and linked each tablet to its catalogue ID, while most of 6 to 20 fed OCR text to an LLM for alignment. ByT5-XL appears in 4 of the top 5. And staged training, teaching Old Assyrian fundamentals before scholarly style, appears in the top 5, and 5th's ladder credits continued pre-training alone with +3.5 points. At the other end, 19th and 20th are single ByT5-base models on Gemini pseudo-labels at about 38.9: data alone got within 4 points of 1st.

**Where the writeups contradict each other.** On model size, 7th found "small << base << large ≲ XL" and 13th's XL beat its large, while 9th's best was base and 15th found large and XL did not work in its setup. On weight averaging, 4th, 5th, 19th and 20th gained and 6th and 15th lost. On curriculum, 8th and 12th call noisy-then-clean their biggest contributor, while 20th found gold fine-tuning after silver pre-training lowered its score. On sampled candidates inside MBR, 6th and 9th measured a loss and 13th and 1st shipped them. These are interactions with each team's data; no ablation transfers blindly.

## 6. The winning solution, explained: DataTech Club

![DataTech Club's pipeline: GLM-OCR maps tablets to page ranges, Gemini-3.0-pro reads the page images into sentence pairs, and 11 ByT5-XL models each emit 10 candidates for an MBR selector](/imgs/blogs/kaggle-deep-past-akkadian-translation-solutions-7.webp)

DataTech Club (guoqingu, hongori, leechengju, xiakie) led the public board and kept the lead. Their writeup is titled "Data Quality Dictates Everything", and the pipeline reads that way: the model side is ordinary, the data side is where the work went.

### Stage 1: throw away the training file

They discarded train.csv and rebuilt the training data from PDFs, both host-provided and self-collected. The given pairs were document-level, inconsistently segmented, and already inside the books; starting from the books gives more pairs, at the right granularity, in one style.

### Stage 2: let OCR find the tablets, let a VLM read them

For well-structured books (the AKT series, ICK 4, Larsen 2002, Michel 2020, POAT), they ran **GLM-OCR** layout analysis plus regular expressions to build a map from each tablet to the page range where its transliteration and translation sit. (GLM-OCR's own design is covered in [the GLM-OCR training recipe](/blog/machine-learning/frontier-models/glm-ocr-training-recipe).) Then **Gemini-3.0-pro-preview** read the page images for each tablet and emitted sentence-level transliteration and English pairs, translating German and Turkish editions to English as it went. Two details matter. The prompts segment by logical sentence, using the English translation as the anchor rather than the tablet's physical lines, which matches how the test was cut. And poorly structured sources went through a small screenshot-to-API front end the team built quickly, so no source was skipped for being awkward.

This is the "LLM as data engineer" pattern at its cleanest: a cheap OCR model handles layout, an expensive multimodal LLM reads, aligns and translates, and neither ships in the notebook.

### Stage 3: an ordinary ensemble, many candidates

On the model side: **11 ByT5-XL** variants trained on different data mixes, 10 with Hugging Face's Seq2SeqTrainer at batch size 64 for 3 epochs and one with a data-quality-weighted loss at batch 48 for 4 epochs. At inference each model produces 10 candidates using beam search and sampling at several temperatures, and a community **minimum Bayes risk (MBR)** selector picks one output per sentence. Everything ran through CTranslate2 in int8 with float32 on two T4s, using almost the full 9 hours.

MBR treats all candidates as samples of a good translation and picks the one that agrees most with the rest:

```python
# Illustrative toy, not any team's code: minimum Bayes risk (MBR) selection.
# Each candidate is scored by its average similarity to all the other candidates;
# the one that agrees most with the crowd wins.
from sacrebleu.metrics import CHRF

chrf = CHRF(word_order=2)

def mbr_pick(candidates):
    best, best_u = None, -1.0
    for i, hyp in enumerate(candidates):
        others = [c for j, c in enumerate(candidates) if j != i]
        u = sum(chrf.sentence_score(hyp, [o]).score for o in others) / len(others)
        print(f"  {u:5.1f}  {hyp}")
        if u > best_u:
            best, best_u = hyp, u
    return best

candidates = [
    "I paid 1 mina of silver to your messenger.",      # beam, model 1
    "I paid 1 mina of silver to your messengers.",     # beam, model 2
    "I have paid 1 mina of silver to your messenger.", # beam, model 3
    "I paid 1 mina of tin to your messenger.",         # sample, T=0.8
    "Your messenger paid me 1 mina of silver.",        # sample, T=1.0
]
print("MBR pick:", mbr_pick(candidates))
```

```console
   82.1  I paid 1 mina of silver to your messenger.
   78.6  I paid 1 mina of silver to your messengers.
   80.6  I have paid 1 mina of silver to your messenger.
   66.1  I paid 1 mina of tin to your messenger.
   60.2  Your messenger paid me 1 mina of silver.
MBR pick: I paid 1 mina of silver to your messenger.
```

The tin outlier and the reordered sample lose because they disagree with the consensus, and chrF++ as the agreement function aims selection at the metric that grades it. The cost is quadratic in candidates per sentence, affordable for ByT5 on T4s.

### The submission they did not pick

Their best submission was one they did not select: 41.5 public and 43.2 private. The selected one added a cleanup of repeated n-grams and scored 41.6 public, 42.8 private. Their stated principle: preprocess carefully and post-process as little as possible, because post-processing tuned on the public board is the main shake-up risk.

## 7. 2nd and 3rd in depth, then the rest of the top 20

### 2nd: KE WU, one model and three LLM pipelines (41.9311)

KE WU, solo, never changed the model: a plain ByT5-large. Every gain came from three LLM pipelines, each assigned to a different partition of the data. The **Breaker** split full document translations into sentence pairs, using word indices from the OARE database as anchors, turning 1,558 documents into 9,378 pairs (about six per document). The **Fixer** re-anchored fragments that had been pre-segmented badly. The **Generator** synthesised translations for untranslated texts from their morphological metadata. On top: about 60 more OCR'd academic publications and orthography normalisation. Hold the model fixed, and the data pipeline is the whole leaderboard.

### 3rd: darraghdog and conjuring92, a synthetic curriculum and an LLM judge (41.5091)

The 3rd-place pair asked a different question: what does a model need to know about Old Assyrian before it sees scholarly translations? Their answer was a **synthetic curriculum written by Claude 4.5 Sonnet**. Vocabulary drills covered about 1,000 lemmas from the CAD and eSAD dictionaries crossed with their senses, about 90,000 pairs, plus about 60,000 pairs mixing polysemous senses. Grammar drills came from about 500 rules extracted from Kouwenberg's 2017 grammar. Each ByT5-Large and ByT5-XL model was first given continued sequence-to-sequence pre-training on those drills, then fine-tuned on scholar-quality pairs. (The general recipe for generating such data is in [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation), and the ordering question in [data mixing and curriculum](/blog/machine-learning/training-data/data-mixing-domain-weighting-and-curriculum).)

The second LLM role was at inference: a fine-tuned **Qwen3-8B pairwise reward model** compared the ByT5-Large and ByT5-XL translations of each sentence and kept the better one. They were the only documented team to put an LLM in the notebook as a judge rather than a translator.

### Ranks 4 to 20, by approach family

![Two ablation ladders: 5th place climbed from 35.4 to 40.8 mostly through pre-training and synthetic data, and 20th place from 31.1 to 38.9 with Gemini pseudo-labels as the largest single step](/imgs/blogs/kaggle-deep-past-akkadian-translation-solutions-8.webp)

These two ladders agree: the large steps are data and training order; model tricks add tenths.

**Page-image extraction with heavy pipelines.** *4th, Wisdom and Vitaly*, spent two weeks fixing Tesseract errors in one book, then switched to Gemini 3.1 Flash in the AI Studio web UI, 15 to 20 page images at a time, for 14.5k training pairs. Ten candidates from mT5-Large and three ByT5 variants went through chrF++ MBR, and averaging two top checkpoints beat any single one. *5th, ebi & okumura*, built the most engineered extractor: Qwen3.5-Plus (397B) detects excavation numbers on page images, links them to the catalogue by alias matching, extracts with layout-specific prompts, and re-prompts failures: about 20,251 pairs for roughly USD 15 in API cost. Their ablation (top ladder) shows pre-training on untranslated texts worth +3.5 and synthetic back-translation from Qwen3.5-27B English worth +0.9. *6th, M & J & M*, triaged PDFs into image, clean-text and broken-text types with a different path for each, scraped word definitions from the OARE site to build a dictionary that guided an LLM's sentence splitting and pseudo-labelling, and trained with **dynamic context sampling** (a random number of neighbouring sentences as context) to survive misalignment. Their data ablation climbs from 34.3 private on cleaned train.csv alone to 39.9 with everything, before a 15-model MBR ensemble at 40.7.

**Scale the noisy stage, then polish.** *7th, Yurnero*, grew the data about 4 times to roughly 36k samples, including about 18k pseudo-labelled chunks of untranslated texts, and found "size matters": ByT5-small far below base, base below large, large about equal to XL. *8th, Hrithik Reddy*, in his first Kaggle competition, used only open-weight models: GLM-OCR plus a previous, current and next page sliding window into GPT-OSS-120B, because at least 40% of translations spill onto the next page. Stage 1 trained ByT5-XL on about 350k noisy pairs; stage 2 on about 65k curated pairs for exactly one epoch in fp32, since more caused collapse. *10th, Kelvin*, ran continued pre-training, then sentence-level fine-tuning, then pseudo-labelled the untranslated texts with the best model and retrained, and added MADLAD-400-3B to the MBR ensemble for diversity. He tied DataBoom as the biggest riser in the top 20, 19th public to 10th private. *12th, Ha-ar-pì-im*, had no API budget and extracted everything in Google NotebookLM, added about 11k template pairs (seals, witness lists, debts), and took a ByT5-base from about 35 to about 39 through three stages ending in a LoRA on the competition training set. *13th, DataBoom*, pre-trained on public Hugging Face Akkadian corpora plus OCR pairs and added **test-time source repair**, regex mappings that move corrupted test transliterations toward the training distribution.

**Clean relentlessly, small models.** *9th, Eleftheria 14*, extracted about 95k pairs with the Gemini batch API, then ran **more than 30 cleaning passes**: sample a batch, have Claude Opus judge alignment and OCR errors, find the systematic error class, write a programmatic fix plus an invariant test, rerun on everything. Gemini also generated about 124k **gap-augmented** pairs, deleting aligned spans from both sides and inserting matching `<gap>` tokens. Four ByT5-base models in FP32 (the writeup says BF16 "blurs" š and s) fed a **source-grounded MBR** that also checks that numbers, Sumerograms, divine names and gaps in the source appear in the candidate. Their key finding: multi-checkpoint MBR of one model scored 39.6 public but 38.4 private, while four diverse models scored 39.2 public and 40.1 private. "Diverse training runs generalise; diverse checkpoints of the same run don't." (The writeup quotes the gain as +2.58 over the single model; from the rounded private scores quoted alongside it I get 1.7, so the +2.58 likely uses a different baseline.) *11th, kwa*, pre-trained on multilingual pairs extracted from publications and used the competition metric itself as the MBR utility. *19th, vinpro*, went denoising, then 80k mixed pairs, then 10k gold, souped, and fuzzy-matched names to an onomasticon. *20th, Karl*, shipped one ByT5-base with no ensemble; his ladder (bottom) shows Gemini-3 Pro pseudo-labels of about 40k untranslated rows as the single biggest gain, +4.0.

**The LLM-as-translator route.** *14th, H&B&Z*, went against the field with Qwen3-14B, fine-tuned on RAG-assisted pseudo-labels and then RL (method unstated), and an unusual **transductive online learning** loop on the roughly 4,000 test sentences: Qwen3-14B, Qwen2.5-14B and Phi-4 pseudo-label the test set, a ByT5-Large trains on that, its outputs fine-tune the LLMs, and around again. Only alternating between architectures helped; self-training within one gave marginal gains. *15th, It's showtime!*, extracted about 31k pairs by feeding two GLM-OCR views (full page and cropped lines) to Kimi K2, trained a Gemma3-27B teacher whose translations were added at reduced weight to each ByT5 training row, and ensembled five ByT5 models with Qwen3-14B and two Qwen3-4B under chrF++ MBR.

**No writeup.** 16th John Doe #2, 17th lost in transliteration and 18th JoJoImpact published nothing that could be found; their methods are unknown.

**The best contrasting entry** is 14th: the only team to make the decoder LLM the main translator and the test set part of training, and still inside the top 15. The LLM route was not hopeless, only more expensive per point.

## 8. What the forum taught everyone

The pivotal forum moments were about data. The host's data-quality discussion was 15th's turning point after weeks of focal loss and architecture changes. The mid-competition data updates ("Data Update" and "Dataset Update - Mind the Gaps") changed what train.csv contained, which is why team-reported data sizes do not line up. And once the host published the scholarly PDFs, 4th noticed that all of train.csv already lived inside them, and 1st likewise discarded train.csv.

The shake-up lessons came from teams that measured it. 9th's checkpoint-versus-model experiment above is the cleanest: ensembling near-identical checkpoints overfits the public 34%. 5th found that scaling back-translation from 10k to 100k sentences cost 0.1 public and gained 0.3 private; 6th's pre-training on all publications scored 38.6 public but 39.5 private. Local validation rarely tracked the board: 8th's 500 held-out pairs scored 42 to 43.5 for all 40+ models, and 11th's LLM-built validation ran 41.7 to 47.9 against public 38.0 to 39.0. When validation shares the training set's style and the test does not, the public board becomes the only signal.

## 9. What did not work

Aggregated across the 17 writeups, with who reported it:

- **RL on seq2seq.** DPO (8th, 9th, 20th), minimum risk training (8th, 11th), PPO and GRPO for seq2seq (8th). Only 14th gained, on an LLM.
- **Rerankers.** LGBM (CV +0.2, leaderboard 0) and a fine-tuned LLM listwise reranker (5th).
- **Sampling inside MBR.** Hurt 6th; 9th measured +2.95 on 200 local samples and -2.7 on the board. Sampled or tagged back-translation also failed for 5th, as did logit-averaging ensembles once models reached about 37.
- **Weight averaging and soups.** Worse than single models for 6th; 15th's soups "in general" failed.
- **Longer context.** Document-level training (6th, 20th); context concatenated at inference (20th).
- **Name handling.** Lexicon capitalisation (4th: "a-na" is both a name and a preposition), PN/GN normalisation (11th), placeholders (15th), name augmentation (8th, 20th).
- **Raw-text pre-training.** Pre-training on a dictionary (11th); domain-adaptive pre-training on about 820k raw Akkadian lines, which caused repetition loops (9th, 34.6 against 38.5).
- **Decoder LLM translators.** Below ByT5 for 6th, 7th and 8th; hallucinations on raw extracted data (14th).
- **Post-processing.** LLM post-processing (8th); post-processing tuned on the public board (4th, and 1st by its own account).
- **More than one epoch on the clean stage** (collapse, 8th); disabling early stopping (20th, 37.5 to 36.6).

## 10. Lessons that transfer

Before the lessons, one more toy, because it is the habit behind 9th's thirty cleaning passes. An LLM extractor will occasionally smooth over a break, change a number or truncate a sentence. Cheap invariants catch most of that before it reaches training:

```python
# Illustrative toy, not any team's code: invariant checks on LLM-extracted pairs.
# The LLM proposes (transliteration, English) pairs; cheap rules decide what ships.
import re

def invariants(src, tgt):
    problems = []
    ratio = len(tgt) / max(len(src), 1)
    if not 0.3 <= ratio <= 4.0:                       # length-ratio filter
        problems.append(f"length ratio {ratio:.2f}")
    if src.count("<gap>") != tgt.count("<gap>"):      # damage must survive translation
        problems.append("gap count")
    if re.findall(r"\b\d+\b", src) != re.findall(r"\b\d+\b", tgt):
        problems.append("numbers")                    # 1 mina stays 1 mina
    return problems

pairs = [
    ("1 ma-na KÙ.BABBAR <gap> a-na DUMU ší-ip-ri-kà áš-qúl",
     "I paid 1 mina of silver <gap> to your messenger."),
    ("1 ma-na KÙ.BABBAR <gap> a-na DUMU ší-ip-ri-kà áš-qúl",
     "I paid 1 mina of silver to your messenger."),     # LLM smoothed over the break
    ("1 ma-na KÙ.BABBAR a-na DUMU ší-ip-ri-kà áš-qúl",
     "I paid 2 minas of silver to your messenger."),    # LLM changed a number
    ("um-ma A-šur-i-dí-ma a-na Pu-šu-ke-en6 qí-bi-ma",
     "Thus."),                                          # truncated extraction
]
for src, tgt in pairs:
    print(invariants(src, tgt) or "ship", "|", tgt)
```

```console
ship | I paid 1 mina of silver <gap> to your messenger.
['gap count'] | I paid 1 mina of silver to your messenger.
['numbers'] | I paid 2 minas of silver to your messenger.
['length ratio 0.11'] | Thus.
```

The length-ratio window is the one 5th place used; the gap and number checks are in the spirit of 9th's invariant tests and source-grounded MBR.

1. **Put the frontier model in the data pipeline, not the notebook.** At least 15 of 17 used one to build data; 2 shipped one as translator; the 72B route finished 24th. (1st, 2nd, 3rd; the matrix.)
2. **Read page images, not OCR text, when layout carries meaning.** The top 5 linked tablets to pages and let a multimodal model read the image; most of 6 to 20 aligned OCR text. (1st, 4th, 5th; 8th's sliding window for translations that spill across pages.)
3. **Teach fundamentals, then style.** Noisy or synthetic pre-training followed by clean fine-tuning is one of the largest measured effects in any writeup, +3.5 for one stage in 5th's ladder. Check the order on your own data, because 20th found the reverse. (3rd, 5th, 8th, 12th; 20th.)
4. **Match the metric's surface form.** A byte-level model, normalised orthography and exact names beat fluent paraphrase under BLEU and chrF++. (Toy above; 4th, 6th, 9th normalisation.)
5. **Select across diverse models, not checkpoints of one run.** MBR over different models held up on private; MBR over one model's checkpoints inflated public. (9th; 1st, 4th, 6th MBR.)
6. **Keep post-processing minimal.** Rules tuned on 34% of the test are the cheapest way to fall on the rest. (1st, 4th.)

### If you enter a competition like this

- Inventory every source of parallel text, scans included, and check whether the training file is a subset of it.
- Build the extraction pipeline with a cheap layout model plus an expensive reader, and gate every extracted pair with invariants (lengths, numbers, gaps, names).
- Pick a byte-level or character-aware model when the script has diacritics and editorial marks, and size it to the inference budget with room for many candidates.
- Train in stages: large noisy or synthetic data first, then a short pass on the cleanest pairs.
- Distrust a validation set built by the same LLM pipeline as your training data.
- Spend inference time on diverse candidates and MBR, not on post-processing rules.

The sibling post on [Make Data Count](/blog/machine-learning/learn-from-kaggle/kaggle-make-data-count-solutions) covers another extraction competition where the points came from data rather than the model, and the general version of this playbook for scanned documents is in [OCR, chart and document data](/blog/machine-learning/training-data/ocr-chart-and-document-data).

## Sources

- Competition page: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation ; host: https://www.deeppast.org
- 1st, DataTech Club: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684353
- 2nd, KE WU: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684345
- 3rd, How I Learned to Stop Worrying: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684425
- 4th, Wisdom and Vitaly: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/685925 ; code https://github.com/Anaphase21/Deep-Past-Challenge-2026/blob/main/code/DPC26_Training.ipynb ; submission https://www.kaggle.com/code/anaphase21/vitaly-deep-past-challenge-submission-code-a93e45
- 5th, ebi & okumura: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684847 ; synthetic data https://www.kaggle.com/datasets/yukiokumura1/deep-past-challenge-back-translated-synthetic-data
- 6th, M & J & M: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684231 ; cleaning https://www.kaggle.com/code/wscmisx/dpc-clean-function ; inference https://www.kaggle.com/code/wscmisx/final-dpc-ensemble-mbr-infer
- 7th, Yurnero: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684215
- 8th, Hrithik Reddy: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684329
- 9th, Eleftheria 14: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/685928 ; code https://github.com/Eleftheria14/Deep_Past_Gold_Solution
- 10th, Kelvin: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684211
- 11th, kwa: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684429
- 12th, Ha-ar-pì-im: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684866
- 13th, DataBoom: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/686956
- 14th, H&B&Z: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684236
- 15th, It's showtime!: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684819
- 19th, vinpro (self-titled 20th): https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684381
- 20th, Karl (self-titled 21st): https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684267
- 24th, Geremie Yeo (LLM-as-translator route): https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684189
- Excluded: "17th Place Solution" by kirito1sasa, https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684332 (matches no team in the API's top 60).
- Ranks 16, 17 and 18: no writeup found.

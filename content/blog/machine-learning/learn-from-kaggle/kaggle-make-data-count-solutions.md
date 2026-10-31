---
title: "Make Data Count: how the top 20 teams found dataset citations, and why the LLM was the smallest lever"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "A synthesis of the top 20 finishers in Kaggle's Make Data Count competition: why looking candidates up in the sources the labels were built from beat every extractor, why CatBoost on metadata typed DOIs better than an LLM, and where the LLMs actually earned their points."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "information-extraction", "scientific-literature", "data-citation", "catboost", "qwen", "label-noise", "leaderboard-shake-up"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 22
---

> **TL;DR.** In Make Data Count, Finding Data References (2025), 1,282 teams had to list every research dataset a scientific paper cites and tag it Primary (made for this paper) or Secondary (reused). Ali and Kea won with a private F1 of 0.79732. Their candidates came from the very corpora the hidden labels were built from, their DOI typing was a CatBoost model on registry metadata, and their LLM, a 0-shot Qwen2.5-Coder-32B, only typed accession IDs. The decisive lever was **retrieval from the label source**: 14 of the 15 top-20 writeups looked their DOIs up there, and the one that did not finished 20th. The one lesson: before you choose a model, find out how the labels were made.

![The lever strip with Retrieval lit: Make Data Count was decided by looking IDs up in the citation corpus and Europe PMC mined terms the labels came from, with a weaker metric lever for dropping rows you cannot type](/imgs/blogs/kaggle-make-data-count-solutions-1.webp)

The strip above frames every post in this series, with the decisive lever lit. This competition belongs in a series about LLM competitions with an asterisk. It looked like an LLM task: read a messy PDF, find the dataset mentions, judge from the prose whether the authors made the data or borrowed it. And 12 of the 15 top-20 teams with writeups did put an LLM in their type decision. But when you ask where the private points came from, the answer is uncomfortable for anyone who came to fine-tune: most of the score came from a lookup table and a page of rules, the models added the last few hundredths, and the best DOI model in the competition was not a language model.

This post synthesises the writeups of the top 20 private finishers: 15 published one, five posted nothing and are named as such. New to the series? Start with [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work).

## 1. The task in 60 seconds

Make Data Count is a DataCite initiative that wants datasets cited, and counted, the way papers are. The competition ran from 2025-06-11 to 2025-09-09 with $100,000 in prizes ($40k, $20k, $17k, $13k and $10k for places 1 to 5), and drew 1,282 teams and 36,850 submissions.

![Notebook budget card and an illustrative paper fragment: a GEO accession introduced with a deposit phrase becomes a Primary row, a reused atlas accession becomes a Secondary row, and a gene nomenclature ID produces no row](/imgs/blogs/kaggle-make-data-count-solutions-2.webp)

Input: a paper from the Europe PMC open-access subset, as a PDF and, for most articles, XML. Output: one row per referenced dataset, `(article_id, dataset_id, type)`. A dataset is identified by a **DOI**, such as `https://doi.org/10.5061/dryad.6m3n9` for a Dryad deposit, or an **accession ID** from a biological database: `GSE12345` in GEO, a PDB code like `1Y2T`, `E-MEXP-568` in ArrayExpress, a BioSample `SAMN...`. The type is **Primary** if the data was generated for this paper and **Secondary** if it is reused. Articles citing no dataset must be left out.

Training had 524 PDFs; the hidden test about 2,600 articles. It was a code competition: CPU or GPU notebook, at most 9 hours, internet off, public leaderboard on 30% of the test and the private 70% paying out.

The figure's right half holds the whole difficulty. One paper mentions three identifiers. Two are datasets, one deposited by the authors and one compared against. The third looks exactly like an accession to a regex but is a gene nomenclature ID the labels never contain. A solution must know, for every string, whether it counts at all, and then whose data it is.

## 2. The metric, worked on a toy

The score is a global F1 over unique `(article_id, dataset_id, type)` tuples, pooled across the test set:

$$
F_1 = \frac{2\,TP}{2\,TP + FP + FN}
$$

A tuple is a true positive only if all three fields match. As the 3rd-place team spelled out, finding the right dataset with the wrong type costs you **twice**: a false positive for the tuple you submitted and a false negative for the one you missed.

![Global F1 on a toy with four gold tuples: perfect scores 1.000, dropping the one uncertain row scores 0.857, guessing its type wrong scores 0.750, and extracting everything including three non-dataset IDs scores 0.727](/imgs/blogs/kaggle-make-data-count-solutions-3.webp)

Take four gold tuples. All four right: 1.000. Leave the uncertain one out: 3 TP, 0 FP, 1 FN, so $\frac{6}{7} = 0.857$. Submit it with the wrong type: 3 TP, 1 FP, 1 FN, so $\frac{6}{8} = 0.750$. When you cannot type a dataset, omitting it beats guessing. Now a greedy extractor that finds all four datasets but also submits three identifiers the labels do not contain: 4 TP, 3 FP, $\frac{8}{11} = 0.727$, worse than missing a dataset outright. Precision on *which* IDs count matters as much as recall, and the type decision is worth double.

Mid-competition the host found partially labelled training articles, listed their unlabelled citations as "Missing", changed the metric to ignore unlabelled citations in such articles, and rescored every submission.

## 3. What made it hard: the labels came from a machine

The hard constraint was not the PDFs or the 9 hours. It was the labels. Per the host, quoted in the 1st-place writeup, accession IDs came from **Europe PMC's text-mined terms** (a named-entity tagger run over the open-access corpus) and DOIs from **DataCite's Data Citation Corpus**, a database of dataset-to-article links. Human labellers then judged Primary versus Secondary from the PDF.

Roderic Page hand-annotated the training PDFs and published the result: the official labels scored an F1 of **0.446** against his annotations. A careful human reader agreed with the answer key less than half the time, and the host acknowledged label issues in a data-quality mega-thread.

That reframed the task from "which datasets does this paper cite?" to "which datasets would *this pipeline* have recorded?" A perfect reader would be punished for every dataset the text miner missed. Khoi Nguyen said it early: the leaderboard favoured *fewer* references, so the task was "retrieval vs agreeing with labels". The winners took that literally and downloaded the label generator.

## 4. Top-20 scorecard

Private F1 from the leaderboard; "Public" is the public rank. Gold ended at rank 12 (13th place wrote that they missed gold "by one rank").

| Rank | Team | Private F1 | Public | Key idea | Source |
|---|---|---|---|---|---|
| 1 | Ali / Kea | 0.79732 | 9 | Corpus v4.1 + Europe PMC lookup; CatBoost on metadata for DOIs; 0-shot Qwen2.5-Coder-32B for accessions | [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606853) |
| 2 | Mohsin and Nikhil | 0.78648 | 10 | 100% candidate recall; rules alone gold; MedGemma-4B LoRA + BiomedBERT | [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606786) |
| 3 | BRBR | 0.77292 | 12 | Dump lookup; rules, then 6-fold DeBERTa-v3-large with EMA | [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606879) |
| 4 | [NVIDIA] Raja & Ahmet & David | 0.76788 | 1 | Agent-made synthetic labels; 14B to 32B to 72B Qwen2.5 cascade | [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606921) |
| 5 | cm391 | 0.76167 | 7 | Lookup + family filter; XML context and registry metadata into two 32B Qwens | [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606769) |
| 6 | Team | 0.75866 | 2 | Lookup + ordered hand rules; no model | [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606867) |
| 7 | 反卷局重新起航 | 0.75232 | 5 | Gemini labels distilled into 7B and 14B classifiers, 3-model vote | [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606757) |
| 8 | zhang2dog | 0.74574 | 13 | Fine-tuned Qwen2.5-32B for DOIs; table-aware prompts for accessions | [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606768) |
| 9 | flash attention | 0.74556 | 14 | Deposit rule gated by embeddings; 0-shot Qwen2.5 DOI chain | [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606743) |
| 10 | Khoi Nguyen | 0.74105 | 19 | Lookup, LB-probed prefixes, all SAMN Primary; no model | [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606740) |
| 11 | Have Hope | 0.73952 | 3 | Prefix filter with LLM re-extraction; keyword-gated LLM (overfit) | [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606940) |
| 12 | choya & Okumura & takai | 0.73197 | 6 | 0-shot Qwen2.5-72B-AWQ with corpus-graph features; BioSample date rule | [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606745) |
| 13 | RB, NB, BD | 0.73164 | 15 | GROBID XML for DOIs, provided XML for accessions, Qwen2.5-32B | [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606979) |
| 14 | Takashi Someya | 0.73066 | 11 | unknown | no writeup |
| 15 | idk | 0.72799 | 4 | Registry metadata for all families; LLM author-ownership decision | [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606751) |
| 16 | Viktoria | 0.69469 | 26 | unknown | no writeup |
| 17 | wolf | 0.68027 | 23 | unknown | no writeup |
| 18 | hamlet | 0.66881 | 18 | unknown | no writeup |
| 19 | Ebi | 0.66631 | 8 | unknown | no writeup |
| 20 | Kawsar Hossain | 0.65779 | 20 | Regex only, no lookup; DeBERTa filters, QLoRA Qwen2.5-3B, Light-R1-14B | [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606746) |

The spread from 1st to 13th is 0.066 F1, smaller than any top team's public-to-private drop: the winners lost 0.094, the public leaders 0.150.

## 5. The top-20 landscape

![Technique matrix for the top 20: 14 of 15 writeups looked DOIs up in the citation corpus, 13 looked accessions up in Europe PMC mined terms, 12 used author or metadata overlap, 12 used an LLM in the type decision, only 5 fine-tuned one, and five columns are question marks for teams with no writeup](/imgs/blogs/kaggle-make-data-count-solutions-5.webp)

Read the matrix top down and it tells the story in order of importance.

**Table stakes.** Fourteen of 15 writeups took DOI candidates from the Data Citation Corpus or the DataCite dump; 13 took accession candidates from Europe PMC's mined terms (11th did not say; 20th did not). All kept a candidate only if the ID literally appeared in the text. Thirteen filtered families the labels never contain: `GCA_` assemblies, `HGNC:` gene names, `GO:` terms, `RRID:` IDs, figshare and HEPData DOIs. Five said they used leaderboard probes to pick families or rules (1st, 2nd, 3rd, 6th, 10th).

**The type decision.** Nine of 15 typed most accessions by rule: default Secondary, a BioSample (`SAMN`) rule for Primary (2nd place's final accession typing is not stated, so it counts as unknown). Twelve of 15 typed DOIs with author or metadata overlap. That is the consensus insight: for a DOI, the strongest Primary signal is **ownership** (do the dataset's authors overlap the paper's?), not the sentence around it.

**The models.** Twelve of 15 used an LLM somewhere in the type decision; three of the top 10 used none (3rd used DeBERTa, 6th and 10th only rules). Only 5 fine-tuned an LLM (2nd, 4th, 7th, 8th, 20th) and 6 prompted one zero-shot (1st, 5th, 8th, 9th, 12th, 20th). Three trained on synthetic or teacher-LLM labels (4th, 7th, 20th). Named LLMs were nearly always Qwen2.5.

**What separated the top 5.** All five spent model capacity on the hard calls (DOIs and SAMN) *and* fed the model real metadata: Crossref and DataCite fields for CatBoost, DataCite abstracts for MedGemma, titles for DeBERTa, registry metadata in Qwen prompts. Ranks 6 and 10 stopped at rules and still took gold. Ranks 11 to 15 layered LLMs over hand rules that fit the public 30%.

**Contradictory findings.**

- *Drop accessions in articles with DOIs.* Helped 1st and 2nd; 4th used a conditional version; 3rd dropped the DOIs instead; 13th's unselected mutual-exclusivity submission would have been gold. 9th says it lowered his score.
- *Fine-tune or prompt.* Fine-tuning helped 8th, and synthetic warm-up gave 4th about +0.02; 13th says fine-tuning did not work, and 9th found GPT-5, Claude 4 and Grok 3 no better than zero-shot Qwen2.5-32B, so skipped it.
- *Bigger models.* 12th ran Qwen2.5-72B and 4th sent its hardest cases to 72B; 13th and 9th found 72B no better.
- *LLMs for accession type.* 1st and 8th used one; 12th lists it as a failure and 9th chose a rule because the labels were too noisy to learn.

## 6. Where the points came from

This is what the matrix cannot show. Several writeups reported scores for intermediate pipelines, so we can build a ladder of private F1 from "no lookup, all model" to "lookup plus the best models".

![Private F1 ladder from the writeups: pure ML extraction without the corpus 0.613, the same with a corpus filter 0.647, a CPU lookup with rules and no model 0.70, 2nd place's lookup plus heuristics with no model 0.739, 2nd place's final with MedGemma and BiomedBERT 0.786, and 1st place's final 0.797](/imgs/blogs/kaggle-make-data-count-solutions-4.webp)

1. A 22nd-place team (two different topics claim 22nd; I could not resolve which is right) ran "pure ML" without the corpus: regex highlights, Qwen 7B and 14B extraction from an 80-character window, a fine-tuned Qwen 14B filter and a four-prompt type classifier. Private **0.613**; adding the corpus as a filter gave **0.647**.
2. The 5th-place author published a CPU-only notebook: lookup, family filter, rules, no model. Private **0.70**.
3. 2nd place's lookup plus five rules, no model: private **0.739**, gold on its own. Their words: "No model required to get a gold medal!"
4. 2nd place's final, with MedGemma-4B and BiomedBERT on top: **0.786**.
5. 1st place's final: **0.797**.

The models are worth about 0.047 at the top (0.786 minus 0.739, same team, same candidates). The step from a model-heavy extractor to a model-free lookup (0.613 to 0.70) is nearly twice that. 1st place's own split sharpens it: DOI-only scored 0.350 private, accession-only 0.664, and accessions, whose candidates came straight from Europe PMC, were most of the test (they estimated about 70% of rows).

None of this makes LLMs useless. The last 0.05 separates 1st from 13th, and that is where the prize money was. It says the LLM was the last lever, pulled after the two bigger ones.

## 7. The winning solution, explained

![First place pipeline: cleaned PDF text feeds two lanes; DOIs come from Citation Corpus v4.1 and are typed by 6-fold CatBoost on Crossref and DataCite metadata, accessions come from the corpus plus Europe PMC mined terms and are typed by a 0-shot Qwen2.5-Coder-32B with A/B logits](/imgs/blogs/kaggle-make-data-count-solutions-6.webp)

### The intuition: retrieve from the label generator, type with the right tool

Kea Kohv and Ali Salhi started from the host's description of where labels came from: if the answer key was made by a citation corpus and a text miner, the best candidate generator *is* that corpus and that miner. No extractor agrees with an NER system's misses better than the NER system. Then they split the problem, because a DOI's type is about *who made the dataset*, a metadata question, while an accession's type is about *how the paper talks about it*, a language question. Each half got a different model.

### The mechanism: DOIs

Text came from PyMuPDF, with whitespace cleaning and Unicode normalisation. DOI candidates were Data Citation Corpus v4.1 pairs with source DataCite, kept only if the DOI appeared in the PDF or XML. On train: 321 predictions, 302 true positives, 19 false positives, 23 misses. The full DataCite dump gave 100% recall but more false positives.

For type they threw the text away. Article metadata came from Crossref (authors, title, year, publisher, journal), dataset metadata from DataCite (authors, year). Features compared titles, authors and years, with publisher, journal and repository as categoricals and the DOI count per article. The model was a **6-fold CatBoost**, stratified by type and grouped by article, with out-of-fold type F1 0.87. Qwen for DOI type, even with metadata and a snippet, lost; so did Qwen as a CatBoost reviewer, and so did article-context features. For DOIs, the language model had nothing to add that a table of author overlaps did not already say. (For gradient boosting's failure modes on small tabular data, see [debugging gradient boosting and class imbalance](/blog/machine-learning/debugging-training/gradient-boosting-and-imbalance-debugging).)

### The mechanism: accessions

Accession candidates came from corpus v4.1 rows with source `eupmc` plus the raw Europe PMC mined terms (Cellosaurus, for one, is missing from the corpus but present in the labels), through a hand-chosen family whitelist (ArrayExpress, InterPro, ChEMBL, UniProt, PDB, BioSample, GEO and others; ENA only with prefix filters). They predicted accessions only for articles with **no** predicted DOI, a leaderboard gain against public advice.

Type went to **Qwen2.5-Coder-32B**, AWQ-quantised and served with vLLM, prompted zero-shot on the context snippet; Coder beat the base model. A `MultipleChoiceLogitsProcessor` forced the output to A or B, [constrained decoding](/blog/machine-learning/inference-engineering/constrained-decoding-from-first-principles-masking-logits-with-an-fsm) in its simplest form. For BioSample IDs the prompt added article title, authors and year plus the BioSample submitter and date from the NCBI API. A final rule set each family's type to its majority within the article. AWQ is what fit 32B into the notebook; see [weight-only quantization with GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq).

Validation dropped sparsely labelled articles, following mccocoful; correlation with the leaderboard stayed shaky. They used submissions to learn which accession families existed in the public test.

### A toy version

The core idea in about 40 lines, an illustrative toy, not any team's code: a greedy extractor against a lookup in the label source, with the same naive type rule.

```python
# Illustrative toy, not any team's code: why "look it up in the label source" beats "extract everything".
import re

papers = {
    "A1": "Reads were deposited at GEO under GSE20001; we compared them with GSE11111. "
          "Gene names follow HGNC:1100 and terms GO:0008150.",
    "A2": "Our data are available from Dryad (doi:10.5061/dryad.x1y2) and we reused "
          "structure PDB 1Y2T. See also 10.1038/nature1234 for methods.",
}
# What the hidden labels were built from: a citation corpus (DOIs) and text-mined terms (accessions).
label_source = {
    "A1": {"GSE20001", "GSE11111"},
    "A2": {"10.5061/dryad.x1y2", "1Y2T"},
}
gold = {("A1", "GSE20001", "Primary"), ("A1", "GSE11111", "Secondary"),
        ("A2", "10.5061/dryad.x1y2", "Primary"), ("A2", "1Y2T", "Secondary")}

ID = re.compile(r"10\.\d{4,9}/[\w.]+\w|GSE\d+|HGNC:\d+|GO:\d+|PDB (\w{4})")

def extract_all(text):                      # a greedy extractor
    return {m.group(1) or m.group(0) for m in ID.finditer(text)}

def lookup(aid, text):                      # keep only label-source IDs that occur in the text
    return {d for d in label_source[aid] if d in text}

def type_rule(aid, d, text):                # toy ownership rule: deposit phrase or Dryad => Primary
    near = text[max(0, text.find(d) - 40): text.find(d)]
    return "Primary" if ("deposited" in near or "dryad" in d or "Our data" in near) else "Secondary"

def f1(pred):
    tp = len(pred & gold); fp = len(pred - gold); fn = len(gold - pred)
    p = tp / (tp + fp) if pred else 0.0; r = tp / (tp + fn)
    return tp, fp, fn, round(2 * p * r / (p + r), 3) if tp else 0.0

for name, cands in [("extract everything", extract_all), ("label-source lookup", None)]:
    pred = set()
    for aid, text in papers.items():
        ids = extract_all(text) if cands else lookup(aid, text)
        pred |= {(aid, d, type_rule(aid, d, text)) for d in ids}
    print(f"{name:20s} TP,FP,FN,F1 = {f1(pred)}")

# A right dataset with the wrong type is one FP and one FN at once:
flipped = (gold - {("A1", "GSE11111", "Secondary")}) | {("A1", "GSE11111", "Primary")}
print(f"{'one type flipped':20s} TP,FP,FN,F1 = {f1(flipped)}")
```

```console
extract everything   TP,FP,FN,F1 = (4, 3, 0, 0.727)
label-source lookup  TP,FP,FN,F1 = (4, 0, 0, 1.0)
one type flipped     TP,FP,FN,F1 = (3, 1, 1, 0.75)
```

The extractor finds every real dataset and still loses 0.27 F1, because the gene ID, the GO term and the methods-paper DOI all look like datasets to a regex. The lookup never sees them. The real lookup was not perfect (19 false positives and 23 misses on DOIs), but its errors were the label's own errors, the only kind the metric forgives.

## 8. Second and third in depth, then ranks 4 to 20

### 2nd: Mohsin and Nikhil, recall first, rules second, models last

They built for **100% candidate recall**, then filtered. DOIs: the Data Citation Corpus (v2 validated best) and the DataCite Public Data File 2024, a superset that alone gave 100% recall; drop figshare, CCDC and HEPData (tagged Missing in train); keep IDs a whitespace-tolerant regex finds in the text; strip Dryad version tags. That reached 0.964 type-agnostic F1 on train. Accessions: Europe PMC mined terms minus `:`-style families and a few others, minus accessions in articles with a DOI, plus a rule for SAMN IDs in an online-only table: 0.98.

Then the rules: SAMN and EMDB are Primary; for a DOI cited by several papers, the earliest citer is Primary; an `isSupplementTo` relation is Primary; more than four same-repository DOIs together are Secondary. Gold already. Then models: Mohsin fine-tuned **MedGemma-4B with LoRA** for DOI type on at most 2,048 tokens, the first match's context plus the top three 1,024-character BM25 chunks retrieved against the dataset's DataCite abstract (0.784 private). Nikhil trained two **BiomedBERT** classifiers that masked every other ID as "other dataset id", with batch size 256 for stability. Note that the LLM's key input was the dataset's abstract, fetched from a registry: ownership again, phrased as text. For LoRA mechanics, see [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft).

### 3rd: BRBR, an encoder, stabilised

Arc144 and tomirol matched regex candidates against the 2024 DataCite dump and Europe PMC terms, took metadata from the 2025 Crossref dump, and dropped figshare DOIs, `GCA_`, `HGNC:` and `rs` IDs. When an article had both kinds, they dropped the DOIs, the mirror of 1st place's choice, and also a gain. Rules first: Dryad DOIs and SAMN Primary; a DOI whose title or authors resemble the paper's Primary; an accession cited more than five times in Europe PMC Secondary; a DOI with five or more citations Secondary.

The rest went to a **6-fold DeBERTa-v3-large** (StratifiedGroupKFold by type and article) reading a 1,000-character context, the article's first 500 characters and titles. Qwen2.5-7B with a classification head was unstable; DeBERTa was faster and better once stabilised with gradient clipping and a weight EMA (decay 0.9995). A podium finish with no LLM in the final pipeline is the cleanest evidence for this post's thesis.

### Ranks 4 to 20, by approach family

**Family A: an LLM as teacher (4, 7, 20).** The NVIDIA team (Raja Biswas, Ahmet Erdem, David) built a tool-calling agent with keyword and semantic search that read Europe PMC articles of 50,000 tokens and more and wrote synthetic labels, used to warm up Qwen2.5 models before fine-tuning on competition data (about +0.02). Public 1st (0.91837), private 4th (0.76788). 7th had Gemini label corpus mentions plus train, distilled that into 7B and 14B classifiers (family not stated) and voted three models; an unselected `EPI_` author lookup would have scored 0.771 private and beaten their final. 20th (Kawsar Hossain) used no lookup at all: regex candidates, DeBERTa false-positive filters, a QLoRA Qwen2.5-3B accession classifier and Light-R1-14B-AWQ for DOIs, trained on a consensus of GPT-5, Qwen3-30B-thinking, Qwen2.5-72B and Llama-70B labels. Built in a few hours a day after a 9-hour job, it was the best pure-extraction pipeline in the top 20, and still 0.14 behind the winner.

![Fourth place uncertainty cascade drawn as a funnel: Qwen2.5-14B sees every candidate, the uncertain 50 percent of DOIs and 20 percent of accessions go to Qwen2.5-32B, and the hardest 10 percent go to Qwen2.5-72B](/imgs/blogs/kaggle-make-data-count-solutions-7.webp)

The cascade is the one idea here purely about fitting LLMs into a budget: 14B labels everything, its uncertain half of DOIs and fifth of accessions go to 32B, and only the hardest 10% reach 72B. The big model is paid for only where the small one hesitates.

**Family B: rebuild the labeller's view, prompt a big model (5, 8, 12, 13).** mccocoful (5th) titled his post "standing on the shoulders of giants": mid-competition, keakohv (half of the eventual winners) described the DOI lookup in a public comment, and a host comment confirmed where accession labels came from. He rebuilt around that, then added XML-aware context (every place the ID's element is referenced, only the matching table row), sent only DOIs and SAMN to the LLM because they split roughly 50/50, and voted Qwen2.5-32B-AWQ and Qwen3-32B-AWQ, prompted with an Owner / User / None rubric ("usually one author overlap is enough to assume Owner"), with his rules. zhang2dog (8th) fine-tuned Qwen2.5-32B for DOIs and prompted it for accessions with table context: caption, footnote, headers, three lines either side. 12th fed Qwen2.5-72B-AWQ zero-shot with corpus-graph features written as text (citation count, whether this paper is the first citer, days since the dataset's release), forcing A or B by the larger logit; metadata lifted their DOI-only score from 0.333 to 0.345. Their BioSample rule: Secondary if more than three years separate ID issue and publication. 13th ran GROBID-parsed XML for DOI context and the provided XML for accessions (every train accession is in the XML), Qwen2.5-32B under [vLLM](/blog/machine-learning/inference-frameworks/vllm-deep-dive) for both.

**Family C: no model at all (6, 10).** The 6th-place team (public 2nd) typed DOIs with an *ordered* rule list: default Primary; Secondary if the DOI was cited earlier by another article, appears only in the references, or sits among more than four DOIs, or if the paper's authors do not overlap the citation after it; Primary for Dryad, Zenodo and F1000, self-citations and deposit phrases. Probing showed some families (NCT, EudraCT, EBiSC, ChEBI, EFO) lived only in the private split. Khoi Nguyen (10th) used DOIs from corpus v3, accessions from v4's Europe PMC rows, a probed prefix list (removing one prefix he could not identify was worth +0.08 on both boards) and all SAMN Primary. "Nothing beat probing."

**Family D: rules, then an LLM, then overfit (9, 11, 15).** Geremie Yeo (9th, solo gold) typed accessions by rule: Primary if the sentence contains "deposit" or "submit" (about 95% of Primary), with false hits filtered by Qwen3-Embedding-0.6B similarity of at least 0.667 to a seed deposition sentence, plus a 64-per-article cap on ENA and protein IDs. DOIs went through a zero-shot Qwen2.5 chain. He was stuck at 0.64 public for a week and reached 0.82 in five days by debugging, not by changing models. Have Hope (11th, public 3rd) wrote that their keyword gates before the LLM were the overfit part; an unselected pure-LLM run did better. "idk" (15th, public 4th) matched authors for every family using DataCite, BioSample and GEO metadata, and believes extending that to GSE accessions cost him; "all SAMN Primary" would have done better.

**No writeup: 14 (Takashi Someya), 16 (Viktoria), 17 (wolf), 18 (hamlet), 19 (Ebi).** No writeup was found; their methods are unknown. Ebi fell from public 8th to 19th; Viktoria climbed from 26th to 16th.

### The best contrasting entry

The sharpest contrast is inside 2nd place: the same lookup with five rules (0.739) and with MedGemma and BiomedBERT on top (0.786). Same candidates, same team, the models' entire contribution visible as one number, and the first submission was already gold.

## 9. What the forum taught everyone

![Public rank against private rank for the top 20: the three winners came from public 9th, 10th and 12th, while the public top six fell to private 4th, 6th, 7th, 11th, 12th and 15th](/imgs/blogs/kaggle-make-data-count-solutions-8.webp)

The decisive idea was published mid-competition. keakohv's comment describing the DOI lookup and the host's comment on label sources were the turning point; 5th place and a commenter on the 20th-place thread both say they rebuilt afterwards. Roderic Page's annotations (the 0.446 thread, 75 votes) made the label noise impossible to ignore, and Hzwlf's "My Rule-Based Method Beats My LLM Attempts" put numbers on the theme: an all-SAMN-Primary rule scored 0.660 against 0.600 for LLM classification.

Then the shake-up. Top-20 public scores ran from 0.76 to 0.92, private from 0.66 to 0.80. The winners came from public 9th, 10th and 12th; the public top six finished 4th, 6th, 7th, 11th, 12th and 15th. The writeups agree on why: family whitelists and accession rules tuned on a 30% public split of noisy, partially labelled data. Four teams (7th, 9th, 11th, 13th) left a better private submission unselected.

The 22nd-place team raised the fairness question: corpus lookup cannot work for new papers not yet in any corpus. Commenters replied that the host updates both sources continuously. Both are right, and that tension belongs in the lessons.

## 10. What did not work

Aggregated across the top-20 writeups, with who reported it:

- **Regex extraction without external data:** 13th, 20th, 22nd. The most consistent negative result.
- **An LLM for DOI type**, with metadata and snippet, or as a CatBoost reviewer: 1st.
- **An LLM for accession type:** 12th; 9th chose a rule because the labels were too noisy.
- **Fine-tuning:** 13th; 9th (frontier models no better than zero-shot). It helped 8th and 4th.
- **Qwen2.5-72B or 14B instead of 32B:** 13th; 72B too slow and no better, 9th.
- **Qwen2.5-7B with a classification head:** unstable, 3rd.
- **Keyword gates before the LLM:** 11th. **Author matching extended to GSE accessions:** 15th.
- **Dropping accessions when a DOI is predicted:** 9th (it worked for 1st, 2nd and, mirrored, 3rd).
- **Reasoning before the answer, chunked NER sweeps, semantic search for context, publication versus dataset date:** 9th.
- **DataCite relation types, article-context features, fancier or Chinese prompts, article metadata in the accession prompt:** 1st (kept only for SAMN, where it scored 0.001 private below simply forcing SAMN to Primary).
- **The Marker PDF parser:** 1st and 9th. **Fixing DOI version suffixes, GPT-OSS labelling:** 13th.
- **Choosing the final submission by public score:** 7th, 9th, 11th and 13th.

## 11. Lessons that transfer

1. **Find out how the labels were made, then use that process.** The winners used the label source as their candidate generator (1st, 2nd). A better extractor than the label source is, by the metric's definition, worse.
2. **Put the LLM where language decides the answer, nowhere else.** 1st typed accessions with Qwen because context mattered, and DOIs with CatBoost because ownership is metadata. The LLM on DOIs lost.
3. **Build the rule baseline first and make the model beat it.** 2nd place's rules alone were gold; 6th and 10th never needed a model; Hzwlf's rule beat his LLM by 0.06.
4. **When labels are scarce, the LLM can be the labeller.** The NVIDIA agent's warm-up labels were worth about +0.02; 7th distilled Gemini into small local models.
5. **Spend big-model compute only where the small model hesitates.** The cascade fit a 72B model into 9 hours by sending it 10% of the work.
6. **Never select on a 30% public split of noisy labels.** The public top six all fell.

### If you enter a competition like this

- Read every host comment on how labels were produced. If it is a database or a model, download it.
- Measure candidate recall and precision separately from the type decision.
- Write and submit the rule baseline before any model.
- Split the problem by what drives the answer (metadata versus text) and pick a model per half.
- A wrong type costs a FP and a FN: when unsure, leave the row out.
- Validate on fully labelled articles; pick finals by local validation, not public rank.

For the rest of this track, see [Santa 2024's perplexity puzzle](/blog/machine-learning/learn-from-kaggle/kaggle-santa-2024-perplexity-puzzle-solutions), [Drawing with LLMs](/blog/machine-learning/learn-from-kaggle/kaggle-drawing-with-llms-solutions), [Jigsaw's agile community rules](/blog/machine-learning/learn-from-kaggle/kaggle-jigsaw-agile-community-rules-solutions) and [Deep Past's Akkadian translation](/blog/machine-learning/learn-from-kaggle/kaggle-deep-past-akkadian-translation-solutions). The closest cousin is [PII Data Detection](/blog/machine-learning/learn-from-kaggle/kaggle-pii-data-detection-solutions), another extraction task where the winning LLMs never ran at inference.

## Sources

Competition: [Make Data Count, Finding Data References](https://www.kaggle.com/competitions/make-data-count-finding-data-references).

- 1st, Ali / Kea: [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606853), [notebook](https://www.kaggle.com/code/keakohv/mdc-1st-place-solution-catboost-and-qwen/notebook?scriptVersionId=262235918), [earlier DOI comment](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/600330)
- 2nd, Mohsin and Nikhil: [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606786)
- 3rd, BRBR: [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606879)
- 4th, NVIDIA team: [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606921), [code](https://github.com/rbiswasfc/mdc-4th-place-solution), [agent thread](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606752)
- 5th, cm391: [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606769), [CPU notebook](https://www.kaggle.com/code/mccocoful/mdc-cpu-submission)
- 6th, Team: [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606867)
- 7th: [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606757); 8th: [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606768)
- 9th, flash attention: [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606743), [code](https://github.com/bogoconic1/9th-place-kaggle-mdc-finding-data-references)
- 10th: [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606740); 11th: [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606940); 12th: [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606745); 13th: [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606979)
- 15th: [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606751); 20th: [writeup](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606746)
- Ranks 14, 16, 17, 18, 19: no writeup found.
- Beyond the top 20: 22nd (claimed), [pure ML vs corpus](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606794)
- Threads: [updated labels and metric](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/589314), [data quality](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/584899), [host on label sources](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/584337), [Roderic Page's labels](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/586075) ([data](https://github.com/rdmpage/make-data-count-training-data)), [train vs test labels](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/584792), [rules beat LLM](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/598550)

# Make Data Count - Finding Data References: top-20 writeup addendum

- Competition: Make Data Count - Finding Data References
- Slug: `make-data-count-finding-data-references` (id 82370, forum 5476697)
- URL: https://www.kaggle.com/competitions/make-data-count-finding-data-references
- Existing coverage: `.claude/plans/kaggle-llm-writeup-bank-part3.md`, H2 "Make Data Count - Finding Data References" (lines ~490-632). It covers ranks 1 to 4 in depth (candidate generation from Data Citation Corpus + Europe PMC TextMinedTerms, CatBoost DOI typing, Qwen accession typing, MedGemma/BiomedBERT, DeBERTa + EMA, NVIDIA agent synthetic labels + 14B/32B/72B cascade), plus the label/metric update, key threads and the shake-up. This file does not repeat that detail; it adds ranks 5 to 21 and the cross-team synthesis.
- Recap of the task in one line: for each paper list every dataset DOI or accession ID it references and tag Primary/Secondary; metric is global F1 over (article, dataset, type) tuples; public/private 30/70; labels were derived from the DataCite Data Citation Corpus (DCC) and Europe PMC text-mined terms, so "retrieve from the label generator" was the winning idea.

## Top-20 table

Ranks and scores from the leaderboard API (`lb.json`). Public rank in parentheses shows the shake-up. Rank 21 included because several ranks 14 to 21 have no writeup.

| Private rank | Team | Members | Private | Public rank (score) | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | Ali / Kea | asalhi, keakohv | 0.79732 | 9 (0.89095) | https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/1st-place-solution | DCC v4.1 + EUPMC candidates; DOI type by 6-fold CatBoost on Crossref/DataCite metadata; accession type by 0-shot Qwen2.5-Coder-32B-AWQ |
| 2 | Mohsin and Nikhil | tezdhar, nikhilmishradev | 0.78648 | 10 (0.88680) | https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/2nd-place-solution | DCC + DataCite dump + EUPMC for full recall; gold-level heuristics; MedGemma-4B LoRA + BiomedBERT classifiers |
| 3 | BRBR | arc144, tomirol | 0.77292 | 12 (0.87635) | https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/3rd-place-solution | Regex matched against DataCite/EUPMC dumps; heuristics then 6-fold DeBERTa-v3-large with EMA |
| 4 | [NVIDIA] Raja & Ahmet & David | aerdem4, tivfrvqhs5, conjuring92 | 0.76788 | 1 (0.91837) | https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/4th-place-solution | Agent-made synthetic labels warm up Qwen2.5; 14B to 32B to 72B uncertainty cascade |
| 5 | cm391 | mccocoful | 0.76167 | 7 (0.89711) | https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/5th-place-by-standing-on-the-shoulders-of-giants | DCC v3 + EUPMC lookup with train-derived family filter; XML-structured context + metadata into Qwen2.5-32B-AWQ and Qwen3-32B-AWQ, majority vote with CPU rules |
| 6 | Team | cakiki, conormacamhlaoibh, samson8 | 0.75866 | 2 (0.91080) | https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/6th-place-solution | DCC4 + EUPMC lookup; pure ordered heuristic rules for type (no model) |
| 7 | 反卷局重新起航 | pingfan, simingtan, zxc123cc, xuanmingzhang777, tonyarobertson | 0.75232 | 5 (0.90413) | https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/7th-place-solution | DCC v4 + EUPMC lookup; author-overlap DOI rule; Gemini-labelled DCC v4 distilled into 7B/14B classifiers, 3-model vote |
| 8 | zhang2dog | zhang2dog | 0.74574 | 13 (0.86573) | https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/8th-place-solution | Regex candidates filtered by DCC/EUPMC; fine-tuned Qwen2.5-32B-Instruct for DOI type, prompted Qwen2.5-32B with table context for accessions |
| 9 | flash attention | yeoyunsianggeremie | 0.74556 | 14 (0.85390) | https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/9th-place-solution | DCC v4 eupmc + v3; deposit/submit rule gated by Qwen3-Embedding similarity for accessions; 0-shot Qwen2.5-32B multi-step DOI pipeline |
| 10 | Khoi Nguyen | suicaokhoailang | 0.74105 | 19 (0.83063) | https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/tricks-10char | DCC v3 DOIs + DCC v4 eupmc accessions, LB-probed prefix list, all SAMN Primary; no model |
| 11 | Have Hope | patricklo01, junhaowang0182, tang0310, fhfdgrhghj, phenixalano | 0.73952 | 3 (0.90950) | https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/11th-place-solution | Regex + DCC v4 + DataCite-prefix filter with LLM re-extraction; author overlap for reference DOIs; keyword-gated LLM for accessions (overfit) |
| 12 | choya & Okumura & takai | chome0910, yukiokumura1, takai380 | 0.73197 | 6 (0.90091) | https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/12th-place-solution | DCC v4 datacite + EUPMC; 0-shot Qwen2.5-72B-AWQ with corpus/metadata features for DOIs; BioSample date rule for accessions |
| 13 | RB, NB, BD | rashmibanthia, benbla, nbroad | 0.73164 | 15 (0.84976) | https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/13th-place-solution | Two pipelines (GROBID XML + DCC for DOIs, provided XML + EUPMC for accessions), Qwen2.5-32B via vLLM for both |
| 14 | Takashi Someya | takashisomeya | 0.73066 | 11 (0.88401) | no writeup found | unknown |
| 15 | idk | guoooooooss | 0.72799 | 4 (0.90623) | https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/shake-down-solution | DataCite + EUPMC + BioSample/GSE metadata; GROBID authors; multi-stage LLM author-ownership decision |
| 16 | Viktoria | viktoriaskorik | 0.69469 | 26 (0.76429) | no writeup found | unknown |
| 17 | wolf | tangtang1999 | 0.68027 | 23 (0.79749) | no writeup found | unknown |
| 18 | hamlet | ghosthamlet | 0.66881 | 18 (0.83575) | no writeup found | unknown |
| 19 | Ebi | ebinan92 | 0.66631 | 8 (0.89322) | no writeup found | unknown |
| 20 | Kawsar Hossain | kawchar85 | 0.65779 | 20 (0.81806) | https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/20th-place-solution-regex-qwen-deberta-for-mdc | Regex only (no corpus lookup); DeBERTa FP filters; Qwen2.5-3B QLoRA accession classifier; Light-R1-14B-AWQ DOI type on LLM-consensus labels |
| 21 | M & K | mbilichenko, ksenikas | 0.65236 | 33 (0.74777) | no writeup found (partial forum post 606764) | DCC v3/v4 extraction; LLM GBIF-species check + author cross-check for DOI type |

Gold appears to end at rank 12 (13th place says they lost gold "by one rank").

## Per-team synthesis

### Ranks 1 to 4 (bank covers these; additions only)
- Rank 1's approach was partly public before the end: 5th place quotes a mid-competition comment by keakohv (https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/600330) describing the DOI-only pipeline (DCC v3 2025-02-01 DataCite pairs, keep pairs whose DOI appears in the PDF/XML, metadata features, Primary/Secondary classifier; DOI-only LB 0.27). 5th place, and a commenter on the 20th-place thread, say they rebuilt their pipelines after that comment. This is the competition's "giant".
- Nothing else missing from the bank for ranks 1 to 4.

### Rank 5: cm391 (mccocoful, solo)
Source: https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606769
- Framing: early work tried to find every possible accession, which was "not the point"; the competition began once keakohv's comment and a host comment on the welcome post (https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/584337) revealed that (1) all labelled accessions come from Europe PMC TextMinedTerms and DCC v3, and (2) human labellers judged Primary/Secondary from the PDF only.
- Retrieval: download all accessions into an (article_id, dataset_id) table; compare with train_labels to find families present in the sources but not in the labels, and filter them. The hardest family was ENA-style `gen.csv`; he dropped every prefix that hit in training but was absent from train_labels. This CPU-only lookup scored 0.82 public / 0.70 private and was published as a notebook (https://www.kaggle.com/code/mccocoful/mdc-cpu-submission).
- Context engineering from XML: locate the container element holding the ID; if it has a tag id, collect every place that tag id is referenced (handles IDs that only live in a table or the reference list); for tables include only the row containing the ID.
- Scope: only DOIs and SAMN were classified by LLM, because they are roughly 50/50 Primary/Secondary; most other accession families are simply Secondary. Metadata (dataset title/authors from the registries, article title/authors/date) was added to the prompt.
- Prompt: Owner / User / None rubric, "one author overlap is usually enough to assume Owner", answer in `\boxed{}`.
- Inference: Qwen2.5-32B-Instruct-AWQ and Qwen3-32B-AWQ, answers parsed from `\boxed{}`, majority vote with the CPU rule submission.
- Final trick: drop accessions in an article when its DOI is one that appears in both EUPMC TextMinedTerms and DCC (polars code in post). Did not improve LB; kept because CV said so; CV was "relatively stable" for him.
- Distinguishing idea: reconstruct the labellers' view (structured XML context plus registry metadata) rather than train anything.

### Rank 6: Team (cakiki, conormacamhlaoibh, samson8)
Source: https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606867
- Data: DCC4, all DOIs plus accessions with source "eupmc" (excluding GCA, HGNC, dbGaP for public FPs); Europe PMC TextMinedTerms accessions (excluding GO, RRID, OMIM). NCT / EudraCT / EBiSC / ChEBI / EFO did not change public LB; probing showed those articles were in the private split, so one final sub excluded them for diversity. Both final subs scored the same private, implying those articles were sparsely labelled ("Missing").
- Retrieval: DOI kept if the unnormalised DOI appears in the PDF text (XML too for eLife); for Dryad/Zenodo/F1000 also match the unversioned DOI, the Zenodo record URL, or the base record. Accessions kept if found in PDF text or XML, PDF only for SAMN.
- Type, DOIs (ordered rules): default Primary; Secondary if the DOI has entries on multiple dates and this article is a later one; Secondary if the DOI is only in the reference section; Secondary if more than 4 DOIs found; Secondary if paper authors do not overlap the citation right after the DOI; Primary for Dryad/Zenodo/F1000; Primary on self-citation author overlap in the same sentence; Primary on deposit phrases ("we have deposited", "have been submitted").
- Type, accessions: default Secondary; SAMN Primary (one sub) or SAMN Primary only near BioSample/BioProject context phrases (other sub); Primary if the ID shares a sentence with deposit phrases.
- Quirk: articles with "eLife" in the ID skip all DOIs, "elife" skip all accessions (train observation).
- Distinguishing idea: a zero-model, fully ordered rule list reached 6th private (2nd public).

### Rank 7: 反卷局重新起航 (pingfan, simingtan, zxc123cc, xuanmingzhang777, tonyarobertson)
Source: https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606757
- Recall: DOIs from DCC v4 by article_id, kept if present in the article; accessions from Europe PMC TextMinedTerms ("all ACC IDs in the training set are included"), kept if present; otherwise fall back to normal extraction. Filters drop `GCA_`, anything containing `/` or `:`, GO:, HGNC:, RRID:, pure digits, NCT. Accessions only, SAMN Primary, rest Secondary: LB 0.783.
- DOI type: author overlap between article and dataset where authors are available; otherwise a 7B model distilled from DCC v4 mentions labelled by Gemini, using context.
- Accession type: DCC v4 + train labelled by Gemini, distilled into 7B and 14B models; three models combined. Strategy 1: SAMN fixed Primary, EPI fixed Secondary, rest majority vote (better public). Strategy 2: SAMN Primary if any model says Primary, EPI Secondary if any says Secondary, rest majority (better private).
- Unselected: author lookup for EPI_ IDs, public 0.850 / private 0.771 (would have beaten their final 0.75232).
- Model family and fine-tuning recipe not stated (UNVERIFIED that the 7B/14B are Qwen).
- Distinguishing idea: Gemini as teacher on the external corpus, distilled into small local models.

### Rank 8: zhang2dog (solo)
Source: https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606768
- DOIs: whitespace-tolerant regex `10\s*\.\s*\d{4,9}\s*/\s*\S+`, filtered by the Data Citation Corpus; type by a fine-tuned Qwen2.5-32B-Instruct on paper abstract, data section and DOI context (fine-tuning raised LB).
- Accessions: a large family regex (ArrayExpress, BioModels, BioSample, CATH, Cellosaurus, ChEMBL, ComplexPortal, EMPIAR, Ensembl, HPA, InterPro, Pfam, PRIDE, UniProt, RefSeq, RefSNP, BioProject, GEO, GISAID, PDB, dbGaP, Reactome, Rfam, BRENDA, 1000 Genomes, ENA), with HGNC, ORPHA, EudraCT, NCT, OMIM, ChEBI, EFO, etc. commented out; filtered by Europe PMC; type by prompted (not fine-tuned) Qwen2.5-32B-Instruct with ID context, table information (caption, footnote, headers, 3 lines either side), abstract and data section. Prompt engineering raised LB.
- Fine-tuning details (LoRA, lr, epochs) not stated.
- Distinguishing idea: table-aware context for accession prompts.

### Rank 9: flash attention (Geremie Yeo, solo gold)
Source: https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606743 ; code https://github.com/bogoconic1/9th-place-kaggle-mdc-finding-data-references ; notebook https://www.kaggle.com/code/yeoyunsianggeremie/9-mdc-final-solution-notebook
- Accessions: DCC v4 (eupmc only). Rule: Primary if the sentence contains "deposit" or "submit" (captures ~95% of Primary), then filter false hits with Qwen3-Embedding-0.6B cosine similarity to a seed "DNA Deposition ... GenBank: PRJNA... BioSample: SAMN..." sentence, threshold 0.667 (tuned to match all such training sentences without matching table noise). Post-processing: cap ENA/protein predictions at 64 per article (train had 74/89/111 FPs in such articles); delete IDs falling inside a range in the PDF (e.g. KT123456-KT123489).
- DOIs: DCC v3. Qwen2.5-7B extracts DOI links with context; Qwen2.5-32B extracts title/authors/year from the paper head; Qwen2.5-32B classifies data vs literature, then Primary vs Secondary (prompt uses metadata + 500 chars before the DOI); all Dryad Primary; for Dryad, add corpus IDs not found in the text as Primary.
- No fine-tuning: GPT-5, Claude 4 and Grok 3 scored similar to Qwen2.5-32B-Instruct on his CV, so synthetic-data fine-tuning was dropped; 72B too slow and not better.
- Final subs: 0.853 / 0.729 (all BioSample/BioModels Primary) and 0.797 / 0.745 (without); best unselected 0.802 / 0.750.
- Did not work: pub-date vs dataset-date feature; chunked NER sweep with Qwen 32B; dropping accessions for articles with a predicted DOI (score fell); reasoning; Marker; semantic/keyword search for context.
- Distinguishing idea: rule plus embedding gate instead of a classifier, after deciding the accession labels were too noisy to learn.

### Rank 10: Khoi Nguyen (solo)
Source: https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606740
- All DOIs from DCC v3; all accessions from DCC v4 eupmc source, else matched by known prefixes. One accession prefix gives +0.08 on both LBs when removed (identity unknown to him); the second most popular new prefix should be whitelisted. All SAMN Primary. Many modelling attempts, "nothing beat probing". Short post; the bank already has this.

### Rank 11: Have Hope (patricklo01, junhaowang0182, tang0310, fhfdgrhghj, phenixalano)
Source: https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606940
- DOI extraction: regex candidates; full DataCite metadata download to build a set of dataset DOI prefixes; keep candidates in DCC v4 or matching the prefix list; for prefix matches absent from v4, feed the text chunk to an LLM to re-extract, then re-filter.
- DOI type: main-text DOIs by semantic context; reference-section DOIs Primary if dataset authors overlap the paper authors, paper authors extracted by an LLM from the first 2,000 chars.
- Accessions: regex, then rule-based keyword patterns gate which chunks go to an LLM. They believe this overfit: a pure LLM approach without rules scored 0.80 (their stated numbers "LB 0.80, Public LB 0.76" look like private 0.80 / public 0.76, UNVERIFIED) and was not selected.
- Model names not stated. Public 3rd to private 11th.
- Distinguishing idea / lesson: the hand-written gating rules were the overfit component.

### Rank 12: choya & Okumura & takai (chome0910, yukiokumura1, takai380)
Source: https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606745
- Candidates: DOIs from DCC v4 (https://zenodo.org/records/16901115), datacite source only, figshare dropped; accessions from EUPMC TextMinedTerms, dropping IDs with `:` or GCA.
- DOI type: zero-shot Qwen2.5-72B-Instruct-AWQ, forced A/B choice by larger logit. Input: ~3,000-char context; corpus features (number of citations of the dataset, whether this is the first citing paper, citations before, days from dataset release to paper publication); paper and dataset title, authors, abstract. Prompt text is in the comments. Adding metadata contributed most.
- Accessions: rule only. Everything Secondary except BioSample; BioSample Secondary if more than three years elapsed between ID issuance and paper publication, else Primary.
- Did not work: LLM type classification for accessions.
- Distinguishing idea: corpus-graph features (first citer, elapsed days) fed as text into a 72B zero-shot prompt.

### Rank 13: RB, NB, BD (rashmibanthia, benbla, nbroad)
Source: https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606979
- Two pipelines. DOIs: start from DCC (https://zenodo.org/records/16901115), parse GROBID-generated XML for structured citations, extract author/abstract/title/section features, classify with Qwen2.5-32B under vLLM. Accessions: provided XML only (every train accession is in the XML, so PDF-only articles have no accessions), EUPMC TextMinedTerms, PMCID to DOI mapping, exclude GO:, HGNC:, GCA_, and `10.`; Qwen2.5-32B; drop duplicates and low-confidence predictions.
- Merge: drop figshare and HEPData (only "Missing" in train), +0.004 public.
- Insights: DOI articles and accession articles are mutually exclusive in train; a submission exploiting this scored 0.742 private (gold) but was not selected because it did not help public. All IPR (InterPro) citations Secondary. DOIs appear in PDF only, XML only or both.
- Did not work: fixing DOI version suffixes; fine-tuning a classifier; regex without external data; Qwen 72B or 14B; labelling open-access data with GPT-OSS.
- Distinguishing idea: GROBID structure for DOI context; XML-only rule for accessions.

### Rank 14: Takashi Someya
No writeup found on the forum topic list or by web search (GitHub/Zenn/Medium/Qiita).

### Rank 15: idk (guoooooooss, solo, titled "shake-down solution")
Source: https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606751
- Downloaded DataCite corpus and Europe PMC data into two (paper, citation) tables; also downloaded full DataCite metadata and all BioSample and GEO (GSE) metadata to extract citation IDs, author names and other metadata.
- GROBID for each paper's author list and title; a multi-stage LLM decision on whether the dataset authors match the paper authors (Primary vs Secondary). Model not named.
- Good CV and public (4th public), 15th private. He suspects GSE handling caused the drop: marking all SAMN Primary or excluding GSE entirely might have won; author-name matching for accessions was meant to reduce risk but did not.
- Lesson: extending author matching from DOIs to accession families (GSE) generalised worse than the blunt SAMN rule.

### Rank 16: Viktoria (viktoriaskorik)
No writeup found.

### Rank 17: wolf (tangtang1999)
No writeup found.

### Rank 18: hamlet (ghosthamlet)
No writeup found.

### Rank 19: Ebi (ebinan92)
No writeup found. Public 8th to private 19th.

### Rank 20: Kawsar Hossain (kawchar85, solo)
Source: https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606746
- Accessions: regex candidates; fine-tuned DeBERTa as FP filter; some ID types forced Secondary; the rest classified by Qwen2.5-3B (base) fine-tuned with QLoRA, input = identifier + context in a structured template with cues, representation = global mean pooling + target-span mean pooling, span-gated GELU classifier head.
- DOIs: regex prefix `10.\d{4,9}` with non-data prefixes removed; Qwen3-4B detects DOIs in noisy windows; DeBERTa FP filter; Qwen2.5-14B extracts paper title and authors; Light-R1-14B-AWQ decides with rules (data vs non-data, author overlap if available, else context cues, else None).
- Labels: training data from DCC v3.0 and v4.1 mentions, labelled by consensus of GPT-5, Qwen3-30B-thinking, Qwen2.5-72B and LLaMA-70B. Planned Light-R1 fine-tuning for DOIs not completed.
- In comments: the submission used regex only, no corpus lookup; trying corpus v3 lowered his LB (cleaning issue suspected); DCC accessions had odd FPs (e.g. `ECE3123` for article `10.1002_ECE3.123`); he missed the EUPMC accession database. Worked a few hours a day after a 9 h job.
- Distinguishing idea: the best pure-extraction (no label-generator lookup) pipeline in the top 20.

### Rank 21: M & K (mbilichenko, ksenikas)
No solution writeup. Maksym Bilychenko's post "Alternative approaches behind the leaderboard" (https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606764): success came down to extracting DOIs and accessions from the corpus (V3 and V4) and separate Primary/Secondary classifiers for DOIs and accessions; worked for them: an LLM check of GBIF references (read the introduction to see whether the species is newly introduced, which makes it Primary) and author cross-checks; early regex approaches gave many FPs. Public 33rd to private 21st.

### Below top 20 (for contrast)
- 22nd (alihaiderahmad's team, https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606794): "pure ML" without the corpus (regex highlight, Qwen 7B/14B extraction from an 80-char window, fine-tuned Qwen 14B non-data filter and 4-prompt type classifier) scored 0.80 public / 0.613 private; adding the MDC corpus filter gave 0.846 / 0.647. Argues corpus lookup is not a usable solution for new papers; commenters reply the host updates DCC and EUPMC continuously.
- Another "22nd Place Solution" topic (https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606822, Muhammad Ibrahim's team): regex + fine-tuned SciDeBERTa, trained on rdmpage's unofficial labels. Rank attribution not verified against the board (UNVERIFIED; two different posts claim 22nd).

## Technique frequency matrix

Y = used in final solution, N = not used / tried and dropped, ? = not stated. Ranks 14, 16 to 19 have no writeup (all ?). Rank 21 not a column. Ranks 1 to 4 cells come from the bank.

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DOI candidates from DCC or DataCite dump (label-generator lookup) | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | ? | Y | ? | ? | ? | ? | N |
| Accession candidates from EUPMC TextMinedTerms / DCC eupmc | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | ? | Y | Y | ? | Y | ? | ? | ? | ? | N |
| Family / prefix blacklist or whitelist (GCA, HGNC, GO, RRID, figshare...) | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | ? | Y | Y | ? | ? | ? | ? | ? | ? | Y |
| LB probing to choose families | Y | ? | ? | ? | ? | Y | ? | ? | ? | Y | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Accession type mostly by rule (default Secondary, SAMN/BioSample rule) | N | Y | Y | Y | Y | Y | Y | N | Y | Y | Y | Y | N | ? | N | ? | ? | ? | ? | N |
| Author / metadata overlap for DOI type | Y | Y | Y | ? | Y | Y | Y | N | Y | N | Y | Y | Y | ? | Y | ? | ? | ? | ? | Y |
| Any LLM in type classification | Y | Y | N | Y | Y | N | Y | Y | Y | N | Y | Y | Y | ? | Y | ? | ? | ? | ? | Y |
| Zero-shot / prompted LLM | Y | N | N | N | Y | N | N | Y | Y | N | ? | Y | ? | ? | ? | ? | ? | ? | ? | Y |
| Fine-tuned LLM (LoRA/QLoRA/full) | N | Y | N | Y | N | N | Y | Y | N | N | ? | N | N | ? | ? | ? | ? | ? | ? | Y |
| Encoder or tabular classifier (DeBERTa/BERT/CatBoost) | Y | Y | Y | N | N | N | N | N | N | N | ? | N | N | ? | ? | ? | ? | ? | ? | Y |
| Synthetic or LLM-teacher labels on external papers | N | N | N | Y | N | N | Y | ? | N | N | ? | N | N | ? | ? | ? | ? | ? | ? | Y |
| Largest LLM 32B or bigger | Y | N | N | Y | Y | N | N | Y | Y | N | ? | Y | Y | ? | ? | ? | ? | ? | ? | N |
| AWQ quantisation stated | Y | ? | N | ? | Y | N | ? | ? | ? | N | ? | Y | ? | ? | ? | ? | ? | ? | ? | Y |
| Logit-constrained answer (A/B choice) | Y | ? | N | ? | N | N | ? | ? | ? | N | ? | Y | ? | ? | ? | ? | ? | ? | ? | ? |
| Ensemble / vote / cascade of models | Y | Y | Y | Y | Y | N | Y | N | N | N | ? | N | N | ? | ? | ? | ? | ? | ? | N |
| Drop accessions when article has DOIs (or vice versa) in final | Y | Y | Y | ? | Y | N | ? | ? | N | ? | ? | ? | N | ? | ? | ? | ? | ? | ? | ? |
| GROBID parsing | ? | ? | ? | ? | ? | N | ? | ? | ? | ? | ? | ? | Y | ? | Y | ? | ? | ? | ? | ? |

Notes: "drop accessions" for rank 5 is the narrower filter (only when the DOI appears in both EUPMC and DCC). Rank 4 "accession type by rule" = several families forced Secondary. Rank 7 accession type is rule for SAMN/EPI plus 3-model vote for the rest. Rank 11 rule gate then LLM. Rank 20 "blacklist" = non-data DOI prefixes and forced-Secondary ID types.

Counted summary (15 teams with writeups: ranks 1 to 13, 15, 20):
- 14 of 15 looked candidates up in the label generator (DCC or DataCite) for DOIs; the exception (20th) is the lowest-scoring writeup.
- 13 of 15 used EUPMC TextMinedTerms or DCC eupmc for accessions (11th unknown, 20th did not).
- 13 of 15 filtered families or prefixes (11th and 15th not stated); 3 say they probed the LB to pick them (1, 6, 10).
- 10 of 15 typed most accessions by rule (default Secondary plus a BioSample/SAMN rule).
- 12 of 15 used author or metadata overlap for DOI type.
- 12 of 15 used an LLM somewhere in type classification; 3 of the top 10 used none (3rd DeBERTa, 6th and 10th pure rules).
- 5 of 15 fine-tuned an LLM (2, 4, 7, 8, 20); 6 of 15 used zero-shot prompting (1, 5, 8, 9, 12, 20; 8 does both).
- 3 of 15 trained on synthetic or teacher-LLM labels (4 agent, 7 Gemini, 20 GPT-5/Qwen/Llama consensus).
- Qwen2.5 was the LLM backbone for nearly everyone who named one (1, 4, 5, 8, 9, 12, 13, 20); MedGemma (2), Qwen3 (5, 9 embedding, 20) and Light-R1 (20) are the others.

## Consensus vs divergence

Table stakes (nearly every top-20 writeup):
- Retrieve candidates from the sources the labels were built from (DCC versions for DOIs, Europe PMC TextMinedTerms for accessions), keep only IDs that literally occur in the PDF/XML. Pure extraction without this lookup (20th, 22nd) lost 0.03 to 0.15 private.
- Remove families the labels never contain (GCA, HGNC, GO, RRID, figshare, HEPData, CCDC, `:`-style IDs).
- Treat accessions as mostly Secondary, with a BioSample/SAMN Primary rule.
- For DOIs, the strongest signal is ownership: author overlap and metadata similarity between paper and dataset, not the sentence context.

What separated the top 5 from 6 to 20:
- Top 5 all spent model capacity on the DOI and SAMN type decision with real metadata (CatBoost on Crossref/DataCite fields, MedGemma with DataCite abstract BM25, DeBERTa with titles, agent-labelled Qwen cascade, Qwen with registry metadata). Ranks 6 and 10 stopped at rules; ranks 11 to 15 added LLM layers on top of hand rules that overfit the public 30%.
- Top 3 came from public 9th to 12th; teams ranked public 1st to 6th mostly fell (4th, 6th, 11th, 15th, 12th). Less public-LB tuning correlated with private rank.

Contradictory findings:
- Drop accessions in articles that have DOIs: helped 1st (https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606853), 2nd (https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606786), 3rd (drop DOIs instead, https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606879), and 13th's unselected mutual-exclusivity sub would have been gold (https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606979); 9th says the score decreased (https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606743).
- Fine-tuning vs zero-shot: 8th says fine-tuning Qwen2.5-32B raised LB for DOIs (https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606768) and 4th gained ~0.02 from synthetic warm-up; 13th says fine-tuning a classifier did not work (606979) and 9th found frontier models no better than zero-shot Qwen2.5-32B, so skipped fine-tuning (606743).
- Bigger model: 12th used Qwen2.5-72B-AWQ zero-shot and 4th routed the hardest 10% to 72B; 13th says 72B did not help and 9th found 72B too slow and not better.
- LLM for accession type: 1st (Qwen2.5-Coder-32B) and 8th (prompted Qwen2.5-32B) used it; 12th lists it under "not worked" (https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606745) and 9th chose a rule because labels were too noisy.
- All SAMN Primary: 10th and 2nd use it; 9th's sub with all BioSample/BioModels Primary was its worse private (0.729 vs 0.745); 12th uses a 3-year date rule instead; 15th thinks not using it cost the win (https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606751).

## Aggregated "what did not work"

- Marker PDF parser (better text, too slow or no gain): 1, 9.
- Qwen for DOI type or as reviewer of CatBoost: 1.
- DataCite relation types (IsSupplementTo) as features: 1 (contrast: 2nd uses isSupplementTo as a Primary rule).
- Article-context features for DOI type: 1.
- Fancier or Chinese prompts: 1.
- Qwen2.5-7B with a classification head (unstable): 3.
- Publication date vs dataset creation date: 9 (contrast: 12th's 3-year BioSample rule and elapsed-days feature).
- Chunked NER sweep with Qwen 32B to find IDs: 9.
- Reasoning before the answer: 9.
- Semantic or keyword search for context: 9.
- Dropping accessions when a DOI is predicted: 9 (worked for 1, 2, 3, 13).
- Synthetic-data fine-tuning (frontier LLMs no better than zero-shot): 9.
- LLM type classification for accessions: 12.
- Fixing DOI version suffixes: 13.
- Fine-tuning a classifier: 13.
- Regex extraction without external data: 13, 20, 22.
- Qwen2.5 72B or 14B instead of 32B: 13 (and 9 for 72B).
- Labelling open-access data with GPT-OSS: 13.
- Hand-written keyword gates before the LLM for accessions (overfit; pure LLM was better): 11.
- Author matching extended to GSE/BioSample accessions: 15.
- Public-LB-best sub selection (SAMN strategy 1 vs 2): 7, 9, 11, 13 all had a better unselected sub.

## Shake-up note

Public scores ~0.85 to 0.92 vs private ~0.65 to 0.80. Biggest falls among the top 20: 15th (public 4th), 19th (public 8th), 11th (public 3rd), 12th (public 6th), 6th (public 2nd), 4th (public 1st). Biggest climbs: 21st (public 33rd), 16th (public 26th), 10th (public 19th), and the winners (1st public 9th, 2nd public 10th, 3rd public 12th).
Explanations given in writeups: 11th blames rule gates for accessions fitted to public; 15th blames GSE handling and author-matching for accessions; 6th found families (NCT, EudraCT, EBiSC, ChEBI, EFO) present only in the private split and sparsely labelled; 13th and 7th and 9th each left a better private sub unselected because it did not win public; 10th's prefix choices were LB probes. The common cause: family whitelists and accession rules tuned on a 30% public split of noisy, partially labelled data.

## UNVERIFIED

- Ranks 14, 16, 17, 18, 19 (and 21 beyond a forum post): no writeup found in the forum topic list or via web search; approaches unknown.
- 7th place: base model family of the distilled 7B/14B models and the fine-tuning recipe not stated.
- 8th place: fine-tuning recipe (LoRA rank, lr, epochs) not stated.
- 11th place: model names not stated; the "LB 0.80 / Public LB 0.76" numbers for their pure-LLM run are read as private/public, not confirmed.
- 13th place: whether Qwen2.5-32B was zero-shot is inferred from "fine-tuning classifier did not work".
- 15th place: LLM used not named.
- Two different topics claim 22nd place (606794, 606822); not checked against the board.
- Matrix "?" cells mean not stated, not "not used".

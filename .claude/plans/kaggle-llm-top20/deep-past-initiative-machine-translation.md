# Deep Past Challenge - Translate Akkadian to English: top-20 writeup synthesis

- Slug: `deep-past-initiative-machine-translation` (competition id 121150, forum id 9344890)
- Existing coverage: `.claude/plans/kaggle-llm-writeup-bank-part3.md`, H2 `## Deep Past Challenge - Translate Akkadian to English (discovered; LLMs as data engineers)` (facts, metric, top-6 table, ranks 1 to 3 summarised). Not repeated here.
- Leaderboard from `competitions.LeaderboardService/GetLeaderboard`, pulled 2026-10-28. Public = 34% of test. Metric = geometric mean of corpus BLEU and chrF++. No ties in the top 20.
- **Rank-label caveat:** several self-titled places disagree with the API by one (vinpro's "[20th]" is API rank 19; Karl's "[21th]" is API 20), while "[22nd]" epikt and "[24th]" Geremie Yeo match the API. A "17th Place Solution" by kirito1sasa (684332) belongs to no team in the API's top 60. Treat self-titles as approximate; ranks below are the API's.

## Top-20 table

| Priv rank | Team | Members | Private | Public (rank) | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | DataTech Club | guoqingu, hongori, leechengju, xiakie | 42.8957 | 41.6694 (1) | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684353 (bank) | Discard train.csv; GLM-OCR + Gemini-3.0-pro PDF extraction; 11x ByT5-XL, 10 candidates each, MBR |
| 2 | KE WU | wukeneth | 41.9311 | 41.0478 (3) | 684345 (bank) | Single ByT5-large; three LLM alignment pipelines (Breaker/Fixer/Generator) + ~60 OCR'd publications |
| 3 | How I Learned to Stop Worrying | darraghdog, conjuring92 | 41.5091 | 41.6451 (2) | 684425 (bank) | Claude-generated vocab/grammar drills for CPT; ByT5-L + XL; Qwen3-8B pairwise reward-model selection |
| 4 | Wisdom and Vitaly | vitalykudelya, anaphase21 | 40.8899 | 39.9734 (8) | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/685925 | 14.5k pairs via Gemini 3.1 Flash (AI Studio UI) + Google Sheets translate; mT5-L/ByT5-L/XL, checkpoint averaging, chrF++ MBR over 10 |
| 5 | ebi & okumura | ebinan92, yukiokumura1 | 40.8077 | 40.4156 (4) | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684847 | ~20k PDF pairs via Qwen3.5-Plus VLM; CPT → EvaCun → PDF; back-translation; single fold-averaged ByT5-XL |
| 6 | M & J & M | wscmisx, marisyn, jackvd | 40.7360 | 40.0310 (7) | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684231 | see below |
| 7 | Yurnero | samson8 | 40.4372 | 40.2765 (6) | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684215 | see below |
| 8 | Hrithik Reddy | hrithik22122 | 40.2797 | 40.3472 (5) | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684329 | see below |
| 9 | Eleftheria 14 | eleftheria14 | 40.1770 | 39.5449 (11) | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/685928 | see below |
| 10 | Kelvin | xyzdivergence | 39.9717 | 38.5589 (19) | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684211 | see below |
| 11 | kwa | ikaroskaggle | 39.8929 | 39.4983 (12) | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684429 | see below |
| 12 | Ha-ar-pì-im | hasinarazafindrazaka, nathanaelrk | 39.7225 | 38.9239 (15) | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684866 | see below |
| 13 | DataBoom | darkartitry, mjjsty, xxbar19 | 39.6175 | 38.4482 (22) | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/686956 | see below |
| 14 | H&B&Z | pingfan, zxc123cc, leehann | 39.5518 | 38.4536 (21) | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684236 | see below |
| 15 | It's showtime! | aman1391, kpriyanshu256 | 39.2971 | 39.6487 (10) | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684819 | see below |
| 16 | John Doe #2 | johndoe90843 | 39.2743 | 39.1694 (14) | no writeup found | - |
| 17 | lost in transliteration | rashmibanthia, liuzhangzhen, arunodhayan, nbroad, tonylica | 39.1371 | 38.7083 (16) | no writeup found | - |
| 18 | JoJoImpact | yiruicui, liaoyuqian, jonathanxuyf, yuanchuan64 | 39.1357 | 38.5569 (20) | no writeup found | - |
| 19 | vinpro | vinprofessionalapps, evanastase | 38.9826 | 38.6004 (18) | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684381 (titled "20th") | see below |
| 20 | Karl | dreamweaver | 38.9230 | 38.3640 (23) | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684267 (titled "21th") | see below |

## Per-team synthesis

### Ranks 1 to 3
Covered in the bank. Not re-read for this file beyond the API facts above.

### 4th - Wisdom and Vitaly (vitalykudelya, anaphase21)
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/685925 ; code: submission notebook https://www.kaggle.com/code/anaphase21/vitaly-deep-past-challenge-submission-code-a93e45 , training https://github.com/Anaphase21/Deep-Past-Challenge-2026/blob/main/code/DPC26_Training.ipynb
- **Data:** discarded train.csv as soon as host PDFs appeared (all of train.csv was in them); built **14.5k training pairs + 392 validation** (AKT 6d held out). Barjamovic's *Historical Geography of Anatolia* was first done with Tesseract + PyMuPDF (superscripts, indentation) but took two weeks of manual hallucination fixes for one book, so they switched to **Gemini 3.1 Flash in the AI Studio UI** (no API): 15 to 20 page images at a time, JSON pairs, superscripts in braces, subscripts with underscores.
- **Non-English sources** (AKT 1-4, 7, 9, 10, Hecker): translated with the `=GOOGLETRANSLATE()` Google Sheets formula, not an LLM.
- **Alignment:** heuristic splitting (punctuation, ~34 words per sentence, paragraph breaks, per-document rules), then manual correction with Gemini for unsure splits.
- **Normalisation:** map every transliteration style to one form at character and word level (e.g. GUŠKIN → KÙ.GI), applied to test too.
- **Models:** mT5-Large, ByT5-Large, ByT5-XL (bf16; Large CV slightly better than XL), a ByT5-Large with dropout 0.2; **length_penalty 1.7** in eval and inference.
- **Checkpoint weight averaging** (e.g. 50/50 of two top checkpoints) beat any single checkpoint.
- **Inference:** 10 candidates (mT5-L 3, ByT5-XL 1, ByT5-L 3, ByT5-L-dropout 3; beams 5-6) and **chrF++ MBR** (pick the candidate with highest mean chrF++ to the others).
- **Post-processing:** only fractions to Unicode; LB-tuned post-processing tanked public LB.
- **Did not work:** capitalising personal names from OA_Lexicon_eBL ("a-na" is both a name and a preposition); re-splitting to match Sentences_Oare and removing the "translate Old Assyrian into English" prefix (lower CV). Weaknesses: very short inputs, proper-noun capitalisation. A pseudo-labelled + Gemini-corrected set from 7,953 published_texts (9,237 chunks) was built on the last day but never trained (compute).

### 5th - ebi & okumura (ebinan92, yukiokumura1)
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684847 ; synthetic data https://www.kaggle.com/datasets/yukiokumura1/deep-past-challenge-back-translated-synthetic-data
- **PDF extraction (~20,251 pairs from 20 volumes; train.csv had 1,561):** 4-phase VLM pipeline. (1) **Qwen 3.5 Plus (397B)** detects excavation numbers on page images; (2) link them to published_texts.csv (transliterations) and train.csv (translations) with alias matching; (3) extract line-numbered transliterations and range-grouped translations with layout-specific prompts (side-by-side vs sequential), stitching multi-page tablets, fixing misordered PDF pages, auto-validating and re-prompting failures; (4) Turkish/German → English with **Gemini 3.1 Flash Lite** using Levenshtein-retrieved few-shots. Length-ratio filter [0.3, 4.0]. Total API cost ~$15 via OpenRouter.
- **External:** EvaCun / ORACC Akkadian parallel corpus (45k pairs), with immediate word-repeat collapsing.
- **Training (ByT5 variants, best ByT5-XL):** Stage 1: CPT (T5 span corruption) on published_texts transliterations → FT EvaCun → FT PDF data; a reverse En→Akk model trained the same way. Stage 2: pseudo-labels on published_texts; **back-translation** of 10,000 English sentences written by **Qwen3.5-27B** (few-shot). Scaling to 100k: -0.1 public, +0.3 private. Stage 3: from the EvaCun checkpoint, FT on synthetic, then FT on PDF data with **tablet context** (up to 2 previous lines in `[context]…[/context]`).
- **Inference:** CTranslate2 (~5x faster than HF; patched its tensor-parallel bug to run ByT5-XXL int8 on 2xT4, but XXL unused for lack of training time); **fold models weight-averaged** into one; beam search; no test-time ensemble.
- **CV:** 25% of train.csv hold-out stratified by length ratio; token-level validation loss correlated well with LB.
- **Ablation (private/public):** PDF only 35.4/35.1; +CPT 38.9/38.0; +EvaCun 39.7/38.7; +tablet context 39.8/39.2; +synthetic 40.7/40.2; +4-fold weight average 40.8/40.1.
- **Did not work:** LGBM reranker (CV +0.2, LB 0), fine-tuned LLM listwise reranker, sampling-based or tagged back-translation, decoding-time logit-averaging ensembles (gain vanished once models reached ~37).

### 6th - M & J & M (wscmisx, marisyn, jackvd)
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684231 ; clean function https://www.kaggle.com/code/wscmisx/dpc-clean-function ; inference https://www.kaggle.com/code/wscmisx/final-dpc-ensemble-mbr-infer
- **PDF triage into three types:** *image* (OCR layer with artifacts: AKT 1-4, 6a-e, 11a, 12), *text* (clean PyMuPDF layer with fixable per-book character rules, e.g. AKT 5 macron markers), *broken_text* (treated as image). Text PDFs: regex finds tablet headings, heading + block go to an LLM API for document pairs. Image PDFs: tuned, parallelised **GLM-OCR**, regex headings, LLM on heading + image, then a **second image+text pass for fine-grained correction**. Turkish translated with LLM APIs + dictionaries.
- **OARE scraping:** for each train.csv `oare_id`, scrape word definitions from oare.byu.edu, giving a dictionary for most train words; train.csv rows matched to the PDF text to recover the correct form; LLM uses the dictionary to split documents into sentence pairs. **published_texts.csv pseudo-labelled by an LLM prompted with per-word dictionary glosses** (form + senses). publications.csv: fixed page batches to the LLM (lots of data, hard to align).
- **Cleaning:** extensive rule set (NFKC, diacritic simplification, editorial brackets, Unicode fractions, determinative normalisation to `{d}`, `<gap>` unification, translation-side removal of parentheticals and "lit."/"sic", slash-alternative resolution, Roman month numerals, measurement expansion).
- **Training tricks:** **dynamic context sampling** (number of context sentences K ~ Zipf, k_max 10; left/right split ~ Binomial; 20% random same-document context) to make the model robust to misalignment and unknown test lengths; **R-Drop** (alpha 0.8); **PGD adversarial embedding perturbations** (3 steps, eps 0.5, alpha 0.2); EMA; DeepSpeed ZeRO-2, gradient checkpointing. Prompt: "Translate TARGET only. Use CONTEXT only as reference" with CONTEXT BEFORE/AFTER at train and inference.
- **Data ablation (ByT5-large, public/private):** cleaned train.csv only 34.0/34.3; + Turkish→English 36.9/37.0; + high-quality publications 38.7/38.9; + 5-epoch pre-train on all publications 38.6/39.5; + published_texts pseudo-labels in pre-train 38.9/39.3; ByT5-**base** with all of it in formal training 39.3/39.9 (the noisy bulk helped once included).
- **Ensemble:** 15 models (Large/XL/Base on data versions v3 to v6, cleaning variants, a name-augmented model, plus 5 folds of the best base), 4 beams x 4 outputs each, **MBR**, no sampling (sampling inside MBR hurt), 2xT4, 208 min. Final 40.7.
- **Did not work:** weight averaging (worse than single models; replaced by MBR); document-level long-maxlen training; LLM fine-tunes (Qwen, LLaMA; a 14B was "average"); bidirectional training; name/dictionary augmentation gave little single-model gain (kept for ensemble diversity).

### 7th - Yurnero (samson8, solo)
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684215 ("short note")
- Hours of **manual sentence alignment** in a Google Sheet early on (low impact; stopped when LB stopped improving; he notes sentences mostly end with a verb and anchors like um-ma, šu-ma, ma-na, KIŠIB, IGI make alignment easy).
- Training set grown ~4x to **~36k samples:** Turkish→English for non-English AKTs (~6k), pairs from ~19 other PDFs (Innaya, Barjamovic 2011, Stratford 2010, …; ~6k), and published_texts deduplicated, split into ~30-word strings and **pseudo-labelled** (~18k).
- **Models:** size matters (ByT5-small << base << large ≲ XL); final ByT5-XL, 3 epochs (iterated on large). Causal-LM fine-tunes (Mistral, Qwen) were worse than ByT5 but "decent enough for a gold zone"; other T5 variants much worse; T5-Gemma untried.

### 8th - Hrithik Reddy (hrithik22122, solo; first Kaggle competition)
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684329
- **ByT5-XL** (NLLB, mT5 1 to 1.5 points worse). **Two-stage SFT was the biggest contributor:** Stage 1 on **~350k noisy pairs** (dictionary entries, partial sentences) for 3 epochs, global batch 128 (bf16, 16 x grad-accum 4 x 2 devices); Stage 2 on **~65k curated pairs for exactly 1 epoch in fp32** (more caused collapse).
- **100% open-source stack:** publications normalised to English first with **Mistral 14B**, later re-done with **GPT-OSS-120B (fp4)** with better few-shots. PDFs: regular-layout books via **GLM-OCR** + a (previous, current, next) page **sliding window** into GPT-OSS-120B (≥40% of translations spill onto the next page); columnar books via VLM (**GLM-4.7V-Flash** for throughput; Qwen3.5-VL 35B later ran fast enough under vLLM but scored worse on LB). Plain OCR into the publication pipeline lowered LB.
- Replicable on one A100 or 2x RTX 6000.
- **Did not work:** dictionary/RAG pseudo-labelling with open models; name and commodity augmentation; automated or manual sentence alignment; >1 epoch in stage 2; LLM post-processing; **RL (DPO, minimum risk training, PPO, GRPO for seq2seq)**; CV (first 500 train pairs scored 42 to 43.5 for all 40+ models, no correlation with LB).

### 9th - Eleftheria 14 (eleftheria14, solo)
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/685928 ; code https://github.com/Eleftheria14/Deep_Past_Gold_Solution ; submission notebook linked in the post.
- **Data was the biggest lever:** ~**95k pairs** extracted from publications.csv (~950 PDFs) with the **Gemini batch API** (JSON schema: original-language translation kept plus an English version, sentences of 3+ words only, completeness flag); Gemini Vision for scanned PDFs without a text layer (re-extracting PDFs that had host OCR gave no LB gain after cleaning). **30+ iterative cleaning passes**: sample a batch, have **Claude Opus** judge alignment/OCR/format, find the systematic error class, write a programmatic fix plus an **invariant test** as a quality gate, rerun on everything. Claude Code (Opus 4.6) wrote the cleaning scripts. LLM sentence splitting with anchors ("um-ma X-ma", "KIŠIB").
- **Augmentation:** **gap augmentation (~124k pairs)**: Gemini aligns source and target spans and deletes aligned spans from both sides, inserting matching `<gap>` tokens (rules: equal gap counts, no adjacent gaps); **entity-swap** augmentation (~40k) swapping Akkadian names and their English equivalents.
- **Models:** four **ByT5-base in FP32** (byte-level diacritics; BF16 "blurs" š vs s), deliberately diverse: different data mixes, LR 1e-4 or 3e-4, one with encoder freeze after epoch 5, one two-stage base → gap fine-tune; checkpoints include a soup of epochs 3/5/6. Trained on an NVIDIA DGX Spark (GB10), 6 to 10 h per model.
- **Inference:** **translation memory** returns gold for the ~5% of test sources that exactly match train; 12 beams, top 4 per model (16 candidates); **source-grounded MBR**: chrF consensus + lexicon coverage + an SG score that checks numbers, Sumerograms, divine names and gaps in the source are reflected in the candidate (weights per source, e.g. 0.49 chrF + 0.44 SG); adaptive length penalty; 9-step post-processing.
- **CV:** 90/10 hold-out chrF diverged from LB (+4 chrF with no LB gain), so checkpoint selection was LB-driven.
- **Key finding:** single-model multi-checkpoint MBR scored 39.6 public but 38.4 private; 4 diverse models with SG-MBR 39.2 public → **40.1 private** (+2.58 over the single model). "Diverse training runs generalise; diverse checkpoints of the same run don't."
- **Did not work:** ByT5-large (one run, memorised noise); DAPT on ~820k raw Akkadian lines (encoder improved, decoder misaligned, repetition loops, 34.6 vs 38.5); DPO (eval up, LB down); **sampling MBR** (+2.95 on 200 local samples, **-2.7 on LB**).

### 10th - Kelvin (xyzdivergence, solo team name)
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684211
- **External data:** host grammar references and Kültepe tablet PDFs (discussion 668485). **GLM-OCR (0.9B)** per page; **Gemini, GPT and Claude APIs** for sentence alignment, OCR noise removal and normalisation (same pipeline for published_texts).
- **Three stages per model:** (1) **CPT for 3 epochs** on document-level Akkadian text (official + cleaned external); (2) sentence-level SFT (CPT → SFT consistently beat direct FT); (3) best single model pseudo-labels published_texts at sentence level, then retrain.
- **Ensemble + MBR of three:** ByT5-Large 37.4/39.3, ByT5-XL 37.6/39.2, **MADLAD-400-3B-MT** 37.3/38.9 (public/private); ensemble 38.5/39.9. MADLAD added for diversity; mT5 variants also tried. One held-out fold kept out of every stage for validation.
- Biggest riser in the top 20 (public 19 → private 10).

### 11th - kwa (ikaroskaggle, solo)
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684429
- **Pre-training on multilingual pairs:** an LLM turned publications.csv `page_text` into JSON pairs (translation and language may be null); kept English, French, German, Turkish. ByT5-large, Seq2SeqTrainer, ~20 h on one Colab A100: 3 epochs, batch 8, lr 1e-4, max length 512, label smoothing 0.1, EMA 0.999.
- **Fine-tuning:** an LLM segmented train.csv into sentence pairs with a 100 to 400 character constraint (test transliterations are 129 to 267 chars); length-ratio outliers (<0.66 or >2.0) replaced by LLM-corrected translations with few-shots; published_texts also LLM-translated for pre-training. FT 2 epochs, lr 1e-5.
- **Inference:** 3 ByT5-large fine-tuned from different pre-training mixes; candidates from normal beam + group beam search (beat nucleus and epsilon sampling); **MBR with the competition metric itself as utility**; soft length-ratio penalty and PN/GN lemmatisation for a small boost.
- **CV:** LLM-built validation set shared train's style and correlated poorly with LB (val 41.7 to 47.9 vs public 38.0 to 39.0); final 39.4 public / 39.8 private.
- **Did not work:** PN/GN normalisation from OA_Lexicon_eBL; pre-training on eBL_Dictionary; minimum risk training.

### 12th - Ha-ar-pì-im (hasinarazafindrazaka, nathanaelrk)
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684866
- **Google NotebookLM** for all extraction, preprocessing and pseudo-labelling (budget too small for APIs): screenshots of 1 to 5 tablets (1 to 10 images) per batch gave more pairs than larger batches; multiple notebook tabs in parallel; NotebookLM also matched tablet references to aliases in published_texts to recover original transliterations.
- **Custom template data (~11k):** seals, witnesses, "from … to", month eponyms, "owes" statements filled with random onomasticon names and numbers, with 0.2 probability of `<gap>`; train.csv names and numbers replaced by PN/NUM placeholders and randomised.
- **~81k unique pairs:** ~28k human-labelled, ~11k custom, ~3k from publications.csv, rest pseudo-labelled.
- **Single ByT5-base, three stages:** (1) all 81k, batch 8, lr 5e-5, 5 epochs (~35 alone, below a human-only model's ~37); (2) human-labelled 28k, batch 2, lr 1e-5, 2 epochs (38+; +2 to +3 BLEU locally vs human-only); (3) **LoRA on the 1.5k competition train set** to align output vocabulary (~39).
- Plateau at 38.1 broke only after adding publications.csv sources (CMK, TPAK, APU) into stage 1. Did not work: Sentences_Oare_FirstWord_LinNum extraction; a few ByT5-large runs.

### 13th - DataBoom (darkartitry, mjjsty, xxbar19)
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/686956
- **Single ByT5-large**: pre-train on merged public Akkadian-English datasets (HF) + an OCR-derived PDF parallel corpus (dedup, broken-pair filter, `|len(translation) - len(transliteration)| < 70`), then fine-tune on the organiser train set.
- **Test-time source repair:** regex + mappings learned from clean train data to repair corrupted transliteration symbols conservatively (context-sensitive, precision-focused), moving test inputs toward the train distribution.
- **MBR** with sentence-level chrF++ over beam + multi-temperature sampling candidates.
- **Results (private/public):** ByT5-base beam 38.35/37.15; large beam 37.88/37.40; **XL beam 39.59/38.10**; XL+large MBR 39.62/38.28; selected large + MBR (T 0.75/0.85) 39.62/38.45; best unselected large OCR-only pre-train 39.67/38.31.
- Lesson: data, preprocessing and decoding beat scaling; raw uncleaned OCR data much less effective; top beam worse than MBR. Public 22 → private 13.

### 14th - H&B&Z (pingfan, zxc123cc, leehann)
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684236 ("LLM + Online Learning is all you need")
- **LLM-backbone route:** Qwen3, Qwen2.5, Phi-4 and ByT5 tried. Pipeline SFT (knowledge distillation) → RL → **online learning on the test set** (only ~4,000 test sentences).
- **Data:** Gemini 3.1 Flash extracted sentence pairs from publications.csv (SFT on it gave Qwen hallucinations); published_texts pseudo-labelled with **RAG**: Qwen3-8B Embedding retrieves top-3 similar extracted pairs as context (Gemini 3.1 with web tool also used for external data). Qwen3-14B SFT: 37.9 private / 36.1 public. train.csv cleaned with the same RAG approach plus an LLM filter; then **RL** on that data: Qwen3-14B 38.1 / 36.9.
- **Online learning (transductive):** Qwen3-14B, Qwen2.5-14B and Phi-4 pseudo-label the whole test set; train a ByT5-Large on it; use ByT5's outputs to fine-tune the LLMs; iterate. **Self-training (LLM on LLM, ByT5 on ByT5) gave marginal gains; only heterogeneous alternation between architectures helped.** Final = ensemble of the online-trained models. Public 21 → private 14.
- RL algorithm, reward and hyperparameters not stated.

### 15th - It's showtime! (aman1391, kpriyanshu256)
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684819
- Early weeks of focal loss, annotation tricks and architecture changes did not move LB; breakthrough after the host's data-quality discussion (678899).
- **PDF extraction:** **GLM-OCR** two ways (full page = good characters, no structure; Tesseract-cropped lines = layout order), both given to **Kimi K2** (via Groq) to emit aligned pairs, with layout-specific prompts (Turkish line-marker variant). AKT 6 needed a multimodal Llama-4-Maverick pass on super-resolved pages (model deprecated before completion). ~31k PDF samples, extracted iteratively. Claude wrote the cleaning code.
- **Models:** ByT5-small, ByT5-base, **Qwen3-14B**, **Qwen3-4B-Instruct**; **Gemma3-27B** trained on competition + PDF data pseudo-labels and acts as **teacher**: each ByT5 row appears twice, gold label and Gemma translation at reduced weight (KD added ensemble diversity).
- **Ensemble:** 5 ByT5 + 1 Qwen3-14B (T 0.0 and 0.7) + 2 Qwen3-4B, **MBR with chrF++ (word_order=2) sentence score** (suggested by Claude; the competition metric as utility worked worse for them); ~9 h runtime. 10% hold-out CV.
- **Did not work:** ByT5 Large/XL in their setup; CPT+SFT model soup (soups in general failed); training Qwen3.5 for pseudo-labels; PN/GN placeholders.

### 16th - John Doe #2 (johndoe90843), 17th - lost in transliteration (rashmibanthia, liuzhangzhen, arunodhayan, nbroad, tonylica), 18th - JoJoImpact (yiruicui, liaoyuqian, jonathanxuyf, yuanchuan64)
No writeup found for any of the three (forum titles of all 225 topics and comment authors checked). Only trace: rashmibanthia's earlier thread "Translate Gemma" (https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/668621, title only, not read). A "17th Place Solution" by kirito1sasa (https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684332: ByT5-base ensemble, Gemini data pipeline, Optuna-tuned weighted MBR, Qwen LLMs underperformed) cannot be matched to any API team and is excluded from the matrix.

### 19th - vinpro (vinprofessionalapps, evanastase; self-titled 20th)
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684381
- ~80k mixed-quality pairs (sentence-segmented train, **Gemini** extraction/alignment from AKTs and papers plus synthetic translations) and a ~10k "gold" set (clean English, deliberately noisy Akkadian for robustness). Normalisation (gaps, Unicode fractions, sz → š).
- **ByT5-base, 3 phases:** (1) **span-corruption denoising** on all transliterations (noise density 0.15, mean span 15); (2) SFT on the 80k with two seeds (Adafactor, lr 5e-5, bf16, grad-accum 4, label smoothing 0.1); (3) 5 epochs on gold.
- **Soup:** 50/50 of the two phase-2 seeds, then 0.75 x that + 0.25 x the gold model (a low-LR polish failed).
- **Inference:** split >512 inputs at spaces and restitch (+0.02); 12 beams, no_repeat_ngram 20, repetition_penalty 1.6; post-processing: capitalisation regex, terminology fixes ("pašallum gold", "šadduātum tax"), **onomasticon fuzzy matching** (difflib, cutoff 0.7).
- Regret: abandoning ByT5-large (validation set misled them).

### 20th - Karl (dreamweaver, solo; self-titled 21st)
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684267
- Single ByT5-base on one AutoDL GPU, no ensemble. Final 38.9 private / 38.3 public; ~140 submissions, 4 weeks.
- **Ablation:** subscript removal +0.7; **DeepSeek** paragraph → sentence splitting (~6,750 gold rows) +2.0; **Gemini-3 Pro pseudo-labels of published_texts (~40k silver rows) +4.0, the biggest gain**; gold cleaning + folding val into train +0.5; ~7,800 **Gemini synthetic "related paragraph" pairs** (reply letters, counter-claims) with gold weighted above silver +0.5; checkpoint averaging epochs 3-5 +0.1. Trajectory 31.1 → 33.8 → 37.5 → 38.3 → 38.9.
- Training: AdamW wd 0.01, lr 3e-4, 6% warmup + cosine, BF16, max length 512, 5 epochs; beam 4; `~` as gap placeholder.
- **Did not work:** 1024 max length on paragraphs; inference-time context concatenation; DPO; **gold fine-tuning after silver pre-training (decreased scores)**; bidirectional training; MBR for a single model (0.0 to 0.3, not worth 2x time); MADLAD-400-3B (33.1); ByT5-small (~36.5 ceiling); name-swap augmentation; disabling early stopping (37.5 → 36.6).

## Technique frequency matrix

17 of 20 teams have a writeup (ranks 1-15, 19, 20). Cells for 1 to 3 from the bank's summary (? where the bank is silent).

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ByT5 in the final submission | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | ? | ? | ? | Y | Y |
| Largest ByT5 used ≥ XL | Y | N | Y | Y | Y | Y | Y | Y | N | Y | N | N | N | N | N | ? | ? | ? | N | N |
| Other seq2seq in ensemble (mT5, MADLAD) | N | N | N | Y | N | N | N | N | N | Y | N | N | N | N | N | ? | ? | ? | N | N |
| Decoder LLM as translator in final | N | N | N | N | N | N | N | N | N | N | N | N | N | Y | Y | ? | ? | ? | N | N |
| LLM used at inference as selector/judge | N | N | Y | N | N | N | N | N | N | N | N | N | N | N | N | ? | ? | ? | N | N |
| Frontier/large LLM or VLM builds training data (OCR-to-pairs, alignment, translation) | Y | Y | Y | Y | Y | Y | ? | Y | Y | Y | Y | Y | ? | Y | Y | ? | ? | ? | Y | Y |
| ...open-weight only for data (no commercial API) | N | ? | N | N | N | ? | ? | Y | N | N | ? | N | ? | N | Y | ? | ? | ? | N | N |
| GLM-OCR in the PDF pipeline | Y | ? | ? | N | N | Y | ? | Y | N | Y | ? | N | ? | ? | Y | ? | ? | ? | ? | ? |
| Discarded or rebuilt train.csv from PDFs | Y | ? | ? | Y | N | Y | N | ? | N | N | Y | N | N | Y | N | ? | ? | ? | Y | Y |
| Pseudo-labelled published_texts | ? | ? | ? | N | Y | Y | Y | ? | ? | Y | Y | Y | ? | Y | Y | ? | ? | ? | ? | Y |
| Synthetic data (back-translation, LLM drills, templates, gap/entity aug) | ? | Y | Y | N | Y | N | N | N | Y | N | N | Y | N | N | Y | ? | ? | ? | Y | Y |
| External corpora beyond host (EvaCun/ORACC, HF sets) | ? | ? | ? | N | Y | N | N | N | N | N | N | N | Y | N | N | ? | ? | ? | N | N |
| CPT / denoising / noisy pre-training stage before clean SFT | ? | N | Y | N | Y | Y | N | Y | N | Y | Y | Y | Y | N | N | ? | ? | ? | Y | N |
| Neighbouring-line context in the input | ? | ? | ? | N | Y | Y | ? | ? | N | N | N | N | N | N | N | ? | ? | ? | N | N |
| Multi-model ensemble at inference | Y | N | Y | Y | N | Y | N | N | Y | Y | Y | N | N | Y | Y | ? | ? | ? | N | N |
| MBR selection | Y | ? | N | Y | N | Y | ? | ? | Y | Y | Y | N | Y | ? | Y | ? | ? | ? | N | N |
| Weight averaging / checkpoint soup (shipped) | ? | ? | ? | Y | Y | N | ? | ? | Y | ? | ? | N | ? | ? | N | ? | ? | ? | Y | Y |
| CTranslate2 inference | Y | ? | ? | ? | Y | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| RL / DPO / MRT attempted | ? | ? | ? | N | N | N | N | Y | Y | N | Y | N | N | Y | N | ? | ? | ? | N | Y |
| ...and it helped | - | - | - | - | - | - | - | N | N | - | N | - | - | Y | - | ? | ? | ? | - | N |
| Minimal post-processing (argued explicitly) | Y | ? | ? | Y | ? | N | ? | Y | N | ? | ? | ? | N | ? | N | ? | ? | ? | N | N |

**Counted summary (17 teams with writeups):**
- 17 of 17 shipped ByT5 (sizes: XL in 8, Large in 4 more, Base-only in 5). Only 2 of 17 (14th, 15th) put a decoder LLM in the final translation ensemble, and 1 (3rd) used an LLM as the final selector; 7th says LLM fine-tunes were "decent enough for gold" but below ByT5; 6th, 17th-titled and 8th found LLMs worse.
- At least 15 of 17 used an LLM/VLM to manufacture training data (Gemini in 8 teams; also Claude, GPT, Qwen3.5-Plus, Kimi K2, DeepSeek, GPT-OSS-120B, NotebookLM). 2 of 17 (8th, 15th) did it with open-weight models only.
- 5 of 17 named **GLM-OCR** as the OCR front end.
- 9 of 17 pseudo-labelled published_texts; 8 of 17 added synthetic data of some form.
- 9 of 17 had a pre-training/noisy stage before clean fine-tuning; 20th reports the opposite order hurt, and 9th's DAPT failed.
- 9 of 17 ensembled multiple models; 8 of 17 used MBR; 5 shipped weight averaging (2 others say soups failed).
- 5 of 17 tried RL/DPO/MRT; it helped only 14th (on a Qwen LLM, details unstated).

## Consensus vs divergence

**Table stakes:** byte-level ByT5; training data rebuilt or expanded far beyond the 1.5k train.csv pairs from host PDFs and publications using LLM/VLM extraction, sentence alignment and German/Turkish → English translation; heavy normalisation of transliteration conventions (determinatives, subscripts, gaps, fractions); sentence-level training to match the test granularity.

**What separated the top 5 from 6 to 20:**
- *Scale and cleanliness of extracted data* rather than model size: 1st (Gemini-3.0-pro from page images, discarded train.csv), 4th (14.5k hand-checked pairs), 5th (~20k pairs through a 4-phase VLM pipeline with automatic re-prompting) all invested in **page-image extraction with tablet-ID linking**; most of 6 to 20 used OCR text + LLM alignment.
- *ByT5-XL* in 4 of the top 5 (2nd used Large) vs mostly Base/Large below 10th.
- *Teaching fundamentals before style:* staged training (3rd's Claude drills CPT, 5th's CPT → EvaCun → synthetic → PDF ablation +5.4 points) appears in the top 5 and is the largest measured effect in any writeup.
- The bottom of the top 20 (19th, 20th) are single ByT5-base models with Gemini pseudo-labels, reaching ~38.9: data alone gets within 4 points of 1st.

**Contradictory findings:**
- *Model size:* bigger is better (7th: small << base << large ≲ XL, 684215; 8th: XL, 684329; 13th: XL beam beat large beam, 686956) vs ByT5-base beat large (9th, 685928; 6th's best single model was base, 684231; 15th: Large/XL "did not work", 684819).
- *Weight averaging:* helped (4th 685925, 5th fold averaging 684847, 19th soup 684381, 20th +0.1 684267) vs worse than single models (6th, 684231) and soups "in general did not work" (15th, 684819).
- *Curriculum order:* noisy-then-clean is the biggest contributor (8th, 684329; 12th, 684866; 6th) vs gold fine-tuning after silver pre-training decreased scores (20th, 684267).
- *MBR utility:* competition metric as utility (11th, 684429) vs competition metric worked worse than chrF++ (15th, 684819); chrF++ MBR (4th, 13th); source-grounded MBR (9th). *MBR for a single model:* minimal gain (20th) vs single-model MBR shipped (13th).
- *Sampling inside MBR:* hurt (6th, "sampling within MBR leads to a decrease"; 9th: -2.7 on LB) vs beam + multi-temperature sampling pool shipped (13th, 1st's 10 candidates include sampling per bank).
- *Bidirectional training:* used in all 4 of 9th's models (685928) vs no gain (6th, 20th).
- *Neighbouring context:* tablet context helped (5th +0.1 private / +0.5 public; 6th dynamic context) vs inference-time context concatenation lost to single sentences (20th).
- *LLMs as the translator:* 14th's Qwen3-14B + transductive online learning and 15th's Qwen3 ensemble reached gold vs 6th (14B "average"), 8th, 9th-titled-17th and the 24th-place Qwen2.5-72B route (bank) all below ByT5.

## Aggregated "what did not work"
- RL on seq2seq: DPO (8th, 9th, 20th), minimum risk training (8th, 11th), PPO and GRPO for seq2seq (8th).
- Rerankers over beam candidates: LGBM reranker (CV +0.2, LB 0) and fine-tuned LLM listwise reranker - 5th.
- Sampling-based MBR candidates - 6th, 9th; sampling-based or tagged back-translation - 5th; decoding-time logit-averaging ensembles - 5th.
- Weight averaging / model soups - 6th, 15th (CPT+SFT soup).
- Document-level / long max-length training - 6th, 20th; inference-time context concatenation - 20th.
- Personal/geographic-name handling: capitalising names from OA_Lexicon_eBL (4th), PN/GN normalisation (11th), PN/GN placeholders (15th), name and commodity augmentation (8th), name swap (20th). (Contrast: 9th's entity-swap augmentation was kept in one of four models.)
- Pre-training on eBL_Dictionary - 11th; DAPT on raw Akkadian (repetition loops) - 9th.
- Bidirectional training - 6th, 20th.
- Decoder LLM fine-tunes as translator (Qwen, LLaMA, Mistral) - 6th, 7th (worse than ByT5), hallucinations on raw extracted data - 14th.
- Re-splitting to match Sentences_Oare and dropping the task prefix - 4th; extracting from Sentences_Oare_FirstWord_LinNum - 12th.
- LLM post-processing - 8th; LB-tuned post-processing - 4th (and 1st per bank).
- Manual or automatic sentence alignment of train.csv (low impact) - 7th, 8th.
- More than 1 epoch on the clean stage (collapse) - 8th; disabling early stopping - 20th.
- ByT5-large in their setup - 9th, 12th, 15th; MADLAD-400-3B single model (33.1) - 20th; ByT5-small - 20th.
- Focal loss and architecture changes - 15th.
- Re-extracting with a stronger VLM (Qwen3.5-VL 35B vs GLM-4.7V-Flash, worse LB) - 8th; Gemini Vision re-extraction of already-OCR'd PDFs (no gain) - 9th.
- CV on LLM-segmented or train-style validation sets (no LB correlation) - 8th, 9th, 11th.

## Shake-up note
Moderate (34% public). Largest moves inside the top 20: **10th Kelvin public 19 → 10**, 13th DataBoom 22 → 13, 14th H&B&Z 21 → 14 ("we are the lucky team and did not shake down"), 4th 8 → 4; fallers 15th It's showtime! 10 → 15, 8th Hrithik 5 → 8, and **claw 13 → 21** (out of the top 20). Just outside, epikt rose 51 → 22.
Explanations offered:
- 9th measured that **multi-checkpoint MBR of one model inflates public** (39.6 public → 38.4 private) while cross-model SG-MBR held (39.2 → 40.1) (685928).
- 5th: scaling back-translation 10k → 100k was -0.1 public, +0.3 private (684847). 6th: pre-training on all publications 38.6 public but 39.5 private (684231). 13th: ByT5-XL's private 39.59 beat its public-selected large (686956).
- 1st (bank) and 4th: LB-tuned post-processing is the main shake-up risk; both kept post-processing minimal.
- Common thread: CV rarely correlated with LB (8th, 9th, 11th), so teams that selected on public LB alone were exposed.

## UNVERIFIED
- Mapping of self-titled places to API ranks: vinpro "20th" = API 19 and Karl "21th" = API 20; the reason for the off-by-one (a removed team or a pre-finalisation board) is not established. kirito1sasa's "17th" post matches no API team in the top 60.
- 1st to 3rd cells marked ? (pseudo-labelling, CPT, context) are not covered by the bank summary; the full posts were not re-read for this file.
- 14th's RL method, reward and the number of online-learning rounds are not stated.
- 6th's "LB/PB" columns are read as public/private.
- 9th's per-model data table was truncated in extraction; the model-level details (LRs, encoder freeze, soups) are as quoted, the exact data sizes per model are not.
- No writeup located for ranks 16, 17, 18 (forum titles, comment authors; GitHub/Zenn not searched).

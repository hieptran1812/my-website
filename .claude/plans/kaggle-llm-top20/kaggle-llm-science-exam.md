# Kaggle - LLM Science Exam (2023): top-20 addendum

- **Slug:** `kaggle-llm-science-exam` (competition id 54662, forumId 3445358, 2,664 teams on the API board)
- **Existing deep coverage:** `.claude/plans/kaggle-llm-writeup-bank-part1.md`, H2 `## 1. Kaggle - LLM Science Exam (2023)`
  (lines ~45-247): ranks 1-3 in depth, 4/5/6 in short paragraphs, key public threads (Radek, Deotte 60k, simjeg 70B,
  MB 270k, CPMP 0.894). This file does not repeat that detail; it adds ranks 7-20 and the few facts the bank lacks for 4-6.
- **Rank-label caveat (important for the post):** a team that sat 10th was disqualified after the deadline (illegal
  team merge, see /discussion/446797). Several writeups were posted before the board was re-ranked, so their titles are
  one place too high: "10th(11th)" kami = API rank 10, "12th" chicm = rank 11, "13th" few-shot-learner = rank 12,
  "13th(14th)" kurupical = rank 13, "14th" Maxim Tsygankov = rank 14. Always cite the API rank.
- Source of all ranks/scores: `competitions.LeaderboardService/GetLeaderboard` (pulled 2026-10-28).

## Top-20 table (private LB, API)

| Priv | Team | Members (userName) | Private | Public rank | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | Team H2O LLM Studio | philippsinger, ybabakhin, ilu000 | 0.933208 | 1 | /discussion/446422 (+446240) | (bank) RAG over 30-60M chunks, 5 embedders, 7B/13B LLM binary classifiers + KV-cache trick |
| 2 | lytic | sorokin | 0.931853 | 6 | /discussion/448256 | (bank) Lucene BM25 + DeBERTa reranker + DeBERTa/Mistral MC + MLM, XGBRanker fusion |
| 3 | Podpall | podpall | 0.928415 | 10 | /discussion/446358 | (bank) own wiki parse, passage retrieval, tuned cross-encoder reranker, encoder ensemble + 70B on hard Qs |
| 4 | Preferred Scantron | mitmul, yoichi7yamakawa, hayaocelot, linshokaku | 0.927633 | 4 | /discussion/446307 | (bank) in-kernel Elasticsearch over cirrussearch, 3 context orderings, DeBERTa-v3-large up to 1280 tok |
| 5 | Preferred ChattyKids | flowlight, zaburo, corochann, qhapaq49, charmq | 0.926279 | 14 | /discussion/446293 | (bank) BM25 + 2 dense retrievers; Mistral-7B all, Llama-2-70B on least-confident 40%/5% |
| 6 | RAG to riches | abhishek, trushk, ubamba98, conjuring92, nbroad | 0.925914 | 7 | /discussion/447647 | (bank) crawled STEM corpus, fine-tuned retrievers + reranker, DeBERTa + Mistral LoRA, 70B on top-5% entropy |
| 7 | days | thedrcat, anonamename, kashiwaba, takashisomeya | 0.925184 | 5 | /discussion/447155 | 4 context sources (incl. Cohere paragraph TF-IDF) + 8 DeBERTa-v3-large + 3 LLMs (Mistral reward-model trick) in a confidence cascade |
| 8 | TT | yingfengyu, ianianh, leehann, bladerunner2022 | 0.924507 | 13 | no writeup found | - |
| 9 | SUSTech BDI | simjeg, camilahwang, xiezejian, zhangsongxin, janderchu | 0.923205 | 19 | no writeup found | (simjeg authored the public zero-shot 70B notebooks, /discussion/440620, 443334) |
| 10 | kami | kami634 | 0.922215 | 16 | /discussion/446248 | 4 wiki sources x 2 retrievers, single DeBERTa-large, max-probability ensemble, n-gram TF-IDF post-process |
| 11 | ScienceGPT | idv2005, berserker408, zzy990106, fankaixie | 0.921329 (tie) | 20 | /discussion/446660 | 4 contexts (incl. own 480k STEM set), DeBERTa MC + "one-shot" DeBERTa + Llama-2-7B LoRA |
| 12 | kaggle is few shot learner | pongtsu, yoyoismee, kunato, teetouchjaknamon, natnitarach | 0.921329 (tie) | 2 | /discussion/446301 | 56M-vector "Faiss batch" over 2 wiki versions + 270k, TF-IDF rerank, DeBERTa-large max ensemble |
| 13 | kurupical | kurupical | 0.921121 | 15 | /discussion/446231 | cirrussearch, sliding 90-word chunks, gte/e5 FAISS, DeBERTa-large 256-train/768-infer, retrieval-rank TTA, mean+max ensemble |
| 14 | Maxim Tsygankov | chubasik | 0.920495 | 47 | /discussion/446484 | reverse-engineered the category-based generation; 750k GPT-3.5 Qs from category pages; 3 DeBERTas x 4 corpora |
| 15 | Life is full of surprise | hengzheng, chizhu2018, sayoulala, syzong, tilbur | 0.920183 | 3 | /discussion/446816 | 5-way recall with SimCSE-trained hard-negative embedders + TF-IDF/BM25/LGBRanker, 2 DeBERTas |
| 16 | CSP | nahman, tmarechaux | 0.918672 | 100 | no writeup found | - |
| 17 | Sadge | nlztrk | 0.918620 | 9 | /discussion/446242 | own MediaWiki server to parse 1.1M STEM pages, FAISS + TF-IDF, single DeBERTa-large max ensemble |
| 18 | Chan Kha Vu | chankhavu | 0.918203 | 8 | /discussion/446261 | full-Wikipedia passage index (54.5M vectors, IVF256-PQ96 6 GB) + science sub-index, 6 DeBERTas |
| 19 | steubk | steubk | 0.917838 | 25 | /discussion/446395 | 42M-paragraph MiniLM FAISS + TF-IDF rerank; orca-mini-v3-70B yes/no + DeBERTa |
| 20 | SkyFish Up Up | wuwenmin | 0.916901 | 17 | /discussion/448114 | 4 RAG contexts + 1 DeBERTa-v3-large; max+mean ensemble |

Note the exact tie at 0.921329 between ranks 11 and 12 (API order kept). All discussion URLs are
`https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/<id>`; the API's `solutionWriteUpUrl` gives the
`/writeups/<slug>` mirror of the same text.

## Per-team synthesis

### Ranks 4-6: facts missing from the bank

- **4th Preferred Scantron** (/discussion/446307): validation = 2,000 instances randomly drawn from the open datasets
  without domain restriction (raw `train.csv` "too easy ... hardly useful"); they show a CV-vs-LB scatter with good
  correlation. Training sets: Deotte 60k + 40k. Tokens 512 -> 768 -> 1280 kept improving.
- **5th ChattyKids** (/discussion/446293): explicit discipline "we did not submit methods which do not work well in our
  local dataset" to avoid public-LB overfitting. Did not work: MB-style custom query construction; embedding-model
  fine-tuning (both negative on local val, so never submitted). Note they fell 14 public -> 5 private, consistent with that.
- **6th RAG to riches**: bank is complete; nothing to add.

### 7th - days (Takashi Someya, Darek Kleczek, anonamename, YumeNeko) - /discussion/447155

Retrieval: four context sources. (1) Wikipedia-20230801 dump (6.67M articles vs 6.27M in the common 2023-07-01
dataset): article retrieval with `gte-small` embeddings of the **full article** (chunked to 256 tokens), then TF-IDF
sentence retrieval; they patched JJ's `sectionize_documents` for better sentence splits. (2) MB 270k + TF-IDF.
(3) MB 270k + `gte-small` (max_seq_length 512). (4) **Cohere en-wiki paragraphs + TF-IDF**: hypothesis that sparse
paragraph search beats dense on these jargon-heavy questions; TF-IDF fitted on the first 10M paragraphs with the BERT
tokenizer vocab, indexes precomputed in 4 x 10M-row batches, top-5 via `sparse_dot_topn` (fast sparse KNN). Validation:
`train.csv` saturated at ~0.99, so they generated ~3,000 GPT-3.5 questions from the 130k STEM articles; CV-LB
"correlate relatively well" (scatter in post); final weights chosen on this CV. Models: 4 DeBERTa-v3-large (2 x
microsoft 512, 1 x `deepset/deberta-v3-large-squad2` 512, 1 x 1024 tokens) from Deotte's notebook. LLMs via `trl`
SFTTrainer + QLoRA, prompt asks for one letter; inference = softmax over the A-E logits of one generated token;
shuffling choices during training helped slightly. **Distinctive idea: "CausalLM reward modeling"**: modified trl's
RewardTrainer so a causal LM scores `<context> #### <prompt> #### <option> ####` by the next "yes" token logit, trained
on (correct, wrong) option pairs. This beat plain SFT: Mistral-7B-v0.1 SFT 0.874 private vs reward-model 0.897;
Mistral-7B-OpenOrca 0.882 vs 0.896; OpenOrca-Platypus2-13B SFT 0.872; Llama2-chat-AYT-13B 0.872. Inference: 4-stage
cascade; DeBERTa first, questions with max prob > 0.7 are "easy" and dropped from later stages; this freed time for 3
LLMs inside 9 h. Final = weighted average of 8 DeBERTa runs (weights 0.5-1.0) and 3 LLM runs (weights 1.25-1.5);
private 0.925 / public 0.931. A YumeNeko Zenn article exists (https://zenn.dev/yume_neko/articles/7347ba6b081e93, not read).

### 8th TT, 9th SUSTech BDI, 16th CSP - no writeup found

No solution topic in the 444-topic forum (all 23 pages scanned). SUSTech BDI includes simjeg, whose public
"[LB 0.836] Zero-shot 70B model + RAG" (/discussion/440620) and "[LB 0.872] ... improved RAG" (/discussion/443334)
notebooks are covered in the bank; teammate xiezejian commented on /discussion/446261 that their DeBERTa "was still
not well-trained enough ... only 0.91-0.92 on legacy retrieval", which hints DeBERTa was part of their final mix.
Nothing found for TT or CSP (CSP is the largest top-20 climber, public 100 -> private 16).

### 10th - kami (solo) - /discussion/446248 (code: kaggle.com/code/kami634/11th-solution)

Four wiki sources: wikiextractor on the database dump (all articles but drops bullet lists/sections and some formulas
show as "formula_1"), cirrussearch dump (no sections), MB 270k Cohere, MB 270k parsed. On the dumps: embed 200-250
character pieces, **IVFPQ-quantised** FAISS to fit memory, retrieve 10 articles, then sliding-window passages over them
searched with IndexFlat (sliding window because a sentence split could cut the answer sentence). On 270k: TF-IDF and
flat embedding search; bigger query encoders (`bge-large-en-v1.5`) "did not make much difference". Model: **one**
DeBERTa-large for everything, inferred with a context twice as long as in training. Training data: Deotte 60k.
Validation: hold-out of 200 train + 1,000 own GPT-3.5 questions; "no high correlation, but informative", the GPT set
had wrong/multi-answer items. **Distinctive idea: max-probability ensemble** across contexts: if any context holds the
answer the model is confident, otherwise flat; "robust, less likely to overfit" and better than blending. Plus a
rule-based post-process with (4,7)-gram TF-IDF to fix "obviously wrong" predictions under specific conditions.

### 11th - ScienceGPT (chicm, LIGHT, Leon, Tifo) - /discussion/446660

Four contexts: (1) own English-wiki FAISS index, top-5 from each of three recall paths (`bge` prompt+answer, `gte`
prompt+answer, `gte` prompt only); (2) JJ's sentence-level open-book context; (3) MB 270k (Cohere + parsed); (4) **own
480k STEM set** built with MB's clustering notebook but re-tuned so the clusters cover all 154 source articles they had
identified (270k did not). Models: DeBERTa-v3-large multiple choice (Deotte); a **"one-shot" DeBERTa-v3-large** that
concatenates all five options and runs once per question (architecture borrowed from Feedback Prize Effectiveness,
/competitions/feedback-prize-effectiveness/discussion/347433); Llama-2-7B LoRA (5 forward passes per question, 3 h at
512 tokens, worth +0.003 in the blend). 8 DeBERTa checkpoints across contexts in the final. Finding: training with
*better* context did not improve the model, so they kept checkpoints trained on sentence-level context.

### 12th - kaggle is few shot learner (5 members) - /discussion/446301 (GitHub nat-nischw/kaggel-llm-science-exam-2023)

Public #2 -> private #12. Retrieval "has the big impact": `bge-small-en` FlatL2 over Cohere wiki (35M vectors) +
Wikipedia 2023-07-01 chunked at 1,000 characters (21M vectors) = 56M vectors, 100 GB+. **"Faiss batch"**: ~6 x 10 GB
shards per wiki, each loaded to GPU, searched, freed, merged by offset (code in post). Top ~100 articles (6 shards x 15)
then **TF-IDF rerank, which beat `bge-reranker`**. Plus 270k + TF-IDF. Three contexts, each model predicts three times.
Model: DeBERTa-v3-large, Deotte pipeline with embeddings unfrozen, 512-768 tokens, trained on 60k + ~5k own GPT-3.5
questions; ensemble by max probability. Did not work: compressing the index (IVF, PQ) cost accuracy;
Platypus2-70B-instruct with their retrieval exceeded 9 h; PEFT on DeBERTa.

### 13th - kurupical (solo) - /discussion/446231

Switched from JJ's 2023-07-01 dump to **cirrussearch** (missing numbers). Built a 250-question ChatGPT retrieval eval
restricted to articles starting with "a" to iterate fast: recall@1 0.94, recall@30 1.0 (about 0.85@1 on whole wiki).
Chunking: read sentences until > 90 words, save, keep last 3 sentences as overlap; grid over (60,2) ... (150,6), chosen
by LB. FAISS `{"nlists":1, "M":64, "nbits":8}`; embedders `gte-base` + `e5-base` (swept gte/bge/e5 x small/base/large).
Lesson: his retrieval-accuracy metric **did not correlate with LB**, so final choices were made on LB. Model: Deotte
baseline; train at max_length 256, infer at 768 (256 beat 384/512 when all inferred at 768); lr 1e-6, cosine,
warmup 0.01, batch 2 x grad-accum 8, fp16, weight decay 0.01. 4-model ensemble over backbones
(`OpenAssistant/reward-model-deberta-v3-large-v2`, `deepset/deberta-v3-large-squad2`, `microsoft/deberta-v3-large`),
retrievers and 60k subsets (dropping sources 3,7,8,9 hurt; dropping 10 helped). **Distinctive: retrieval-rank TTA**:
four views, each = top-1 chunk + a different rank band (1-5, 6-10, 11-15, 16-20); combine by `mean + max` per question.
Did not work: own ChatGPT question generation (score dropped); a CV correlated with LB (tried 300/dataset and
wuwenmin's 300 + train 200); complex TTA; model-side improvements ("not the key point").

### 14th - Maxim Tsygankov (solo) - /discussion/446484

Public 47 -> private 14. **Reverse-engineered the generator**: manually traced each `train.csv` question to its source
page and noticed many came from the same Wikipedia *categories*, including non-science pages inside a science category
("Droste effect", "Triskelion" in Category:Symmetry), so the test set was likely generated automatically from science
categories. Built an "Adjacent by Category" (AC) corpus: every page in those categories, split into ~1,500-char
segments; GPT-3.5 wrote 5 MCQs per segment: **~750k segment-question-answer triplets**, his only training data
(dataset `chubasik/kaggle-llm-science-exam-gpt-ac-dataset`). Inference-only corpus variants: Grandparent categories
(GP), Natural-sciences BFS to depth 5 (covered 193/200 train pages, plus 4 extra category roots), GP+1 with hundreds of
irrelevant categories removed by hand, AC-750 (750-char segments). Hybrid search: e5 and bge **fine-tuned on AC** plus
TF-IDF. Final: 3 DeBERTa-ForMultipleChoice variants (output layers, input order, option awareness) x 4 contexts.

### 15th - Life is full of surprise (5 members) - /discussion/446816

Public #3 -> private #15 ("looks like overfit"); they put "all attention on RAG" and used only 2 DeBERTas, and suspect the
overfit sat in DeBERTa rather than in an LLM reader. Embedders all trained with **SimCSE** (+0.015 LB), then a second
round with **hard negatives** mined by the first model (top-4 non-gold retrievals as negatives, +0.005 more). TF-IDF
sped up to 20 min. Five recall paths: (1) 6.8M-article wiki: sentence model top-1000 articles -> BM25 or LightGBM ranker
-> top-30 -> sentence model top-20 sentences (alone 0.885 LB); (2-5) MB 270k at 2.1M and 2.8M paragraph splits, each with
sentence-model top-5 and TF-IDF top-8. Sharded FAISS indexes merged by score to save memory. They call it "a Discussion
Competition": after MB's 270k set, every trick was public.

### 17th - Sadge / Anil Ozturk (solo) - /discussion/446242

**Rebuilt the STEM subset properly**: downloaded category SQL dumps, reconstructed the category-links graph, BFS 4 levels
from STEM roots -> ~1.1M STEM articles. Then ran a **local MediaWiki server** and pulled those pages through the
TextExtract API so LaTeX and Lua templates render exactly (F2). Other sources: full JSON dump (F1, broken numbers) and
Cohere wiki (F3). Unicode-normalised; chunks = 3-sentence windows, stride 1. Retrieval: `bge-small-en` FAISS over F1,
~150 GB **uncompressed** (compression hurt); TF-IDF (1-2 grams, no stop words) on F2 and F3; FAISS-then-TF-IDF rerank
(F4). One DeBERTa-v3-large on all four contexts, **max ensemble**.

### 18th - Chan Kha Vu (solo) - /discussion/446261 (code: kaggle.com/code/chankhavu/fork-of-0-92-submission)

Public 8 -> private 18. **Passage-level retrieval over all of Wikipedia** (answers often live in a less-related
article): 128-token passages, stride 96 -> 54.4M 768-d vectors, 154 GB. Index study with GPT-3.5 questions whose source
passage is known: best = **IVF256 + PQ 96 bytes** (max for GPU), < 6 GB, ~20 s for 4,000 questions at K=100,
nprobe=96. This alone + one public Deotte model = 0.891 LB, before MB's 270k existed. Science sub-index: 270k articles
(stride 64, IVF256 FP16) -> 0.895; 12 h before the end added 300k more science articles (GPT-4 topic list + `bge-base`
clustering) -> 600k articles / 7.6M passages. Rerank 3-sentence chunks with `bge-large-en-v1.5` plus BM25 over the
retrieved pool. 6 DeBERTa models trained in the last 3 days on 1x A100 + 1x 3090 (likely under-trained), on own GPT-3.5
data prompted to include numbers/formulas, plus 60k. **"Training on worse context gives better models"**: train contexts
built with the weaker `bge-small` pipeline, ~+0.005 LB. Smaller lr + more QA pairs helped. Did not work: LightGBM/XGBoost
LambdaRank reranker over embedding/BM25/TF-IDF scores (worse than raw embedding score). His Azure account was banned
11 h before the deadline, so his last models missed the final subs.

### 19th - steubk (solo) - /discussion/446395 (code: kaggle.com/code/steubk/llm-mc-qa-openbook-v3-inf-val-tfidf-db450)

Retrieval: HF `wikipedia_20220301.en`, boilerplate sections removed, split on blank lines, paragraphs < 10 words dropped;
`all-MiniLM-L6-v2` (max 512) into **10 FAISS indexes of 4.2M rows**, searched sequentially on 2x T4 (< 10 min / 200
prompts); then TF-IDF rerank of the candidates ("very important"). Readers: zero-shot **orca-mini-v3-70b** via simjeg's
layer-wise code, prompt asks yes/no for each proposed answer; DeBERTa-v3 (max_length 450) MC trained on 24k prompts from
mgoksu's dataset filtered to What/Which/How/Who and deduped, validated on the 200. Final = merged softmax of both; each
step in a separate script to dodge memory leaks; runtime 7 h 20 min up to timeouts in the final days. Ablation
(public / private): DeBERTa no context 0.822 / 0.805; + step-1 context 0.889 / 0.884; 70B + step-1 0.897 / 0.889;
merge 0.906 / 0.901; 70B + TF-IDF rerank 0.904 / 0.906; **merge + rerank 0.921 / 0.917**.

### 20th - SkyFish Up Up / william.wu (solo) - /discussion/448114 (code: kaggle.com/code/wuwenmin/llm-science-ensemble-exp-max)

Short: 4 RAG contexts + 1 DeBERTa-v3-large; switching the mean ensemble to **max + mean** would reach 0.917.
DeBERTa-v3-large trained on 8x RTX 4090 with DeepSpeed (itsuki9180's notebook). Did not work: pre-fine-tuning on RACE then
fine-tuning on 60k at lower lr. He also authored the 300-question validation set others used (kurupical) and the
FAISS+TF-IDF fusion comment credited by 15th.

### Outside top 20 but useful

- **21st Contextual kagglers** (alexryzhkov, titericz, ...) /discussion/447589 - not read in detail.
- **Disqualified former 10th** (lizhecheng) /discussion/446797: ~303k training rows (150k own GPT-3.5 + 99k/60k/17k
  MMLU/70k public sets), `gte-small` beat MiniLM-L6/L12 and bge-small (0.843 -> 0.851 public), extended MB's
  `target_articles` with manually identified titles; DeBERTa-v3 only. Cautionary tale: merged teams over the
  submission cap by adding a new account.

## Technique frequency matrix

Y = stated in writeup (or bank), N = stated not used / tried and dropped, ? = unknown / no writeup.

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RAG over Wikipedia | Y | Y | Y | Y | Y | Y | Y | ? | ? | Y | Y | Y | Y | Y | Y | ? | Y | Y | Y | Y |
| Re-parsed wiki to recover numbers (cirrus / MediaWiki / patched extractor) | Y | N | Y | Y | Y | ? | N | ? | ? | Y | ? | N | Y | N | ? | ? | Y | ? | N | ? |
| MB 270k STEM set used | ? | N | ? | ? | Y | ? | Y | ? | ? | Y | Y | Y | ? | N | Y | ? | ? | Y | N | ? |
| Sparse retrieval (BM25 / TF-IDF / ES) | N | Y | ? | Y | Y | ? | Y | ? | ? | Y | ? | Y | ? | Y | Y | ? | Y | Y | Y | ? |
| Dense retrieval | Y | N | Y | Y | Y | Y | Y | ? | ? | Y | Y | Y | Y | Y | Y | ? | Y | Y | Y | ? |
| Fine-tuned retriever / reranker | N | Y | Y | N | N | Y | N | ? | ? | N | N | N | N | Y | Y | ? | N | N | N | ? |
| Multiple context variants fused late | Y | Y | ? | Y | Y | Y | Y | ? | ? | Y | Y | Y | Y | Y | Y | ? | Y | Y | Y | Y |
| DeBERTa-v3-large reader | N | Y | Y | Y | N | Y | Y | ? | Y | Y | Y | Y | Y | Y | Y | ? | Y | Y | Y | Y |
| Fine-tuned 7B-13B LLM reader | Y | Y | N | N | Y | Y | Y | ? | ? | N | Y | N | N | N | N | ? | N | N | N | N |
| 70B zero-shot / QLoRA on Kaggle GPU | N | N | Y | N | Y | Y | N | ? | Y? | N | N | N | N | N | N | ? | N | N | Y | N |
| LoRA / QLoRA | Y | ? | N | N | Y | Y | Y | ? | ? | N | Y | N | N | N | N | ? | N | N | N | N |
| Own GPT-generated training data | N | N | N | N | N | Y | N | ? | ? | N | N | Y | N | Y | ? | ? | N | Y | N | ? |
| Own harder validation set | Y | Y | ? | Y | Y | ? | Y | ? | ? | Y | ? | ? | Y | ? | ? | ? | N | Y | N | ? |
| Confidence cascade (big model on hard Qs) | N | Y | Y | N | Y | Y | Y | ? | ? | N | N | N | N | N | N | ? | N | N | N | N |
| Max-probability ensemble across contexts | ? | N | ? | ? | ? | ? | N | ? | ? | Y | ? | Y | Y | ? | ? | ? | Y | ? | N | Y |
| Option-order / retrieval TTA | N(shaky) | ? | ? | ? | Y | ? | ? | ? | ? | N | ? | ? | Y | ? | ? | ? | N | ? | ? | ? |

Counted summary (17 teams with writeups: 1-7, 10-15, 17-20):
- 17/17 used RAG over Wikipedia; 17/17 fused several context variants.
- 15/17 used a DeBERTa-v3-large reader (exceptions: 1st, 5th, which used only LLM readers).
- 6/17 fine-tuned a 7B-13B LLM reader (1, 2, 5, 6, 7, 11); all of them are in the top 11.
- 4/17 ran a 70B model on Kaggle GPUs (3, 5, 6, 19) plus SUSTech BDI likely via simjeg (unverified).
- 5/17 used a confidence cascade (2, 3, 5, 6, 7): every one is in the top 7.
- 11/17 used sparse retrieval somewhere; 6/17 reported re-parsing Wikipedia to fix numbers.
- 5/17 used a max-probability ensemble across contexts (10, 12, 13, 17, 20), all solo-or-DeBERTa-only teams ranked 10-20.
- 4/17 generated their own GPT training data (6, 12, 14, 18); the 1st-place team said quantity mattered little.

## Consensus vs divergence

**Table stakes (ranks 1-20):** retrieval is the model; several Wikipedia sources and/or retrievers fused late; DeBERTa
or a 7B LLM read 3-5 chunks; validation beyond the saturated 200-row train.csv.

**What separated the top 7 from 10-20:** (1) an **LLM reader** (fine-tuned 7B/13B or 70B) in addition to DeBERTa: present
in 6 of the top 7 (4th is the DeBERTa-only exception, compensated by 1280-token contexts and Elasticsearch), absent from
every DeBERTa-only team at 10-20 except 11th's small Llama-2-7B and 19th's zero-shot 70B; (2) a **confidence cascade**
to fit big models in 9 h (top-7 only); (3) a **trained reranker or retriever** (2, 3, 6). Teams 10-20 mostly polished
retrieval and ensembled a single DeBERTa.

**Contradictions:**
- *Better training context.* 1st: training on the true source context was worse than on own noisy RAG (bank).
  18th: "training on worse context gives better models" (+0.005) /discussion/446261. 11th: better train context did not
  help /discussion/446660. Consistent: noisy train context is better.
- *Index compression.* 18th: IVF256-PQ96 index fine, 0.891 LB alone /discussion/446261; 10th uses IVFPQ
  /discussion/446248. Versus 12th: "compress the index (IVQ, PQ) resulted in low accuracy" /discussion/446301 and 17th:
  compression decreased retrieval, kept 150 GB flat /discussion/446242.
- *Generating own questions.* 14th trained only on 750k self-generated Qs /discussion/446484 and 6th/18th used GPT data;
  13th: "Create question by ChatGPT ... score is decreased" /discussion/446231.
- *Retrieval metric as proxy.* 18th validated index choices by passage recall on GPT questions /discussion/446261;
  13th found his recall metric uncorrelated with LB /discussion/446231.
- *Fine-tuning embedders.* 6th, 14th, 15th (SimCSE + hard negatives, +0.02 LB) fine-tuned retrievers; 5th tried
  embedding fine-tuning and dropped it on local val /discussion/446293.
- *Rerankers.* 3rd's tuned cross-encoder worth +0.017 (bank); 12th found TF-IDF rerank beat `bge-reranker`
  /discussion/446301; 18th's LambdaRank GBDT reranker lost to raw embedding score /discussion/446261.

## Aggregated "what did not work"

- Compressing FAISS indexes (IVF/PQ) - 12th, 17th (contested by 10th, 18th).
- Platypus2-70B with their retrieval inside 9 h (timeouts) - 12th.
- PEFT/LoRA on DeBERTa - 12th.
- Own ChatGPT-generated questions - 13th.
- Building a CV that correlates with LB from public sets / 300-row sets - 13th; 10th's 1k GPT set "no high correlation".
- Retrieval-recall proxy metric uncorrelated with LB - 13th.
- Complex TTA - 13th; option-order TTA "shaky" - 1st (bank).
- Larger query encoders (bge-large) - 10th ("not much difference").
- LightGBM/XGBoost LambdaRank reranker - 18th.
- Pre-fine-tuning on RACE - 20th.
- Custom MB-style query construction; embedding fine-tuning - 5th.
- Training on true source context instead of retrieved context - 1st (bank), 11th, 18th.
- Focal loss; removing Llama causal mask - 2nd (bank).

## Shake-up

Computed from the API (public rank -> private rank), top-20 only:
- Fallers into the top 20: kaggle is few shot learner 2 -> 12, Life is full of surprise 3 -> 15, Chan Kha Vu 8 -> 18,
  Sadge 9 -> 17, Contextual kagglers 11 -> 21. 15th explicitly blames DeBERTa overfitting to the public 20%; 18th blames
  missing his last models.
- Risers: CSP 100 -> 16 (no writeup), Maxim Tsygankov 47 -> 14 (credits a category-reverse-engineered corpus "robust
  enough to survive"), steubk 25 -> 19, SUSTech BDI 19 -> 9, ScienceGPT 20 -> 11, ChattyKids 14 -> 5 (local-val
  discipline), Podpall 10 -> 3.
- The public split was 20% (~800 questions), so a 0.003 MAP gap is a handful of questions; the top 20 span only
  0.933 -> 0.917.
- The board was re-ranked after the former 10th team was removed (/discussion/446797), which is why several writeup
  titles are off by one.

## UNVERIFIED

- Whether SUSTech BDI (9th) used simjeg's 70B pipeline in their final submission: inferred from membership only.
- TT (8th) and CSP (16th) approaches: unknown; no Kaggle post found; one web search returned nothing.
- 20th's four RAG sources are in an image only; not transcribed.
- 7th's exact LoRA hyperparameters (rank, alpha, lr, epochs) are not in the post.
- Matrix cells for ranks 1-6 are taken from the bank's summaries; "?" means the bank does not say, not that they did not.
- 21st (Contextual kagglers) writeup was fetched by URL list only, not read.

# Learn from Kaggle: LLM Competitions -- Writeup Bank, Part 1 (2023-2025)

Built 2026-10-27. Scope: the six Kaggle LLM-era NLP competitions of 2023-2025 that a "Learn from Kaggle: LLM competitions"
series draws on first: LLM Science Exam, Detect AI Generated Text, Prompt Recovery, LMSYS Chatbot Arena, WSDM Cup
Multilingual Chatbot Arena, plus a short LLM-adjacent section on PII Data Detection.

## How this bank was verified (read first)

- **Primary source = Kaggle's own JSON endpoints** (kaggle.com is JS-rendered), called through
  `.claude/scripts/kaggle-api.sh` (anonymous session + XSRF cookie):
  - `competitions.CompetitionService/GetCompetition` -> id, forumId, dates, team counts, prize pool, GPU/CPU runtime cap,
    public-LB percentage, total submissions.
  - `competitions.PageService/ListPages` -> Description / Evaluation / Timeline / Data / Prizes / Code Requirements /
    Rules page text. Every metric formula, prize split and notebook limit below is quoted from these pages.
  - `competitions.LeaderboardService/GetLeaderboard` -> final **private** LB (rank, score, team name, members, linked
    writeup URL) and the public LB (used for shake-up numbers).
  - `discussions.DiscussionsService/GetTopicListByForumId` (`sortBy:"TOP"`; `"VOTES"` returns HTTP 400) -> top-voted
    threads, vote counts as of 2026-10-27.
  - `discussions.DiscussionsService/GetForumTopicById` -> full markdown of each winner's post; technical detail below is
    paraphrased closely from those posts.
- Raw JSON lives in `/private/tmp/claude-501/kaggle-bank-p1/` (scratch, not committed).
- Writeup URLs: the leaderboard API links the new `/competitions/<slug>/writeups/<name>` form. The original forum thread
  `/competitions/<slug>/discussion/<id>` is also given where known (both resolve on kaggle.com).
- Scores are as the API returns them (display precision varies by competition). Vote counts drift; treat them as rough.
- The API `deadline` field sometimes falls after the final-submission day (it marks the LB finalisation / forecasting end);
  each section gives both.
- **UNVERIFIED** marks anything not confirmed from a primary source. No rank, score, name or number here was invented.

## Summary table

| Slug | Title | Open -> final submission (API deadline) | Teams | Prize pool | Metric |
|---|---|---|---|---|---|
| `kaggle-llm-science-exam` | Kaggle - LLM Science Exam | 2023-07-11 -> 2023-10-10 | 2,664 | $50,000 | MAP@3 over 5 options |
| `llm-detect-ai-generated-text` | LLM - Detect AI Generated Text | 2023-10-31 -> 2024-01-22 | 4,358 | $110,000 (LB $60k + efficiency $50k) | ROC AUC |
| `llm-prompt-recovery` | LLM Prompt Recovery | 2024-02-27 -> 2024-04-16 | 2,175 | $200,000 | Sharpened cosine similarity (exp 3) of sentence-t5-base embeddings |
| `lmsys-chatbot-arena` | LMSYS - Chatbot Arena Human Preference Predictions | 2024-05-02 -> 2024-08-05 (LB final 2024-08-12) | 1,849 | $100,000 | Multiclass log loss (A / B / tie) |
| `wsdm-cup-multilingual-chatbot-arena` | WSDM Cup - Multilingual Chatbot Arena | 2024-11-18 -> 2025-02-03 (forecast end 2025-03-10) | 950 | $50,000 | Accuracy (A vs B) |
| `pii-detection-removal-from-educational-data` | The Learning Agency Lab - PII Data Detection | 2024-01-17 -> 2024-04-23 | 2,048 | $60,000 (LB $28k + efficiency $32k) | Micro F-beta, beta = 5 |

All six are **code competitions**: submit a notebook, internet off, freely available external data and pre-trained
models allowed, `submission.csv`. Max team size 5, 5 submissions/day.

---

## 1. Kaggle - LLM Science Exam (2023)

### Facts

- **URL:** https://www.kaggle.com/competitions/kaggle-llm-science-exam (competition id 54662, forumId 3445358)
- **Dates:** 2023-07-11 -> final submission 2023-10-10 (entry + team-merger deadline 2023-10-03).
- **Task:** answer 5-option multiple-choice science questions that **GPT-3.5 wrote** from Wikipedia snippets on
  scientific topics, after which easy questions were filtered out. The host framed it as "can a model ~10x
  smaller than the question-writer ace its test?" (Description page: "the largest models run on Kaggle are around 10
  billion parameters, whereas gpt3.5 clocks in at 175 billion").
- **Data:** only **200 labelled sample questions** (`train.csv`); hidden test of **~4,000** questions, "may be different in
  subject matter". Columns `prompt, A..E, answer`. No generation details disclosed. Everyone built their own training sets.
- **Metric:** MAP@3, predict up to 3 labels per question:
  `MAP@3 = (1/U) * sum_u sum_{k=1..min(n,3)} P(k) * rel(k)`; once the correct label is hit, later repeats are ignored.
  So correct at rank 1 = 1.0, rank 2 = 0.5, rank 3 = 0.333.
- **Public / private split:** public LB = 20% of test (API `leaderboardPercentage`), private = the rest.
- **Prizes:** $50,000 -- 1st $12k, 2nd $10k, 3rd $10k, 4th $10k, 5th $8k.
- **Teams:** 2,664 teams, 3,299 competitors, 57,886 submissions.
- **Limits:** CPU or GPU notebook <= 9 h, internet off, public external data and pre-trained models allowed. Kaggle GPUs
  were 2x T4 (16 GB each) or 1x P100 -- simjeg's post: "these models can't fit a T4 or a P100 GPU".
- **Rule clarifications:** licence of models was debated: 3rd and 6th place used `Platypus2-70B-instruct` (non-commercial
  licence) and said so in their posts; CPMP switched to Xwin-LM-70B "for safety" (446232). Podpall's update: "Turns out it
  was fine to use Platypus2." Thread: "Acceptable Model/Data Use Clarification" (discussion/425681). No leak.

### Top placements (private LB, from the API)

| Rank | Team | Private | Public rank | Approach (one line) | Writeup |
|---|---|---|---|---|---|
| 1 | Team H2O LLM Studio (Psi, Yauhen Babakhin, Pascal Pfeiffer) | 0.933208 | 1 | RAG over 30-60M Wikipedia chunks (5 embedders) + 7B/13B LLMs fine-tuned as binary classifiers with a KV-cache trick | https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446422 (short: /discussion/446240) |
| 2 | lytic (ivan, solo) | 0.931853 | 6 | Lucene BM25 over full Wikipedia + DeBERTa reranker + DeBERTa MC + Mistral MC + masked-LM, fused by XGBRanker | https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/448256 |
| 3 | Podpall (Pavel Podberezko, solo) | 0.928415 | 10 | Own Wikipedia parse + small-embedder retrieval + tuned cross-encoder reranker + DeBERTa/Electra/RoBERTa ensemble + 70B LLMs on hard questions | https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446358 |
| 4 | Preferred Scantron (mitmul, Yiemon773, hayaocelot, deoxy) | 0.927633 | 4 | Elasticsearch over cirrussearch Wikipedia, 3 context-ranking variants, DeBERTa-v3-large with up to 1280 tokens | https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446307 |
| 5 | Preferred ChattyKids (flowlight, KaizaburoChubachi, corochann, qhapaq49, charmq) | 0.926279 | 14 | BM25 + 2 dense retrievers; Mistral-7B on all, Llama-2-70B on the least-confident 40% / 5%, run layer-by-layer | https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446293 |
| 6 | RAG to riches (Abhishek Thakur, Trushant Kalyanpur, Udbhav Bamba, Raja Biswas, Nicholas Broad) | 0.925914 | 7 | Custom STEM wiki corpus, fine-tuned retrievers + cross-encoder reranker, DeBERTa + Mistral-7B LoRA, 70B on top-5% hardest | https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/447647 |
| 7 | days (Darek Kleczek, anonamename, YumeNeko, Takashi Someya) | 0.925184 | 5 | not read | /competitions/kaggle-llm-science-exam/writeups/days-7th-place-solution |
| 12 | kaggle is few shot learner | 0.921329 | **2** | public #2 fell to private #12 | /competitions/kaggle-llm-science-exam/writeups/kaggle-is-few-shot-learner-13th-place-solution-ens |
| 15 | Life is full of surprise | 0.920183 | **3** | public #3 -> private #15 | /competitions/kaggle-llm-science-exam/writeups/life-is-full-of-surprise-public-top3-private-top15 |
| 54 | CPMP (solo) | 0.908408 | -- | **No model training at all**: zero-shot Xwin-LM-70B + paragraph RAG, plus a logistic-regression head | https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446232 |

Gold = top 15. Other writeups linked by the LB API: 10th kami, 13th kurupical, 18th Chan Kha Vu ("8th public 18th private,
full Wikipedia"), 64th "One RAG to rule them all" (Elasticsearch).

### 1st place -- Team H2O LLM Studio (Psi / philippsinger, Yauhen Babakhin, Pascal Pfeiffer)

Sources: short summary https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446240 (Psi), full writeup
https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446422 (Pascal Pfeiffer).

- **Framing.** Plain classification (DeBERTa or LLM, no context) hit a ceiling early. Their first retrieval submission
  with `all-MiniLM-L6-v2` beat everything else, so from then on they mostly improved **retrieval** and kept the model part
  roughly fixed. They had a 0.930 private-LB submission ~30 days before the end. Psi: the score has a theoretical
  ceiling because GPT-3.5 mislabels a share of its own answers; grinding 0.93x -> 0.940 public took weeks.
- **Retrieval.** Tried most of the MTEB top-20 embedders; ~300 local combinations of retriever x LLM x Wikipedia dump.
  Final contexts from `e5-base-v2`, `e5-large-v2`, `gte-base`, `gte-large`, `bge-large`. Wikipedia chunks encoded as
  `title + chunk`. Question encoded two ways: (1) `"{prompt} {A}, {B}, {C}, {D}, {E}"`; (2) one search per
  `"{prompt} {option}"`, merged by similarity.
- **Search engine = matrix multiply.** No FAISS. 30M-60M chunks split into parts, each part loaded in turn, similarity
  matrix vs. all test questions via torch matmul on GPU, keep a running global top-k; split across 2 GPUs for 2x speed.
- **Wikipedia dumps.** Started from the public 2022-12 wiki embeddings dataset; rebuilt a newer snapshot, merged tiny
  paragraphs (subtitles) into the next one. Found that public parsers **drop Lua-rendered values** (numbers in science
  articles), so used the **cirrussearch** dump (fully rendered pages, but no newlines) and re-chunked it into 256 / 512 /
  1024-character pieces without splitting sentences. Best single corpus: cirrussearch at 512 chars. Filtering to science
  articles never helped significantly; they kept full Wikipedia.
- **Context length.** Train with 3 chunks, infer with 5 chunks (1k max_length). Reversing context order at inference
  hurt a lot, suggesting the model learned that the best chunk comes first.
- **Validation.** The 200 given samples reached 0.99 once RAG was in, so useless; they validated on **6k STEM questions**
  from a public dataset. Best private sub = best CV sub.
- **Training data.** Public synthetic MCQ sets (Radek's early data "does the job quite well"); quantity mattered little.
  Contexts for training generated with their own RAG (noisy on purpose); training on the true source context was worse.
- **Model.** Binary formulation: one sample per (question, option), score = P(correct), rank by probability. Makes it
  order-independent. LoRA on **all linear layers**, binary classification head on the **final next-token logits over the
  full vocabulary**, lower lr for the head than for LoRA, 1 epoch, cosine decay, BCE, last checkpoint. LoRA rank / lr
  values not given. Backbones that worked: `Llama-2-7b`, `Mistral-7B-v0.1`, `xgen-7b-8k-base`, `Llama-2-13b`. Bigger
  models were "not useful"; DeBERTa did not even help in the ensemble.
- **KV-cache trick.** Forward (context + question) once, keep `past_key_values`; then run the five answers as a batch of 5
  reusing that cache; take each answer's last-token logits -> head -> probability.
- **Cross-option head.** To give the binary model some cross-option signal without positional bias, the head also takes
  the **mean next-token logits of the other four options**. Boosted CV and LB and blended well with the plain head.
  Multiclass models and TTA over option orderings were tried: TTA helped but was "shaky", never beat binary.
- **Inference.** Five 7B models + one 13B; each model gets a different (wiki, embedder, top-k) context = late fusion of
  retrieval variants. Fits "precisely" in the 9 h limit and uses **2.5 TB** of input data. Code merged into H2O LLM
  Studio (PR h2oai/h2o-llmstudio#449, causal classification). Notebooks: `ybabakhin/1st-place-team-h2o-llm-studio`
  (0.933 private), `ybabakhin/1st-place-single-model-inference` (single model 0.932 private).

### 2nd place -- lytic (ivan, solo)

Source: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/448256 -- "mostly based on Deberta type
models and only half of the test questions are handled by Mistral LLM."

1. **Knowledge base:** HF `graelo/wikipedia` 20230601.en, sentence-split, grouped into overlapping ~1,000-char chunks,
   indexed with `pyserini.index.lucene` (BM25). Index 23 GB; 200x5 queries in ~5 min.
2. **Reranker:** `deberta-v3-base` built like `DebertaV2ForMultipleChoice` but choosing the best **context** for a
   (question, answer) pair. Labels from a `deberta-v3-large` teacher: retrieve a few contexts per correct QA pair, the
   one the teacher scores best is the positive.
3. **MC readers:** standard `DebertaV2ForMultipleChoice` for DeBERTa and Mistral; loss scaled by
   `num_labels / num_choices_in_sample` to mix 4- and 5-option datasets.
4. **Masked LM** for questions whose options differ by only a few words (MC models did badly there): pairwise alignment
   between options with `sed.standard_sed_backtrace`.
5. **"None of the above" stress test:** replacing correct answers with "None of the above" in validation dropped MAP@3
   from 0.9 to 0.3, so he added a second-stage ranker trained with more such examples.
6. **XGBRanker fusion** on validation sets: retrieval scores, DeBERTa/Mistral MC logits, a "None of the above" flag.
- **Training data:** Radek's 39,249; stem_1k_v1 928; eduqg 3,297; mozattt 3,084; cdeotte MMLU 17,433; OpenBookQA 5,957;
  ARC 7,787; QASC 9,060; SciQ 13,679.
- **Ablation (private / public):** k=1, 2x DebertaMC 0.9098 / 0.9078 -> k=8 + Reranker 0.9161 / 0.9078 -> + XGBRanker
  0.9160 / 0.9099 -> + MLM 0.9155 / 0.9116 -> **+ MistralMC 0.9315 / 0.9315**. Biggest single jump = adding Mistral.
- **Did not work:** focal loss for DeBERTa MC; removing the causal mask in Llama attention.

### 3rd place -- Podpall (Pavel Podberezko, solo)

Source: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446358 (code: `podpall/3rd-place-reranker-training`,
`podpall/3rd-place-full-inference`).

- **Wikipedia:** modified `attardi/wikiextractor` to stop it deleting numeric values.
- **Retrieval:** one-stage **passage-level** search (not article-then-sentence), because article titles/openings can miss
  the entity a question is about. Retrievers: `bge-small` and `all-MiniLM-L6-v2` (from his ablation table).
- **Reranker is the key component.** Two ingredients: (1) (question, candidate) pairs, positive and negative, drawn from
  the **same distribution as inference** (his own Wikipedia passages, not the 70k dataset's contexts); (2) **hard
  negatives**, which only work starting from a reranking-pretrained checkpoint (`ibm/re2g-reranker-nq`) or in 2 stages.
  Tuned with sugupoko's 70k dataset.
- **Readers:** DeBERTa ensemble + 3 Electra + 5 RoBERTa; 70B LLMs (`Platypus2`, `Xwin`) via simjeg's layer-by-layer code
  answer ~500 hard questions.
- **Progression (private / public):** 5 DeBERTas + untuned reranker 0.894 / 0.893 -> + Platypus on 500 hard Qs 0.91 /
  0.909 -> + new DeBERTas, 3 Electra, 5 RoBERTa 0.912 / 0.914 -> **+ tuned reranker 0.927 / 0.922** -> + Xwin 0.928 / 0.926.
- **Ablation of final (private):** final 0.9284; w/o Xwin 0.9272; **w/o Platypus2 0.9288** (slightly better without it);
  w/o both LLMs 0.9165; w/o bge-small 0.9272; w/o MiniLM 0.9258; w/o Electra+RoBERTa 0.9246; **w/o reranker 0.9113**.
  His conclusion: the QA part is strong given correct context; the weak link is first-stage retrieval with two tiny
  embedders that were never tuned on competition data.

### 4th and 5th (short, useful for diagrams)

- **4th, Preferred Scantron** (https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446307): Elasticsearch
  server **inside the Kaggle kernel** over cirrussearch Wikipedia split into sentences; sentence-wise keyword queries for
  every sentence of prompt and options; three context orderings (v3 ES score, v5 edit distance, v7 semantic with
  `msmarco-bert-base-dot-v5` for prompt and `all-mpnet-base-v2` for options). Retrieval tuned zero-shot (no training in
  the loop) with Llama-2-7B, then a pretrained-with-context model on 2,000 held-out rows of the 60k set. Reader:
  DeBERTa-v3-large on the 60k + 40k sets; 512 -> 768 -> 1280 tokens kept helping; prompt+choices first, then pack as
  much context as fits. Inference ensembles the three context types. Trick: symlink read-only ES files from
  `/kaggle/input` because `/kaggle/temp` I/O was unstable.
- **5th, Preferred ChattyKids** (https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446293): own
  Wikipedia parse (`wikitextparser` + keeping `{val}`/`{math}` templates and `<math>` tags); **BM25 via pyserini/Lucene**
  over 74M paragraphs (ORQA argument: when the questioner knows the answer, BM25 suffices); dense: `instructor-xl` over
  sentence blocks with a FAISS index quantized 300 GB -> 10 GB, and `bge-large-en` over mbanaei's STEM-270k paragraphs.
  Models: base (non-instruct) `Mistral-7B-v0.1` and `Llama-2-70b-hf`, QLoRA with near-paper hyperparameters, read the
  next-token probability of " A".." E". **70B on Kaggle**: per-layer quantized weights stored as a dataset, layer-by-layer
  inference, xformers memory-efficient attention, ~6 GB GPU. **TTA**: 5 rotations of option order packed into one
  sequence with a custom attention mask so the shared context is encoded once. Single 70B + BM25 top-2 + 5x TTA =
  0.925 private. Final 3-stage cascade: Mistral-7B ensemble on all -> Llama-2-70B on the least-confident 40% -> Llama-2-70B
  with 2-3x longer bge context on the least-confident 5%; 0.926 private / 0.928 public. Validation on 200 + 300 + their
  own 1,000 GPT-3.5-generated questions (prompt in the post), never used for training.
- **6th, RAG to riches** (https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/447647, GitHub
  https://github.com/rbiswasfc/llm-science-exam): custom STEM corpus by crawling Wikipedia categories (~500k pages);
  synthetic MCQs from GPT-3.5, GPT-4, Llama-2-70B and Falcon-180B used to **fine-tune the retrievers** (`gte-base`,
  `bge-base-en-v1.5`, NCE loss with in-batch hard negatives) as well as the readers; `deberta-v3-base` cross-encoder
  reranks top-10 and keeps 2-4 by a dynamic threshold. Readers: DeBERTa (span classification with cross-option pooling,
  MC head, PET) + Mistral-7B-OpenOrca LoRA (lr 3e-4, r=16, alpha=16, all linear + embeddings, dropout 0.1, bf16). Top 5%
  by prediction entropy go to zero-shot `platypus2-70b-instruct` and `sheep-duck-llama-2` (yes/no logit difference).

### Key discussion threads and notebooks

| Thread | Votes | Takeaway |
|---|---|---|
| Radek Osmulski, "levelling the playing field - 6000 new high quality train examples" https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/426174 | 159 | GPT-3.5-generated MCQs shared so people without OpenAI access could train; later 15k (431786) and 5.9k (432316) more. The winners' training data came from here. |
| Chris Deotte, "60k Dataset Achieves LB 0.830+" https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/436383 | 282 | Merged all public MCQ sets + Wikipedia context ("Open Book" technique diagram); single DeBERTa reaches LB 0.830+. Became the standard training/validation set (4th place validated on it). |
| Chris Deotte, "Bonus 40k Dataset" /discussion/440908 and "Is your LLM truthful? Bonus 99k" /discussion/444202 | 156 / 89 | More data incl. MMLU-derived rows. |
| simjeg, "[LB 0.836] Zero-shot 70B model + RAG" https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/440620 | 182 | Ran Platypus2-70B **layer-by-layer, sample-by-sample** on a Kaggle GPU; asking "is this answer correct? yes/no" 5 times beat asking "which is best?" (0.81 vs 0.75 LB without RAG). On the 200 samples MAP@3: DeBERTa 0.75, Beluga-7B 0.81, Platypus2-13B 0.85, Platypus2-70B 0.91 (with RAG). Unlocked 70B for 3rd, 5th, 6th place. |
| simjeg, "[LB 0.872] Zero-shot 70B model + improved RAG" /discussion/443334 | 55 | Follow-up with better retrieval. |
| Mohammadreza Banaei, "[86.2%] Retrieval with 270K Wiki STEM articles + long context support" https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/442595 | 137 | Found a 270k-article STEM cluster that covers most LB gold articles; re-fetched via Wiki API to fix missing numbers; TF-IDF + Longformer-large. Used by 5th place. |
| Psi, "Simple loss/perplexity LLM baseline" /discussion/424242 | 95 | Early LLM-scoring baseline from the eventual winners. |
| Abhishek Thakur, "LLM finetuning tutorial" /discussion/424519 and "LLAMAv2 7B finetuning in Kaggle Notebooks" /discussion/430614 | 144 / 54 | How-tos that pushed people from DeBERTa toward LLM fine-tuning. |
| CPMP, "0.894 without training any model (+56th)" https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446232 | 53 | Paragraph-level retrieval over 35M paragraphs (~140 GB embeddings, streamed 1M per file, `faiss.knn_gpu` without an index, ~25 min); query = mean(prompt emb, mean of option embs) so it is order-free; Xwin-LM-70B yes-logit -> 0.892 private; a logistic-regression head on the last hidden state -> 0.908 private. Ran 8 h 58 min of the 9 h limit. |
| Chris Deotte, "Top 100 Solution - Fast RAPIDS TF-IDF RAG" /discussion/446318 | 73 | GPU TF-IDF retrieval on 2xT4. |
| "Wikipedia pages used to generate the training data!" /discussion/425242 | 56 | Early evidence that questions trace back to specific Wikipedia pages -> retrieval is the game. |

### Shake-up

Mild at the very top: H2O was #1 on both boards. Public #2 ("kaggle is few shot learner") finished private #12; public #3
("Life is full of surprise") private #15; public #6 lytic rose to private #2; public #10 Podpall to #3. Largest jump in
the private top 50: PhysicsPhriends public 178 -> private 36. The 20% public split (~800 questions) made MAP@3 noisy;
5th place notes "MAP@3 is a bit unstable".

### Transferable lessons

1. **When the questions were written from a corpus, retrieval is the model.** Every top-6 team was RAG; H2O spent most of
   its effort on retrieval and kept the reader fixed; Podpall's ablation shows the reranker worth +0.017.
   (/discussion/446422, /discussion/446358)
2. **Parse the corpus yourself.** Three of the top five found public Wikipedia dumps dropped numbers and templates and
   rebuilt them (cirrussearch, patched wikiextractor, wikitextparser). (/discussion/446422, /discussion/446358, /discussion/446293)
3. **Turn the LLM into a classifier, and exploit shared prefixes.** Binary per-option heads + KV-cache reuse (1st) and
   packed TTA with attention masks (5th) are how 7B-70B models fit 9 hours. (/discussion/446422, /discussion/446293)
4. **Cascade by confidence.** Cheap model on everything, 70B only on the uncertain 5-40% (3rd, 5th, 6th). (/discussion/446293, /discussion/447647)
5. **Build a validation set that is harder than the given one.** 200 samples saturated at 0.99; winners validated on
   6k / 2k / 1k held-out synthetic questions and picked the best-CV submission. (/discussion/446422, /discussion/446307)
6. **Zero-shot giants are a real baseline.** No-training 70B + RAG reached 0.892-0.908 private. (/discussion/440620, /discussion/446232)

### UNVERIFIED

- Exact Kaggle GPU type for this competition is not on the competition pages; "2x T4 or P100" is from participants' posts.
- Podpall's 70B model sizes: the post names "Platypus2" and "Xwin" and credits simjeg's 70B code, but does not state
  the parameter count explicitly.
- H2O LoRA rank, lr and epochs-per-model were not published in the posts.
- The 7th-place (days) and 10th+ writeups were not read.

---

## 2. LLM - Detect AI Generated Text (2023-24) -- "DAIGT"

### Facts

- **URL:** https://www.kaggle.com/competitions/llm-detect-ai-generated-text (id 61542, forumId 3863365). Hosts:
  Vanderbilt University + The Learning Agency Lab.
- **Dates:** 2023-10-31 -> final submission 2024-01-22 (entry / merger deadline 2024-01-15).
- **Task:** binary -- was this middle/high-school essay written by a student (0) or an LLM (1)? Output a probability.
- **Data:** ~10,000 essays across **7 essay prompts**. Training set = essays from **2 prompts**, "nearly all" written by
  students, with only a few generated examples ("You may wish to generate more essays"). Hidden test ~9,000 essays from
  the other prompts, from "multiple, unknown generative models". `train_prompts.csv` gives the instructions and source
  articles. Competitors quickly showed the essays come from the **PERSUADE 2.0 corpus** (also used in Feedback Prize)
  and identified which PERSUADE prompts the test used (discussion/452172, /453410).
- **Metric:** ROC AUC on `generated` probability. Only the ranking matters, which is why rank-averaging ensembles worked.
- **Public / private:** public = 46% of test.
- **Prizes:** $110,000. Leaderboard: 1st $20k, 2nd $10k, 3rd $8k, 4th $7k, 5th-7th $5k each. **Efficiency track** (CPU
  only, no GPU): 1st $20k, 2nd $10k, 3rd $8k, 4th $7k, 5th $5k. Efficiency score as printed on the page:
  `Efficiency = AUC / (Benchmark - maxAUC) + RuntimeSeconds / 32400` (lower is better; Benchmark = sample_submission AUC,
  maxAUC = best private AUC). The AUC term is negative because Benchmark < maxAUC.
- **Teams:** 4,358 teams, 5,264 competitors, **110,052 submissions** (the most of any comp in this bank).
- **Limits:** CPU or GPU notebook <= 9 h, internet off, external data allowed, 3 final submissions selectable.
- **Quirk:** test essays contained **artificial character-level obfuscations / typos** (a "hidden attack"), and some
  characters present in PERSUADE never appear in `train.csv` (452172). Spotting and modelling these mattered: 4th place
  concluded the obfuscations were "completely random" character insertions/replacements (470179).

### Top placements (private LB)

| Rank | Team | Private | Public rank | Approach (one line) | Writeup |
|---|---|---|---|---|---|
| 1 | Udbhav Bamba, Raja Biswas, Nicholas Broad (team name = 4 emoji) | 0.987824 | 9 | 160k-essay "datamix" built for diversity (many LLMs, decoding configs, PERSUADE-finetuned generators, attack augmentations); rank-ensemble of Mistral-7B QLoRA, DeBERTa, Ghostbuster features, unsupervised | https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/473295 (short: /discussion/470121) |
| 2 | Guanshuo Xu (solo) | 0.983412 | 17 | DeBERTa "pretrained" on ~500k SlimPajama human/AI pairs, then fine-tuned on essays from LLMs fine-tuned to mimic students | https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470395 |
| 3 | nlp team (Ivan Isaev, KKY, Muhammad Ahmed, Dmitrii Khizbullin, Yevhenii Maslov) | 0.974994 | 15 | 12 DeBERTa-v3-large (3 trained on 0.5-1.2M Pile/SlimPajama continuations from ~35 LLMs) + TF-IDF pipeline with confident pseudo-labels | https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470333 |
| 4 | Ertugrul & Chase (Chase Bowers, Ertugrul Demir) | 0.973915 | 3 | "Combined arms": TF-IDF with BPE tokenizer + random-typo augmentation, Mistral-7B log-prob features, Longformer source-classifier, DeBERTa-large on 700k rows | https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470179 |
| 5 | Linguistic Ninjas (tarobxl, Marcel Guina, TGK, James Day, Dom De Jonge) | 0.972147 | 22 | DeBERTa-v3-large + Mamba-790m on 1.7M examples; **test-time domain adaptation**: short-context students distilled on the test set | https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093 |
| 6 | Davide Cozzolino (solo) | 0.969132 | **2,995** | biggest climber in the gold zone | /competitions/llm-detect-ai-generated-text/writeups/davide-cozzolino-6nd-place-solution-with-code |
| 7 | Hao Mei (solo) | 0.965133 | **2,357** | "generate data with non-..." (title truncated in API) | /competitions/llm-detect-ai-generated-text/writeups/hao-mei-7th-place-solution-generate-data-with-non- |
| 8 | Abdullah Meda (solo) | 0.956501 | 1,031 | linguistic features: perplexity + GLTR | https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470224 |
| 9 | LLMLab (Darragh, David Austin, Dieter) | 0.947710 | **1** | TF-IDF + reverse-engineered deobfuscation + 2 DeBERTa-v3-large + 1 RoBERTa on 200k-700k diverse rows | https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470255 |
| 21 | Secret Sauce | 0.932421 | **2** | 0.986 public, selected private 0.932 (best unselected 0.957) | https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470148 |
| 40 | Kelvin | 0.919670 | -- | **1st place Efficiency Prize** (title per LB API) | /competitions/llm-detect-ai-generated-text/writeups/kelvin-1st-place-efficiency-prize-scientific-journ |

Gold = top 18. Public #4 CPMP finished private #163; public #5 J.M -> #105; public #8 -> #128; public #10 -> #161.

### 1st place -- Raja Biswas, Udbhav Bamba, Nicholas Broad

Sources: short https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470121 (Raja Biswas), comprehensive
https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/473295 (Nicholas Broad). Code
https://github.com/rbiswasfc/llm-detect-ai, inference `nbroad/r100-ensemble`, data `conjuring92/ai-mix-v26`.

- **Thesis: data beats model.** "The modeling approach was less important, as we had multiple single models in the
  0.970+ range due to the quality of the dataset." Each individual model "would have been in the gold medal range".
- **Datamix (~160k rows, ~40k human)** built incrementally, each iteration plugging blind spots of the previous model:
  - Human text: **all 15 PERSUADE prompts**, plus GPT-2 output dataset (human side), ELLIPSE, NarrativeQA, Wikipedia,
    NLTK Brown, IMDB reviews.
  - Generated text, four sources: (1) proprietary LLMs: gpt-3.5, gpt-4, claude, cohere, gemini, palm; (2) open LLMs:
    llama, falcon, mistral, mixtral; (3) existing sets: T5-generated synthetic essays from Feedback Prize 2, DAIGT V2
    subset, OUTFOX, Ghostbuster data, gpt-2-output-dataset; (4) **LLMs fine-tuned on PERSUADE** (mistral, llama, falcon,
    deci-lm, t5, pythia, BLOOM, GPT2) three ways: instruction-tuned with metadata (prompt name, holistic score, ELL status,
    grade) -> essay; one prompt held out (generate only the held-out prompt, to force new styles); span-wise generation
    (one discourse span at a time, conditioned on the rest).
  - Decoding diversity: **contrastive search**, guidance scale, typical_p, suppress_tokens, high temperature, large
    top-k; prompts with/without source texts, rewrite prompts, fill-in-the-blank (mask words, ask LLM to reconstruct).
  - **Attack augmentations** on a random subset: spelling correction, char delete/insert/swap, synonym replacement,
    obfuscations, back-translation, random capitalisation, sentence swap. Plus a heavily pre-processed copy (lowercase,
    no special chars, normalised whitespace) to resist character attacks.
- **Models:**
  - **Mistral-7B-v0.1 (Q)LoRA, sequence classification**: `r=64, lora_alpha=16, dropout 0.1, target q/k/v/o`. Best single
    model: **0.984 private / 0.966 public**.
  - **Ghostbuster re-implementation** with Llama-7B + TinyLlama-1.1B (same tokenizer) instead of OpenAI davinci/ada;
    unigram + trigram models trained on Brown with the Llama tokenizer; skip first 25 tokens; 10 aggregations (min, max,
    mean, median, 10/25/75/90% quantiles, L2, variance) over 4 token-prob sequences and 6 ratios = **100 features**;
    SVM + Random Forest via RAPIDS. 0.974 private / 0.957 public.
  - **DeBERTa-v3-small trained from scratch at inference time**: custom tokenizer on train + test, MLM pretraining on
    train + test, fine-tune on datamix + Mistral pseudo-labels of test, 1 epoch. 0.943 / 0.942.
  - DeBERTa-v3-large classification (BCE with continuous labels, e.g. instruction-tuned-LLM essays < 1) and
    DeBERTa-v3-large **pairwise ranking loss with margin**: 0.963 / 0.961.
  - Ahmet's unsupervised approach (discussion/465156) with Mistral-confident weak labels and mean (not min) aggregation:
    0.971 / 0.957.
- **Ensemble:** convert each model's scores to **ranks** (1..n), average ranks, light weights by public LB and intuition,
  most weight to Mistral.

### 2nd place -- Guanshuo Xu (solo)

Source: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470395 (inference `wowfattie/daigt-2nd-place`,
training data `wowfattie/daigtpretraindata`).

- Started late. DeBERTa fine-tuned on shared datasets gave "nearly perfect CV but much lower and unstable public LB".
- Inspired by James Day's post (465882): generated **~500k human/AI pairs from SlimPajama** with various LLMs; DeBERTa-v3-large
  on it = 0.916 public / 0.967 private. Call this the **"pretraining"** step: LM -> general human-vs-AI classifier.
- **Domain adaptation:** test prompts all ask students to write about source articles he could not find (except Venus),
  so AI essays could look very student-like. He **fine-tuned LLMs on student essays with H2O LLM Studio** (no code, CSV +
  GUI) so they mimic students incl. citations and typos, generated essays, fine-tuned the classifier on them -> ~0.94 /
  0.98.
- **Final:** 2x DeBERTa-v3-large + 1x DeBERTa-large, plus the same three fine-tuned on DAIGT-V4 -> **0.967 public / 0.983
  private**. Blending with a 0.961 TF-IDF kernel gave his best public score but a lower private one.

### 3rd place -- nlp team (writeup by Yevhenii Maslov)

Source: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470333 (inference
`evgeniimaslov2/llm-daig-3rd-place-solution`).

- **Preprocessing:** sorokin's T5 deobfuscator (457819) applied only to texts with >15 errors; drop symbols not in
  train; normalise encodings.
- **Hard-example mining (4 models, 11k rows):** start from alejopaullier's DAIGT external data; each iteration add the 500
  human + 500 generated samples the previous model got most wrong; check LB every 4th iteration; stop when LB stops
  improving. Best single 0.927 public / **0.845 private** -- i.e. it overfit public.
- **Large-scale continuations (3 models):** continuations of **Pile and SlimPajama** text by **~35 open-source LLMs**;
  filter too short/long, code, maths, non-English, high non-letter ratio; 3 sampling scenarios by temperature with
  random top_p/min_p and presence/frequency penalties. Trained on 500k, 1M, 1.2M rows; max_len 256 train / 1512 infer,
  batch 48. Best single (~1M): **0.956 public / 0.967 private**. 5 more models fine-tuned from these on the 11k set: +public,
  -0.005 private.
- **TF-IDF pipeline:** public notebook "0.960 phrases are keys" with more CatBoost/LGBM iterations, voting weights
  [0.05, 0.225, 0.225, 0.5], +1k confident test pseudo-labels (p<0.01 or >0.99): public flat, **private 0.893 -> 0.927**.
- **Ensemble:** step 1 blend TF-IDF with 11k-models only where transformers are confident (<0.1 or >0.9), TF-IDF alone in
  the middle; step 2 plain weighted average with the large-data models. **Post-processing:** per prompt with >1000 rows,
  UMAP on TF-IDF, distance to 7 nearest human vs 7 nearest generated, scale by the ratio clipped to [0.9, 1.1].

### 4th and 5th (short)

- **4th, Ertugrul & Chase** (https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470179; Chase's
  LLM part in /discussion/471934): TF-IDF n-grams (3,7) on a **BPE tokenizer trained on the data** (from Ertugrul's
  post 458522) over DAIGT V2, with **random typos injected into 30% of generated essays** to stop typos being a
  human-only signal; Mistral-7B token-prob/perplexity features -> classifiers re-trained on TF-IDF pseudo-labels of test;
  Longformer predicting the **source dataset**; DeBERTa-v3 on ~700k public rows, 1 epoch, 1024 tokens, as the regulariser
  for uncertain rows. Weights set by "common sense", not public LB.
- **5th, Linguistic Ninjas** (https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093, code
  https://github.com/jday96314/AiContentDetection): data = PERSUADE 26k human / 327k generated, **Pile completions
  512k/512k**, SlimPajama completions 233k/233k, "Tricky Crawl" 125k human; generated with vLLM on 2x RTX 3090 + 1x 4090,
  AWQ 4-bit for >=13B, temperature 0-2, top-k {off,20,40}, top-p 0.5-1, frequency penalty 0-0.5 per request. Best models
  used 62-99% Pile data; PERSUADE always 1%. **Test-time domain adaptation:** teacher (1 DeBERTa + 2 Mamba, 1024 tokens)
  soft-labels the test set; two "short-context students" (128 and 256 **characters**) are fine-tuned on random test
  chunks to imitate the teacher, then predict overlapping chunks (stride = half). Full-context DeBERTa 0.970 private;
  students 0.977 without Mamba (would have been 3rd) vs 0.972 with Mamba; the best submission was not selected.

### Key discussion threads and notebooks

| Thread | Votes | Takeaway |
|---|---|---|
| Radek Osmulski, "500 LLM generated essays specifically for this competition" https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452155 | 282 | First shared generated essays; the train set had almost no positives. |
| Nicholas Broad, "Train prompts exact match with PERSUADE corpus" https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452172 | 95 | Train essays = PERSUADE essays after cleaning; lists all 15 PERSUADE prompts; some characters were stripped from competition text. |
| Carl McBride Ellis, "A '7 Prompts' training dataset" /discussion/453410 | 189 | Test prompts identified among PERSUADE's 15. |
| Darek Kleczek, DAIGT datasets: /discussion/452401, /455517, /463087, /468246 (V4) | 139 / 170 / 132 / 97 | The community DAIGT V2/V4 train sets most teams used (1st, 2nd, 4th all cite them). |
| James Day, "0.963 with BERT - Transformers love diverse data" https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/465882 | 155 | ~500k Pile continuations from 12 LLMs moved DeBERTa from ~0.75-0.80 to 0.958-0.963 LB; inspired 2nd and 3rd place. |
| Ertugrul Demir, "Using Custom Tokenizers to Boost Your Score" https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/458522 | 140 | Train a BPE tokenizer on the corpus so typos become distinct vocabulary pieces for TF-IDF; also seeded 1st place's from-scratch DeBERTa. |
| Ahmet Erdem, "Superfast Unsupervised Baseline [0.91 LB]" /discussion/465156 | 96 | Similarity-based weak labelling; adopted by 1st place. |
| ivan (sorokin), "T5 deobfuscator" /discussion/457819 | 85 | Undo the injected typos; used by 3rd place. |
| narsil, "Find Typos - 0.714 LB in 14 lines of code" /discussion/452750 | 72 | Typo count alone is a signal: humans make typos, LLMs do not -- until the hosts' obfuscation muddies it. |
| CPMP, "The curse of early success" https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470154 | 77 | Reached 0.970 public with GPT-2 tokenizer, n-grams (3,9), vocabulary fit on test, min_df=2, GPT-2 n-gram perplexity in place of TF, LinearSVC -- and fell to #163 private because he never built general data/models. |
| Alejandro Mosquera, "On the performance of non-NN detectors" /discussion/454448 | 50 | Asked whether TF-IDF's "unreasonable" strength was an artefact of this test set; the shake-up answered yes. |
| Piotr Kozlowski, "0.961 -> 0.963 by using Levenshtein distance for error correction" /discussion/468767 | 67 | Typo-correction trick for TF-IDF pipelines. |

### Shake-up

One of the biggest in Kaggle NLP. Public LB was dominated by test-fitted TF-IDF pipelines (vocabulary built on test
essays). Public #1 LLMLab -> private #9; #2 Secret Sauce -> #21; #4 CPMP -> #163; #8 -> #128; #10 -> #161. Winners came
from public 9, 17, 15, 3, 22 -- and 6th / 7th place climbed from public **2,995** and **2,357**. The winning team: "We are
pleasantly surprised by the unexpected outcome." The general-purpose detectors trained on big diverse data held; the
test-specific n-gram models did not.

### Transferable lessons

1. **Build the training set; the model is secondary.** 1st place: multiple single models at 0.970+ purely from datamix
   quality. (/discussion/473295)
2. **Diversity of generators and decoding settings is what generalises** -- many LLM families, temperatures, contrastive
   search, fine-tuned student-mimicking generators. (/discussion/473295, /discussion/470395, /discussion/470093)
3. **"Pretrain" on large out-of-domain human/AI pairs, then adapt to the target domain** (2nd: SlimPajama -> student-like
   essays; 5th: teacher -> short-context students on the test set). (/discussion/470395, /discussion/470093)
4. **Anticipate adversarial noise.** Add typo / obfuscation / paraphrase augmentation rather than relying on typos as a
   label signal. (/discussion/473295, /discussion/470179)
5. **For AUC, ensemble ranks, not probabilities.** (/discussion/473295)
6. **A public LB that rewards test-fitting is a trap.** Vocabulary-on-test TF-IDF topped public and collapsed; pick
   final subs that you would trust on unseen prompts. (/discussion/470154, LB API shake-up numbers)

### UNVERIFIED

- Efficiency-prize ranking beyond the LB API title "Kelvin -- 1st place efficiency prize" (rank 40) and "SuperFei --
  solution of 4th place in efficiency LB" (rank 47) was not checked against an official efficiency leaderboard.
- 6th (Davide Cozzolino) and 7th (Hao Mei) writeups were not read; approach not summarised.
- The exact obfuscation process used by the hosts was never published in what I read; "random character-level" is 4th
  place's inference.
- Exact count of test essays per prompt and the generating LLMs are not disclosed on the data page.

---

## 3. LLM Prompt Recovery (2024)

### Facts

- **URL:** https://www.kaggle.com/competitions/llm-prompt-recovery (id 67121, forumId 4360200). Host: Google (Gemma launch).
- **Dates:** 2024-02-27 -> final submission 2024-04-16 (entry / merger 2024-04-09).
- **Task:** given `original_text` and `rewritten_text` (Gemma's rewrite), recover the **`rewrite_prompt`** that was given
  to Gemma (e.g. "Rewrite this essay but do it using the writing style of Dr. Seuss").
- **Data:** **one** example row in train.csv and test.csv. Hidden test "roughly 1,400" texts (Description: "1300+"),
  rewritten by **Gemma 7b-it** (Kaggle model variation `7b-it-quant`) with undisclosed prompts. Competitors had to generate
  all training data (host starter: `wlifferth/starter-notebook-generating-more-data-with-gemma`). Kaggle later shared
  "Supplementary Texts Rewritten by GenAI Models" (discussion/480288).
- **Metric:** for each row, embed prediction and truth with **sentence-t5-base** (the TensorFlow/Kaggle Models version
  `st5-base`), score = **Sharpened Cosine Similarity with exponent 3**, i.e. roughly `cos(e_pred, e_true)^3`, averaged over
  rows. Host's reason: "to attenuate the generous score given by embedding vectors for incorrect answers." Empty
  predictions throw an error.
- **Public / private:** public = 15% of test.
- **Prizes:** **$200,000** -- 1st $100k, 2nd $40k, 3rd $20k, 4th $14k, 5th $11k, 6th $10k, 7th $5k (+ Kaggle merch for 25
  people who uploaded a model to Kaggle Models and used it in a public notebook).
- **Teams:** 2,175 teams, 2,664 competitors, 43,837 submissions.
- **Limits:** CPU/GPU notebook <= 9 h, internet off; "submission runtimes have been slightly obfuscated" (up to 15 min
  variance). External data and models allowed.
- **The exploit (central to the story):** the metric, not the model, decided the podium. Because scores are averaged
  cosines-cubed in a fixed embedding space, a single **"mean prompt"** string scores ~0.6 on every row, and adversarial
  token strings that pull embeddings toward a central point (`lucrarea`, a Romanian word piece close to the `</s>` token in
  sentence-t5 space) add up to ~+0.05. All top-7 writeups used a mean prompt and/or `lucrarea`.

### Top placements (private LB)

| Rank | Team | Private | Public rank | Approach (one line) | Writeup |
|---|---|---|---|---|---|
| 1 | Khoi Nguyen (solo) | 0.7165 | 1 | Concatenated Mistral-7B / Gemma-7B-1.1-it predictions + an adversarial suffix of 8x `lucrarea` | https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494343 |
| 2 | Team Danube (Psi, Yauhen Babakhin, Pascal Pfeiffer) | 0.7139 | 3 | Models that predict the **target embedding** (cosine loss) + greedy token brute-force to turn the embedding back into a string, seeded with few-shot + LLM predictions + mean prompt | https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494497 |
| 3 | prompt = "don't say anything" (Pedro Bernardo, Eduardo Rocha de Andrade, matheus) | 0.7128 | 4 | Beam-searched mean prompt (with `lucrarea`) + Mistral-7B-Instruct-v0.2 full-prompt prediction behind a gate model + tags model + 12-cluster mean prompts | https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494621 |
| 4 | Yusef A. (solo) | 0.7092 | 2 | 0.69 mean prompt + Mistral-7B-it with `response_prefix = "Modify this text by"`; "ST5 tokenizer attack" | https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494362 |
| 5 | Z.D.Z (zengzhy5, Dmitry Abulkhanov, yuanzhe zhou) | 0.7071 | 5 | not read | /competitions/llm-prompt-recovery/writeups/z-d-z-5th-place-solution |
| 6 | flg (solo) | 0.7033 | 6 | T5 decoding + iterative mean-prompt refinement + LLMs (post titled "7th place") | https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494755 |
| 7 | Taras & Andre | 0.7026 | 7 | "Mastering mean prompt" | /competitions/llm-prompt-recovery/writeups/taras-andre-7th-place-solution-mastering-mean-prom |
| -- | Dipam Chakraborty et al. | -- | -- | Post titled "6th Place": **Universal Adversarial Triggers** (gradient token search, HotFlip-style) adapted to sentence similarity, rediscovered `lucrarea` automatically | https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494348 |
| 19 | ironbar (Guillermo Barbadillo) | 0.6729 | -- | ensembling 4 transformers, no mean-prompt hacks emphasised | https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494536 |

Gold = top 14. Shake-up was tiny: public top 10 = private top 10 (reordered).

### 1st place -- Khoi Nguyen (solo): "adversarial attack"

Source: https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494343 (inference `suicaokhoailang/1st-place-0-71`).

- **The trick:** append `" 'it 's ' something Think A Human Plucrarealucrarealucrarealucrarealucrarealucrarealucrarealucrarea"`
  to any prediction -> up to **+0.05**.
- **Why it works (his explanation):** in the HuggingFace sentence-t5, appending an extra `</s>` to one of two vaguely
  similar sentences pulls their cosine toward ~0.9 (up if lower, down if higher). cos 0.9 ~ **0.73 on the metric**
  (0.9^3 = 0.729). But the host scored with the **TensorFlow** version, whose SentencePiece config tokenises `</s>`
  literally as `<`, `/`, `s`, `>` -- so the special token cannot be injected. Searching the T5 vocabulary for tokens whose
  embeddings sit next to `</s>` turns up `lucrarea`; it is "a Walmart version of `</s>`" that pulls scores toward ~0.71.
  His theory: the token drags a sentence toward a focal point (perhaps the centre) of the space; if prediction and truth
  are far apart this shortens the distance, if they are already close it can hurt.
- **Models:** Mistral-7B (base and instruct) and **Gemma-7b-1.1-it**, trained on different datasets; Mistral slightly
  better; **no single model beat 0.65**. Predictions of all models are **concatenated**, each model prompted with a
  different opening verb ("Rewrite... Alter... Turn... Make...") for diversity: 0.70 -> 0.71.

### 2nd place -- Team Danube (Psi, Yauhen Babakhin, Pascal Pfeiffer)

Source: https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494497. Code: inference
`ilu000/2nd-place-team-danube-llm-prompt-recovery`, LLM Studio code `philippsinger/kaggle-prompt-llmstudio-psi-working-v3`,
data `ybabakhin/llm-prompt-recovery-generated-data`.

1. **Mean prompt by brute force.** For the mean of the target vectors, try each of the ~32k T5 tokens greedily. First
   submission was bad: `</s>` dominated the optimisation, but TF SentencePiece "has protection against special token
   injection". Excluding special tokens, the optimiser "really started to like tokens like `lucrarea`". Mean prompt alone
   ~0.65 LB.
2. **Embedding-prediction models.** H2O LLM Studio classification model whose target is the **768-d sentence-t5
   embedding**, trained with a **cosine-similarity loss** = the metric directly. Backbones: H2O-Danube / H2O-Danube2
   (1.8B) or Mistral-7B, "little difference". Local ~0.75+ on embeddings.
3. **Embedding -> string.** Greedy brute-force token search to match the predicted embedding; loses 3-4 points; local
   ~0.71, LB 0.68-0.69.
4. **Better seeds.** Initialise the optimisation with: few-shot predictions + LLM predictions trained to output only the
   change ("as a shanty") + a good mean prompt. Final string = `few-shot + LLM preds + mean prompt + 20 optimised tokens`.
   Example final prediction in the post is a long string mixing "shanty", "lucrarea", and many style words.
- **Data / CV:** Kaggle's supplementary texts were the most useful seed; Gemma used to generate original texts and
  rewrite prompts few-shot; extra random texts and prompts for diversity. ~350-sample validation set with good mean-prompt
  CV-LB correlation. Best CV = best LB = best private ("public and private was a fair split").

### 3rd place -- prompt = "don't say anything" (Eduardo Rocha de Andrade, Pedro Bernardo, matheus)

Source: https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494621. Code: https://github.com/pedromb/llm-prompt-recovery,
https://github.com/matheuspf/llm-prompt-recovery, https://github.com/arc144/kaggle_llm_prompt_recovery_public; inference
`arc144/cluster-prompts`.

Five components; 1-4 alone gave 0.71:
1. **Mean-prompt template.** Generate candidate prompts; pick a subsample whose distribution matches the public LB (by
   matching LB scores of single-prompt submissions); **beam search** over a couple thousand words to maximise score on
   it. Final global mean prompt starts `"improve phrasing text {full prompt + tags} lucrarea tone lucrarea rewrite this
   creatively formalize..."`. Predictions are inserted **after the third word**; only words not already present are added.
2. **Full-prompt model:** LoRA-fine-tuned **Mistral-7B-Instruct-v0.2** via **unsloth** ("probably the best finding";
   far better than v0.1 and Gemma). Zero-shot scored 0.61, most fine-tunes <= 0.6, final single 0.62. Data pipeline:
   (a) generate prompts with Gemini and GPT-3.5-turbo, asking also for expected input-text characteristics; (b) generate
   **paraphrased variants** of each prompt (same intent, different words); (c) generate matching original texts with
   Gemini / GPT-3.5; (d) rewrite with **Gemma-2b-it-quant** (7b was too slow) + unsloth; (e) cluster prompts by T5
   embeddings with **HDBSCAN** and balance per cluster; (f) strip "Sure, here is..." boilerplate. Steps (b)+(e) were the
   breakthrough to 0.62.
3. **Gate model:** `MistralForSequenceClassification`, 1 class, judges (original, candidate prompt, rewritten) -- 40%
   positives, 20% random negatives, 40% **hard negatives** (nearest prompt in T5 space). Bad full-prompt predictions are
   dropped.
4. **Tags model:** a second Mistral CausalLM predicting tags ("shanty", "summarize", "formal tone"); the main verb is
   essentially unpredictable, tags are not.
5. **Cluster mean prompts:** LBFGS directly in T5 space showed a single best point scores 0.72 CV (~0.70 reachable with
   real words). **KMeans with 12 clusters**: per-cluster ceilings 0.726-0.833, weighted ~0.76 CV. A Mistral classifier
   assigns test rows to a cluster; used only if it agrees with KMeans on the predicted prompt, else the global mean
   prompt. +0.005 CV, not enough for 0.72.

### 4th and the "6th" post (short)

- **4th, Yusef A.** (https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494362): "tldr 1: `lucrarea`";
  tldr 2: 0.69 mean prompt + Mistral-7B-it with `response_prefix = "Modify this text by"`. Theory: T5's tokenizer was
  built for English plus German and Romanian translation, so Romanian pieces live in the vocab. Noted that the TF/KerasNLP
  ST5 and the PyTorch ST5 **differ**: TF lacks the `<extra_id_X>` sentinel tokens and has **max length 128**, so long
  prompts score differently than local PyTorch CV suggests; posted a check comparing the two encoders.
- **Dipam Chakraborty's team, post "6th Place"** (https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494348):
  adapted **Universal Adversarial Triggers** (Wallace et al.) to sentence similarity: iteratively replace tokens using
  gradients of sentence-t5 to maximise similarity to a dataset of rewrite prompts; their optimised suffix contains
  `lucrarea` several times ("we didn't discover `lucrarea` ... seems like our method did"). Also found test prompts split
  into style/tone vs topic/keyword kinds; a zero-shot classifier routes between a style LLM and a keyword LLM plus a
  zero-shot Mistral. TextAttack was tried first but works on words, not tokens.

### Key discussion threads and notebooks

| Thread | Votes | Takeaway |
|---|---|---|
| moth, "How This Competition Works" https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/480683 | 154 | Explains generation stage -> metric stage; the community primer. |
| Kishan Vavdara, "All Datasets at One Place" /discussion/481811 | 191 | Index of community-generated (original, prompt, rewrite) sets. |
| Khoi Nguyen, "My thoughts on why the baseline is so hard to beat" https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/480703 | 49 | A generic "Improve that text" is already semantically close to most labels; being specific and wrong is punished. Early statement of the mean-prompt problem by the eventual winner. |
| Geremie Yeo, "[LB Probing Result] no rewritten texts begin with 'Sure'" https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/483374 | 44 | Probing showed the hosts cleaned Gemma boilerplate; training data should be cleaned too. "Improve that text." alone scored 0.58. |
| Rich Olson, "New Mistral 7B Public Notebook - .62 LB" https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/488354 and "T5 Travesty" /discussion/488128 | 85 / 38 | Force short outputs "Improve this text by [X]"; wrong prefixes and extra details are heavily penalised by ST5 (e.g. 0.66 -> 0.75 by shortening). |
| Vincent Sirius, "Merge the prompt to LB 0.61" /discussion/481116 | 31 | Concatenating good generic prompts raises the floor -- the seed of mean-prompt optimisation. |
| Paul Mooney, "Supplementary Texts Rewritten by GenAI Models" /discussion/480288 | 51 | Host-provided data; 2nd place found it the most useful seed. |
| Darien Schettler, "Subjective Competition Learnings" /discussion/483916 | 105 | List of practical issues (Gemma instability on Kaggle, Keras/TF tooling). |
| sokazaki, "Why doesn't kaggle ban obvious cheaters?" /discussion/493789 | 31 | End-of-comp complaints about suspicious accounts. |

### Shake-up

Essentially none at the top: public top 10 = private top 10. Psi: "Thankfully, public and private was a fair split". The
risk here was not overfitting but **metric mismatch** between local PyTorch ST5 and the scorer's TF ST5 (494362, 494497).

### Transferable lessons

1. **Read the metric's implementation, not its description.** The tokenizer difference between TF and PyTorch ST5 decided
   whether `</s>` injection worked and how long prompts scored. (/discussion/494343, /discussion/494362, /discussion/494497)
2. **With a similarity metric, optimise toward the centroid first.** A single mean prompt ~0.6-0.69; per-cluster centroids
   raise the ceiling. (/discussion/494621, /discussion/494497)
3. **Train on the metric.** Predict the 768-d target embedding with a cosine loss, then decode to tokens. (/discussion/494497)
4. **Adversarial token search is an optimisation tool** (greedy brute force, beam search, universal triggers) -- and a
   warning about embedding-based judges. (/discussion/494348, /discussion/494621)
5. **When labels are unseen, generate paraphrase-diverse data and clean it like the host did** (no "Sure, here is").
   (/discussion/494621, /discussion/483374)

### UNVERIFIED

- Dipam Chakraborty's team does not appear in the final private LB returned by the API, and LB rank 6 (flg) titles its
  post "7th place"; the team was apparently removed after the post. Reason not verified.
- "Romanian word piece" for `lucrarea` is Yusef A.'s theory, not confirmed.
- 5th (Z.D.Z), 6th (flg) and 7th (Taras & Andre) writeups were not read in detail.

---

## 4. LMSYS - Chatbot Arena Human Preference Predictions (2024)

### Facts

- **URL:** https://www.kaggle.com/competitions/lmsys-chatbot-arena (id 66631, forumId 4305895).
- **Dates:** 2024-05-02 -> final submission 2024-08-05 (entry / merger 2024-07-29). API deadline 2024-08-12 21:30 UTC =
  LB finalised after the rescoring on a new test set ("The leaderboard has been finalized", /discussion/527576).
- **Task:** given a prompt (possibly multi-turn) and two anonymous chatbots' responses from Chatbot Arena, predict the
  probability that the human judge preferred A, preferred B, or called a tie. Explicitly framed as building a **reward /
  preference model** (RLHF) and warned about position, verbosity and self-enhancement bias.
- **Data:** **55k** training rows (`id, model_a, model_b, prompt, response_a, response_b, winner_model_a/b/tie`; model
  names in train only); test **~25,000** rows. Text fields are JSON lists of turns. Data page warns of offensive text.
- **Metric:** multiclass **log loss** (eps=auto) over the three columns. Lower is better.
- **Public / private:** public = 55% (`leaderboardPercentage`). After the leak the **private set was replaced with
  unpublished data** and submissions re-run, so private scores (~0.97-0.98) are on a different, harder set than public
  scores (~0.83-0.88).
- **Prizes:** $100,000 -- 1st $25k, 2nd $20k, 3rd $20k, 4th $20k, 5th $15k.
- **Teams:** 1,849 teams, 2,210 competitors.
- **Limits:** CPU/GPU notebook <= 9 h, internet off, external data allowed, runtime obfuscated by up to 15 min. Kaggle
  GPU = **2x T4** (2nd and 5th place discuss T4 / sm75 limits).
- **The leak (2024-08-04/05):** Yurnero's team noticed the top public score "plunged" and found the source within a day:
  a random 1/3-table submission with leak-altered rows scored **1.030 vs 1.098** (/discussion/524079). Cause (host Lisa
  Dunlap, /discussion/524374): LMSYS's public **leaderboard-analysis notebook** contained past battles' **metadata and
  human labels** (no chat text) -- including Kaggle train and test battles -- matchable via fields like token counts. Fix:
  deadline kept, private test replaced with unpublished data, all selected submissions re-run. Kaggle: "don't select
  leak-based submissions" (/discussion/524356). Public #1 "Leak User" scored **0.29588** public and finished private #174.
  Many top teams added "(no leak)" to their names.

### Top placements (private LB on the replacement test set)

| Rank | Team | Private (log loss) | Public rank | Approach (one line) | Writeup |
|---|---|---|---|---|---|
| 1 | BlackPearl (no leak) -- sayoulala (solo) | 0.96898 | 6 | Distil Llama-3-70B + Qwen2-72B (5-fold, after UltraFeedback post-pretrain) into Gemma-2-9B; average 5 LoRAs; GPTQ 8-bit + TTA | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629 |
| 2 | [Rist] pref (no leak) -- tascj, sakaku, kapenon | 0.97425 | 5 | **Full-parameter** fine-tuning of Gemma-2-9B/27B + ArmoRM-Llama3-8B, "full swap" augmentation, 240k pseudo-labels, custom Triton kernels for T4 | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685 |
| 3 | In The Arena (no leak) -- Raja Biswas, Mark Tenenholtz | 0.97770 | 8 | Start from **reward models** (FsfairX-Gemma2-RM, RLHFlow pair-preference Llama3-8B) + 500k pseudo-labelled pairs generated with vLLM | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766 |
| 4 | CAST (no leak) -- DL, xxyc, w1623843225, fufu2022 | 0.98031 | 15 | Gemma-2-9b-it LoRA r=64, smart per-turn truncation, UltraFeedback pseudo-labels, rule post-processing for empty/identical responses | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/529067 |
| 5 | Team Danube -- Psi, dott, Pascal Pfeiffer | 0.98129 | 14 | Pre-train Gemma-2-9B on UltraFeedback win/loss, then fine-tune; 2 models, one on swapped order; INT8 on 2xT4 | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527669 |
| 9 | Ebi (no leak) | 0.98280 | 21 | not read | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527704 |
| 16 | Chris Deotte (no leak) | 0.98532 | -- | Step-by-step public-notebook -> near-gold recipe; LoRA tuning method; **truncate left** | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527596 |
| 18 | Shelton (no leak) | 0.98551 | 104 | "More data + higher rank (QLoRA)"; biggest climber in top 20 | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527595 |
| 174 | Leak User | -- | **1 (0.29588)** | leak exploitation; dropped on the new test set | -- |

Gold = top 13. Other writeups via LB API: 13th "gpt2-chatbot-which-trusted-cv", 19th MRIC, 21st "Human bias" (pseudo
labels + feature engineering, /discussion/527627), 22nd Team LMFlow, 26th "Dont shake pls" (/discussion/527591).

### 1st place -- sayoulala (BlackPearl), "Distill is all you need"

Source: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629. Code: https://github.com/shyoulala/LMSYS_BlackPearl,
inference `sayoulala/v7-gemma-gptq`.

- **Data:** Kaggle train + "ut" data = UltraFeedback converted to competition format (Darek Kleczek's 157k set,
  /discussion/499756) + the 33k extra LMSYS rows (/discussion/500973).
- **Backbones:** Llama-3-70B, Qwen2-72B, Gemma-2-9B. `AutoModelForSequenceClassification`; LoRA for 9B, **QLoRA for the
  70B/72B**; all linear layers; **r=64, alpha=128**; max_len 1024; 2 epochs; global batch 64.
- **Step 1, post-pretrain:** 1 epoch on UltraFeedback for all three models, lr 1e-5.
- **Step 2, teacher logits:** 5 folds (train = 4/5 Kaggle + 33k; dev = 1/5 Kaggle) for Llama-3-70B and Qwen2-72B; infer
  the class distribution on the training set.
- **Step 3, distil:** fine-tune Gemma-2-9B with a distillation loss on the teachers' logits ("at least three loss for
  training", lr 5e-5).
- **Step 4, merge:** **average the LoRA layers of the 5 fold models** into one.
- **Step 5, quantise:** GPTQ 8-bit; TTA with length 2000 at submission.
- **CV (5 folds, log loss):** Qwen2-72B 0.875 / 0.881 / 0.869 / 0.880 / 0.875; Llama-3-70B 0.874 / 0.877 / 0.877 / 0.873 /
  0.873; distilled Gemma-9B 0.862 / 0.876 / 0.858 / 0.872 / 0.868 (student at or below teachers). Public LB 0.882 (0.876
  with TTA), **private 0.96898**. He called CV and LB "very consistent". Hardware per secondary source (Nebius blog,
  https://nebius.com/blog/posts/chatbot-arena-competition-review): 8x A100 80GB -- not stated in the Kaggle post.

### 2nd place -- [Rist] pref (tascj, sakaku/liushuzhi, kapenon)

Source: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685. Code: https://github.com/tascj/kaggle-lmsys-chatbot-arena,
inference `tascj0/lmsys-0805`.

- **CV:** `StratifiedGroupKFold` grouped by prompt, 20% held out. +21k rows from the deduplicated 33k set.
- **Input:** MT-Bench judge instructions (arXiv 2306.05685) as template; prompt-A-B ("PAB") format mostly (PAPB better
  only for 1.5B); gemma2-9b max length ~4340 tokens.
- **Head:** custom `Dropout -> Linear(h, h/2) -> Dropout -> GELU -> Linear(h/2, 3)`, re-initialised after loading (the
  default seq-cls head init gave high early loss).
- **No LoRA -- full-parameter training.** BF16 + an optimizer with **Kahan summation** lets a 7B train on one A100 80G;
  9B needs two. Last 10 days on 4x A100 80G.
- **"Full swap":** include each sample **and** its A/B-swapped twin, accumulating both gradients into the same optimizer
  step -- stable **-0.003** log loss vs random swap for gemma2-9b; adding PAB/PAPB formats the same way gave -0.001.
- **Stage 1:** fine-tune `gemma-2-9b-it` (val 0.891), `gemma-2-27b-it` (0.883; only trained after batch 80 and no grad
  clip), `RLHFlow/ArmoRM-Llama3-8B-v0.1` (0.899); average 0.876.
- **Stage 2:** pseudo-label **240k** rows (110k from lmsys-1m, 130k from external sets screened with a 1.5B model) with the
  stage-1 ensemble; fine-tune gemma-2-9b and ArmoRM on them (window attention disabled for sm75 kernels).
- **Stage 3:** continue on 55k + 21k. Val 0.884 and 0.890, ensemble 0.876-0.877. ArmoRM run on AB-swapped input;
  ensemble 2:1; old-LB 0.868.
- **Speed:** flash-attn 2.6.2 (logit softcapping), `flash_attn_varlen_func` with concatenated sequences and `cu_seqlens` --
  **no padding anywhere**; transformer_engine RMSNorm / fused RoPE. **Inference on 2x T4:** layers split across the two
  GPUs and pipelined; Triton kernels from lightllm (`context_attention_fwd` with softcapping, `gelu_and_mul`,
  `silu_and_mul`) and lmdeploy (`rms_norm`, `fused_rotary_emb`); xformers for Llama-3.

### 3rd place -- In The Arena (Raja Biswas, Mark Tenenholtz)

Source: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766. Code: https://github.com/mtenenholtz/lmsys-chatbot-arena-solution,
https://github.com/rbiswasfc/lmsys-arena, submission `conjuring92/lmsys-3rd-place-submission`.

- **Key idea 1 -- start from reward models, not chat models:** `sfairXC/FsfairX-Gemma2-RM-v0.1` instead of gemma-2-9b-it,
  `RLHFlow/pair-preference-model-LLaMA3-8B` instead of Llama-3.1-8B-Instruct. Gemma2-RM alone ~0.895 LB.
- **Key idea 2 -- large-scale pseudo-labelling:** 500k+ new pairs by generating responses to lmsys-1m prompts with diverse
  open LLMs via **vLLM** (temperature 0.7, top_p 0.9), batching by **number of turns** rather than length for throughput,
  plus `mlabonne/orpo-dpo-mix-40k`. Soft-labelled by a baseline ensemble, mixed with competition data, retrained. Gemma2-RM
  with PL: 0.880 LB without TTA at 8k length. 2 PL rounds; second added little. UltraFeedback no better than ORPO-DPO mix.
- **Mark's training:** QLoRA r=64, alpha=16, all linear, batch 16, seq len 1800; `<PROMPT>..<RESPONSE A>..<RESPONSE B>`
  tags not added to vocab (training embeddings hurt). **Disabling Gemma-2 logit softcapping** improved CV and LB by ~0.002.
  A/B flip TTA ~0.007.
- **Raja's baseline:** QLoRA r=32, alpha=64, dropout 0, **label smoothing 0.03**; ~300 high-loss rows removed; >4 turns ->
  keep first 2 + last 2; rslora / lora+ / DoRA / full-tuning not clearly better. 0.913 public.
- **Final:** 2x Gemma-2-9b checkpoints (0.880 and 0.896 public), max length 4096 / 3072, sorted by length, one with A/B
  swap.
- **Did not work:** Gemma-2-27B (could not train it well -- "seems to have been the sauce"); distillation from larger
  models ("totally convinced ... was the key, but we couldn't get it right"); predicting the model identity as auxiliary
  loss; Llama-3.1 worse than Llama-3. PL loosened CV-LB correlation.

### 4th and 5th (short)

- **4th, CAST** (https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/529067): official 55k + 33k dedup,
  20 folds but trained on one; stage 2 adds pseudo-labels for 30k UltraFeedback rows (>100k total), retrain from scratch.
  `gemma-2-9b-it` + `Gemma2ForSequenceClassification`, LoRA bf16 r=64, alpha=16, dropout 0.05; max_length 2048 train /
  3072 infer; 4x A100; stage 1 ~10 h, stage 2 ~15 h. **Truncation:** "Round k" template; when the last round overflows,
  split remaining budget 20% prompt / 40% A / 40% B with "......", drop the round if <80 tokens remain. A/B swap TTA.
  **Post-processing:** empty response (`[null]`, `[]`...) -> [0.04, 0.88, 0.08] for (empty, non-empty, tie); identical
  responses -> tie [0.06, 0.06, 0.88].
- **5th, Team Danube** (https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527669): no pseudo-labels, no
  distillation, no 1M set. Noticed RLHFlow reward models fine-tuned better than base/instruct -> **pre-trained Gemma-2-9B
  on all UltraFeedback with binary win/loss**, which improved CV and public LB "by up to 20 points"; single fold ~0.880.
  Final = same config trained twice on full data with A/B order swapped, one run inferred on swapped input (TTA); blend
  ~0.873. Inference: half the data per T4 concurrently, length-sorted batches, bitsandbytes INT8 + fp16 compute; 4k
  context.

### Key discussion threads and notebooks

| Thread | Votes | Takeaway |
|---|---|---|
| Chris Deotte, "16th Place - So Close to Gold Medal" https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527596 | 205 | Most-voted post. LoRA tuning recipe: pick modules (all), pick head LR, sweep alpha at r=16, then sweep r with alpha fixed (rank <16 poor; r=1024 +0.002). Public-notebook ladder: TTA 0.941 -> 0.926, r=64/a=16 0.913, all modules 0.903, +33k 0.899, max 2048 0.895, r=1024 0.894, infer 3072 0.893, 2 Gemma TTA 0.891, extra heads 0.890, **truncate left 0.885**. |
| Chris Deotte, "Starter Notebook 34B LLM" /discussion/521294 | 179 | How to fit a 34B model in Kaggle's limits; links his KDD Cup 2024 work. |
| Darek Kleczek, "External data - additional 157k human preference ratings" https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/499756 | 106 | UltraFeedback converted to the comp format (GPT-4 as judge, not humans). Used for pretraining by 1st and 5th, pseudo-labels by 4th. |
| Abdullah Meda, "Additional 21k Labelled Conversations" /discussion/500973 | 90 | The "33k" LMSYS arena set (~21k after dedup) used by every top team. |
| Yurnero, "We found a leak" https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/524079 | 99 | Leak confirmed by probing (1.030 vs 1.098). |
| Lisa Dunlap (host), "Follow-up on the leak: new test set" https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/524374 | 34 | Root cause = public Arena analysis notebook with battle metadata + labels; private set replaced. |
| Sohier Dane, "Leak next steps interim update" https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/524356 | 38 | Don't select leak-based submissions. |
| RihanPiggy, "Cheat Issue" /discussion/524381; Sercan Yesiloz, "Suspicious Surge of Novice Accounts" /discussion/523394 | 67 / 47 | Community alarm before the leak was explained. |
| Yixiao Yuan, "The magic behind Gemma2" /discussion/522188 | 47 | Gemma-2-9B became the default backbone once released (Gemma 2 release: /discussion/515292). |
| Valentin Werner, "Data Annotation Intuition - why the labels are noisy" /discussion/502449 | 34 | Human votes are noisy; log-loss floor is high. |
| Marilia Prata, "Bradley-Terry model. Elo ratings" /discussion/499803; takaito, "Ranking of models estimated from train data" /discussion/499822 | 42 / 35 | Model identity predicts wins (in train only). |

### Shake-up

Driven by the leak, not by overfitting: the public top 4 were leak-affected (#1 Leak User -> 174, #2 Artem Volgin ->
1,135, #3 billxbf -> 143, #4 "We did the right thing" -> 141). Among no-leak teams, ranks were stable: public 6/5/8 ->
private 1/2/3. Largest rise in the private top 20: Shelton public 104 -> private 18.

### Transferable lessons

1. **Distil big into small when inference is capped.** 70B/72B teachers trained off-Kaggle -> 9B student with KL on
   logits -> LoRA averaging -> 8-bit. (/discussion/527629)
2. **Initialise from the closest pretrained task.** Reward models (FsfairX-Gemma2-RM, ArmoRM, RLHFlow) or a reward
   pretraining stage on UltraFeedback beat chat/instruct starts by large margins. (/discussion/527766, /discussion/527669, /discussion/527629)
3. **Symmetry is free data.** A/B swap as training augmentation (both orders in one step) and as TTA. (/discussion/527685, /discussion/527766)
4. **Pseudo-label at scale.** 240k-500k self-generated pairs soft-labelled by the ensemble. (/discussion/527685, /discussion/527766)
5. **Truncation policy matters with long chats** -- keep the end of the conversation (truncate left) or budget tokens per
   field. (/discussion/527596, /discussion/529067)
6. **Engineering for T4s is a skill**: INT8/GPTQ, length sorting, split across 2 GPUs, padding-free kernels. (/discussion/527685, /discussion/527669)

### UNVERIFIED

- 1st place training hardware (8x A100 80GB) comes from a secondary source (Nebius blog), not the Kaggle post.
- Exact size of the replacement private test set was not found on the pages read.
- 9th (Ebi) and 13th writeups not read.

---

## 5. WSDM Cup - Multilingual Chatbot Arena (2024-25)

### Facts

- **URL:** https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena (id 86946, forumId 6102659). Selected as
  the **WSDM Cup 2025**; top teams invited to present at WSDM (travel at own cost).
- **Dates:** 2024-11-18 -> final submission **2025-02-03**; then a **forecasting phase** -- selected notebooks re-run once
  on newly collected Arena battles; competition end 2025-03-10 (API deadline).
- **Task:** LMSYS sequel, now **multilingual** and **binary**: given one prompt and two responses, predict `model_a` or
  `model_b` (ties removed). Single-turn ("a judge provides one prompt").
- **Data:** `train.parquet` with `id, prompt, response_a, response_b, winner, model_a, model_b, language` (model names and
  language in train only). Training-phase test ~10,000 historical rows; forecasting-phase test "at most 25,000 rows"
  collected after the deadline. 1st place counts the competition train set as **48k** rows. Hosts said Arena "may release
  additional data during the model training phase".
- **Metric:** **accuracy** (categorisation accuracy). No probability calibration, unlike LMSYS log loss.
- **Prizes:** $50,000 -- 1st $12k, 2nd $10k, 3rd $10k, 4th $10k, 5th $8k.
- **Teams:** 950 teams, 1,150 competitors (890 on the final private LB returned by the API).
- **Limits:** CPU/GPU notebook **<= 4.75 h in the training phase, 12 h in the forecasting phase** (API
  maxGpuRuntimeMinutes = 720), internet off. 3rd place budgeted 265 min / 690 min for the two phases.
- **No leak reported.** A "Concerns about the possibility of manipulation" thread (/discussion/551090) flagged that the
  forecast data comes from a public Arena that competitors could vote in -- outcome not verified here.

### Top placements (private = forecasting-phase LB)

| Rank | Team | Private (accuracy) | Approach (one line) | Writeup |
|---|---|---|---|---|
| 1 | whitefebruary (Michael Pieler, Max) | 0.716092 | LMSYS-1st distillation + LMSYS-2nd unpadded inference on **Qwen2.5-14B-Instruct**; full fine-tune; 5x Qwen2.5-72B teachers; 50k synthetic hard multilingual prompts | https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902 |
| 2 | zhudong1949 (solo) | 0.716047 | Start from tascj's LMSYS checkpoints (gemma2-9b, ArmoRM-Llama3-8B), pseudo-label 95k vLLM-generated multilingual pairs, English-then-multilingual epochs | https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567948 |
| 3 | PLaMo 1000000B (RihanPiggy, solo) | 0.714967 | Qwen2.5-14B (post-pretrained) + Phi-4 as **CausalLM scoring the "A"/"B" token** under vLLM; self-distillation; auto-round quantisation | https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567584 |
| 4 | HKUST-GZ DSA KiMi Lab (HengweiDAI) | 0.714067 | CoT prompt control, CausalLM with vLLM `allowed_token_ids`, GPTQ 8-bit (from post's table of contents) | https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/568522 |
| 5 | Just doing (Myo Min Htet, Chenyu Lin, Albert Wang, +2) | 0.712627 | Qwen2.5-14B-Instruct; middle-truncation of long prompts (per secondary summary) | https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567856 (also /discussion/567724) |
| 6 | Oh Your Mercy Never Fails (Zhecheng Li et al.) | 0.712582 | Pseudo-labels from LMSYS-3rd data + API responses; distil Llama-3.3-70B + Qwen2.5-72B into Gemma-2-9B and Qwen2.5-14B | https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567600 |
| 7 | Team Turing (Trushant Kalyanpur, Raja Biswas) | 0.712222 | **Uncertainty cascade**: gemma-2-9b on all; 35% most uncertain to Qwen2.5-14B; top 15% to Qwen2.5-32B or Mistral-Small-24B | https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567589 |
| 8 | sayoulala (LMSYS winner) | 0.712042 | no writeup in API | -- |
| 10 | tascj (LMSYS 2nd) | 0.711817 | no writeup in API | -- |

Gold = top 11. Margins are tiny: ranks 1-2 differ by 0.000045 accuracy; 1st-10th span 0.0043.

### 1st place -- whitefebruary (Michael Pieler, Max)

Source: https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902. Code:
https://github.com/maxreciprocate/kaggle-lmarena-1st-place, inference `orangeaugust/lmsys2`.

- **Recipe = LMSYS 1st (distillation) + LMSYS 2nd (inference engine)**, base **Qwen2.5-14B-Instruct**.
- **Setup:** `ForSequenceClassification` with **2 classes**; concat prompt + both responses, **proportional truncation
  from the middle** of each field; both A/B orders in one gradient step.
- **Stage 1, pretraining:** student Qwen2.5-14B-Instruct and teacher Qwen2.5-72B-Instruct on `mlabonne/orpo-dpo-mix-40k`
  (40k) + `opencsg/UltraFeedback-chinese` (50k). **Full-parameter** fine-tuning, DeepSpeed ZeRO-3; 50 warm-up steps,
  linear decay to 10% of peak lr **4e-6**; grad-clip threshold raised to 64.
- **Stage 2, teachers:** 5 folds of Qwen2.5-72B on `lmarena-ai/arena-human-preference-55k` (57k) + competition data (48k);
  after dropping ties, multi-turn and duplicate prompts/responses -> 80k. Teachers see **model names** before each
  response.
- **Stage 3, distillation:** out-of-fold soft labels; **manually reviewed the samples where teachers disagreed most with
  humans and corrected mislabels**. Extra soft-labelled data: arena sets with ties and first turns (106k),
  `lmsys/chatbot_arena_conversations` (33k), `lmarena-ai/PPE-Human-Preference-V1` (16k; suspected to correlate with the LB,
  included "fearing others would"), Llama-3-70b and gpt-4o-mini battles (1k each), nbroad's open-model sets v1-v3 (25k),
  and **their own 50k synthetic set**: lmsys-chat-1m prompts filtered to Arena-Hard complexity >= 3, 50% non-English,
  8 MT-Bench categories matched to the comp, responses from Llama-3.1-405B-Instruct-FP8, Llama-3.3-70B, Qwen2.5-72B,
  Athene-V2-Chat via vLLM. Loss = weighted **CE (original or hard label) + KL + cosine** on soft labels. Final model =
  linear merge of a full-data model and one without the last two datasets.
- **Inference:** tascj's padding-free sequence concatenation ported to Qwen2.5. Pass 1: merged model on all rows. Pass 2:
  A/B-swapped, full-data model on the **33% most uncertain** rows.
- **Did not work:** a GPT-4o critique model (20k critiques filtered by agreement with humans) to add CoT -- soft labels
  with critiques were repeatedly worse; Qwen2.5-72B AWQ under vLLM for pass 2 (only 5% of rows fit); activation-similarity
  **layer pruning** of gemma-2-27b / Qwen2.5-14B (no better than gemma-2-9b); forcing order-invariant embeddings.

### 2nd place -- zhudong1949 (solo)

Source: https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567948. Code:
https://github.com/zhudongwork/wsdm-cub-2nd-place-solution.git, inference `zhudong1949/lmsys-0201`.

- Framework: tascj's LMSYS training/inference code (full fine-tuning under limited GPUs).
- Prompt / response A / response B **truncated in the middle in proportion to length** (`MIDV2ProcessorPAB`).
- **Initialised from tascj's LMSYS-trained gemma2-9b and ArmoRM-Llama3-8B**, 3-class head replaced with 2-class.
- **Submission 1:** fine-tune these + gemma2-27b on WSDM 80/20; soft-label nbroad's v1 (8.5k) + v2 (13k) with the 3-model
  average -> `hf-21k` (v3 dropped: response quality too close, noisy); fine-tune on hf-21k + WSDM; TTA gemma on PAB,
  llama on PBA, weights 3.3:1 -> 0.709 public / 0.708 private.
- **Submission 2 (the winner):** 74k more pairs: lmsys-chat-1m prompts deduped, split English (424k) vs other (103k),
  sampled with **rising non-English share** (v1 10k = 7k/3k, v2 27k = 21k/6k, v3 38k = 36k/2k); responses from 13 models
  via vLLM (Qwen2.5-72B, Llama-3.3-70B, gemma-2-27b, phi-4, DeepSeek-R1-Distill-Qwen-32B/14B with think sections removed,
  Mistral-7B-v0.3, internlm, Hermes-3, Llama-3.2-1B/3B ...). Pseudo-labelled 3.3:1; dropped rows with
  `|p_a - p_b| <= 0.03`. Re-pretrain on 95k pseudo-labels, then 2 epochs on full WSDM data: **epoch 1 English subset, epoch
  2 multilingual subset**. Inference: gemma on all (PAB), llama only where `|p_a - p_b| < 0.8` (PBA), 2.5:1 -> 0.709
  public / **0.716 private**.
- **Did not work:** Qwen2.5-72B distillation (time/GPU), Skywork reward models as pseudo-labellers or backbones, Skywork
  preference 80K data, Bradley-Terry loss, multi-loss weighting.

### 3rd place -- PLaMo 1000000B (RihanPiggy / LIHANG-HONG, solo)

Source: https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567584. Code:
https://github.com/LIHANG-HONG/WSDM-Cup-3rd-place-solution.

- **CausalLM, not SequenceClassification**, re-implementing the Eedi 1st-place rerank pipeline, so **vLLM** can serve it;
  custom `SFTTrainer.compute_loss` on the logits of tokens "A" and "B" only. Chat template asks "which response is best,
  answer only the letter", with a `<Language>` field from **fastText** language ID. Each of prompt / A / B capped at 2048
  tokens (input ~6144).
- **Models:** Qwen2.5-14B-Instruct (with post-pretrain on UltraFeedback-binarized + tascj team's collected data) + Phi-4
  (no post-pretrain); seed-merged. Train data: LMSYS 55k + LMSYS 33k + WSDM. Teachers Qwen2.5-72B and Athene-V2-Chat via
  DeepSpeed.
- **CV (fold 0, accuracy):** Qwen2.5-14B lr 1e-4 = 0.7051; **lr 1.4e-4 collapses to 0.508** (loss spikes); Phi-4 ~0.697-0.702.
  Distillation: T=5 and soft-loss weight 0.9 best (0.7135); more teachers 0.7169. **Self-distillation from the 14B's own
  logits matched 72B distillation** -- "maybe it is the effect of label cleaning brought by soft logit that really matters".
  KL alone enough (KL + cosine no clear gain).
- **Quantisation:** auto-round, seqlen 6144, nsamples 512, iters 3000 (sweep table in post).
- **Inference engineering:** budget 265 min (train phase) / 690 min (forecast); T4x2 throughput ~70k tokens/min (~74k with
  vLLM prefix cache for TTA); sort by length; TTA with Qwen-14B on 25% of rows; spend the remaining time on Phi-4 (~60% of
  rows), with **Phi-4 weight by prompt-length quantile** (0.5 below median, 0.15 for q2-q3) because Phi-4 is good on short
  prompts only.
- **Gave up:** a Qwen2.5-3B CoT model describing the user's task (+0.001-0.002 but timeouts).

### 7th place -- Team Turing (short, a clean cascade diagram)

https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567589: score all rows with fine-tuned
gemma-2-9b-it; uncertainty `1 - |p - 0.5|`; top 15% most uncertain re-scored by Qwen2.5-32B-Instruct (sub 1) or
Mistral-Small-24B-Instruct-2501 (sub 2) with weight 1.5; next 35% by Qwen2.5-14B-Instruct with weight 1.0; vLLM.

### Key discussion threads and notebooks

| Thread | Votes | Takeaway |
|---|---|---|
| sayoulala, "LSMSY 1st Solution and Code" https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/554766 | 93 | Re-opened his LMSYS code for fairness; the LMSYS winning recipe became everyone's baseline. |
| Sinan Calisir, "Top Solutions From the Previous Competition (LMSYS)" /discussion/547480 | 31 | Index of LMSYS writeups. |
| Nicholas Broad, "8.5k Open Model Samples" /discussion/552166, "13k more" /discussion/556059, "final dump from GIANT open models" /discussion/557746 | 54 / 29 / 20 | Multilingual open-model response sets; used by 1st and 2nd (2nd skipped v3 as too noisy). |
| hengck23, "[lb 0.682] speedup gemma2 training: HF+FA2 versus unsloth+FA2" /discussion/554660 | 28 | Training-speed comparison. |
| lhwcv, "CV vs LB thread" /discussion/552368; Cody_Null, "Stunning results from EDA and OOF" /discussion/551674 | 31 / 29 | Accuracy LB was noisy; CV trust debates. |
| Benedikt Droste, "Concerns about the possibility of manipulation" /discussion/551090 | 16 | Forecast data comes from a live public Arena. |
| hengck23, "survey on public ranking vs GPU resource" /discussion/561080 | 14 | Compute disparity discussion. |

### Shake-up

The forecasting phase used new battles, so the final LB is a true out-of-time test. The API returns **0.000000 for every
public-LB score** (public ranks listed but scores wiped), so public-vs-private movement cannot be computed from the API.
2nd place's own numbers show the effect: submission 1 public 0.709 -> private 0.708, submission 2 public 0.709 -> private
0.716 (/discussion/567948).

### Transferable lessons

1. **A previous competition's open-sourced winners are your starting checkpoint.** 1st and 2nd built directly on LMSYS
   1st/2nd code and weights. (/discussion/569902, /discussion/567948, /discussion/554766)
2. **Soft labels clean noisy human labels.** Self-distillation matched 72B distillation; 1st place also hand-corrected
   high-disagreement rows. (/discussion/567584, /discussion/569902)
3. **Match the target distribution when synthesising data** -- language share, difficulty (Arena-Hard score), MT-Bench
   categories; 2nd place scheduled English then multilingual epochs. (/discussion/569902, /discussion/567948)
4. **Spend extra compute only where the model is unsure** -- 33% TTA (1st), `|p_a - p_b| < 0.8` (2nd), 15%/35% cascade
   (7th). (/discussion/569902, /discussion/567948, /discussion/567589)
5. **CausalLM + vLLM can beat SequenceClassification on throughput** under a 4.75 h T4 budget. (/discussion/567584)
6. **When the test is collected after the deadline, choose robustness over public-LB fit** (merged seeds, conservative
   weights). (/discussion/567584)

### UNVERIFIED

- Public-LB scores (API returns zeros) and therefore shake-up magnitudes.
- 4th and 5th place details are taken from post tables of contents and a secondary Japanese summary
  (https://zenn.dev/dalab/articles/d0e6938bf50db1), not read in full.
- Size of the final forecasting test set actually used.

---

## 6. The Learning Agency Lab - PII Data Detection (2024) -- short, LLM-adjacent

Included because the winning ingredient was **LLM-generated synthetic training essays with Faker-injected PII**; the
models themselves were encoders (DeBERTa-v3-large).

### Facts

- **URL:** https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data (id 66653, forumId 4310355).
  Hosts: Vanderbilt University + The Learning Agency Lab.
- **Dates:** 2024-01-17 -> final submission 2024-04-23 (entry / merger 2024-04-16).
- **Task:** token classification (NER) of **7 PII types** in student essays from a MOOC: `NAME_STUDENT` (students only,
  not instructors or cited authors), `EMAIL`, `USERNAME`, `ID_NUM`, `PHONE_NUM`, `URL_PERSONAL`, `STREET_ADDRESS`, in **BIO**
  format (so 15 labels incl. `O`). Original PII had already been replaced by **surrogates** of the same type.
- **Data:** ~22,000 essays answering a single assignment prompt; **70% reserved for test**; JSON with `full_text`, spaCy
  `tokens`, `trailing_whitespace`, `labels`. Train = 6,807 essays, of which only 945 contain PII (moth, /discussion/469493).
- **Metric:** micro **F-beta with beta = 5** (recall weighted 5x precision) over positive (`document, token, label`)
  predictions; `O` must not be submitted. **Metric fix (2024-03-12):** duplicate predictions for the same document-token
  pair are now all dropped before scoring (/discussion/483538).
- **Public / private:** public = 25%.
- **Prizes:** $60,000 -- Leaderboard 1st $13k, 2nd $10k, 3rd $5k; **Efficiency (CPU-only)** 1st $15k, 2nd $12k, 3rd $5k,
  score `F5 / (Benchmark - maxF5) + RuntimeSeconds / 32400` (same form as DAIGT).
- **Teams:** 2,048 teams, 2,510 competitors, 55,847 submissions. **Limits:** CPU/GPU <= 9 h, internet off.
- **Quirk:** the public LB contained **no PHONE_NUM** at all (only 4 of 6,807 train essays had one) (/discussion/480067).

### Top placements (private LB)

| Rank | Team | Private F5 | Public rank | Approach (one line) | Writeup |
|---|---|---|---|---|---|
| 1 | Fold Zero (Patrick Chan, Gaurav Rawat, YoungPaik, Geremie Yeo, Yevhenii Maslov) | 0.96988 | 14 | 10 DeBERTa-v3-large variants (multi-sample dropout, BiLSTM head, knowledge distillation, name-swap aug), Optuna-weighted voting, rule post-processing | https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374 |
| 2 | dott (solo) | 0.96970 | 7 | Bag of 6 DeBERTa-v3-large on words (is_split_into_words), 7 classes without B/I, scaled-down `O` probability | https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497352 |
| 3 | Rai (solo) | 0.96950 | 28 | single DeBERTa-v3-large pretrained on mpware data then fine-tuned (secondary source) | /competitions/pii-detection-removal-from-educational-data/writeups/rai-3rd-place-solution |
| 4 | Luca Massaron, Rashmi B, steubk, AK, Priyanshu Kumar | 0.96929 | 17 | 10 DeBERTa-v3-large with focal loss + BiLSTM/GRU head; data generated with **Llama-3-70B-Instruct** | https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497367 |
| 5 | R & T & M (Ryota, takai380, min fuka) | 0.96891 | **1** | not read in full | https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497306 |
| 9 | Nicholas Broad | 0.96755 | 159 | not read | https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497177 |

Gold = top 14. Public #2 DAIG -> private 26, #3 -> 50; 25th place called itself "THE SHAKING AGENCY LAB" (public 9).

### 1st place -- Fold Zero (writeup by Geremie Yeo)

Source: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374. Code:
https://github.com/bogoconic1/pii-detection-1st-place (mirror https://github.com/ympaik87/pii-detection-1st-place),
inference `yeoyunsianggeremie/pii-1st-place-solution`.

- **Data:** nbroad's Mixtral-generated set "by far the most useful", mpware's Mixtral-8x7B essays, plus their own ~2k
  generated essays.
- **Training:** 4 folds (`document % 4`) then 1-2 full fits; **13 labels**; max length 1600-2048; 3-4 epochs; lr 1e-5; no
  downsampling of PII-free essays; **`o_weight = 0.05`** (down-weight the `O` class).
- **Diversity:** multi-sample-dropout model (smoother training); BiLSTM-before-classifier model (NaN loss fixed by careful
  init and `save_safetensors=False`); **knowledge distillation** from teachers trained on other datasets with the same
  folds (+0.005-0.01 CV and LB); first/last-name swap augmentation.
- **Post-processing ("the key"):** per-label thresholds; drop NAME_STUDENT not title-cased or with digit/underscore;
  propagate a predicted name to every occurrence in the document; PHONE_NUM with >=9 digits -> ID_NUM; repair `\n` in
  STREET_ADDRESS; B-USERNAME/-/B-USERNAME -> I-USERNAME; drop ID_NUM <4 or >25 chars, URL_PERSONAL <10 chars, EMAIL
  without `@`; span repair; regex fallback.
- **Ensemble:** weighted voting over 7 groups / 10 models, group weights and voting threshold tuned with **Optuna on OOF**.

### 2nd place -- dott (solo)

Source: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497352 (inference
`dott1718/piid-2nd-place-solution`).

- Tokenise the given word list with `is_split_into_words=True`, average sub-token probabilities per word, ignore
  whitespace tokens. Drop B-/I- prefixes -> **7 classes** (spans are always separated, so BIO is recoverable exactly).
- nbroad's generated set mixed in at **sample weight 0.5**.
- Bag of **6 deberta-v3-large** full-data models; max length 512 / 1024 / 2048, stride 32, batch 8, lr 2e-5, 4 epochs, cosine,
  1 warm-up epoch, AdamW. Augmentations did not help the final sub.
- Post-processing: propagate NAME_STUDENT within a document but relabel 1-char and non-title-case strings; handle the
  rare `"\n"` that only appears inside STREET_ADDRESS. Decision rule: argmax after multiplying `O` probability by a
  factor tuned on public LB, consistently **0.02-0.03** -- the F5 metric rewards recall.

### 4th place -- "Llama3 70B is all you need" (Rashmi B, Luca Massaron, steubk, AK, Priyanshu Kumar)

Source: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497367.

- **Synthetic data recipe:** generate personas with Mistral-7B-Instruct-v0.2 / Gemini Pro 1.0 / GPT-3.5-turbo; add a
  sample of each "tool" type (Learning Launch, Visualization, Mind Mapping, Story Telling) to the prompt; inject PII with
  **Faker**; generate ~4,600 essays with **Meta-Llama-3-70B-Instruct** and Mixtral-8x7B-Instruct via vLLM. "A single model
  trained with dataset generated from Llama3 70B Instruct model is better than our best ensemble." Also paraphrased
  false-positive non-PII names with `dipper-paraphraser-xxl`.
- Models: 2x5-fold DeBERTa-v3-large, **focal loss**, BiLSTM + GRU head, 3 epochs, lr 1e-5/2e-5, max len 1280 train, **4000
  with stride 1024 at inference** (<7 h). Whitespace strings -> a new `[SPACE]` token; Unicode normalisation. CV:
  5-fold stratified on PII yes/no (`document % 4` had high fold variance).

### Key threads

| Thread | Votes | Takeaway |
|---|---|---|
| Raja Biswas, "Impact of External Datasets" https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/473139 | 169 | Each shared synthetic set (pjmathematician 2k, moth 4.4k, nbroad 2.3k) gave a strong boost; external data would decide the comp. |
| Nicholas Broad, "2355 essays generated by Mixtral 8x7b" https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/472221 | 134 | Persona-conditioned essays + Faker PII; LB 0.854 -> 0.888; the dataset every top-4 team used. |
| moth, "+4400 external generated texts" https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/469493 | 107 | Only 945 of 6,807 train essays had PII. |
| Raja Biswas, "Truncation of Input Sequence" /discussion/473011 | 108 | Long essays need stride/long max length. |
| Damien Mourot, "There is no PHONE_NUM in the public LB dataset" https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/480067 | 37 | LB probing found a whole class missing from public. |
| inversion (host), "Important: Metric Fix" https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/483538 | 29 | Duplicate (document, token) predictions could game the metric; now removed. |

### Transferable lessons

1. **An LLM is a data factory for a small model**: persona-prompted essays + programmatic PII labels; the stronger the
   generator (Llama-3-70B), the better the encoder. (/discussion/497367, /discussion/472221)
2. **Bias the decision toward the metric**: F5 rewards recall, so down-weight or scale `O` (0.05 weight, x0.02-0.03).
   (/discussion/497374, /discussion/497352)
3. **Post-processing from OOF error analysis is worth a model.** (/discussion/497374)
4. **Probe what the public LB does not contain** before trusting it. (/discussion/480067)

### UNVERIFIED

- 3rd place (Rai) details come only from a secondary Japanese summary (https://zenn.dev/sinchir0/articles/396967387196dc);
  the Kaggle post was not fetched.
- Efficiency-prize winners were not checked.
- 5th place (497306) was fetched but not summarised.


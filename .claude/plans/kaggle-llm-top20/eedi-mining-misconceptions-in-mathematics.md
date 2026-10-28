# Eedi - Mining Misconceptions in Mathematics: top-20 writeup synthesis

- Slug: `eedi-mining-misconceptions-in-mathematics` (competition id 82695, forum 5528027). Dates 2024-09-12 to 2024-12-12.
- Task recap: for each (math MCQ, wrong answer) pair, rank ~2.5k misconceptions; MAP@25; public LB = 28% of test; 9 h notebook on 2x T4, internet off.
- Existing deep coverage: `.claude/plans/kaggle-llm-writeup-bank-part2.md`, H2 "## 4. Eedi — Mining Misconceptions in Mathematics (2024)" (ranks 1-3 in depth; 4-5 listed but "not read"). This file adds ranks 4-20 and does not repeat ranks 1-3.
- Sources: leaderboard via `competitions.LeaderboardService/GetLeaderboard`; writeups via `discussions.DiscussionsService/GetForumTopicById` (team `writeUpForumTopicId`). Raw JSON in `/private/tmp/claude-501/kaggle-top20/eedi-mining-misconceptions-in-mathematics/` and `/private/tmp/claude-501/kaggle-bank-p2/md/`.

## Top-20 table (from the leaderboard API, private order)

Public column = public rank (public score). Rank 21 included because it is the first team below the cut. No ties in the top 21.

| Private rank | Team | Members (Kaggle usernames) | Private | Public rank (score) | Writeup | Approach |
|---|---|---|---|---|---|---|
| 1 | MTH 101 | conjuring92 | 0.63881 | 6 (0.64379) | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution | Qwen2.5-14B retriever, 14B then 32B pointwise rerankers, 72B listwise top-5; Claude-generated grouped synthetic MCQs; distilled CoT (bank) |
| 2 | CQYR | copasta, wangqihanginthesky, yokuyama, honglihang | 0.60480 | 8 (0.62947) | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/cqyr-2nd-place-solution | Linq-Embed-Mistral + 2x Qwen2.5-14B retrievers; sliding-window listwise rerank with Qwen2.5-14B, then Qwen2.5-72B + Llama-3.3-70B (bank) |
| 3 | Waseda Pochi | threerabbits | 0.60248 | 1 (0.67081) | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/waseda-pochi-3rd-place-solution-with-magic-boost | Public Qwen-14B embedders + Qwen-32B-AWQ reranker (6 LoRAs) + LB-probed unseen-misconception boost (bank) |
| 4 | takoi & charmq & kami | takoihiraokazu, kami634, charmq | 0.60044 | 3 (0.65568) | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/takoi-charmq-kami-4th-place-solution | Qwen2.5-14B + 32B LoRA retrievers fed LLM-written misconception guesses; top 25 overall + top 15 unseen; 3 Qwen2.5-32B-GPTQ pointwise rerankers (fold-LoRA averaging); seen-misconception scores x0.40 |
| 5 | Ebi & ktr | kcotton21, ebinan92 | 0.59765 | 4 (0.65374) | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/ebi-ktr-5th-place-solution | stella_en_1.5B bi-encoder + Qwen2.5-32B listwise reranker choosing among 52 single-letter options (2 windows = top 104); Gemini-1.5-pro synthetic; Qwen-32B CoT as input; SubjectId GroupKFold |
| 6 | Nikita Babich | nikitababich | 0.58998 | 5 (0.65040) | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/nikita-babich-6th-place-solution | 4 retrievers (gte-Qwen2-7B, SFR-Embedding-2_R, bge-multilingual-gemma2, Qwen2.5-14B) + Qwen2.5-32B-AWQ 50-option reranker with self-distillation, rephrased-misconception TTA and a 2-item permutation second stage |
| 7 | terekaerumasahmet | tereka, aerdem4, kaerunantoka, irrohas | 0.58693 | 2 (0.66270) | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/terekaerumasahmet-private-7th-public-2nd-place-sol | Rank-voting ensemble of 3 retriever+reranker pipelines (Qwen2.5-32B-AWQ: 40-option MC, yes/no pointwise, 9-option shuffled MC with TTA); iterative unseen-misconception boost |
| 8 | Yannan Chen | yannan90 | 0.58160 | 13 (0.61350) | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/yannan-chen-8th-place-solution | 4x Qwen2.5-14B retrievers (weighted sum) -> 40 -> single Qwen2.5-32B yes-logit reranker, wise-FT LoRA merge (alpha 0.8), AWQ + vLLM prefix caching; ~6 h runtime |
| 9 | Study Hard | zuoyouzuo, zhenshitainanle, hinsay, lumoumou, gjp2master | 0.57114 | 7 (0.64171) | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/study-hard-private-9th-public-7th-place-solution | GPT-4o synthetic data filtered by the best retriever; Qwen2.5-32B-AWQ 50-token rationales; Qwen2.5-14B retriever; Qwen2.5-32B causal-LM yes-logit reranker |
| 10 | sayoulala | sayoulala | 0.56616 | 19 (0.59813) | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/sayoulala-10st-place-solution-summary | 7k GPT-4o synthetic rows; Qwen2.5-14B + Qwen2.5-Coder-14B retrievers (5-fold LoRA merged); Qwen2.5-32B-AWQ 25-option multiple-choice reranker |
| 11 | H&Y&L | huangxiaojian, yuchengzhi, linbeo | 0.56496 | 9 (0.62309) | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/h-y-l-private-11th-public-9th-place-solution-summa | FlagEmbedding pipeline with 3-stage curriculum (external set 1 -> MalAlgoQA -> train) for Qwen2.5-32B retriever and reranker; cross-device negatives; rerank top 18 |
| 12 | chizhu | chizhu2018 | 0.56429 | 18 (0.60023) | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/chizhu-12th-place-solution | Concatenated embeddings of 3 Qwen2.5-14B retrievers (one with Qwen-32B error guesses, one public); 3 rerankers (Qwen2.5-32B-AWQ, 2x Qwen2.5-14B) on random 25-of-top-50 subsets |
| 13 | qwenqwenqwenqwenqwenqwenqwenqw | rashmibanthia, benbla, nbroad, abdullahmeda | 0.56133 | 26 (0.58401) | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/qwenqwenqwenqwenqwenqwenqwenqw-13th-place-solution | Qwen2.5-Math-7B solution text -> Qwen2.5-14B retriever top 100 -> Qwen2.5-32B listwise over 100 -> Qwen2.5-72B listwise over 25 (fp16 layer-by-layer) |
| 14 | eeee (rsi) | andreygalichin, pipmos, rolf110, danyaivanov | 0.55874 | 16 (0.60220) | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/eeee-rsi-14th-place-solution | Qwen2.5-14B retriever with per-epoch hard-negative refresh and false-negative masking; Qwen2.5-32B pointwise reranker sees the whole candidate list, reuses KV cache, new classification head; 13.9k GPT-4o synthetic |
| 15 | iws | tkyiws | 0.55392 | 10 (0.62060) | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/iws-15th-place-solution | Progressive hard-negative mining SFR -> Qwen2.5-14B -> Qwen2.5-32B retrievers; 2 Qwen2.5-32B-bnb-4bit rerankers trained on 2-25 random options; overlap-averaged ensemble |
| 16 | mathminds | andy2709, thanhhau097a, hoanganhpham, locbaop | 0.55306 | 17 (0.60143) | no writeup found | no public method found (see below) |
| 17 | B.Z.Z.Z.L. | pingfan, xuanmingzhang777, xiaoqinglong1996, lizhecheng, tonyarobertson | 0.55085 | 11 (0.61427) | no writeup found | no public method found (see below) |
| 18 | Ilya Novoselskiy | ilya9711nov | 0.54993 | 23 (0.59492) | no writeup found | no public method found (see below) |
| 19 | LRKBY | lihaoweicvch, xyzdivergence, yeoyunsianggeremie, evgeniimaslov2, roschildrui | 0.54805 | 12 (0.61422) | no writeup found | no public method found (see below) |
| 20 | JIN | jinssaa | 0.54603 | 29 (0.57508) | no writeup found | no public method found (see below) |
| 21 | Sohaib | sohaibahmed9920 | 0.54565 | 22 (0.59687) | no writeup found | no public method found (see below) |

## Per-team synthesis, ranks 4 to 20

Ranks 1-3: the bank section is complete for method; nothing material to add from the writeups beyond what it lists.

### 4th - takoi & charmq & kami (private 0.60044, public 3rd)
Source: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/takoi-charmq-kami-4th-place-solution (topic 551559). Code: https://github.com/TakoiHirokazu/kaggle-Eedi-4th-solution , https://github.com/charmq00/kaggle_eedi_public ; inference https://www.kaggle.com/code/charmq/eedi-pp040-ret15-exp345-341-348-multi-32b-ret-c015 . Slides (Rist): https://speakerdeck.com/rist/4th-place-solution-eedi-mining-misconceptions-in-mathematics
- **Framing:** most test misconceptions were expected to be unseen in train, so every stage was tuned for unseen labels.
- **CV:** GroupKFold on QuestionId, 5 folds, and a separate score on the validation rows whose misconception is absent from train.
- **Synthetic data:** Qwen2.5-72B-Instruct-AWQ wrote a full MCQ (construct, subject, question, 4 options, correct, incorrect) for each misconception never seen in train, with **100 random train questions as few-shot examples** (~20k-token prompt); "it was crucial to include a large number of examples". ~8,000 questions generated; 2,500 sampled for reranker training (time budget).
- **Misconception generation:** untuned Qwen2.5-32B-Instruct writes a guessed misconception for each (question, wrong answer) with 2 shots; appended to the retriever input.
- **Retrievers:** Qwen2.5-14B-Instruct (LoRA r=32, alpha=64, MNRL, 1 positive : 47 negatives per pair, batch 6; MAP@25 0.511, recall@25 0.923) and Qwen2.5-32B-Instruct-GPTQ-Int4 (1:255 negatives, 10 epochs, ~10 h/fold on an A100; MAP@25 0.554, recall@25 0.926; ~50 min/fold inference so only 2 folds used). Embeddings concatenated; vLLM modified to compute embeddings. Negatives drawn only from misconceptions that appear as positives in train. Candidates: top 25 overall plus top 15 among unseen misconceptions, deduplicated.
- **Rerankers:** three Qwen2.5-32B-Instruct-GPTQ-Int4 LoRA pointwise models; Models 1/2 average the LoRA weights of folds 1+2 and 3+4 (1:9 negatives; fold MAP@25 0.653, unseen-only 0.601); Model 3 adds the 2,500 synthetic rows (1:19 negatives, 2 epochs, ~10 h/fold on 4x A100; 0.664 / 0.605). Model 3 lifted private by ~0.01. Restricting negatives to train-positive misconceptions lifted a single reranker by ~0.025 on public and private (author's comment).
- **Post-processing:** multiply scores of misconceptions seen in train by **0.40** (best public among tried coefficients).
- **Distinguishing idea:** explicit unseen-label handling at three points (unseen-only candidate slots, synthetic data only for unseen labels, seen-label down-weighting).

### 5th - Ebi & ktr (private 0.59765, public 4th)
Source: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/ebi-ktr-5th-place-solution (topic 551391). Code: https://github.com/ebinan92/Eedi-5th-solution
- **CV:** switched from GroupKFold(QuestionId) to **GroupKFold(SubjectId), 5 folds**, which put more validation-only misconceptions in each fold and brought CV close to LB (e.g. CV 0.626 / LB 0.633).
- **Synthetic data:** one generated problem per misconception without train examples, by **gemini-1.5-pro** with 4 shots chosen as problems whose misconception names are closest under stella_en_1.5B_v5 embeddings (+~0.01 CV vs random shots).
- **Knowledge distillation:** Qwen2.5-32B-Instruct explains (<=200 words) why the correct answer is right and what misconception produced the wrong one; this text is input to every later model (+0.04 bi-encoder CV, +~0.01 reranker CV).
- **Retriever:** dunzhang/stella_en_1.5B_v5, SentenceTransformers, CachedMNRL, LoRA r=32/alpha=64, 1 epoch; negatives mined with range_min 512 (relaxed, because hard negatives hurt); CV 0.41 / LB 0.39. Weak, but recall is rescued by reranking a deep top-104.
- **Listwise reranker (core):** idea from kcotton21 based on single-token logits (arXiv 2406.15657). Qwen2.5-32B-Instruct sees 52 candidates labelled A-Z, a-z (ordered by the retriever) plus the option "NA", and outputs one token; probabilities over the 52 letters sort the list. Trained on 4 non-overlapping windows of the top 208 (1-52, 53-104, ...); plain full-vocabulary CE (restricting to candidate tokens made little difference); LoRA r=24, alpha=48, lr 5e-5 cosine, 10 warmup steps, 1 epoch, global batch 8, Unsloth; ~7 h per model on a rented A100 (~$350 total). Inference: 2 windows (top 104), GPTQ 4-bit + vLLM, ~120 min per run, ensembled over 3 folds (the third fold only partly made it in, blamed on Kaggle runtime delays, discussion/550743).
- **Scores (CV / public / private):** fold 4, 52x2 windows 0.626 / 0.633 / 0.581; folds 3+4 - / 0.650 / 0.590; folds 2(60%)+3+4 - / 0.653 / 0.597.
- **Did not work:** other single-token label sets (letter pairs, letters+digits, Japanese hiragana); QwQ-32B-Preview; two-step 7B (104) -> 32B (52) reranking; few-shot examples in the reranking prompt; a third window (CV only).

### 6th - Nikita Babich (solo, private 0.58998, public 5th)
Source: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/nikita-babich-6th-place-solution (topic 551565).
- **CV:** one fold, but **seen and unseen misconceptions weighted equally**; argues the usual QuestionId split (~20% unseen) rewards overfitting to seen labels (0.65 x 0.8 + 0.3 x 0.2 = 0.58 example).
- **Synthetic data:** Qwen2.5-72B-Instruct few-shot (32B struggled on tricky misconceptions); 3 shots from the most similar train misconceptions; full test-like MCQ with one relevant distractor; 3 questions x 3 seeds = 9 per misconception; 3 h on 2x A40. LoRA pretraining for generation hurt reasoning, so few-shot only (consistent with arXiv 2404.02124).
- **Text cleaning:** LaTeX converted to plain text with `pylatexenc`; two ensemble members kept raw LaTeX.
- **Retrievers:** gte-Qwen2-7B-instruct, SFR-Embedding-2_R, bge-multilingual-gemma2, and the public Qwen2.5-14B (anhvth226 notebook); LoRA r=64/alpha=128, batch 32, 5 negatives from the top 30, two rounds of negative mining (1,600 then 400 steps), bnb 4-bit; ~9 h per model on one RTX 3090. Retriever ensembling mattered (author's comment on 5th place).
- **Reranker (Qwen2.5-32B-Instruct, LoRA r=64/alpha=128, batch 2 x 6 accum, AWQ, ~6 h on an A40):** completion-only loss (`DataCollatorForCompletionOnlyLM`); **excluding all test-only misconceptions from training** gave the biggest gain (~0.61-0.63 public); 49 negatives sampled from the top 100 with the positive at a random position (fights retriever order bias).
- **Self-distillation:** train with seed A, infer logits on a seed-B dataset (new negatives and positions), retrain with KL + CE: ~+0.01 public. Rationale: several misconceptions can be valid, soft labels regularise.
- **Misconception rephrasing:** Qwen2.5-32B rewrites each misconception into "missing knowledge + student belief"; logits from original and rephrased lists combined (~+0.01 public).
- **Second stage:** Qwen keeps the retriever's order when unsure (arXiv 2306.17563), so the top-2 are rescored in both orders and averaged; window 2 was optimal.
- **Did not work:** fp16 rerankers with all LoRAs then quantised; Qwen2.5-72B reranker (same as 32B); synthetic data in reranker training. ~$300 of rented A40/A100.

### 7th - terekaerumasahmet (tereka, aerdem4, kaerunantoka, irrohas; private 0.58693, public 2nd)
Source: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/terekaerumasahmet-private-7th-public-2nd-place-sol (topic 551388); Masaya's part: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551430 .
- **Structure:** rank-voting ensemble (sum of 1/rank across pipelines) of three retriever + reranker pipelines, all rerankers Qwen2.5-32B served with vLLM (AWQ 4-bit, prefix caching).
- **Pipeline 1 (Masaya):** a pretrained Qwen (writeup says "qwen35B") first answers the question; SFR-Embedding-2_R retriever with 100 hard negatives, removing negatives unseen in train; unknown misconceptions filled with GPT-4o-mini data; 60 candidates; reranker Qwen2.5-32B-AWQ + LoRA as multiple choice over 40 options labelled 0-9, A-Z, a-z (original data only), run twice ([:40] and [20:]) and merged by logprobs. Masaya-part detail: generated wrong-answer reasoning as retriever context (inspired by arXiv 2204.07496); SFR-Embedding-2_R in bnb-4bit, LoRA r=4/alpha=8, 4 negatives, query 128 / passage 64 tokens, 15 epochs, lr 2e-5 (Qwen2.5, gte-Qwen2, bge-en-icl did not beat it); per the post, "Remove unseen example in train" left CV unchanged but improved LB much; reranker Qwen2.5-32B-Instruct-AWQ, LoRA r=8/alpha=16, lr 1e-4, 2 epochs, trained as last-token classification on OOF retriever candidates; merge keeps top 20 from the first window and top 5 from the second (idea by Ahmet); 210 min for both windows.
- **Pipeline 2 (tereka + kaeru):** SFR-Embedding-2_R with 100 hard negatives on original + Gemma-2-27B / Qwen2.5-32B synthetic data (LB 0.378) averaged with sayoulala's public model; 70 candidates filtered by pipeline 1; reranker Qwen2.5-32B-AWQ + LoRA as yes/no pointwise; single fold.
- **Pipeline 3 (Ahmet):** Qwen2.5-32B multiple choice over each pipeline's top 9, trained with shuffled option order and run with order TTA, so order-independent.
- **Post-processing:** boost misconceptions unseen in train, iteratively more, unless they become rank 1 for some pair (assumption: they are in the mapping for a reason).
- **Did not work:** concatenating or averaging many retriever vectors; full-data reranker; QwQ.
- **Data note:** this team is the one linked to the "private dataset accidentally shared" thread (bank, discussion/550619); the writeup does not mention using it.

### 8th - Yannan Chen (solo, private 0.58160, public 13th)
Source: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/yannan-chen-8th-place-solution (topic 551412). Inference https://www.kaggle.com/code/yannan90/eedi-recall-rerank-lb-0-601-pb-0-584 ; training https://github.com/yannan90/kaggle-2024-Eedi-8th-place-training-code
- Only 14B and 32B models; ~6 h runtime. Built on sayoulala's recipe (discussion/543519) and Chris Deotte's LMSYS quantisation / wise-FT / vLLM ideas.
- **Synthetic:** 983 ChatGPT few-shot rows (3-4 random train examples) to cover missing misconceptions, regenerated per fold for misconceptions unique to the validation fold.
- **Retriever:** Qwen2.5-14B, bnb 4-bit + LoRA r=32/alpha=64 on all projections, train_group_size 4, effective batch 64, 7 epochs, cosine; 5-fold; first 4-5 epochs without synthetic data, last 2-3 with it; **manual curriculum**: lr lowered 5e-4 -> 1e-5 and negative pool shrunk 80 -> 4 when validation stalls. Iterative hard-negative mining "very helpful". 4 fold models (picked by LB) ensembled by weighted sum; each fits a T4 in 4-bit, no merge.
- **Reranker:** Qwen2.5-32B, bnb 4-bit + LoRA r=32/alpha=64 incl. lm_head, "Yes" logit per (query, misconception), group size 4, batch 64, 4 epochs, lr 5e-4, full data, negatives = top 40 from the retriever ensemble. Merged with **wise-FT alpha 0.8**, AWQ (128 calibration rows), vLLM with prefix caching, 40 candidates per query.
- **Public LB ladder:** synthetic data 0.3x -> 0.49; retriever ensemble -> 0.51; reranker -> 0.57; wise-FT -> 0.59.
- **Gotcha:** AutoAWQ did not export the lm_head/embedding weights for Qwen2.5-0.5B (random letters at inference); fixed by copying the missing keys from the unquantised safetensors.

### 9th - Study Hard (private 0.57114, public 7th)
Source: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/study-hard-private-9th-public-7th-place-solution (topic 551420). Inference https://www.kaggle.com/code/hinsay/kfold-recall-and-reranker
- **Synthetic:** GPT-4o writes subject, construct, question, correct and incorrect answers for a given misconception; for misconceptions in train, rows are kept only if the best retriever ranks the target in its top 5-10 (filtering); unseen ones kept unfiltered.
- **Rationale:** Qwen2.5-32B-Instruct-AWQ (vLLM) writes a <=50-token rationale for the wrong answer.
- **Retriever:** Qwen2.5-14B-Instruct, contrastive (sayoulala recipe, code adapted from the KDD Cup 2024 OAG 1st-place repo), hard negatives from SFR, upsample misconceptions with <3 train rows. Without synthetic data the 14B scored 0.447 LB.
- **Rerankers:** 14B binary classifier (dropped for speed) and **Qwen2.5-32B causal LM using the "yes" token logit**, then AWQ (inference after Deotte's 34B vLLM notebook). Public / private: retriever 0.547 / 0.491; binary reranker 0.606 / 0.542; causal-LM reranker 0.641 / 0.571. Reranker training began only a week before the deadline.

### 10th - sayoulala (solo, private 0.56616, public 19th)
Source: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/sayoulala-10st-place-solution-summary (topic 551722). Author of the widely used recipe thread discussion/543519.
- 7k GPT-4o external rows. Retrievers Qwen2.5-14B-Instruct and Qwen2.5-Coder-14B-Instruct: LoRA pretrain on the 7k rows, then fine-tune on train; 5 folds merged by LoRA fusion; three retrievers combined (14B 5-fold, Coder-14B 5-fold, full-data 14B).
- Reranker: Qwen2.5-32B-AWQ with LoRA, **multiple choice over the top 25** with CE only over the 25 option tokens; final = 5-fold LoRA fusion trained on 7k + train ("LoRA fusion remains effective in any scenario").
- Did not work (offline): distilling Qwen-72B outputs; not submitted.

### 11th - H&Y&L (private 0.56496, public 9th)
Source: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/h-y-l-private-11th-public-9th-place-solution-summa (topic 551424). Code https://github.com/MilchstraB/Eedi
- FlagEmbedding retrieve + rerank, both Qwen2.5-32B; rerank only the top 18 (time).
- **External data curriculum** instead of synthetic generation: ext1 (discussion/533764) then ext2 = MalAlgoQA (discussion/541222, wrong-answer rationale treated as the misconception) then train; mixing ext1 with train was worse than staged training. ext1 relabelled by a Qwen2.5-7B-Math retriever trained on it. External data gave +0.02-0.03 at retrieval.
- Reranker trained in the same 3 stages with hard negatives from the matching stage's retriever.
- Cross-device negatives in the last retrieval stage: public 0.567 -> 0.574. Public: retrieval 0.582, + reranker 0.623.

### 12th - chizhu (solo, private 0.56429, public 18th)
Source: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/chizhu-12th-place-solution (topic 551429).
- **Synthetic:** 4 batches, >10,000 rows from a CoT prompt (`<thought>` then `<output>`) that turns a misconception into a 4-option MCQ; recall 0.430 -> 0.470 (single fold).
- **Retriever:** SFR top-100 negatives, then Qwen2.5-14B contrastive; embeddings concatenated from (a) 14B with Qwen2.5-32B-AWQ-inferred error text, (b) 14B on the raw query, (c) the public anhvth226 14B; each 5-fold with averaged LoRA weights. 0.526 public / 0.485 private.
- **Rerankers:** 25-class prediction on 25 random candidates drawn from the top 50 (OOF); Qwen2.5-32B-AWQ (vLLM, public 0.570), Qwen2.5-14B raw (0.567), Qwen2.5-14B + reason (0.558); weighted: 0.597 / 0.564.
- **Did not work:** Qwen2.5-Coder; LightGBM/CatBoost post-reranker on OOF; last-token-pool 0/1 reranker.

### 13th - qwenqwenqwen... (rashmibanthia, benbla, nbroad, abdullahmeda; private 0.56133, public 26th)
Source: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/qwenqwenqwenqwenqwenqwenqwenqw-13th-place-solution (topic 551673).
- CV: 4-fold GroupKFold by MisconceptionId.
- Qwen2.5-Math-7B-Instruct (base, vLLM fp16) solves the question; that reasoning is input to a Qwen2.5-14B retriever (FlagEmbedding, sayoulala code, SFR-mined hard negatives; 0.486 public / 0.433 private) that returns top 100.
- **Two listwise rerankers:** Qwen2.5-32B-Instruct sees all 100 candidates and emits one letter id (logprobs give the top 25); Qwen2.5-72B-Instruct reranks those 25 the same way. 72B trained with LLaMA-Factory LoRA + DeepSpeed ZeRO-3; 32B with HF SFTTrainer. 32B and 72B run in **fp16 one layer at a time** across both GPUs; 14B in bnb. 500 min total. Final 0.578 public / 0.561 private.
- **Did not work:** ensembling 72B models; step-by-step solutions in the reranker context; several synthetic-data variants; 32B retriever (0.470 / 0.453); AWQ 14B retriever (needs vLLM >= 0.6.4).

### 14th - eeee (rsi) (andreygalichin, pipmos, rolf110, danyaivanov; private 0.55874, public 16th)
Source: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/eeee-rsi-14th-place-solution (topic 551492). Weights: https://www.kaggle.com/models/andreygalichin/qwen2.5-32b-r32-a16-synth-merge-smooth-full/
- **Synthetic:** 13,921 GPT-4o rows so each misconception has >=7 examples; reference example from the nearest misconception (SFR embeddings), prompt with a verify-and-modify step and "the student must answer without the example" (prompt design after arXiv 2403.04706); ~500 exact duplicates removed.
- **Validation:** 20% holdout split into OOD (no shared MisconceptionId) and ID (no shared QuestionId); OOD tracked the LB.
- **Retriever:** Qwen2.5-14B, QLoRA r=32/alpha=64, InfoNCE, 8 negatives from the top 100 refreshed each epoch, query token masking 0.1, lr 5e-5, 3 epochs, in-batch negatives with **same-misconception false negatives masked to -inf**. Recall@25 0.88, MAP@25 0.42 OOD; 0.438 public / 0.434 private.
- **Reranker:** Qwen2.5-32B pointwise binary classifier that also sees a shuffled list of ~30 candidates; question + list encoded once and `past_key_values` reused for batches of 5 candidates; new classification head on the logits; trained in H2O LLM Studio, QLoRA r=16/alpha=32, lr 2e-5 (head 1e-5), 1 epoch, label smoothing 0.05, positives oversampled 4x. 0.571 OOD / 0.589 public / 0.558 private. No vLLM; ~7 h.
- **Did not work:** GPT-4o-mini rewrites of positives for oversampling; other answer options in the prompt; a tournament-style 10-then-5 listwise scheme (worked on 14B, could not port to 32B in time). Team's own post-mortem: scoring one misconception per call cost far more time than listwise.

### 15th - iws (tkyiws, solo, private 0.55392, public 10th)
Source: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/iws-15th-place-solution (topic 551501). Inference https://www.kaggle.com/code/tkyiws/sub-embedder-reranker-ensemble-v3?scriptVersionId=212088687
- GPT-4o synthetic from 10 sample questions: 985 x 2 rows for never-seen misconceptions and 847 for once-seen ones.
- **Retriever:** progressive mining: SFR-Embedding-2_R top 200 -> train Qwen2.5-14B with 30 random negatives from that 200 (batch 1) -> new top 200 -> same with Qwen2.5-32B. Random rather than hardest negatives because the job is recall@25, not top-1. Two retrievers: 0.513 / 0.455 and 0.515 / 0.450 (public / private).
- **Reranker:** Qwen2.5-32B-Instruct-bnb-4bit instruction-tuned to pick the right answer from 2-25 randomly sampled options (positive injected if absent). Two variants 0.610 / 0.538 and 0.596 / 0.536.
- **Ensemble:** average scores for IDs both pipelines returned, keep single-pipeline IDs as is: +0.02. Final 0.617 / 0.553.

### 16th-20th - no writeup on the leaderboard
- 16 mathminds (andy2709, thanhhau097a, hoanganhpham, locbaop), 17 B.Z.Z.Z.L. (pingfan, xuanmingzhang777, xiaoqinglong1996, lizhecheng, tonyarobertson), 18 Ilya Novoselskiy, 19 LRKBY (lihaoweicvch, xyzdivergence, yeoyunsianggeremie, evgeniimaslov2, roschildrui), 20 JIN (jinssaa): no `writeUpForumTopicId` on the leaderboard and no title matching their rank in the forum topic list (198 topics scanned, sorted by votes); a web search found nothing. Method unknown.
- Nearest neighbours that did post: 26th (discussion/551406, public 14th), 28th retriever part (551455), 30th public-notebook-only (551409), 31st (553317). Not summarised here.

## Technique frequency matrix

Y = used, N = not used / not mentioned as used, ? = unknown, P = partial (only some stages). Ranks 16-20 have no writeup, so all ?. Ranks 1-3 from the bank section; N in a writeup that is silent on a point means "not reported", not proven absent.

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Decoder LLM (Qwen2.5 7-32B) as retriever/embedder | Y | Y | Y | Y | N | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | ? | ? | ? | ? | ? |
| Qwen2.5-32B in the final reranking stage | Y | N | ? | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | ? | ? | ? | ? | ? |
| 70B+ model in final inference (72B / Llama-3.3-70B) | Y | Y | N | N | N | N | N | N | N | N | N | N | Y | N | N | ? | ? | ? | ? | ? |
| LLM-generated synthetic MCQs | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | N | Y | N | Y | Y | ? | ? | ? | ? | ? |
| Frontier API model for synthetic data (Claude/GPT-4o/Gemini) | Y | N | Y | N | Y | N | Y | Y | Y | Y | N | Y | N | Y | Y | ? | ? | ? | ? | ? |
| Real external data (non-synthetic) | N | Y | N | N | N | N | N | N | N | N | Y | N | N | N | N | ? | ? | ? | ? | ? |
| LLM-written reasoning/error text as model input | Y | Y | N | Y | Y | N | Y | N | Y | N | N | Y | Y | N | N | ? | ? | ? | ? | ? |
| Misconception text rewritten/enriched by an LLM | N | Y | N | N | N | Y | Y | N | N | N | N | N | N | N | N | ? | ? | ? | ? | ? |
| Pointwise (yes/no) reranker | Y | N | ? | Y | N | N | Y | Y | Y | N | ? | N | N | Y | N | ? | ? | ? | ? | ? |
| Listwise / multiple-choice reranker (letters as single tokens) | Y | Y | ? | N | Y | Y | Y | N | N | Y | N | Y | Y | N | Y | ? | ? | ? | ? | ? |
| Order-permutation / reversed-order TTA in reranking | N | Y | Y | N | N | Y | Y | N | N | N | N | N | N | N | N | ? | ? | ? | ? | ? |
| Distillation (teacher pseudo-labels or self-distillation) | Y | ? | ? | ? | ? | Y | N | N | N | Y | N | N | N | N | N | ? | ? | ? | ? | ? |
| LoRA fine-tuning (vs full FT) | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | ? | ? | ? | ? | ? |
| 4-bit quantisation for inference (AWQ/GPTQ/auto-round/bnb) | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | ? | Y | N | Y | Y | ? | ? | ? | ? | ? |
| vLLM serving | Y | Y | ? | Y | Y | N | Y | Y | Y | N | ? | ? | P | N | N | ? | ? | ? | ? | ? |
| Fold-LoRA weight averaging / wise-FT merge | N | ? | ? | Y | N | N | N | Y | N | Y | N | Y | N | N | N | ? | ? | ? | ? | ? |
| Explicit unseen-misconception handling (boost, unseen slots, down-weight seen) | N | N | Y | Y | N | N | Y | N | N | N | N | N | N | N | N | ? | ? | ? | ? | ? |
| CV split designed to include unseen misconceptions | Y | ? | ? | N | Y | Y | N | N | N | N | ? | N | Y | Y | N | ? | ? | ? | ? | ? |
| Ensemble of >=3 retrievers | N | Y | Y | N | N | Y | Y | Y | N | Y | N | Y | N | N | N | ? | ? | ? | ? | ? |

Counted summary (ranks 1-15, the 15 teams with writeups; ? counted as not used):

- Decoder LLM (Qwen2.5 7-32B) as retriever/embedder: 14 of 15 teams with writeups.
- Qwen2.5-32B in the final reranking stage: 13 of 15 teams with writeups.
- 70B+ model in final inference (72B / Llama-3.3-70B): 3 of 15 teams with writeups.
- LLM-generated synthetic MCQs: 13 of 15 teams with writeups.
- Frontier API model for synthetic data (Claude/GPT-4o/Gemini): 10 of 15 teams with writeups.
- Real external data (non-synthetic): 2 of 15 teams with writeups.
- LLM-written reasoning/error text as model input: 8 of 15 teams with writeups.
- Misconception text rewritten/enriched by an LLM: 3 of 15 teams with writeups.
- Pointwise (yes/no) reranker: 6 of 15 teams with writeups.
- Listwise / multiple-choice reranker (letters as single tokens): 9 of 15 teams with writeups.
- Order-permutation / reversed-order TTA in reranking: 4 of 15 teams with writeups.
- Distillation (teacher pseudo-labels or self-distillation): 3 of 15 teams with writeups.
- LoRA fine-tuning (vs full FT): 15 of 15 teams with writeups.
- 4-bit quantisation for inference (AWQ/GPTQ/auto-round/bnb): 13 of 15 teams with writeups.
- vLLM serving: 8 of 15 teams with writeups.
- Fold-LoRA weight averaging / wise-FT merge: 4 of 15 teams with writeups.
- Explicit unseen-misconception handling (boost, unseen slots, down-weight seen): 3 of 15 teams with writeups.
- CV split designed to include unseen misconceptions: 5 of 15 teams with writeups.
- Ensemble of >=3 retrievers: 7 of 15 teams with writeups.

## Consensus vs divergence

**Table stakes (nearly every top-15 writeup):**
- Two stages: a fine-tuned embedding retriever (most often Qwen2.5-14B with LoRA, trained with MNRL/InfoNCE on sayoulala's FlagEmbedding-style recipe, discussion/543519) then a Qwen2.5-32B LoRA reranker, 4-bit quantised, usually served with vLLM. 14 of 15 used a decoder LLM as the embedder; 13 of 15 had a 32B reranker.
- Synthetic MCQs for misconceptions with no training rows (13 of 15). The pool has ~2.5k misconceptions, most never or once seen in train, so without synthetic coverage the retriever cannot place them.
- Hard-negative mining with an off-the-shelf embedder (SFR-Embedding-2_R appears in 7th, 9th, 12th, 13th, 15th).

**What separated 1-5 from 6-20:**
- Bigger or more varied rerank stacks: both 1st and 2nd ran 70B-class listwise rerankers; 4th used three 32B rerankers incl. one on synthetic data; 5th reranked a deep top-104 with 52-letter listwise prompts.
- Every top-5 team did something explicit about unseen misconceptions or the generalisation axis: CV grouped by ConstructId (1st) or SubjectId (5th), seen-label down-weighting and unseen-only candidate slots (4th), LB-probed unseen boost (3rd), misconception enrichment (2nd). Below 5th, most teams used QuestionId splits and no unseen handling (exceptions: 6th equal-weight CV, 7th boost, 14th OOD split).
- Synthetic data built with care for confusable labels (1st clusters; 5th nearest-misconception shots; 4th 100-shot prompts) vs plain few-shot generation lower down.

**Contradictory findings:**
- Hard negatives: helped 8th ("iterative hard negative mining very helpful", discussion/551412) and 14th (per-epoch refresh, 551492); 5th found hard negatives "not effective" and used relaxed ones (551391); 1st reported iterative hard-negative mining did not work (bank); 15th deliberately used random negatives from the top 200 (551501).
- Synthetic data in the reranker: helped 4th (+0.01 private with 2,500 rows, 551559) and 1st (bank); did not help 6th ("did not work", 551565) or 3rd (bank, "added little").
- 72B reranker: decisive for 1st and 2nd (bank) and used by 13th (551673); 6th got "the same results as for 32B" (551565).
- QwQ-32B-Preview: did not work for 1st (bank), 5th (551391), 7th (551388); 2nd found QwQ a good retriever (0.500 private single, bank).
- Hard vs soft unseen correction: 4th down-weights seen labels x0.40 (551559); 7th iteratively boosts unseen (551388); 6th instead removes test-only misconceptions from reranker training and relies on generalisation (551565).
- CoT in the reranker: 1st gained +0.019 from CoT (bank); 13th found inserting step-by-step solutions in the reranker context did not work (551673); 12th's 14B-with-reason reranker was its weakest member (0.558 vs 0.567 public, 551429).

## Aggregated "what did not work"
- Iterative hard-negative mining (1st, bank); hard negatives in general (5th).
- QwQ-32B-Preview as reranker (1st bank, 5th, 7th "QwQ model").
- Qwen2.5-72B reranker over 32B (6th); ensembling several 72B models (13th).
- Synthetic data in reranker training (6th); GPT-4o-mini synthetic for reranker "added little" (3rd, bank); several synthetic variants (13th).
- Rewriting positive misconceptions with GPT-4o-mini for oversampling (14th); misconception rephrasing for recall (7th Masaya part: recall "almost same"). Contrast: 6th found rephrasing useful as TTA.
- Other answer options in the prompt (14th); few-shot examples in the listwise reranking prompt (5th); step-by-step solution in reranker context (13th).
- Multi-step reranking 7B -> 32B (5th); a third 52-option window (5th, CV only).
- Alternative single-token label sets (letter pairs, letters + digits, hiragana) (5th).
- Concatenating or averaging many retriever vectors (7th). Contrast: 4th and 12th concatenated embeddings successfully.
- Full-data-trained reranker (7th).
- fp16 rerankers with all LoRAs, quantised afterwards (6th).
- Qwen2.5-Coder (12th). Contrast: 10th used a Coder-14B retriever.
- LightGBM/CatBoost post-reranker on OOF (12th); last-token-pool 0/1 reranker (12th).
- 32B retriever over 14B (13th: 0.453 vs 0.433 private, kept but not best); AWQ 14B retriever (13th, needs newer vLLM).
- Distilling 72B outputs (10th, offline only).
- LoRA-merge continuous fine-tuning recipe (1st, bank).
- LoRA pretraining of the generator for synthetic data (6th: hurt reasoning).

## Shake-up note
Public was 28% of test. Largest moves inside the top 20 (public -> private): 13th from 26th (+13), 20th from 29th (+9), 10th from 19th (+9), 8th from 13th, 12th from 18th (+6); down: 7th from 2nd (-5), 15th from 10th, 19th from 12th (-7), 17th from 11th (-6). The public 1st (Waseda Pochi, LB-probed unseen boost) fell to 3rd. 6th place credited the host's balanced seen/unseen split across public and private for a "very moderate shuffle" (551565), and 2nd place (bank) trusted CV over a public score that made its 72B look worse. Teams that tuned post-processing coefficients on public (4th's 0.40, chosen on public) still held their place, so the shake-up was mild overall.

## UNVERIFIED
- Ranks 16-20 methods: no writeup found.
- 7th place pipeline 1 says retrieval used a "pretrained qwen35B" to generate the test answer; no 35B Qwen2.5 exists, likely Qwen2.5-32B (Masaya part says "Qwen2.5-32B"). Option-token sets differ between the team post (0-9, A-Z, a-z, "40 charactors") and Masaya's part ([1-9], [a-z], [A-Z]).
- 3rd place reranker format (pointwise vs listwise) and 2nd/3rd CV scheme: not stated in the bank; marked ? in the matrix.
- 11th place reranker format (FlagEmbedding reranker implies pointwise cross-encoder style) and quantisation: not stated.
- 14th place reranker quantisation at inference: not stated (no vLLM).
- 6th place vLLM use: not stated.
- Matrix "N" cells for silent writeups mean "not reported".

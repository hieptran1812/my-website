# MAP: Charting Student Math Misunderstandings (2025), top-20 addendum

- Slug: `map-charting-student-math-misunderstandings` (competition id 104383). Metric MAP@3 over `Category:Misconception`; 15 shared QuestionIds in train and test; 2x T4, 9 h, internet off.
- Already covered in depth: `.claude/plans/kaggle-llm-writeup-bank-part2.md`, H2 "## 5. MAP: Charting Student Math Misunderstandings (2025)" (lines 270 to 339): facts, ranks 1 to 3 in depth, one-liners for 4 to 6, 18th (Deotte pyramid) and 22nd. This file does not repeat that detail; it adds ranks 4 to 21 and the cross-team synthesis.
- Sources for this file: leaderboard API (ranks, scores, public ranks), the writeups at the URLs below (read in full for ranks 4 to 10, 12, 13, 15, 18).

## 1. Top-20 table

Ranks and scores from the leaderboard API, in API order. Ties: ranks 9/10 (0.94801), 11/12 (0.94797) and 20/21 (0.94748) share private scores; rank 21 is included because it ties rank 20. Public column = public rank (public score).

| Rank | Team | Members (Kaggle ids) | Private | Public rank (score) | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | tascj | tascj0 | 0.94919 | 14 (0.95204) | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/1st-place-solution | Suffix classification over per-question candidates with FlexAttention shared prefix; full-FT Qwen3-32B + GLM-Z1-32B, 3 seeds; W8A8 + layer-wise inference |
| 2 | looking for remote job | pingfan, leehann | 0.94890 | 2 (0.95373) | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/map2025-private-and-public-2nd | 80k LLM-written explanations for rare labels + soft labels from 4 teachers; Qwen3-14B letter generation + soft CE |
| 3 | monsaraida & Masaya | irrohas, monsaraida | 0.94866 | 1 (0.95443) | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/3rd-place-solution | Qwen3-14B multi-task classifier (R-Drop, AWP, EMA) + Qwen2.5 14B/32B/72B CausalLM on the question's label set; low-confidence re-inference |
| 4 | THLUO | ilovearsenal | 0.94835 | 13 (0.95211) | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/4th-place-solution-single-model-pb-0-948-lb-0-951 | Pointwise Yes/No reranker (Qwen3-Reranker-8B) over 4 to 6 candidates per question, few-shot explanations per candidate in prompt, distilled from Qwen2.5-Math-72B; + Qwen3-32B-AWQ |
| 5 | Naoism | naoism | 0.94830 | 24 (0.95133) | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/5th-place-solution | Diversity ensemble of 6 (Gemma2-9B, Qwen2.5-14B, Qwen3-8B, DeepSeek-Math-7B; 37 and 65 classes, two public-notebook models), picked on public LB; per-question mask |
| 6 | Manan Jhaveri | mananjhaveri | 0.94828 | 22 (0.95145) | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/mananjhaveri-map-6th-place-solution | 37-class QLoRA seq-classification "Qwen-semble" (Qwen3-Embedding-8B, Qwen3-14B, Qwen2.5-14B); GPT-4o synthetic for 10 rarest labels; pseudo-labelled duplicates |
| 7 | S T I T Y | sinchir0, takaito, isakatsuyoshi, takai380, yukitake93 | 0.94820 | 10 (0.95247) | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/private-7th-public-10th-place-solution | Label-token logit CE (vLLM-friendly); ~50k GPT-4.1/GPT-5-mini synthetic rows; Qwen2.5-32B soft-label distillation + label smoothing; Gemma2-9B, Qwen2.5-32B, R1-Distill-32B, Qwen3-8B |
| 8 | Takashi Someya | takashisomeya | 0.94818 | 18 (0.95173) | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/8th-place-solution | Qwen2.5-7B / Qwen2.5-14B-AWQ / Qwen3-8B / Qwen3-14B-AWQ LoRA CausalLM with single-token symbol labels; reconstructed a to d options in the question; ~30k gpt-5/o3 synthetic rows for weak cells; 50 models |
| 9 | kanto-kaggler | takumuwakamatsu, neilus, rikitomo0526 | 0.94801 | 28 (0.95109) | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/9th-place-solution | Qwen2.5-32B-Instruct + DeepSeek-R1-Distill-Qwen-32B, 5 folds each, QLoRA CausalLM with 65 special label tokens; vLLM + AWQ; label-error fixes |
| 10 | Liwen Zheng | zhengliwen | 0.94801 | 17 (0.95190) | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/10th-place-solution | 65-class seq-classification QLoRA, 5 folds x {Qwen3-Reranker-8B, Qwen3-Embedding-8B, Qwen2.5-32B}; focal + smoothed CE with class-weight warmup; 32B served via GPTQ backbone in vLLM embed mode + external head |
| 11 | sayoulala | sayoulala | 0.94797 | 4 (0.95310) | no writeup found | unknown (no writeup) |
| 12 | Eisuke Okuda | eisukeokuda | 0.94797 | 19 (0.95160) | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/12th-place-solution | Five Qwen2.5-14B-Instruct classifiers distilled from Qwen2.5-72B-Instruct and Qwen2.5-Math-72B teachers, EMA; LLM-rubric score used to rebuild CV |
| 13 | iws | tkyiws | 0.94792 | 3 (0.95356) | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/map-competition-13th-place-gold-medal-solution | Qwen3-14B-Base classifier x2 + listwise CausalLM over the question's 4 to 6 targets (Qwen3-32B x2, Qwen2.5-72B x2); model choice via held-out synthetic data |
| 14 | AnnieGo & kyoumonemui | sjtuwangshuo, kyoumonemui | 0.94790 | 20 (0.95154) | no writeup found | unknown (no writeup) |
| 15 | You've Already Won | jjinho, ttyn4519, ky8ori, masasato1999, lizhecheng | 0.94786 | 12 (0.95217) | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/15th-place-solution-writeup | Qwen3-14B + Phi-4 + Phi-4-reasoning LoRA classifiers, 37 classes, question-valid labels only, Neither shared / Correct per-question label scheme; shaken out of gold |
| 16 | Takoi | takoihiraokazu | 0.94775 | 7 (0.95268) | no writeup found | unknown (no writeup) |
| 17 | RAG | titericz, aerdem4, conjuring92 | 0.94767 | 5 (0.95308) | no writeup found | unknown (no writeup) |
| 18 | Chris Deotte | cdeotte | 0.94756 | 9 (0.95259) | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/18th-place-pyramid-ensemble | "Pyramid" cascade: small models on all rows, larger ones on progressively more uncertain subsets; ~1,000 models trained |
| 19 | Hacker | dianliuvu, iamcranberry, lucioxing, jokerjjjjjj | 0.94752 | 70 (0.95021) | no writeup found | unknown (no writeup) |
| 20 | Kelvin | xyzdivergence | 0.94748 | 99 (0.94976) | no writeup found | unknown (no writeup) |
| 21 | D.Imanishi | dimanishi | 0.94748 | 26 (0.95124) | no writeup found | unknown (no writeup) |

Spread: 1st to 21st private is 0.00171. Writeups found: 14 of the 20 top-20 ranks (1 to 10, 12, 13, 15, 18); none for 11, 14, 16, 17, 19, 20 (and the tied 21st).

## 2. Per-team synthesis

### Ranks 1 to 3: facts the bank lacks
- 2nd (HZM + Baiph), from writeup tables: Baiph-side members with public / private / inference time: Qwen2.5-14B-AWQ full data 0.948 / 0.946 / 20 min; Qwen3-14B-AWQ fold1 0.948 / 0.943; Qwen2.5-32B-AWQ fold1 0.948 / 0.944 / 40 min; Qwen2.5-32B-AWQ full data 0.949 / 0.945; QwQ-32B-AWQ with the 80k external data, full data, distilled 0.949 / 0.948; Qwen2.5-32B-AWQ 80k distilled fold1 0.949 / 0.947; Qwen3-14B-AWQ 80k distilled fold1 0.950 / 0.945. Final fusion = four 37-class plus four 65-class models, equal-weight average.
- 3rd (Masaya): ablation without the other choices in the prompt: Qwen2.5-32B-AWQ CV 0.9474 / public 0.945 / private 0.941, 72B-GPTQ 0.946 / 0.940 (versus 0.949 / 0.944 and 0.950 / 0.946 with choices). CV is fold 0 only (GPU-limited). Masaya's cascade pipeline was the team's public-best but scored slightly lower on private than monsaraida's; monsaraida's more complex multi-stage schemes also did not give the best private score. 72B inference used auto-round GPTQ with CPU offload.
- 1st: nothing material missing from the bank.

### 4th: THLUO (solo), single-model story
Source: writeup URL in table. Validation: 5-fold StratifiedKFold, then a single fold "slightly below average" as the holdout, to avoid folds that score abnormally high or low. Applied the known fix for the 12 rows of QuestionId 31778 (MC_Answer 9 labelled True while the correct answer is 6, per discussion/589400) at both train and inference. Reduced 65 labels to 37 (True/False derived from the shared 15 questions), then to the 4 to 6 candidate targets each question shows in train.

Formulation: a **pointwise reranker**, borrowed from the Eedi Mining Misconceptions 1st place: each prompt contains one candidate diagnosis and asks "Does this diagnosis is correct? (Yes/No)". All 4 to 6 candidates of one row_id are put in a single batch and a cross-entropy over the group's Yes logits optimises within-row ranking (so effective batch size is 4 to 6). Mixing several row_ids per batch with averaged group losses did not help; he suspects groups of different length give losses of different scale.

Backbones (multi-fold ordering): Qwen3-32B >= Qwen3-Reranker-8B >= QwQ-32B >= Qwen3-14B >= Qwen2.5-Math-7B-Instruct > Qwen2.5-14B-Instruct > Qwen3-8B > Qwen2.5-7B-Instruct. Iterated on Qwen3-Reranker-8B; teacher for distillation was Qwen2.5-Math-72B-Instruct (limited GPU, few teachers tried).

The distinguishing idea is **prompt_v2: few-shot explanation samples for every candidate label inside the prompt** (1 to 3 random examples per target in training, 6 per target at inference). Progression (CV / public / private): v1 1 epoch 0.9473 / 0.944 / 0.943 (40 min); + distill 0.9480 / 0.947 / 0.945; v2 + distill 0.9493 / 0.949 / 0.946 (4 h inference); + full data 0.950 / 0.947; full data 2 epochs 0.9499 / 0.951 / 0.949 (best single); Qwen3-32B-AWQ v1 + distill 0.9502 / 0.950 / 0.947 (2.5 h). Final = reranker-8B + 32B-AWQ, CV 0.9518, public 0.95211, private 0.94835, 6.5 h total; the ensemble scored below his best single model on private because the 32B was weaker there. Inference with vLLM; the 2-epoch full-data model produced NaNs on T4 after bf16 to fp16 conversion (overflow), so he re-trained a substitute from the 2-epoch validation checkpoint plus the holdout fold.

Did not work: listwise ranker; chain of thought (his biggest time sink, no gain); retrieving the most similar labelled explanations with a fine-tuned Qwen3-Embedding-8B and putting them in the prompt (CV 0.9493 but worse public and private). Not tried: generating extra explanations.

### 5th: Naoism (solo), diversity over a single best model
Source: writeup URL in table. Thesis: the baseline already gets ~94% right; the errors live in rare `True_Misconception:*` rows, `False_Correct:NA` rows and ambiguous explanations, and a single model's "quirks" bias those, so he optimised for ensemble **diversity**.

CV: started with StratifiedKFold on Misconception, switched to **StratifiedGroupKFold grouped by StudentExplanation with spaces and periods removed** because identical explanations inflated CV. Once a single model reached ~0.947 public, CV-LB correlation "almost completely disappeared" (also for ensembles); he chose single models on CV + LB and ensemble membership on public LB only, with near-uniform weights to limit LB overfit, and calls it a gamble that paid off.

Ensemble (weighted average of probabilities), model / classes / method / CV / public / private / weight: Gemma2-9B-it 37 LoRA 0.946 / 0.946 / 0.943 / 0.25; Qwen2.5-14B-Instruct 37 QLoRA with augmentation 0.945 / 0.946 / 0.943 / 0.25; Gemma2-9B-it 65 QLoRA - / 0.947 / 0.942 / 0.25; Qwen2.5-14B-Instruct 37 QLoRA 0.938 / - / - / 0.1; Qwen3-8B 65 full FT (sinchir0's public notebook) - / 0.942 / 0.938 / 0.1; DeepSeek-Math-7B-instruct 65 QLoRA (jaytonde's public notebook) - / 0.944 / 0.940 / 0.25. Mixing 37- and 65-class heads was deliberate. Each model has its own prompt; one lists all 65 labels with short guidelines.

Augmentation (one model only): per question, sample 10% of `False_Misconception:X` rows, rewrite the student's answer to the correct one and relabel `True_Misconception:X`; tiny gain, added diversity. Post-processing: zero the probability of any misconception never seen with that QuestionId in train. Distinguishing idea: two unmodified public-notebook models inside a gold ensemble, chosen for diversity rather than score.

### 6th: Manan Jhaveri (solo), "Qwen-semble"
Source: writeup URL in table; code https://github.com/mananjhaveri/MAP-Charting-Student-Math-Misunderstandings-6th-place-solution ; notebook https://www.kaggle.com/code/mananjhaveri/map-6th-place-solution-qwen-semble-ftw . Treated it purely as 37-label misconception classification (True/False prefix removed). Prompt: Question, Answer, Correct: Yes/No, Explanation, plus a final line "Task: Classify the misconception in the explanation." which improved CV and LB. Max length 160 with dynamic padding. Holdout was a 20% split; submissions retrained on full data.

Data: dropped conflicting duplicates (same answer and explanation, different label), later **pseudo-labelled those duplicates with the ensemble and added them back** (a commenter singled this out as the biggest win). ~50 rule-based augmentations per category (punctuation, words to digits, "plus" to "+", small typos, guided by fuzzy near-duplicate analysis). Synthetic: ChatGPT 4o, 100 to 120 samples for each of the 10 lowest-support categories.

Models: started with GTE and Ettin encoders, then LLMs; `Qwen3-Embedding-8B` with a sequence-classification head was best. 4-bit QLoRA, rank 128, alpha 32 to 128; A5000/A100 on Jarvis Labs. Public LB per member: Qwen3-Embedding-8B dedup 0.946; Qwen3-14B 0.947; Qwen2.5-14B-IT 0.947; Embedding-8B + augmentation 0.945 (weak alone but removing it always hurt the ensemble); Embedding-8B + pseudo-labelled duplicates + synthetic **0.949 best single**; Qwen3-14B same data 0.948; same with another seed 0.943. Ensembling method from kishanvavdara's public notebook. First four models crossed 0.950 public; six gave 0.951 / 0.947; adding the poor-scoring reseeded Qwen3-14B gave his best public and private.

Did not work: many backbones and prompts; ~10k augmented samples (stuck at 0.945, no ensemble value); 65 labels; a two-stage Category then Misconception model; generation over simplified label descriptions; textual-entailment formulation; MoE classification head; focal loss.

### 7th: S T I T Y (5 members; public 10th)
Source: writeup URL in table (discussion/612134). Overview: assign a token to every misconception label and train cross-entropy directly on those **label-token logits** of a CausalLM; classification heads scored similarly but logits allowed fast vLLM inference. True/False added back by rule; only the question's relevant misconception candidates go in the prompt (much shorter); the other MC options are listed too.

Synthetic data (isakatsuyoshi): for every (True/False, QuestionId, Misconception) group with 5+ rows, sample 3 real explanations as references and ask GPT-4.1 (versions 2 to 46) or GPT-5-mini (47 to 56) for a new one; ~200,000 generated, ~50,000 (versions 2 to 13) used in the final solution, no filtering; dataset https://www.kaggle.com/datasets/isakatsuyoshi/map-synthetic-dataset . On Deotte's Gemma2-9B baseline it gave CV +0.003 to +0.004 and LB +0.002 to +0.005, but other models showed mixed results. Label noise: label smoothing 0.1 on some models; Qwen2.5-32B trained on competition data provides soft labels instead of trusting the synthetic labels.

Pipelines: (1) yukitake93: competition + synthetic (v2 to 13), two prompt types (relevant candidates with Claude-Sonnet-written descriptions of each misconception, or the full catalogue); Gemma-2-9B-IT 0.950 / 0.946 and Qwen2.5-32B-AWQ 0.950 / 0.947, ensemble 0.952 / 0.948; targets = 50% original label + 50% Qwen2.5-32B OOF on competition rows, 32B logits only on external rows; Gemma uses smoothing 0.1 + temperature 1.5, Qwen-32B none + T 1.0. (2) takai380: DeepSeek-R1-Distill-Qwen-32B, CE + KL to per-fold OOF soft labels, external rows kept only if OOF top-1 matches their label: 0.950 / 0.945 (competition data only: 0.948 / 0.943). (3) sinchir0: his public full-FT Qwen3-8B SFTTrainer notebook; using existing tokens instead of added special tokens as answers (CV +0.005, LB +0.005); taking a slightly later checkpoint than min eval loss (CV +0.008, LB +0.007); lr 2e-5 (1e-4 failed to converge), 3 epochs, full train: 0.948 / 0.945.

Did not work: per-question dynamic ensemble weights (fold-0 CV only, too little data); adding fast small models.

### 8th: Takashi Someya (solo)
Source: writeup URL in table (discussion/612101). CausalLM chosen because vLLM accelerates it; Qwen2.5 and Qwen3 were the best families he tried. Final members: Qwen2.5-7B, Qwen2.5-14B-Instruct-AWQ, Qwen3-8B, Qwen3-14B-AWQ, all LoRA.

Distinguishing idea: **reconstruct the a to d answer options**. Discussion/589350 showed students pick from 4 options; he recovered the MC_Answer-to-letter mapping from student explanations and rewrote QuestionText as e.g. "Which number is the greatest? (a) 6 (b) 6.2 (c) 6.079 (d) 6.0001", so the model can compare the chosen answer with the distractors. Prompt: role line, Question with options, Answer, Correct: Yes/No, Student Explanation, "Type of Misunderstanding:". LaTeX stripped for readability (no score effect). Labels mapped to 65 single-token symbols (■, □, ▲, ...) following a public Qwen2.5-32B-GPTQ notebook (aleaiest).

Training: 3 epochs, AdamW wd 0.01, cosine with warmup 0.03, batch 12, lr 2e-4, LoRA r 16 / alpha 16 / dropout 0.05 on all 7 projections. Small changes in batch size and lr swung CV a lot, so he grid-searched those two; r and alpha did not matter; LoRA beat full FT in his (admittedly less tuned) runs.

Synthetic data: train a model, compute OOF MAP@3 per (QuestionId, Category, Misconception), and for low-MAP or low-count cells ask gpt-5 and o3 for up to 50 new samples per call using real rows (random, or low-OOF ones) as few-shot; ~30,000 total; example prompts at https://www.kaggle.com/datasets/takashisomeya/map-synthetic-data-o3-example . Gain looked modest, at most ~0.001 LB; no filtering method he tried helped.

Final: 10 settings x 5 folds = 50 models, weights from LB and CV; every member is public 0.949 alone and private 0.944 to 0.946. He submitted the LB-best and the CV-best ensembles; LB-best was better on private (0.951 public, 0.948 private).

### 9th: kanto-kaggler (3 members)
Source: writeup URL in table (discussion/612105), by rikitomo0526. Two 32B CausalLMs, 5 folds each, 10 models, vLLM + AWQ, 460 min inference, public 0.950 / private 0.948. Built on sinchir0's public SFTTrainer notebook: 65 labels as **added special tokens**, QLoRA with `modules_to_save=["lm_head"]` and PEFT `trainable_token_indices` for the label-token embeddings. Qwen2.5-32B-Instruct: 2 epochs, lr 1e-4, batch 8 x grad accum 2, r 32 / alpha 64, dropout 0.01, all 7 projections. DeepSeek-R1-Distill-Qwen-32B: lr 8e-5, r 64 / alpha 128, otherwise the same. Google Colab A100 80GB, 220 min per fold. Folds: StratifiedKFold on QuestionId + Category + Misconception.

Preprocessing: fixed the known annotation errors by row_id (6 rows answer 6 to 9 and 12 rows 9 to 6) and merged the duplicate label `Wrong_Fraction` into `Wrong_fraction`, redistributed by QuestionId at post-processing. A simpler prompt (Question, Answer, Correct, Student Explanation, Label) slightly improved CV and cut time. Inference: logits of the label tokens, plain average across models. Results (valid / public / private): Qwen2.5-32B fold0 0.9502 / 0.946 / 0.944, 5-fold - / 0.950 / 0.946; R1-Distill-32B fold0 0.9480 / 0.948 / 0.945, 5-fold - / 0.950 / 0.946; both 0.950 / 0.948.

Extra detail from rikitomo0526's Zenn post (https://zenn.dev/mkj/articles/1f374627377474, 2026-01-06): switching inference quantisation from bitsandbytes to AWQ kept accuracy and cut inference time to about 1/6; training used DeepSpeed; a 32B x5 + 72B x2 submission scored below 32B x10 on private; soft-label training through SFTTrainer needed `remove_unused_columns=False` and a custom collator.

Did not work: pairwise and pointwise rerankers for top-1/top-2 reordering; 70B/72B (better CV, ~150 min inference too long); Llama-3.3-70B, gpt-oss-20b, Mistral, Hermes, Gemma 3, Qwen3 all below Qwen2.5; soft labels from another 32B (no public gain, +0.001 private); merging 5 fold LoRAs into one (worse than one model); training on Misconception only without the True/False part.

### 10th: Liwen Zheng (solo, first medal)
Source: writeup URL in table (discussion/612038). Plain 65-class sequence classification (`AutoModelForSequenceClassification`, as in Deotte's Gemma2-9B notebook), 15 QLoRA models: 5 folds x {Qwen3-Reranker-8B, Qwen3-Embedding-8B, Qwen2.5-32B-Instruct}. Data hygiene: auto-fix True/False inconsistencies; near-duplicates kept in the same fold.

Loss (distinguishing idea): mix of focal loss (on unsmoothed CE, to emphasise hard rows) and CE with label smoothing 0.1, with **per-class weights phased in by a separate weight warmup over ~33% of steps** to stabilise the long tail. 8B: 4-bit NF4 + gradient checkpointing, single RTX 4090, lr 1e-4, 4 epochs, cosine, effective batch 16. 32B: A800-40GB, lr 1e-4, 3 epochs, effective batch 32 (comment: ~3 h on 2x A100 40GB). LoRA r 64, alpha 128, dropout 0.05, Q/K/V/O + MLP, `score` head saved.

Inference engineering: 8B models keep LoRA unmerged on a 4-bit base. 32B: merge LoRA, split backbone from the classifier head, GPTQ-4bit the backbone (group 64, symmetric, group-aware reordering, true_sequential, ~2k calibration prompts in training format), run vLLM in `task="embed"` with LAST pooling and no normalisation, then apply the FP16 head outside vLLM. This is a way to serve a classifier head through vLLM. Ensemble: weighted **logit** blend, 2 per 32B fold and 1 per 8B fold, more stable than averaging probabilities. Results (public / private): single 8B 0.946 / 0.942; 8B x 5 folds 0.950 / 0.946; 32B x 5 folds 0.950 / 0.947; all 15 0.951 / 0.948.

Did not work: hierarchical multi-task classifier with hard logic constraints; 32B as pointwise reranker; 32B as listwise reranker (top-5 from k-fold recall models, A to E output, best 14B 5-fold 0.948 / 0.945); TTA.

### 11th: sayoulala (solo; public 4th)
No writeup found. No solution post in the competition forum (all 160 listed topics scanned by title and author), no MAP post or comment by any member in the fetched threads, nothing found by web search (GitHub/Zenn/Medium). Public 4th to private 11th, a small slide. sayoulala's only related trace is an Eedi Mining Misconceptions (2024) recall thread (discussion/543519), different competition.

### 12th: Eisuke Okuda (solo, first solo gold)
Source: writeup URL in table (discussion/612114). Final = simple probability average of **five Qwen2.5-14B-Instruct** sequence classifiers distilled from 72B teachers; distillation was the biggest single gain (+0.003 to +0.005). Teachers: Qwen2.5-72B-Instruct and Qwen2.5-Math-72B as `AutoModelForSequenceClassification`, QLoRA r 64 / alpha 128, 4 folds, 2 epochs, lr 1e-4. Students: QLoRA r 64 / alpha 128 (LoRA itself gave no standout gain), custom distillation head, 3 epochs, lr 1e-4, full training data, EMA decay 0.999; the five variants differ only in the distillation loss weight. Ensembling added +0.001 to +0.003 public and helped private too; `torch.compile()` was needed to fit the five merged models in time.

Distinguishing idea: `True_Neither` (third most frequent class) was very hard, so he wrote an additive 5-point LLM rubric scoring an explanation's relevance, absence of misconceptions, partial grasp and reasoning trace (with "subtract all" rules for flawless or misconceived answers). Synthetic data generated with it did not raise CV, but MAP@3 varied across rubric-score groups, so he **used the rubric score to rebuild the CV split**. Distilling into small encoders (ettin-encoder-1b) lost 0.002 to 0.004 against the 14B. Did not work: listing the multiple choices in the prompt; vLLM inference; hard-negative sampling.

### 13th: iws (tkyiws, solo; public 3rd)
Source: writeup URL in table (discussion/612069). Hybrid of 6 models, two per configuration with different seeds, all trained on all 36,696 rows with no validation split: Qwen3-14B-Base classification x2 (QLoRA 4-bit, 2 epochs, 37 classes, scores read only for the question's 4 to 6 targets; private 0.943/0.944, public 0.949/0.948, weight 0.9); Qwen3-32B listwise x2 (private 0.943, public 0.949/0.948, weight 1.0); Qwen2.5-72B-Instruct listwise x2 (private 0.945, public 0.947/0.949, weight 1.0). Ensemble public 0.952, private 0.947.

Listwise model: CausalLM shown the question's 4 to 6 targets as numbered options in randomly shuffled order, outputs the option number; QLoRA 4-bit, 1 epoch; LoRA merged, 72B `intermediate_size` padded 29568 to 29696 so vLLM tensor parallelism works, then GPTQ 4-bit with `gptqmodel`; vLLM logprobs per option. Scores unified (L1-normalised classifier probabilities, softmax of listwise logprobs), weighted average, top 3, True/False prefix added. The 4 answer choices were added to every prompt. Training: 14B on RTX 5090 2 h, 32B RTX 5090 10 h, 72B H200 4 h; inference 8 h on 2x T4.

Distinguishing idea: model selection on **held-out synthetic data** rather than public LB. Public was optimistic and almost any combination scored alike; seeds moved scores by about +/-0.002; accuracy rose 8B to 14B to 32B then plateaued (72B no better); on synthetic data, listwise beat classification, so listwise got more weight. He says relying on public ranking would likely have cost the medal. Post-hoc: Qwen2.5-32B matched Qwen2.5-72B while Qwen3-32B lagged (Qwen3 less suited); private showed less seed variance than public, so architecture diversity might have beaten seed pairs. Did not work: synthetic data for training and CoT (synthetic used only for evaluation, weakly correlated with LB).

### 14th: AnnieGo & kyoumonemui
No writeup found. No solution post in the competition forum (all 160 listed topics scanned by title and author), no MAP post or comment by any member in the fetched threads, nothing found by web search (GitHub/Zenn/Medium).

### 15th: You've Already Won (5 members; shaken down from gold)
Source: writeup URL in table (discussion/612062); code https://github.com/T3pp31/map-charting-student-math-misunderstandings . Four sub-reports.
- lizhecheng and ky8ori (8x A100 80GB): 5-fold LoRA sweeps on Qwen3-14B/8B/4B, Qwen3-4B-Instruct-2507, gemma-2-9b-it, DeepSeek-R1-0528-Qwen3-8B, deepseek-math-7b-instruct, each with 8 settings (rank 32 or 64, lr 1.75e-4 to 2.5e-4, alpha 32, 2 epochs, linear schedule), simplest prompt. Qwen3 best and larger better; gemma-2-9b hard to train (gradient explosions); deepseek-math and Qwen2.5 dropped. Removing True/False prefixes about +0.001. Masked loss over the question's possible labels: CV up, LB flat. Richer prompts (options listed, rewritten question): CV up, LB down, and needed max length 300+. Focal, bootstrapping and label-smoothing losses: no consistent per-fold gain. Soft labels from embedding-based semantic grouping of misconceptions: inconsistent. A binary Correct-vs-Neither swap model for the top-2 positions reached 0.965+ MAP on those rows but lowered LB.
- t fuku (RTX 5090): screened 62 model-size combinations; lr 2e-4, cosine, 3 epochs, LoRA 64/128: Phi-4 CV 0.9483 / public 0.948, Phi-4-reasoning-plus 0.9480 / 0.948, Qwen3-32B-4bit 0.9483 / 0.946, Qwen2.5-32B-4bit 0.9485 / 0.946. A fixed `<think>Let me analyze this mathematical misconception...</think>` prefill in the prompt substantially raised his CV; simple Deotte-style prompts beat listing all options. Regret: over-tuning ensemble weights to public LB; equal weights on phi-4, phi-4-reasoning-plus, Qwen3-32B and Qwen3-14B would have scored 0.948 private (gold).
- M Sato: manual a to d option reconstruction (little effect); masked loss (CV only); QuestionId label filtering at post-processing (effective).
- JJ (2x 4090, later RTX 6000 Pro Blackwell): Qwen3 models trained locally degraded badly on Kaggle inference even with quantisation-matched training, so he used Phi-4; LoRA 64/128, lr 2e-4, 3 epochs. Distinguishing idea: **label sharing**: `Neither:NA` shared across all questions but `Correct:NA` question-specific (e.g. `31772:Correct:NA`), because Neither language is generic and Correct language is question-specific; best CV 0.9496 (inverse scheme 0.949, kept for diversity).
Final: retrained Qwen3-14B on full data plus phi-4 and Phi-4-reasoning, 5 to 6 models at roughly equal weights, only question-valid labels scored; public 0.952. They had 15+ unselected submissions at 0.948+ private and blame weak experiment management and CV construction.

### 16th: Takoi (solo; public 7th)
No writeup found. No solution post in the competition forum (all 160 listed topics scanned by title and author), no MAP post or comment by any member in the fetched threads, nothing found by web search (GitHub/Zenn/Medium). Public 7th.

### 17th: RAG (titericz, aerdem4, conjuring92; public 5th)
No writeup found. No solution post in the competition forum (all 160 listed topics scanned by title and author), no MAP post or comment by any member in the fetched threads, nothing found by web search (GitHub/Zenn/Medium). Public 5th. Member conjuring92 won Eedi Mining Misconceptions (2024; code https://github.com/rbiswasfc/eedi-mining-misconceptions), whose pointwise reranker THLUO (4th) reused here, but no MAP-specific detail is published.

### 18th: Chris Deotte (solo)
Covered in the bank (pyramid ensemble). Nothing added here.

### 19th: Hacker (4 members; public 70th)
No writeup found. No solution post in the competition forum (all 160 listed topics scanned by title and author), no MAP post or comment by any member in the fetched threads, nothing found by web search (GitHub/Zenn/Medium). Second-largest upshake in the top 20: public 70th to private 19th.

### 20th: Kelvin (solo; public 99th)
No writeup found. No solution post in the competition forum (all 160 listed topics scanned by title and author), no MAP post or comment by any member in the fetched threads, nothing found by web search (GitHub/Zenn/Medium). Public 99th to private 20th, the largest upward move in the top 20.

### 21st: D.Imanishi (solo; ties 20th on private)
No writeup found. No solution post in the competition forum (all 160 listed topics scanned by title and author), no MAP post or comment by any member in the fetched threads, nothing found by web search (GitHub/Zenn/Medium).

## 3. Technique frequency matrix

Columns are private ranks. Y = used in the final solution, N = not used (or tried and dropped), ? = unknown. Ranks 11, 14, 16, 17, 19, 20 have no writeup and are all ?; they are omitted from the columns for width. Rank 18 (Deotte) is filled only from the bank's summary.

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 13 | 15 | 18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen family in final ensemble | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | ? |
| Non-Qwen-architecture member (GLM, Gemma, DeepSeek-Math, Phi-4) | Y | N | N | N | Y | N | Y | N | N | N | N | N | Y | ? |
| Largest final member >= 32B | Y | Y | Y | Y | N | N | Y | N | Y | Y | N | Y | ? | ? |
| Sequence-classification head | N | N | Y | N | Y | Y | N | N | N | Y | Y | Y | Y | Y |
| CausalLM label-token / letter / Yes-No scoring | N | Y | Y | Y | Y | N | Y | Y | Y | N | N | Y | N | Y |
| Custom candidate scoring (suffix / pointwise reranker / listwise) | Y | N | N | Y | N | N | N | N | N | N | N | Y | N | N |
| Restrict to the question's seen label set | Y | N | Y | Y | Y | ? | Y | ? | ? | ? | ? | Y | Y | Y |
| True/False removed (37 classes) | N | Y | Y | Y | Y | Y | Y | N | N | N | ? | Y | Y | Y |
| Other answer choices in the prompt | Y | ? | Y | N | Y | N | Y | Y | N | ? | N | Y | N | ? |
| LLM-generated synthetic training data | N | Y | ? | N | N | Y | Y | Y | N | N | N | N | ? | ? |
| Soft-label distillation from a bigger teacher | N | Y | N | Y | N | N | Y | N | N | N | Y | N | N | ? |
| Pseudo-labels on train duplicates | N | N | N | N | N | Y | N | N | N | N | N | N | N | ? |
| Full fine-tuning (any member) | Y | N | N | N | Y | N | Y | N | N | N | N | N | N | ? |
| LoRA / QLoRA (any member) | N | Y | Y | ? | Y | Y | Y | Y | Y | Y | Y | Y | Y | ? |
| Quantised inference (AWQ / GPTQ / INT8 / 4-bit) | Y | Y | Y | Y | ? | ? | Y | Y | Y | Y | ? | Y | ? | ? |
| vLLM at inference | N | Y | Y | Y | ? | ? | Y | Y | Y | Y | N | Y | ? | ? |
| Cascade: re-infer uncertain rows with bigger models | N | N | Y | N | N | N | N | N | N | N | N | N | N | Y |
| Several seeds of the same config in the ensemble | Y | ? | ? | N | N | Y | ? | N | N | N | Y | Y | ? | ? |
| Known label-error fix (QuestionId 31778) | ? | ? | N | Y | ? | ? | ? | ? | Y | Y | ? | ? | ? | ? |
| Regularisers for noise (label smoothing, R-Drop, EMA, AWP) | N | N | Y | N | N | N | Y | N | N | Y | Y | N | N | ? |
| Ensemble size (final members) | 4 used of 6 | 8 | several | 2 | 6 | 7 | ~6 | 50 | 10 | 15 | 5 | 6 | 5-6 | pyramid of many |

Counted summary (14 teams with writeups; "?" excluded from the denominator only where noted):
- 13 of 13 known used Qwen-family backbones (Qwen2.5, Qwen3, QwQ, Qwen3-Reranker/Embedding, R1-Distill-Qwen); only 4 of 14 put a non-Qwen architecture in the final mix.
- 8 of 12 known had a final member of 32B or larger; 4 (5th, 6th, 8th, 12th) won gold with 14B or smaller members only.
- 9 of 14 dropped the True/False prefix (37 classes); 8 of 14 explicitly restricted outputs to the question's observed label set.
- 9 of 14 used some CausalLM label-scoring formulation; 8 of 14 used a classification head (several used both).
- 9 of 14 served quantised models; 8 of 14 used vLLM.
- 4 of 14 trained on LLM-generated explanations (2nd, 6th, 7th, 8th); 2 more tried and dropped it (12th, 13th).
- 4 of 14 used soft-label distillation (2nd, 4th, 7th, 12th); 2 more tried it with little or no gain (9th, 15th).
- Only 3 of 14 used full fine-tuning in any member; LoRA/QLoRA was the default.
- Only 2 of 14 (3rd, 18th) used an uncertainty cascade.

## 4. Consensus vs divergence

Table stakes (nearly every top-20 writeup):
- Qwen backbones at 7B to 72B, fine-tuned (mostly LoRA/QLoRA r 16 to 128, lr 1e-4 to 2e-4, 1 to 4 epochs).
- Exploit that test shares the same 15 questions: derive True/False by rule and/or limit candidates to labels seen for that question.
- An ensemble (only 4th is near-single-model, and its 2-model blend did not beat its best single on private).
- Short prompts and fast T4 inference (vLLM + AWQ/GPTQ, fp16 not bf16, no max-length padding).

What separated the top 5 from 6 to 20 (with only 0.0009 between 1st and 6th, treat as tendencies):
- Top 3 invested in a formulation that scores only the question's candidates (1st suffix classification, 3rd Masaya CausalLM over the question's options, 4th pointwise reranker), plus either 32B+ models or strong systems work (1st W8A8 layer-wise, 3rd cascade).
- Top 4 all used either soft labels (2nd, 4th) or heavy regularisation and noise-aware validation (1st multi-seed loss-based selection, 3rd R-Drop/AWP/EMA). Ranks 6 to 15 mostly relied on ensembles of standard classifiers.
- 5th is the exception: no model above 14B, a diversity ensemble picked on public LB, including two unmodified public-notebook models.

Contradictory findings:
- Other answer choices in the prompt: helped 3rd (+0.003 to 0.004, https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/3rd-place-solution), 8th (reconstructed a to d options, https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/8th-place-solution), 7th and 13th; did not work for 12th (https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/12th-place-solution) and raised CV but lowered LB for 15th (https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/15th-place-solution-writeup); 9th found a simpler prompt slightly better. 15th's M Sato found option reconstruction had little effect.
- LLM synthetic data: central for 2nd (80k rows) and 7th (+0.002 to +0.005 LB on Gemma2-9B, https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/private-7th-public-10th-place-solution); at most ~0.001 for 8th; no CV gain for 12th; weak LB correlation and dropped for 13th (https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/map-competition-13th-place-gold-medal-solution). 7th itself says the effect is model-dependent.
- Rerankers: pointwise reranker was 4th's whole solution (https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/4th-place-solution-single-model-pb-0-948-lb-0-951) and listwise was 13th's main family; pointwise and pairwise failed for 9th (https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/9th-place-solution), pointwise and listwise failed for 10th (https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/10th-place-solution), listwise failed for 4th.
- Soft labels: +0.003 to +0.005 for 12th and a step gain for 4th; only +0.001 private (zero public) for 9th; inconsistent across folds for 15th.
- Qwen3 vs Qwen2.5: 9th and 13th found Qwen2.5 better (13th: Qwen2.5-32B matched 72B while Qwen3-32B lagged); 15th found Qwen3 best in its sweep; 1st's best members were Qwen3-32B and GLM-Z1-32B. Forum thread on this: discussion/612201.
- Bigger models: 1st says larger is better up to 32B; 13th saw a plateau beyond 32B; 9th saw better CV at 70B/72B but could not afford the time, and 32B x5 + 72B x2 scored below 32B x10 on private.
- LoRA vs full FT: 8th found LoRA better than full FT; 1st used full FT throughout; 7th's sinchir0 pipeline was full FT.
- Selecting by public LB: 5th chose ensemble members on public only and it paid off; 8th's LB-best pick beat its CV-best on private; 13th says public-based selection would have cost the medal; 15th over-tuned weights to public and fell from a gold zone it had unselected.
- Label-noise handling for QuestionId 31778: 4th, 9th, 10th corrected the rows; 3rd's attempt to hedge by inserting flipped True/False predictions did not work.

## 5. Aggregated "what did not work"

- Chain-of-thought / reasoning generation: 4th (biggest time sink), 13th.
- Listwise reranker: 4th, 10th. Pointwise or pairwise reranker on top of a classifier: 9th, 10th (32B). Pairwise CausalLM: 3rd (too slow).
- Retrieving similar labelled explanations into the prompt (RAG-style few-shot): 4th (CV up, LB and private down). Note 4th's per-label random few-shot samples DID work.
- 70B/72B models under the time limit: 9th (too slow; worse private when mixed); 13th (no gain over 32B); Llama-3.3-70B: 3rd, 9th.
- Non-Qwen families: Gemma 2/3, Mistral, Hermes, gpt-oss-20b, Llama: 3rd, 9th; gemma-2-9b unstable to train: 15th; deepseek-math and Qwen2.5 weaker than Qwen3: 15th.
- Too much rule-based augmentation (~10k rows): 6th.
- Synthetic data for training: 12th, 13th (see contradictions).
- 65 labels instead of 37: 6th. Training on Misconception only without True/False: 9th (opposite of most teams).
- Two-stage Category then Misconception model: 6th. Hierarchical multi-task with hard logic constraints: 10th.
- Generation over label descriptions, textual entailment, MoE head: 6th.
- Focal loss: 6th; focal, bootstrapping, label smoothing gave no consistent gain: 15th (10th, by contrast, used focal + smoothing).
- Masked loss over the question's labels: 15th (CV only).
- Correct-vs-Neither swap post-processor: 15th.
- Per-question dynamic ensemble weights: 7th. Adding fast small models: 7th. Distilling into ~1B encoders: 12th.
- Merging fold LoRAs: 9th. TTA: 10th. vLLM (for a classification pipeline): 12th. Hard-negative sampling: 12th.
- Unifying `Wrong_Fraction` / `Wrong_fraction`: 15th (no gain; 9th did it and kept it).
- Eedi-pretrained weights, fixed per-label letters: 3rd.

## 6. Shake-up note

Score gaps were tiny (0.00171 from 1st to 21st) and the public LB showed only 3 decimals, so rank moves are large while score moves are small.
- Up: Kelvin 99th to 20th (+79) and Hacker 70th to 19th (+51), both without writeups; kanto-kaggler 28th to 9th; Naoism 24th to 5th; Manan Jhaveri 22nd to 6th; tascj 14th to 1st (one submission all competition, per the bank).
- Down: iws 3rd to 13th; sayoulala 4th to 11th; RAG 5th to 17th; Takoi 7th to 16th; monsaraida & Masaya 1st to 3rd; 15th place (public 12th) fell out of gold.
- Explanations offered: 13th says public scores were optimistic and nearly any combination looked alike, with seed noise ~+/-0.002; 5th saw CV-LB correlation vanish above ~0.947 public and attributes it to rare hard rows and unseen phrasings; 4th's Qwen3-32B member was strong on CV and public but weaker on private; 15th blames over-tuned ensemble weights and weak CV construction, noting 15+ unselected 0.948+ private submissions. The bank adds label noise and seed variance (1st place) as the root cause.

## 7. UNVERIFIED

- No writeup located for ranks 11 (sayoulala), 14 (AnnieGo & kyoumonemui), 16 (Takoi), 17 (RAG), 19 (Hacker), 20 (Kelvin), 21 (D.Imanishi). The forum topic API returned only 160 topics (count field 160), so a later or off-forum writeup (Kaggle `/writeups/` page not linked from the forum) cannot be ruled out.
- 5th: whether the Qwen2.5-14B model #4 (no LB score shown) and the public-notebook models were retrained or used as published is not stated.
- 7th: exact ensemble weights and final member count are not stated ("~6" in the matrix is an estimate from the listed pipelines).
- 3rd and 2nd ensemble sizes in the matrix are approximate ("several", "8" from the 2nd writeup's fusion line).
- 18th (Deotte) matrix cells beyond formulation are unknown here; read the bank or the writeup before citing.
- 12th: whether inference was quantised is not stated (merged models + torch.compile).
- 9th's Zenn author is rikitomo0526 (Rick); his teammates are taken from the API member list, not from the article.
- Matrix "N" for teams that never mention a technique means "not reported", not a confirmed absence, for rows like label-error fix and seeds (those are marked ? where silence is ambiguous).

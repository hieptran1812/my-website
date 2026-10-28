# Jigsaw - Agile Community Rules Classification: top-20 writeup addendum

- Competition: Jigsaw - Agile Community Rules Classification
- Slug: `jigsaw-agile-community-rules` (id 94635, forum 7097115), https://www.kaggle.com/competitions/jigsaw-agile-community-rules
- Metric: column-averaged (per-rule) AUC. Code competition, 2x T4, 12 h, internet off. Public/private 30:70.
- Already covered in depth: `.claude/plans/kaggle-llm-writeup-bank-part3.md`, H2 "Jigsaw - Agile Community Rules Classification" (ranks 1 to 3 deep; 4, 5, 6, 8 as one-liners; 120th Chris Deotte; hidden-rule probing; train-on-test). This file does not repeat the bank's deep detail for ranks 1 to 3; it adds ranks 4 to 20 in full.
- Leaderboard source: Kaggle LeaderboardService API (lb.json). Rank 21 is listed too because ranks 19 and 20 share a score (0.92613).

## Top-20 table (private leaderboard, API order)

Discussion URLs use `https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/<id>`; writeup URLs are the API-linked `/writeups/` pages.

| Rank | Team | Members | Private | Public rank (score) | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | Guanshuo Xu | wowfattie | 0.92930 | 1 (0.93476) | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/1st-place-solution (discussion 613305) | Online Unsloth LoRA SFT of 6 models (Qwen3-14B/8B/4B, Qwen2.5-14B, Llama3.1-8B, Ettin-400M) on test examples, loss on the Yes/No token only, per-rule rank-normalised blend |
| 2 | ^^ | andyy12345, yaoqingjin, trudih, yfr999, hoatha | 0.92923 | 6 (0.93316) | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/2nd-place-solution (discussion 614279) | Majority-vote label cleaning, drop subreddit, 4 Unsloth 4-bit LoRA models (phi-4, Qwen3-14B/8B, Qwen2.5-7B) on 2 GPUs in parallel |
| 3 | Sergio Manuel Papadakis | socom20 | 0.92871 | 5 (0.93335) | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/3rd-place-solution (discussion 613324) | Offline-pretrained LoRAs then TTT; LLM last-token embeddings into per-rule LightGBM/XGB/SVC; slow LLM + fast embedding ensembles |
| 4 | Dream Team | rusalmighty | 0.92860 | 3 (0.93382) | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/4th-place-solution (discussion 613473) | SALSA single-token logit classifier, soft labels from vote ratios, Generalized CE q=0.955, AWQ Qwen3-4B then 14B with uncertainty-sampled pseudo-labels |
| 5 | Flash | ivanisaev, benbla, raki21, cnumber, kishanvavdara | 0.92855 | 2 (0.93472) | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/5th-place-solution-diverse-ensemble (discussion 613421) | Paradigm-diverse stack: TTT Llama3.1-8B/Qwen3-8B (r=256), DeBERTa NLI TTT, 6 embedders + LR/kNN, Qwen-32B with asymmetric example selection |
| 6 | ducnh279 | ducnh279 | 0.92781 | 12 (0.93249) | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/6th-place-solution (discussion 613150) | Deep Mutual Learning (online peer distillation) of Qwen3-14B, Qwen3-8B, Qwen3Guard-Gen-4B trained jointly on test examples |
| 7 | ktr | kcotton21 | 0.92760 | 9 (0.93281) | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/7th-place-solution (discussion 613215) | Train-on-test QLoRA of 8-14B models incl. safety models (shieldgemma-9b best single), Gemma2 fp16 unlocked in vLLM |
| 8 | Reddit comment moderators | williamwu88, justforfun44, muyouqian4 | 0.92759 | 4 (0.93370) | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/8th-place-solution-qwen3-14b3-llama2-13b1-b (discussion 613195) | 3x Qwen3-14B + Llama2-13B + bge-base, Unsloth QLoRA with sequence-classification head, seed ensembles, unlabelled negatives |
| 9 | lllleeeo + mango | mango789, lllleeeo | 0.92747 | 7 (0.93315) | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/9th-place-solution (discussion 613185) | Small generative TTT LLMs (Qwen3-4B-2507, Llama3.2-3B, Phi-3.5-mini, Qwen2.5-7B) + DeBERTa/E5/MPNet classifiers + BGE triplet retrieval, 0.7/0.25/0.05 |
| 10 | Nikita Babych | nikitababich | 0.92745 | 13 (0.93244) | no writeup found | unknown |
| 11 | takaito/Muj!rush!/yukko/miyaki | takaito, mujrush, yukko08, sankurero | 0.92734 | 18 (0.93207) | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/11th-place-solution (discussion 613506) | Online BERT (gte-multilingual-base) + per-rule contrastive SentenceBERT + Llama3.2-3B/Qwen3-4B with example TTA, hard pseudo-labels |
| 12 | losingself | level14taken | 0.92702 | 16 (0.93219) | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/12th-place-solution (discussion 613096) | Found the 2M-comment unlabelled source, soft-labelled 800k rows with a public LLM ensemble, pretrained DeBERTa-large on it, plus public Llama-3B TTT and triplet |
| 13 | ISE@JNU-1 | zwzwyang, mavicbf, mikeoscar1, yangjiahua | 0.92698 | 10 (0.93279) | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/13th-place-solution-using-positive-examples-from (discussion 613265) | Cross-rule negatives (30% of positives relabelled as negatives under another rule), Qwen3-8B/1.7B GPTQ-int4 + DeBERTa |
| 14 | next gen VNG warriors | bibanh, locbaop | 0.92678 | 8 (0.93288) | no writeup found | unknown |
| 15 | hasibirok0 / kfsky / Kinosuke | chiman3se, kengofujii, hasibirok0 | 0.92657 | 15 (0.93219) | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/15th-place-solution (discussion 613611) | 7-model diversity ensemble: Qwen3-4B "compliance" few-shot, Qwen2.5-7B + DeBERTa with 19k synthetic rows, Gemma2-2B cls, ModernBERT/BGE/E5 |
| 16 | PPD | philippsinger, dott1718, ilu000 | 0.92633 | 25 (0.93087) | no writeup found | unknown |
| 17 | Ebi | ebinan92 | 0.92625 | 11 (0.93270) | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/18th-solutionprivate-11th (discussion 613071) | In-notebook Qwen3-32B paraphrase augmentation per rule, QLoRA phi-4, Qwen2.5-14B, Qwen3-14B with Unsloth DDP |
| 18 | YF | raininbox, zhuwanglju, haiwenhong, xzwyyds, mccocoful | 0.92617 | 29 (0.93034) | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/18th-place-solution (discussion 617588) | Two-stage: 4-option multiple-choice pretraining on retrieved hard negatives, then Yes/No; Qwen3-4B/8B, Llama3.1-8B x2 seeds; no-leak local CV |
| 19 | Fidel Escalona | fbescalo | 0.92613 | 23 (0.93128) | no writeup found | unknown |
| 20 | Artem Silevich | amsilevich | 0.92613 | 27 (0.93065) | no writeup found | unknown |
| 21 | ACRC | mks2192, shimondanziger | 0.92608 | 17 (0.93218) | no writeup found | unknown (listed only because of the 19/20 tie) |

Writeups found: 15 of 20 (ranks 1-9, 11-13, 15, 17, 18).

## Per-team synthesis

### Ranks 1 to 3: only what the bank lacks
- 1st (613305): no LoRA rank, alpha or learning rate is stated in the post; the bank already has every concrete number given. Nothing further to add.
- 2nd (614279): epochs fixed to 1 as an explicit rule of their recipe; otherwise covered.
- 3rd (613324): ArcFace variant with one class per rule and a `'{rule}: {body}'` input for the Qwen3-Embedding model were both tried and did not help (body-only input kept). Writeup URL per API: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/3rd-place-solution

### 4th, Dream Team (Ruslan Berdichevsky, solo) - SALSA
Source: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613473 ; paper arXiv 2510.22691.
- **Framing:** SALSA (Single-pass Autoregressive LLM Structured Classification): constrain the answer to a tiny vocabulary ("0"/"1") and treat the next-token logits as a Bayesian estimator of the label; one forward pass, no decoding. The stronger the zero-shot instruct model, the stronger the signal.
- **Data:** the 2 positive + 2 negative examples on every row used as labels next to train labels; subreddit removed (examples were unrelated to the subreddit field); duplicates collapsed and the per-text **violation ratio used as a soft target**.
- **Models:** AWQ Qwen3-4B and Qwen3-14B (AWQ chosen over GPTQ for capability per bit); 32B+ judged infeasible in the budget.
- **Loss:** Generalized Cross-Entropy (Tsallis), q = 0.955, tuned on local CV and public LB sampling. Qwen3-14B GCE 0.92313 private vs CE 0.92309: effect marginal on private.
- **Prompt/inference:** short template, rule then comment last, assistant turn pre-filled with empty `<think></think>` and `<ANSWER>`; batches grouped by rule so the rule prefix stays in the KV cache.
- **Two-stage pseudo-labels:** Stage 1 small model infers on test; ~8,000 most-uncertain test rows become soft pseudo-labels; Stage 2 Qwen3-14B trained on original data plus those rows; Stage-1 and Stage-2 blended 0.8 : 0.2.
- **Ablations (private/public):** final 0.9286/0.93382; Stage-2 only 0.9264/0.93111; 4B single 0.91874/0.92482; 14B single 0.92313/0.92909; two stages with 4B 0.92239/0.92642; 4B+14B blend 0.92469/0.92921. About 0.004 AUC from the two-stage pipeline.
- **CV:** local CV plus public LB; dataloader-seed noise about 0.002 AUC, so smaller gains are not significant.
- **Distinguishing idea:** pseudo-labels pick the *most uncertain* test rows, used to anchor relative scores rather than add knowledge.

### 5th, Flash (ivanisaev, benbla, raki21, cnumber, kishanvavdara) - diverse ensemble
Source: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613421
- **Data:** drop subreddit; remove exact duplicates (train under 10k rows, no LB drop). Validation: trust public LB.
- **TTT LLMs:** Llama-3.1-8B-Instruct (0.92084 public / 0.92618 private, about 130 min) and Qwen3-8B (0.91971 / 0.92314) on 2x T4 DDP, fp16, gradient checkpointing, seq len 512; LoRA r=256, alpha=32, dropout 0, all 7 projection modules; paged_adamw_8bit, lr about 2e-4, cosine, about 10% warmup. Inference vLLM TP=2, max_model_len 512, max_tokens 1 constrained to Yes/No, truncate only the comment.
- **Embeddings, per rule:** E5 base/large, Qwen3-Embedding-0.6B, BGE small/large, GTE-large; near-duplicate clusters (cosine > 0.95-0.97) down-weighted; LR (31 C values) and distance-weighted kNN (K 10-15) on 6-fold CV, blended 0.7 LR + 0.3 kNN; per-rule non-negative weight search over the 6 embedders; final 0.9 x ensemble prob + 0.1 x rank-mean. 0.90067 / 0.90976.
- **DeBERTa NLI:** `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`, premise = body, hypothesis "This comment violates the rule: {rule}." plus 8 paraphrases; per-rule TTT on test examples upsampled to about 25k pairs, 2 epochs, only biases + head + LayerNorms trained; 0.89853 / 0.90636.
- **Qwen-32B (GPTQ + LoRA, trained on official data, inference only):** asymmetric example selection: most similar positive, least similar negative per comment (Qwen3-0.6B embeddings); +0.006 vs vanilla few-shot; 0.89065 / 0.89671, about 300 min.
- Public DeBERTa recipe and the public triplet + FAISS notebook (with train set, extra epoch) also included.
- **Ensemble:** per-rule rank normalisation, then "stack as you go": start 50/50 embeddings + DeBERTa-NLI, add each new model with a fixed mixing ratio and geometric decay of the previous stack; clip to [1e-6, 1-1e-6].
- **Distinguishing idea:** many weak, paradigm-different models (retrieval, NLI, LLM-TTT) beat a pure LLM stack on the shake; their best single (Llama 8B) moved up from public to private.

### 6th, ducnh279 (solo) - Deep Mutual Learning
Source: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613150
- **Data:** test positive/negative examples with their rules only; Qwen chat-template prompt with an empty `<think>` block and "Answer:"; no subreddit in the prompt.
- **Why DML:** no room on 2x T4 for a larger teacher; offline KD would only cover the 2 train rules; teacher softmax nearly one-hot and temperature did not soften it. DML adds a KL loss between peer students to the CE loss.
- **Results:** Qwen3-14B independent 0.925/0.921 -> DML 0.929/0.925; Qwen2.5-14B-Instruct 0.923/0.919 -> 0.929/0.924; DML pair ensemble 0.932/0.9271 vs independent pair 0.929/0.925.
- **Final:** 3-peer cohort Qwen3-14B, Qwen3-8B, Qwen3Guard-Gen-4B, chosen because 2-peer DML correlates errors; 0.93237 public / 0.92781 private. Joint DML training about 5 h; inference about 5 h, test split into two length-balanced halves, sorted by length, one per GPU. No LoRA hyperparameters stated.
- **Follow-up:** the 13th-place team added their data augmentation to this recipe and reported 0.92827 -> 0.92894 (https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/614439, post-competition).
- **Distinguishing idea:** online distillation among same-size peers when no bigger teacher fits.

### 7th, ktr (kcotton21, solo)
Source: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613215
- **Data:** subreddit fully ignored (host said it was not used in annotation; +0.007); dedup on `rule-body-label`, labels replaced by majority vote, ties discarded; done on train+test combined. Training rows = all test examples + train bodies.
- **CV:** built on train, **no correlation with LB**; used only as a sanity check. Trusted LB.
- **Models (public/private):** Qwen3-14B 0.926/0.919, phi-4 0.924/0.920, gemma-2-9b-it 0.924/0.920, **shieldgemma-9b 0.927/0.922 (best single)**, Qwen3-8B-Guard 0.923/0.921 (vanilla 0.920 public). Safety-tuned moderation models used as already-trained bases (not trained by him).
- **Recipe:** QLoRA with Unsloth, two models trained at once on different GPU ids; 4 models about 10 h total; the 5th (Qwen3-8B-Guard) inferred only on 2 rules. phi-4 run about 240 min, Gemma2 about 140 min. 32B QLoRA + DDP explored but not implemented.
- **Trick:** vLLM 0.10.0 refuses Gemma2 in fp16; he emptied `_FLOAT16_NOT_SUPPORTED_MODELS` in `vllm/config.py` with sed; one-token inference stayed stable.
- **Did not work:** Qwen3-32B trained on train data, inferring only public rules (much worse); synthetic data (loss went to near zero, did not match test).
- **Distinguishing idea:** pick backbones that were already moderation-tuned (ShieldGemma, Qwen3Guard).

### 8th, Reddit comment moderators (williamwu88, justforfun44, muyouqian4)
Source: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613195
- **Validation:** no CV, LB only (random split, 29% of 55k rows, stable mean AUC); every submission is the same config over 4 seeds so changes are not noise.
- **Final subs:** seeds 42-44 Qwen3-14B + seed 45 Llama2-13B + seed 46 bge-base-en-v1.5 (0.933 LB / 0.927 PB); variant with 2x Llama3.1-8B instead of bge (0.932 / 0.926).
- **Recipe:** Unsloth QLoRA on 2x T4 within 12 h; replaced Unsloth's SFT classification with a standard `AutoModelForSequenceClassification` head (unsure which is better); custom parallel training (Qwen3-14B could only run two single models in parallel, not batch x2; Unsloth must be initialised first); `group_by_length=True` gave about 2x training speed.
- **Data:** majority label per (body, rule) then dedup; from the official unlabelled corpus they added "absolute negatives"; guessed positives added only when LB did not drop.
- **Inference:** data split evenly over 2 GPUs, sorted short to long (about 4x faster), GPU1 waits a few seconds; 3 length buckets with different batch sizes freed time for one more 8B model.
- **Ensemble:** global ranking beat within-rule ranking.
- **Did not work:** pseudo-labelling; simulating CV during training.

### 9th, lllleeeo + mango
Source: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613185
- **Part 1, generative TTT** (from wasupandceacar's public Llama-3.2-3B pseudo-training notebook): swapped to Qwen3-4B-Instruct-2507 with only 5% of test data (0.916 -> 0.919), Unsloth (7.5 h on 2 GPUs -> 2.5 h on 1 GPU, 0.920), dropped subreddit (2.1 h, 0.922). Extra training variants used all deduplicated examples with randomly matched pos/neg pairs. Qwen3-4B 0.924/0.918, Llama3.2-3B 0.923/0.916, Phi-3.5-mini 0.918/0.913 (LB/PB); 3-model ensemble 0.930/0.924, 9 h.
- **Part 2, encoders:** DeBERTa-v3-base, E5-base, MPNet-base with 2x head lr, mean pooling, layer-wise lr decay (about +0.003); 0.909/0.902, 0.910/0.904, 0.906/0.900; ensemble 0.921/0.913.
- **Part 3:** BGE triplet loss + FAISS retrieval with single-centroid and UMAP clustering ensembled, 0.909/0.904.
- **Blend:** rank-normalise within each part, then 0.7 Part1 + 0.25 Part2 + 0.05 Part3 (set by LB). Sub 1: four generative models (adds Qwen2.5-7B) on 2x T4, 9 h, 0.933/0.927. Sub 2: 3 models x 2 data configs, 11 h, 0.933/0.926.
- **Did not work:** two-stage training, semantic matching of pos/neg examples, rule-specific prompts, label smoothing, logit-based ensembling, layer freezing, SENet fusion of CLS/mean/max pooling, rule-based ensemble weighting.
- **Note:** they deliberately avoided tuning test-data volume to the LB; in hindsight that more stable choice was suboptimal on private.

### 10th, Nikita Babych - no writeup found
No solution post in the forum topic list (233 topics scanned) and no external writeup found by web search.

### 11th, takaito / Muj!rush! / yukko / miyaki (takaito, mujrush, yukko08, sankurero)
Source: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613506 ; an NTT DOCOMO press release names Meisaku Suzuki and Kenichiro Miyaki in the 11th-place team (https://www.docomo.ne.jp/english/info/media_center/pr/2025/1121_00.html).
- Fully online (train-on-test). Diversity via prompt variants: examples + body, or body only.
- **BERT online:** two models trained in parallel on 2 GPUs; base beat large; multilingual strong; freezing input-side layers kept accuracy and saved time; final pick gte-multilingual-base (0.914 LB, 9 min). Public DeBERTa-v3-base notebook kept.
- **SentenceBERT per rule:** bge-large-en-v1.5 (tuned to LB) and gte-large-en-v1.5 (tuned to local CV); test data only; rule as anchor, pull pos examples, push neg examples, CosineSimilarityLoss; score = cosine(rule, body); 6 rules in about 35 min.
- **LLMs:** wasupandceacar's notebook as base; tried Llama-3.2-3B/1B, Qwen3-4B-Instruct, Gemma-2B-it; kept Llama-3.2-3B and Qwen3-4B. **TTA** over different pos/neg example combinations; a zero-shot (no examples) mode.
- **Hard pseudo-labels** for both BERT and LLMs: top/bottom "xx%" of ensemble scores labelled 1/0 (percentage not stated), LLMs restart from earlier LoRA weights; post-deadline check: private 0.92579 -> 0.92734 from pseudo-labelling.
- **Did not work:** few-shot pseudo-labelling (prompts too long); models above 4B (Llama3-8B, Gemma2-9B, Qwen2.5-14B GPTQ) too slow at inference; **continual pretraining on about 38 GB of Reddit** collected via the API (stable but no LB gain; suspected loss of instruction following).

### 12th, losingself (level14taken, solo)
Source: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613096 ; code https://github.com/level14taken/jigsaw-comp/blob/main/dinosaur-with-deberta-large.ipynb
- **External data:** probing showed under 0.5% of test bodies are missing from a 2M-comment unlabelled Reddit dataset he published (https://www.kaggle.com/datasets/level14taken/jigsaw-2m-reddit-unlabelled/data). Rows sampled interactively with Claude Code: judge 100 rows per subreddit per rule, keep subreddits with violation rate of at least 1%, giving 800k rule-body rows out of 12M possible.
- **Distillation:** soft labels on those rows from the best public LLM ensemble (bibanh's re-finetuned Qwen2.5-32B-GPTQ notebook), used to pretrain DeBERTa (stage 1: BCE with pos_weight, 1 epoch), then fine-tune. DeBERTa-base went 0.912 -> about 0.924 with 2 seeds in about 1 h of submission time; final used DeBERTa-large.
- **Ensemble:** with the public triplet notebook (8x augmentations, 0.907 public) and public Llama-3B fp16 TTT (0.920 public). Whole submission under 6 h, so 6 h were unused.
- **Did not work:** including low-violation (<1%) subreddits; keeping only bodies over 50 chars.
- **Tip:** public notebooks using bf16 on T4 ran about 6x slower.
- **Distinguishing idea:** recover the test's source corpus and distil an LLM teacher into a cheap encoder offline.

### 13th, ISE@JNU-1 (zwzwyang, mavicbf, mikeoscar1, yangjiahua)
Source: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613265 ; code https://www.kaggle.com/code/yangjiahua/finall-qwen3-8b-int4
- **Data trick:** sample 30% of training rows, move each to a different random rule and set label 0 (a positive for one rule is almost always a negative for another).
- **Models (public / public with trick / private):** Qwen3-8B GPTQ-int4 0.922 / 0.927 / 0.923 (8 h training); Qwen3-1.7B GPTQ-int4 0.911 / 0.918 / 0.912 (1.5 h); DeBERTa-base 0.906 / 0.911 / 0.903 (1 h); DeBERTa-small 0.900 (no augmentation; it hurt the ensemble).
- **Side results:** Qwen2.5-14B-Instruct on train.csv only scored 0.886 (Qwen3 was worse than Qwen2.5 train-only); LLM-generated (Gemini, Qwen) data for private rule 3 lifted Qwen2.5-14B 0.886 -> 0.888, other rules' synthetic data hurt; a modified GRPO on Qwen3-1.7B beat LoRA SFT (no numbers, code unreleased).
- **Distinguishing idea:** cross-rule negatives as free augmentation.

### 14th, next gen VNG warriors (bibanh, locbaop) - no writeup found
No solution post found. bibanh authored the public "LB 0.9 re-finetune Qwen2.5-32B GPTQ" notebook that 12th place used as its teacher (https://www.kaggle.com/code/bibanh/lb-0-9-re-finetune-qwen2-5-32b-gptq); whether the team's final solution built on it is UNVERIFIED.

### 15th, hasibirok0 / kfsky / Kinosuke
Source: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613611 ; slides (Japanese) https://speakerdeck.com/hasibirok0/map-jigsaw-code-golf-zhen-rifan-rihui-by-guan-dong-kagglerhui-jigsaw-15th-solution
- Trusted public LB, no CV. **Kept subreddit** as `r/{subreddit}` context (contrary to most). Dedup on [body, rule] for classifiers only, not for causal LMs.
- **Synthetic data:** 19,000 rows (gpt-oss-120b 4,000, Qwen3-235B 7,500, Qwen3-Next-80B 7,500) for DeBERTa stage-1 pretraining (+0.003, 0.906 -> 0.909) and Qwen2.5-7B.
- **Models (public/private, min):** Qwen3-4B few-shot 0.920/0.914 (278); Qwen2.5-7B zero-shot + synthetic 0.902/0.899 (193); Gemma2-2B seq-cls 0.914/0.906 (90; LoRA r=16, alpha=32, dropout 0.1, 7 modules, 2 epochs, lr 2e-5, batch 16, max len 384); DeBERTa-v3-base 0.909/0.901; ModernBERT-base 0.895/0.889; BGE-base 0.903/0.895; E5-base retrieval-only 0.887/0.877.
- **Trick:** asking for *compliance* ("Yes" if it complies) instead of violation, reading P("No"), lifted Qwen3-4B 0.918 -> 0.920; reason unknown.
- **Ensemble:** global rank average, weights 0.20/0.25/0.20/0.125/0.075/0.075/0.075; weights barely mattered. Qwen2.5-7B steps cut for runtime, lowering its single score but kept for diversity.
- **Did not work:** EMA; weight optimisation; concatenated embeddings + PCA; synthetic data for TTT except Qwen2.5-7B zero-shot; DeepSeek-R1-7B, MPNet, Ettin, Gemma2-9B (time vs score); step-by-step / CoT prompts.

### 16th, PPD (philippsinger, dott1718, ilu000) - no writeup found
No forum post or external writeup found.

### 17th, Ebi (ebinan92, solo)
Source: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613071 (titled "18th Solution (public 11th)"; the API lists private rank 17) ; notebook https://www.kaggle.com/code/ebinan92/18th-solution/notebook?scriptVersionId=270004249
- Everything runs inside the submission. **Qwen3-32B paraphrases** one extra sample per training body using rule-specific rewrite instructions that preserve violation or compliance (prompts given for all 6 revealed rules plus a default).
- QLoRA of phi-4, Qwen2.5-14B-Instruct and Qwen3-14B, input body + rule, output Yes/No; Unsloth's official multi-GPU DDP with a patch for RL-related imports and `device_id` in `device_map`. No hyperparameters stated.
- **Did not work:** pseudo-labels, Muon optimizer, new training data via few-shot prompting.
- **Distinguishing idea:** test-time paraphrase augmentation by a 32B model inside the 12-hour budget.

### 18th, YF (raininbox, zhuwanglju, haiwenhong, xzwyyds, mccocoful)
Source: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/617588 ; code https://www.kaggle.com/code/xzwyyds/18th-place-solution
- **No-leak local validation:** train on all example columns (dedup, conflicts removed); validate on train.csv bodies minus any (rule, body) that also appears as an example. Built because LB swung and identical notebooks crashed.
- **Stage 1:** for each positive, retrieve the 3 most similar negatives of the same rule with Qwen3-Embedding-8B and build a 4-option "which one violates the rule" question; loss on the answer letter only. **Stage 2:** continue to Yes/No, loss on the Yes/No token only, plus "rule keyword signals" (not detailed). NFKC normalisation, subreddit removed. TTT on test.
- **Models:** Unsloth LoRA, Qwen3-4B-Instruct-2507 (seeds 1337, 3407), Qwen3-8B (1337, 1237), Llama-3.1-8B-Instruct (3407, 340). Local AUC single vs two-stage: e.g. Qwen3-4B seed 3407 0.6537 -> 0.7448; ensemble 0.7515 -> 0.7610 (local, absolute values not comparable to LB).
- **Inference:** vLLM with a MultipleChoiceLogitsProcessor restricted to Yes/No; per-model rank normalisation, average seeds, weights 0.33/0.33/0.34 by family.
- **Distinguishing idea:** contrastive multiple-choice pre-task over hard retrieved negatives.

### 19th Fidel Escalona, 20th Artem Silevich - no writeup found
No forum post or external writeup found for either (tied at 0.92613).

### 21st, ACRC (mks2192, shimondanziger) - no writeup found (outside top 20)
mks2192 published "jigsaw-llama3-1-8b-instruct-training-one-epoch" (https://www.kaggle.com/code/mks2192/jigsaw-llama3-1-8b-instruct-training-one-epoch), the starting point credited by 2nd place.

## Technique frequency matrix

Y = used in final solution, N = not used (or tried and dropped), ? = not stated. Ranks 10, 14, 16, 19, 20 have no writeup (all ?).

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Train on test-set pos/neg examples (TTT/online) | Y | Y | Y | Y | Y | Y | Y | Y | Y | ? | Y | Y | Y | ? | Y | ? | ? | Y | ? | ? |
| Drop subreddit | Y | Y | Y | Y | Y | Y | Y | ? | Y | ? | ? | ? | ? | ? | N | ? | ? | Y | ? | ? |
| Label dedup / majority vote or soft vote ratio | Y | Y | ? | Y | Y | ? | Y | Y | Y | ? | ? | ? | ? | ? | Y | ? | ? | Y | ? | ? |
| Largest fine-tuned LLM >= 13B | Y | Y | Y | Y | Y | Y | Y | Y | N | ? | N | N | N | ? | N | ? | Y | N | ? | ? |
| Qwen3 in final | Y | Y | Y | Y | Y | Y | Y | Y | Y | ? | Y | N | Y | ? | Y | ? | Y | Y | ? | ? |
| Llama in final | Y | N | N | N | Y | N | N | Y | Y | ? | Y | Y | N | ? | N | ? | N | Y | ? | ? |
| Safety/guard-tuned base (Qwen3Guard, ShieldGemma) | N | N | N | N | N | Y | Y | N | N | ? | N | N | N | ? | N | ? | N | N | ? | ? |
| Unsloth | Y | Y | Y | ? | ? | ? | Y | Y | Y | ? | ? | ? | ? | ? | ? | ? | Y | Y | ? | ? |
| 4-bit weights (QLoRA/AWQ/GPTQ) | ? | Y | ? | Y | N | ? | Y | Y | ? | ? | ? | ? | Y | ? | ? | ? | Y | ? | ? | ? |
| Single next-token Yes/No (or 0/1) readout | Y | Y | N | Y | Y | Y | Y | N | Y | ? | Y | ? | ? | ? | Y | ? | Y | Y | ? | ? |
| Loss masked to answer token | Y | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | Y | ? | ? |
| vLLM inference | ? | ? | Y | ? | Y | ? | Y | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | Y | ? | ? |
| Encoder / embedding models in blend | Y | N | Y | N | Y | N | N | Y | Y | ? | Y | Y | Y | ? | Y | ? | N | N | ? | ? |
| External unlabelled Reddit data | N | N | N | N | N | N | N | Y | N | ? | N | Y | N | ? | N | ? | N | N | ? | ? |
| LLM-generated synthetic / paraphrase data | ? | ? | ? | ? | ? | ? | N | ? | ? | ? | ? | ? | N | ? | Y | ? | Y | ? | ? | ? |
| Pseudo-labels on test | ? | ? | ? | Y | ? | ? | ? | N | N | ? | Y | Y | ? | ? | ? | ? | N | ? | ? | ? |
| Distillation (KD/DML/soft-label teacher) | N | N | N | N | N | Y | N | N | N | ? | N | Y | N | ? | N | ? | N | N | ? | ? |
| Few-shot examples inside the prompt | N | N | N | N | Y | N | ? | ? | ? | ? | Y | ? | ? | ? | Y | ? | N | N | ? | ? |
| TTA | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | Y | Y | ? | ? | ? | ? | ? | ? | ? | ? |
| Multi-seed ensembles | ? | ? | ? | ? | ? | ? | ? | Y | Y | ? | ? | Y | ? | ? | ? | ? | ? | Y | ? | ? |
| Per-rule rank normalisation before blend | Y | ? | ? | ? | Y | ? | ? | N | N | ? | ? | ? | ? | ? | N | ? | ? | N | ? | ? |
| Trust public LB, no local CV | Y | N | Y | N | Y | ? | Y | Y | Y | ? | N | ? | ? | ? | Y | ? | ? | N | ? | ? |
| Ensemble size (models in final) | 6 | 4 | ~14 | 2 | 7+ | 3 | 5 | 5 | 4+3+1 | ? | 6+ | 3 | 3 | ? | 7 | ? | 3 | 6 | ? | ? |

Notes: 3rd's "Single next-token readout" is N because the LLM gives a last-token embedding to tree/SVC heads. 8th uses a sequence-classification head. 12th TTA = the triplet notebook's 8x augmentations. 3rd's "~14" counts 8 LLMs + about 6 fast models (approximate). 5th's >=13B model is Qwen-32B GPTQ+LoRA trained offline on official train data only; its TTT models are 8B. Rank 17's training-row source (test examples vs train only) is not stated.

### Counted summary (15 teams with writeups: ranks 1-9, 11, 12, 13, 15, 17, 18)
- 14 of 15 trained on the test set's own labelled examples (17th not stated).
- 14 of 15 used Qwen3 (all but 12th); 7 of 15 used Llama.
- 9 of 15 fine-tuned an LLM of 13B or more; all of ranks 1-8 did (5th only offline, for a 32B); of ranks 9-18 only 17th did.
- 9 of 15 explicitly dropped subreddit; 1 (15th) kept it.
- 9 of 15 put encoder or embedding models in the final blend.
- 7 of 15 explicitly trusted public LB without local CV; 4 (2nd, 4th, 11th, 18th) relied on local CV too.
- 3 of 15 used pseudo-labels in the final (4th, 11th, 12th); 3 report pseudo-labels or two-stage failing (8th, 9th, 17th).
- 2 of 15 used external unlabelled Reddit data (8th, 12th); 2 used LLM synthetic data in the final (15th, 17th); 2 used distillation (6th, 12th); 2 used safety-tuned backbones (6th, 7th).

## Consensus vs divergence

**Table stakes (nearly everyone):**
- Treat the 4 labelled examples per test row as a training set and fine-tune in the 12-hour notebook (train-on-test, from neibyr's 598099 notebook).
- Unsloth LoRA/QLoRA on 2x T4, two models in parallel, one per GPU, sorted-by-length inference.
- Read P(Yes) from one next-token logit instead of generating.
- Ignore subreddit and deduplicate with label voting.
- Ensemble several backbones with rank averaging.

**What separated the top 5 (and 6-8) from 9-20:**
- **Model size engineered into the budget.** Ranks 1-4 and 6-8 all fit 13-14B LLMs (Qwen3-14B, Qwen2.5-14B, phi-4, Llama2-13B) into the run; ranks 9, 11, 12, 13, 15, 18 topped out at 4-8B and leaned on encoders. 11th explicitly dropped >4B models for inference time.
- **A distinctive training signal on top of plain SFT:** answer-token loss masking (1st), label cleaning by vote (2nd), offline-pretrained LoRAs as TTT init (3rd), GCE + soft labels + uncertainty pseudo-labels (4th), DML (6th), safety-tuned bases (7th).
- 5th is the exception: largest *test-time-trained* model 8B (its 32B saw only train data), but the most paradigm-diverse blend.

**Contradictions:**
- **Pseudo-labels:** helped 4th (about +0.004 two-stage, 613473), 11th (+0.0016 private, 613506), 12th (soft labels on unlabelled data, 613096); failed for 8th (613195), 17th (613071); 9th's "two-stage training" did not work (613185).
- **Per-rule vs global rank:** 1st says per-rule rank normalisation gave a consistent gain (613305); 8th says global ranking beat within-rule ranking (613195); 9th says rule-based ensemble weighting did not work (613185); 7th argues per-rule ranking does not change the metric (613215).
- **Subreddit:** 7th measured +0.007 from ignoring it (613215) and most dropped it; 15th kept it as context (613611).
- **Synthetic data:** helped 15th's DeBERTa (+0.003, 613611) and 17th's paraphrases were in the final (613071); hurt 7th (613215); mixed for 13th (one rule helped, others hurt, 613265).
- **Examples in the prompt:** 5th's similarity-chosen examples gave +0.006 for Qwen-32B (613421) and 11th/15th used few-shot prompts; 9th found semantic matching of pos/neg examples did not work (613185).
- **32B models:** 5th kept an inference-only Qwen-32B in the blend (613421); 7th found a 32B trained on train data much worse (613215).
- **Local CV:** 7th found no CV-LB correlation (613215); 18th built a no-leak local split and used it to choose the two-stage recipe (617588); 2nd used local CV because LB was noisy (614279).
- **LM head vs classification head:** most read the Yes/No token; 8th used `AutoModelForSequenceClassification` and could not say which is better (613195).

## Aggregated "what did not work"

| Item | Team(s) |
|---|---|
| Pseudo-labelling | 8th, 17th (9th: two-stage training) |
| Few-shot pseudo-labelling (prompts too long) | 11th |
| Synthetic training data | 7th; 13th (all rules except private rule 3); 15th (for TTT except Qwen2.5-7B zero-shot) |
| New training data via few-shot prompting | 17th |
| Qwen3-32B trained on train data for public rules only | 7th |
| Models above 4B for a small-model stack (inference time) | 11th; 15th (DeepSeek-R1-7B, Gemma2-9B, MPNet, Ettin rejected for time) |
| Continual pretraining on about 38 GB Reddit | 11th |
| Simulating CV during training / local CV correlated with LB | 8th; 7th (no correlation) |
| Semantic matching of pos/neg examples | 9th |
| Rule-specific prompts | 9th |
| Label smoothing | 9th |
| Logit-based ensembling | 9th |
| Rule-based ensemble weighting / within-rule ranking | 9th; 8th |
| Ensemble weight optimisation | 15th |
| Layer freezing (DeBERTa) | 9th (11th found freezing input-side BERT layers harmless and faster) |
| SENet fusion of CLS/mean/max pooling | 9th |
| EMA of weights | 15th |
| Concatenated embeddings + PCA | 15th |
| CoT / step-by-step prompts | 15th |
| Muon optimizer | 17th |
| Keeping low-violation subreddits / only long bodies in unlabelled pretraining | 12th |
| ArcFace with per-rule classes; `{rule}: {body}` input for embedding model | 3rd |
| bf16 on T4 (6x slower in public notebooks) | 12th |
| Generalized CE on private (marginal) | 4th |

## Shake-up note

Moves are small in AUC: the top 20 private span 0.92613-0.92930, below 4th place's measured seed noise (about 0.002). Biggest risers (public -> private): 18th YF (29 -> 18), 16th PPD (25 -> 16), 20th Artem Silevich (27 -> 20), 11th (18 -> 11), 6th ducnh279 (12 -> 6), 2nd (6 -> 2). Biggest fallers: 14th next gen VNG warriors (8 -> 14), 17th Ebi (11 -> 17), 21st ACRC (17 -> 21), 5th Flash (2 -> 5), 8th (4 -> 8). Public 1st (Guanshuo Xu) held.
- 18th attributes its stability to a no-leak local CV and multi-seed rank-normalised blending instead of LB chasing (617588).
- 6th: DML gains were larger on private than its public position suggested (613150).
- 5th: its best single (Llama-3.1-8B TTT) scored higher on private than public; the diverse blend still slipped from 2nd to 5th (613421).
- 9th notes their choice not to tune test-data volume to the LB was suboptimal on private (613185).
- Most teams trusted public LB (30% of about 55k rows, random split), and the small top-20 shake-up supports that.

## UNVERIFIED
- Ebi's post is titled "18th Solution (public 11th)" but the API gives private rank 17 (public 11); the title probably reflects a pre-finalisation leaderboard.
- 4th's 0.8 : 0.2 blend: which stage gets 0.8 is not explicit in the post.
- 3rd place final ensemble size (~14) is my count from the bank, approximate.
- 11th pseudo-label percentage ("top/bottom xx%") not stated.
- 13th's modified GRPO result: no numbers given.
- 18th "rule keyword signals" not described.
- 14th: link to bibanh's public Qwen2.5-32B notebook is a membership fact, not evidence of their final method.
- 11th team member real names from the DOCOMO press release (via web search summary, page not opened).
- No writeup located for ranks 10, 14, 16, 19, 20 (forum list of 233 topics scanned + one web search); a later post or external blog may exist.

# LMSYS - Chatbot Arena Human Preference Predictions: top-20 addendum

- **Slug:** `lmsys-chatbot-arena` (competition id 66631, forumId 4305895)
- **Existing deep coverage:** `.claude/plans/kaggle-llm-writeup-bank-part1.md`, H2 `## 4. LMSYS - Chatbot Arena Human Preference Predictions (2024)` (lines ~596-765). It covers the task, metric (3-class log loss, lower is better), the leak and replacement private set, ranks 1-3 in depth, 4 and 5 briefly, and Chris Deotte's 16th-place ladder. This file does not repeat that detail; it adds ranks 6-20 and the facts the bank lacks.
- **Medals (API):** 1,688 teams on the final board; gold = top 13, silver = next 79.
- **Important:** private scores (0.969-0.986) are on the **replacement test set** after the leak. Public scores (0.83-0.88) are on the original, leak-affected public set. So the public-to-private comparison mixes two different test sets.
- Raw data: `/private/tmp/claude-501/kaggle-top20/lmsys-chatbot-arena/lb.json`, topic dumps under `/private/tmp/claude-501/kaggle-top20/_arena/lm_*.md`.

## 1. Top-20 table (from `LeaderboardService/GetLeaderboard`)

| Priv | Team | Members | Private (log loss) | Public rank (public score) | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | BlackPearl (no leak) | sayoulala | 0.96898 | 6 (0.83705) | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629 | Distill Llama-3-70B + Qwen2-72B into Gemma-2-9B, average 5 fold LoRAs, GPTQ 8-bit (bank) |
| 2 | [Rist] pref(no leak) | tascj, sakaku, kapenon | 0.97425 | 5 (0.83101) | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685 | Full-parameter Gemma-2-9B/27B + ArmoRM, full-swap augmentation, 240k pseudo-labels, Triton kernels (bank) |
| 3 | In The Arena (no leak) | Raja Biswas, Mark Tenenholtz | 0.97770 | 8 (0.84122) | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766 | Start from reward models + 500k vLLM-generated pseudo-labelled pairs (bank) |
| 4 | CAST(no leak) | DL, xxyc, w1623843225, fufu2022 | 0.98031 | 15 (0.85301) | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/529067 | Gemma-2-9b-it LoRA r=64, per-turn truncation, UltraFeedback pseudo-labels, empty/identical post-processing |
| 5 | Team Danube | Psi, dott, Pascal Pfeiffer | 0.98129 | 14 (0.84982) | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527669 | Reward-pretrain Gemma-2-9B on UltraFeedback win/loss, 2 runs with swapped order, INT8 |
| 6 | whak(no leak) | soso | 0.98156 | 23 (0.86078) | no writeup found | unknown |
| 7 | Qlora is all we need | Xujiaying, Cao Zhiyuan, Jayden Szeto, Xinyang Wang | 0.98162 | 34 (0.86381) | no writeup found | unknown (team name suggests QLoRA) |
| 8 | Safari | Stephen, zzzilos, Haidiche, odette01 | 0.98274 | 61 (0.86720) | no writeup found | unknown |
| 9 | Ebi (no leak) | Ebi | 0.98280 | 21 (0.85893) | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527704 | Gemma-2-9b-it x2 (one on swapped data) + lmsys-chat-1m pseudo-labels + RLHFlow pair-preference Llama-3-8B on short/long rows |
| 10 | Ilya Novoselskiy | Ilya Novoselskiy | 0.98310 | 86 (0.87172) | no writeup found | unknown |
| 11 | LLMT | M Lei6666, huangyufeng, Yao Liu, Danna Fu | 0.98353 | 65 (0.86823) | no writeup found | unknown |
| 12 | tq (no leak) | tmp (baosenguo), qdv206 | 0.98379 | 22 (0.85937) | no writeup found | unknown |
| 13 | gpt2-chatbot-which-trusted-cv | heartkilla (aruchomu) | 0.98392 | 57 (0.86657) | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/540876 | 3-stage transfer (700k task + 900k Nectar domain full FT, then QLoRA), Llama-3.1-8B + Gemma-2-9B, CE+focal, rsLoRA, post-processing |
| 14 | NHA | NguyenThanhNhan, Thanh Hau Nguyen, Anh Pham | 0.98392 (tie with 13 on displayed score) | 9 (0.84169) | no writeup found | unknown |
| 15 | M (no leak) | Mathurin Ache | 0.98427 | 58 (0.86672) | no writeup found | unknown |
| 16 | Chris Deotte (no leak) | Chris Deotte | 0.98532 | 18 (0.85617) | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527596 | Public QLoRA notebook upgraded: LoRA hyper-sweep, r=1024, LoRA-fp16 train / 8-bit infer, 2-Gemma TTA, truncate left |
| 17 | less is more (no leak) | xiaotingting, justin1357, hwz13, Littlemelon | 0.98541 | 29 (0.86218) | no writeup found | unknown; member justin1357 posted insights threads (see below) |
| 18 | Shelton (no leak) | Ray (faykudbq) | 0.98551 | 104 (0.87897) | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527595 | emiz6413 public Gemma-2 QLoRA notebook, r=64, more data, max_length 2100 TTA |
| 19 | MRIC (no leak) | lhwcv, Mohamed Eltayeb, Cody_Null, Roschild.Rui | 0.98564 | 28 (0.86167) | https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/528288 | emiz6413 notebook + o_proj/gate_proj, freeze 0, 3072 len, +21k data, r=64, deeper custom head |
| 20 | luckydog (no leak) | Mr.T, DevinEgber, CHEN, LinCWei, tinkzy | 0.98585 | 16 (0.85340) | no writeup found | unknown |

Writeups found: **10 of 20** (ranks 1, 2, 3, 4, 5, 9, 13, 16, 18, 19; rank 19 has a main post plus a teammate's failed-experiments post).

Ties: ranks 13 and 14 both show 0.98392 (the API orders them 13 then 14; the hidden score has more decimals or submission time breaks the tie).

## 2. Per-team synthesis

### Ranks 1-3: facts the bank lacks

Nothing material. The bank covers the three posts thoroughly. One addition from the 4th-place post: CAST tested Llama-3-8B and Llama-3.1-8B against gemma-2-9b-it and found Gemma "much better" than both, matching the 3rd place's "Llama-3.1 worse than Llama-3" and the 17th-place member's thread (below).

### 4th: CAST (no leak), additions to the bank

Source: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/529067. The bank already has data, LoRA config, truncation and post-processing. Missing facts:
- Backbone choice was empirical: gemma-2-9b-it "performed much better than both Llama3 8b and Llama3.1 8b".
- Hardware: team members had different setups; the best experiment ran on **4x A100**. Stage 1 took about 10 h and stage 2 about 15 h on **4x A100 40G**.
- Inference raises max_length from 2048 (train) to 3072 and averages the original and A/B-swapped passes.
- The post-processing constants are explicitly motivated by log loss's sensitivity to extreme values plus label noise: they did not predict 1.0 for the non-empty side even when the other response was empty.

### 5th: Team Danube, additions to the bank

Source: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527669.
- All models trained with **H2O LLM Studio**.
- How they found reward pretraining: they fine-tuned top RewardBench models and saw RLHFlow's reward models beat base and instruct versions of the same backbone, so they reproduced the pretraining (UltraFeedback-style binary win/loss) themselves. "The amount of data was also not particularly crucial."
- Reward pretraining gave "up to 20 points" (0.020 log loss) on CV and public LB.
- Inference: 8k context fits for two models with their setup, but they used 4k "to be safe for the re-run" on the replacement test set. Final inference notebook: https://www.kaggle.com/code/philippsinger/cool-crab-v44-rev-ff-sub/
- The team explicitly did not use pseudo-labels, distillation or lmsys-chat-1m, and still placed 5th, which is the clearest evidence in the top 20 that reward pretraining alone carries most of the gain that pseudo-labelling gave others.

### 9th: Ebi (no leak), solo

Source: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527704

**Data.** Kaggle train + the 33k LMSYS arena set (/discussion/500973). For pseudo-labels he took **lmsys-chat-1m**, kept English, non-redacted rows, extracted unique prompts and randomly sampled **two responses from different models per prompt**, deliberately only two to cover as many prompts as possible with limited volume: about **45k** extra pseudo-labelled rows.

**Input format.** `<context>\n{context}\n<response_A>\n{response_a}\n<response_B>\n{response_b}`, where response_a/b are only the *last* responses and context holds all earlier turns (with only one model's earlier responses). Tokenizer `truncation_side="left"`; max length 1536 train (256x6), 1792 inference (256x7).

**Models.**
- gemma-2-9b-it, LoRA on all linear layers, r=32, alpha=64, 1 epoch, lr 8e-5, linear schedule, no warm-up, global batch 16.
- Llama-3-8B initialised from **RLHFlow/pair-preference-model-LLaMA3-8B** (a reward model that already takes two responses), LoRA all linear, r=128, alpha=128, 1 epoch, lr 5e-5, global batch 16.

**Ladder (pre-leak public LB).** One Gemma + swap TTA 0.890 -> instead train a second Gemma on swapped data and ensemble 0.887 -> add chat-1m pseudo-labels to both 0.881 (private 0.9859) -> add Llama-3 without TTA 0.881 public but **private 0.9828**. So the Llama-3 reward-model member did not move public at all but improved the replacement test by 0.003.

**Inference.** Three models did not fit in 9 h, so **Llama-3 was run only on rows with <= 512 or >= 1500 tokens**. Speed: 8-bit instead of 4-bit, `merge_and_unload()` (no score loss at 8-bit), dynamic batch size by token length.

**Distinguishing idea.** Training a second model on swapped data instead of swap-TTA, and a length-gated third model to fit the time budget.

### 13th: gpt2-chatbot-which-trusted-cv (heartkilla), solo gold

Source: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/540876

**Three-stage transfer learning,** each stage closer to the final task:
1. **Task adaptation (full fine-tune):** binary preference model on ~**700k** pairs from RLHFlow's preprocessed preference collection (arXiv 2405.07863), excluding SHP (human completions) and PKU (safety focus).
2. **Domain adaptation (full fine-tune):** **Nectar**, only its lmsys-chat-1m subset (7 completions per prompt ranked by GPT-4) -> ~**900k** binary pairs.
3. **Final (4-bit QLoRA):** ternary model on all competition data + LMSYS 33k + **toxic-dpo** (toxic-but-helpful answers labelled as winners, to remove leftover "safe" refusal behaviour) + **HelpSteer2** (tie if helpfulness differs by < 1, tuned so label distribution matches the competition). Deduplicated across stages. Two final models with slightly different data mixes.

**Models.** 7-9B range was best; larger gave marginal gains. Final backbones `meta-llama/Llama-3.1-8B-Instruct` and `google/gemma-2-9b-it`; Llama-3.1 blended better with Gemma than Phi, InternLM or Qwen. **Multi-sample dropout** in the head: higher train loss but ~0.005 better validation loss.

**Prompting (~0.01 CV total).** Left truncation; **turn alternation** (prompt_1, a_1, b_1, prompt_2, ...) instead of all-A-then-all-B; an instruction appended at the end ("pick the best response ... Answer only with 'A', 'B', or 'Tie'"), which sped convergence.

**Position bias.** Train one model on AB, predict on BA; train another on BA, predict on AB; weighted average.

**Training.** Loss = 0.7 CE + 0.3 focal. LoRA on Q/K/V/O, r=32, alpha=32, **rsLoRA (~0.02 gain from the start)**, cosine with 10% warm-up, bf16, max length 1.8k, 8-bit Adam. Hardware: mostly one A100, sometimes H100; 8x H100 a few times for pretraining.

**Inference.** SDPA (T4 has no Flash Attention), so Gemma softcapping was disabled in training. Max length 1.8k for one model, 4.1k for the other. Post-processing: temperature scaling tuned on validation; a "null" response -> 1/3 each; refusal phrases ("sorry", "as an AI", "apologize") -> boost the other side by 1.05 and renormalise; renormalise so mean class probabilities match training priors. Length-sorted, test split over two GPUs, autocast.

**Did not work:** DoRA, LoftQ; more LoRA target layers; Lion, Sophia; pooling variants (max, GRU); layer freezing; TF-IDF + LightGBM and DeBERTa blending; label smoothing; Unsloth/vLLM/AWQ/GPTQ speed-ups (little gain for classification); Arena-Hard, MT-Bench data; LM pretraining on prompts; AWP/FGM; distillation from Gemma-27B ("should have tried distilling from an ensemble of large models"); stacking; LoRA averaging, SWA, EMA; pseudo-labelling; ArmoRM labels as auxiliary regression; auxiliary model-identity loss; length-based probability scaling. Also regression and Siamese reformulations.

**Distinguishing idea.** Staged full-FT preference pretraining at 1.6M pairs before the competition QLoRA, plus refusal/prior post-processing.

### 16th: Chris Deotte (no leak), additions to the bank

Source: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527596. The bank has the LoRA tuning recipe and LB ladder. Missing:
- **Hardware:** 8x V100 32GB, sometimes 4x A100 80GB (NVIDIA compute).
- **LR-alpha coupling he measured:** with lr 2e-4 (1 epoch, linear warm-up) the best alpha is 4 for every rank; with lr 2e-5 best alpha is 192; with lr 6e-5 best alpha is 64. Effective backbone lr = alpha/r x head lr.
- **"Infer 3 heads"** (preference, model_a identity, model_b identity; discard the last two) gave 0.891 -> 0.890.
- **LoRA fp16 training + 8-bit inference** instead of QLoRA 4-bit/4-bit: ~10% faster inference, which bought the 3072-token inference length.
- Best single model: CV 0.878, original public LB 0.888; public Train/Quantize/Infer notebooks: https://www.kaggle.com/code/cdeotte/16th-place-train-1-of-3 (and -quantize-2-of-3, -infer-3-of-3). Under Kaggle-only training (QLoRA 4-bit, 2048 tokens) the same pipeline gives 0.889.
- **Did not work: Llama-3.1-405B as a judge for synthetic data.** Three few-shot variants (generate whole rows; generate response B + target for lmsys-1m prompt/response A; label existing lmsys-1m pairs). The labelling variant reached ~0.95 log loss locally but the data did not help.
- He trained on 8x V100 with HF Trainer and notes gradient checkpointing "didn't seem to work correctly" with it.

### 18th: Shelton (no leak), Ray (faykudbq), solo

Source: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527595

Built directly on @emiz6413's public "Gemma-2 9b 4-bit QLoRA" training/inference notebooks, modifying only the config. Base `unsloth/gemma-2-9b-it-bnb-4bit`; max_length 1024 train; `n_splits=800` with fold 0 (so essentially all data in training); `adamw_8bit`; per-device batch 4 x grad-accum 16; 1 epoch; **lr 1e-5**, 20 warm-up steps; **LoRA r=64, alpha=32**, dropout 0.05; target modules q/k/v/o/gate_proj; **freeze_layers 0** (the public notebook froze the first 16). The title's "more data" means extra external data (the post does not name the set). Inference with simple swap TTA at **max_length 2100**: https://www.kaggle.com/code/faykudbq/18th-inference-gemma2-max-length-2100-tta

He reports a clear local-CV / public-LB correlation and no significant shake-up for himself (though he jumped public 104 -> private 18 on the replacement set). Warning he gives: raising max_length "often results in higher LB scores" but risks timeouts, and identical submissions could time out on one run and pass on another.

**Distinguishing idea.** Minimal change to a public notebook: unfreeze all layers, higher rank, more data, longer inference. A public-notebook baseline can reach rank 18.

### 19th: MRIC (no leak)

Sources: main https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/528288 (Roschild.Rui); failed experiments https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527651 (Mohamed Eltayeb); weights https://huggingface.co/RoschildRui/LMSYS_Gemma2_QLoRA_ft; model post /discussion/532364.

Also built on emiz6413's QLoRA notebooks. Their key tricks with reported gains (the post writes deltas as "0.12" etc.; the table below shows they are log-loss changes of about 0.01, e.g. 0.935 -> 0.908 CV):
1. Add `o_proj`, `gate_proj` to LoRA targets (CV 0.921 -> 0.908, LB 0.933 -> 0.925 single model).
2. Training max_len 3072.
3. Freeze 0 layers.
4. +21k deduplicated external 33k data.
5. LoRA r=64 (alpha 128 / 64 variants).
6. A deeper custom head (hidden -> 2x hidden -> hidden/2 -> 3, no bias).

Experiment table (gemma-2-9b-it, 1 epoch, lr 2e-4 then 1e-4, A100-40G/80G): best CV around 0.83-0.86 with many-split (n_split 50/100, i.e. near-full-data) training; best public LB 0.892 (v38 with TTA at 2048). Llama-3.1-8B-it trials: original 0.981 CV, with SFT-style change 0.931, + mixed data 0.921, adding o/gate/dense projections did not help. lhwcv solved the inference time-outs.

**Unresolved / did not work:** two-stage fine-tuning would not converge; could not find a working distillation method; full-parameter FT did not reach good CV. Listed as "potential" but untried: left truncation, 8-bit inference, model-identity auxiliary heads, EMA.

**Eltayeb's failed experiments:** (a) rows where OOF loss is worse than uniform 1/3: replacing them with 1/3 would give CV 0.717, and most were subjective prompts; he labelled every training prompt Objective/Other with **Qwen2-72B** and tried predicting 1/3 for subjective ones or relabelling them as 1/3 in training: no gain. (b) Sorting training by length (long -> short) so noisy short samples get the low end of the lr schedule: no difference. (c) Expanding multi-turn conversations into cumulative samples and averaging: much worse.

### No-writeup teams (one line each)

- **6 whak (no leak), soso:** no writeup. Posted "Do not share your solution after deadline" (https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/524181) asking teams to wait until the reproduce phase so leak users could not copy solutions.
- **7 Qlora is all we need:** no writeup found; no approach detail in the forum pages read.
- **8 Safari:** no writeup found.
- **10 Ilya Novoselskiy:** no writeup found. Largest public-to-private climb inside the top 10 (86 -> 10).
- **11 LLMT:** no writeup found.
- **12 tq (no leak):** no writeup found.
- **14 NHA:** no writeup found.
- **15 M (no leak), Mathurin Ache:** no writeup found.
- **17 less is more (no leak):** no writeup. Member justin1357 posted: UltraFeedback "165k" set hurt his LB, which he attributes to its prompts coming from eval sets (UltraChat, ShareGPT, Evol-Instruct, TruthfulQA, FLAN) rather than Arena users, and DPO/SimPO-tuned Gemma-2 variants did poorly (https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/523527); Llama-3.1 was worse than Gemma-2 after 1-4k steps (/discussion/522024); DeBERTa-transferred ideas failed on LLMs (/discussion/524795). He also posted the "Crazy lb 0.707 run in just 1h" thread (/discussion/523968) and "[leak source] data here" (/discussion/524207) that surfaced the leak.
- **20 luckydog (no leak):** no writeup found.

## 3. Technique frequency matrix

Columns are private ranks. Y = used, N = stated not used or tried and dropped, ? = unknown (mostly no writeup). Ranks without writeup are all `?` and omitted from columns to keep the table readable: 6, 7, 8, 10, 11, 12, 14, 15, 17, 20.

| Technique | 1 | 2 | 3 | 4 | 5 | 9 | 13 | 16 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma-2-9B in final | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Llama-3/3.1-8B family in final (incl. RM inits) | N (teacher only) | Y (ArmoRM) | N | N | N | Y (RLHFlow pair-RM) | Y | N | N | N |
| Model > 9B in the submitted ensemble | N | Y (27B) | N | N | N | N | N | N | N | N |
| Larger teachers / distillation | Y (70B/72B) | N | N (tried, failed) | N | N | N | N (27B tried, failed) | N | N | N (tried, failed) |
| Reward-model init or reward pretraining | Y (UltraFeedback post-pretrain) | Y (ArmoRM) | Y | N | Y | Y (Llama) | Y (700k + 900k) | N | N | N |
| 33k LMSYS extra set | Y | Y | ? | Y | Y | Y | Y | Y | ? | Y |
| Pseudo-labels on external prompts | N | Y (240k) | Y (500k) | Y (30k UF) | N | Y (45k chat-1m) | N (tried, failed) | N | N | N |
| LLM-generated synthetic responses | N | N | Y (vLLM) | N | N | N | N | N (405B judge failed) | N | N |
| LoRA/QLoRA (vs full FT) | LoRA/QLoRA | Full FT | QLoRA | LoRA | ? (H2O LLM Studio) | LoRA | Full FT pretrain + QLoRA | LoRA (fp16) | QLoRA | QLoRA |
| A/B swap used (aug or TTA) | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Left truncation / keep conversation end | ? | ? | Y (first 2 + last 2 turns) | Y (per-turn budget) | ? | Y | Y | Y | ? | N (listed as untried) |
| 8-bit quantisation at inference | Y (GPTQ) | N (custom kernels) | ? | ? | Y (bnb INT8) | Y | N (autocast) | Y | ? | ? |
| Rule post-processing | ? | ? | ? | Y (empty/identical) | ? | ? | Y (null, refusal, priors, temp) | ? | ? | ? |
| Ensemble size (models at inference) | 1 merged | 3 | 2 | 1 + TTA | 2 | 3 | 2 | 2 | 1 + TTA | ? |

Counted summary over the 10 writeups (ranks 1, 2, 3, 4, 5, 9, 13, 16, 18, 19):
- **10 of 10** used Gemma-2-9B as the main backbone.
- **10 of 10** used A/B swapping in some form (augmentation, second model on swapped data, or TTA).
- **8 of 10** used the 33k extra LMSYS set (2 not stated).
- **6 of 10** started from reward models or did a reward/preference pretraining stage (1, 2, 3, 5, 9, 13). All of the top 5 except CAST did.
- **4 of 10** used pseudo-labels on external prompts (2, 3, 4, 9); 2 more tried and failed (13, and 3rd for a second round).
- **1 of 10** succeeded with distillation from larger models (1st); 3 tried and failed (3, 13, 19).
- **2 of 10** used full-parameter fine-tuning in the final stage (2nd) or pretraining stages (13th).
- **5 of 10** explicitly kept the end of the conversation (3, 4, 9, 13, 16).
- **4 of 10** quantised to 8-bit at inference (1, 5, 9, 16).

## 4. Consensus vs divergence

**Table stakes (nearly every writeup):** Gemma-2-9B (it or a reward-tuned variant) as the backbone, LoRA on all or most linear layers with r >= 32, A/B swap symmetry, the 33k extra Arena set, length-sorted batched inference split over the two T4s, and 1 epoch of fine-tuning.

**What separated the top 5 from 6-20:** extra *preference signal* injected before the competition fine-tune. Ranks 1, 2, 3 and 5 all had either (a) larger teachers' logits (1st), (b) hundreds of thousands of pseudo-labelled pairs (2nd, 3rd), or (c) reward-model initialisation / reward pretraining on UltraFeedback (1st, 3rd, 5th). The writeups from ranks 16-19 are public-notebook upgrades (emiz6413's QLoRA Gemma-2) without any of these. The 9th and 13th sit between: 9th used 45k pseudo-labels and a reward-model Llama; 13th used 1.6M pairs of preference pretraining yet still landed 13th, suggesting the pretraining data's distribution mattered (Nectar/GPT-4 rankings vs Arena humans).

**Contradictory findings:**
- **Pseudo-labelling:** decisive for 2nd and 3rd (https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685, /discussion/527766) and helpful for 9th (/discussion/527704), but 13th lists pseudo-labelling as unsuccessful (/discussion/540876) and 5th placed without it (/discussion/527669).
- **Distillation:** the winning idea for 1st (/discussion/527629), yet 3rd "couldn't get it right" (/discussion/527766), 13th failed from Gemma-27B (/discussion/540876), 19th could not find a working method (/discussion/528288). 13th's own diagnosis: distil from an *ensemble* of large models, not one.
- **UltraFeedback:** 1st and 5th used it for pretraining with clear gains; 4th used it for pseudo-labels; 3rd found it no better than ORPO-DPO mix; 17th-place member justin1357 found it hurt when added as training data (/discussion/523527). Plausible reconciliation (not stated by any one team): useful as a pretraining/warm-up signal, harmful mixed directly into the final fine-tune.
- **Label smoothing:** 3rd (Raja) used 0.03; 13th found it unhelpful.
- **LoRA averaging / weight averaging:** 1st averaged 5 fold LoRAs; 13th lists LoRA weight averaging, SWA and EMA as not working.
- **Model-identity auxiliary heads:** 16th got 0.001 from 3 heads; 3rd and 13th found auxiliary model-identity loss unhelpful.
- **Speed tools:** 13th found vLLM/AWQ/GPTQ gave little speed-up for classification; 1st used GPTQ 8-bit and 9th/16th 8-bit bnb for speed with no score loss.

## 5. Aggregated "what did not work"

| Item | Reported by |
|---|---|
| Gemma-2-27B (could not train well / distil from) | 3rd, 13th |
| Distillation from larger models (single teacher or method not found) | 3rd, 13th, 19th |
| Llama-3.1-8B worse than Llama-3 / Gemma-2 | 3rd, 4th, 17th (justin1357), 19th |
| Auxiliary model-identity loss | 3rd, 13th |
| Training new tag tokens / embeddings | 3rd |
| rsLoRA / LoRA+ / DoRA / full-tuning not clearly better | 3rd (Raja); DoRA, LoftQ also 13th (but 13th found rsLoRA +0.02) |
| Label smoothing | 13th |
| Other optimizers (Lion, Sophia) | 13th |
| Pooling variants (max, GRU) | 13th |
| Layer freezing | 13th; freezing 16 layers in the public notebook also worse (16th, 18th, 19th unfroze) |
| Classical / encoder blending (TF-IDF + LightGBM, DeBERTa) | 13th, 17th (justin1357) |
| Arena-Hard, MT-Bench as extra data | 13th |
| UltraFeedback 165k as direct training data | 17th (justin1357) |
| Pseudo-labelling | 13th (3rd: second round added little) |
| Stacking; SWA, EMA, LoRA weight averaging | 13th |
| ArmoRM scores as auxiliary regression target | 13th |
| Length-based probability scaling | 13th |
| Regression / Siamese reformulations | 13th |
| Llama-3.1-405B as judge or generator of synthetic training data | 16th |
| Two-stage fine-tuning (would not converge); full-parameter FT | 19th |
| Treating subjective prompts as uniform (LLM-classified) | 19th (Eltayeb) |
| Long-to-short length-sorted training | 19th (Eltayeb) |
| Cumulative multi-turn sample expansion | 19th (Eltayeb) |
| Predicting model identity / PL loosened CV-LB correlation | 3rd |

## 6. Shake-up note

The leak made public ranks unreliable: the public set was the leak-affected original; private is a replacement set (see bank). Among the top 20 private:
- **Biggest climbers:** Shelton 104 -> 18, Ilya Novoselskiy 86 -> 10, LLMT 65 -> 11, Safari 61 -> 8, M 58 -> 15, gpt2-chatbot-which-trusted-cv 57 -> 13. Only Shelton explains it, and only loosely (more data, higher rank, trusted CV). The 13th team's name ("which trusted CV") states its strategy: select by CV, not by the leak-distorted public board.
- **Fallers inside the top 20:** NHA public 9 -> private 14; luckydog 16 -> 20; Team Danube 14 -> 5 is a rise. No-leak teams' public ranks were compressed below leak users, so a public rank of 15-30 was roughly equivalent to a no-leak top 10.
- Ebi's numbers show the replacement set rewarded diversity the public set did not: adding Llama-3 left public unchanged (0.881) but improved private 0.9859 -> 0.9828.
- Danube used 4k context instead of 8k "to be safe for the re-run": the rerun on a new set with a time limit added runtime risk, which 18th also flags (identical submissions timing out non-deterministically).

## 7. UNVERIFIED

- Approaches of ranks 6, 7, 8, 10, 11, 12, 14, 15, 17, 20: no writeup found on the forum (all 292 topics scanned) or the API `solutionWriteUpUrl`; one web search found nothing. The team-name hint "Qlora is all we need" is not evidence of approach.
- 18th's "more data": the post does not name the dataset.
- 19th's gain numbers are written as "0.12", "0.17" etc. in the post; interpreted here as ~0.01-scale log-loss changes from their own table. The exact mapping is the authors'.
- 5th's LoRA vs full FT and hyperparameters are not stated (H2O LLM Studio only).
- How the API breaks the 13/14 tie on 0.98392.

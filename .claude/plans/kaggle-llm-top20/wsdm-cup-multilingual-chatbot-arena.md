# WSDM Cup - Multilingual Chatbot Arena: top-20 addendum

- **Slug:** `wsdm-cup-multilingual-chatbot-arena` (competition id 86946, forumId 6102659)
- **Existing deep coverage:** `.claude/plans/kaggle-llm-writeup-bank-part1.md`, H2 `## 5. WSDM Cup - Multilingual Chatbot Arena (2024-25)` (lines ~767-923). It covers task (binary A/B, accuracy), data, the 4.75 h / 12 h time limits, the forecasting-phase private set, ranks 1-3 in depth and a short note on the 7th-place cascade. This file adds ranks 4-20 and what the bank lacks.
- **Medals (API):** 890 teams on the final board; gold = top 11, silver = next 39.
- **Public LB is unusable from the API:** every team's public `displayScore` is `0.000000`, so the "public rank" the API returns (e.g. 824 for the winner) is an artefact of sorting zeros, not a real rank. Public scores below come only from the teams' own posts.
- Raw data: `/private/tmp/claude-501/kaggle-top20/wsdm-cup-multilingual-chatbot-arena/lb.json`; topic dumps `/private/tmp/claude-501/kaggle-top20/_arena/ws_*.md`.

## 1. Top-20 table (from `LeaderboardService/GetLeaderboard`)

| Priv | Team | Members | Private (accuracy) | Public (self-reported) | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | whitefebruary | Michael Pieler, Max (orangeaugust) | 0.716092 | ? | https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902 | Qwen2.5-14B full FT distilled from 5x Qwen2.5-72B, 50k synthetic multilingual prompts, padding-free inference (bank) |
| 2 | zhudong1949 | zhudong1949 | 0.716047 | 0.709 | https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567948 | Start from tascj's LMSYS gemma2-9b/ArmoRM checkpoints, 95k vLLM pseudo-labels, English-then-multilingual epochs (bank) |
| 3 | PLaMo 1000000B | RihanPiggy (honglihang) | 0.714967 | ? | https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567584 | CausalLM A/B-token Qwen2.5-14B + Phi-4 under vLLM, self-distillation, auto-round (bank) |
| 4 | HKUST-GZ DSA KiMi Lab | HengweiDAI | 0.714067 | 0.703 | https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/568522 | CausalLM A/B token, full-vocab post-pretrain then binary finetune, distil Athene-V2 + Nemotron-70B + Qwen2.5-72B, vLLM `allowed_token_ids`, GPTQ 8-bit |
| 5 | Just doing | Myo Min Htet, Chenyu Lin, Albert Wang, dlondmike, ZHENG NIE233 | 0.712627 | 0.706812 | https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567856 (+ /567724, /567617) | Qwen2.5-14B-it CausalLM LoRA r=64, middle truncation, swap, 2-stage hard->soft labels, vLLM + logit-tie post-processing |
| 6 | Oh Your Mercy Never Fails | Baiph, zzz, Xianhui Zhang, Zhecheng Li, no fit just luck | 0.712582 | ? | https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567600 | 560k pseudo-labelled pairs + distil Llama-3.3-70B/Qwen2.5-72B into Gemma-2-9B and Qwen2.5-14B; Gemma only on hard rows |
| 7 | Team Turing | Trushant Kalyanpur, Raja Biswas | 0.712222 | ? | https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567589 (not in API) | Uncertainty cascade gemma-2-9b -> Qwen2.5-14B -> Qwen2.5-32B / Mistral-Small-24B; 70B+ pseudo-labels on 100k generated pairs |
| 8 | sayoulala | sayoulala | 0.712042 | ? | no writeup found | unknown (LMSYS 1st-place author; reopened his LMSYS code, /discussion/554766) |
| 9 | quenn | Hanger Chou, Osmond, Jingcheng Deng, 李天成 | 0.711817 | ? | no writeup found | unknown |
| 10 | tascj | tascj | 0.711817 (same displayed score as 9th) | ? | no writeup found | unknown (LMSYS 2nd-place author whose code 1st and 2nd reused) |
| 11 | Evalthon | klein.peng, invalidpointer, hccngu, ZachZhang | 0.711412 | ? | no writeup found | unknown |
| 12 | monsaraida | monsaraida | 0.710872 | ? | https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567614 | Qwen2.5-14B-it QLoRA r=8, 2-stage (UltraFeedback/DPO pretrain then Arena data), "(snip N chars)" marker, model-name aux head, TTA on 40% most uncertain |
| 13 | x-creative | x-creative | 0.710737 | ? | no writeup found | unknown |
| 14 | Takoi | Takoi | 0.709432 | ? | no writeup found | unknown |
| 15 | KYRL | lhwcv, Kelvin, Yevhenii Maslov, Roschild.Rui | 0.709252 | ? | no writeup found | unknown final; lhwcv's CV-vs-LB thread documents a Gemma-2-9b baseline (below) |
| 16 | Tilbur | Tilbur | 0.708802 | ? | no writeup found | unknown |
| 17 | yukichi0403 | yukichi0403 | 0.708667 | ? | no writeup found | unknown |
| 18 | zxc123cc | zxc123cc | 0.708442 | ? | no writeup found | unknown |
| 19 | Shaomu & Shiyi | Shaomu, morita0913 | 0.708172 | ? | no writeup found | unknown |
| 20 | RankMe | Sanahe, KLMN_GG, Jianbo Wang, Haojie Shi, moment02 | 0.708127 | ? | no writeup found | unknown |

Writeups found: **8 of 20** (ranks 1, 2, 3, 4, 5, 6, 7, 12). Rank 7's post exists on the forum though the API has no `solutionWriteUpUrl` for it.

Ties: ranks 9 (quenn) and 10 (tascj) share 0.711817; the API orders quenn first.

Spread: rank 1 to rank 20 spans only 0.008 accuracy, about 200 test rows if the forecasting set is ~25k (size UNVERIFIED).

## 2. Per-team synthesis

### Ranks 1-3: facts the bank lacks

None material; the bank covers all three in depth. Note from the 4th-place post's comments that sayoulala (8th here) congratulated the 4th-place author, and that 4th place credits sayoulala's LMSYS pipeline as his base.

### 4th: HKUST-GZ DSA KiMi Lab (HengweiDAI), solo

Source: https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/568522; inference https://www.kaggle.com/code/daihengwei/wsdm-vllm-gen-inference-pb714

**Hardware:** 40x A100 80GB from the KiMi Lab at HKUST (Guangzhou). Built on sayoulala's LMSYS pipeline.

**CausalLM instead of SequenceClassification.** A judge prompt (system message listing the input tags and "Return only one letter: A or B", user message with `<Query>`, `<Response_A>`, `<Response_B>` separated by dashes) ending with `' Choice: '`. The score is the logit of the "A" token vs the "B" token at the last position. He calls this "CoT as initial prompt control"; it is an instruction prompt, not generated reasoning. Left padding.

**Stage-wise loss design.**
- *Post-pretrain:* UltraFeedback + `C4AI-Community/multilingual-reward-bench`; cross-entropy of the A/B token **against the whole vocabulary**; both AB and BA versions in the same batch; 1024 tokens.
- *Fine-tune:* LMSYS (ties removed) + WSDM; cross-entropy **between A and B logits only**; AB order only; 2048 tokens.
- He reports the vocabulary-wide loss in pretraining "can raise the upper limit" but needs more steps.

**Distillation.** Teachers Athene-V2-Chat, nvidia/Llama-3.1-Nemotron-70B-Instruct-HF and Qwen2.5-72B-Instruct trained with the same procedure to produce soft labels for a 14B student (family not named in the post). Training the student for **2 epochs** helped: 5-fold best CV rose from 0.7184/0.7108/0.7134/0.7044/0.7085 (epoch 1) to 0.7209/0.7153/0.7145/0.7091/0.7115 (epoch 2), about +0.002.

**Inference.** vLLM, tensor_parallel 2, one generated token with `allowed_token_ids=[a_tok_id, b_tok_id]` plus `logprobs` (plain `logprobs=N` can miss both A and B in the top N). Quantisation: fp16 CV 0.714; **GPTQ 8-bit CV 0.713, public 0.703, private 0.714**; AWQ 4-bit CV 0.710, public 0.708, private 0.709. He picked GPTQ despite AWQ's better public score, and private confirmed it.

**Did not work:** a ranking model; a fourth teacher (Llama-3.3-70B; < 0.0005 difference); filtering the 5-10% of samples where six 70B OOF models disagreed most with the label, then training a 14B (no significant change).
**Not realised:** fusion with Gemma (single 0.713 might reach 0.715-0.716); an MoE-style router choosing per-question the most reliable model.

**Distinguishing idea.** Choosing the quantisation by CV rather than public LB (AWQ won public, GPTQ won private).

### 5th: Just doing (three posts by teammates)

Sources: detailed https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567856 (Chenyu Lin; code https://github.com/lcy80366872/kaggle-project/tree/main/wsdm, inference https://www.kaggle.com/code/linchenyu/5th-place-solution); roadmap /discussion/567724 (Albert Wang); overview /discussion/567617 (Myo Min Htet).

**Data.** WSDM 48k + LMSYS 55k + LMSYS additional 33k + nbroad open-model sets (8.5k + 13k + 5k); deduplicated and doubled by swap: **2 x 120k rows**. The overview also lists "dpo 40k".

**Model choice.** Tried Qwen2.5-14B-it, a Gemma-2 model and DeepSeek-R1-Distill-Qwen-14B. R1-Distill gave +0.001 CV but -0.003 LB (they suspect a one-token answer wastes its reasoning). Final: **Qwen2.5-14B-Instruct, CausalLM**, following the Eedi 1st-place approach (/eedi discussion 551688, conjuring92).

**Prompt.** Judge instruction listing eight criteria (completeness, accuracy, clarity, concision, examples, tone, reasoning, format), then the tagged query and responses, ending "Which response is more likely to be selected by a user? (A or B)". **Middle truncation** of the query and both responses with an ellipsis.

**Two-stage training.**
- Stage 1: WSDM + LMSYS 55k + 33k with hard labels; **LoRA fp16, r=64, alpha=128**; separate lrs **1e-5 for lora_A and 5e-5 for lora_B** (LoRA+ style, credited to conjuring92's Eedi post); batch 16; max length 4096; 1 epoch; loss only on the A/B token. ~12 h on **6x A100**. Then predict logits on the open-model sets.
- Stage 2: same data with smoothed labels ([0,1] -> [0.05, 0.95]) + open-model sets with stage-1 soft labels. ~16 h on 6x A100.

**Inference.** vLLM, fp16, prompt+output max 8192 with CPU offload (T4 memory), one generated token, compare A/B logits. **Post-processing** when the logits are within a threshold: a heuristic based on counts of numbers in the responses (Albert Wang's idea: when the LLM is undecided, the response with more numbers tends to be preferred).

**LB ladder (Chenyu Lin, public):** zero-shot 0.6254 -> mid-cut 0.6938 -> + LMSYS + 33k 0.6964 -> + swap 0.7038 -> + post-process 0.7039 -> + open-model soft labels **0.706812 (private 0.712)**. Other rows: left-cut 0.6907; random few-shot 0.6910; 0.7 DPO + 0.3 BCE 0.689; CoT multi-task 0.6973 (vs 0.6938 base).
Albert Wang's earlier ladder: Gemma-2-9B QLoRA 0.661 -> post-processing 0.665 -> **LightGBM ensemble** on handcrafted features 0.685 -> Qwen2.5-14B on vLLM 0.690 -> head-and-tail truncation 0.693 -> swap 0.697 -> + LMSYS data 0.703 -> soft-label swap 0.706.

**Did not work:** contrastive loss on mean-pooled query/answer vectors (CV only); prompt wording; random few-shot (CV up, LB down); max length 8192 in training (LB down); 2 epochs (LB down); DPO loss alongside BCE (LB down). A larger-model CoT + multi-task gave ~0.003 but was not adopted because they did not understand why. The overview says they did not use distillation or CoT in the final.

**Distinguishing idea.** A plain instruction-tuned CausalLM with middle truncation and soft labels matched teams using 70B teachers.

### 6th: Oh Your Mercy Never Fails

Source: https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567600

**Pseudo-labels (the core).** ~**560k** samples after removing short responses: the LMSYS 3rd-place team's generated data (conjuring92), lmsys-chat-1m prompts with **API-generated** responses from various models, and ~10 open-source DPO datasets (e.g. RLHFlow). Labeller: Gemma-2-9B fine-tuned on competition data (70B/72B labellers were slightly better but much slower). Training only on pseudo-labels, without the competition data, already gave **> 0.693 LB**. DPO datasets' original labels were useless for post-training but their prompts/responses were useful for pseudo-labelling.

**Distillation.** Teachers Qwen2.5-72B and Llama-3.3-70B trained on WSDM + LMSYS; losses KL, CE and their equal average. Distillation helped Gemma-2-9B measurably, not Qwen2.5-14B.

**Final training.** LoRA directly on **4-bit quantised** Qwen2.5-14B and Gemma-2-9B; max length 2500 (3072 gave nothing).

**Inference.** Qwen2.5-14B logits on all rows; Gemma-2-9B only on rows where Qwen was unsure; filled the 12 h forecasting budget.

**Multilingual.** Not a focus; prioritised the top five languages, especially English, where accuracy was weakest.

**Did not work:** TTA (no effect for sequence classification); LoRA merge (large unexplained degradation); multi-LoRA ensembles; length-based dynamic token allocation; CoT prompting; dynamic truncation (+0.003 at first, gone after other changes); other backbones; Gemma-2-27B teachers; mDeBERTa multiple-choice; fine-tuned GPT-3.5 as pseudo-labeller (10k samples) and 2-32-shot GPT-3.5 judging.

**Distinguishing idea.** Scale of pseudo-labelled data (560k) with a cheap 9B labeller.

### 7th: Team Turing, additions to the bank

Source: https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567589; inference https://www.kaggle.com/code/conjuring92/wsdm-7th-place-solution. Built in a **3-week sprint**. The bank has the cascade; missing:
- **Labelled data:** WSDM (45k), LMSYS (30k), `lmarena-ai/PPE-Human-Preference-V1` (10k).
- **Unlabelled data, pseudo-labelled by fine-tuned Qwen2.5-72B-Instruct and Llama-3.1-Nemotron-70B-Instruct:** nbroad open models (6k sampled), Skywork-Reward-Preference-80K-v0.2 (25k), allenai tulu-3-sft-olmo-2-mixture (30k), and **100k team-generated pairs** from WildChat-1M + lmsys-chat-1m prompts, **cluster-sampled to match the Arena Explorer topic distribution**.
- **Heads:** Gemma models used A/B token logits, trained with a combined LM + reward-modelling loss (arXiv 2406.10216, "Regularizing Hidden States..."); other models used a single linear `ForSequenceClassification` head ("kept the head simple" for generalisation).
- **Timing per 512 rows (vLLM):** gemma-2-9b fp16 @4096 6.5 min; Qwen2.5-14B fp16 @3072 10.5 min; Qwen2.5-32B GPTQ @2048 20 min; Mistral-Small-24B GPTQ @2048 16.4 min.
- Generalisation reasoning: larger LLMs on uncertain rows, mixed families and head types, 70B+ distillation over diverse prompts.

### 12th: monsaraida, solo

Source: https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567614

**Model.** Qwen2.5-14B-Instruct (beat gemma2-9b-it fp16 and DeepSeek-R1-Distill-Qwen-14B). **QLoRA, lr 2e-4, r=8**, 1 epoch, max length 2048.
**Two stages.** Stage 1 pretrain on dpo_44k + ut_157k (UltraFeedback); stage 2 on WSDM 48k + LMSYS 39k + add_23k (15k after processing), from shelterw's `wsdm-extra-dataset`.
**"Snip" marker.** Middle truncation that inserts `(snip 123 chars)` with the actual count omitted, which improved accuracy.
**Swap.** Trained with A/B swapped copies; doubling time for a slight gain; swapped pairs **must be in the same batch** to help.
**CV.** 5-fold, evaluating fold 0 only; final retrained on all data.
**Auxiliary head.** Classify the model name of A and B with weight 0.1 (language-ID auxiliary did not help).
**Inference.** Sort by descending token count, alternate rows across the two GPUs; second pass averages with swapped input only for the **40% of rows closest to 0.5**.
**Note.** Discussed strategy nightly with ChatGPT o1, which also found bugs.

### No-writeup teams

- **8 sayoulala:** no writeup. Re-opened his LMSYS winning code mid-competition for fairness (https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/554766); the 4th place says it built on that pipeline.
- **9 quenn:** no writeup found.
- **10 tascj:** no writeup found. His LMSYS code and checkpoints were the starting point for 1st and 2nd (bank).
- **11 Evalthon:** no writeup found.
- **13 x-creative, 14 Takoi, 16 Tilbur, 17 yukichi0403, 18 zxc123cc:** no writeup found.
- **15 KYRL:** no writeup. Member lhwcv ran the "CV vs LB thread" (https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/552368): Gemma-2-9b-it, bf16 training (fp16 inference on T4 cost 0.7011 -> 0.699), 5-fold CV 0.694-0.701, LB 0.687-0.694 at 3072 tokens without TTA, all-data model 0.695 (0.697 with origin+flip TTA); bnb 4-bit inference at that point; added LMSYS and dpo44k; prompt `<context> ... <assistant_a> ... <assistant_b>`; time-guard that stops inference at 4.75 h. He judged a ±0.005 CV-LB fluctuation normal.
- **19 Shaomu & Shiyi:** no writeup. Shaomu asked why his public rank rose without new submissions (/discussion/556013).
- **20 RankMe:** no writeup. Member Jianbo Wang asked 12th place about the model-name auxiliary head.

## 3. Technique frequency matrix

Columns with writeups only (ranks 8-11, 13-20 have none; KYRL's partial forum data is not a final solution).

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 12 |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-14B in final | Y | N | Y | Y (14B, family not named) | Y | Y | Y (cascade) | Y |
| Gemma-2-9B in final | N | Y | N | N | N | Y | Y | N |
| Other final members | N | ArmoRM-Llama3-8B | Phi-4 | N | N | N | Qwen2.5-32B / Mistral-24B | N |
| CausalLM A/B-token head (vs SeqCls) | N | N | Y | Y | Y | N | Y (Gemma) | N |
| vLLM at inference | N (custom unpadded) | N (tascj engine) | Y | Y | Y | ? | Y | ? |
| LMSYS 55k / 33k as extra labelled data | Y | ? | Y | Y | Y | Y | Y | Y |
| Reward / preference pretraining stage | Y | Y (LMSYS checkpoints) | Y | Y | N | N | N | Y |
| LLM-generated responses (synthetic pairs) | Y (50k) | Y (95k) | N | N | N | Y (API) | Y (100k) | N |
| Pseudo / soft labels | Y | Y | Y (self) | Y | Y | Y (560k) | Y | N |
| Distillation from >= 70B teachers | Y | N | Y | Y | N | Y | Y | N |
| Full FT vs LoRA | Full | Full | LoRA | ? | LoRA | QLoRA | ? | QLoRA |
| Middle / proportional truncation | Y | Y | ? | ? | Y | ? | ? | Y (snip) |
| A/B swap in training | Y | Y | ? | Y (pretrain) | Y | ? | ? | Y |
| Swap TTA on uncertain rows only | Y (33%) | Y | Y (25%) | N | N | N | N (cascade) | Y (40%) |
| Uncertainty-gated second model | Y | Y | Y (Phi-4) | N | N | Y | Y | N |
| 8-bit or lower quantisation | ? | ? | Y (auto-round) | Y (GPTQ 8) | N (fp16 + offload) | Y (4-bit) | Y (GPTQ for 24B/32B) | Y (4-bit) |
| Rule / tie post-processing | N | N | N | N | Y | N | N | N |

Counted summary over the 8 writeups:
- **7 of 8** used Qwen2.5-14B in the final submission (all but 2nd). Gemma-2-9B is in 3 of 8.
- **7 of 8** used soft or pseudo labels; **5 of 8** distilled from 70B+ teachers (1, 3, 4, 6, 7); 3rd found self-distillation as good as 72B.
- **4 of 8** used a CausalLM A/B-token formulation (3, 4, 5, 7-Gemma); 5 of 8 used vLLM.
- **6 of 8** spent extra compute only on uncertain rows (swap TTA or a second model).
- **7 of 8** added LMSYS data (2nd uses LMSYS-trained checkpoints).
- **4 of 8** generated their own response pairs with LLMs (1, 2, 6, 7).
- **1 of 8** used rule post-processing (5th).

## 4. Consensus vs divergence

**Table stakes:** Qwen2.5-14B (or LMSYS-trained Gemma-2-9B) as the main model; LMSYS 55k + 33k as extra labels; both A/B orders in training; soft labels; length-sorted inference split over two T4s; spending leftover time on uncertain rows. LMSYS winners' open code (sayoulala, tascj) was the starting point for several teams (1st, 2nd, 4th).

**Top 5 vs 6-20:** the visible gap is small (0.004 between 1st and 6th) and the top 7 overlap heavily in technique. What only the top 3 share: an *engineered inference engine* (1st and 2nd: tascj's unpadded kernels; 3rd: vLLM CausalLM with prefix caching and length-quantile weighting) that let them run a 14B model plus a second pass within the limits; and careful label cleaning (1st hand-corrected disagreements; 3rd self-distilled). 5th shows a simpler path (no distillation) reaching 0.7126; 12th shows QLoRA r=8 single model reaching 0.7109.

**Contradictory findings:**
- **TTA:** 6th: "TTA showed no measurable impact" (https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567600); 12th, 1st, 3rd found swap TTA on uncertain rows helpful (/discussion/567614, /discussion/569902, /discussion/567584); KYRL's lhwcv found origin+flip helpful on CV and LB (/discussion/552368).
- **Distillation:** core for 1st, 4th, 6th-Gemma; "no significant effect" on Qwen2.5-14B for 6th; 3rd says self-distillation matched 72B; 5th did not distil at all; 2nd skipped 72B for compute.
- **More epochs:** 4th gained ~0.002 CV from a second epoch; 5th saw LB drop with 2 epochs.
- **Longer context:** 5th's 8192-token training dropped LB; 6th saw no gain past 2500; 1st/3rd used long inputs (3rd ~6144).
- **CoT:** 5th saw +0.003 from CoT multi-task (not adopted); 3rd +0.001-0.002 but timeouts; 6th and 1st (GPT-4o critiques) found it unhelpful.
- **DeepSeek-R1-Distill-Qwen-14B:** 5th +0.001 CV / -0.003 LB; 12th worse than Qwen2.5-14B; 2nd used R1-Distill only as a response generator.
- **LoRA merging:** 1st linearly merged full-FT models successfully; 6th saw severe degradation from LoRA merge.
- **Auxiliary model-name head:** used by 12th (weight 0.1); not reported elsewhere.

## 5. Aggregated "what did not work"

| Item | Reported by |
|---|---|
| CoT / critique-augmented training or prompting | 1st (GPT-4o critiques), 3rd (timeouts), 6th; 5th not adopted |
| Layer pruning of larger models | 1st |
| Qwen2.5-72B AWQ under vLLM for second pass (too slow) | 1st |
| Order-invariant embeddings | 1st |
| 72B distillation (compute), Skywork RMs as labellers/backbones, Skywork 80K data, Bradley-Terry loss, multi-loss weighting | 2nd |
| nbroad v3 open-model set (too noisy) | 2nd |
| lr 1.4e-4 for Qwen2.5-14B (collapse) | 3rd |
| Ranking model; 4th teacher; filtering high-disagreement samples | 4th |
| AWQ 4-bit (better public, worse private than GPTQ 8-bit) | 4th |
| Contrastive auxiliary loss; random few-shot; DPO + BCE; 8192 training length; 2 epochs | 5th |
| DeepSeek-R1-Distill-Qwen-14B | 5th, 12th |
| TTA; LoRA merge; multi-LoRA ensemble; dynamic token allocation; dynamic truncation (faded) | 6th |
| Gemma-2-27B teachers | 6th |
| mDeBERTa; fine-tuned GPT-3.5 labeller; few-shot GPT-3.5 judge | 6th |
| DPO datasets' original labels as training labels | 6th |
| Language-ID auxiliary task | 12th |
| Gemma-2-9B vs Qwen2.5-14B as single model | 5th, 12th, 6th (Gemma kept only as helper) |

## 6. Shake-up note

The API's public scores are wiped (all `0.000000`), so no rank-movement table is possible. Self-reported public -> private: 2nd 0.709 -> 0.716; 4th 0.703 -> 0.714 (GPTQ) vs 0.708 -> 0.709 (AWQ); 5th 0.7068 -> 0.7126. Private scores run ~0.005-0.01 above public across these teams, consistent with the forecasting set being easier or differently distributed than the training-phase public set. The 4th-place quantisation pair is the cleanest evidence that the public set mis-ranked submissions. 7th says it designed for generalisation (larger models on uncertain rows, mixed families, simple head) and was "pleased to see the winning solutions generalize well".

## 7. UNVERIFIED

- 4th place's student model family (post says "a 14B model"; the dataset class is named `qWenSFTDataset`, suggesting Qwen2.5-14B).
- 4th's LoRA vs full FT and hyperparameters; 6th's and 7th's training hyperparameters (not stated).
- 5th's exact post-processing threshold (the roadmap mentions 0.05 logit difference; a commenter tried 0.05).
- Size of the forecasting-phase test set (so the row-count reading of the 0.008 spread is approximate).
- Approaches of ranks 8-11 and 13-20: no writeup found among all 123 forum topics, the API writeup URLs, or one web search.

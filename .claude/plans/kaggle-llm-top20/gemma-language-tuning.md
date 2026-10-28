# Google: Unlock Global Communication with Gemma (2024-25): all prize winners

- **Slug:** `gemma-language-tuning`
- **URL:** https://www.kaggle.com/competitions/gemma-language-tuning
- **Dates:** 2024-10-02 to 2025-01-15 (competition API). Review-delay update: https://www.kaggle.com/competitions/gemma-language-tuning/discussion/562683 ; winners: https://www.kaggle.com/competitions/gemma-language-tuning/discussion/575770
- **Entrants:** "hundreds of entries" (host); 7,019 users joined (bank). Notebook-judged: no Writeup objects, no track/prize API data (tr json is empty).
- **Prize pool:** $150,000 as 5 x $30,000 plus a physical trophy (bank).
- **Brief and rubric (bank):** adapt **Gemma 2** to one of 73 eligible languages or a cultural context; public notebook covering dataset creation, fine-tuning and inference/evaluation; model published on Kaggle Models. Pass/fail gates (Compliant, Topical, Open, eligible Language), then Technical 10 (few-shot, RAG and/or fine-tuning used efficiently), Descriptive 10, Useful 10, Robust 10. Host reference notebook: Gemma 2 for Japanese (/discussion/541342; not on disk, not needed for winners).
- **Bank coverage:** `kaggle-llm-writeup-bank-part4.md` H2 "## 7." covers all five winners at moderate depth (Swahili LoRA config and data profiling; Kyara data pipeline and headline scores; ArGemma task adapters; Italian Spectrum + DPO; Ancient Chinese 252k mix). Below: only new facts per winner, re-read from the notebooks already on disk in `/private/tmp/claude-501/kaggle-bank-p4/nb_*.md` (all five pulled; no WebFetch needed).

## Winners table (host 575770, ranked places)

| Place | Prize | Team | Project | Language | Notebook | Model / code links |
|---|---|---|---|---|---|---|
| 1st | $30k | Alfaxad Eyembe, Mrina Eliseus (Mtenga) | Gemma 2 Swahili | Swahili | [nb](https://www.kaggle.com/code/alfaxadeyembe/introducing-gemma-2-swahili) | [Kaggle Models](https://www.kaggle.com/models/alfaxadeyembe/gemma-2-swahili), [GitHub](https://github.com/Alfaxad/gemma2-swahili-models), [HF](https://huggingface.co/Alfaxad) |
| 2nd | $30k | Justin Yang (zake7749) | Kyara: Retrieval Augmentation for LLM Fine-Tuning | Traditional Chinese | [nb](https://www.kaggle.com/code/zake7749/kyara-retrieval-augmentation-for-llm-fine-tuning) | see section |
| 3rd | $30k | Taha Alselwi | ArGemma: Fine-Tuning Gemma for Arabic | Arabic (MSA) | [nb](https://www.kaggle.com/code/tahaalselwii/fine-tuning-gemma-for-arabic-argemma) | see section |
| 4th | $30k | Stefano Fiorucci (anakin87) | Post-Training Gemma for Italian and Beyond | Italian | [nb](https://www.kaggle.com/code/anakin87/post-training-gemma-for-italian-and-beyond) | see section |
| 5th | $30k | Shiner (judith007) | Ancient Chinese Expert | Classical Chinese | [nb](https://www.kaggle.com/code/judith007/ancient-chinese-expert-gemma2-chatgpt/notebook?scriptVersionId=216814746) | see section |

All five places paid the same $30k; the host still ranks them 1st to 5th. No honorable mentions announced.

## Per-winner synthesis

### 1st. Gemma 2 Swahili, Alfaxad Eyembe and Mrina Eliseus
New facts beyond the bank (from the notebook):
- **Suite and methods:** Gemma2-2B-Swahili-Preview is **continued pretraining** of 2B on Swahili monolingual text; 2B-IT and 9B-IT are **LoRA** (r=8, alpha=32, dropout 0.1); **27B-IT is QLoRA** (4-bit NF4, double quantisation, bf16 compute), adapters merged with `merge_and_unload()`. Stack: Hugging Face transformers + PEFT + bitsandbytes + accelerate.
- **Data:** pretraining slice from Lelapa AI's **Inkuba-Mono Swahili** (~58M sentences overall; 1M-example slice, max_length 256). Instruction data: **67,017 Swahili pairs from Bactrian-X** (16.27M tokens, avg 242.83 tokens/example), max_length 2048. Both are existing public datasets, not self-built.
- **Hardware and schedule:** Google Colab Pro, **one A100 40GB**. 2B-IT: batch 2 x accum 32, 2,500 steps, LR 2e-4, ~7.5 h, ~15 GB VRAM; 9B-IT: 400 steps, LR 2e-4, ~8 h, ~25 GB; 27B-IT 4-bit: batch 1 x accum 64, **150 steps**, LR 1.5e-4, ~9 h, ~35 GB; preview 2B: 2,500 steps, ~8 h. Training outside Kaggle, reproduced in GitHub notebooks.
- **Tokenizer:** unchanged Gemma tokenizer (no vocabulary extension mentioned).
- **Evaluation (small samples, base vs fine-tuned):** Swahili sentiment (Neurotech/Jinamizi, 3,925 texts; **200-sample** stratified eval): 2B 49.00 to 66.50%, 9B 83.42 to 86.50%, 27B 87.50 to 88.50%. **Swahili MMLU (120-question sample)**: 2B 15.00 to 34.17%, 9B 43.33 to 55.83%, 27B 20.00 to 54.17%. Translation: **20 Wikimedia en-sw pairs**, BLEU-1..4 and **chrF** (figures only; the notebook itself says 9B shows "slight translation regression" and 2B "mixed"). Plus qualitative side-by-sides (creative writing, explanations) vs gemma-2-2b-it. (notebook)
- **Note:** the bank's "MMLU up to 57.89%, sentiment up to 90.00%" comes from the notebook's summary bullets; the notebook's own tables top out at 55.83% and 88.50%. Report the table numbers.
- **Why it won (inference; host gave no reasons):** four published sizes + docs + a benchmark report covers Useful and Robust; but eval samples are tiny (120/200/20).

### 2nd. Kyara: Retrieval Augmentation for LLM Fine-Tuning, Justin Yang (zake7749)
[Notebook](https://www.kaggle.com/code/zake7749/kyara-retrieval-augmentation-for-llm-fine-tuning) · [HF model gemma-2-2b-it-chinese-kyara-dpo](https://huggingface.co/zake7749/gemma-2-2b-it-chinese-kyara-dpo) · [GitHub](https://github.com/zake7749/kyara)

New facts beyond the bank (from the notebook):
- **Tuning method is full-parameter, not LoRA** (the bank's UNVERIFIED "LoRA rank" question is answered: there is none). SFT with **Axolotl**: base `google/gemma-2-2b-it`, dataset `zake7749/kyara-zh-sample-1M` (chat template, train on assistant turns only), sequence_len 4096 with **sample packing**, micro-batch 1 x grad-accum 32 x **8 GPUs = global batch 256**, **3 epochs**, LR 2e-5 cosine, warmup 5%, `paged_adamw_8bit`, bf16, **DeepSpeed ZeRO-3**, flash attention on (noted as needed for Gemma stability), gradient checkpointing.
- **DPO with LLaMA-Factory**, full fine-tuning from `kyara-lite-sft-1M`: LR 3e-7 constant, 1 epoch, cutoff 2048, beta 0.1, **`pref_ftx` 0.85 (SFT loss mixed into DPO)**, sigmoid loss, global batch 256 on 8 GPUs, ZeRO-3 with offload. Dataset `zake7749/kyara-dpo-zh`.
- **Preference pair sources:** (1) reward model **Skywork-Reward-Gemma-2-27B-v0.2** scoring responses from a sampled model pool (best = chosen, worst = rejected, near-ties dropped), served via SGLang + liteLLM, pipelined with distilabel; (2) **pairwise LLM judges (GPT-4o, Gemini-Flash-2.0, DeepSeek V3) with A/B position swap for consistency**; (3) environment-interactive filtering of sampled outputs.
- **Tokenizer:** unchanged (`AutoTokenizer`, `<end_of_turn>` EOS).
- **Training hardware:** 8 GPUs, outside Kaggle (type not stated in the extracted parts); inference recommended on Kaggle P100.
- **Evaluation (the most extensive among the five):** ground-truth benchmarks via predicted probabilities or answer extraction (**TMMLU+** fold-wise, MMLU-Redux, GSM8K, MATH-L5, CRUX, ZebraLogic via ZeroEval) and LLM-as-judge benchmarks (AlpacaEval/-LC, Arena-Hard, **MT-Bench-TW**, MT-Bench). Kyara-1.5-2B-it vs Gemma-2-2b-it: TMMLU+ 42.57 vs 36.90, MATH-L5 8.88 vs 4.3, Arena-Hard 22.60 vs 19.4, MT-Bench 8.28 vs 8.17; also beats ChatGPT-3.5-1106 on AlpacaEval-LC (35.35 vs 19.30). Kyara-2.5-9B-it vs Gemma-2-9b-it: TMMLU+ 59.87 vs 54.77, GSM8K 90.60 vs 87.41, Arena-Hard 41.1 vs 33.6, but MMLU-Redux 72.57 vs 72.82 (slightly lower). (notebook)
- **Extras:** qualitative samples per category (open QA, advice, professional knowledge, math, summarisation, writing, RAG with distractors) and a **naive RAG agent** demo (query expansion, digest, replan, reply); a limitations section (safety, hallucination, retrieval quality).

### 3rd. ArGemma: Fine-Tuning Gemma for Arabic, Taha Alselwi
[Notebook](https://www.kaggle.com/code/tahaalselwii/fine-tuning-gemma-for-arabic-argemma) · Kaggle model input `argemma` (LoRA `.h5` files, version 11, path `/kaggle/input/argemma/other/default/11/`) · an `ArGemma.mp4` demo video embedded in the notebook · no GitHub.

New facts beyond the bank (from the notebook):
- **Goals:** Arabic-English translation; creative generation (stories, dialogues, letters, summaries); and **bridging Classical Arabic (Quran, poetry, proverbs) and Modern Standard Arabic**.
- **Stack and hardware:** **KerasNLP** `GemmaCausalLM.from_preset("gemma2_instruct_9b_en")` (2B also trained), Keras distribution API with a **(1, 8) device mesh sharding weights across 8 TPU cores** (Kaggle TPU; no model parallelism for 2B). **LoRA rank 8** via `backbone.enable_lora(rank=8)`, sequence length 512, AdamW LR 1e-4, weight decay 0.01 (bias and scale excluded), batch size 1, 1 to 3 epochs per task, stopping while sparse categorical accuracy stays under ~70% "to avoid the model hallucinating and generating words from another language". (Answers the bank's UNVERIFIED LoRA rank.)
- **Adapters per task family** because of compute limits: `tr` translation, `sm` summarisation, `sd` stories/dialogues/letters, `br` Classical-MSA bridging; saved separately with `save_lora_weights`.
- **Hybrid "br" method:** a custom Arabic semantic-search model (**Arabert-all-nli-triplet-Matryoshka** bi-encoder via Sentence Transformers) embeds a parallel Classical/MSA knowledge base; for each training query, **top-3 retrieved examples are prepended as context and the model is fine-tuned on those context-rich inputs**; at inference the same retrieval builds a few-shot prompt. RAG is used in both training and inference.
- **Tokenizer:** unchanged.
- **Evaluation:** **BLEU** for translation on 1,000 test pairs with a custom weighting **(0.7, 0.2, 0.1, 0)** (unigram-heavy, not standard BLEU-4): 2B 0.3094 to 0.7022, 9B 0.4788 to 0.7714 (before vs after). **ROUGE-1** for summarisation with custom Arabic functions: F1 2B 0.0099 to 0.5134, 9B 0.0838 to 0.4505. Stories/dialogues and Classical-MSA bridging assessed qualitatively by **native speakers and language experts** (no numbers). (notebook) The non-standard BLEU weights inflate scores relative to sacreBLEU; flag if quoting.

### 4th. Post-Training Gemma for Italian and Beyond ("Gemma Neogenesis"), Stefano Fiorucci (anakin87)
[Notebook](https://www.kaggle.com/code/anakin87/post-training-gemma-for-italian-and-beyond) · [HF model gemma-2-2b-neogenesis-ita](https://huggingface.co/anakin87/gemma-2-2b-neogenesis-ita) (also as Kaggle model input `gemma-2-2b-neogenesis-ita`) · [GitHub gemma-neogenesis](https://github.com/anakin87/gemma-neogenesis) · [HF Space demo](https://huggingface.co/spaces/anakin87/gemma-2-2b-neogenesis-ita)

New facts beyond the bank (from the notebook):
- **Motivation from observed failures:** gemma-2-2b-it sometimes switches to English mid-answer and invents non-existent Italian words. The author argues Gemma 2's **256k vocabulary** makes language adaptation feasible **without tokenizer changes** (none made).
- **SFT data:** `efederici/capybara-claude-15k-ita` (multi-turn, Claude-translated/generated) plus the author's **`fine-instructions-ita-70k`**: FineTome-100k single-turn conversations, MinHash-deduplicated (threshold 0.9), instructions and responses each **translated and then judged by Llama-3.1-70B-Instruct** (HF Serverless Inference API), low scorers removed.
- **Spectrum:** SNR scan of gemma-2-2b-it, **top 25% of layers** unfrozen (all layer types scanned); the rest frozen. Framed as partial full fine-tuning that limits catastrophic forgetting, not LoRA.
- **SFT (TRL `SFTTrainer`):** max_seq_length 1536, packing, 1 epoch ("no benefit beyond 1"), cosine, warmup 0.2, **LR 5e-6** (2e-5 hurt when continuing from an instruct model), batch 2 x accum 8 = 16, bf16/tf32, **single A6000 48GB, ~5 h**.
- **DPO data:** `efederici/evol-dpo-ita` (20K) **re-ranked by Llama-3.1-70B as judge**, an Italian subset of `ruggsea/wsdm2024-cot-dataset` (269), plus **on-policy pairs**: prompts from `sapienzanlp/it-Magpie-Llama-3.1-Pro-300K-Filtered-easy`, two samples from the SFT model, ranked by **Llama-3.3-70B-Instruct**, with a strong alternative generated if both are poor (target 25K).
- **DPO (TRL `DPOTrainer`, Spectrum layers again):** beta 0.1, sigmoid loss, LR 5e-6 (5e-7 and 1e-6 changed nothing), 1 epoch, batch 1 x accum 16, max_prompt 350, max_length 1260.
- **Evaluation:** lm-evaluation-harness on Open ITA LLM Leaderboard tasks (MMLU_IT 5-shot, ARC_IT, HellaSwag_IT), and **submitted to the public leaderboard**. Base 47.65 / 40.03 / 54.69 (avg 47.46); SFT 47.77 / 41.15 / 55.66 (48.19); **DPO 48.03 / 40.46 / 56.97 (48.49)**. Gains are about +1 point average; the author is candid that these metrics barely capture preference tuning and on-policy data gave only "moderate" gains. Plus 11-prompt side-by-side vibe checks via two identical HF Spaces. (notebook)
- **Application:** a **Haystack Web-RAG** pipeline (DuckDuckGo search, prompt builder, local generator) answering Italian culture questions with cited URLs.

### 5th. Ancient Chinese Expert, Shiner (judith007)
[Notebook](https://www.kaggle.com/code/judith007/ancient-chinese-expert-gemma2-chatgpt/notebook?scriptVersionId=216814746) · [self-made dataset ancient-chinese-english-translate](https://www.kaggle.com/datasets/judith007/ancient-chinese-english-translate) · Kaggle model input `judith007/gemma-2/transformers/gemma-2-9b` (variant `gemma-2-9b-ancient-chinese`) · no GitHub.

New facts beyond the bank (from the notebook):
- **Language/target:** Classical (ancient) Chinese, written in Simplified script, plus a **new task: Classical Chinese to and from English**, which the author says translators handle poorly. Baseline audit of gemma-2-2b first (translation, poems, couplets, idioms), noting it often drifts into English.
- **Data (252k):** FireFly and COIG-CQIA classical subsets; the self-made translation set built by chaining classical-to-modern Chinese pairs and translating the modern side into English; a **CoT translation dataset**; an LLM identity set. Plus a separate **30k idiom corpus for RAG**.
- **Tuning method:** **LoRA via HF PEFT** on all attention and MLP projections (`q,k,v,o,gate,up,down`). **The real run (outside Kaggle, 1 x A800): r=128, alpha=256, dropout 0.1, max_length 768, 3 epochs, batch 1 x accum 64, LR 1e-5.** The in-notebook demo cell uses a toy config (r=8, alpha=16, 2 epochs, LR 1e-4, fp16); quote the A800 values. (Answers the bank's missing rank.) 9B is loaded in **int8** (bitsandbytes) for inference because of GPU memory.
- **Tokenizer:** unchanged.
- **Evaluation:** **ACLUE** (15 tasks, five categories). Fine-tuned 9B: Lexical 44.13, Syntactic **58.20**, Semantic 45.30, Inference 38.43, Knowledge **39.60**; it beats ChatGPT (34.53 / 43.00 / 36.40 / 41.23 / 33.61) in four of five categories. **But base gemma-2-9b scores higher on three of five** (Lexical 45.80, Semantic 51.40, Inference 47.22); fine-tuning only wins Syntactic and Knowledge. Fine-tuned 2B (31.47 / 34.00 / 29.10 / 35.56 / 30.15) sits below ChatGPT on most. The bank's "beats ChatGPT" is accurate but omits that the tune regressed vs its own base on 3/5 categories. (notebook)
- **Extras:** FAISS **RAG over the 30k idiom set** for idiom explanation; **few-shot prompting** for idiom sentiment classification; a robustness check that the tuned model still answers in English, Japanese, Korean and French (the rubric's "Robust" criterion).

## Feature matrix

Columns: 1 Swahili, 2 Kyara, 3 ArGemma, 4 Italian Neogenesis, 5 Ancient Chinese.

| Row | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Target language | Swahili | Traditional Chinese | Arabic (MSA + Classical) | Italian | Classical Chinese (+ English translation) |
| Gemma size(s) | 2B, 9B, 27B (+2B CPT) | 2B, 9B | 2B, 9B | 2B | 2B, 9B |
| Tuning method | CPT (2B) + LoRA r8 (2B/9B) + QLoRA 4-bit (27B) | full FT SFT + full FT DPO | LoRA r8 (KerasNLP), per-task adapters | Spectrum top-25% layers SFT + DPO | LoRA r128/alpha256 |
| Fine-tuned (Y/N) | Y | Y | Y | Y | Y |
| Preference tuning (DPO) | N | Y | N | Y | N |
| Tokenizer changes | N | N | N | N (relies on 256k vocab) | N |
| Own dataset built | N (Inkuba-Mono, Bactrian-X) | Y (retrieval-synthesised SFT + DPO) | partial (curated ~10 public sources + RAG KB) | Y (judge-filtered translation, on-policy DPO) | Y (translation, CoT, idiom RAG sets) |
| Synthetic / LLM-generated data | N | Y | N | Y | Y (translation chain) |
| LLM-as-judge in data pipeline | N | Y (pairwise, swap-checked) + reward model | N | Y (Llama-3.1/3.3-70B) | N |
| RAG / few-shot | N | Y (data synthesis + RAG agent demo) | Y (RAG in train + inference) | Y (Haystack web-RAG app) | Y (idiom RAG + few-shot) |
| Eval metric | accuracy (MMLU-sw, sentiment), BLEU-1..4, chrF | TMMLU+, MMLU-Redux, GSM8K, MATH, AlpacaEval, Arena-Hard, MT-Bench(-TW) | BLEU (custom weights), ROUGE-1/L, native-speaker review | MMLU_IT, ARC_IT, HellaSwag_IT (lm-eval-harness + public leaderboard) | ACLUE |
| Eval rigour | small quantitative (120/200/20 items) | benchmark with baseline | small quantitative + human | benchmark with baseline, public leaderboard | benchmark with baseline |
| Human eval | qualitative side-by-sides | qualitative samples | Y (native speakers, experts) | 11-prompt vibe check | qualitative samples |
| Training hardware | 1 x A100 40GB (Colab Pro) | 8 GPUs (type not stated) | 8 TPU cores (Kaggle) | 1 x A6000 48GB | 1 x A800 |
| Trained on Kaggle | N | N | Y | N | N |
| Framework | HF transformers + PEFT + bitsandbytes | Axolotl + DeepSpeed ZeRO-3; LLaMA-Factory | KerasNLP / Keras 3 | TRL SFT/DPOTrainer + Spectrum | HF transformers + PEFT |
| Published model (Kaggle/HF) | Y (Kaggle + HF) | Y (HF) | Y (Kaggle) | Y (HF + Kaggle) | Y (Kaggle) |
| Video | N | N | Y (embedded mp4) | N | N |
| Live demo | N | N | N | Y (HF Space) | N |
| Open code (GitHub) | Y | Y | N | Y | N |
| On-device / offline | N | N | N | N | N (int8 9B only for memory) |
| Agentic | N | partial (naive RAG agent) | N | N | N |

**Counts (5 winners):** 5/5 fine-tuned Gemma 2; 0/5 changed the tokenizer; 3/5 used LoRA/QLoRA, 2/5 trained more than adapters (Kyara full FT, Italian Spectrum layer subset); 2/5 added DPO (2nd and 4th); 4/5 built at least partly new data; 4/5 report an external public benchmark against the base model; 2/5 use BLEU-family metrics (1st, 3rd), **none uses chrF as the headline** (Swahili reports chrF only in figures); 4/5 trained off Kaggle; 4/5 combine tuning with RAG or few-shot (the rubric's Technical criterion names all three); 3/5 have GitHub; 1/5 has a video; 1/5 a live demo; 0/5 on-device.

## Consensus vs divergence

**Table stakes:** a published model artifact; a before/after comparison against the stock Gemma 2 instruct model; a notebook that walks dataset creation, training and inference (the "Descriptive" criterion); and some RAG or few-shot component alongside tuning (4/5), which maps directly onto the rubric's "few-shot, RAG and/or fine-tuning" wording.

**What separated 1st/2nd from 4th/5th (all $30k, but ranked):** breadth of artifacts and evaluation. 1st shipped four sizes up to 27B with a benchmark report; 2nd shipped the most rigorous evaluation (ten benchmarks, LLM-judge suites, 2B and 9B) and a reusable data-synthesis method. The 4th-place Italian notebook has arguably the cleanest method and the only independent public-leaderboard submission, but a single 2B model and a +1-point average gain. 5th has a compelling niche and new task (classical-to-English) but its 9B tune regressed vs base on 3 of 5 ACLUE categories.

**Contradictory choices:**
- **Adapters vs updating base weights:** LoRA r=8 with LR 2e-4 to 1e-4 (Swahili, ArGemma) vs LoRA r=128 at LR 1e-5 (Ancient Chinese) vs full fine-tuning at LR 2e-5 (Kyara) vs Spectrum layer subsets at LR 5e-6, with the explicit warning that 2e-5 hurt when continuing from an instruct model ([Italian](https://www.kaggle.com/code/anakin87/post-training-gemma-for-italian-and-beyond), [Kyara](https://www.kaggle.com/code/zake7749/kyara-retrieval-augmentation-for-llm-fine-tuning)).
- **Epochs:** Italian found no benefit beyond 1 epoch; Kyara and Ancient Chinese trained 3; ArGemma capped training accuracy at ~70% to avoid cross-language word leakage ([ArGemma](https://www.kaggle.com/code/tahaalselwii/fine-tuning-gemma-for-arabic-argemma)).
- **Data source:** existing public corpora only (Swahili: Inkuba-Mono + Bactrian-X) vs heavy LLM synthesis and judge filtering (Kyara, Italian). Both reached the top two places, so data provenance alone did not decide ranking.
- **Shared failure observed:** Italian, Ancient Chinese and ArGemma all name the same base-model failure, **unwanted switching into English** (or generating foreign words), as a primary target of tuning.

## UNVERIFIED
- The host thread gives places only; no judges' comments or scores per criterion exist, so "why it won" lines are inference.
- All benchmark numbers are authors' own (notebook), except the Italian DPO model, which the author says is on the public Open ITA LLM Leaderboard (not checked).
- Swahili: summary bullets claim MMLU up to 57.89% and sentiment up to 90.00%, which do not match the notebook's own tables (max 55.83% and 88.50%); translation BLEU/chrF values are only in figures.
- Kyara: GPU type for the 8-GPU runs not stated in the notebook sections read; hardware for 9B training not stated.
- ArGemma: BLEU uses non-standard weights (0.7, 0.2, 0.1, 0); scores are not comparable to sacreBLEU. Exact training-set sizes per task not extracted.
- Ancient Chinese: the notebook's in-cell demo hyperparameters differ from the reported A800 run; the reported run's values are used here.
- Host reference notebook /discussion/541342 (Japanese) not on disk and not fetched; not needed to identify winners.

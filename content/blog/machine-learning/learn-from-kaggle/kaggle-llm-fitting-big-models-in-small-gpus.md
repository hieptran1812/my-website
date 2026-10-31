---
title: "Fitting big models in small GPUs: what eight Kaggle LLM competitions taught about the notebook box"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "Distillation, 8-bit and 4-bit quantisation, vLLM, KV-cache reuse, layer-by-layer 70B inference and time managers, compared across LMSYS, WSDM, Science Exam, AIMO 1 and 3, Eedi, MAP and Jigsaw, with every memory number derived from one formula."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "quantization", "knowledge-distillation", "vllm", "kv-cache", "inference-optimization", "gpu-memory"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 13
---

> **TL;DR.** From 2023 to 2025 a Kaggle LLM competition ran on two 16 GB T4s for 9 to 12 hours, internet off. Winners treated that box as the problem: 70B models moved into training and a distilled 9B or 14B shipped (LMSYS, WSDM, MAP); 8-bit when speed mattered, 4-bit when size did, chosen by CV (WSDM 4th); shared prefixes encoded once (Science Exam 1st, MAP 1st); layer-by-layer streaming when nothing else fit (Science Exam 5th, Eedi 13th, MAP 1st); and time managers in AIMO. One line, `GB = params x bits / 8`, predicts most of it.

Every Kaggle LLM competition hands you a small apartment and asks you to move a piano in. The memory chart below is the floor plan: four model sizes at three precisions against the boxes Kaggle has handed out.

## The arithmetic of the box

![GPU memory bars: 9B, 14B, 32B and 72B models at 16, 8 and 4 bits, against lines for one T4 at 16 GB, two T4s at 32 GB and one H100 at 80 GB](/imgs/blogs/kaggle-llm-fitting-big-models-in-small-gpus-2.webp)

Weight memory is parameters times bytes per parameter. With parameters in billions and bits per weight $b$:

$$
\text{GB}_{\text{weights}} = N_{\text{params}}\,[\text{B}] \times \frac{b}{8}
$$

A nominal 9B needs $9 \times 2 = 18$ GB at 16-bit, already more than one T4; 9 GB at 8-bit, 4.5 GB at 4-bit. A 14B is 28, 14 and 7 GB. A 32B at 4-bit is ${32 \times 4 / 8 = 16}$ GB, the entire T4 with nothing left for activations, which is why 32B rerankers in Eedi were split across both cards. A 72B at 4-bit is 36 GB, larger than the 32 GB of both T4s together. Yet Eedi 1st and 2nd ran 72B-class rerankers at 4-bit on two T4s; the writeups we read do not say what was offloaded, so treat that as open (the forum thread "Qwen2.5-72B & Llama 3.3-70B on 2xT4" worked it out). Nominal counts understate real ones slightly, so every margin is tighter than it looks.

The KV cache stores a key and a value per layer, per KV head, per token:

$$
\text{bytes}_{\text{KV}} = 2 \times L \times H_{kv} \times d_{\text{head}} \times T \times \text{bytes per value}
$$

For Qwen2.5-14B ($L = 48$, $H_{kv} = 8$, $d_{\text{head}} = 128$, from its public config) at fp16 that is ${2 \times 48 \times 8 \times 128 \times 2 = 196{,}608}$ bytes per token. Eight rows at the 6,144-token input cap WSDM 3rd place used is ${8 \times 6{,}144 = 49{,}152}$ tokens, or 9.7 GB of cache. This illustrative toy (not any team's code) adds the two:

```python
# Illustrative toy: does a model fit the box? Weights + KV cache, decimal GB.
def weight_gb(params_b, bits):
    return params_b * bits / 8                      # 1e9 params x bits/8 bytes = GB

def kv_gb(layers, kv_heads, head_dim, tokens, bytes_per=2):
    per_token = 2 * layers * kv_heads * head_dim * bytes_per   # K and V
    return per_token * tokens / 1e9

BOX_GB = 32        # two T4s, 16 GB each, before any runtime overhead
# Qwen2.5-14B shape: 48 layers, 8 KV heads, head_dim 128 (from its public config)
for bits in (16, 8, 4):
    w = weight_gb(14, bits)
    kv = kv_gb(48, 8, 128, tokens=8 * 6144)        # 8 rows at a 6,144-token cap
    verdict = "fits" if w + kv < BOX_GB else "does not fit"
    print(f"14B @ {bits:>2}-bit: weights {w:5.1f} GB + KV {kv:4.1f} GB = {w + kv:5.1f} GB -> {verdict}")
```

```console
14B @ 16-bit: weights  28.0 GB + KV  9.7 GB =  37.7 GB -> does not fit
14B @  8-bit: weights  14.0 GB + KV  9.7 GB =  23.7 GB -> fits
14B @  4-bit: weights   7.0 GB + KV  9.7 GB =  16.7 GB -> fits
```

"Fits" means across both cards; even 16.7 GB overflows one T4, hence tensor parallelism or one model copy per T4 in nearly every writeup. The T4 also lacks bfloat16 and FlashAttention-2: Jigsaw 12th place measured public notebooks running bf16 on T4 about 6 times slower ([Jigsaw post](/blog/machine-learning/learn-from-kaggle/kaggle-jigsaw-agile-community-rules-solutions)).

The boxes themselves, as each competition's sources record them:

| Competition | Box | Largest model run in the notebook | How it fit |
|---|---|---|---|
| [LLM Science Exam](/blog/machine-learning/learn-from-kaggle/kaggle-llm-science-exam-solutions) | 2x T4 or 1x P100, 9 h | Llama-2-70B (5th) | layer-by-layer on the least-confident 40% |
| [LMSYS Chatbot Arena](/blog/machine-learning/learn-from-kaggle/kaggle-lmsys-chatbot-arena-solutions) | 2x T4, 9 h | Gemma-2-9B student of 70B/72B teachers (1st) | distillation, GPTQ 8-bit |
| [AIMO 1](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-1-solutions) | 2x T4 or 1x P100, 9 h | DeepSeekMath-7B (all top 4) | 8-bit GPTQ (1st), one copy per T4 (4th) |
| [WSDM Cup](/blog/machine-learning/learn-from-kaggle/kaggle-wsdm-cup-multilingual-arena-solutions) | 2x T4, 4.75 h training phase / 12 h forecast | Qwen2.5-14B, Qwen2.5-32B GPTQ on uncertain rows (7th) | distillation, vLLM, 8-bit or 4-bit |
| [Eedi](/blog/machine-learning/learn-from-kaggle/kaggle-eedi-mining-misconceptions-solutions) | 2x T4, 9 h | Qwen2.5-72B listwise reranker (1st) | AWQ, 72B only on the final top 5 candidates |
| [MAP](/blog/machine-learning/learn-from-kaggle/kaggle-map-student-math-misunderstandings-solutions) | 2x T4, 9 h | Qwen3-32B + GLM-Z1-32B (1st) | W8A8 SmoothQuant, two layers resident |
| [Jigsaw Rules](/blog/machine-learning/learn-from-kaggle/kaggle-jigsaw-agile-community-rules-solutions) | 2x T4, 12 h | 13B to 14B in ranks 1 to 8 | 4-bit LoRA, one model per T4 |
| [AIMO 3](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-3-solutions) | 1x H100 80 GB, 5 h | gpt-oss-120b (all 5 writeups) | FP8 KV cache, prefix caching, time allocation |

## The cross-competition landscape

![Matrix of eight competitions against seven model-fitting moves, each cell naming the ranks or counting the writeups that used the move, grey where no writeup reported it](/imgs/blogs/kaggle-llm-fitting-big-models-in-small-gpus-1.webp)

By column: **quantisation is table stakes from 2024 on** (13 of 15 Eedi writeups ran 4-bit; every MAP writeup stating its precision was quantised, 9 of 9). **vLLM arrived with the CausalLM formulation**: 4 of 8 WSDM writeups (recounted from the matrix cells; the top-20 notes said 5) and 8 of 15 in Eedi, one of them partially. **Distillation is decisive where it appears but rare**: 1 of 10 in LMSYS (where three more teams tried and failed), 5 of 8 in WSDM, 3 of 15 in Eedi, 4 of 14 in MAP. **Streaming layers** is a last resort. **Time managers** live where problems arrive one at a time: AIMO.

Read it by row and the decisive move tracks the box. In 2023 teams streamed 70B; in 2024 and 2025 they moved big models into training; in AIMO 3 the box became an H100, weights stopped binding, and the fight moved to the KV cache and the clock.

## Distillation: put the 70B in the training run, not the notebook

![Size-to-scale comparison: two 70B-class teachers of 140 and 144 GB at 16-bit outside the Kaggle box, five fold LoRAs averaged into one, and a 9 GB Gemma-2-9B student inside a 32 GB, 9-hour notebook](/imgs/blogs/kaggle-llm-fitting-big-models-in-small-gpus-3.webp)

A 70B cannot be resident on two T4s at any common precision (140 GB at 16-bit, 35 GB at 4-bit). LMSYS 1st place (sayoulala, "Distill is all you need") built around that. Llama-3-70B and Qwen2-72B, QLoRA-trained over 5 folds, produced logits; Gemma-2-9B learned from them; the **5 fold LoRAs were averaged into one** (one inference pass instead of five); then GPTQ 8-bit: ${9 \times 8 / 8 = 9}$ GB. The student's fold log losses (0.858 to 0.876) beat both teachers' (0.869 to 0.881). The [knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals) post covers the loss.

It was not a free lunch: in the same LMSYS top 20, 3rd "couldn't get it right", 13th failed from Gemma-2-27B (its diagnosis: distil from an ensemble, not one model), and 19th found no working method. In WSDM, 5 of 8 writeups distilled from 70B+ teachers, and 3rd place (RihanPiggy) found **self-distillation from the 14B's own logits matched 72B distillation**, suspecting soft-logit label cleaning was what mattered. 6th place found distillation helped Gemma-2-9B measurably and Qwen2.5-14B not at all. MAP 12th (Eisuke Okuda) called 72B-to-14B distillation its biggest single gain, +0.003 to +0.005. With no room for a teacher, Jigsaw 6th (ducnh279) used **Deep Mutual Learning**, online distillation among Qwen3-14B, Qwen3-8B and Qwen3Guard-Gen-4B trained jointly as peers, and 12th distilled an LLM ensemble into a DeBERTa that ran in about an hour.

## Quantisation: 8-bit for speed, 4-bit for size, CV for the decision

![Slope chart of WSDM 4th place's quantisation choice: GPTQ 8-bit at CV 0.713, public 0.703, private 0.714 against AWQ 4-bit at CV 0.710, public 0.708, private 0.709](/imgs/blogs/kaggle-llm-fitting-big-models-in-small-gpus-4.webp)

The cleanest published experiment is WSDM 4th place (HengweiDAI): fp16 CV 0.714, GPTQ 8-bit 0.713, AWQ 4-bit 0.710. Public preferred AWQ (0.708 against 0.703); private reversed it (0.714 against 0.709), as CV had said. Going from 8 to 4 bits cost 0.003 of CV and 0.005 of private score.

Why 8-bit when 4-bit is smaller? Speed on old hardware. LMSYS 9th (Ebi) chose 8-bit over 4-bit for speed and merged the LoRA with no loss. LMSYS 16th (Chris Deotte) trained in fp16 and inferred in 8-bit instead of 4-bit both ways: about 10% faster, which bought a 3,072-token inference length. MAP 1st (tascj) went further: a T4 sustains about 20 TFLOPS in fp16 in practice, LMDeploy's W8A8 INT8 kernels 40+, so SmoothQuant (alpha 0.75) roughly doubled usable compute with no measurable ensemble loss. The [SmoothQuant and KV-cache quantisation post](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache) explains why activations need the smoothing.

4-bit wins when size binds, as with Eedi's standard 32B reranker. Teams then argued over the quantiser:

| Choice | Who | What they reported |
|---|---|---|
| AWQ with task-specific calibration | Eedi 1st (MTH 101) | used for every quantised model |
| AWQ over GPTQ, "capability per bit" | Jigsaw 4th (Dream Team) | AWQ Qwen3-4B and 14B |
| bitsandbytes to AWQ | MAP 9th (kanto-kaggler) | same accuracy, about 1/6 the inference time |
| auto-round 4-bit, 512 calibration samples | Eedi 2nd (CQYR) | under 2% loss |
| GPTQ 4-bit of a merged 32B backbone | MAP 10th (Liwen Zheng) | served in vLLM `task="embed"` |
| fp16 rerankers, quantised afterwards | Eedi 6th (Nikita Babich) | did not work |

A gotcha worth stealing: Eedi 8th (Yannan Chen) found AutoAWQ did not export `lm_head` and embedding weights for Qwen2.5-0.5B (random letters at inference) and copied them from the unquantised safetensors. AIMO 1 winner Numina chose 8-bit GPTQ for DeepSeekMath-7B partly because upload to Kaggle was twice as fast. One correction: our source notes quote Numina's reason as "~32 GB just for weights" at 16-bit; the formula gives ${7 \times 2 = 14}$ GB, so we use 14 GB, halved to 7 GB at 8-bit. See [GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq) for the mechanics.

## Serving: vLLM, or no padding anywhere

A causal LM answering one token can ride a serving engine, which is why the CausalLM formulation spread. WSDM 4th ran vLLM with tensor parallelism 2 and one generated token restricted by `allowed_token_ids=[a_tok_id, b_tok_id]`, noting that plain top-N `logprobs` can miss both A and B. MAP 7th (S T I T Y) and 8th (Takashi Someya) both chose label-token logits *because* vLLM accelerates them.

WSDM 3rd measured about 70k tokens per minute on two T4s (about 74k with prefix caching for TTA). Its 265-minute training-phase budget is therefore ${265 \times 70{,}000 = 18.55}$ million tokens, about 3,019 rows if every row hit its 6,144-token cap. WSDM 7th (Team Turing) timed 512 rows in vLLM at 6.5 min for Gemma-2-9B, 10.5 min for Qwen2.5-14B and 20 min for Qwen2.5-32B GPTQ, each at a shorter length than the last: 1.6 to 1.9 times more per size step, so the 32B ran on uncertain rows only.

Counterexamples: LMSYS 13th (heartkilla) found vLLM, AWQ and GPTQ gave little speed-up for classification. LMSYS 2nd place ([Rist] pref: tascj, sakaku, kapenon) skipped engines entirely: `flash_attn_varlen_func` over concatenated sequences, **no padding anywhere**, Triton kernels ported from lightllm and lmdeploy, and layers split across the two T4s and pipelined. WSDM 1st and 2nd reused that engine six months later.

## KV-cache tricks: encode the shared prefix once

![Token strips to scale: five naive sequences of 1,020 tokens totalling 5,100, against one cached 1,000-token prefix with five 20-token options totalling 1,100](/imgs/blogs/kaggle-llm-fitting-big-models-in-small-gpus-5.webp)

Multiple-choice and reranking repeat a long prefix with short suffixes. LLM Science Exam 1st place (Team H2O LLM Studio) ran context plus question once, kept `past_key_values`, and scored the five options as a batch of five reusing that cache. With illustrative lengths of 1,000 prefix tokens and 20 per option, the naive cost is ${5 \times 1{,}020 = 5{,}100}$ tokens and the cached cost ${1{,}000 + 5 \times 20 = 1{,}100}$, about 4.6 times fewer. That is how five 7B readers and a 13B fit "precisely" into 9 hours. The idea recurs:

- **MAP 1st (tascj)**: prefix plus all candidate suffixes packed into one sequence under a FlexAttention mask.
- **Science Exam 5th (Preferred ChattyKids)**: five option-order rotations packed into one masked sequence.
- **Eedi 14th (eeee)**: `past_key_values` reused for batches of 5 candidates.
- **Jigsaw 4th**: batches grouped by rule so the rule prefix stays cached.
- **Eedi 2nd and 7th**: vLLM prefix caching, about 10% faster for 2nd.

On an H100 weights fit, so the cache sets how many attempts run at once. In AIMO 3, 4 of 5 writeups used an FP8 or 8-bit KV cache: halving bytes per value doubles the tokens a fixed budget holds, letting 8 attempts at about 65,536 tokens each (${8 \times 65{,}536 = 524{,}288}$ tokens at worst) share one card. The [KV-cache optimisation](/blog/machine-learning/model-serving/kv-cache-optimization) and [prefix caching](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) posts cover the serving side.

## Layer-by-layer: when the model cannot fit at all

![Two-lane timeline: disk-to-GPU loads of layers 1 to 5 overlapping GPU compute of layers 1 to 4 on 640 samples, two layers resident, with the arithmetic of 0.5 GB per layer, 25 passes and 800 GB read](/imgs/blogs/kaggle-llm-fitting-big-models-in-small-gpus-6.webp)

If the model will not fit, keep only the layer you are computing. simjeg's public Science Exam notebook ran Platypus2-70B layer-by-layer, sample-by-sample, on a Kaggle GPU. Science Exam 5th stored per-layer quantised Llama-2-70B weights as a dataset and ran them in about 6 GB of GPU memory on the least-confident 40% of questions, then again on the hardest 5%. At 16-bit each of Llama-2-70B's 80 layers is ${140 / 80 = 1.75}$ GB, so the reported 6 GB covers a layer plus activations. Eedi 13th ran its 32B and 72B listwise rerankers in fp16, one layer at a time across both GPUs, inside 500 minutes.

Every pass reads the whole model from disk; MAP 1st made that affordable by keeping two layers resident, so loading layer $k+1$ overlaps computing layer $k$, and pushing 640 samples through each pass. For Qwen3-32B at INT8, ${32 / 64 = 0.5}$ GB per layer. Over 16,000 rows that is ${16{,}000 / 640 = 25}$ passes and ${25 \times 32 = 800}$ GB read; at batch 16 it would be 1,000 passes and 32 TB. tascj reported about 65 minutes for those 16k rows. Gotcha: Kaggle's `/tmp` is copy-on-write with a cap; layer files there crashed runs and only 4 of 6 uploaded models made the final. Use `/kaggle/working`.

Streaming has limits: Science Exam 12th timed out with Platypus2-70B, and 3rd place's ablation scored 0.9288 private *without* Platypus2 against 0.9284 with it. Stream a giant only for rows a small model is unsure about.

## Time managers: the queue is part of the problem

![Toy time manager over 10 problems and 3,600 seconds: budget equals remaining time divided by problems left, clipped to 300 to 900 seconds; hard problems 3, 6, 8 and 10 are cut off at 406, 439, 525 and 750 seconds](/imgs/blogs/kaggle-llm-fitting-big-models-in-small-gpus-7.webp)

In AIMO hidden problems arrive one at a time, in unknown order. Nine hours over 50 problems in AIMO 1 is ${540 / 50 = 10.8}$ minutes each; 5 hours over 50 in AIMO 3 is 360 seconds. AIMO 1 4th place (codeinter) wrote a `TimeManager`: when the running average per problem exceeds the remaining budget per problem, the next problem's attempts shrink to $0.8 \times \text{default} \times (\text{budget} / \text{average})$, and geometry problems with `[asy]` figures get only 3 attempts. One model copy per T4 raised samples per problem from 21 to 30 or 40. Numina deliberately kept fixed width 48 and depth 4, because early stopping gave high variance under random order.

AIMO 3 5th place (TAMU-TACO) published the cleanest rule: per-problem budget equals remaining time over problems left, clipped to [300, 900] seconds, up to 8 attempts, stop at 4 matching answers. They tuned it over 64 rounds on 64 H100s of their own. This toy reproduces the rule on 10 made-up problems:

```python
# Illustrative toy of the AIMO 3 5th-place rule (TAMU-TACO): budget = remaining / left, clipped.
FLOOR, CEIL = 300, 900

def run(needs, total):
    remaining = total
    for i, need in enumerate(needs):
        left = len(needs) - i
        budget = min(CEIL, max(FLOOR, remaining / left))
        used = min(need, budget)                    # a hard problem is cut off at its budget
        remaining -= used
        flag = "cut off" if used < need else ""
        print(f"p{i + 1:<2} need {need:>3}s budget {budget:5.0f}s used {used:5.0f}s left {remaining:5.0f}s {flag}")

run([150, 200, 900, 250, 400, 900, 180, 700, 300, 900], total=3600)
```

```console
p1  need 150s budget   360s used   150s left  3450s 
p2  need 200s budget   383s used   200s left  3250s 
p3  need 900s budget   406s used   406s left  2844s cut off
p4  need 250s budget   406s used   250s left  2594s 
p5  need 400s budget   432s used   400s left  2194s 
p6  need 900s budget   439s used   439s left  1755s cut off
p7  need 180s budget   439s used   180s left  1575s 
p8  need 700s budget   525s used   525s left  1050s cut off
p9  need 300s budget   525s used   300s left   750s 
p10 need 900s budget   750s used   750s left     0s cut off
```

The same 900-second problem gets 406 seconds in slot 3 and 750 in slot 10: robustness depends on where hard problems fall, TAMU-TACO's own framing. Their tests rejected longer budgets (they eat time early) and more aggressive early stopping (it locks in repeated wrong answers).

Classifiers spend leftover time on uncertain rows instead. LMSYS 9th ran its third model, Llama-3, only on rows of at most 512 or at least 1,500 tokens; WSDM 3rd ran TTA on 25% of rows and Phi-4 on about 60%; WSDM 6th ran Gemma-2-9B only where Qwen2.5-14B was unsure; Jigsaw 8th's 3 length buckets freed time for one more 8B model.

## What did not work

| Attempt | Who reported it |
|---|---|
| Distilling from a single bigger model | LMSYS 3rd, 13th, 19th |
| vLLM / AWQ / GPTQ for pure classification speed | LMSYS 13th, MAP 12th (vLLM) |
| Qwen2.5-72B AWQ under vLLM as a second pass (too slow) | WSDM 1st |
| Quantising fp16 rerankers after training | Eedi 6th |
| bf16 on T4 via patched vLLM (unstable) | AIMO 1 3rd |
| Static KV cache + torch.compile on T4 | AIMO 1 Numina |
| Models above 4B under the budget | Jigsaw 11th |
| More than 8 parallel attempts | AIMO 3 2nd, 3rd |

## If you enter a competition with a box like this

- [ ] Write `params x bits / 8` plus the KV formula before choosing a model; leave headroom.
- [ ] Use big models as teachers; try self-distillation before buying 70B compute.
- [ ] 8-bit when compute-bound, 4-bit when memory-bound; decide by CV.
- [ ] One-token CausalLM for vLLM, or padding-free if you stay a classifier.
- [ ] Share every repeated prefix through the KV cache.
- [ ] Stream layers only for uncertain rows, with large batches per pass.
- [ ] Budget time per item from what remains, with a floor and a ceiling.

## Sources

The competition posts are linked from the table at the top; the writeups cited:

- Science Exam: [1st](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446422), [3rd](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446358), [5th](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446293), [12th](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446301), [simjeg 70B thread](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/440620)
- LMSYS: [1st](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629), [2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685), [3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766), [9th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527704), [13th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/540876), [16th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527596), [19th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/528288)
- AIMO 1: [Numina](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/numina-numina-1st-place-solution), [after exams](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/after-exams-3rd-place-solution), [codeinter](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/codeinter-4th-place-solution)
- AIMO 3: [1st](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/1st-place-solution-for-the-aimo3-competition), [2nd](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/2nd-place-solution-ai-mathematical-olympiad-prog), [TAMU-TACO](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/aimo3-tamu-taco-solution)
- WSDM: [3rd](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567584), [4th](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/568522), [6th](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567600), [7th](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567589)
- Eedi: [1st](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution), [2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/cqyr-2nd-place-solution), [6th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/nikita-babich-6th-place-solution), [8th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/yannan-chen-8th-place-solution), [13th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/qwenqwenqwenqwenqwenqwenqwenqw-13th-place-solution), [14th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/eeee-rsi-14th-place-solution)
- MAP: [1st](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/1st-place-solution), [7th](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/private-7th-public-10th-place-solution), [8th](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/8th-place-solution), [9th](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/9th-place-solution), [10th](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/10th-place-solution), [12th](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/12th-place-solution)
- Jigsaw: [4th](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/4th-place-solution), [6th](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/6th-place-solution), [8th](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/8th-place-solution-qwen3-14b3-llama2-13b1-b), [11th](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/11th-place-solution), [12th](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/12th-place-solution)

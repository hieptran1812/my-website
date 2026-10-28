---
title: "WSDM Cup Multilingual Chatbot Arena: what the top 20 carried over from LMSYS, and what multilinguality changed"
date: "2026-10-28"
publishDate: "2026-10-28"
description: "The LMSYS sequel went binary, multilingual and out-of-time. Eight of the top twenty teams published, almost all ran a distilled Qwen2.5-14B, and 0.008 accuracy separated 1st from 20th. Every published idea, and the few that moved the needle."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "preference-modeling", "knowledge-distillation", "pseudo-labeling", "multilingual", "vllm", "quantization", "chatbot-arena"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 22
---

Six months after the LMSYS Chatbot Arena competition closed, Kaggle ran it again with three changes. The target went binary: ties were removed, so the model only had to say whether response A or response B won. The metric went from log loss to plain accuracy. And the data went multilingual, with a final test set made of Arena battles that did not exist yet when submissions closed. The WSDM Cup 2025 drew 950 teams and produced the most compressed leaderboard in this series: the winner scored 0.716092, second place 0.716047, and the 20th team 0.708127.

That compression is the story. When the previous competition's winners open-source their code and checkpoints, the starting line moves for everyone. What separated teams was not a new idea about preference modelling. It was how much model you could fit into two T4 GPUs, how clean your labels were, and where you spent the last hour of the clock.

> **TL;DR**
> - **Winner:** whitefebruary (Michael Pieler and Max), 0.716092 accuracy, by stacking LMSYS 1st place's distillation recipe on LMSYS 2nd place's padding-free inference engine, with Qwen2.5-14B as the student and five Qwen2.5-72B folds as teachers.
> - **Decisive lever:** model fit. A 14B model distilled from 72B teachers, run fast enough to afford a second pass on the rows it was unsure about. Synthetic multilingual data was the second lever.
> - **The one lesson:** a sequel is won by whoever reuses the last competition's open winners fastest and then fixes their labels. Seven of the eight teams that published trained on soft or pseudo labels.

![The lever strip: of Data, Retrieval, Model fit, Test-time compute and Metric insight, model fit is lit as the decisive lever and data as the secondary one](/imgs/blogs/kaggle-wsdm-cup-multilingual-arena-solutions-1.webp)

The strip above is the frame this series uses: every competition is decided by one of five levers. Here the lit one is model fit (distil a big judge into a small one, quantise it, run it inside the notebook), with data second because the top two teams both generated tens of thousands of new multilingual battles. This post reads best after [the LMSYS Chatbot Arena solutions](/blog/machine-learning/learn-from-kaggle/kaggle-lmsys-chatbot-arena-solutions); if the notebook format is new to you, start with [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work).

## The task in 60 seconds

Imagine sitting next to someone using Chatbot Arena. They type one prompt, often not in English. Two anonymous chatbots answer side by side, and the person clicks the one they prefer. Your job is to guess which one they clicked, for thousands of battles, without seeing who answered.

Formally: each row has a `prompt`, a `response_a` and a `response_b`; predict `winner`, either `model_a` or `model_b`. Conversations are single turn. Training rows also carry `model_a`, `model_b` (the chatbot names) and `language`; test rows do not.

![The notebook budget drawn to scale: two T4 GPUs with 16 GB each and no internet, a 4.75 hour training-phase limit against a 12 hour forecasting limit, and roughly 10,000 historical test rows against up to 25,000 new ones](/imgs/blogs/kaggle-wsdm-cup-multilingual-arena-solutions-2.webp)

The budget card above is the constraint every technique answers to. Submissions were Kaggle notebooks with the internet off. In the training phase a notebook had 4.75 hours to score about 10,000 historical battles. After the deadline on 2025-02-03, selected notebooks were re-run once, with a 12 hour limit, on newly collected Arena battles, "at most 25,000 rows". That re-run produced the private leaderboard. Third place budgeted 265 and 690 minutes for the two phases and measured about 70,000 tokens per minute for a 14B model on the two T4s.

Those numbers make the problem concrete. If rows averaged 3,000 tokens, one pass over 25,000 of them would take around 1,070 minutes of prefill alone. Everyone had to truncate, sort, quantise and choose.

![An illustrative training row: a short German prompt, a concise German answer A, a long English answer B, the winner column as target, and the model and language columns marked as training-only](/imgs/blogs/kaggle-wsdm-cup-multilingual-arena-solutions-3.webp)

The row above is illustrative, not from the dataset, but it shows the shape of the problem. The prompt is German. Response A answers in German and stops; response B answers in English and keeps going. A human who asked in German probably prefers A, and a model that learned "longer and more detailed wins" from English battles gets it wrong. The bottom strip shows the input format nearly every top team converged on: prompt and both responses concatenated, each truncated from the middle so the opening and ending of every field survive.

The training file had about 48,000 rows. Outside data was fair game, and every serious team used a lot of it: the LMSYS 55k and 33k Arena sets, preference datasets such as UltraFeedback, and open-model response sets posted to the forum. Prizes totalled USD 50,000, from USD 12k for first to USD 8k for fifth.

## The metric: accuracy, and why it compresses everything

LMSYS used log loss over three classes, which rewards calibrated probabilities. WSDM used accuracy over two:

$$
\text{accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\left[\hat{y}_i = y_i\right], \qquad \hat{y}_i = \begin{cases} A & \text{if } p_i(A) \geq 0.5 \\ B & \text{otherwise} \end{cases}
$$

Here $N$ is the number of test rows, $y_i$ the human's choice and $p_i(A)$ the model's probability that A won. Everything the model knows about a row collapses into which side of 0.5 it lands on.

![A toy batch of eight rows scored for accuracy: five correct gives 0.625, and nudging one prediction from 0.51 to 0.49 flips it and lifts accuracy to 0.750](/imgs/blogs/kaggle-wsdm-cup-multilingual-arena-solutions-4.webp)

In the toy batch above, five of eight rows are right, so accuracy is ${5/8 = 0.625}$. Row 1 at $p(A) = 0.90$ and row 8 at $p(A) = 0.51$ count exactly the same. Nudge row 8 to 0.49 and the prediction flips to B, the truth, and accuracy jumps to ${6/8 = 0.750}$. All of the action lives near 0.5.

The top teams exploited two consequences. Calibration work from LMSYS stopped paying: temperature scaling cannot change a single prediction. And the rows near 0.5 are the only rows where extra compute can change the score. A second model applied to a row scored 0.95 is wasted; applied to a row at 0.52, it is the whole game. Six of the eight published solutions spend extra compute only where the model is unsure.

The flip side is noise. Ranks 1 to 20 span ${0.716092 - 0.708127 \approx 0.008}$ accuracy. If the forecasting set was near its 25,000-row ceiling (its actual size was never confirmed), that is about 200 rows. First and second differ by 0.000045, about one row in 22,000.

## What made it hard

**The notebook, not the cluster.** Training was unconstrained: 4th place trained on 40 A100 80GB cards, 5th ran 12 and 16 hour stages on six A100s. But inference had to fit on two T4s. A Qwen2.5-72B judge was the best model anyone trained, and nobody could run it at test time; first place tried a 72B AWQ model under vLLM for its second pass and found only 5% of rows fit.

**Labels from strangers.** Arena votes are noisy, so some fraction of training labels is plain wrong. Several top teams used teacher soft labels as a cleaning step.

**Multilinguality with an English-heavy supply.** The extra labelled data that made LMSYS work is mostly English. Teams that wanted multilingual training pairs had to generate them and pseudo-label them.

**A test set from the future.** The private board came from battles collected after the deadline on a live public Arena, a true out-of-time test that made the public board a weak guide. A forum thread worried that competitors could vote in the Arena that would become the test set; that concern was never resolved in public.

## The top-20 scorecard

The leaderboard API returns `0.000000` for every public score in this competition, so its "public rank" is an artefact of sorting zeros and no rank movement can be computed. The public column below uses only scores teams reported in their own posts.

| Rank | Team | Private (accuracy) | Public, self-reported | Key idea | Source |
|---|---|---|---|---|---|
| 1 | whitefebruary (Michael Pieler, Max) | 0.716092 | n/a | LMSYS 1st distillation + LMSYS 2nd padding-free inference; full-FT Qwen2.5-14B; five 72B teachers; 50k synthetic multilingual prompts | [writeup](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902) |
| 2 | zhudong1949 (solo) | 0.716047 | 0.709 | tascj's LMSYS checkpoints; 95k vLLM-generated pseudo-labelled pairs; English epoch, then multilingual | [writeup](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567948) |
| 3 | PLaMo 1000000B (RihanPiggy, solo) | 0.714967 | n/a | CausalLM scoring the "A"/"B" token under vLLM; Qwen2.5-14B + Phi-4; self-distillation | [writeup](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567584) |
| 4 | HKUST-GZ DSA KiMi Lab (HengweiDAI, solo) | 0.714067 | 0.703 | A/B-token CausalLM, vocabulary-wide post-pretrain, three 70B-class teachers, GPTQ 8-bit chosen by CV | [writeup](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/568522) |
| 5 | Just doing (five members) | 0.712627 | 0.706812 | Qwen2.5-14B CausalLM, LoRA r=64, hard then soft labels, number-count tie-break | [writeup](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567856) |
| 6 | Oh Your Mercy Never Fails | 0.712582 | n/a | 560k pairs pseudo-labelled by a 9B; Gemma only where Qwen was unsure | [writeup](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567600) |
| 7 | Team Turing (Trushant Kalyanpur, Raja Biswas) | 0.712222 | n/a | Uncertainty cascade 9B, 14B, then 32B or 24B; 100k topic-matched generated pairs | [writeup](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567589) |
| 8 | sayoulala | 0.712042 | n/a | LMSYS 1st-place author; reopened his LMSYS code | no writeup |
| 9 | quenn | 0.711817 | n/a | unknown | no writeup |
| 10 | tascj | 0.711817 | n/a | LMSYS 2nd-place author whose code 1st and 2nd reused | no writeup |
| 11 | Evalthon | 0.711412 | n/a | unknown | no writeup |
| 12 | monsaraida (solo) | 0.710872 | n/a | Qwen2.5-14B QLoRA r=8, "(snip N chars)" marker, model-name auxiliary head | [writeup](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567614) |
| 13 | x-creative | 0.710737 | n/a | unknown | no writeup |
| 14 | Takoi | 0.709432 | n/a | unknown | no writeup |
| 15 | KYRL | 0.709252 | n/a | final unknown; a member's thread documents a Gemma-2-9b baseline | no writeup |
| 16 | Tilbur | 0.708802 | n/a | unknown | no writeup |
| 17 | yukichi0403 | 0.708667 | n/a | unknown | no writeup |
| 18 | zxc123cc | 0.708442 | n/a | unknown | no writeup |
| 19 | Shaomu & Shiyi | 0.708172 | n/a | unknown | no writeup |
| 20 | RankMe | 0.708127 | n/a | unknown | no writeup |

Eight of twenty teams published: ranks 1 to 7 and 12 (7th's post is on the forum though the API lists no link). Ranks 9 and 10 share a displayed score; gold went to the top 11.

## The top-20 landscape

![The top-20 technique matrix: ranks 1 to 7 and 12 have writeups and filled cells, the other twelve teams are greyed as no writeup, with counts such as 7 of 8 using soft or pseudo labels and 6 of 8 spending extra compute only on uncertain rows](/imgs/blogs/kaggle-wsdm-cup-multilingual-arena-solutions-5.webp)

The matrix above is the most useful single picture of this competition. Counts on the right are out of the eight teams that published; a hatched cell means the writeup does not say.

**Table stakes:**

- **Qwen2.5-14B-Instruct as the main model.** Six writeups name it (1st, 3rd, 5th, 6th, 7th, 12th); 4th says only "a 14B model", though its dataset class is named `qWenSFTDataset`. Second place kept the LMSYS-trained Gemma-2-9B and ArmoRM-Llama3-8B it started from. Gemma-2-9B, the LMSYS workhorse, survives in three finals (2nd, 6th, 7th), and in 6th only as a helper.
- **LMSYS data as extra labels:** seven of eight; 2nd inherited it through its checkpoints.
- **Soft or pseudo labels:** seven of eight; only 12th stayed on hard labels.
- **Both A/B orders in training**, often in the same batch or gradient step.
- **Extra compute on uncertain rows only:** six of eight.

**What separated the top 3.** Not much: 1st to 6th is 0.0035. But two things appear only at the very top. An engineered inference path: 1st and 2nd ran tascj's padding-free kernels from LMSYS, and 3rd rebuilt the task as a causal LM so vLLM with prefix caching could serve it. Both bought room for a 14B model plus a second pass. And deliberate label cleaning: 1st hand-corrected rows where teachers disagreed with humans; 3rd showed self-distillation matched 72B distillation. Simpler recipes still got close: 5th used no distillation and reached 0.712627, and 12th's single QLoRA at rank 8 reached 0.710872.

**Where the writeups contradict each other:**

| Question | Yes | No |
|---|---|---|
| Swapped-order TTA helps? | 1st, 3rd, 12th (on uncertain rows); KYRL's lhwcv | 6th: "no measurable impact" |
| 70B distillation worth it? | 1st, 4th; 6th for Gemma | 6th for Qwen2.5-14B; 3rd (self-distillation matched); 5th never distilled |
| A second epoch? | 4th, CV up | 5th, LB down |
| Longer context? | 3rd used about 6,144 input tokens | 5th: 8,192 hurt LB; 6th: nothing past 2,500 |
| Chain of thought? | 5th +0.003, not shipped; 3rd +0.001 to 0.002 with timeouts | 1st, 6th |
| Merging? | 1st: linear merge of full fine-tunes | 6th: LoRA merge degraded badly |

With a spread this small, a +0.002 on one team's CV and a -0.003 on another's public board are both inside the noise of a 10,000-row public set.

## The winning solution, explained

The intuition: in LMSYS, sayoulala won by distilling giant teachers into a small student, and tascj placed second with an inference engine fast enough to run big models in time. Whitefebruary took both, swapped in a stronger, more multilingual base, and added what LMSYS did not need: multilingual training data that looked like the test.

![First place's pipeline: preference pretraining, five 72B teacher folds on 80k cleaned rows, distillation into Qwen2.5-14B with extra soft-labelled data sets drawn to scale, a linear merge, and a two-pass inference with the second pass on the 33% most uncertain rows](/imgs/blogs/kaggle-wsdm-cup-multilingual-arena-solutions-6.webp)

The pipeline above has three training stages and two inference passes.

**Stage 1, preference pretraining.** Student (Qwen2.5-14B-Instruct) and teacher (Qwen2.5-72B-Instruct) first train on `mlabonne/orpo-dpo-mix-40k` and `opencsg/UltraFeedback-chinese` (50k) with a two-class `ForSequenceClassification` head. Full-parameter training under DeepSpeed ZeRO-3, peak learning rate 4e-6 with 50 warm-up steps and linear decay to 10% of peak, gradient clipping raised to 64. Each field is truncated proportionally from the middle, and both A/B orders go into the same gradient step.

**Stage 2, the teachers.** Five folds of Qwen2.5-72B train on the Arena 55k set (57k rows) plus the 48k competition rows, about 80k after dropping ties, multi-turn conversations and duplicates. The teachers see the **model names** before each response. The student never can, but a teacher that knows "this is GPT-4o against a 7B model" produces sharper soft labels.

**Stage 3, distillation.** Out-of-fold teacher predictions become soft labels. First, the team **manually reviewed rows where teachers disagreed most with the human label and corrected the mislabels**. Then they soft-labelled extra data: Arena sets with ties and first turns (106k), `chatbot_arena_conversations` (33k), PPE-Human-Preference-V1 (16k, included because they suspected it correlated with the leaderboard and feared others would use it), nbroad's open-model sets (25k), two small 1k battle sets, and their own 50k synthetic set.

That synthetic set is where multilinguality enters. Prompts from lmsys-chat-1m, filtered to Arena-Hard complexity 3 or higher, half non-English, matched to eight MT-Bench categories, answered by Llama-3.1-405B-Instruct-FP8, Llama-3.3-70B, Qwen2.5-72B and Athene-V2-Chat under vLLM. The loss is a weighted sum of cross-entropy on the hard label, KL to the soft label, and a cosine term. The final model linearly merges a full-data model with one trained without the last two datasets.

**Inference.** tascj's padding-free sequence packing, ported to Qwen2.5, spends no compute on pad tokens. Pass 1 runs the merged model on every row. Pass 2 runs the full-data model on the A/B-swapped input for the 33% of rows where pass 1 was least certain. Here is that gate as an illustrative toy, not the team's code (theirs is linked in Sources):

```python
# Illustrative toy of uncertainty-gated swap TTA. Not any team's actual code.
import numpy as np
import torch

def score(model, tok, prompts, a_list, b_list, max_len=3072):
    """Return p(A wins) for each row with a 2-class sequence classifier."""
    out = []
    for p, a, b in zip(prompts, a_list, b_list):
        text = f"<prompt>{p}</prompt><response_a>{a}</response_a><response_b>{b}</response_b>"
        enc = tok(text, truncation=True, max_length=max_len, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[0].float()
        out.append(torch.softmax(logits, dim=-1)[0].item())
    return np.array(out)

def two_pass(model_main, model_swap, tok, prompts, a_list, b_list, frac=0.33):
    p1 = score(model_main, tok, prompts, a_list, b_list)
    margin = np.abs(p1 - 0.5)                 # small margin = unsure
    k = int(len(p1) * frac)
    unsure = np.argsort(margin)[:k]           # the 33% closest to 0.5
    # Swap A and B, so the model's "A wins" now means the original B won.
    p_swap = score(model_swap, tok,
                   [prompts[i] for i in unsure],
                   [b_list[i] for i in unsure],
                   [a_list[i] for i in unsure])
    p_final = p1.copy()
    p_final[unsure] = 0.5 * p1[unsure] + 0.5 * (1.0 - p_swap)
    return np.where(p_final >= 0.5, "model_a", "model_b")
```

Real solutions also sort by token length, split rows across both GPUs and pack sequences, which is where most of the throughput comes from. The gate is the part worth copying.

What did not work for them: a GPT-4o critique model meant to add chain-of-thought signal (soft labels with critiques came out worse every time), layer pruning gemma-2-27b or Qwen2.5-14B (no better than plain gemma-2-9b), and forced order-invariant embeddings.

## 2nd and 3rd place in depth

### 2nd: zhudong1949, the multilingual curriculum

The solo second place missed first by 0.000045 by a different route. He started from **tascj's LMSYS-trained gemma2-9b and ArmoRM-Llama3-8B checkpoints**, swapped their three-class heads for two-class ones, and kept tascj's full fine-tuning framework with proportional middle truncation.

His first submission soft-labelled nbroad's v1 (8.5k) and v2 (13k) open-model sets with a three-model average (v3 was dropped as too close in quality to label reliably) and ensembled gemma on the normal order with llama on the swapped order at 3.3:1. It scored 0.709 public, 0.708 private.

The winning submission added 74k pairs he generated. He split deduplicated lmsys-chat-1m prompts into English (424k) and other languages (103k), sampled three batches (7k plus 3k, 21k plus 6k, 36k plus 2k English and other), and answered them with 13 models under vLLM, including Qwen2.5-72B, Llama-3.3-70B, phi-4 and DeepSeek-R1-Distill-Qwen with think sections removed. He pseudo-labelled at 3.3:1 and dropped rows whose two probabilities were within 0.03. Then a curriculum: re-pretrain on the 95k pseudo-labelled rows, then fine-tune on WSDM data with **epoch 1 on the English subset and epoch 2 on the multilingual subset**. At inference llama re-scores only rows where $|p_a - p_b| \lt 0.8$, combined 2.5:1. Public 0.709, private 0.716: the public board could not tell his two submissions apart, and the future test could.

Dropped along the way: Qwen2.5-72B distillation (too costly), Skywork reward models as labellers or backbones, Skywork 80K data, Bradley-Terry loss and multi-loss weighting.

### 3rd: PLaMo 1000000B, the causal-LM reformulation

Third place made the most consequential engineering choice of the competition. Instead of a classification head he trained a **causal language model** and read the logits of exactly two tokens, "A" and "B", borrowing the pattern from Eedi's first-place reranker. The point was serving: a causal LM runs under vLLM with paged attention, continuous batching and prefix caching, which a custom classification head does not get for free (the [vLLM deep dive](/blog/machine-learning/inference-frameworks/vllm-deep-dive) covers those).

The template asks which response is best, answer with the letter only, and includes a `<Language>` field filled by fastText language ID. Prompt, A and B are each capped at 2,048 tokens. His CV shows how sharp the learning-rate cliff is: Qwen2.5-14B at 1e-4 scored 0.7051 on fold 0, and at 1.4e-4 the loss spiked and accuracy collapsed to 0.508, a coin flip. Distillation at temperature 5 reached 0.7135, more teachers 0.7169. Then the surprise: **self-distillation from the 14B's own logits matched 72B distillation**. In his words, "maybe it is the effect of label cleaning brought by soft logit that really matters."

He quantised with auto-round and planned the clock like a scheduler. Rows sorted by length; Qwen-14B scores everything; swap TTA on 25% of rows, helped by vLLM's prefix cache (about 70k to 74k tokens per minute); the rest of the time goes to Phi-4 on about 60% of rows, **weighted by prompt-length quantile** (0.5 below the median, 0.15 in the next band) because Phi-4 was good on short prompts only. Seed-merged models were an explicit robustness choice for a test nobody could see.

## Ranks 4 to 20, by approach family

![Where the second pass went: horizontal bars for 1st, 3rd, 7th, 12th and 2nd place, each full width for the base model with an overlaid segment showing the share of rows that got extra compute, from 15% to 60%](/imgs/blogs/kaggle-wsdm-cup-multilingual-arena-solutions-7.webp)

The bars above line up the uncertainty gates. Every bar is the whole test set; the overlay is the share that got a second opinion. The shares range from 15% to 60%, but the shape is identical: one cheap pass on everything, an expensive pass where it can change the answer.

### Family 1: A/B-token causal LMs (4th, 5th)

**4th, HKUST-GZ DSA KiMi Lab, 0.714067.** Built on sayoulala's LMSYS pipeline. The judge prompt ends in `' Choice: '`, scored by the A logit against the B logit. His distinctive idea is a two-stage loss: in post-pretraining on UltraFeedback and a multilingual reward bench, the A/B token is trained with cross-entropy **against the whole vocabulary**; in fine-tuning, **between the A and B logits only**. He distilled Athene-V2-Chat, Llama-3.1-Nemotron-70B and Qwen2.5-72B into a 14B student, and a second epoch lifted all five folds by 0.0011 to 0.0047. At inference, vLLM with `allowed_token_ids` restricted to the two letters, since plain top-N logprobs can miss both.

**5th, Just doing, 0.712627.** **Qwen2.5-14B-Instruct as a causal LM, LoRA r=64, alpha 128**, with separate learning rates for the LoRA A and B matrices (1e-5 and 5e-5). Stage 1 trains on hard labels; stage 2 smooths labels to 0.05 and 0.95 and adds open-model sets with stage-1 soft labels. No distillation, no chain of thought. Their public ladder is the best ablation published here: zero-shot 0.6254, middle truncation 0.6938, plus LMSYS data 0.6964, plus swap 0.7038, plus a tie-break rule 0.7039, plus open-model soft labels 0.706812. The rule: when the A and B logits are nearly tied, prefer the response with more numbers in it.

### Family 2: pseudo-label at scale (6th)

**6th, Oh Your Mercy Never Fails, 0.712582.** About **560k** pairs: the LMSYS 3rd-place team's generated data, lmsys-chat-1m prompts answered through commercial APIs, and about ten open DPO datasets, all labelled by a fine-tuned Gemma-2-9B (70B labellers were slightly better but far slower). Pseudo-labels alone, with no competition data, already scored above 0.693 public. The DPO datasets' own labels were useless, but their prompts and responses were excellent raw material. Final models were LoRAs on 4-bit Qwen2.5-14B and Gemma-2-9B, with Gemma consulted only where Qwen was unsure.

### Family 3: the cascade (7th)

**7th, Team Turing, 0.712222.** Built in three weeks. Gemma-2-9b scores every row; uncertainty is ${1 - |p - 0.5|}$; the next 35% most uncertain go to Qwen2.5-14B at weight 1.0 and the top 15% to Qwen2.5-32B or Mistral-Small-24B at weight 1.5. Their data work matched distributions most carefully: 100k generated pairs from WildChat-1M and lmsys-chat-1m prompts, **cluster-sampled to the Arena Explorer topic mix**, pseudo-labelled by fine-tuned 70B-class models. Their vLLM timings per 512 rows show the economics: gemma-2-9b 6.5 minutes, Qwen2.5-14B 10.5, Qwen2.5-32B GPTQ 20.

### Family 4: the lean single model (12th)

**12th, monsaraida, 0.710872.** One Qwen2.5-14B-Instruct, **QLoRA rank 8**, one epoch, 2,048 tokens. Middle truncation inserts a literal `(snip 123 chars)` marker, which improved accuracy. An auxiliary head predicts the model names at loss weight 0.1, using the training-only columns as free signal (a language-ID auxiliary did not help). Swapped pairs only helped in the same batch, and a swapped second pass covered the 40% of rows closest to 0.5.

### The twelve teams with no writeup

Ranks 8 to 11 and 13 to 20 published nothing we could find, so their methods stay unknown. Two names matter anyway. **sayoulala (8th)** won LMSYS and reopened that code mid-competition "for fairness"; 4th built on it. **tascj (10th)** placed second in LMSYS; his code and checkpoints started both 1st and 2nd here. The two people whose work shaped the top of this board finished 8th and 10th on it. From **KYRL (15th)**, lhwcv's public thread documents a Gemma-2-9b baseline at 0.687 to 0.694 public with 5-fold CV of 0.694 to 0.701.

### The best contrasting entry

Fifth place: no 70B teachers, no custom engine, no distillation. A LoRA, a good truncation rule, swapped data and its own soft labels, 0.0035 behind the winner, under 90 rows on a 25,000-row test. With one GPU node and three weeks, that is the recipe.

## What the forum taught everyone

The most influential thread was sayoulala reopening his LMSYS winning code (93 votes): from week one, the LMSYS winners' recipes were everyone's baseline. Nicholas Broad posted three multilingual open-model response sets; 1st and 2nd used them. Threads on CV versus LB argued over trusting a noisy accuracy board, and a survey of public rank against GPU resources made the compute gap explicit. No leak was reported.

![Self-reported public scores against private scores for five submissions: 2nd place's two submissions start level and split, 4th place's AWQ and GPTQ lines cross, and 5th place rises](/imgs/blogs/kaggle-wsdm-cup-multilingual-arena-solutions-8.webp)

**The shake-up, from the teams' own numbers.** With API public scores wiped, the chart above uses only scores teams wrote down. Private ran 0.006 to 0.011 above public for the three submissions these teams chose, which suggests the forecasting set was easier or differently distributed. And the public board mis-ranked at least two decisions. Second place's two submissions tied at 0.709 public and split to 0.708 and 0.716 private. Fourth place's quantisation choice is cleaner still: AWQ 4-bit scored 0.708 public and 0.709 private; GPTQ 8-bit scored 0.703 public and 0.714 private. He picked GPTQ because its CV (0.713 against 0.710) said so, despite a 0.005 public deficit. For why the two methods diverge, see [weight-only quantization with GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq).

## What did not work

Aggregated from all eight published writeups:

| Tried | Result | Reported by |
|---|---|---|
| Chain of thought or critique-augmented training | Worse, or small gains lost to timeouts | 1st, 3rd, 6th; 5th did not ship its +0.003 |
| Running a 72B judge in the notebook | Only 5% of rows fit (AWQ, vLLM) | 1st |
| Layer pruning; order-invariant embeddings | No gain | 1st |
| Skywork reward models and 80K data; Bradley-Terry loss | No gain | 2nd |
| nbroad open-model set v3 | Too noisy to label | 2nd |
| Learning rate 1.4e-4 on Qwen2.5-14B | Collapse to 0.508 | 3rd |
| Ranking model; a fourth teacher; dropping high-disagreement rows | No significant change | 4th |
| AWQ 4-bit | Better public, worse private | 4th |
| Few-shot prompts; DPO plus BCE; 8,192-token training; 2 epochs | CV up or LB down | 5th |
| DeepSeek-R1-Distill-Qwen-14B as the judge | Worse on LB | 5th, 12th |
| LoRA merge; multi-LoRA ensembles; TTA for sequence classification | Degraded or no effect | 6th |
| mDeBERTa; GPT-3.5 labellers and judges; Gemma-2-27B teachers | Worse | 6th |
| Language-ID auxiliary head | No gain | 12th |

Two patterns run through this table. Reasoning did not help a one-token judge: four teams tried chain of thought and none shipped it. And the 72B model was always the best teacher but never a usable judge inside the notebook.

## Lessons that transfer

**1. A sequel starts from the previous winners' open code.** First and second built on LMSYS 1st and 2nd place code and weights; 4th on sayoulala's pipeline. Spend week one reproducing them on the new data (/discussion/569902, /discussion/567948).

**2. Soft labels clean labels.** Third place's self-distillation matched 72B distillation, and first place hand-fixed teacher-human disagreements. With crowd votes as labels, a teacher's probability is often closer to the truth than the vote; [knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals) explains why soft targets carry more signal (/discussion/567584).

**3. Synthesise data to match the test, on purpose.** First place matched language share, difficulty and MT-Bench categories; 7th cluster-sampled to the Arena topic mix; 2nd scheduled English before multilingual (/discussion/567589).

**4. Spend extra compute only where the decision is close.** Six of eight published solutions gated their second pass on uncertainty (/discussion/567614).

**5. Reformulate for your serving stack.** A causal LM emitting one of two tokens gets vLLM's batching and prefix cache. Three of the top five went that way (/discussion/568522).

**6. When the test comes from the future, pick by CV.** Fourth place's GPTQ choice and 2nd's tied submissions both show the public board ranking decisions wrongly.

### If you enter a competition like this

- Reproduce the previous competition's top two public solutions in week one.
- Write the inference budget as tokens per minute on the target hardware, then count affordable passes.
- Truncate from the middle, sort by length, keep both A/B orders in the same batch.
- Train a teacher with everything the test lacks (model names included); distil it into the largest student the notebook runs.
- Read the rows where teacher and human disagree most; some are wrong labels.
- Gate every second pass on the margin around 0.5, and choose the final submission by CV.

## Sources

- Competition: https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena
- 1st: /discussion/569902 ; code https://github.com/maxreciprocate/kaggle-lmarena-1st-place
- 2nd: /discussion/567948 ; code https://github.com/zhudongwork/wsdm-cub-2nd-place-solution
- 3rd: /discussion/567584 ; code https://github.com/LIHANG-HONG/WSDM-Cup-3rd-place-solution
- 4th: /discussion/568522 ; inference https://www.kaggle.com/code/daihengwei/wsdm-vllm-gen-inference-pb714
- 5th: /discussion/567856 , /discussion/567724 , /discussion/567617 ; code https://github.com/lcy80366872/kaggle-project/tree/main/wsdm
- 6th: /discussion/567600
- 7th: /discussion/567589 ; inference https://www.kaggle.com/code/conjuring92/wsdm-7th-place-solution
- 12th: /discussion/567614
- Threads: sayoulala's LMSYS code /discussion/554766; lhwcv CV vs LB /discussion/552368; Nicholas Broad's open-model sets /discussion/552166 , /discussion/556059 , /discussion/557746; manipulation concern /discussion/551090; GPU survey /discussion/561080

All `/discussion/...` paths are under https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena.

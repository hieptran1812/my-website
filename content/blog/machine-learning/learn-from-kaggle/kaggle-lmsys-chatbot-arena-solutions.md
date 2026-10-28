---
title: "LMSYS Chatbot Arena on Kaggle: how the top 20 taught a 9B model to judge like a human"
date: "2026-10-28"
publishDate: "2026-10-28"
description: "A synthesis of the top-20 LMSYS Human Preference Predictions solutions: why every winner ran Gemma-2-9B, how 1st place distilled 70B-class teachers into it, what reward pretraining and pseudo-labels bought, and how a data leak split the leaderboard across two test sets."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "reward-model", "preference-learning", "knowledge-distillation", "lora", "quantization", "pseudo-labeling", "gemma"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 22
---

> **TL;DR.** Winner: **sayoulala (team BlackPearl), solo, private log loss 0.96898.** Decisive lever: **model fit.** He trained Llama-3-70B and Qwen2-72B as teachers off Kaggle, distilled their predictions into Gemma-2-9B, averaged five fold adapters into one, and quantised it to 8 bits so it ran on two T4s in nine hours. The lesson: when the inference box is fixed, the competition is about how much of a big model's judgement you can pour into a small one. All ten published top-20 solutions ran Gemma-2-9B; what separated the top five was extra preference signal injected *before* the final fine-tune.

![The five-lever strip with Model fit lit as the decisive lever and Data half-lit for pseudo-labels and reward pretraining](/imgs/blogs/kaggle-lmsys-chatbot-arena-solutions-1.webp)

The strip above is the frame this series puts on every Kaggle LLM competition: five places an edge can come from, with the decisive one lit. In LMSYS it was model fit, with data a close second. The one thing that did break, a data leak in the last week, moved the leaderboard more than any technique.

The competition ran from May to August 2024 and asked a question that sounds trivial: given a prompt and two chatbot answers, which one did the human prefer? It is the question a reward model answers inside RLHF. Yet across 1,849 teams, the winning private score was only about 0.13 better than guessing one third for everything. Human preference is noisy, and the top 20 fought over the last few hundredths of a nat.

This post synthesises the published writeups of the top 20 private-leaderboard teams. Ten of the twenty published. The other ten are named in the scorecard and marked "no writeup"; we do not guess their methods.

## 1. The task in 60 seconds

![The notebook budget: two T4 GPUs, nine hours, internet off, about 25,000 hidden test rows split 55 percent public and 45 percent private](/imgs/blogs/kaggle-lmsys-chatbot-arena-solutions-2.webp)

Each row is one Chatbot Arena battle: a user types a prompt, two anonymous models answer, and the user votes A, B or tie. Kaggle gave **55k** training battles with columns `id, model_a, model_b, prompt, response_a, response_b` and three one-hot targets `winner_model_a`, `winner_model_b`, `winner_tie`. The text fields are JSON lists, because a battle can run for several turns. The model names appear in train only, so you cannot lean on "GPT-4 usually wins" at test time.

The hidden test set held about **25,000** rows. Your submission was a notebook that ran offline: internet off, at most **9 hours** of CPU or GPU time, with Kaggle's **2x T4** as the GPU. External data was allowed, so most training happened off Kaggle; the notebook only had to *run* the result. The overview warned about position, verbosity and self-enhancement bias. Prize money was USD 100,000: 25k for 1st, 20k each for 2nd to 4th, 15k for 5th.

![An illustrative battle row with one-hot vote, and the same row with responses A and B swapped and the label mirrored](/imgs/blogs/kaggle-lmsys-chatbot-arena-solutions-3.webp)

The row above is a made-up example in the dataset's shape, not a real row. Notice the right panel. If you swap the two responses and swap the label, you get a new training example that encodes exactly the same human judgement. That symmetry is the single idea every published top-20 solution used, and it shows up three ways: as data augmentation, as a second model trained on swapped inputs, and as test-time augmentation (predict both orders, un-swap, average).

## 2. The metric, with a worked toy score

Scoring was multiclass log loss over the three columns, lower is better:

$$
\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c \in \{A,B,\text{tie}\}} y_{i,c}\,\ln p_{i,c}
$$

where $N$ is the number of test rows, $y_{i,c}$ is 1 for the class the human chose and 0 otherwise, and $p_{i,c}$ is your predicted probability for class $c$. Since only one $y_{i,c}$ is 1, each row contributes $-\ln p$ of the class that actually happened.

![Per-row log loss bars for confident hits, a uniform guess and confident misses, with reference lines at ln 3 and the winning private score](/imgs/blogs/kaggle-lmsys-chatbot-arena-solutions-4.webp)

Work it on five rows. A confident hit at $p=0.88$ costs $-\ln 0.88 = 0.128$. A decent hit at $p=0.60$ costs ${0.511}$. A uniform guess, one third for each class, costs $\ln 3 = 1.099$ no matter what the human did. A confident miss, where you gave the true class only ${0.05}$, costs ${2.996}$, and at ${0.04}$ it costs ${3.219}$. Measured against the uniform guess, each confident hit saves ${1.099 - 0.128 = 0.971}$ and each confident miss at ${0.05}$ loses ${2.996 - 1.099 = 1.897}$. One miss erases almost two hits, and the asymmetry only grows as $p$ shrinks.

The chart's reference lines frame the whole competition. Uniform guessing scores ${1.0986}$; first place scored ${0.96898}$ on the private set, a gap of ${0.1296}$. That is what label noise looks like. Valentin Werner's forum thread made the point early: the same pair of answers draws different votes from different humans, so the log-loss floor is high.

Two consequences follow. **Never predict a hard 0 or 1**: CAST (4th) gave the non-empty side of an empty-response battle only 0.88, citing this sensitivity plus label noise. And **calibrate**: the 13th-place solution tuned a temperature on validation and renormalised predictions to the training class priors.

## 3. What made it hard

Three constraints decided the shape of every solution.

**The judgement needs a big model; the box only fits a small one.** Two T4s, with no FlashAttention 2 kernels for their sm75 architecture, had to push 25,000 long multi-turn battles through the model in under nine hours. A 70B model does not fit that budget; a 9B one does, if quantised and batched carefully. Every top solution answers "how do we get 70B quality out of a 9B forward pass?"

**Conversations are long.** Prompts and responses can each run thousands of tokens over several turns. The top teams trained at 1,024 to about 4,340 tokens and inferred at up to 4k or 8k. What you cut matters, and the writeups disagree on where to cut.

**The leaderboard broke a week before the end.** On 4 to 5 August 2024 Yurnero's team noticed the top public score had "plunged" and found the cause within a day. A random submission of one third everywhere scores 1.098, which is $\ln 3$. The same submission with leak-altered rows scored 1.030. Host Lisa Dunlap traced it: a public leaderboard-analysis notebook published by LMSYS contained past battles' metadata and human labels (no chat text), including battles that were in Kaggle's train and test sets, matchable through fields such as token counts. Kaggle kept the deadline, replaced the private test set with unpublished battles and re-ran every selected submission. The public number-one, "Leak User", scored **0.29588** on public and finished **174th** on private.

![Dumbbell chart of public rank versus private rank for the top 20, with Leak User falling from public 1 to private 174](/imgs/blogs/kaggle-lmsys-chatbot-arena-solutions-5.webp)

The chart above needs one warning before you read it, and it applies to every public number in this post. **Public scores (about 0.83 to 0.88) were computed on the original, leak-affected public split. Private scores (0.969 to 0.986) were computed on the replacement set, which is a different and harder set of battles.** So "public 104 to private 18" is not an overfitting shake-up; it compares ranks on two different tests. Among no-leak teams, ranks were stable: the eventual top three sat at public 6, 5 and 8, and many top teams appended "(no leak)" to their names.

## 4. The top-20 scorecard

Private scores are on the replacement test set; public ranks are on the original set. 1,688 teams appear on the final board; gold went to the top 13.

| Rank | Team | Private | Public rank | Key idea | Source |
|---|---|---|---|---|---|
| 1 | BlackPearl (no leak), sayoulala | 0.96898 | 6 | Distil Llama-3-70B + Qwen2-72B into Gemma-2-9B, average 5 fold LoRAs, GPTQ 8-bit | [writeup](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629) |
| 2 | [Rist] pref (no leak) | 0.97425 | 5 | Full-parameter Gemma-2-9B/27B + ArmoRM, "full swap", 240k pseudo-labels, Triton kernels on T4 | [writeup](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685) |
| 3 | In The Arena (no leak) | 0.97770 | 8 | Start from reward models; 500k+ vLLM-generated pairs, pseudo-labelled | [writeup](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766) |
| 4 | CAST (no leak) | 0.98031 | 15 | Gemma-2-9b-it LoRA, per-turn truncation budget, UltraFeedback pseudo-labels, empty/identical rules | [writeup](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/529067) |
| 5 | Team Danube | 0.98129 | 14 | Reward-pretrain Gemma-2-9B on UltraFeedback win/loss; two runs, one on swapped order; INT8 | [writeup](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527669) |
| 6 | whak (no leak) | 0.98156 | 23 | unknown | no writeup |
| 7 | Qlora is all we need | 0.98162 | 34 | unknown | no writeup |
| 8 | Safari | 0.98274 | 61 | unknown | no writeup |
| 9 | Ebi (no leak) | 0.98280 | 21 | Second Gemma trained on swapped data, 45k chat-1m pseudo-labels, length-gated Llama-3 reward model | [writeup](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527704) |
| 10 | Ilya Novoselskiy | 0.98310 | 86 | unknown | no writeup |
| 11 | LLMT | 0.98353 | 65 | unknown | no writeup |
| 12 | tq (no leak) | 0.98379 | 22 | unknown | no writeup |
| 13 | gpt2-chatbot-which-trusted-cv | 0.98392 | 57 | Three-stage transfer: 700k + 900k pair full-FT pretraining, then QLoRA; refusal and prior post-processing | [writeup](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/540876) |
| 14 | NHA | 0.98392 | 9 | unknown | no writeup |
| 15 | M (no leak) | 0.98427 | 58 | unknown | no writeup |
| 16 | Chris Deotte (no leak) | 0.98532 | 18 | Public QLoRA notebook upgraded by measured LoRA sweeps, r=1024, 8-bit inference, truncate left | [writeup](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527596) |
| 17 | less is more (no leak) | 0.98541 | 29 | unknown (a member posted insight threads) | no writeup |
| 18 | Shelton (no leak) | 0.98551 | 104 | Public Gemma-2 QLoRA notebook, all layers unfrozen, r=64, more data, 2,100-token TTA | [writeup](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527595) |
| 19 | MRIC (no leak) | 0.98564 | 28 | Public notebook + extra LoRA targets, 3,072-token training, deeper head | [writeup](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/528288) |
| 20 | luckydog (no leak) | 0.98585 | 16 | unknown | no writeup |

Ranks 13 and 14 show the same displayed score; the API orders them 13 then 14. The whole top 20 spans only 0.01687 of log loss. The gap from 1st to 2nd (0.00527) is larger than the gap from 5th to 20th (0.00456): the winner had something the others did not.

## 5. The top-20 landscape

![Technique matrix: twenty rank columns, ten with writeups and ten shaded as no writeup, rows for Gemma-2-9B, A/B swap, extra data, reward pretraining, truncation, pseudo-labels, 8-bit, full fine-tuning, post-processing and distillation](/imgs/blogs/kaggle-lmsys-chatbot-arena-solutions-6.webp)

Read the matrix row by row. Two rows are solid across every writeup: **10 of 10** ran Gemma-2-9B in the final submission and **10 of 10** used A/B swapping. Gemma-2 was released mid-competition, and once it appeared it became the default; Yixiao Yuan's "magic behind Gemma2" thread captured the moment. CAST (4th) tested Llama-3-8B and Llama-3.1-8B against `gemma-2-9b-it` and found Gemma "much better" than both.

**Table stakes**, shared by nearly everyone: Gemma-2-9B with LoRA on all or most linear layers at rank 32 or above; one epoch of fine-tuning; the extra "33k" Arena set that Abdullah Meda posted (about 21k rows after deduplication, used by **8 of 10**, the other two not stating); A/B symmetry; length-sorted batched inference split across the two T4s.

**What separated the top five from 6 to 20** was *preference signal injected before the competition fine-tune*. Ranks 1, 2, 3 and 5 all had at least one of: a larger teacher's logits (1st), hundreds of thousands of pseudo-labelled pairs (2nd and 3rd), or a reward-model starting point or reward-pretraining stage (1st, 2nd via ArmoRM, 3rd, 5th). That row reads **6 of 10**, and all six sit in the gold zone (ranks 1 to 13). The writeups from ranks 16, 18 and 19 are upgrades of public notebooks with none of this.

The writeups contradict each other on almost every second-tier technique:

| Technique | Worked for | Did not work for |
|---|---|---|
| Pseudo-labelling | 2nd, 3rd (decisive), 9th, 4th | 13th; 5th placed without it |
| Distillation from bigger models | 1st (the winning idea) | 3rd, 13th, 19th could not make it work |
| UltraFeedback | 1st, 5th as pretraining; 4th as pseudo-labels | 3rd (no better than ORPO-DPO mix); 17th member found it hurt as direct training data |
| Label smoothing | 3rd (0.03) | 13th |
| Weight / LoRA averaging | 1st averaged 5 fold LoRAs | 13th: LoRA averaging, SWA and EMA all failed |
| Model-identity auxiliary heads | 16th (0.891 to 0.890) | 3rd, 13th |
| 8-bit speed-ups | 1st (GPTQ), 5th, 9th, 16th (bitsandbytes) | 13th found little speed gain for classification |

A plausible reconciliation for UltraFeedback, stated by no single team: it helps as a warm-up stage and hurts inside the final fine-tune, because its prompts come from evaluation sets rather than Arena users, the provenance problem justin1357 (17th) pointed at.

## 6. The winning solution: "Distill is all you need"

![BlackPearl pipeline: UltraFeedback post-pretraining, two 70B-class teachers over five folds, distillation into Gemma-2-9B, averaging five fold LoRAs, GPTQ 8-bit, inference on two T4s](/imgs/blogs/kaggle-lmsys-chatbot-arena-solutions-7.webp)

sayoulala's insight: you cannot run 70B in the notebook, but you can run it *on the training set*, off Kaggle, and teach a 9B model to reproduce its full probability distribution, not just the hard human label. The teacher's probabilities carry information the one-hot label throws away: "this was a near tie, B slightly better" is a far richer target than "B". If you want the general mechanism, our posts on [knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals) and [distillation in LLMs](/blog/machine-learning/large-language-model/distillation-in-llm) build it from scratch. Here is the pipeline step by step, from his writeup.

**Data.** Kaggle train, the 33k Arena rows, and "ut" data: UltraFeedback converted to competition format, the 157k set Darek Kleczek shared in the forum. Note UltraFeedback's labels come from GPT-4 as a judge, not humans.

**Backbones.** Llama-3-70B, Qwen2-72B and Gemma-2-9B, each as `AutoModelForSequenceClassification` with a 3-way head. LoRA for the 9B and QLoRA for the 70B and 72B; all linear layers; **r=64, alpha=128**; max length 1,024; 2 epochs; global batch 64.

**Step 1, post-pretrain.** One epoch on UltraFeedback for all three models at lr 1e-5: a reward-pretraining stage before any Arena data.

**Step 2, teacher logits.** Five folds: train on four fifths of Kaggle plus the 33k, hold out one fifth. Do this for both 70B-class teachers, then infer the class distribution on the training rows.

**Step 3, distil.** Fine-tune Gemma-2-9B on the teachers' logits with a distillation loss at lr 5e-5. The writeup says "at least three loss for training" without spelling out every term.

**Step 4, merge.** Average the LoRA layers of the five fold students into one adapter. One forward pass at test time instead of five.

**Step 5, quantise.** GPTQ 8-bit, then test-time augmentation at length 2,000 in the submission. Our [GPTQ and AWQ post](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq) covers why 8-bit weight-only quantisation of a 9B model costs almost nothing in accuracy.

The cross-validation numbers are the proof that the idea works. Per fold, Qwen2-72B scored 0.875, 0.881, 0.869, 0.880, 0.875 (mean 0.876); Llama-3-70B scored 0.874, 0.877, 0.877, 0.873, 0.873 (mean 0.8748); the distilled Gemma-9B scored 0.862, 0.876, 0.858, 0.872, 0.868 (mean 0.8672). **The student beat both teachers in four of five folds and on average.** Soft targets from two different teachers regularise the student, which ends up acting like an ensemble of both. Public LB was 0.882 (0.876 with TTA), private 0.96898, and he called CV and LB "very consistent". A secondary source (Nebius's competition review) says training used 8x A100 80GB; the Kaggle post does not state it, so treat that as unverified.

Below is an illustrative toy of the distillation objective and the swap TTA. It is our sketch of the idea, not BlackPearl's code (his repository is linked in Sources). Because the writeup does not list the exact loss terms, the mix here, hard-label cross-entropy plus a KL term to each teacher, is an assumption.

```python
# Illustrative toy, NOT the team's code: 3-way preference distillation + swap TTA.
import torch
import torch.nn.functional as F

def distill_loss(student_logits, teacher_probs_list, hard_labels, alpha=0.5, T=1.0):
    """student_logits: [B, 3] for (A wins, B wins, tie).
    teacher_probs_list: list of [B, 3] class distributions from the 70B-class teachers.
    hard_labels: [B] int64 in {0, 1, 2}."""
    ce = F.cross_entropy(student_logits, hard_labels)
    log_p = F.log_softmax(student_logits / T, dim=-1)
    kl = sum(
        F.kl_div(log_p, t.clamp_min(1e-6), reduction="batchmean") * T * T
        for t in teacher_probs_list
    ) / len(teacher_probs_list)
    return alpha * ce + (1 - alpha) * kl

SWAP = torch.tensor([1, 0, 2])  # A-wins <-> B-wins, tie stays tie

@torch.no_grad()
def predict_with_swap_tta(model, batch_ab, batch_ba):
    """batch_ab: inputs in (prompt, A, B) order; batch_ba: same rows with A and B swapped."""
    p_ab = model(**batch_ab).logits.softmax(-1)
    p_ba = model(**batch_ba).logits.softmax(-1)[:, SWAP]  # un-swap the columns
    return 0.5 * (p_ab + p_ba)

# Five fold students -> one adapter: average matching LoRA tensors.
def average_lora_state_dicts(state_dicts):
    keys = state_dicts[0].keys()
    return {k: torch.stack([sd[k].float() for sd in state_dicts]).mean(0) for k in keys}
```

A caution on the last function: averaging LoRA A and B matrices separately is not the same as averaging the low-rank products $BA$, so it only works when the folds start from the same initialisation and stay close. It evidently worked for BlackPearl; the 13th-place team reports LoRA averaging did *not* work for them.

## 7. 2nd and 3rd in depth

### 2nd: [Rist] pref, full fine-tuning and a kernel rewrite

tascj, sakaku and kapenon took the opposite road on model fit: **no LoRA at all**. They trained every parameter in BF16, using an optimizer with **Kahan summation** to keep BF16 updates accurate, which let a 7B model train on one A100 80G and the 9B on two. The last ten days ran on 4x A100 80G.

Their best single trick is "**full swap**": put each sample *and* its A/B-swapped twin into the same optimizer step, accumulating both gradients. Against random swapping it gave a stable **-0.003** log loss on Gemma-2-9B. Adding prompt-A-B and prompt-A-prompt-B input formats the same way gave another -0.001. They used the MT-Bench judge instructions as the prompt template and replaced the default classification head with `Dropout -> Linear(h, h/2) -> Dropout -> GELU -> Linear(h/2, 3)`, re-initialised after loading, because the default head init produced a high early loss.

Stage 1 fine-tuned Gemma-2-9B (val 0.891), Gemma-2-27B (0.883) and the reward model `ArmoRM-Llama3-8B` (0.899), averaging 0.876. Stage 2 pseudo-labelled **240k** rows (110k from lmsys-chat-1m, 130k screened external rows) with that ensemble and retrained Gemma-2-9B and ArmoRM; stage 3 continued on the 55k plus 21k. The final 2:1 ensemble scored 0.868 on the old public board.

On inference, they split layers across the two T4s and pipelined them, removed padding entirely, and ported Triton kernels from lightllm and lmdeploy for softcapped attention, RMSNorm and rotary embeddings. That is how a 9B plus an 8B fit in nine hours with no 8-bit quantisation in the writeup.

### 3rd: In The Arena, reward models and half a million new battles

Raja Biswas and Mark Tenenholtz had two ideas. **First, start from reward models, not chat models.** They swapped `gemma-2-9b-it` for `sfairXC/FsfairX-Gemma2-RM-v0.1` and Llama-3.1-8B-Instruct for `RLHFlow/pair-preference-model-LLaMA3-8B`. The Gemma reward model alone reached about 0.895 on LB. A model already trained to score answers is a far better starting point for a preference task than one trained to chat; our post on [reward modeling from human preferences](/blog/machine-learning/reinforcement-learning/reward-modeling-from-human-preferences) explains what those checkpoints learned.

**Second, manufacture data.** They took prompts from lmsys-chat-1m, generated responses with a variety of open LLMs through vLLM (temperature 0.7, top_p 0.9), batching by number of turns rather than length for throughput, and soft-labelled the resulting **500k+** pairs with a baseline ensemble. Gemma2-RM trained with these pseudo-labels reached 0.880 without TTA at 8k length. A second round of pseudo-labelling added little, and UltraFeedback was no better than the ORPO-DPO mix they also used.

Details that transfer: Mark used QLoRA r=64, alpha=16; adding new tag tokens to the vocabulary hurt; **disabling Gemma-2's logit softcapping** improved CV and LB by about 0.002; A/B flip TTA was worth about 0.007. Raja used label smoothing 0.03, removed about 300 high-loss rows, and for chats over four turns kept the first two and last two. Their candid failures matter as much: they could not train Gemma-2-27B well, and were "totally convinced" distillation was the key "but we couldn't get it right". First place proved them right.

## 8. Ranks 4 to 20, grouped by approach

### Family A: reward pretraining and pseudo-labels (4, 5, 9, 13)

**4th, CAST.** Gemma-2-9b-it LoRA r=64; stage 2 added pseudo-labels for 30k UltraFeedback rows and retrained from scratch. Their truncation is the most careful in the top 20: when the last round of a "Round k" template overflows, the remaining budget splits 20% prompt, 40% A, 40% B with an ellipsis, dropping the round entirely below 80 tokens. Post-processing set empty responses to [0.04, 0.88, 0.08] for (empty, non-empty, tie) and identical responses to [0.06, 0.06, 0.88].

**5th, Team Danube.** Psi, dott and Pascal Pfeiffer used no pseudo-labels, no distillation, no lmsys-chat-1m. They noticed that RLHFlow reward models fine-tuned better than the base and instruct versions of the same backbone, so they reproduced that recipe: **pre-train Gemma-2-9B on all of UltraFeedback with binary win/loss**, then fine-tune. That gave "up to 20 points" (0.020) on CV and public LB. The final was the same config trained twice, one on swapped order, inferred in INT8 on the two T4s. They used 4k context instead of 8k "to be safe for the re-run". Fifth without pseudo-labels is the top 20's best evidence that reward pretraining alone carries most of what pseudo-labels gave others.

**9th, Ebi.** Instead of swap TTA, Ebi trained a *second* Gemma on swapped data and ensembled (0.890 to 0.887 public). He added about 45k pseudo-labelled lmsys-chat-1m rows, then RLHFlow's pair-preference Llama-3-8B, but three models did not fit in nine hours, so **Llama-3 ran only on rows of at most 512 or at least 1,500 tokens**. That member left public unchanged at 0.881 yet improved private from 0.9859 to 0.9828: diversity the old public set did not reward.

**13th, gpt2-chatbot-which-trusted-cv.** heartkilla, solo gold, ran three-stage transfer: full fine-tuning on about 700k binary pairs from RLHFlow's preference collection, then about 900k pairs from Nectar's lmsys-chat-1m subset, then 4-bit QLoRA on competition data plus toxic-dpo and HelpSteer2. rsLoRA gave about 0.02 from the start. Post-processing: temperature scaling, "null" responses set to one third each, refusal phrases ("sorry", "as an AI") boosting the other side by 1.05, and priors renormalised. That 1.6M pretraining pairs still landed 13th suggests distribution (GPT-4 rankings, not Arena humans) mattered more than size.

### Family B: public-notebook upgrades (16, 18, 19)

![Chris Deotte's public-leaderboard ladder from 0.941 to 0.885 across ten measured changes, ending with left truncation](/imgs/blogs/kaggle-lmsys-chatbot-arena-solutions-8.webp)

**16th, Chris Deotte.** The competition's most-voted post is a reproducible ladder from a public QLoRA notebook at 0.941 to 0.885, every step measured on the old public board. Read the deltas: the first three changes are worth 0.038 together, the next six only 0.013, and then truncating from the left, keeping the end of the conversation where the final answers live, is worth 0.005 on its own. His LoRA method is worth stealing: pick the modules, tune the head LR, sweep alpha at r=16, then sweep r with alpha fixed. He measured that the best alpha moves with learning rate (alpha 4 at lr 2e-4, 64 at 6e-5, 192 at 2e-5), and trained in fp16 LoRA with 8-bit inference, about 10% faster than 4-bit QLoRA, which bought the longer inference length. Our [LoRA debugging post](/blog/machine-learning/debugging-training/debugging-lora-and-peft) covers why alpha and r interact this way.

**18th, Shelton.** Ray changed only the config of emiz6413's public Gemma-2 QLoRA notebook: all layers unfrozen (the public one froze the first 16), r=64 and alpha=32, lr 1e-5, extra external data the post does not name, and swap TTA at 2,100 tokens. A public notebook plus a few config changes reached rank 18.

**19th, MRIC.** Same public base, plus `o_proj` and `gate_proj` as LoRA targets (CV 0.921 to 0.908), 3,072-token training, no frozen layers, the deduplicated 21k, r=64 and a deeper custom head. They could not get two-stage fine-tuning to converge, distillation to work, or full fine-tuning to reach good CV.

### The no-writeup ten

Ranks 6, 7, 8, 10, 11, 12, 14, 15, 17 and 20 published no solution that the forum or the API lists; a team name like "Qlora is all we need" is not evidence of method. whak's soso asked teams in the forum not to share solutions until the reproduce phase, so leak users could not copy them. Ilya Novoselskiy's climb from public 86 to private 10 is the largest inside the top ten, and unexplained.

### The best contrasting entry

Put 5th next to 1st. BlackPearl spent the most compute in the field on two 70B-class teachers. Danube spent none on teachers or pseudo-labels and reproduced one known recipe, reward pretraining, then placed 0.0123 behind. If you have a weekend rather than a cluster, Danube's writeup is the one to copy.

## 9. What the forum taught everyone

| Thread | Takeaway |
|---|---|
| Darek Kleczek, "157k human preference ratings" | UltraFeedback in competition format; pretraining for 1st and 5th, pseudo-labels for 4th |
| Abdullah Meda, "Additional 21k labelled conversations" | The 33k Arena set nearly every top team used |
| Yixiao Yuan, "The magic behind Gemma2" | Gemma-2-9B became the default backbone |
| Marilia Prata and takaito on Bradley-Terry / Elo | Model identity predicts wins, but names exist in train only |
| Yurnero; Lisa Dunlap (host) | Leak found by probing, then root cause and replacement set |

The decisive external data and the default backbone both arrived through forum posts, and so did the leak. Before the host explained it, threads titled "Cheat Issue" and "Suspicious Surge of Novice Accounts" were already raising alarms, and justin1357 (17th) posted the thread that surfaced the leaked source. The forum was the competition's real infrastructure.

## 10. What did not work

Aggregated across the ten top-20 writeups, with who reported it:

| Idea | Reported by |
|---|---|
| Gemma-2-27B (could not train well, or distil from) | 3rd, 13th |
| Distillation from a larger model | 3rd, 13th, 19th (13th: "should have tried distilling from an ensemble") |
| Llama-3.1-8B as backbone (worse than Llama-3 or Gemma-2) | 3rd, 4th, 17th member, 19th |
| Auxiliary model-identity loss | 3rd, 13th |
| Training new tag-token embeddings | 3rd |
| DoRA, LoftQ, LoRA+ (no clear gain) | 3rd, 13th |
| Label smoothing; Lion and Sophia optimizers; max/GRU pooling | 13th |
| Freezing layers | 13th; 16th, 18th and 19th all unfroze the public notebook's 16 |
| TF-IDF + LightGBM or DeBERTa blends | 13th, 17th member |
| Arena-Hard and MT-Bench as extra data | 13th |
| UltraFeedback as direct training data | 17th member |
| Llama-3.1-405B as judge or generator of synthetic data | 16th (labelling variant reached about 0.95 locally, did not help) |
| Two-stage fine-tuning; full-parameter FT | 19th |
| Treating LLM-classified subjective prompts as 1/3 | 19th (Eltayeb, using Qwen2-72B as classifier) |
| Long-to-short length-ordered training; expanding multi-turn chats into cumulative samples | 19th (Eltayeb) |
| A second pseudo-labelling round | 3rd |

Eltayeb's number deserves a sentence: replacing every row whose out-of-fold loss was worse than uniform with one third would give CV 0.717, so much of the loss lives in unpredictable rows, yet spotting those rows ahead of time with a 72B classifier did not help.

## 11. Lessons that transfer

1. **Distil big into small when inference is capped.** Train the large model where compute is cheap, transfer its full distribution, and ship the small one. The student can beat its teachers (1st, CV 0.8672 versus 0.8748 and 0.876).
2. **Start from the closest pretrained task.** Reward models or a reward-pretraining stage beat chat starts by large margins: about 0.020 for Danube, and the Gemma reward model alone at about 0.895 for 3rd. Six of ten writeups did this. Our [LoRA and QLoRA for RLHF post](/blog/machine-learning/reinforcement-learning/lora-qlora-for-rlhf) shows the same initialisation logic on the training side.
3. **Symmetry is free data.** Swap A and B in training (in the same step, for -0.003) and at inference (about 0.007 for 3rd). Ten of ten used it.
4. **Pseudo-label at scale, from the right distribution.** 240k to 500k self-generated pairs drove 2nd and 3rd; 1.6M pairs of GPT-4-ranked data still finished 13th. Arena-like prompts mattered more than volume.
5. **Decide what to throw away.** Keep the end of the conversation (16th: 0.005), or budget tokens per field (4th). Five of ten writeups say so explicitly.
6. **Trust CV when the board is compromised.** The 13th team put the lesson in its name: "which trusted CV".

### If you enter a competition like this

- Compute the uniform-guess score on day one ($\ln 3$ here) so you know how much signal exists.
- Group folds by prompt (2nd used `StratifiedGroupKFold`), and treat any unexplained public jump as a leak until proven otherwise.
- Re-sweep backbones whenever a new open model drops.
- Look for a checkpoint already trained on your task family before fine-tuning a chat model.
- Budget the notebook backwards from nine hours: length-sort, split across both GPUs, quantise to 8-bit, then spend leftover time on longer context or a second model.
- Never predict exactly 0 or 1 under log loss; calibrate on validation.

The next competition in this track, the [WSDM Cup multilingual arena](/blog/machine-learning/learn-from-kaggle/kaggle-wsdm-cup-multilingual-arena-solutions), reran the question with a binary, multilingual target; its winners built directly on the LMSYS 1st and 2nd place recipes, and both sayoulala and Raja Biswas came back. Background on notebook budgets: [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work). A contest decided by retrieval instead: the [LLM Science Exam solutions](/blog/machine-learning/learn-from-kaggle/kaggle-llm-science-exam-solutions).

## Sources

- Competition: https://www.kaggle.com/competitions/lmsys-chatbot-arena
- 1st, BlackPearl: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629 · code https://github.com/shyoulala/LMSYS_BlackPearl
- 2nd, [Rist] pref: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685 · code https://github.com/tascj/kaggle-lmsys-chatbot-arena
- 3rd, In The Arena: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766 · code https://github.com/mtenenholtz/lmsys-chatbot-arena-solution and https://github.com/rbiswasfc/lmsys-arena
- 4th, CAST: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/529067
- 5th, Team Danube: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527669 · inference https://www.kaggle.com/code/philippsinger/cool-crab-v44-rev-ff-sub/
- 9th, Ebi: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527704
- 13th, gpt2-chatbot-which-trusted-cv: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/540876
- 16th, Chris Deotte: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527596 · notebooks https://www.kaggle.com/code/cdeotte/16th-place-train-1-of-3
- 18th, Shelton: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527595 · inference https://www.kaggle.com/code/faykudbq/18th-inference-gemma2-max-length-2100-tta
- 19th, MRIC: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/528288 · failed experiments https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527651 · weights https://huggingface.co/RoschildRui/LMSYS_Gemma2_QLoRA_ft
- Leak: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/524079 · https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/524374 · https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/524356
- External data thread: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/499756
- 17th member's threads: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/523527
- whak's request: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/524181
- Secondary (unverified hardware claim): https://nebius.com/blog/posts/chatbot-arena-competition-review

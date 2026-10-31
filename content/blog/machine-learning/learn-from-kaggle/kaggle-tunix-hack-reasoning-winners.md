---
title: "Google Tunix Hack: how 15 winners taught a 2B Gemma to show its work"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "A judged Kaggle hackathon asked 319 teams to post-train Gemma 2 2B or Gemma 3 1B to reason in tags inside one 9-hour TPU session. All 15 prize winners, their GRPO variants, reward designs and data, and what the rubric actually rewarded."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "hackathon", "grpo", "reinforcement-learning", "reward-design", "gemma", "tunix", "tpu", "distillation"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 22
---

> **TL;DR.** The Google Tunix Hack (Kaggle, 11 November 2025 to 12 January 2026, 319 teams, 100,000 USD) asked for a Gemma 2 2B or Gemma 3 1B that writes `<reasoning>…</reasoning><answer>…</answer>`, post-trained with Google's JAX library Tunix inside **one 9-hour Kaggle TPU session**. Judges scored the notebook (35), the model on held-out prompts with an LLM judge plus humans (45), and a 3-minute video (20). Six entries were placed; nine got unranked Honorable Mentions. **The decisive lever was reward design for answers nobody can check:** five of the six placed entries found a dense, cheap signal for creative writing and summaries. **The one lesson:** once SFT has taught the format, the format reward goes flat, and whatever you reward next *is* your model.

![The brief on one card: a Kaggle TPU v5e-8 for 9 hours, a Gemma 2 2B or Gemma 3 1B, a tagged output under 1K tokens, four deliverables, 319 teams and a 100,000 USD pool split across six places plus nine Honorable Mentions](/imgs/blogs/kaggle-tunix-hack-reasoning-winners-1.webp)

The card above is the whole contest. There is no leaderboard: a panel read the notebooks, sampled the models and watched the videos. A slightly better model that could not prove it, or could not run in one session, lost to one that did both.

Every number below is **the team's own claim** unless marked as the host's. Judges never published per-team scores or the held-out prompts. The six placed entries are ranked; the nine Honorable Mentions (HMs) are listed in host order, which is not a rank.

The [Nemotron Reasoning Challenge](/blog/machine-learning/learn-from-kaggle/kaggle-nvidia-nemotron-reasoning-challenge-solutions) is its scored cousin; [AIMO Progress Prize 2](/blog/machine-learning/learn-from-kaggle/aimo-progress-prize-2-published-solutions) is the opposite extreme.

## 1. The brief in 60 seconds

Told to a friend: "Teach the smallest Gemma to think out loud before it answers, on a borrowed TPU, in nine hours, and explain how."

| Item | Value |
| --- | --- |
| Library | Tunix, Google's JAX-native post-training library; starter = Tunix GRPO demo on Gemma2-2B |
| Model | Gemma 2 2B or Gemma 3 1B in practice |
| Output | `<reasoning>trace</reasoning><answer>…</answer>`, under 1K tokens, English, no tools, no images |
| Hardware | Kaggle TPU v5e-8, 9 h per session, 20 h per week |
| Deliverables | Writeup of at most 1,500 words, public notebook, video of at most 3 minutes, cover image |
| Field | 319 teams, 444 competitors (the Developers Blog's "over 11,000 entrants" presumably counts joiners) |
| Prizes | 1st 30,000, 2nd 25,000, 3rd 15,000, 4th to 6th 10,000 each (USD), plus 9 HMs |

The output format is the brief's real content, and the redrawn sample below shows why it is harder than it looks.

![Two illustrative toy prompts: a pen-price word problem whose tagged answer can be checked against a reference by format and answer rewards, and a two-line rain poem that has no reference at all, so only a dense reward can score its reasoning](/imgs/blogs/kaggle-tunix-hack-reasoning-winners-3.webp)

The math item is easy to reward: check the tags, parse `8`, compare. The poem has tags too, but nothing to compare against, and the held-out set explicitly included creative writing, ideation and summarisation. Every serious entry had to answer one question: **what reward fires on the poem?**

## 2. The judging rubric, with its weights

![The rubric drawn to scale as one 100-point bar: notebook quality 35, model quality 45 on held-out verifiable and non-verifiable prompts, video 20, and a dashed optional 15 beyond the bar for a multi-session model](/imgs/blogs/kaggle-tunix-hack-reasoning-winners-2.webp)

| Criterion | Points | What it covers |
| --- | --- | --- |
| Notebook quality | 35 | Training data, hyperparameters, prompt, strategy |
| Model quality, single session | 45 | The checkpoint the notebook produces in **one** 9 h TPU session, judged by "a combination of LLM-as-a-judge and human judgment" on held-out creative writing, ideation, summarization, math, coding and basic science |
| Video quality | 20 | 3 minutes or less |
| Optional bonus | +15 | A multi-session or private-data model, submitted as a Kaggle Model that loads in Tunix's Gemma code |

Two consequences shaped every winning recipe.

**55 of the base 100 points are about explanation.** Notebook plus video outweigh model quality, which is why the host's reasons read like paper reviews. Between another hour of GRPO and an ablation table, the rubric says take the table.

**Model quality is measured on the single-session checkpoint.** A hackathon about RL was, underneath, a scheduling problem: fit SFT, maybe a preference stage, GRPO, maybe a judge, and export into 540 minutes. The host published reasons, not scores, so I infer none.

## 3. What made it hard

**The judged set was mostly unverifiable.** Only math, coding and arguably basic science have a reference to match. GRPO, the starter's algorithm, needs a scalar reward per sampled answer. For a summary there is none.

**Format rewards saturate.** After a short SFT warm-up the model emits perfect tags nearly every time; Gemma3Think measured 99%. From then on, a format reward gives every sampled answer the same score. GRPO computes advantages *relative to the group*:

$$A_i = \frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}{\mathrm{std}(r_1, \dots, r_G)}$$

where $r_i$ is the reward of the $i$-th of $G$ sampled answers to one prompt. Identical rewards mean zero advantage and zero gradient. The 4-Tier team saw the model learn the format "within tens of samples" while loss sat near 1.0: it was wrapping its base answers in tags and learning nothing else. The [GRPO walkthrough](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) covers the full objective; the toy below shows the failure and the fix.

```python
# Illustrative toy, not any team's code: why a saturated format reward
# gives GRPO nothing to learn from, and why a dense judge score does.
import math

def grpo_advantages(rewards, eps=1e-6):
    """Group-relative advantage: (r - mean) / std within one prompt's group."""
    n = len(rewards)
    mean = sum(rewards) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / n)
    return [round((r - mean) / (std + eps), 2) for r in rewards]

def expected_rating(probs):
    """G-Eval style: expectation over rating tokens 1..5, mapped to [0, 1]."""
    e = sum(p * k for k, p in enumerate(probs, start=1))
    return round((e - 1) / 4, 3)

# Four rollouts for one creative-writing prompt, all perfectly formatted
# after SFT. There is no reference answer, so correctness cannot fire.
format_reward = [1.0, 1.0, 1.0, 1.0]
print("format only     :", grpo_advantages(format_reward))

# Same four rollouts, now also scored by a judge's rating distribution.
judge_probs = [
    [0.05, 0.10, 0.20, 0.40, 0.25],
    [0.00, 0.05, 0.15, 0.45, 0.35],
    [0.10, 0.30, 0.40, 0.15, 0.05],
    [0.00, 0.00, 0.10, 0.50, 0.40],
]
dense = [expected_rating(p) for p in judge_probs]
print("judge scores    :", dense)
total = [f + d for f, d in zip(format_reward, dense)]
print("format + judge  :", grpo_advantages(total))
# Argmax rating would give rollouts 1, 2 and 4 the same score of 4.
print("argmax ratings  :", [p.index(max(p)) + 1 for p in judge_probs])
```

```console
format only     : [0.0, 0.0, 0.0, 0.0]
judge scores    : [0.675, 0.775, 0.438, 0.825]
format + judge  : [-0.02, 0.65, -1.61, 0.99]
argmax ratings  : [4, 4, 3, 4]
```

Format alone: four zeros. Add a dense score and the group spreads from −1.61 to 0.99. Note the last line too: a judge that returned only its top rating would tie three of the four rollouts at 4. Hold that thought for 1st place.

**The compute had to hold everything at once.** Policy, reference, KV cache and any judge share one v5e-8. Yuchen20 wrote that an in-process judge "would not fit"; 1st place proved otherwise.

## 4. The winners table: all 15

| Place | Team | Project | Base | Recipe in one line | Source |
| --- | --- | --- | --- | --- | --- |
| 1st | Dubai Chewy Cookie | G-RaR | Gemma 2 2B IT | LoRA SFT, then GRPO with a rubric reward from an on-TPU Gemma-3-12B judge | [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1768107128907) |
| 2nd | Distilled Santa (David Babu) | Pinocchio-1B | Gemma 3 1B | SFT on gpt-oss-120B traces, SimPO, GRPO with an async Gemini judge | [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1767940008062) |
| 3rd | Kevin Power | IDEA-E | Gemma 2 2B IT | GRPO with a TF-IDF-to-teacher reward and a 4-phase curriculum | [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/idea-e-distillation-with-curriculum-guided-grpo) |
| 4th | SandSpec | Tunix Hackathon | Gemma 2 2B IT | SFT, failure-mined DPO, two GRPO stages in 2 h 17 min | 9-word writeup; [host notebook copy](https://www.kaggle.com/code/windmaple/mattys737-sandeeplleb-submission) |
| 5th | Shanai Nair | On-Policy Distillation | Gemma 3 1B | Offline 4B teacher; student SFT then OPD | [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/eliciting-reasoning-via-on-policy-distillation) |
| 6th | TheItCrow | Gemma2-Deep | Gemma 2 2B IT | GRPO only, 17 reward functions and a trained reasoning RM | [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/theitcrow-gemma2-deep-1767955818518) |
| HM | One_Horizon | Gemmatron-Tunix | Gemma 2 2B IT | SFT, GRPO, then self-distill best-of-N back into SFT | [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/gemmatron-tunix) |
| HM | Yuchen20 | SFT First, DPO Second | Gemma 2 2B IT | No RL: 8K-token SFT, then DPO | [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1763082571151) |
| HM | TraceCraft | Precision Alignment | Gemma 3 1B IT | GRPO only, patched to 2-GRPO plus Dr. GRPO, no KL | [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1763040229455) |
| HM | Green | Optimized Tunix | Gemma 2 2B IT | No RL: forked Tunix for full fine-tune at 4K context, then SimPO | [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/optimized-tunix) |
| HM | OINK BOINK | Project Glassbox | Gemma 2 2B base | SFT, GRPO, best-of-N self-distillation | [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1764443807640) |
| HM | Keegan Carey | Gemma3Think | Gemma 3 1B IT | Short Qwen3-14B traces, GRPO with a category-routed embedding reward | [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/gemma3think) |
| HM | chenchienyi et al. | Rubric-Guided Writing | Gemma 3 1B | SFT, GRPO with tiered writing rubrics and a discounted math reward | [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1766911313944) |
| HM | heissanjay | 4-Tier Hybrid Process Rewards | Gemma 2 2B IT | GRPO only, Gemini judge over five reasoning rubrics | [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1766324142382) |
| HM | DoD | DOD Google Tunix Hack | Gemma 2 2B IT | Two-stage SFT, GRPO with a likelihood reward for open tasks | [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/dod-google-tunix-hack) |

The HMs share one host line: "creativity and technical expertise". The placed entries each got a specific reason, quoted in their sections below.

## 5. The landscape: a feature matrix across all 15

![Fifteen winners as columns, placed entries left and Honorable Mentions right, thirteen recipe features as rows with filled, empty, unknown and not-applicable cells and a count per row: Gemma 2 2B in 10 of 15, SFT warm-start in 11, GRPO-family RL in 13, teacher traces in 13 of 14 known](/imgs/blogs/kaggle-tunix-hack-reasoning-winners-4.webp)

I recounted every row from the cells. Gemma 2 2B won 10 of 15 slots. An SFT warm-start appears in 11 of 15; 3rd, 6th, TraceCraft and 4-Tier went straight to GRPO. GRPO-family RL appears in 13 of 15; Yuchen20 and Green used none. Traces from a larger teacher appear in 13 of the 14 entries where it is known. Among the 13 RL entries, 5 put an LLM judge in the reward and 6 found a dense reward with no judge at all. Only 4 of 15 report numbers against a baseline: 2nd, 3rd, 5th and the Rubric HM.

### Table stakes

A Gemma 2 2B IT or Gemma 3 1B policy with LoRA. Traces distilled from a far bigger model (gpt-oss-120B, DeepSeek-Reasoner, Gemini 2.5 Flash, Qwen3-14B, GLM 4.7) and re-tagged into the XML. A format reward plus a verifiable math reward, usually on GSM8K.

### What separated the six placed entries from the HMs

1. **A dense, cheap reward for unverifiable tasks.** On-TPU judge (1st), async API judge (2nd), TF-IDF to a teacher (3rd), token-level teacher supervision (5th), a learned process reward (6th). Only 4th is the exception, and it was cited for model quality.
2. **Multi-stage pipelines.** 2nd and 4th both run SFT, a preference stage, then GRPO; 5th runs five stages across teacher and student.
3. **Evidence.** Three of the four baseline-reporting entries are placed. Six of nine HMs report no numbers or only training monitors.
4. **Single-session engineering.** Split meshes, per-algorithm mesh choice, checkpoint chaining. Several HMs carry a single-session caveat instead.

### Where the record contradicts itself

- **Is GRPO worth it on a 1B to 2B model in 9 hours?** Yuchen20: no, their GSPO runs reward-hacked and SFT teaches format more sample-efficiently. Green: GRPO is "repeatedly shown to not converge for small models". Thirteen winners, all six placed, used it anyway.
- **Can an LLM judge fit?** Yuchen20 says no; 1st place fit a 12B one. 4-Tier measured under 2% variance from an API judge at temperature 0; Gemma3Think calls LLM judges "very unreliable".
- **Keep the KL penalty?** TraceCraft dropped KL and the reference, saying KL "often hurt when mis-scaled"; 1st, 4th, 6th, Gemmatron and Gemma3Think kept one, with β from 0.04 to 0.2.
- **Is SFT before RL mandatory?** 1st place argues yes; 3rd placed with no SFT stage per its writeup (the Developers Blog recap says otherwise; I follow the writeup).

## 6. 1st place, G-RaR: the judge moved onto the TPU

Dubai Chewy Cookie (HojunLee, R-Jelly, iph1xs, Kim_daeminmin, Jaeyeon Bang) treated "a judge won't fit" as a sharding problem. The host's reason: "clever training setup (training model and judge model both on the same TPU), innovative reward design".

![Eight v5e chips split in two: chips 0 to 3 hold the Gemma-2-2B-IT policy and its reference on a (1,4) mesh, chips 4 to 7 hold a Gemma-3-12B judge on its own (1,4) mesh, rollouts flow right and a G-RaR score in [0,1] flows back, with rubrics pre-generated offline by Gemma-3-27B](/imgs/blogs/kaggle-tunix-hack-reasoning-winners-5.webp)

**The split mesh.** Devices 0 to 3 hold policy and reference as a (1,4) mesh; devices 4 to 7 hold a Gemma-3-12B judge as a second (1,4) mesh. Rollouts are sampled on one half and scored on the other, in parallel, inside the single-session rule.

**The reward.** Format, exact-answer match where a reference exists, and G-RaR, a G-Eval-style rubric reward. Gemma-3-27B **pre-generated three-tier rubrics per question offline**, so rubrics cost nothing at train time. During RL the 12B judge reads rubric and response and produces logits over the rating tokens 1 to 5. Instead of the top token, the team takes the **softmax expectation**, normalised to [0, 1]:

$$r_{\text{G-RaR}} = \operatorname{norm}\Big(\sum_{k=1}^{5} k \cdot p(k)\Big), \quad p(k) = \operatorname{softmax}(\ell)_k$$

where $\ell$ are the judge's logits on the five rating tokens. The map $(E-1)/4$ in my toy is my illustrative choice; the writeup only says the expectation is normalised.

![Two toy responses whose judge distributions both peak at rating 4: response A has expected rating 3.70 and response B 4.10, so the argmax reward ties them at 4 while the expected reward separates them, 0.675 against 0.775](/imgs/blogs/kaggle-tunix-hack-reasoning-winners-6.webp)

This is the section 3 toy again: an argmax judge returns a few integers and groups tie, while the expectation separates "a 4, leaning 5" from "a 4, leaning 3". The rubric keeps the judge consistent about what a 4 means.

**Why SFT first.** The writeup calls it "mandatory": without format, verification rewards stay near zero; unstructured outputs make policy-gradient estimates noisy; cold-start GRPO must learn format and reasoning at once and risks collapse.

**Data.** 69,392 samples, published as HF `comoZ/reasoning-dataset`: SFT 33,142 (OpenThoughts 12k plus chimbiwide creative and general sets, the same author as the Gemma3Think HM) and RL 36,250 across GSM8K, Code Contests, SciQ, WritingPrompts, XSum and Dolly. The halves sum to the total.

## 7. Places 2 to 6 in depth

### 2nd: Pinocchio-1B, three acts in one session (David Babu)

Host: "excellent write-up and notebook, sophisticated pipeline and high quality video". Gemma 3 1B through SFT, SimPO and GRPO inside one session, with custom safetensors converters to hand checkpoints between trainers.

- **SFT distillation.** 70k prompts over 17 domains, traces from gpt-oss-120B. Gemini labels 34 task types, prepended to the trace (a "virtual MoE": the 1B model announces the task type before reasoning). A Gemini 2.0 Flash filter, "NoseCheck", drops bad traces.
- **SimPO.** A custom loss injected into Tunix's `DPOTrainer`: no reference model, length-normalised, label smoothing 0.1, aimed at the "verbosity hack". Dropping the reference also saves memory. The [DPO post](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) explains the loss SimPO modifies.
- **GRPO.** Gemini-judge reward (correctness, reasoning quality 0 to 10, fatal-flaw penalties, format), `num_generations=8`, `beta=0.07`, async judging at about 1,000 evaluations per minute.

On 1,000 held-out samples: accuracy 57% to about 72% (the headline "15% improvement" is 72 − 57 = 15 points, which checks), format 98.7%, reading-comprehension hallucination about 36% to about 8%. The author also reports a math and code regression, an "alignment tax". David Babu also won in the [gpt-oss-20b red-teaming challenge](/blog/machine-learning/learn-from-kaggle/kaggle-gpt-oss-red-teaming-challenge-findings).

### 3rd: IDEA-E, curriculum GRPO with a CPU-side reward (Kevin Power)

Host: "comprehensive ablation study, well designed curriculum learning strategy and excellent report".

- **Scaffold.** Identify, Define, Ethics (a safety gate), Act, Explain.
- **Data.** 27,198 traces over 9 benchmarks from DeepSeek-Reasoner (T=0.4), reviewed by GPT-4o-mini, kept only if correct, with retries at higher temperature and about 400 fixed by hand; each tagged difficulty 1 to 3.
- **GRPO** on Gemma-2-2B-IT: LoRA r=64, α=64, G=2, β=0.08, ε=0.2, LR 1e-6, micro batch 1, 15,000 steps, about 9 hours on a Colab TPU v6e-1. The single-chip ablations score lower than the final Kaggle eight-chip runs.
- **Reward.** A tiered answer score (4.0 exact down to 0), format, a section-marker penalty, and **2.25 × TF-IDF cosine between response and teacher trace**, which runs on CPU and never blocks the TPU.
- **Curriculum.** Format first, then a difficulty ramp, then weak-bucket remediation, then replay.
- **Ablation at 15k samples:** answer-only reward +7.5 points; adding the reasoning reward +28.0; adding the curriculum +32.0.

That ablation is the contest's most useful number: a similarity reward computable in microseconds accounts for most of the gain.

### 4th: SandSpec, the 9-word writeup

SandSpec's writeup reads, in full: "A pipeline that fine-tunes Tunix on how to reason." The 177-second video has no description and its transcript could not be retrieved. Yet the host cited "exceptional model quality and clear write-up". **Everything below comes from a copy of the submission notebook published under the host's account** ([windmaple/mattys737-sandeeplleb-submission](https://www.kaggle.com/code/windmaple/mattys737-sandeeplleb-submission)), not from the team's writeup; that it is the judged run is an inference. The notebook's markdown is the real writeup, and code cells in my source pull were truncated, so some settings may be incomplete.

![Four 9-hour session lanes to scale: SandSpec trains SFT, DPO and two GRPO stages in 2 h 17 min and leaves most of the session unused, IDEA-E fills about 9 h with GRPO, Gemmatron's 2 h SFT plus 7 h GRPO already reach the limit before its SFT-2 stage, and Gemma3Think spends 30 min on SFT and about 7 h on GRPO](/imgs/blogs/kaggle-tunix-hack-reasoning-winners-7.webp)

- **Model.** Gemma 2 2B IT on a v5e-8, LoRA rank 64, alpha 64 on attention and MLP.
- **Stages and stated time.** SFT 22 min on about 3,000 records; DPO 16 min on about 5,800 pairs; GRPO on about 4,500 GSM8K problems, 56 min; GRPO on about 2,500 extra-domain problems, 43 min. Total 137 minutes, a quarter of the cap.
- **SFT data.** Public instruction sets across math, code, writing, summarisation, puzzles and science, rewritten by Gemini-2.5-Flash into the XML, deduplicated, weighted toward instruction following and creative writing.
- **DPO data, the distinctive part.** Round 1: answers from Gemma-2-2B-IT, Gemma-3-4B-IT and Gemini-2.0-Flash ranked by Gemini-2.5-Flash. Round 2: train an offline Gemma on round 1, run it on unseen prompts, harvest its *own* failures and add them as pairs. Preference data aimed at the non-verifiable domains, mined from the model's actual mistakes.
- **Hyperparameters printed.** SFT LR 2e-5, effective batch 32. DPO β 0.2, LR 3e-6. GRPO 4 generations, β 0.08, ε 0.2, LR 3e-6, temperature 0.9. Mesh chosen per algorithm: FSDP (8,1) for compute-bound SFT and DPO, tensor parallel (1,4) for decode-heavy GRPO.
- **Rewards, no judge.** Math: exact format 3.0 + approximate format 2.0 + answer 3.0 + numbers 1.5, a 9.5-point maximum. Extra domains: one 1.0-point reward, format 0.4 + accuracy 0.4 + heuristic reasoning quality 0.1 + bonus 0.1.
- **Evaluation.** A held-out set run greedily after every stage; no score reported.

The uncomfortable lesson: model quality is 45 points on a set nobody saw, and a clean pipeline with failure-mined DPO beat many better-documented entries.

### 5th: On-Policy Distillation from scratch (Shanai Nair)

Host: "algorithmic innovation (implemented OPD from scratch)". A Gemma 3 **4B** teacher was trained offline and published: full SFT on 100k gpt-oss-120B "baby-step" traces (LR 2e-5, batch 128), length-normalised DPO on about 57k judge-ranked pairs, then GRPO with joint multi-sample judge scores. Inside the session, the Gemma 3 **1B** student runs SFT then **on-policy distillation**: it samples its own answers and gets token-level teacher supervision on them, with a PPO-style loss. Student-to-teacher reverse KL converges to about 0.3.

Evaluation is the field's most careful: 16k held-out prompts, an LLM judge *ranking* sampled completions rather than scoring single answers. Traces have the student classify the task, then pick a mode. The [on-policy distillation post](/blog/machine-learning/large-language-model/on-policy-distillation) explains why sampling from the student fixes plain SFT's exposure bias.

### 6th: Gemma2-Deep, seventeen rewards (Kevin Bönisch)

Host: "creative reward design and well-made video". Pure GRPO on Gemma2-2b-it with LoRA and a frozen reference; no SFT stage described. The notebook could not be pulled, so rank, LR, steps and TPU time are unknown here.

- **Data, Deep-CoRGI.** 100k+ tasks unified from the a-m-Team distilled reasoning collection, mostly capped at 10,000 per source.
- **A learned process reward, CoRGI-ThoughtTeacher.** The author corrupts each trace (deleted sentences, shortened reasoning), asks Gemini which version is better, and trains an encoder with a linear head to score (prompt, reasoning) in [0, 1]. It never sees the answer.
- **Seventeen reward functions** by task type and span, including a **reasoning-budget reward from −1.5 to 3 inspired by Nemotron 3's thinking budget**, so verbosity first pays less, then costs.

Mean total reward rose from 6.60 to 10.63 (+61%). The author's caveat is the honest part: the ThoughtTeacher component *fell*, 0.090 to 0.080, and "I should have increased the weight of the CoRGI-TT more". Most of the gain is format and length compliance, measured on the training reward itself.

## 8. The nine Honorable Mentions, by family

![Reward designs for answers with no reference, placed on one axis of extra training-time machinery: DoD's policy likelihood needs nothing, IDEA-E's TF-IDF and two embedding cosines run on CPU, Gemma2-Deep trains a small reward model, Pinocchio and 4-Tier call the Gemini API, and G-RaR runs a 12B judge on the same TPU](/imgs/blogs/kaggle-tunix-hack-reasoning-winners-8.webp)

The figure is the field's answer to the poem problem, ordered by cost. Placed entries sit at both ends.

### No RL at all: Yuchen20 and Green

**Yuchen20.** Two LoRA stages on Gemma 2 2B IT: SFT at 8,192 tokens on a filtered `allenai/Dolci-Think-SFT-7B` subset plus about 25% general reasoning data, then DPO at 4,096 tokens on `Dolci-Think-DPO-7B`, whose chosen side comes from a 32B Qwen3 reasoner and rejected from a 0.6B one. DPO here smooths mixed-teacher style drift rather than teaching correctness. Length shaping oversamples short Q&A to cure endless reasoning. No held-out numbers.

**Green.** Tunix full fine-tuning ran out of memory above 1,024 tokens, so the author forked Tunix and added flash attention, cut-cross-entropy and full-model rematerialisation ("four days of vibe coding", no prior JAX), saving "over 100 GB" for batch 64 at context 4,096 ([PR 966](https://github.com/google/tunix/pull/966)). Stage 1 full fine-tunes on about 0.5B tokens of long-reasoning rows; stage 2 is SimPO on about 25k rows with GLM 4.7 writing the chosen side. The stages would not chain reliably in one notebook, so judges were asked to fall back to stage 1.

### RL, then distill the RL back into SFT: Gemmatron and Glassbox

**Gemmatron.** SFT-1 teaches tags, GRPO with deterministic verifiers improves reasoning, then **SFT-2 fine-tunes on the best-of-N GRPO rollouts**, compressing RL gains into a stable supervised policy. GRPO LR 1e-5, 6 generations, β 0.04. The GRPO set is described as 1,244 questions, but its listed parts (500 GMAT, 400 truth tables, 100 safety) sum to 1,000. Its 2 h SFT plus 7 h GRPO fill the session before SFT-2; how that fits is unclear.

**Project Glassbox.** The same shape on the Gemma 2 2B **base** model: SFT on 50k rows at exactly 10k per domain, GRPO on a hand-cleaned 5,000-row subset, then self-distillation keeping 10 to 15% of about 4,000 candidates. Penalties for reasoning under 20 characters and for n-gram repetition were added after the model learned empty `<reasoning></reasoning>` shortcuts and phrase loops.

### Trainer surgery: TraceCraft

GRPO only, on Gemma 3 1B IT, monkey-patched into **2-GRPO** (group size 2) plus **Dr. GRPO**, which removes the length and std normalisation that favours long wrong answers, with KL and the reference dropped. A cosine bonus fires when generated reasoning matches a gpt-oss-120b teacher trace above 0.5. The ablation over algorithm × curriculum × LoRA rank {32, 64} picked 2-GRPO + Dr. GRPO, curriculum on, rank 32. (The listed per-source counts sum to 12,960, one short of the stated 12,961 training rows.) The [beyond-GRPO post](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo) derives Dr. GRPO; here is the bias in one toy:

```python
# Illustrative toy, not TraceCraft's patch: the per-token weight a GRPO-style
# loss puts on each token of a response, with and without the 1/|o| length
# normalisation and the per-group std division that Dr. GRPO removes.
import math

rewards = [1.0, 1.0, 0.0, 0.0]          # two right, two wrong
lengths = [120, 150, 90, 480]           # tokens; the last wrong answer rambles
mean = sum(rewards) / len(rewards)
std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))

print(f"{'resp':>4} {'reward':>6} {'len':>4} {'GRPO w/token':>13} {'Dr.GRPO w/token':>16}")
for i, (r, n) in enumerate(zip(rewards, lengths)):
    adv_grpo = (r - mean) / std          # scaled by the group's std
    adv_dr = r - mean                    # Dr. GRPO: no std division
    w_grpo = adv_grpo / n                # spread over this response's own length
    w_dr = adv_dr / max(lengths)         # Dr. GRPO: one constant for everyone
    print(f"{i:>4} {r:>6.1f} {n:>4} {w_grpo:>13.5f} {w_dr:>16.5f}")
```

```console
resp reward  len  GRPO w/token  Dr.GRPO w/token
   0    1.0  120       0.00833          0.00104
   1    1.0  150       0.00667          0.00104
   2    0.0   90      -0.01111         -0.00104
   3    0.0  480      -0.00208         -0.00104
```

With per-response length averaging, the 480-token wrong answer is penalised about 5.3 times less per token than the 90-token one (0.01111 / 0.00208). Rambling is cheaper than being briefly wrong, so a small model drifts long. Dr. GRPO gives every token the same magnitude. TraceCraft's own lesson: small trainer details "skew gradients as much as the reward does".

### Judge-free dense rewards: Gemma3Think and DoD

**Gemma3Think.** Citing "Small Models Struggle to Learn from Strong Reasoners", Keegan Carey synthesised **short** traces with Qwen3-14B. One category-aware reward routes each of 9 categories to a strategy; the semantic one is 40% all-mpnet-base-v2 cosine to the reference + 15% length + 25% reasoning + 20% format, on CPU in about 20 ms. GRPO: LoRA r32/α64, LR 5e-6, β 0.2, 4 generations. The failure log is the best in the field: LR 1e-6 learned nothing; a separate format reward saturated at 99%, leaving loss near 0 until format was folded into a 20% bonus; one LoRA adapter shared across SFT and GRPO made the model forget its tags.

**DoD.** Math gets rule-based rewards. Everything else gets an **RLPR-style probability reward**: the policy's own likelihood of the reference answer, teacher-forced after its reasoning, minus a prompt-only baseline. Reasoning that makes the reference more probable earns reward, with no judge model. The SFT data mixes long and short CoT at 1:4, because pure long CoT made the model overthink easy GSM8K items.

### Rubrics and judges: Rubric-Guided Writing and 4-Tier

**Rubric-Guided Writing.** An LLM turns each writing task into weighted criteria tiered Essential, Important and Bonus, plus negative Penalty rubrics; math content counts fully only above a correctness threshold. Which model scores the rubrics during RL is not stated. Greedy results: MATH-500 6.2 to 28.0 (+21.8), HellaSwag 34.6 to 38.6 (+4.0). The claim that more *writing* RL raised *math* rests on one seed with no control for simply training longer.

**4-Tier.** GRPO only on Gemma 2 2B IT, 1,125 steps on a (2,4) FSDP + TP mesh. Format +0.5 (deliberately weak), reasoning quality ±1.0 from Gemini 2.0 Flash over five weighted rubrics, correctness ±1.0, and a conciseness band. An MD5 response cache hit 85%+ and cut failed judge calls from about 15% to under 1%. Data is 70% non-verifiable by design (3,500 of 5,000).

## 9. What the host and the forum taught everyone

- **The winners thread is the authority on places.** The Developers Blog recap files 5th and 6th under Honorable Mentions; the winners thread and prize data give them 10,000 USD each.
- **Pin your library.** An unpinned Tunix install broke Yuchen20's LoRA sharding late in the contest.
- **The library improved because of the contest.** Green's fork went upstream as a PR, and the recap promoted the split-mesh judge and the CPU-side TF-IDF reward as reusable patterns.
- **Checkpoint plumbing was a hidden tax.** Pinocchio wrote its own converters; Green saved safetensors only because Orbax hung.

## 10. What did not work

| Attempt | Who reported it | What happened |
| --- | --- | --- |
| GRPO without SFT warm-up | 1st (as argument) | Rewards near zero, noisy gradients, risk of collapse |
| Separate format reward after SFT | Gemma3Think, 4-Tier | Saturated; model wraps base answers in tags |
| GSPO with a length reward | Yuchen20 | Reward hacking: loops, rambling |
| GRPO on small models | Green | "Repeatedly shown to not converge"; dropped RL |
| Mis-scaled KL penalty | TraceCraft | "Often hurt"; removed KL and the reference |
| Learned process reward at low weight | 6th | Its own score fell 0.090 to 0.080 |
| Rule rewards without anti-shortcut terms | Glassbox | Empty reasoning tags and phrase loops |
| One LoRA adapter for SFT and GRPO | Gemma3Think | Forgot the tags |
| Continued pretraining on the base model | Green | Benchmarks regressed by over 50% |
| Pure long CoT for SFT | DoD | Overthinking easy GSM8K items |

## 11. Lessons that transfer

1. **After SFT, the format reward is dead weight.** Fold it into a small bonus or a gate and spend the reward budget elsewhere (Gemma3Think, 4-Tier).
2. **Reward density beats reward sophistication.** TF-IDF to a teacher trace drove most of 3rd place's ablation gain; 1st place's expectation-over-logits is the same idea applied to a judge. Never hand GRPO a reward that ties.
3. **Memory is a layout problem.** "The judge won't fit" was wrong once someone split the mesh. SandSpec chose a mesh per algorithm; Green rewrote attention.
4. **Mine your own model's failures for preference data.** SandSpec placed on model quality without any judge in the reward.
5. **Trainer details are rewards in disguise.** Length normalisation, group size and KL weight moved outcomes as much as reward functions did.
6. **Evidence is points.** Three of the four baseline-reporting entries were placed.

### If you enter a hackathon like this

- Read the rubric weights, then plan the schedule backwards from the single-session rule.
- Time every stage on the target hardware in week one.
- SFT on teacher traces to fix the format, then decide what reward fires on a poem.
- Prefer rewards that differ across a group: expectations, similarities, likelihoods, ranks.
- Pin library versions and test checkpoint hand-offs between trainers early.
- Report a baseline, an ablation and one failure.

## 12. Sources

- Competition: https://www.kaggle.com/competitions/google-tunix-hackathon
- Winners announcement with host reasons: https://www.kaggle.com/competitions/google-tunix-hackathon/discussion/691572
- Evaluation-progress thread: https://www.kaggle.com/competitions/google-tunix-hackathon/discussion/670878
- Google Developers Blog recap: https://developers.googleblog.com/how-the-community-trained-gemma-to-think-with-tunix-and-tpus/
- 1st G-RaR: [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1768107128907), [notebook](https://www.kaggle.com/code/hojuna/g-rar-from-sft-to-grpo-for-structured-reasoning)
- 2nd Pinocchio-1B: [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1767940008062), [notebook](https://www.kaggle.com/code/davidacad10/tunix-gemma-to-pinocchio)
- 3rd IDEA-E: [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/idea-e-distillation-with-curriculum-guided-grpo), [GitHub](https://github.com/kpower7/tunix_hackathon)
- 4th SandSpec: [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1768112993955), [host-published notebook copy](https://www.kaggle.com/code/windmaple/mattys737-sandeeplleb-submission)
- 5th OPD: [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/eliciting-reasoning-via-on-policy-distillation)
- 6th Gemma2-Deep: [writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/theitcrow-gemma2-deep-1767955818518)
- HMs: [Gemmatron-Tunix](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/gemmatron-tunix), [SFT First, DPO Second](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1763082571151), [Precision Alignment](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1763040229455), [Optimized Tunix](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/optimized-tunix) and [Tunix PR 966](https://github.com/google/tunix/pull/966), [Project Glassbox](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1764443807640), [Gemma3Think](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/gemma3think), [Rubric-Guided Writing](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1766911313944), [4-Tier Hybrid Process Rewards](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1766324142382), [DOD Google Tunix Hack](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/dod-google-tunix-hack)

Related in this series: [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work), [AIMO Progress Prize 3](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-3-solutions), and the same-wave hackathon posts on [Gemma impact hackathons](/blog/machine-learning/learn-from-kaggle/kaggle-gemma-impact-hackathons-winners), the [BigQuery AI hackathon](/blog/machine-learning/learn-from-kaggle/kaggle-bigquery-ai-hackathon-winners) and [Measuring AGI](/blog/machine-learning/learn-from-kaggle/kaggle-measuring-agi-cognitive-abilities-winners).

---
title: "Index-Translate: How a 9B Translator Shares One Foundation Across Text, Speech, Dubbing and Whole Books"
date: "2026-11-25"
description: "A technique-by-technique reading of Index-Translate: pivot-organized mid-training, gated RL rewards, weight interpolation plus on-policy distillation, and the three specialists it spins off for speech, syllable-controlled dubbing and native book-length translation."
tags: ["paper-reading", "machine-translation", "index-translate", "multilingual", "grpo", "model-merging", "on-policy-distillation", "speech-translation", "long-context", "dubbing"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 14
paper:
  title: "Index-Translate: A Multilingual Translation Model Family"
  authors: "Tianjiao Li et al. (Bilibili Index LLM Team)"
  venue: "arXiv 2026"
  url: "https://arxiv.org/abs/2609.40181"
---

> [!tldr]
> - **What it is.** A family of translation models (2B, 9B, and a 35B-A3B preview) built on Qwen3.5 base checkpoints, covering 150 languages, plus three specialists: Index-Echo (speech-to-text and speech-to-speech), Index-Homura (translation to an exact syllable count, for dubbing) and Index-NativeLong (translate a whole book chapter in one generation).
> - **The key mechanism.** One shared multilingual mid-training, then separate General, Instruction and Meme experts trained with SFT and RL, merged by plain weight averaging (0.8/0.1/0.1) and patched with targeted multi-teacher on-policy distillation.
> - **Why it matters.** The 9B model sits within a few thousandths of 30B-to-218B translation models on FLORES and beats all of them on instruction following (instTrans IFscore 0.8209 versus 0.6415 for Hy-MT2-30B-A3B).
> - **Most surprising result.** On native long-document translation, Index-NativeLong-9B scores 0.7382 at 64K tokens on GuoFeng while Qwen3.8 Flash collapses to 0.4939 and Hy-MT2-7B to 0.0001.
> - **Where it fails.** Precise control does not survive integration. A syllable-control expert diluted to 0.1 weight in a merge hits its target 15% of the time, versus 54% as a standalone specialist. The authors therefore ship it separately.

![Figure 1 from Li et al. (2026): seven-category comparison of Index-Translate against the best non-Index model per category](/imgs/blogs/index-translate-fig1.webp)

The radar is the headline: on seven translation categories, the Index models press against the best non-Index score per axis. The more interesting question is the paper's own: *which translation skills can one compact model share, and which need their own model?*

## The problem

A translation model serving a video platform does not do one task. A product listing must keep its placeholders intact, a meme needs cultural interpretation, a dubbed line must hit a syllable budget, and a novel must keep a character's name the same on page 3 and page 300. Speech adds an acoustic front end and, for dubbing, a voice.

The industry default is either one giant frontier model doing all of it through prompting, or a zoo of separate systems. Small dedicated translation models (Hy-MT2, TranslateGemma) are strong on sentence-level benchmarks but, as Table 4 below shows, poor at following translation instructions: TranslateGemma-12B scores 0.3068 on instTrans IFscore. The gap the paper claims to fill is a compact family where shared skills are trained once and consolidated, and only the skills that refuse to share get a dedicated model.

The hope behind it is the one from [task arithmetic](/blog/paper-reading/computer-vision/understanding-and-enforcing-weight-disentanglement-in-task-arithmetic): experts fine-tuned from one initialization can be added in weight space. This paper field-tests how far that goes.

## Contributions

1. A 150-language mid-training recipe that separates ordinary parallel text from **pivot-organized** groups and shows broad coverage lifts non-core languages without hurting core ones.
2. Three post-trained experts with **gated multiplicative rewards** and adversarial reward-model refresh (RIVAL) to resist reward hacking.
3. An integration recipe: **parameter interpolation, then targeted MOPD**, with ablations that expose what survives each step.
4. Three dedicated extensions (speech, syllable count, native long documents) and two new benchmarks: instTrans and NativeLongBench.

## Method

![Figure 2 from Li et al. (2026): the text-model training pipeline, from Qwen3.5 base through mid-training, three specialists, interpolation and targeted MOPD](/imgs/blogs/index-translate-fig2.webp)

Read Figure 2 left to right: one mid-training run feeds three specialists, which are averaged, and MOPD repairs what the average broke.

### 1. Mid-training with pivot-organized data

**The problem.** A base LLM knows many languages but has weak cross-language alignment, especially outside the top 20. Throwing random sentence pairs at it teaches pairwise mappings, and with 150 languages there are over 22,000 directed pairs, most of which you will never have data for.

**Intuition.** Think of an airline network. Point-to-point routes between every city pair is the ordinary parallel corpus: expensive and sparse. A hub-and-spoke network routes everything through one hub. A pivot group is a hub: one English sentence, with its Chinese, Japanese, Spanish and Vietnamese renderings all attached, so the model sees five languages saying the same thing in one training example.

![Ordinary parallel pairs versus a pivot group anchored on one English sentence](/imgs/blogs/index-translate-1.webp)

**Mechanism.** The paper mixes four streams: general replay $G$, monolingual text $M$, ordinary parallel pairs $P$, and pivot groups $V$. *Parallel degree* is the number of languages aligned to the same content, two for $P$ and up to many for $V$. Training follows the stable and decay phases of a warmup-stable-decay schedule. The stable phase samples general, parallel and monolingual at 1:1:1 for 7,000 steps (117.44B tokens) at a constant learning rate of $5 \times 10^{-5}$. The decay phase switches to general, a 20-language pivot stream and a 148-language pivot stream at 1:4:2 for 3,000 steps (50.33B tokens), annealing to $5 \times 10^{-6}$. With sequences of 4,096 tokens and a global batch of 4,096, each step is about 16.8M tokens, so the total is 167.77B.

The sampling probability of each stream is just its normalized weight. For a stream $k$ with weight $w_k$:

$$
p_k = \frac{w_k}{\sum_{j} w_j}, \qquad \text{decay: } p_{\text{pivot-20}} = \tfrac{4}{7} \approx 0.57,\; p_{\text{pivot-148}} = \tfrac{2}{7} \approx 0.29,\; p_G = \tfrac{1}{7} \approx 0.14
$$

So during decay, roughly six of every seven sampled sequences are multilingual pivot data.

**Evidence and failure modes.** In the 1:1:1 sweep, dropping parallel data (1:1:0) costs the most translation score (0.7750 versus 0.7914). Expanding decay coverage from 20+ to 100+ languages at 2B lifts the 61 non-core targets from 0.5605 to 0.6570 COMET while the 20 core languages barely move (0.8231 to 0.8219). The cost is general capability: 11-task NLU at 2B drops from 0.5920 (base) to 0.5303 after the stable phase and only recovers to 0.5431 after decay. The 9B general-capability table shows the same scar, with GPQA-Diamond at 36.3 versus 75.8 for Qwen3.5-9B. Coverage has to be held all the way through SFT too: a 22-language-only SFT control dropped 60-language FLORES COMET from 0.7441 to 0.7062 within 200 steps.

One honest caveat the paper flags itself: its pivot-versus-parallel ablation is a tie. The mixture without $P$ scored 0.7991 and the one with $P$ scored 0.7971, a 0.002 gap that also changes the shares of every other stream. Pivot organization is the chosen recipe, not a demonstrated win.

### 2. Gated rewards for RL

**The problem.** RL on translation with a learned metric invites reward hacking. A model can score well on a neural metric while answering in the wrong language, or silently drop a clause that the metric does not punish hard enough.

**Intuition.** A job interview with hard requirements. If you lack the work permit, nothing else in the interview matters. If you have it, you are graded on quality, and a missing reference cuts your score proportionally.

**Mechanism and math.** General translation (trained with GRPO) multiplies a binary language gate by two graded scores:

$$
R_{\text{gen}} = g_{\text{lang}} \cdot q_{\text{XCOMET}} \cdot q_{\text{adeq}}
$$

where $g_{\text{lang}} \in \{0, 1\}$ is 1 only if the output is in the target language, $q_{\text{XCOMET}} \in [0,1]$ is the reference-based XCOMET-XXL score, and $q_{\text{adeq}} \in [0,1]$ is a source-grounded adequacy score from Gemini 2.5 Flash that discounts omissions and additions. Instruction translation adds a hard-constraint gate and an additive soft term:

$$
R_{\text{inst}} = g_{\text{lang}} \cdot g_{\text{hard}} \cdot \left( q_{\text{qual}} + q_{\text{soft}} \right)
$$

where $g_{\text{hard}}$ is zero if any required structure, terminology, code span or placeholder is broken, $q_{\text{qual}}$ is a rubric-based quality score and $q_{\text{soft}}$ averages soft constraints such as style and contextual interpretation (judged by Gemini 3.1 Pro under Rubric-as-Reward).

![Three candidate translations scored by the gated product reward](/imgs/blogs/index-translate-2.webp)

**Worked example.** The figure traces three candidates. A fluent answer in the wrong language has $q_{\text{XCOMET}} = 0.90$ but $g_{\text{lang}} = 0$, so its reward is exactly zero. A right-language output that drops a clause gets $1 \times 0.82 \times 0.60 = 0.49$. A complete one gets $1 \times 0.88 \times 0.97 \approx 0.85$. In GRPO, rewards are compared within a group of samples for the same prompt, so the wrong-language sample becomes a strong negative advantage rather than a mild one.

```python
def r_gen(g_lang: int, q_xcomet: float, q_adeq: float) -> float:
    return g_lang * q_xcomet * q_adeq

def r_inst(g_lang: int, g_hard: int, q_qual: float, q_soft: float) -> float:
    return g_lang * g_hard * (q_qual + q_soft)

group = [r_gen(0, 0.90, 0.95), r_gen(1, 0.82, 0.60), r_gen(1, 0.88, 0.97)]
mean = sum(group) / len(group)
std = (sum((r - mean) ** 2 for r in group) / len(group)) ** 0.5
advantages = [(r - mean) / std for r in group]
print([round(r, 2) for r in group])       # [0.0, 0.49, 0.85]
print([round(a, 2) for a in advantages])  # [-1.28, 0.12, 1.16]
```

**Why it works, when it fails.** The product makes the gates non-negotiable and the adequacy score a multiplier, so the policy cannot buy back a dropped clause with fluency. The weak point is that two of the three factors are LLM judges, which can themselves be gamed. That is what RIVAL is for: it alternates reward-model updates on current policy outputs with policy updates against the refreshed reward. In the authors' earlier 7B study, adding quantitative supervision at RIVAL iteration 2 lifted WMT English-to-Chinese BLEU from 30.14 to 39.39. The 9B RL results (Table 17) show the payoff: instruction IFscore rises by 0.0934 and low-resource off-target rate falls from 3.4% to 1.2%.

### 3. Interpolation, then targeted MOPD

**The problem.** You now have three 9B experts that each beat the others on their own task. Serving three models is costly, and each one is mediocre outside its lane.

**Intuition.** Averaging weights is like mixing paints from the same base: because every expert started from the same mid-trained checkpoint, they live in one basin and a weighted mix is still a working model. But mixing dilutes the strong colors. MOPD is then a touch-up: the merged student writes its own translations, and the relevant expert corrects it token by token.

**Mechanism and math.** First, a convex combination in parameter space (the model-soup recipe):

$$
\theta_{\text{merge}} = w_G\,\theta_G + w_{\text{Inst}}\,\theta_{\text{Inst}} + w_M\,\theta_M, \qquad w_G + w_{\text{Inst}} + w_M = 1
$$

with $w_G/w_{\text{Inst}}/w_M = 0.8/0.1/0.1$ for the released 2B and 9B models. Then MOPD samples rollouts from the student $\pi_\theta$ and, for each prompt $x$, picks the domain-matched teacher $\pi_{T(x)}$. The paper states the objective is reverse KL on student-generated tokens but does not print it; the standard form is

$$
\mathcal{L}_{\text{MOPD}}(\theta) = \mathbb{E}_{x,\; y \sim \pi_\theta(\cdot \mid x)} \sum_{t=1}^{|y|} \mathrm{KL}\!\left( \pi_\theta(\cdot \mid x, y_{\lt t}) \,\big\|\, \pi_{T(x)}(\cdot \mid x, y_{\lt t}) \right)
$$

where $y_{\lt t}$ is the student's own prefix. Reverse KL is mode-seeking: the student is penalized for putting mass where the teacher puts none, which is why it cleans up off-target language switches so well.

![What survives integration: quality blends, control collapses](/imgs/blogs/index-translate-3.webp)

**Worked numbers.** The figure collects the paper's two dilution ablations. Cutting the general branch from 1.0 to 0.5 at 2B nudges WMT XCOMET-XXL up from 0.8222 to 0.8522 but craters IFscore from 0.7395 to 0.2828. A syllable-control expert at 0.1 weight in a four-way merge reaches a SandGlass hit rate of 0.1544, against 0.5439 for the same skill as a specialist. On the MOPD side (Table 19), every variant halves low-resource off-target from 4.2% to 2.2%, and a 9B-teacher variant lifts instTrans IF from 0.8359 to 0.8669, but instTrans quality drops from 0.5979 to 0.5572 and MEME falls from 0.6429 to 0.6275.

**When it fails.** Broad, smooth skills (translation quality) average well. Sharp, rule-like skills (a constraint that is either met or not) average badly, and MOPD did not rescue length constraints either. The authors guess token-level supervision is the culprit, which is plausible: a syllable budget is a sequence-level property that no single token's distribution carries.

### 4. The three specialists

![Figure 3 from Li et al. (2026): Index-Echo speech-to-text and speech-to-speech architecture](/imgs/blogs/index-translate-fig3.webp)

**Index-Echo** bolts a Qwen3-Omni AuT audio encoder and connector onto the 2B or 9B translator and trains end to end on speech translation. For speech-to-speech, it freezes that model and trains a roughly 30M-parameter Hidden2CV mapper that turns the translator's last hidden states into CosyVoice3's text-embedding slot, then polishes it with DiffRO, which backpropagates a content-consistency reward through Gumbel-Softmax samples of speech tokens. The source clip supplies the speaker prompt, so the dub keeps the original voice.

**Index-Homura** fine-tunes for a requested target syllable count $s^\star$ with GRPO, jointly rewarding quality and closeness to $s^\star$. The paper does not print the length reward; any shape that peaks at $|s - s^\star| = 0$ fits the description. The central budget is subtitle duration times a speaking rate (6.19 syllables per second for English), so a 2.0 s English line gets about 12 syllables, with short and long variants at 0.75 and 1.25 times that.

![Figure 5 from Li et al. (2026): translation quality versus syllable-control accuracy on SandGlass](/imgs/blogs/index-translate-fig5.webp)

Figure 5 is the cleanest picture of a trade-off in the paper. RL moves Homura-9B from 45.75% to 81.92% within 10% of target while quality falls from 0.8581 to 0.7863. Doubling the 2B syllable-reward weight buys 63.08% to 74.19% control for 0.7615 to 0.6900 quality. Frontier models sit upper-right on quality but far lower on control: GPT-5.6-Sol hits 47.64%.

**Index-NativeLong** is the most interesting bet. Instead of chunking a book and stitching chunks with a glossary, the model receives the whole source (4K to 64K Chinese-side tokens) and writes the whole translation in one pass. Training is full-parameter SFT on aligned book passages, with long-sequence decay; ablations found RoPE and NoPE comparable after this SFT. The argument is consistency: in the paper's glacial-history excerpt, the native 9B model keeps "ice age" (冰河时代) distinct from "glacials" (冰期), while chunked Qwen3.8-Flash merges them even with a full-source glossary.

Scoring a 60K-token translation against a reference whose sentence boundaries differ is its own problem. SEGALE aligns output blocks to source blocks semantically, scores each matched block with COMET, and assigns zero to unmatched blocks, so a model that stops early or loops is punished for lost coverage, not only for bad wording.

## Experiments and results

| Model | FLORES COMET-22 | WMT26 judge | instTrans IFscore | Low-res off-target | GuoFeng 64K |
| --- | --- | --- | --- | --- | --- |
| Index-Translate-35B-A3B (preview) | **0.8794** | 76.76 | **0.8336** | 2.4% | n/a |
| Index-Translate-9B | 0.8789 | 75.35 | 0.8209 | 4.0% | **0.7382** (NativeLong) |
| Index-Translate-2B | 0.8655 | 60.26 | 0.7569 | 4.2% | 0.5460 (NativeLong) |
| Hy-MT2-30B-A3B | 0.8787 | 66.81 | 0.6415 | 14.5% | 0.0664 |
| North-Small-Translate (218B-A25B) | 0.8784 | 68.37 | 0.5294 | 13.2% | 0.2353 |
| TranslateGemma-12B | 0.8732 | 71.19 | 0.3068 | 1.7% | n/a |
| DeepSeek-V4.1-Flash | 0.8762 | 83.55 | 0.6374 | **1.3%** | n/a |
| GPT-5.6-Sol | 0.8650 | **89.10** | 0.7624 | 10.4% | n/a |
| Qwen3.8 Flash | n/a | n/a | n/a | n/a | 0.4939 |

The FLORES column is nearly flat: six of these systems sit between 0.873 and 0.880. The real separation is on instruction following and low-resource language discipline, where Hy-MT2-1.8B goes off-target 54.2% of the time and Index-Translate-2B 4.2%. The WMT26 column, judged by GPT-5.6-Sol, still has frontier models ahead by 8 to 14 points.

![Figure 4 from Li et al. (2026): native long-document translation quality by length group](/imgs/blogs/index-translate-fig4.webp)

In Figure 4 everyone is close at 4K. By 64K, Hy-MT2 stops after a short fragment, North degenerates into repetition, and Qwen3.8 Flash leaves sentences unfinished. Index-NativeLong-9B stays above 0.73 on all three corpora. Speech is similar in shape: Index-Echo-9B's MT judge score (0.857) sits between Qwen3.8-Omni-Flash (0.887) and Gemini-3.1-Pro with thinking (0.849), but its start-time error is 0.488 s versus 1.004 s and 1.824 s, which matters more for subtitles than a few points of quality.

**What might not transfer.** Three things are load-bearing. First, the Qwen3.5 base: the image-text result (comparable to post-trained Qwen3.5 with zero image training) is inherited, and at 2B it is not comparable at all (0.5204 versus 0.6826 on MMTIT). Second, the judges: WMT26, instTrans and MEME are all scored by LLMs, including GPT-5.6-Sol, which is also a compared system. Third, the data pipeline: SFT translations are regenerated by frontier models, so part of what is being distilled is frontier quality.

## Critique

**Strong.** The paper is unusually candid: it reports the pivot-versus-parallel tie, the general-capability drop, MOPD's quality cost and the syllable-control collapse, and uses those failures to justify its architecture. The gated rewards are easy to copy. NativeLongBench, with SEGALE alignment and zero credit for unmatched blocks, measures the right thing for book translation.

**Weak.** NativeLongBench's fixed evaluation sets "have also been used during model development," which the authors disclose but which makes the 64K gap an upper bound. The speech benchmarks are in-house (140 windows for S2TT), and SeamlessM4T-v2 is fed 57-second inputs beyond its roughly 30-second range, so its 0.060 is not a fair number. The 35B-A3B model is a preview with thin training details.

**Missing ablations.** There is no merge-weight sweep at 9B, no comparison of interpolation against TIES or task-arithmetic merging, and no comparison of native long-document translation against chunking with the *same* model at 64K (the glossary case is qualitative). The parallel-versus-pivot question deserved a matched-share experiment.

**What would change my mind:** if a chunked Index-Translate-9B with a sliding translation-history window matched NativeLong-9B's 0.74 at 64K on held-out books never used in development, I would read the native-generation result as a length-robustness training effect, not evidence that single-pass translation is the better formulation.

## What I'd build with this

These are my extrapolations, not the paper's claims.

1. **Port the gated reward to structured generation.** JSON extraction and SQL generation have hard gates (parses, schema matches) that should multiply, not add.
2. **Test skill shape before merging.** Thresholded skills (length, format) belong in a separate model or a sequence-level reward, not a weight-space blend.
3. **A dubbing stack from Echo plus Homura.** Timestamps plus syllable budgets are the text half of what [MuseTalk](/blog/paper-reading/multimodal/musetalk-video-dubbing) does for lips; [training ASR models](/blog/machine-learning/speech-to-text/training-asr-models) covers the encoder side, and [attention sinks](/blog/paper-reading/large-language-model/streaming-llm-attention-sinks) are the other answer to long sequences.

## References

- Li, T., Yu, M., Shi, C., et al. *Index-Translate: A Multilingual Translation Model Family.* arXiv:2609.40181, 2026. [https://arxiv.org/abs/2609.40181](https://arxiv.org/abs/2609.40181)
- Li, T., et al. *RIVAL: Reinforcement Learning with Iterative and Adversarial Optimization for Machine Translation.* arXiv:2506.05070, 2025.
- Ma, W., et al. *MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training.* arXiv:2606.30406, 2026.
- Cui, Z., et al. *HOMURA: Taming the Sand-Glass for Time-Constrained LLM Translation via Reinforcement Learning.* arXiv:2601.10187, 2026.
- Wortsman, M., et al. *Model soups.* ICML 2022.
- Related on this blog: [weight disentanglement in task arithmetic](/blog/paper-reading/computer-vision/understanding-and-enforcing-weight-disentanglement-in-task-arithmetic), [MuseTalk video dubbing](/blog/paper-reading/multimodal/musetalk-video-dubbing), [training ASR models](/blog/machine-learning/speech-to-text/training-asr-models).

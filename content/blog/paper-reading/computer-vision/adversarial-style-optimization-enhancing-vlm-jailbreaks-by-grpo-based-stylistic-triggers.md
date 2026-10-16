---
title: "Adversarial Style Optimization: Enhancing VLM Jailbreaks by GRPO-based Stylistic Triggers Optimization"
publishDate: "2026-10-16"
date: "2026-10-16"
description: "A defender's reading of the CVPR 2026 oral that turns visual style into an attack axis: why a pencil-sketch filter weakens VLM refusals, how a GRPO-tuned image editor with a two-tier reward pushes further, and what the numbers do and do not support."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["safety-robustness", "cvpr-2026", "cvpr-2026-award-candidate", "cvpr-2026-oral", "vlm-safety", "jailbreak", "red-teaming", "grpo", "flow-matching", "safety-alignment", "adversarial-style-optimization"]
author: "Hiep Tran"
featured: true
readTime: 19
image: "/imgs/blogs/adversarial-style-optimization-enhancing-vlm-jailbreaks-by-grpo-based-stylistic-triggers-fig2.webp"
excerpt: "A vision-language model reads a charcoal sketch as easily as a photo, but its refusal reflex does not travel as well. This paper turns that gap into an optimizable attack axis, and in doing so maps a blind spot defenders need to close."
paper:
  title: "Adversarial Style Optimization: Enhancing VLM Jailbreaks by GRPO-based Stylistic Triggers Optimization"
  authors: "Bingjun Luo, Jialin Guo, Yue Yao, Xinpeng Ding"
  venue: "CVPR 2026 (oral, award candidate)"
  url: "https://arxiv.org/abs/2607.21619"
---

> [!tldr]
> **Claim.** Multimodal LLMs understand an image's content regardless of its visual style, but their safety refusals do not hold up equally well across styles. The authors call this *Stylistic Inconsistency* and build ASO, a plug-in that restyles an existing jailbreak image with a GRPO-tuned image editor to raise its attack success rate (ASR).
>
> **Why it matters.** Almost every VLM safety evaluation tests *what* an image shows. This paper shows that *how* it is drawn is a separate axis of failure, and that it can be searched automatically.
>
> **Most surprising result.** Unstructured Gaussian noise *lowers* ASR on GPT-4.1-mini (54.26% to between 48.03% and 50.75%), while a coherent, optimized style raises it to 57.36%. Safety is robust to noise and brittle to style.
>
> **Where it is thin.** Gains are typically 2 to 8 points, one configuration goes backwards, and the optimization loop selects successes with the same HarmBench judge used for evaluation. The paper also asserts, but never measures, that comprehension survives the restyling.

We read this red-teaming paper as defenders: the method and evidence at the paper's level, plus what it implies for safety training. No harmful prompt or target behavior is reproduced, and paper figures are cropped so no harmful text is visible.

## 1. The mismatch the paper starts from

Existing visual jailbreaks (FigStep, QR-Attack, HADES, IDEATOR, HIMRD) attack *content*: typography or crafted objects that split harmful intent across modalities. They change **what is depicted**.

The authors keep content fixed and change **how it is rendered**: lighting, medium, texture, palette. A human recognizes a knife in a photo, a charcoal drawing or pixel art. If a VLM's comprehension is equally style-invariant but its refusal is not, then style is a free knob for an attacker.

![Bottom half of Figure 1 from Luo et al. (2026): the same base attack image restyled, the restyled version labelled "Bypass", and the ASR improvement on four model families](/imgs/blogs/adversarial-style-optimization-enhancing-vlm-jailbreaks-by-grpo-based-stylistic-triggers-fig1.webp)

The crop above is the lower half of the paper's teaser (the upper half holds the attack prompt). One base image passes through style transfer, and the bars show ASR gains of about 3 to 7 points on four model families. The paper calls this **Stylistic Inconsistency**: two abilities that should move together, and do not.

![Conceptual sketch: one harmful concept rendered in several styles, all inside the wide "content recognized" region, but only some inside the narrower "refusal fires" region](/imgs/blogs/adversarial-style-optimization-enhancing-vlm-jailbreaks-by-grpo-based-stylistic-triggers-1.webp)

This is our sketch, not the authors'. Each point is the same content in a different style; points inside the wide region but outside the narrow one are what ASO searches for. Why the gap exists is our hypothesis, not the paper's: comprehension comes from huge, stylistically diverse pretraining, while safety tuning is a small stage on mostly photographic or typographic data. It echoes the text-LLM finding that [safety alignment is only a few tokens deep](/blog/paper-reading/ai-interpretability/safety-alignment-should-be-made-more-than-just-a-few-tokens-deep): a refusal learned on a narrow input distribution, which style escapes without changing the meaning.

## 2. Problem setup: an enhancement generator

ASO is not a new attack but an **amplifier** for existing ones. Formally, the base dataset is

$$
\mathcal{D}_{\text{base}} = \{(I_{\text{base},j},\ P_{\text{harm},j})\}_{j=1}^{N},
$$

where $I_{\text{base},j}$ is the $j$-th attack image, $P_{\text{harm},j}$ is the paired text prompt that carries the harmful intent, and $N$ is the number of pairs. A target VLM $\mathcal{M}$ produces a response $R = \mathcal{M}(I, P)$. A judge model $\mathcal{J}$ (HarmBench, a LLaMA-based classifier) returns $y = 1$ if the response is a successful jailbreak and $y = 0$ otherwise.

The goal is an image-to-image generator $\mathcal{G}$ that maximizes the mean judge verdict over the restyled set:

$$
\mathcal{G}^{*} = \arg\max_{\mathcal{G}} \ \frac{1}{N} \sum_{j=1}^{N} \mathcal{J}\big(\mathcal{M}(\mathcal{G}(I_{\text{base},j}),\ P_{\text{harm},j}),\ P_{\text{harm},j}\big).
$$

Read it inside out: restyle the image, query the target with the unchanged prompt, ask the judge, average. Only the image changes, which is what makes ASO stackable on any content-based attack.

![Figure 2 from Luo et al. (2026): the ASO framework. Left, Style Sensitivity Probing with a frozen generator and a style pool; right, GRPO-based Style Enhancement with a trainable generator, tiered reward and the DB-GRPO curate and evict loop](/imgs/blogs/adversarial-style-optimization-enhancing-vlm-jailbreaks-by-grpo-based-stylistic-triggers-fig2.webp)

The left stage (snowflake) uses frozen weights $\Theta_0$ to *pick a style*; the right stage (flame) trains weights $\Theta$ to *tune that style* for the target.

## 3. Stage 1: style sensitivity probing

**The problem.** Style space is huge and RL on an image editor is expensive, so start in a direction the target is actually sensitive to.

**The analogy.** A locksmith tries a ring of key blanks before filing one. Probing is the ring: cheap, coarse, and only meant to find which blank is closest.

**The mechanism.** A pool of 16 styles in four families, each tied to a hypothesis about *why* it might weaken safety. Every base image is restyled by the frozen editor ("in the style of $S_i$"), the target answers, the judge scores, and the highest-ASR style wins.

| Family | Styles in the pool | Authors' hypothesis |
| --- | --- | --- |
| Medium & Texture ($\mathcal{S}_{\text{med}}$) | pencil sketch, charcoal, ink wash, black and white photo, vintage photo | texture artifacts interfere with recognition |
| Geometric & Abstract ($\mathcal{S}_{\text{geo}}$) | cubism, abstract, pixel art, low poly, surrealism | broken forms disrupt spatial semantics |
| Thematic & Atmospheric ($\mathcal{S}_{\text{atm}}$) | cyberpunk, steampunk, vaporwave | genre mood primes unsafe context |
| Domain-Specific ($\mathcal{S}_{\text{dom}}$) | 90s anime, manga, children's book | artistic domains may have looser alignment |

**The math.** For style $S_i$, the restyled dataset is built with the frozen weights $\Theta_0$ and a text directive $P_{\text{style},i}$:

$$
y_{i,j} = \mathcal{J}\big(\mathcal{M}(\mathcal{G}(I_{\text{base},j}, P_{\text{style},i}; \Theta_0),\ P_{\text{harm},j}),\ P_{\text{harm},j}\big) \in \{0, 1\},
$$

$$
\mathrm{ASR}(S_i) = \frac{1}{N}\sum_{j=1}^{N} y_{i,j}, \qquad S^{*} = \arg\max_{S_i \in \mathcal{S}} \mathrm{ASR}(S_i).
$$

Here $y_{i,j}$ is the judge's verdict on pair $j$ under style $i$, and $S^{*}$ is the winning style. Nothing is trained; this is a 16-way grid search.

**A worked example.** Suppose $N = 200$ and pencil sketch yields 81 judged successes while cubism yields 74. Then $\mathrm{ASR} = 0.405$ versus ${0.370}$, and pencil sketch becomes $S^{*}$. (These counts are illustrative.)

![Supplementary Figure 3 from Luo et al. (2026): ASR per style for QR Attack and HIMRD on Qwen3-VL (top) and GPT-4.1-mini (bottom); blue bars are Medium & Texture styles](/imgs/blogs/adversarial-style-optimization-enhancing-vlm-jailbreaks-by-grpo-based-stylistic-triggers-fig3.webp)

**What the probe found.** For a defender this is the paper's most informative chart. The blue Medium & Texture bars beat the grey ones across both models and attacks ("macro-level consensus"). Under HIMRD on Qwen3-VL, texture styles average above 87% ASR. The micro-level winner differs: charcoal peaks on Qwen3-VL, while black and white photo peaks on GPT-4.1-mini.

**When it fails.** Probing scores only the *frozen* editor's generic rendition, so a style potent after tuning can lose here. It costs ${16 \times N}$ target and judge calls, and alone buys only 1 to 2 points.

## 4. Stage 2: the structurally tiered reward

**The problem.** A binary reward gives almost no signal: early restyles are mostly refused, all score zero, and a near-miss looks like a confident refusal. Worse, a model can *accept* and still answer harmlessly. Acceptance is not success.

**The analogy.** Think of a two-stage exam. Before you pass the gate, you are graded on how close you came to passing. After you pass, you are graded on how well you did. Crucially, the worst score after the gate is still better than the best score before it.

**The mechanism.** If the target refuses, the reward is its accept-versus-reject log-odds, shifted to lie below a threshold. If it accepts, the reward is the judge's harmfulness log-odds, clipped from below at that threshold.

**The math.** With threshold $C_{\text{thresh}} = -10$, restyled image $I_h = \mathcal{G}(\cdot;\Theta)$, prompt $P_h = P_{\text{harm}}$ and response $R = \mathcal{M}(I_h, P_h)$:

$$
r_t(I_h, P_h) =
\begin{cases}
C_{\text{thresh}} + \log \dfrac{P_{\mathcal{M}}(\text{accept} \mid I_h, P_h)}{P_{\mathcal{M}}(\text{rejected} \mid I_h, P_h)} & \text{if rejected} \\[2ex]
\max\left( \log \dfrac{P_{\mathcal{J}}(\text{yes} \mid R, P_h)}{P_{\mathcal{J}}(\text{no} \mid R, P_h)},\ C_{\text{thresh}} \right) & \text{if accepted}
\end{cases}
$$

$P_{\mathcal{M}}(\text{accept})$ and $P_{\mathcal{M}}(\text{rejected})$ are the target's probabilities for its initial decision; the paper allows a classifier proxy when these are not exposed. $P_{\mathcal{J}}(\text{yes})$ and $P_{\mathcal{J}}(\text{no})$ come from the judge's logits on its "Yes" and "No" tokens.

Each branch does one job. In the rejected branch, the model chose to refuse, so $P_{\mathcal{M}}(\text{accept}) \lt P_{\mathcal{M}}(\text{rejected})$, the log term is negative, and the reward is strictly below $-10$. Refusing *less confidently* raises it toward $-10$. In the accepted branch, the clip means even a perfectly harmless acceptance scores exactly $-10$, never below any refusal. That preserves the ordering *refuse, then accept-harmless, then accept-harmful* while keeping both branches continuous.

![The tiered reward plotted against the model's state: a rising segment strictly below C = -10 for refusals, a clipped floor at -10 for harmless acceptances, then a rising segment for responses the judge scores as harmful](/imgs/blogs/adversarial-style-optimization-enhancing-vlm-jailbreaks-by-grpo-based-stylistic-triggers-2.webp)

**A worked example** (natural log; the paper does not state the base):

| Target's state | Quantity | Reward |
| --- | --- | --- |
| Confident refusal, $P(\text{rejected}) = 0.9$ | $\log(0.1/0.9) = -2.20$ | $-12.20$ |
| Borderline refusal, $P(\text{rejected}) = 0.6$ | $\log(0.4/0.6) = -0.41$ | $-10.41$ |
| Accepted, judge $P(\text{yes}) \approx 0$ | log-odds below $-10$ | clipped to $-10$ |
| Accepted, judge $P(\text{yes}) = 0.2$ | $\log(0.2/0.8) = -1.39$ | $-1.39$ |
| Accepted, judge $P(\text{yes}) = 0.9$ | $\log(0.9/0.1) = 2.20$ | ${2.20}$ |

The plot above draws these exact points. The jump at the "model accepts" boundary is the point of the design. Moving from a confident to a borderline refusal earns about 1.8 reward, but crossing into acceptance earns far more.

**When it fails.** The rejected branch needs the target's accept and reject probabilities. Open models expose logits; for GPT-4.1-mini and Gemini-2.5-Flash the paper mentions a classifier proxy but never describes it. And the accepted branch trains directly against HarmBench, the evaluation judge (see the critique).

## 5. Stage 2: making a flow model trainable with GRPO

**The problem.** The editor is FLUX.1-dev, a flow-matching model. Standard flow sampling integrates a deterministic ODE: given the input and the noise seed, the output is fixed, so there is no per-step action probability to put in a policy gradient. RL needs stochastic actions with computable log-probabilities.

**The fix, borrowed.** Following Flow-GRPO and DanceGRPO, the authors convert the ODE sampler into an equivalent SDE that injects noise at each denoising step, with noise level $a = 0.9$. Each step is now a Gaussian transition with a tractable log-probability, so the whole 16-step trajectory is a sequence of actions a policy gradient can push on. Only LoRA adapters (rank 64, alpha 128) on the attention layers are trained; the base weights stay frozen.

**GRPO in one paragraph.** [Group Relative Policy Optimization](/blog/machine-learning/reinforcement-learning/grpo-group-relative-policy-optimization) drops PPO's learned value function. For each input it samples a *group* of $G$ outputs and scores each one relative to its siblings:

$$
A_i = \frac{r_i - \operatorname{mean}(\{r_k\}_{k=1}^{G})}{\operatorname{std}(\{r_k\}_{k=1}^{G})},
$$

where $r_i$ is the tiered reward of the $i$-th restyle of the same base image and $A_i$ is its advantage. The policy then maximizes the usual clipped surrogate:

$$
L(\Theta) = \hat{\mathbb{E}}_{j \in B}\Big[\min\big(\rho_j A_j,\ \operatorname{clip}(\rho_j, 1-\epsilon, 1+\epsilon)\, A_j\big)\Big], \qquad \rho_j = \frac{\pi_{\Theta}(a_j \mid s_j)}{\pi_{\Theta_{\text{old}}}(a_j \mid s_j)}.
$$

$\rho_j$ is the probability ratio between the current and the sampling policy for action $a_j$ (the generated image) in state $s_j$ (the base pair), and $\epsilon$ is the clip range, set to ${10^{-4}}$ in the supplement. A very tight trust region, as in Flow-GRPO.

**Worked example.** Take four restyles of one base image with rewards $-12.2$, $-10.4$, $-1.4$ and ${2.2}$. The mean is $-5.45$ and the population standard deviation is about ${6.02}$. The advantages are $-1.12$, $-0.82$, ${0.67}$ and ${1.27}$. Refused restyles are pushed down, accepted ones up, whatever the raw reward scale.

**A detail worth flagging.** The text sets the advantage $A_j$ to the raw reward $r_j$, and Algorithm 1 uses raw $r_j$ in the gradient, but Eq. 9 is the group-normalized form above. These are different estimators, and we cannot tell which produced the numbers.

## 6. Dynamic-Batch GRPO: train for discovery, not generalization

**The problem.** Standard generative RL wants a policy good on average. A red-teamer wants the most successful images for the least compute, so re-training on cracked pairs, or hammering uncrackable ones, is waste.

**The analogy.** A teacher with a stack of unsolved problems. Solved problems leave the stack and go in the "done" folder. A problem that resists ten honest attempts is set aside. Every lesson is spent on the problems still in play.

**The mechanism (Algorithm 1).** DB-GRPO keeps two sets: an Unsolved pool, initialized to the whole base dataset, and a Success collection, initially empty. Each iteration samples a batch of size $B = 3$ from the Unsolved pool only, generates a restyle for each, scores it, and takes a policy-gradient step. Then comes *curate and evict*: any sample with $r_j > C_{\text{thresh}}$ has its image saved to the Success collection and leaves the pool; any sample that has failed $K_{\max} = 10$ times also leaves. The loop ends when the pool is empty.

![Dynamic-Batch GRPO over time: samples move from the Unsolved pool to the Success collection, or are evicted after K_max = 10 failed attempts (illustrative counts)](/imgs/blogs/adversarial-style-optimization-enhancing-vlm-jailbreaks-by-grpo-based-stylistic-triggers-3.webp)

Note the success criterion: $r_j > -10$ means only "the target did not refuse". A harmless acceptance scores exactly $-10$ and so does not count, but anything the judge rates even slightly above the clip floor does.

![Supplementary Figure 1 from Luo et al. (2026): ASR as a function of K_max for IDEATOR on VLBreakBench, Qwen3-VL and LLaVA-OV-1.5](/imgs/blogs/adversarial-style-optimization-enhancing-vlm-jailbreaks-by-grpo-based-stylistic-triggers-fig4.webp)

**The evidence for $K_{\max}$.** With $K_{\max} = 0$ the ASR on Qwen3-VL is 48.28%, which is exactly the IDEATOR baseline in Table 2. At $K_{\max} = 10$ it reaches 53.27% (LLaVA-OV-1.5: 47.06% to 52.28%), and going to 15 adds almost nothing. The authors read the plateau as "the remaining samples are intractable".

**When it fails.** The ASR curve rises with the *number of attempts per sample*. Part of that is the policy learning a better style; part is plain best-of-$K$ sampling from a stochastic sampler and target, selected by the same judge that later scores the result. The paper does not separate the two.

## 7. Results

Four targets, two benchmarks (MM-SafetyBench, VLBreakBench), ASO stacked on several base attacks. ASR is the fraction HarmBench judges harmful; the Harmfulness Score (HS) is the judge's log-odds $\log P_{\mathcal{J}}(\text{yes}) - \log P_{\mathcal{J}}(\text{no})$. Runs used 8 A100 GPUs at 512 by 512.

**MM-SafetyBench (Table 1), ASR, base then with ASO:**

| Base attack | Qwen3-VL | LLaVA-OV-1.5 | GPT-4.1-mini | Gemini-2.5-Flash |
| --- | --- | --- | --- | --- |
| QR Attack | 38.99% to 42.98% | 37.80% to 44.35% | 54.26% to 57.36% | 55.04% to 62.79% |
| SI Attack | 39.31% to 42.58% | 37.82% to 44.25% | 53.49% to 57.36% | 55.81% to 62.02% |
| HIMRD | 87.38% to 89.52% | 52.92% to 55.42% | **72.80% to 71.77%** | 75.97% to 76.74% |

**VLBreakBench (Table 2), ASR, base then with ASO:**

| Base attack | Qwen3-VL | LLaVA-OV-1.5 | GPT-4.1-mini | Gemini-2.5-Flash |
| --- | --- | --- | --- | --- |
| IDEATOR | 48.28% to 53.27% | 47.06% to 52.28% | 74.82% to 75.54% | 66.19% to 66.91% |
| SI Attack | 49.72% to 53.16% | 48.39% to 51.28% | 75.54% to 77.70% | 58.99% to 65.47% |

Three patterns stand out. Gains are largest where the base attack is mid-strength (QR and SI on LLaVA and Gemini, about 6 to 8 points). They shrink where the base attack is already strong (HIMRD, IDEATOR on GPT-4.1-mini, under 1 point). And in one cell, HIMRD on GPT-4.1-mini, ASO *lowers* ASR by 1.03 points and HS from 4.22 to 3.64. The paper's text calls the effect universal; the table says "nearly".

**The ablation (Table 4)** splits each gain into probing and enhancement:

| Attack, model | Base | + Probing | ++ Enhance | Probing gain | Enhance gain |
| --- | --- | --- | --- | --- | --- |
| QR, Qwen3-VL | 38.99% | 40.48% | 42.98% | +1.49 | +2.50 |
| QR, LLaVA-OV-1.5 | 37.80% | 40.12% | 44.35% | +2.32 | +4.23 |
| SI, Qwen3-VL | 39.31% | 40.62% | 42.58% | +1.31 | +1.96 |
| SI, LLaVA-OV-1.5 | 37.82% | 39.96% | 44.25% | +2.14 | +4.29 |
| HIMRD, Qwen3-VL | 87.38% | 89.17% | 89.52% | +1.79 | +0.35 |
| HIMRD, LLaVA-OV-1.5 | 52.92% | 54.23% | 55.42% | +1.31 | +1.19 |

The paper says Enhance gives "a much more significant ASR jump" in all cases. It does in four of six rows; in both HIMRD rows probing contributes more.

![Random Gaussian noise versus optimized style on GPT-4.1-mini (QR Attack, MM-SafetyBench): every noise level falls below the base attack, while ASO rises above it](/imgs/blogs/adversarial-style-optimization-enhancing-vlm-jailbreaks-by-grpo-based-stylistic-triggers-4.webp)

**The control we find most convincing** is the noise ablation in the supplement, plotted above. Adding Gaussian noise at $\sigma$ from 0.1 to 0.5 drops GPT-4.1-mini's ASR from 54.26% to between 48.03% and 50.75%, and HS from $-0.62$ to as low as $-1.51$. On Qwen3-VL noise does nothing consistent (38.81% to 40.48% against a 38.99% base). So "any perturbation helps" is ruled out: degradation makes the attack *easier* to refuse, coherent style makes it harder. That is the paper's real scientific finding.

![Supplementary Figure 2 from Luo et al. (2026): mean ASR and standard deviation over five seeds for the baseline and ASO, QR Attack and IDEATOR on Qwen3-VL and LLaVA-OV-1.5](/imgs/blogs/adversarial-style-optimization-enhancing-vlm-jailbreaks-by-grpo-based-stylistic-triggers-fig5.webp)

**Seed stability.** Over five seeds, baseline and ASO error bars do not overlap for these four open-model cells; commercial targets were not tested this way.

**What might not transfer.** Logit access for the rejected branch; a large query budget (up to 10 attempts per sample, each a 16-step FLUX edit plus target and judge calls); and HarmBench as both reward and evaluator.

## 8. What this means for defenders

The shape of the failure matters more than the recipe. This section is our synthesis, not the paper's.

| Finding | Defensive implication |
| --- | --- |
| Noise lowers ASR, coherent style raises it | Robustness to corruptions (ImageNet-C style) is not evidence of safety robustness. Test with style shifts, not just noise. |
| Medium & Texture styles win across models | Start safety data augmentation with sketch, charcoal, monochrome and vintage renders of existing refusal examples. |
| The optimal style is model-specific | A fixed style-canonicalization filter will be bypassed by a style it was not trained on; pair it with training-time invariance. |
| Only the image changes; the text prompt is fixed | A refusal that depends on visual style is a refusal tied to surface features, not to the meaning of the request. |

Three defenses follow:

1. **Style-invariant refusal training.** Add restyled copies of every image safety example with the same refusal target, and track over-refusal on stylized *benign* images so one failure is not traded for another.
2. **Inference-time consistency checks.** Render the input in a canonical photographic style (or two or three fixed styles) and refuse if any view triggers refusal. Extra forward passes make this a fit for high-risk deployments.
3. **Red-team with the attacker's tool.** Alternate tuning the DB-GRPO editor against the model and fine-tuning the model on its successes, the same loop as [automated red-teaming of video models](/blog/paper-reading/computer-vision/tear-temporal-aware-automated-red-teaming-for-text-to-video-models). For instructions smuggled in through images, steering defenses like [ARGUS](/blog/paper-reading/computer-vision/argus-defending-against-multimodal-indirect-prompt-injection-via-steering-instruction-following) are a complementary layer.

## 9. Critique

**What is strong.** The noise ablation separates "perturbation" from "style", the first confound a skeptic would raise. The tiered reward is clean design: ordered and dense. The style-preference chart is a useful map for defenders.

**What is weak.**

- **Selection with the evaluation judge.** DB-GRPO keeps an image once the reward crosses the threshold, the accepted branch *is* HarmBench, and the reported ASR also comes from HarmBench. Some uplift is likely best-of-10 selection rather than a better style. The missing baseline is best-of-10 with the frozen probed style, scored the same way.
- **Comprehension is asserted, never measured.** The paper measures only the safety half of its title finding. Per-style captioning or VQA accuracy would turn the claim into a result and rule out the cheaper explanation that some restyles just make content harder to see.
- **Small, uneven effects.** Most gains are 2 to 8 points, HIMRD on GPT-4.1-mini goes backwards, and the "much more significant" Enhance claim fails in two of six ablation rows.
- **Method description inconsistencies.** The advantage is described as the raw reward in the text but group-normalized in Eq. 9. Algorithm 1's update subtracts the gradient of an objective it says it maximizes. The table of per-category results (Table 3) does not say which target model it reports.
- **Commercial-target reward.** How the rejected-branch probabilities were obtained for GPT-4.1-mini and Gemini-2.5-Flash is not described.

**What would change my mind.** If a best-of-10 baseline using the frozen probed style, scored by the same judge, matched ASO's numbers, I would read the RL stage as a sampling budget rather than a learned style. Conversely, if ASO's successful images transferred to a *different* judge and a *different* target at a similar rate, I would accept that it learns a genuinely general stylistic trigger.

## 10. Where this goes next

The orchestrating note for this series asks two questions: can safety training be made style-invariant, and which representation layers break the invariance? Here are concrete follow-ups.

1. **Locate the break layer.** *Question:* at which layer of the vision encoder or LLM does the "harmful intent" signal stop being style-invariant? *Why now:* ASO provides matched pairs, the same content in a refused style and in an accepted style, which is exactly what a layer-wise probe needs. *First experiment:* on Qwen3-VL, extract a refusal direction from photo inputs (difference of means between refused and accepted prompts), then project restyled inputs onto it at every layer and plot where the projection collapses for accepted restyles.
2. **Encoder or LLM?** *Question:* is the gap created by the vision encoder's features or by how the LLM's safety tuning reads them? *Why now:* LLaVA-OV-1.5 is fully open, so components can be swapped. *First experiment:* activation patching: patch the projector output from the photo version into the accepted sketch version and see whether the refusal returns.
3. **Style-invariant safety fine-tuning with an adaptive adversary.** *Question:* does adversarial training against a DB-GRPO editor close the gap without raising over-refusal? *Why now:* the paper supplies the adversary and a reward. *First experiment:* three rounds of alternate training on LLaVA-OV-1.5, tracking ASR under a fresh ASO run and refusal rate on stylized benign images from a general VQA set.
4. **Measure the comprehension half.** *Question:* does comprehension really survive every style in the pool? *Why now:* the 16-style pool and editor are released. *First experiment:* restyle a benign VQA benchmark with each style and report accuracy next to ASR; styles where accuracy also drops are a weaker kind of attack than the paper claims.
5. **Judge-robust red-teaming.** *Question:* how much of ASO's gain survives a held-out judge? *Why now:* the reward and evaluator are separable by design. *First experiment:* train with HarmBench in the reward, evaluate with a different safety classifier, and report the gap.

## References

- Luo, Guo, Yao, Ding. *Adversarial Style Optimization: Enhancing VLM Jailbreaks by GRPO-based Stylistic Triggers Optimization.* CVPR 2026. [arXiv:2607.21619](https://arxiv.org/abs/2607.21619), [CVPR page](https://cvpr.thecvf.com/virtual/2026/poster/37412), [supplementary material](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Luo_Adversarial_Style_Optimization_CVPR_2026_supplemental.pdf), [code](https://github.com/bingjunluo/ASO).
- Liu et al. *Flow-GRPO: Training Flow Matching Models via Online RL.* arXiv:2505.05470, 2025.
- Mazeika et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* arXiv:2402.04249, 2024.
- Zhao et al. *Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency.* ICCV 2025.
- On this site: [GRPO explained](/blog/machine-learning/reinforcement-learning/grpo-group-relative-policy-optimization), [Safety alignment should be more than a few tokens deep](/blog/paper-reading/ai-interpretability/safety-alignment-should-be-made-more-than-just-a-few-tokens-deep), [TEAR: automated red-teaming for text-to-video](/blog/paper-reading/computer-vision/tear-temporal-aware-automated-red-teaming-for-text-to-video-models), [ARGUS: defending against multimodal indirect prompt injection](/blog/paper-reading/computer-vision/argus-defending-against-multimodal-indirect-prompt-injection-via-steering-instruction-following).

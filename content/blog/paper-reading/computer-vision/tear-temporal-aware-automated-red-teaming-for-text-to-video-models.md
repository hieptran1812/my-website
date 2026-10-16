---
title: "TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models"
publishDate: "2026-10-16"
date: "2026-10-16"
description: "Why text-to-video safety risks live in time rather than in any single frame, and how TEAR trains a prompt generator that finds them, read with a defender's eye."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["safety-robustness", "cvpr-2026", "cvpr-2026-oral", "red-teaming", "text-to-video", "video-generation", "ai-safety", "ppo", "tear"]
author: "Hiep Tran"
featured: true
readTime: 22
image: "/imgs/blogs/tear-temporal-aware-automated-red-teaming-for-text-to-video-models-fig1.webp"
excerpt: "TEAR shows that a text-to-video prompt can pass every text filter and still yield a policy-violating video, because the harm only exists in the order of events. A walkthrough of the method, the numbers, and what a temporal-aware defense needs."
paper:
  title: "TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models"
  authors: "Jiaming He, Guanyu Hou, Hongwei Li, Zhicong Huang, Kangjie Chen, Yi Yu, Wenbo Jiang, Guowen Xu, Tianwei Zhang"
  venue: "CVPR 2026 (Oral)"
  url: "https://arxiv.org/abs/2511.21145"
---

> [!tldr]
> **What it claims.** Text-to-video (T2V) models have a safety hole that text and image red-teaming cannot see: a prompt made of individually harmless event descriptions, placed in a strict time order, can produce a video whose *sequence* violates policy. TEAR is an automated red-teaming framework that trains an LLM to find such prompts.
>
> **Why it matters.** Most deployed T2V safety stacks check the prompt text and, at best, individual frames. Neither looks at what happens *between* frames.
>
> **Most surprising result.** On Hunyuan-Video, TEAR's prompts pass the text filters about 96% of the time while 82.3% of them produce videos the video judge flags. The strongest adapted baseline (FLIRT) reaches 57.2% and passes the text filters only about half the time.
>
> **Where it is weak.** Success is scored by a single GPT-4o video judge, and that same style of judgment feeds the refinement loop. There is no human evaluation in the main paper, so part of the gain may be optimization against the judge.

This is a red-teaming paper, so a word on framing. I read it as an engineer who has to *defend* a video model. I explain the mechanism at the level the paper does, and I do not reproduce any of the paper's harmful example prompts or describe harmful content. The useful lesson for builders is structural: where in the stack temporal risk hides, and what kind of filter could see it.

## The problem: harm that lives between frames

Red-teaming for language models looks for prompts that make a model *say* something bad. Red-teaming for image models looks for prompts that make a model *draw* something bad. In both cases, the harmful content is present in a single output object you can inspect.

Video adds a dimension. A sequence of frames can each be innocuous while the story they tell together is not. The paper's Figure 1 (which I do not reproduce, because its frames and prompt text are the harmful example) shows the pattern abstractly: a seed that names a harmful act is filtered, but a rewritten prompt describing a few mundane physical events with explicit timing ("after two seconds") passes the filter, and the resulting video depicts the act.

The authors call this a **temporal-aggregation attack**: individually benign descriptions whose concatenation produces an unsafe video. Existing red-teaming methods, adapted from images, treat a video as a stack of independent frames. They have no mechanism for searching over the ordering of events.

![Illustrative: six frames grouped as events A, B and C each score below a per-frame filter threshold, while a sequence-level risk computed over a window of frames rises above the threshold by the last frames](/imgs/blogs/tear-temporal-aware-automated-red-teaming-for-text-to-video-models-1.webp)

The figure above is my illustration, not data from the paper. Every per-frame bar stays under the filter threshold. A score computed over a window of frames is the only one that crosses it. That is the defender's core problem in one picture: **a per-frame classifier is structurally blind to risk that exists only in the composition.**

### The formal objective

The paper states red-teaming as a set-search problem. Let $M$ be the T2V model mapping a prompt $p$ to a video $v = M(p)$. Let $\Phi_P$ be a text judgment system (1 means "unsafe text") and $\Phi_V$ a video judgment system (1 means "unsafe video"). The red-teaming system $R$ takes a target category $T$ and a seed set $P_{vu}$ of unsafe prompts, and returns:

$$
P_v^{*} = R(P_{vu}, T, M, \Phi_P, \Phi_V) \quad \text{s.t.} \quad p \in P_v^{*} \iff \Phi_P(p) = 0 \;\wedge\; \Phi_V(M(p)) = 1
$$

In words: find prompts the text filter calls safe whose videos the video judge calls unsafe. The paper notes that finding the full set is intractable (discrete prompt space, expensive video generation), so the practical goal is a guided search in a continuous space, with emphasis on vulnerabilities unique to time.

![Illustrative scatter: x axis is the prompt judge score, y axis the video judge score; naive prompts sit bottom-left, FLIRT prompts spread across the top on both sides of the text threshold, and TEAR prompts cluster in the top-left blind-spot quadrant](/imgs/blogs/tear-temporal-aware-automated-red-teaming-for-text-to-video-models-2.webp)

This picture places the paper's Table 1 numbers (Hunyuan-Video) onto the two judges' axes; the dot positions are illustrative. The top-left quadrant is the only one that counts as a finding. A baseline that lands top-right has produced a harmful video, but the text filter would have caught the prompt, so it is not a real bypass.

### Threat model

The threat model is a **developer auditing their own model before release**. The developer has auxiliary LLMs and vision-language models acting as automated evaluators of prompts and videos. This matters for how to read the paper: it is a tool for finding failure cases you then fix, and its evaluation assumes white-box access to your own pipeline plus black-box access to commercial APIs for transfer tests.

## TEAR at a glance

TEAR has three components: a **temporal-aware test generator** (an LLM that writes candidate prompts), a **refine model** (a multimodal LLM that revises prompts using judge feedback), and the **target T2V model**. The generator is trained in three stages; the refine loop then runs at test time.

![Figure 2 from He et al. (2026): the TEAR framework, with test-case generator optimization in three stages (dataset construction, initial training, temporal-aware optimization) and the red-teaming test-case generation loop with prompt judge, video judge and refine model](/imgs/blogs/tear-temporal-aware-automated-red-teaming-for-text-to-video-models-fig1.webp)

| Stage | What happens | Signal used |
| --- | --- | --- |
| 1. Dataset construction | Rewrite seed prompts into time-ordered sequences; keep pairs that pass text judges and fail video judges | $\Phi_P$, $\Phi_V$ |
| 2. Initial generator training | Supervised fine-tuning of Llama-3 (LoRA) on seed to rewrite pairs | next-token NLL |
| 3. Temporal-aware online preference learning | PPO against prompt-level and video-level rewards, with a KL leash | $R_{pmt}$, $R_{con}$ |
| Test time: refinement | A Qwen-3-VL refine model revises prompts for up to 8 rounds | $\Phi_P$, $\Phi_V$ feedback |

The paper frames generation as a Markov decision process: the state is the token sequence so far, the action is the next token, and the reward arrives on the finished prompt. This is the standard RLHF setup, applied to an attacker LLM instead of an assistant. If PPO for language models is new to you, [the PPO for LLM fine-tuning walkthrough](/blog/machine-learning/reinforcement-learning/ppo-for-llm-fine-tuning) covers the machinery that stage 3 reuses.

## Stage 1: rule-based temporal rewriting

**Problem.** An LLM fine-tuned on generic red-teaming data has never seen the shape of a temporal attack. It needs examples.

**Analogy.** Think of a film script versus a logline. A logline names the event; a script only lists stage directions and timings. A censor reading the logline objects; a censor skimming stage directions one line at a time may not.

**Mechanism.** The authors build a meta harmful dataset $D_m$ of seed prompts and ask an LLM to rewrite each seed $p_s$ with a time-ordered instruction sequence. Three rules govern the rewrite:

1. **Temporal deconstruction.** Split the directive into a chronological sequence of discrete, static event descriptions.
2. **Sequential enforcement.** Insert explicit temporal connectives ("first", "after two seconds") to force a strict order.
3. **Temporal-space synthesis.** No single description carries the harm; it emerges only from the composition.

The output is a set of pairs $D_p = \{\langle p_s^1, p_t^1\rangle, \dots, \langle p_s^n, p_t^n\rangle\}$, where $p_t$ is the rewritten prompt. A filter keeps only pairs meeting the red-teaming objective: $\Phi_P(p_t) = 0$ and $\Phi_V(M(p_t)) = 1$.

**When it fails.** The rules only cover one attack shape: decomposition plus ordering. They do not cover, for example, harm that emerges from camera motion or from a slow change in a single object. The generator can only discover what the seed distribution makes reachable.

## Stage 2: initial generator training

**Mechanism.** Plain supervised fine-tuning. Given seed $p_s$ and an instruction $I$, train the generator to produce $p_t$ with next-token negative log-likelihood:

$$
\mathcal{L}_{Ini} = -\,\mathbb{E}_{(p_s, p_t, T) \sim D_p} \log p(p_t \mid p_s, I)
$$

Here $p(\cdot)$ is the generator's autoregressive distribution, $I$ is the fixed instruction, and $T$ the target category. The paper trains Llama-3 with LoRA for 4,000 steps at batch 8 and peak learning rate ${1.0 \times 10^{-5}}$ with a cosine schedule.

**Worked example.** If the reference rewrite $p_t$ is 60 tokens and the generator assigns each correct token probability 0.5, the loss for that example is ${60 \times \ln 2 \approx 41.6}$ nats. Training pushes those per-token probabilities up until the generator reliably produces rewrites in this style. The result, $G_{initial}$, knows the *shape* of a temporal prompt but has never seen a video.

**When it fails.** SFT imitates the dataset. It cannot learn which rewrites actually move the target model, because the loss never touches a generated video. That is stage 3's job.

## Stage 3: temporal-aware online preference learning

This is the paper's core contribution. The authors decompose the red-teaming objective into two spaces: **prompt space** (stay textually safe and on-pattern) and **temporal space** (make the video actually match the harmful semantics across time).

### The prompt-level reward

**Problem.** A prompt that drifts toward obviously harmful wording gets caught. A prompt that drifts away from the temporal style loses its attack shape.

**Mechanism.** The prompt-level reward is a weighted sum of two terms:

$$
R_{pmt}(p_t) = \mathbb{E}_{p_t \sim G_\delta(p_s)}\left[\alpha_1 \,\big(1 - g_t(p_t)\big) + \alpha_2 \,\frac{g_r(p_t) + 1}{2}\right]
$$

- $G_\delta$ is the policy being trained.
- $g_t(\cdot) \in [0,1]$ is the confidence of a pre-trained hate speech classifier, so ${1 - g_t}$ rewards text that looks safe.
- $g_r(\cdot) \in [-1, 1]$ is a pattern-alignment score; mapping it with $(g_r + 1)/2$ puts it on $[0,1]$.
- $\alpha_1, \alpha_2$ are weights (the main text does not report their values).

The pattern score is a cosine similarity to a prototype. Pick a subset $P_{ref}$ of representative temporal-style rewrites, embed each with a sentence encoder $T_p(\cdot)$, and average:

$$
g_r(p_t) = \frac{\left\langle T_p(p_t),\; \frac{1}{|P_{ref}|}\sum_{p' \in P_{ref}} T_p(p') \right\rangle}{\lVert T_p(p_t)\rVert \cdot \left\lVert \frac{1}{|P_{ref}|}\sum_{p' \in P_{ref}} T_p(p') \right\rVert}
$$

![Cosine geometry of the pattern-alignment reward: reference prompts cluster in one direction of an embedding plane, their mean is the prototype, a candidate 20 degrees away scores 0.97 after rescaling and one 110 degrees away scores 0.33](/imgs/blogs/tear-temporal-aware-automated-red-teaming-for-text-to-video-models-3.webp)

**Worked example.** Candidate A sits 20 degrees from the prototype, so $g_r = \cos 20^\circ \approx 0.94$ and the rescaled term is 0.97. Candidate B sits 110 degrees away, so $g_r \approx -0.34$ and the rescaled term is 0.33. Take my own illustrative weights $\alpha_1 = \alpha_2 = 0.5$ and a classifier confidence $g_t = 0.1$ for both. Then A scores ${0.5 \times 0.9 + 0.5 \times 0.97 \approx 0.94}$ and B scores ${0.5 \times 0.9 + 0.5 \times 0.33 = 0.615}$. The reward prefers the prompt that both looks safe and keeps the temporal shape.

**When it fails.** $g_t$ is a single hate speech classifier, while evaluation uses three text judges (a toxicity detector, an NSFW detector and Llama-Guard-3-8B). The reward is a proxy for the deployed filter, and any filter feature the proxy does not encode is a gap the policy will not learn to avoid. It works here because the evaluation filters are all text classifiers of similar character.

### The temporal consistency reward

**Problem.** Optimizing prompt space alone does not guarantee the video depicts the target semantics. A prompt can be safe and on-style while the model renders something harmless.

**Mechanism.** For each training step, sample $p_t$ from $G_\delta$, query the T2V model for a video $v_{p'} = M(p_t)$, sample frames $F_{v_{p'}} = \{f_1, \dots, f_i\}$, and encode them with a video encoder $E_v$ into temporally ordered latent tokens. Two scoring models then judge the video:

$$
R_{con}(p_s, p_t) = \mathbb{E}_{v_{p'} \sim M(p_t)} \min\!\left(\beta,\; \frac{g_{gc}\big(p_s, E_v(F_{v_{p'}})\big) - \gamma_1}{\theta_1} + \frac{g_{ic}\big(E_v(F_{v_{p'}})\big) - \gamma_2}{\theta_2}\right)
$$

- $g_{gc}$ is a **global consistency** score: how well the whole video aligns with the meta harmful semantics of the *seed* $p_s$ (not the rewritten prompt). This is the term that rewards harm that emerges only over time.
- $g_{ic}$ is an **inner consistency** score: temporal coherence of the video itself, a utility check so the policy cannot win with broken footage.
- $\gamma_1, \gamma_2$ are offsets and $\theta_1, \theta_2$ are scales that normalize the two scores; $\beta$ caps the total.

Both scorers come from pre-trained video reward models trained on large-scale human preference data for video generation (the paper cites Liu et al., 2025). The key design choice is that $g_{gc}$ compares the video to the seed's meaning, so the reward measures what the *sequence* depicts rather than what any one prompt sentence says.

![Two panels: left, the consistency reward rises linearly with the combined score then goes flat at a cap beta; right, the PPO policy distribution shifts right of the initial generator's distribution, held by a KL penalty](/imgs/blogs/tear-temporal-aware-automated-red-teaming-for-text-to-video-models-4.webp)

**Worked example.** Suppose $g_{gc} = 0.7$, $\gamma_1 = 0.5$, $\theta_1 = 0.4$, giving 0.5; and $g_{ic} = 0.8$, $\gamma_2 = 0.6$, $\theta_2 = 0.4$, giving 0.5. The sum is 1.0. With $\beta = 0.8$, the reward is clipped to 0.8. These constants are mine; the main text gives none. The left panel shows why the cap exists: without it, the policy is paid more and more for ever more extreme videos, which is a classic reward-hacking path.

**When it fails.** The global consistency model is the attacker's notion of "did the harm happen". If it was trained on generic preference data, it may reward videos that *look* aligned with the seed without being policy-violating by a human's judgment. Its score is also only as temporal as its encoder: a scorer that pools frames before comparing would reintroduce the per-frame blind spot.

### The full PPO objective

The two rewards combine in a PPO objective with a KL penalty to the stage-2 generator:

$$
\zeta = \mathbb{E}_{p_s \sim D_m,\; p_t \sim G_\delta}\left[R_{pmt}(p_t) + R_{con}(p_s, p_t) - \lambda \log \frac{G_\delta(p_t \mid p_s)}{G_{initial}(p_t \mid p_s)}\right]
$$

$\lambda$ weights the KL term, which stops the policy from drifting into gibberish that games the rewards (over-optimization). The right panel above shows the intent: the trained distribution shifts toward higher-reward prompts but stays tethered to the fluent, on-distribution SFT model. Training uses AdamW at learning rate ${1.0 \times 10^{-6}}$ with $\gamma = 1.0$ and $\lambda = 0.95$; I read that $\lambda$ as the GAE parameter, since the paper reuses the symbol and does not report a separate KL coefficient.

## Test-case refinement: the closed loop

**Problem.** The trained generator gives a good first draft. Some drafts still trip a text filter or fail to produce the target video.

**Mechanism.** A refine model $R_m$, Qwen-3-VL with few-shot in-context examples, receives the prompt $p_t$, the generated video $M(p_t)$, and structured feedback from both judges: a score, an explanation and a suggestion. It writes a revised prompt $p_{t+1}$. The loop repeats until both judges agree on "text safe, video unsafe" or the round budget runs out. Generation uses beam search with $b = 16$ and a 100-token limit.

![Figure 4 from He et al. (2026): attack success rate and NSFW filter pass rate versus refining rounds (direct, 1, 3, 6, 8) for five T2V models](/imgs/blogs/tear-temporal-aware-automated-red-teaming-for-text-to-video-models-fig4.webp)

The refine loop is doing a lot of the work. ASR starts at 57% to 71% with no refinement, rises sharply over the first three rounds, and reaches 83% to 95% after eight. The NSFW filter pass rate starts at 79% to 86% and climbs past 95%. In other words, the *trained generator alone* already matches or beats the best baseline, but roughly a quarter of the final ASR comes from a black-box MLLM editor.

**When it fails.** The refine model sees the video judge's verdict, and the final evaluation uses the same kind of video judge. That creates a Goodhart risk I return to in the critique.

## Results

### Setup

| Item | Value |
| --- | --- |
| Open-source targets | Wan 2.2-14B, Hunyuan-Video |
| Commercial targets | Veo-3.1, Hailuo-2.3, Luma Ray-2 |
| Seeds | 390 meta-harmful prompts (from T2VSafetyBench plus added categories), 65 per category |
| Categories | Violence, Gore, Self-harm, Pornography, Illegal Activity, Disturbing Content |
| Text judge $\Phi_P$ | Toxicity detector, NSFW text detector, Llama-Guard-3-8B |
| Video judge $\Phi_V$ | GPT-4o with T2VSafetyBench criteria |
| Baselines | ART and FLIRT (adapted from T2I), UVD, T2VSafetyBench prompts, plus a naive baseline |

ASR is the fraction of prompts whose video $\Phi_V$ flags; PSR (prompt successful pass rate) is the fraction each text judge lets through.

![Table 1 from He et al. (2026): success cases by category, attack success rate and prompt pass rate for six methods on Hunyuan-Video and Wan 2.2](/imgs/blogs/tear-temporal-aware-automated-red-teaming-for-text-to-video-models-fig2.webp)

The headline comparison, pulled from Table 1:

| Model | Method | ASR | PSR (TD / NSFW / Llama Guard) |
| --- | --- | --- | --- |
| Hunyuan-Video | Naive | 2.6% | 98.3 / 98.9 / 99.4 |
| Hunyuan-Video | FLIRT | 57.2% | 51.4 / 52.2 / 51.9 |
| Hunyuan-Video | ART | 52.6% | 92.2 / 93.0 / 92.7 |
| Hunyuan-Video | **TEAR** | **82.3%** | 96.7 / 96.5 / 95.6 |
| Wan 2.2 | FLIRT | 56.4% | 48.6 / 50.8 / 50.5 |
| Wan 2.2 | ART | 49.7% | 91.8 / 92.4 / 93.2 |
| Wan 2.2 | **TEAR** | **80.5%** | 94.3 / 97.4 / 94.9 |

Two readings matter for a defender. First, FLIRT's ASR is respectable, but about half its prompts fail the text filters, so its *effective* bypass rate is far lower. ART has the opposite profile: it passes filters but produces fewer harmful videos. TEAR is the only method high on both axes, which is exactly the blind-spot quadrant from the scatter earlier.

Second, the per-category counts are uneven. On Hunyuan-Video, TEAR succeeds on 61 of 65 Violence seeds and 61 of 65 Disturbing Content seeds, but only 29 of 65 Pornography seeds. Content that is harmful in a single frame is easier for existing filters and model alignment to block; content that is harmful in a sequence of actions is where temporal composition pays off.

### Seed-free generation

Without seed prompts, TEAR reaches 79.2% ASR on Hunyuan-Video and 76.9% on Wan 2.2, against FLIRT's 55.1% and 53.8%. NSFW pass rates stay at 96.2% and 97.0%. The generator has internalized the attack shape rather than paraphrasing seeds.

### Commercial services

![Figure 3 from He et al. (2026): radar charts of attack success rate, API-level filter pass rate and NSFW filter pass rate per category for Veo-3.1, Hailuo-2.3 and Ray-2](/imgs/blogs/tear-temporal-aware-automated-red-teaming-for-text-to-video-models-fig3.webp)

On Veo-3.1, Hailuo-2.3 and Ray-2, the prompts pass the API-level and NSFW filters at close to 98%, while ASR is at or above 85% for most categories. Pornography is again the lowest, below 80%. Ray-2 shows a slightly lower ASR, which the authors attribute to weaker prompt-video consistency (a property VBench also reports). That last point is a quiet irony: **a model that follows prompts less faithfully is, for this attack, safer.**

### Transfer across models

![Table 3 from He et al. (2026): transfer attack success rates when prompts optimized on one T2V model are used on another, across five models](/imgs/blogs/tear-temporal-aware-automated-red-teaming-for-text-to-video-models-fig5.webp)

Prompts optimized on one model and replayed on another keep most of their power. The average across all 20 source and target pairs is 76.4%, with most cells between 70% and 82%. The peak is Ray-2 to Hailuo-2.3 at 82.6%; the floor is Veo-3.1 to Ray-2 at 68.5%. The authors read this as a shared weakness of T2V safety rather than a quirk of one model.

### Diversity and generation settings

Prompt diversity stays high: 1 minus average Self-BLEU ranges from about 0.71 (Self-harm) to 0.76 (Illegal Activity), and 1 minus cosine similarity from 0.69 to 0.73. The generator is not collapsing to one template, which matters because a defender who blocks one template would otherwise be done.

Generation settings move ASR too. ASR rises with inference steps and plateaus around 50, peaks at a moderate CFG scale (82.79% at 50 steps and CFG 7.5 in their sweep), and is higher with more frames (85.90% at 256 frames and CFG 7.5). The authors' explanation is that better video quality gives the judge clearer evidence. A defender's reading: longer, higher-fidelity videos give the temporal attack more room to compose events.

### What might not transfer

The load-bearing parts of the setup are the GPT-4o video judge and the reward models from video preference data. A different judge with stricter criteria, or human raters, could move every ASR in this section. The seed set is 390 prompts in six categories, so per-category numbers rest on 65 examples each; a five-point gap there is about three prompts.

## A defender's reading: why static filters miss this

The paper's attack surface lists the places a temporal risk can slip past. Turning that around gives a checklist for builders. This section is my synthesis, not something the paper evaluates.

| Defense layer | What it sees | Why TEAR passes it | What would close the gap |
| --- | --- | --- | --- |
| Prompt keyword or classifier filter | Words, toxicity of each phrase | Each event description is benign | Ask a guard model what the *sequence* of events depicts as a whole |
| Per-frame image classifier | One frame at a time | No single frame is policy-violating | Score sliding windows of frames, or clip-level embeddings |
| Model-level alignment | Training-time refusals | Trained on single-concept harmful prompts | Fine-tune on TEAR-style cases, which is the paper's stated purpose |
| Post-hoc video judge | Whole clip | This is what catches it, if run | Run it on outputs in production, not only in audits |

The cheapest fix sits at the text side: the attack works partly because each sentence is judged alone. A guard prompt that asks a capable LLM to narrate the end state of the described sequence attacks the decomposition directly. The most robust fix sits at the output side: a clip-level judge that reasons over temporal order. The paper's own success metric is exactly such a judge, which tells you it can see the problem.

For broader context on video safety evaluation, see [evaluating and red-teaming video generation](/blog/machine-learning/video-generation/evaluating-and-red-teaming-video-generation). For the guard-model layer, [production LLM guardrails](/blog/machine-learning/large-language-model/production-llm-guardrails-models-evaluation-finetuning) covers how guard models are evaluated and fine-tuned. Two siblings in this wave attack multimodal models from other angles: [style-triggered VLM jailbreaks optimized with GRPO](/blog/paper-reading/computer-vision/adversarial-style-optimization-enhancing-vlm-jailbreaks-by-grpo-based-stylistic-triggers) and [ARGUS, a steering defense against indirect prompt injection](/blog/paper-reading/computer-vision/argus-defending-against-multimodal-indirect-prompt-injection-via-steering-instruction-following).

## Critique

**What is strong.** The framing is right and overdue: temporal composition is a genuinely new attack dimension for video, and the paper names it cleanly. The reward design is thoughtful. Comparing the video to the *seed's* semantics through $g_{gc}$, rather than to the rewritten prompt, is what makes the optimization temporal. Reporting text pass rates alongside ASR exposes the weakness of baselines that look strong on ASR alone. The transfer matrix across five models, three of them commercial, is a strong signal that this is not a single-model artifact.

**What is weak.**

- **Judge overlap.** The refine model optimizes against video-judge feedback, and ASR is measured by a GPT-4o video judge. If those share blind spots or biases, part of the gain is optimizing the metric. A held-out judge or human evaluation would separate real harm from judge agreement.
- **ASR definition.** ASR counts videos the judge flags; PSR is reported separately. It is not stated whether ASR is conditioned on passing the text judges. If not, the joint success rate is lower than 82.3%, though with PSR near 96% the gap is small for TEAR and large for FLIRT.
- **Missing ablations in the main text.** There is no ablation that removes $R_{con}$, removes $R_{pmt}$, or removes the KL term. The refine-rounds curve shows the generator alone reaches 57% to 71% ASR, but we cannot tell which stage-3 reward is responsible for beating the SFT model.
- **Unreported constants.** $\alpha_1, \alpha_2, \beta, \gamma_1, \gamma_2, \theta_1, \theta_2$ do not appear in the main text, and $\lambda$ is overloaded between the KL coefficient and the RL hyperparameters.
- **Baseline adaptation.** ART and FLIRT were designed for images and adapted by the authors. A baseline given access to the same refine MLLM would be a fairer comparison for the loop's contribution.

**What would change my mind.** If a human study on a sample of TEAR's "successful" videos found that a large fraction were not policy-violating by a human rater's judgment, the headline ASR would mostly measure GPT-4o's leniency. Conversely, if a held-out judge (a different VLM, never seen by the refine loop) reproduced ASR within a few points, I would treat the 82% figure as robust.

## Where this goes next

The orchestrating note for this series asked for follow-ups on temporal-aware defenses and on transfer. Here are four, each concrete enough to start.

1. **A temporal-aware safety filter trained on TEAR's own outputs.** *Question:* can a clip-level classifier trained on TEAR-found cases block the blind-spot quadrant without raising false refusals on benign multi-event prompts? *Why now:* TEAR produces a diverse labeled set of exactly the cases static filters miss, with diversity scores around 0.7. *First experiment:* train a small video transformer over sliding windows of frame embeddings on TEAR successes versus naive prompts' videos; report recall at a fixed false-positive rate on a benign multi-event set, against a per-frame classifier baseline.

2. **Prompt-side "end-state narration" guards.** *Question:* does asking a guard LLM to summarize what the full event sequence depicts recover most of the video judge's detections at a fraction of the cost? *Why now:* the attack relies on per-sentence benignity, and the paper's text judges are all per-text classifiers. *First experiment:* run the 390-seed TEAR set through Llama-Guard-3 twice, once on the raw prompt and once on an LLM-written one-sentence end-state summary; measure how much PSR drops.

3. **Why transfer works: shared temporal priors.** *Question:* is cross-model transfer (76.4% average) explained by shared text encoders, shared training data, or simply by prompt-following fidelity? *Why now:* Ray-2's lower ASR and lower prompt-video consistency hint that fidelity is the driver. *First experiment:* correlate each target model's transfer ASR with its VBench prompt-consistency score and its text-encoder family across the 20 pairs.

4. **Temporal unlearning or concept steering.** *Question:* can safety fine-tuning on composed sequences generalize to unseen compositions, or does it only patch the templates seen? *Why now:* TEAR gives a generator that keeps producing new compositions after a patch, so you can test generalization adversarially. *First experiment:* fine-tune Wan 2.2 with refusals on half of TEAR's categories, then re-run TEAR's PPO stage against the patched model and measure how many rounds it needs to recover ASR on the held-out categories.

5. **Judge robustness as a first-class result.** *Question:* how much does ASR change across video judges? *Why now:* the whole field (attack and defense) scores against VLM judges. *First experiment:* re-score a fixed set of TEAR videos with three different VLM judges plus a small human panel and report agreement (Cohen's kappa) per category.

## References

- He, J., Hou, G., Li, H., Huang, Z., Chen, K., Yu, Y., Jiang, W., Xu, G., Zhang, T. *TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models.* CVPR 2026 (Oral). [arXiv:2511.21145](https://arxiv.org/abs/2511.21145). [CVPR page](https://cvpr.thecvf.com/virtual/2026/oral/40357).
- Miao, Y. et al. *T2VSafetyBench: Evaluating the Safety of Text-to-Video Generative Models.* NeurIPS 2024 (seed prompts and video judge criteria).
- Li, G. et al. *ART: Automatic Red-teaming for Text-to-Image Models to Protect Benign Users.* NeurIPS 2024 (baseline).
- Mehrabi, N. et al. *FLIRT: Feedback Loop In-context Red Teaming.* EMNLP 2024 (baseline).
- Liu, J. et al. *Improving Video Generation with Human Feedback.* arXiv:2501.13918, 2025 (source of the consistency reward models).
- On this site: [evaluating and red-teaming video generation](/blog/machine-learning/video-generation/evaluating-and-red-teaming-video-generation), [PPO for LLM fine-tuning](/blog/machine-learning/reinforcement-learning/ppo-for-llm-fine-tuning), [production LLM guardrails](/blog/machine-learning/large-language-model/production-llm-guardrails-models-evaluation-finetuning).

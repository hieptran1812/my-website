---
title: "Persona Vectors: Reading and Steering an LLM's Character with One Direction"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How Chen et al. turn a one-line trait description into a single activation direction that monitors, steers, vaccinates against, and predicts personality drift in chat models, including drift caused by training data that looks harmless."
tags: ["paper-reading", "persona-vectors", "interpretability", "persona", "steering", "activation-steering", "ai-safety", "sycophancy", "hallucination", "emergent-misalignment", "large-language-model", "arxiv-2025"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 18
image: "/imgs/blogs/persona-vectors-monitoring-and-controlling-character-traits-in-language-models-fig1.webp"
excerpt: "One difference of means, four jobs: monitor the Assistant's character, steer it, vaccinate it during finetuning, and flag the training data that would corrupt it."
paper:
  title: "Persona Vectors: Monitoring and Controlling Character Traits in Language Models"
  authors: "Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey"
  venue: "arXiv 2025 (Anthropic Fellows Program)"
  url: "https://arxiv.org/abs/2507.21509"
---

> [!tldr]
> **Claim.** Character traits of a chat model's "Assistant" persona, such as evil, sycophancy and the propensity to hallucinate, live along linear directions in activation space. An automated pipeline can find each direction from nothing more than a trait name and a one-line description.
>
> **Why it matters.** One vector does four jobs: it monitors persona drift before the model speaks, steers it after the fact, prevents it during finetuning, and flags the training samples that would cause it.
>
> **Most surprising result.** *Adding* the evil direction during finetuning makes the finished model *less* evil. And data that looks clean to an LLM judge (vague prompts answered confidently) still gets flagged and still teaches hallucination.
>
> **Where it breaks.** Directions are coarse, supervised and prompt-elicited. Monitoring correlations drop sharply once you control for prompt type (as low as r = 0.245 for hallucination), and everything is measured on two 7–8B models with an LLM judge.

Chat models talk to us through a character. Post-training shapes that character to be helpful, harmless and honest, but the character is not bolted down. Bing's chatbot slid into threatening users, a system-prompt change made Grok praise Hitler, and an April 2025 RLHF update made GPT-4o so sycophantic that OpenAI rolled it back. Betley et al. went further and showed that finetuning on something as narrow as insecure code can make a model broadly misaligned, which they called **emergent misalignment**.

These are failures of *personality*, not capability, and we have had poor tools for them. This paper (Chen, Arditi, Sleight, Evans and Lindsey, from the Anthropic Fellows Program) proposes a single, cheap instrument for the whole lifecycle: a **persona vector**.

![Figure 1 from Chen et al. (2025): the pipeline turns a trait and its description into a persona vector, which then serves monitoring, post-hoc mitigation, preventative steering and data flagging](/imgs/blogs/persona-vectors-monitoring-and-controlling-character-traits-in-language-models-fig1.webp)

The figure above is the mental model for the rest of this post. Read the bottom row left to right: monitor, see finetuning cause a shift, subtract the vector to undo it, add the vector during training to avoid it, and project training data onto the vector to catch the problem before anything is trained.

The idea sits on the [linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis): high-level concepts are encoded as directions, so a dot product reads them and vector addition writes them. What is new here is not the difference-of-means trick, which is old, but the automation and, above all, the use of these directions on the *training* side.

## 1. Extracting a persona vector

**The problem.** To steer or monitor a trait you need its direction, and finding it has usually meant hand-curating contrastive examples for every concept. That does not scale to "every trait you might worry about next month."

**The analogy.** Think of measuring the "accent" of a speaker. Record the same person reading the same sentences twice, once told to speak with a strong regional accent and once told to speak neutrally. Average each set of recordings and subtract. What remains is the accent, because the words and the speaker cancel out.

**The mechanism.** The paper's version runs in two stages.

1. **Generate artifacts.** Given only a trait name and description (for evil: "actively seeking to harm, manipulate, and cause suffering"), one generic prompt template asks Claude 3.7 Sonnet to write **5 pairs** of contrastive system prompts (one eliciting the trait, one suppressing it), **40 questions** likely to evoke the trait (split 20 for extraction and 20 held out for evaluation), and a **judge rubric**. A GPT-4.1-mini judge uses that rubric to score any response 0 to 100 for trait expression.
2. **Contrast activations.** For each extraction question, the target model generates **10 rollouts** under each positive and each negative system prompt. Responses are filtered to keep only those that obeyed their prompt: trait score above 50 under the positive prompt, below 50 under the negative one. For each surviving response, the residual-stream activation at every layer is averaged over the response tokens. The persona vector is the difference between the two class means.

![The extraction pipeline from a one-line description to one vector per layer](/imgs/blogs/persona-vectors-monitoring-and-controlling-character-traits-in-language-models-1.webp)

**The math.** Let $a_\ell(x, y) \in \mathbb{R}^{d}$ be the mean residual-stream activation at layer $\ell$ over the tokens of response $y$ to prompt $x$, where $d$ is the model width ($d = 3584$ for Qwen2.5-7B). Let $\mathcal{T}$ be the set of kept trait-exhibiting (prompt, response) pairs and $\mathcal{N}$ the kept non-trait pairs. Then

$$
v_\ell = \frac{1}{|\mathcal{T}|}\sum_{(x,y)\in\mathcal{T}} a_\ell(x,y) \;-\; \frac{1}{|\mathcal{N}|}\sum_{(x,y)\in\mathcal{N}} a_\ell(x,y)
$$

This yields one candidate vector per layer. The authors pick a single layer by checking which one steers best: on Qwen2.5-7B-Instruct, layer 20 for evil and sycophancy and layer 16 for hallucination. Averaging over **response** tokens matters; Appendix A.3 finds this beats prompt-token positions for steering.

**Worked micro-example.** Pretend $d = 3$. Trait responses average to $[2, 1, 0]$ and non-trait responses to $[0, 1, -1]$. Then $v = [2, 0, 1]$. The middle coordinate, shared by both classes (topic, grammar, the question itself), cancels exactly. Only what differs between "evil" and "helpful" answers to the *same* questions survives.

**When it fails.** The pipeline assumes the trait can be **elicited by a system prompt**. Qwen and Llama will play evil on request with very low refusal rates, but a model with stronger safety training might refuse, leaving the positive class empty. A vague description also gives a vague direction, and the averaging produces a coarse, broad axis rather than fine distinctions.

The judge is load-bearing, so the authors validate it: two humans agreed with GPT-4.1-mini on which of two responses showed the trait more strongly in **284 of 300** comparisons (94.7%). Scores on their 20 evaluation questions also track external benchmarks with r between 0.941 and 0.964.

## 2. Steering: proving the vector is causal

**The problem.** A direction that merely correlates with a trait could be picking up topic or style. We need to know that pushing along it *causes* the behavior.

**The analogy.** A thermostat knob. If turning it up warms the room every time, the knob controls the heat, regardless of what the label says.

**The mechanism and math.** At each decoding step, add a scaled copy of the vector to the residual stream at layer $\ell$:

$$
h_\ell \leftarrow h_\ell + \alpha \, v_\ell
$$

where $h_\ell \in \mathbb{R}^{d}$ is the residual activation at the current token and $\alpha$ is a scalar steering coefficient. Positive $\alpha$ amplifies the trait; negative $\alpha$ suppresses it.

**Worked micro-example.** Continue the toy: $v = [2, 0, 1]$, so $\lVert v \rVert = \sqrt{5} \approx 2.236$ and $\hat v \approx [0.894, 0, 0.447]$. A token with $h = [0.5, 0.2, 0.1]$ projects to $h \cdot \hat v \approx 0.492$. Steer with $\alpha = 1.5$: $h' = [3.5, 0.2, 1.6]$, which projects to about 3.846. The jump is exactly $\alpha \lVert v \rVert \approx 3.354$, and nothing orthogonal to $v$ moved.

![Figure 3 from Chen et al. (2025): steering at each layer with coefficients from 0.5 to 2.5 raises trait expression for evil, sycophancy and hallucination on Qwen2.5-7B-Instruct, with example steered responses](/imgs/blogs/persona-vectors-monitoring-and-controlling-character-traits-in-language-models-fig3.webp)

The top row of Figure 3 shows trait scores rising with the coefficient, peaking in the middle layers. The bottom row is the qualitative proof: steering toward hallucination makes the model describe "traditional Martian dust soup according to Chef Zorg's cookbook" as if the cookbook existed.

**When it fails.** Steering is a blunt edit to every token. Large coefficients degrade fluency and general capability, which section 5 measures directly with MMLU.

## 3. Monitoring: reading the persona before the model speaks

**The problem.** In deployment you want to know that a prompt is pushing the model toward a trait *before* the harmful response is generated.

**The mechanism.** Take the activation at the **last prompt token**, the position just before the Assistant starts answering, and project it onto the persona direction. Formally the score is $s(x) = h_\ell^{\text{last}}(x) \cdot \hat v_\ell$, where $\hat v_\ell = v_\ell / \lVert v_\ell \rVert$.

The authors test it two ways. Claude 4.0 Sonnet writes eight system prompts that interpolate from trait-suppressing to trait-promoting, and a many-shot setting prepends 0, 5, 10, 15 or 20 trait-exhibiting examples. For each configuration they sample 10 responses per question, score them, and compare the mean score with the projection.

Across system prompts, projection and trait expression correlate at **r = 0.75–0.83**. The honest part is Appendix C.2, which splits that correlation in two:

| Trait | Setting | Overall r | Within-condition r |
| --- | --- | --- | --- |
| Evil | System prompt | 0.747 | 0.511 |
| Evil | Many-shot | 0.755 | 0.735 |
| Sycophancy | System prompt | 0.798 | 0.669 |
| Sycophancy | Many-shot | 0.817 | 0.813 |
| Hallucination | System prompt | 0.830 | 0.245 |
| Hallucination | Many-shot | 0.634 | 0.400 |

"Within-condition" holds the prompt fixed and asks whether the projection still ranks individual questions correctly. For hallucination under system prompts it barely does (0.245).

**Why it works and when it fails.** The monitor is excellent at telling an "evil" system prompt from a "helpful" one, and much weaker at subtle per-question variation. The authors say exactly this: the vector detects clear, explicit prompt-induced shifts and may be less reliable for subtle ones. My reading is that a production monitor built on this needs a calibrated threshold per deployment context, not one global cutoff.

## 4. Finetuning moves the model along the vector

**The problem.** Prompts are only half the story. Training can change a persona permanently, and often by accident.

**The setup.** The authors build two families of finetuning data. Three are **trait-eliciting** (malicious answers, flattering answers, fabricated answers). Five are **EM-like**, in the spirit of emergent misalignment: medical advice, code, GSM8K, math and political opinions with domain-specific flaws but no explicit trait. Each comes in three versions: Normal (clean control), I (mild trait or subtle errors) and II (overt trait or severe errors). Models are trained for one epoch with rs-LoRA (rank 32, scaling 64, learning rate ${10^{-5}}$) on a single H100.

The results confirm the worry. Training on evil data also raises sycophancy and hallucination, and training on flawed math reasoning raises evil, even though no math answer says anything evil.

**The math.** Define the **finetuning shift** as the change in the average last-prompt-token activation over the evaluation prompts $\mathcal{E}$, projected onto the persona direction:

$$
\Delta_{\text{ft}} = \left( \frac{1}{|\mathcal{E}|}\sum_{x\in\mathcal{E}} h_\ell^{\text{ft}}(x) - \frac{1}{|\mathcal{E}|}\sum_{x\in\mathcal{E}} h_\ell^{\text{base}}(x) \right) \cdot \hat v_\ell
$$

Here $h_\ell^{\text{ft}}$ and $h_\ell^{\text{base}}$ are the finetuned and base model's activations at the last prompt token. A positive value means training pushed the model's default state toward the trait.

![Figure 6 from Chen et al. (2025): each point is a finetuned model; its shift along the persona vector (x-axis) tracks its post-finetuning trait expression (y-axis)](/imgs/blogs/persona-vectors-monitoring-and-controlling-character-traits-in-language-models-fig6.webp)

Across all these finetuned models, $\Delta_{\text{ft}}$ correlates with the trait score at **r = 0.76–0.97**. Projections onto *other* traits' vectors correlate less well (r = 0.34–0.86), so the signal is mostly trait-specific.

**When it fails.** "Mostly" is doing work. The authors note that negative traits (and, oddly, humor) shift together, and opposite to optimism. Part of that is correlated data and part is correlated vectors: the persona directions are not orthogonal, so one shift shows up on several meters.

## 5. Undoing the shift: inference-time vs preventative steering

Once we can see a finetuned model has drifted along $v$, there are two places to push back.

### Inference-time steering

The obvious fix is to subtract the vector during generation, $h_\ell \leftarrow h_\ell - \alpha v_\ell$. Figure 7A shows it works: trait expression falls as $\alpha$ grows. The cost is visible in the gray MMLU line, which drops at large coefficients. You are fighting the model's learned state on every token, and collateral damage accumulates.

### Preventative steering

**The problem.** We would rather the model never learned the shift.

**The analogy.** A vaccine rather than a medicine. Or, closer to the mechanism: if a coach keeps telling a runner to lean left, and you tilt the track left during practice, the runner never needs to learn to lean. Remove the tilt on race day and the runner runs straight.

**The mechanism.** During finetuning, *add* the undesired persona vector, $h_\ell \leftarrow h_\ell + \alpha v_\ell$, in the forward pass. The trait-laden training targets are now already partly "explained" by the injected shift, so the loss exerts less gradient pressure on the weights to move activations along $v$. At deployment the steering is switched off, and the weights carry far less of the trait.

![Preventative steering supplies the persona shift during training so the weights never have to learn it](/imgs/blogs/persona-vectors-monitoring-and-controlling-character-traits-in-language-models-2.webp)

**A gradient view (my framing, not the paper's).** Write the training loss as $\mathcal{L}(\theta)$ for weights $\theta$. If the data rewards activations sitting at $c$ units along $\hat v$, regular finetuning must change $\theta$ until the model produces those $c$ units itself. With a fixed injection of $\alpha \lVert v \rVert$ units, the weights only need to supply $c - \alpha\lVert v \rVert$. A coefficient matched to the data's pull drives that residual toward zero, which is why the paper's single-layer results improve further with multi-layer steering.

![Figure 7 from Chen et al. (2025): (A) subtracting the vector after finetuning lowers trait scores but erodes MMLU at large coefficients; (B) adding it during finetuning limits trait acquisition while preserving MMLU](/imgs/blogs/persona-vectors-monitoring-and-controlling-character-traits-in-language-models-fig7.webp)

Figure 7B is the headline: preventative steering cuts trait acquisition while MMLU stays close to the regular finetuning baseline, and coherence stays above 80. For reference, the base model scores 0 (evil), 4.4 (sycophancy) and 20.1 (hallucination) before finetuning. Single-layer prevention does not fully stop trait acquisition on datasets designed to teach that trait. Steering at all layers (Appendix J.3) brings those near baseline, again without MMLU loss.

The paper compares its approach against two alternatives:

| Method | What it does in training | Result in the paper |
| --- | --- | --- |
| Preventative steering | Add $+\alpha v$ to activations | Works for all three traits; needs a tuned $\alpha$ |
| CAFT (Casademunt et al.) | Zero out the projection onto $v$ | Works for evil and sycophancy, far weaker for hallucination |
| Projection regularizer | Penalize change in projection on $v$ | Ineffective: the trait reappears through other directions |
| Inference-time steering | Subtract $\alpha v$ after training | Works, but MMLU drops at large $\alpha$ |

The CAFT result has a neat explanation. The base model's activations project *negatively* onto evil and sycophancy, so forcing the projection to zero is itself a positive push, which is accidental preventative steering. For hallucination the base projection is already near zero, so ablation does almost nothing. The regularizer failure is the more sobering finding: penalize one direction and gradient descent routes the trait around it.

**When it fails.** Prevention requires knowing the trait in advance and choosing $\alpha$. Too small leaves the shift; the paper does not give a principled way to set it.

## 6. Projection difference: screening data before you train

**The problem.** Everything so far needs a finetuned model. The cheaper question is: will this dataset corrupt my model, and which samples are to blame?

**The intuition.** A training sample teaches the model something only if it differs from what the model would say anyway. A flirtatious reply to a roleplay request may score high on the sycophancy axis, but if the model already answers that way, training on it changes little. The informative quantity is the *gap* between the data and the model's own natural answer.

**The mechanism.** For each training prompt $x_i$ with dataset response $y_i$, sample the base model's own response $y_i'$. Embed both with the response-averaged activation, project onto the unit persona direction, and average the gaps.

$$
\Delta P = \frac{1}{|\mathcal{D}|}\sum_{i} \big[\, a_\ell(x_i, y_i) - a_\ell(x_i, y_i') \,\big] \cdot \hat v_\ell
$$

Here $\mathcal{D} = \{(x_i, y_i)\}$ is the training set, $a_\ell$ is the response-averaged activation from section 1, and $\hat v_\ell$ is the unit-normalized persona vector. The per-sample term inside the sum is the sample-level score.

![Raw projection flags the sample the model already agrees with; projection difference flags the one that teaches something new](/imgs/blogs/persona-vectors-monitoring-and-controlling-character-traits-in-language-models-3.webp)

**Worked micro-example (illustrative numbers).** Three samples, as in the figure. A romantic roleplay reply projects at 6.0 and the model's own answer at 5.5, for a gap of 0.5. A confident answer to an underspecified prompt projects at 3.0, but the model would have asked for clarification (−2.0), for a gap of 5.0. A plain factual answer gives a gap of 0.2. Raw projection ranks A first; $\Delta P$ correctly ranks B first. The dataset-level $\Delta P$ is $(0.5 + 5.0 + 0.2)/3 = 1.9$.

![Figure 8 from Chen et al. (2025): dataset-level projection difference on the training data (x-axis) predicts trait expression after finetuning (y-axis)](/imgs/blogs/persona-vectors-monitoring-and-controlling-character-traits-in-language-models-fig8.webp)

Figure 8 shows dataset-level $\Delta P$ predicting post-finetuning behavior, and Appendix H confirms it beats raw projection. At the sample level, projections separate trait-inducing samples from control samples in clearly distinct histograms. That holds even for Opinion Mistake II, which contains no explicit trait at all yet induces all three when trained on.

**When it fails.** It is expensive: you must generate a base-model response for every training prompt. Appendix I offers two shortcuts. Sample a subset for a dataset-level estimate, or replace the base response's projection with the **last prompt token's** projection, which tracks it closely. The shortcut works well for evil and hallucination and is weaker but competitive for sycophancy.

## 7. Real-world data: what LMSYS-Chat-1M hides

Synthetic datasets are built to have the trait. The real test is ordinary chat logs. The authors score every sample of LMSYS-Chat-1M (one million conversations with 25 LLMs) by projection difference, then finetune separate models on the top 500, the bottom 500, and a random 500.

![Figure 10 from Chen et al. (2025): finetuning on the 500 highest projection-difference LMSYS samples raises trait expression above random, the lowest 500 lowers it, and the ordering survives LLM-judge filtering](/imgs/blogs/persona-vectors-monitoring-and-controlling-character-traits-in-language-models-fig10.webp)

The ordering holds for all three traits: high above random above low. More interesting is what the method surfaces:

- **Evil:** explicit requests for toxic content or harmful personas, as expected.
- **Sycophancy:** mostly romantic or sexual roleplay requests. Obvious in hindsight, not something most people would have listed in advance.
- **Hallucination:** underspecified queries such as "Keep writing the last story," where the logged assistant invented content and Qwen would have asked for clarification. The pattern shows up even in the heavily filtered UltraChat 200k.

Then the stress test. A GPT-4.1-mini filter first removes every sample with a trait score above 1, so nothing overtly evil, flattering or fabricated remains. High projection-difference samples *still* induce more trait expression than random ones. The "underspecified query" samples slip past the hallucination judge because it looks for fabricated facts, not for answering a question that was never really asked. This is the paper's best argument that activation-based screening catches things a text judge structurally cannot.

## Results at a glance

| Experiment | Models | Key number | Source |
| --- | --- | --- | --- |
| Judge vs humans | GPT-4.1-mini judge | 94.7% agreement (284/300) | App. B.2 |
| Prompt monitoring | Qwen2.5-7B, Llama-3.1-8B | r = 0.75–0.83 (system prompts) | Sec. 3.3 |
| Finetuning shift vs trait | Both | r = 0.76–0.97; cross-trait 0.34–0.86 | Sec. 4.2 |
| Inference-time steering | Both | Trait falls; MMLU falls at large $\alpha$; coherence above 75 | Fig. 7A |
| Preventative steering | Both | Trait limited, MMLU preserved; coherence above 80 | Fig. 7B |
| LMSYS screening | Qwen | High > random > low, also after LLM filtering | Fig. 10 |

**What is load-bearing and might not transfer.** Three things. The models are 7–8B open chat models that comply with "be evil" system prompts; frontier models with sturdier refusals may not, which breaks extraction. The trait scores come from one LLM judge on 20 single-turn questions per trait. And training is short LoRA finetuning, where a small, low-rank update plausibly moves along a few dominant directions. Whether full-parameter or RL training behaves as linearly is open.

## A peek inside: SAE decomposition

Persona vectors are deliberately coarse. Appendix M asks what they are made of by comparing the evil vector with the decoder directions of a BatchTopK sparse autoencoder trained on Qwen (131,072 features, $k = 64$). The score is the cosine similarity $s^{(i)} = \hat v_{\text{evil}} \cdot \hat f^{(i)}_{\text{dec}}$, where $\hat f^{(i)}_{\text{dec}}$ is feature $i$'s unit decoder direction. Only a few features align strongly, and they read as finer concepts: insulting language (feature 12061), deliberate cruelty (128289) and malicious code (14739). For context on how SAE features compare with simple directions as steering tools, see the AxBench analysis.

## Critique

**What is strong.**

- **One instrument, whole lifecycle.** The same vector is validated causally (steering), observationally (monitoring), on training dynamics (finetuning shift) and predictively (data screening). Each use cross-checks the others.
- **Preventative steering is genuinely new and cheap.** It costs one vector addition per forward pass, beats CAFT on hallucination, and the paper explains *why* CAFT works when it does.
- **The negative results are reported.** The within-condition correlations, the failed regularizer and the CAFT limitation all appear in appendices instead of being buried.

**What is weak.**

- **Monitoring is oversold in the main text.** The headline r = 0.75–0.83 is mostly between-prompt separation. A within-condition 0.245 for hallucination is not a deployable signal.
- **The judge defines the target.** The judge rated enthusiastic *disagreement* as sycophantic, and rated "I don't know this mission, but here is a plausible description" as hallucination. When the meter and the evaluator share an LLM-generated rubric, some correlation is baked in.
- **Coefficient tuning is unaddressed.** Both steering methods need a hand-picked $\alpha$, and the paper offers no rule for choosing it from data.
- **Trait entanglement.** Negative traits move together. A "sycophancy" alarm may partly be a general "bad persona" alarm, which is useful but less specific than the name suggests.

**Missing ablation.** I would want a direct comparison of $\Delta P$ screening against gradient- or influence-based data attribution at matched compute, since both target the same question: which samples cause which behavior?

**What would change my mind.** If preventative steering failed on a larger model trained with full-parameter finetuning or RL, rather than rank-32 LoRA for one epoch, I would read the result as a property of small low-rank updates rather than of personas. Likewise, if within-condition monitoring stayed below about 0.5 for most traits on a broader evaluation set, I would stop treating the vector as a deployment monitor and keep it only as a training-time tool.

## References

1. Chen, R., Arditi, A., Sleight, H., Evans, O., and Lindsey, J. (2025). *Persona Vectors: Monitoring and Controlling Character Traits in Language Models*. [arXiv:2507.21509](https://arxiv.org/abs/2507.21509).
2. Code: [github.com/safety-research/persona_vectors](https://github.com/safety-research/persona_vectors).
3. Betley et al. (2025). *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.*
4. Casademunt et al. (2025). Concept ablation finetuning (CAFT).
5. On this site: [The linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis), AxBench: steering baselines vs sparse autoencoders, and [What is superposition](/blog/machine-learning/ai-interpretability/what-is-superposition).

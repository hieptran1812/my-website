---
title: "Nemotron-4 340B: How NVIDIA Aligned a Frontier Model on 98% Synthetic Data"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A deep dive into the Nemotron-4 340B technical report — the synthetic-data flywheel, the multi-attribute reward model, reward-model-as-judge, and the RPO preference objective that aligned a frontier model with only ~20K human annotations."
tags: ["llm", "nemotron", "nvidia", "synthetic-data", "reward-model", "rlhf", "dpo", "rpo", "preference-optimization", "alignment", "helpsteer", "sft"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 50
---

## When the alignment data is the model's own output

The received wisdom of the RLHF era was that alignment is gated by human labeling. To make a model helpful and safe you needed armies of annotators writing demonstrations and ranking responses — InstructGPT-style, with tens or hundreds of thousands of carefully collected human preferences. Data was the moat, and the moat was expensive, slow, and hard to scale.

Nemotron-4 340B is the report that says: mostly, you don't. NVIDIA aligned a 340-billion-parameter frontier model where **over 98% of the alignment data was synthetically generated**, using roughly **20,000 human annotations total** across the entire pipeline — 10K for supervised fine-tuning and 10K (the HelpSteer2 set) for the reward model and preference tuning. The model the team built from that data wins against GPT-4-1106-preview in a non-trivial fraction of head-to-head human evaluations, scores 8.22 on MT-Bench, 92.3 on GSM8K, and ships under a genuinely permissive license alongside the very tools used to make it.

This is not a story about one clever trick. It is a *system*: a reward model good enough to judge data, a prompt-synthesis pipeline that manufactures diversity on purpose, a response-generation-and-filtering loop, and a preference-optimization method (RPO) that fixes a real weakness in DPO. The pieces interlock, and the most important property of the whole is that it **bootstraps** — a weak open model seeds the first batch of data, the model trained on that data surpasses the seed, and the better model generates better data for the next round. This post is a tour of that machine, drawn from the [Nemotron-4 340B technical report](https://arxiv.org/abs/2406.11704). It is the second in a series reading NVIDIA's model reports for their reusable techniques; the first covered [Minitron's pruning-and-distillation recipe](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation), and this one is its alignment-side counterpart.

The mismatch that the whole design resolves:

| Question | The RLHF-era assumption | What Nemotron-4 340B shows |
|---|---|---|
| What gates alignment quality? | Volume of human preference data | Quality of the *reward model* that judges synthetic data |
| How much human labeling? | 100K+ demonstrations and rankings | ~20K total annotations |
| Where do SFT prompts come from? | Collected from users / annotators | Synthesized top-down from seed topics |
| Who ranks preference pairs? | Human raters | Verifiers, then a reward model (0.87 vs a 0.54 LLM judge) |
| Can a weak teacher cap the student? | Yes — you can't exceed your labels | No — weak-to-strong iteration surpasses the seed |
| Is DPO's binary preference enough? | Mostly | No — RPO calibrates to the reward *gap* |

![Pipeline diagram of the weak-to-strong synthetic data flywheel: Mixtral-8x7B as a weak generator synthesizes prompts and responses, which align a 340B Interm-1 model; the resulting Interm-1-Instruct beats the teacher and becomes the new generator that regenerates stronger data, producing the final 340B-Instruct](/imgs/blogs/nemotron-4-340b-synthetic-data-alignment-1.webp)

The diagram above is the mental model for the whole report: alignment is a **flywheel**. A permissively-licensed weak model (Mixtral-8x7B-Instruct) generates the first synthetic dataset; that data aligns an intermediate 340B checkpoint; the aligned intermediate *surpasses the generator that trained it* and becomes the new, stronger generator; and the loop turns again with higher-quality data. The rest of this article is a walk through each component the flywheel needs to spin — the model triplet, prompt synthesis, response generation and reward filtering, the reward model itself, the judging strategy, the staged alignment recipe, and RPO.

Before diving in, it is worth naming why this report mattered beyond NVIDIA. When it landed, the prevailing anxiety about synthetic data was "model collapse" — the fear that training models on model-generated text would degrade them generation over generation. Nemotron-4 340B is the existence proof that the fear is conditional, not absolute: synthetic data degrades a model *when the loop has no external quality anchor*, and improves it *when a strong reward model governs the loop*. That distinction reframed the conversation. The question stopped being "is synthetic data safe?" and became "what is the governor on your loop?" — a much more productive question, because governors are engineerable. Every design choice in the report is, in one way or another, an answer to that question.

A framing to hold onto:

> In the RLHF era, the bottleneck was *collecting* human judgments. Nemotron-4 340B moves the bottleneck to *building a reward model good enough to manufacture* those judgments — and once you have that, data stops being scarce.

## 1. The triplet: one base model, three shipped models

The release is three models, and understanding why there are three — and that they all come from one base — is the foundation for everything else.

![Graph showing 9 trillion training tokens (70% English, 15% multilingual, 15% code) feeding a 340B-Base decoder-only model with 96 layers, d=18432, 96 heads / 8 KV heads, 256k vocab; the base branches via a reward head into 340B-Reward and via alignment into 340B-Instruct](/imgs/blogs/nemotron-4-340b-synthetic-data-alignment-2.webp)

**Nemotron-4-340B-Base** is a standard decoder-only transformer, deliberately conventional so the interesting work can live in the data and alignment rather than the architecture:

- **340B parameters** (9.4B embedding + 331.6B non-embedding), **96 transformer layers**, hidden dimension **18,432**.
- **96 attention heads** with **8 KV heads** — grouped-query attention, the same memory-saving trick discussed in the [KV cache deep dive](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management).
- **256,000-token vocabulary** (SentencePiece), **4,096 sequence length**, **RoPE** positional embeddings, **squared-ReLU** activations.
- Pretrained on **9 trillion tokens** (8T in the main run + 1T of continued pretraining on higher-quality data), with a blend of **70% English, 15% multilingual** (53 languages), **15% code** (43 languages).
- Sized so the deployed model **fits on a single DGX H100 (8 GPUs) in FP8** — a hard product constraint that shaped the parameter count.

Training ran on 768 DGX H100 nodes with 8-way tensor parallelism, 12-way pipeline parallelism, and data parallelism, reaching ~41% Model FLOP/s Utilization. None of this is exotic; the report's own framing is that the base is a solid, unsurprising foundation, and the novelty is downstream.

From that single base, two models are derived:

- **Nemotron-4-340B-Reward** replaces the final softmax (the unembedding that predicts tokens) with a **linear regression head** that outputs five scalars. It is a judge, not a generator.
- **Nemotron-4-340B-Instruct** is the base put through the staged alignment pipeline (SFT → DPO → RPO).

The reason this matters: the reward model and the instruct model **share the base's representations**, and the reward model is then used to *build the data* that trains the instruct model. The judge and the student grew from the same root, which is part of why the judge is good enough to supervise at scale. We will come back to the reward model in §4; first, the data.

### Why ship all three models, not just the Instruct

It would have been easy to ship only Nemotron-4-340B-Instruct — the chat model — and keep the base and reward models internal. Releasing all three was a deliberate decision that reflects the report's whole thesis. The **Base** is released because it is the substrate others can align or compress for their own purposes (this is exactly what the [Minitron work](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) does to NVIDIA's own bases). The **Reward** model is released because, in the synthetic-data paradigm, *the reward model is the most reusable artifact of all* — anyone can use it to filter their own synthetic data, score their own model's outputs, or judge their own preference pairs, without retraining anything. And the **Instruct** model is released both as a usable assistant and, pointedly, as a *synthetic-data generator* for other people's pipelines. The triplet is not three products; it is a self-contained kit for running the flywheel: a base to start from, a reward model to judge with, and an instruct model to generate with. That framing — shipping the *means of production* of aligned models, not just an aligned model — is the strategic core of the release.

### Why the architecture is deliberately boring

It is worth dwelling on how *conventional* the base is, because the conventionality is a choice, not a limitation. There is no Mixture-of-Experts, no novel attention variant, no exotic positional scheme — just a dense decoder with GQA, RoPE, and squared-ReLU. The report's bet is that at this scale, with a clean 9T-token corpus, the marginal return on architectural cleverness is small compared to the marginal return on *data and alignment* cleverness, which is where the team spent their innovation budget. This is a recurring theme in NVIDIA's model work and a useful corrective to architecture-chasing: a boring, well-executed transformer plus an excellent data pipeline beats a clever architecture plus a mediocre one. The squared-ReLU activation and the 256K SentencePiece vocabulary are the only mild departures from the most vanilla recipe, and both are efficiency choices rather than capability bets.

### The single-DGX-H100 constraint shaped the model

The 340B parameter count looks arbitrary until you see the deployment target: the model is sized so that, **quantized to FP8, it fits on one DGX H100 node (8× 80GB = 640GB)**. At FP8, 340B parameters consume ~340GB for weights, leaving headroom for the KV cache and activations within a single node's memory — which means inference needs no cross-node communication, the single biggest latency killer in large-model serving. This is co-design: the architecture serves the serving constraint. It also explains why this report pairs so naturally with the [Minitron compression work](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) and the [quantization deep dive](/blog/machine-learning/large-language-model/quantization-in-llm) — FP8 is not an afterthought applied at deployment, it is a premise the model was designed around. Training itself ran across 768 DGX H100 nodes with a 8×12 tensor-and-pipeline-parallel mesh on top of data parallelism, hitting ~41% MFU, but the *deployment* footprint is the one that fixed the parameter budget.

## 2. Synthesizing prompts for diversity, not just volume

The first thing people get wrong about synthetic data is thinking the hard part is *volume*. It is not — a large model will happily generate millions of prompts. The hard part is **diversity**. If you ask a model "give me a user prompt," it will regress to the mean and hand you a thousand variations of "write a poem about the ocean." A dataset with no coverage of edge cases, formats, and domains produces a model with no competence at edge cases, formats, and domains. So the pipeline generates prompts **top-down**, deliberately spreading coverage across task families before filling each one in.

![Tree diagram of synthetic prompt generation: a root of synthetic prompts from 3000 seed topics branches into single-turn, instruction-following, and two-turn families; single-turn further branches into open Q&A plus writing, closed Q&A on C4 documents, and math plus code from 12k and 17k keywords; instruction-following branches into format constraints like JSON and length](/imgs/blogs/nemotron-4-340b-synthetic-data-alignment-3.webp)

The generation starts by synthesizing a **diversity scaffold**: the pipeline first generates ~3,000 macro-topics and sub-topics, so that downstream prompts are *conditioned* on a topic rather than sampled blindly. Then prompts are generated within each family:

- **Single-turn prompts**:
  - *Open Q&A and writing* — conditioned on the topic scaffold for breadth.
  - *Closed Q&A* — grounded on real documents sampled from the C4 corpus, so the prompts reference concrete material.
  - *Math and coding* — seeded from **12,000 Python keywords and 17,000 math keywords**, so the prompts span the actual surface area of those domains rather than clustering on the few problems a model volunteers unprompted.
- **Instruction-following prompts** — single-turn prompts augmented with **explicit, checkable format constraints**: "respond in JSON," "use exactly three paragraphs," "answer in under 50 words." These are the prompts that teach the model to *follow instructions precisely*, and crucially their constraints are **verifiable**, which matters enormously for the judging step (§5).
- **Two-turn prompts** — built from ShareGPT-style first turns plus model-generated continuations, so the model learns multi-turn coherence rather than treating every message as a fresh context.

The pipeline also keeps a slice of **real-world prompts** — about 1M community conversations from LMSYS, filtered for safety — so the synthetic distribution stays anchored to how people actually talk. The seed generator for all of this is initially **Mixtral-8x7B-Instruct** (permissively licensed, so the output is clean to use), and is later replaced by the intermediate Nemotron models as the flywheel turns.

Here is the shape of topic-conditioned prompt synthesis — the conditioning on a topic is the whole trick for diversity:

```python
def synthesize_prompt(generator, topic, subtopic, task_family):
    if task_family == "open_qa":
        instruction = (
            f"Write a single realistic user question about {subtopic} "
            f"(a sub-area of {topic}). Vary the phrasing and difficulty.")
    elif task_family == "instruction_following":
        constraint = sample_format_constraint()           # JSON, length, headings...
        instruction = (
            f"Write a user request about {subtopic} that also requires: "
            f"{constraint}. The constraint must be checkable.")
    elif task_family == "math_code":
        kw = sample_keyword()                              # from 12k Python / 17k math
        instruction = f"Write a {topic} problem that uses the concept '{kw}'."
    return generator.generate(instruction, temperature=0.9)  # high temp for diversity
```

### How the diversity scaffold is built

The 3,000 topics are themselves synthetic, generated in a two-level hierarchy: the model first proposes broad macro-topics (science, law, cooking, software, …), then for each macro-topic proposes sub-topics (within software: concurrency, parsing, memory management, …). This macro/sub structure matters because it produces *stratified* coverage — you get breadth across domains and depth within each, rather than a flat list that over-samples whatever the model finds salient. Once the scaffold exists, prompt generation conditions on a (topic, subtopic) pair, so the resulting prompts inherit the scaffold's stratification. The keyword seeds for math and code work the same way at finer grain: 17,000 math keywords and 12,000 Python keywords are concept-level handles (eigenvalue, modular arithmetic, `asyncio`, generator, decorator) that force the prompts to span the *actual surface area* of those domains. A model left to generate math problems unprompted will produce a thousand variations of "solve for x"; conditioned on "modular arithmetic" or "Lagrange multipliers," it produces problems it would otherwise never volunteer. The scaffold is, in effect, a curriculum designed in advance and then filled in by generation.

This generalizes into a concrete recipe you can apply to any synthetic-data effort: (1) enumerate the *axes* of variation you care about — domain, sub-domain, format, difficulty, turn count, language; (2) generate or curate an explicit set of values along each axis; (3) condition every generated example on a sampled combination of those values; (4) verify coverage by checking that the resulting distribution actually spans the axes, not just nominally references them. The volume of data you generate is almost irrelevant next to whether it covers the axes — a stratified 100K beats an unconditioned 10M.

### Second-order optimization: diversity is a coverage problem, not a volume problem

The senior insight worth extracting: when you synthesize training data, your job is to **engineer coverage**, and coverage does not happen by default. Every place the pipeline injects external structure — 3,000 topics, 12K/17K keywords, real C4 documents, format constraints, LMSYS anchoring — is a place where it is *forcing* breadth the generator would not produce on its own. If you take one thing from this section into your own synthetic-data work, it is this: enumerate the axes of variation you care about (topic, format, difficulty, domain, turn count) and condition generation on each explicitly. A pile of unconditioned samples is a pile of near-duplicates.

## 3. Generate many responses, keep the best by reward

Prompts are half the data; the other half is responses, and the responses are where quality is won or lost. The pipeline generates **multiple candidate responses per prompt** from intermediate models, then uses the reward model to keep only the good ones.

![Graph showing a synthetic prompt fanning out to three candidate responses A, B, and C generated by intermediate models; all three feed into the 340B-Reward model which scores five attributes, and a threshold filter keeps only the high-quality responses](/imgs/blogs/nemotron-4-340b-synthetic-data-alignment-4.webp)

The mechanics:

- For each prompt, sample several responses (from the current best generator, and sometimes from a mix of intermediate models to get diversity in the candidate pool).
- Score every response with **Nemotron-4-340B-Reward**, which emits the five attribute scores.
- **Threshold-filter**: keep responses (or preference pairs) above a quality bar; discard the rest.

This filtering is what makes synthetic data trustworthy. The generator is imperfect — it produces some excellent responses and some mediocre ones — but the reward model is a far cheaper and more consistent quality gate than human review, and it scales to millions of candidates. The data that survives is, by construction, the data the reward model believes is high quality across helpfulness, correctness, and coherence.

For **code specifically**, the pipeline uses a technique called **Genetic Instruct**: an evolutionary self-instruction loop that takes seed coding problems and applies mutation and crossover operators to evolve new, harder, more varied problems, generating ~800K synthetic coding samples for the Code SFT stage. It is the same coverage philosophy as §2, applied with a genetic-algorithm twist to a domain where you can mechanically check correctness.

```python
def build_sft_examples(prompts, generators, reward_model, threshold):
    examples = []
    for prompt in prompts:
        candidates = [g.generate(prompt) for g in generators]   # many responses
        scored = [(c, reward_model.score(prompt, c)) for c in candidates]
        # keep the best candidate, and only if it clears the quality bar
        best, attrs = max(scored, key=lambda x: x[1].helpfulness)
        if attrs.helpfulness >= threshold and attrs.correctness >= threshold:
            examples.append({"prompt": prompt, "response": best})
    return examples
```

### Genetic Instruct, in detail

The Code SFT data deserves a closer look because it shows the coverage philosophy taken to its logical end. Genetic Instruct treats coding problems as a population to be *evolved*. Starting from a seed set of problems, it applies genetic operators: **mutation** (take a problem and perturb it — change the data structure, add a constraint, increase the input size) and **crossover** (combine two problems into a harder composite). Each generated problem-solution pair is then **verified** — the solution is executed against the problem's checks — so only correct, runnable examples survive into the next generation. Run this for several generations and you get ~800K coding samples that are far more varied and progressively harder than the seeds, with correctness guaranteed by execution rather than trusted from the generator. This is only possible in domains with a cheap correctness oracle (code you can run, math you can check), and where it is possible, it is the strongest synthetic-data method available because the oracle makes aggressive generation safe. The transferable principle: when you can mechanically verify outputs, you should generate aggressively and filter hard, because the verifier — not your trust in the generator — is what guarantees quality.

### Diversity in the candidate pool, not just the prompts

A subtle point about response generation: the multiple candidates per prompt are sampled not just at high temperature from one model but, deliberately, from a *mix* of intermediate models at different alignment stages. This matters because a single model at a single temperature produces candidates that are correlated — they share the same blind spots, so the "best of N" is only marginally better than any one. Drawing candidates from diverse sources widens the quality spread, which makes the reward-model filter more effective: there is a genuinely better option to select, not just N samples of the same mediocrity. The lesson echoes the prompt-diversity theme one level down — diversity has to be engineered into the *response* pool too, or your "keep the best" filter has nothing good to keep.

### Second-order optimization: the reward model is the bottleneck and the multiplier

Notice the dependency: the *quality of every piece of synthetic data is upper-bounded by the quality of the reward model that filters it*. A weak reward model passes garbage; a strong one manufactures a clean dataset from a noisy generator. This is why NVIDIA invested so heavily in the reward model and published it as a first-class artifact — in this paradigm, the reward model is not a means to an RL end, it is **the data-quality engine for the entire pipeline**. Get it right and everything downstream gets easier.

## 4. The reward model: from one scalar to five attributes

Classic reward models, descended from the InstructGPT recipe, are trained on pairwise preferences with a Bradley-Terry objective and emit a **single scalar**: this response is better than that one. Nemotron-4-340B-Reward does something more informative.

![Before-and-after comparison: on the left, a pairwise scalar reward uses a Bradley-Terry chosen-versus-rejected objective, emits one scalar score, and is opaque with no attribution; on the right, a multi-attribute regression head on the base predicts helpfulness, correctness, coherence, complexity and verbosity, scoring 92.0 on RewardBench and remaining interpretable](/imgs/blogs/nemotron-4-340b-synthetic-data-alignment-5.webp)

It is a **multi-attribute regression** model, built by training on the **HelpSteer2** dataset (10K human-annotated examples) to predict **five attributes** of any (prompt, response) pair, each on a 0–4 scale:

- **Helpfulness** — does the response actually address the request?
- **Correctness** — is it factually and logically right?
- **Coherence** — is it consistent and well-structured?
- **Complexity** — how sophisticated is the response (a proxy for depth)?
- **Verbosity** — how much detail, used to detect padding and length-gaming.

Why is regressing five attributes better than ranking one scalar? Three reasons:

1. **Nuance.** A single scalar collapses orthogonal qualities. A response can be correct but unhelpful (a right answer to the wrong question), or helpful but verbose (padding to look thorough). Five attributes keep these distinct, so the filter in §3 can require *correctness AND helpfulness* rather than a muddy weighted average.
2. **Interpretability.** When you filter data or debug the model, "rejected for low correctness" is actionable; "reward = 0.31" is not.
3. **Anti-gaming.** The verbosity attribute is an explicit defense against the classic RLHF failure where models learn that longer = higher reward and start rambling. By scoring verbosity separately you can stop rewarding length.

The verbosity attribute deserves special attention because it targets the single most pervasive reward-hacking failure in RLHF. When reward is a single scalar, models reliably discover that *longer responses score higher* — more words look more thorough, more hedging looks more careful, more bullet points look more structured — and they learn to pad. The result is the familiar over-long, over-qualified assistant style that says in three paragraphs what should take one sentence. Scoring verbosity as a *separate* attribute breaks this: the filter can require high helpfulness and correctness *while penalizing* unnecessary verbosity, so length stops being a free path to higher reward. This is reward hacking defused by decomposition — you cannot game "be helpful" by being verbose if verbosity is measured and discounted independently. The general principle is that **any quality you do not measure separately becomes a vector for gaming the qualities you do measure**, which is the strongest argument for multi-attribute rewards over scalar ones.

The result is the strongest reward model of its time: **92.0 on RewardBench** overall and, tellingly, **87.1 on the Chat-Hard category** — the hard, subtle preference cases — surpassing GPT-4o and Gemini 1.5 Pro as judges. That Chat-Hard number is the one that makes the synthetic-data flywheel viable, because it is precisely the hard cases where a weak judge would let bad data through.

The reward head is a small architectural change on top of the base:

```python
import torch.nn as nn

class MultiAttributeRewardModel(nn.Module):
    """340B base trunk + a 5-way regression head replacing the LM head."""
    ATTRS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

    def __init__(self, base_trunk, hidden_size=18432):
        super().__init__()
        self.trunk = base_trunk                       # the 340B decoder, no unembedding
        self.reward_head = nn.Linear(hidden_size, 5)  # five scalars, not a softmax

    def score(self, input_ids):
        h = self.trunk(input_ids).last_hidden_state[:, -1, :]  # last-token state
        attrs = self.reward_head(h)                            # [B, 5]
        return dict(zip(self.ATTRS, attrs.unbind(-1)))
```

The head is trained with a regression objective (MSE against HelpSteer2's 0–4 human attribute labels), not a Bradley-Terry pairwise loss — which is what makes the five outputs *calibrated scalars* you can threshold on, rather than relative rankings that only mean something pairwise.

### Bradley-Terry versus regression, formally

It is worth being precise about what changes. The classic pairwise reward model maximizes the Bradley-Terry likelihood that the chosen response outscores the rejected one,

$$P(y_c \succ y_l) = \sigma\big(r(x, y_c) - r(x, y_l)\big)$$

and the learned $r$ is only meaningful *up to comparison*: it tells you $y_c > y_l$, but the absolute value of $r(x, y_c) = 0.4$ means nothing on its own. A multi-attribute regression model instead fits each attribute to an absolute human rating,

$$\mathcal{L}_{\text{reg}} = \sum_{a \in \text{attrs}} \big(\hat{r}_a(x, y) - r_a^{\text{human}}(x, y)\big)^2$$

so $\hat{r}_{\text{correctness}} = 3.2$ is a *level*, comparable across prompts, thresholdable, and auditable. This absolute calibration is exactly what the data-filtering step needs: "keep responses with correctness ≥ 3.0" is a sentence you can write only if the reward is a calibrated level, not a pairwise margin. The cost is that you need *rated* data (HelpSteer2's 0–4 labels) rather than just *ranked* pairs — a slightly more expensive annotation, repaid many times over by what the calibrated head enables downstream.

### The HelpSteer2 annotation schema

HelpSteer2 is small — ~10K examples — but its design is why 10K is enough. Rather than asking annotators for a holistic "which is better," it asks them to rate each response on the five attributes on a 0–4 Likert scale, with detailed rubrics for each level. This decomposition does two things. First, it extracts *more signal per annotation*: a single holistic comparison is roughly one bit, while five graded attributes is far richer supervision per labeled example. Second, it makes annotation *more consistent* — raters agree more when asked concrete sub-questions ("is this factually correct, 0–4?") than vague holistic ones ("is this good?"). The combination is why a 10K set trains a RewardBench-topping model: the schema is engineered to maximize signal-per-label, the same coverage-engineering philosophy applied to annotation instead of generation.

### Second-order optimization: a judge you can read is a judge you can trust

The deeper lesson is about *legibility of supervision*. A scalar reward is a black box — when the model games it, you find out from downstream behavior, too late. Attribute-level rewards make the supervision signal **inspectable**: you can audit which attribute a data point failed, notice when verbosity is creeping up, and set per-attribute thresholds. When your reward model is going to manufacture millions of training examples unsupervised, that legibility is not a nicety, it is how you keep the pipeline from silently drifting.

## 5. Who judges the preferences?

For preference data — pairs of (chosen, rejected) responses used in DPO and RPO — the pipeline needs to decide which response is better. There is no single best judge; the right one depends on the task.

![Matrix showing preference judging strategies: rows are math/code, instruction-following, and subjective chat; columns are ground-truth verifier, LLM-as-judge, and reward-model-as-judge; ground-truth verifiers are best for math/code and instruction-following, while for subjective chat the reward-model-as-judge scores 0.87 accuracy versus the LLM judge's 0.54](/imgs/blogs/nemotron-4-340b-synthetic-data-alignment-6.webp)

The pipeline uses three judging strategies, matched to the domain:

- **Ground-truth-as-a-judge.** Where the answer is checkable, use it. Math problems from GSM8K/MATH have answer keys; instruction-following prompts have *verifiable* constraints (you can mechanically check whether the output is valid JSON or exactly three paragraphs). This is the most reliable signal there is — it is not a model's opinion, it is a fact. The earlier decision to make instruction-following constraints *checkable* (§2) pays off here: it converts a subjective "is this good?" into an objective "did it satisfy the constraint?"
- **LLM-as-judge.** Where there is no ground truth — subjective chat, open-ended writing — you can ask a strong LLM to compare two responses. This is the standard "LLM-as-judge" approach, and it is *better than nothing* but noisy and biased (position bias, verbosity bias, self-preference).
- **Reward-model-as-judge.** The pipeline's key move is to **switch from LLM-as-judge to reward-model-as-judge** partway through. On the hard subjective cases (Chat-Hard), the reward model agrees with human preference **0.87** of the time versus the LLM judge's **0.54** — a difference between a useful signal and a coin flip. Because the reward model was trained on exactly the multi-attribute structure of preference, it is a far more reliable judge of subtle quality than a general LLM asked to compare two completions.

The discipline here is to **use the strongest available judge per domain**: verifiers where you can check, the reward model where you cannot. Never use a coin-flip judge to manufacture data you will then train on — you will just teach the model the judge's noise.

```python
def judge_preference(prompt, resp_a, resp_b, domain, reward_model, verifier):
    if domain in ("math", "code", "instruction_following"):
        # Ground truth wins whenever the task is checkable.
        ok_a, ok_b = verifier(prompt, resp_a), verifier(prompt, resp_b)
        if ok_a != ok_b:
            return resp_a if ok_a else resp_b
    # Subjective: the reward model (0.87) beats an LLM judge (0.54) here.
    sa = reward_model.score(prompt, resp_a)
    sb = reward_model.score(prompt, resp_b)
    return resp_a if sa.helpfulness >= sb.helpfulness else resp_b
```

### Why LLM judges fail on hard cases

It is worth understanding *why* the LLM-as-judge sat at 0.54 on Chat-Hard, because the failure modes are general. LLM judges suffer from well-catalogued biases: **position bias** (they favor the first or second response based on order, not content), **verbosity bias** (they rate longer answers higher regardless of quality), **self-preference** (they favor responses in their own style), and **format bias** (they reward superficial structure like bullet points). On easy preference cases these biases are swamped by a real quality difference, so the judge looks fine. On *hard* cases — where two responses are close and the right answer requires noticing a subtle factual error or a missed constraint — the biases dominate and the judge degrades to a coin flip. The reward model does better precisely because it was trained on the multi-attribute structure of quality: it learned to separate correctness from verbosity from coherence, so it does not confuse "longer" with "better." The practical takeaway is to **never assume an LLM judge is reliable on your hard cases without measuring it** — the cases where you most need a good judge are exactly the cases where a naive LLM judge is worst, and the only way to know is to validate against human agreement on a held-out hard set.

## 6. Staged alignment, measured at each step

With a reward model and a synthetic-data engine in place, the actual alignment of the Instruct model proceeds in **measured stages**, each one a deliberate intervention with a benchmark attached.

![Timeline of staged alignment: Code SFT on 800k seed coding samples, then General SFT on 200k tasks reaching MT-Bench 7.99, then DPO on 160k pairs with an added SFT loss, then three reward-aware RPO iterations reaching MT-Bench 8.22, ending at 340B-Instruct with IFEval 80 and GSM8K 92.3](/imgs/blogs/nemotron-4-340b-synthetic-data-alignment-7.webp)

The stages, in order:

1. **Code SFT** (800K Genetic-Instruct samples, one epoch, constant LR 3e-7, batch 128). Counter-intuitively, alignment starts with *code*. The hypothesis: code data instills precise, structured, instruction-following behavior that transfers to general tasks. This is a separate first phase so the general phase that follows does not dilute it.
2. **General SFT** (200K diverse-task samples, three epochs, batch 128, LR searched in [1e-7, 5e-7]). The broad instruction-tuning phase — and critically it **retains ~2% code data** to prevent catastrophic forgetting of the code skills from stage 1. General SFT alone takes MT-Bench from 6.79 to **7.99** and MMLU to 78.3.
3. **DPO** (160K preference pairs). [Direct Preference Optimization](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) on high-quality chosen/rejected pairs, with a **weighted SFT loss on the chosen responses added** to prevent the model from drifting away from good behavior while it learns the preference — a common DPO stabilization trick. Tuned over LR [3e-8, 3e-7], KL coefficient [3e-4, 3e-3], SFT weight [1e-5, 1e-3].
4. **RPO ×3** (300K examples, relaxed quality filtering, three successive iterations). The new **Reward-aware Preference Optimization** method (§7), run three times, each iteration using the previous checkpoint as the reference policy. This is where the biggest instruction-following gains land: **IFEval prompt-strict accuracy jumps from ~46 to ~80**, MT-Bench reaches **8.22**, and GSM8K reaches **92.3**.

The progression is worth seeing as a table, because it shows *which stage buys what*:

| Stage | MT-Bench | MMLU | GSM8K | HumanEval | IFEval (prompt) |
|---|---|---|---|---|---|
| Code SFT | 6.79 | 72.2 | 77.6 | 70.7 | 46.4 |
| + General SFT | 7.99 | 78.3 | 87.9 | 66.5 | 61.4 |
| + DPO | 7.90 | 78.4 | 88.5 | 67.1 | 61.7 |
| + RPO ×3 (final) | **8.22** | **78.7** | **92.3** | **73.2** | **79.9** |

Two things jump out. First, **SFT does most of the knowledge work** (MMLU is basically set after General SFT). Second, **preference optimization — especially RPO — does the instruction-following and chat-quality work** (IFEval nearly doubles, MT-Bench climbs). They are doing different jobs, which is why the recipe stages them rather than blending them.

### Why stage instead of blend?

A reasonable question: why run Code SFT, General SFT, DPO, and RPO as four sequential phases rather than mixing all the data and training once? Because the stages optimize *different objectives over different data*, and blending them muddies each. SFT is maximum-likelihood on demonstrations — it teaches the model what good outputs *look like*. Preference optimization (DPO/RPO) is a contrastive objective on pairs — it teaches the model which of two outputs is *better*. These are different gradients pulling in different ways, and interleaving them means neither converges cleanly. Staging also lets you **gate progression on measurement**: you do not start DPO until General SFT has hit its MT-Bench target, so a regression in an early stage is caught before it compounds. The within-stage choices reinforce this — Code SFT runs a single epoch at a low constant LR (3e-7) to instill structure without overfitting, General SFT runs three epochs with an LR search because it is the broad-coverage workhorse, DPO adds a weighted SFT loss on chosen responses to keep the contrastive objective from dragging the model off the SFT manifold, and RPO relaxes the data-quality filter (because its reward-gap calibration is robust to slightly noisier pairs). Each stage has a hyperparameter posture matched to its job; blending would force one compromise posture for all of them.

### The Code-SFT-first ordering, in detail

The decision to lead with code deserves its own note because it is the least obvious. The intuition is that code is the most *unforgiving* instruction-following domain: a program either parses or it does not, types match or they do not, the function returns the right value or it fails the test. Training on code first instills a disposition toward precision — respect the spec, close every bracket, honor the format — that transfers to general instruction-following, where "respond in JSON with exactly three fields" is the same kind of constraint with a softer failure mode. The benchmark trace bears this out: the model enters General SFT already disciplined, and the instruction-following gains compound from there. The 2% code retention during General SFT is the insurance policy — without it, three epochs of general data would erode the code competence (catastrophic forgetting), so a small code fraction keeps the skill warm. This ordering-plus-retention pattern is a transferable recipe: front-load the domain that teaches precision, then protect it with a small retained fraction during later phases.

### Second-order optimization: measure every stage or you are flying blind

The report's discipline of attaching a benchmark to every stage is a practice worth stealing. Alignment pipelines are long and each stage can silently regress something (note HumanEval *dropping* from 70.7 to 66.5 after General SFT before recovering — exactly the kind of regression you only catch if you measure per stage). If you run SFT → DPO → RPO as one opaque block, you cannot tell which stage helped and which hurt. Instrument each one.

## 7. RPO: aligning to the reward gap, not just the sign

DPO is the workhorse of modern preference tuning, and it has a subtle flaw that RPO is designed to fix. Understanding the flaw is the key to understanding why RPO matters.

![Before-and-after comparison: on the left, DPO uses a binary preference signal where chosen beats rejected, pushes the two apart equally regardless of the gap, and suppresses good near-ties; on the right, RPO targets the reward gap eta times the difference of reward star of chosen and rejected, pushing hard on big gaps and gently on small ones, keeping good responses across three iterations](/imgs/blogs/nemotron-4-340b-synthetic-data-alignment-8.webp)

[DPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) optimizes a binary preference: given a chosen response $y_c$ and a rejected response $y_l$, it maximizes the margin between their implicit rewards,

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\Big(\beta \log \tfrac{\pi(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \tfrac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big)$$

The problem: DPO treats **every preference pair as equally decisive**. Whether $y_c$ is *vastly* better than $y_l$ or only *marginally* better, DPO pushes them apart with the same force, driving the margin as large as it can. But many of your preference pairs are *near-ties* — two genuinely good responses where one is slightly preferred. DPO will aggressively suppress the "rejected" one anyway, teaching the model that a perfectly good response is bad. Over many such pairs, this **over-suppresses good behavior** and distorts the policy. (This is the same family of pathologies that the broader [GRPO/DPO/PPO landscape](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) wrestles with.)

**Reward-aware Preference Optimization (RPO)** fixes this by making the target margin *proportional to the actual reward gap* the reward model assigns. Instead of pushing the implicit-reward difference to infinity, RPO pushes it to **match** $\eta\,(r^*(y_c) - r^*(y_l))$, the scaled reward gap:

$$\mathcal{L}_{\text{RPO}} = \mathbb{D}\!\Big[\beta \log \tfrac{\pi(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \tfrac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \;\Big\|\; \eta\,\big(r^*(y_c) - r^*(y_l)\big)\Big]$$

where $\mathbb{D}$ is a divergence (NVIDIA uses one based on the sigmoid and binary cross-entropy), $r^*$ is the reward-model score, and $\eta$ scales the gap. The behavior this produces:

- **Big reward gap → push hard.** When the reward model says $y_c$ is much better, RPO drives a large margin, like DPO.
- **Small reward gap → push gently.** When $y_c$ is only slightly better, RPO targets a *small* margin and **does not over-suppress** the good "rejected" response.

This is why RPO is run **iteratively** — three rounds, each using the previous checkpoint as the reference policy. Because RPO does not blow up the margins, it is stable enough to iterate, and each round refines the policy against a fresh reference. It is the difference between a preference method that respects how good your data actually is and one that treats every comparison as life-or-death.

```python
import torch.nn.functional as F

def rpo_loss(pol_logp_c, pol_logp_l, ref_logp_c, ref_logp_l,
             reward_c, reward_l, beta=0.1, eta=1.0):
    # Implicit reward margin under the current policy vs the reference.
    margin = beta * (pol_logp_c - ref_logp_c) - beta * (pol_logp_l - ref_logp_l)
    # RPO target: the ACTUAL reward gap from the reward model, scaled by eta.
    target = eta * (reward_c - reward_l)
    # Match the margin to the target gap (sigmoid/BCE divergence), instead of
    # DPO's "drive the margin to +infinity".
    return F.binary_cross_entropy_with_logits(margin, torch.sigmoid(target))
```

### A worked example: DPO versus RPO on a near-tie

Make it concrete. Suppose a preference pair where the reward model scores the chosen response 3.4 and the rejected response 3.1 — both genuinely good, a near-tie. Under DPO, the only thing that matters is the *sign*: chosen beats rejected, so DPO drives the implicit-reward margin $\beta\log\frac{\pi(y_c)}{\pi_{\text{ref}}(y_c)} - \beta\log\frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}$ as large as the optimizer can manage, pushing the probability of the "rejected" 3.1-quality response *down* hard — even though it is a perfectly good answer you would be happy to see in production. Repeat this over thousands of near-ties and the policy learns that lots of good responses are bad, narrowing its output distribution and hurting diversity. Under RPO, the target margin is $\eta(3.4 - 3.1) = 0.3\eta$ — a *small* target. RPO nudges the chosen response slightly above the rejected one and then *stops*, because the loss is minimized at the small target, not at infinity. The 3.1-quality response stays a plausible output. Now contrast a clear-cut pair, chosen 3.8 versus rejected 1.2: RPO's target is $2.6\eta$, a large margin, and RPO pushes hard — just like DPO would. The single objective adapts its force to the actual quality gap. That adaptivity, multiplied over a 300K-pair dataset with a realistic mix of near-ties and clear-cuts, is the difference between a policy that preserves its good behaviors and one that suppresses them.

### Why the divergence choice makes iteration stable

The specific $\mathbb{D}$ NVIDIA uses is built from the sigmoid and binary cross-entropy, which gives RPO a bounded, well-behaved gradient: the loss is minimized when the policy's implicit-reward margin *equals* the target gap, and the gradient shrinks as you approach that target rather than continuing to push. Contrast DPO, whose log-sigmoid objective has a gradient that only vanishes as the margin goes to infinity — there is no finite "correct" margin, so the model keeps inflating it. That difference is exactly why RPO can be **iterated three times** while DPO typically cannot be run repeatedly without the policy drifting and degrading. Each RPO round resets the reference policy to the previous checkpoint and re-targets the (now possibly smaller) reward gaps, so the process converges toward a policy whose implicit rewards are *calibrated to the reward model* rather than maximally separated. Iteration also lets the reward model re-score on the improved policy's distribution, closing a small loop: better policy → fresh reference → re-calibrated targets → better policy. Three iterations is where the report found diminishing returns, but the principle is that a calibrated objective is *safe to repeat* in a way a saturating one is not.

### RPO in the preference-method landscape

It helps to place RPO among its neighbors. PPO-based RLHF uses an explicit reward model and an actor-critic loop — powerful but unstable and infrastructure-heavy. [DPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) removes the separate reward model and the RL loop by deriving the policy directly from preferences — simple and stable but, as we saw, it discards the reward magnitude. [GRPO and its variants](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) keep a relative-reward signal but normalize within groups. RPO occupies a distinctive spot: it keeps DPO's simplicity (no RL loop, derive the policy from a closed-form objective) but **re-injects the reward magnitude** that DPO threw away, by using a *trained reward model's calibrated gap* as the target. It is, in a sense, "DPO that remembers how much better the chosen response was." That only works because the team had a reward model good enough to trust the magnitude — which loops back to why the reward model is the linchpin of the whole report.

### Second-order optimization: calibrated targets beat saturating targets

The general principle, beyond RPO: a preference signal carries *magnitude*, not just *sign*, and throwing away the magnitude is throwing away information. DPO's saturating objective (drive the margin as large as possible) discards the reward model's calibrated gap and replaces it with a binary "this one wins." When you have a *good* reward model — and Nemotron's is excellent — feeding its calibrated gap into the objective rather than just its argmax is strictly more information for the policy to learn from.

## 8. The flywheel, in full: weak-to-strong without a ceiling

Now we can return to the mental-model figure with the components understood, because the most important claim in the whole report is about the *loop*, not any single piece: **the teacher does not impose a ceiling on the student.**

The conventional intuition says a model trained on another model's outputs cannot exceed that other model — you are distilling, and distillation copies, it does not create. Nemotron-4 340B breaks this:

1. The **initial generator is Mixtral-8x7B-Instruct**, a far weaker, permissively-licensed model. It produces the first batch of synthetic data.
2. That data aligns an intermediate checkpoint, **340B-Interm-1-Instruct**.
3. The intermediate **surpasses Mixtral** — because the 340B base is vastly more capable, and even imperfect data is enough to *unlock* capability that is already latent in the base, rather than *install* capability that has to come from the teacher.
4. The stronger intermediate becomes the **next generator**, producing higher-quality data.
5. The loop turns again, and quality compounds.

Why does the student beat the teacher? Because **the base model's pretraining is the real source of capability**, and alignment data only needs to be good enough to *elicit and shape* it. Mixtral's data is not teaching the 340B facts or reasoning it lacks; it is demonstrating the *format* of helpful behavior, which a much stronger base can then execute far better than the demonstrator. Two forces drive each round: a better base (more latent capability) and better data (cleaner elicitation), and they compound. This is the same weak-to-strong principle that makes [distillation](/blog/machine-learning/large-language-model/distillation-in-llm) and self-improvement loops work, applied to alignment.

The reward model is what makes the flywheel safe to spin: at every round, it filters the generator's output, so even as the generator changes, the *quality bar* stays anchored to the reward model's judgment. Without that anchor, an iterative self-generation loop drifts and collapses (the failure mode behind "model collapse" warnings about training on synthetic data). The reward model is the governor on the flywheel.

### Model collapse, and why this loop avoids it

"Model collapse" is the well-documented failure where training a model repeatedly on its own outputs causes the distribution to narrow — tails vanish, diversity erodes, and after a few generations the model produces bland, repetitive, increasingly wrong text. It is the standard cautionary tale against synthetic data, and it is real. Why does the Nemotron flywheel not collapse? Three structural defenses. First, **the quality filter is external to the generator**: the reward model (trained on *human* data) decides what survives, so the surviving distribution is shaped by a fixed human-anchored standard, not by the generator's own drifting preferences. Second, **diversity is injected every round** from outside the model — the topic scaffolds, keyword seeds, C4 documents, and LMSYS real-world prompts re-introduce breadth that a pure self-generation loop would lose. Third, **verifiers provide ground truth** on the checkable fraction (math, code, format), so a hard floor of correctness is maintained regardless of what the generator believes. Collapse happens when a loop has no external anchor; this loop has three. The general lesson for anyone building a self-improvement pipeline: a flywheel is only safe if something *outside the model* — a reward model, a verifier, an injected diversity source — holds the standard. Spin a loop with no external governor and collapse is not a risk, it is the default.

### The economics of synthetic versus human data

There is a cost story underneath the 98% number worth making explicit. Human preference annotation is expensive (dollars per comparison), slow (days to weeks of turnaround), and hard to scale (you cannot 10× your annotator pool overnight). Reward-model filtering is cheap (a forward pass per candidate), fast (as fast as you can run inference), and trivially scalable (add GPUs). Once the fixed cost of building the reward model is paid — the 10K HelpSteer2 annotations plus training — the *marginal* cost of judging a new example collapses from "a human's time" to "a forward pass." That is the economic engine: you convert a small, fixed human-annotation investment into an unlimited supply of machine judgments. The break-even is reached almost immediately, because the reward model judges millions of examples that would each have cost a human comparison. This is why the paradigm is not just *possible* but *economically dominant* for any team that can build a decent reward model: synthetic-plus-filter is cheaper per high-quality example than human annotation by orders of magnitude, and it gets cheaper as inference costs fall. The human annotation does not disappear — it concentrates into the reward model, where it has the highest leverage.

## 9. Failure modes the pipeline guards against

Like the Minitron recipe, the Nemotron alignment pipeline is best understood as a list of failure modes with safeguards bolted on. Each design choice above exists because something goes wrong without it — here is the catalog, which doubles as a checklist for your own synthetic-data alignment.

- **Mode collapse to near-duplicate prompts.** Symptom: millions of synthetic prompts, narrow competence. Cause: unconditioned generation regresses to the mean. Safeguard: top-down generation conditioned on 3,000 topic scaffolds, 12K/17K keyword seeds, and format constraints (§2).
- **Garbage responses entering the training set.** Symptom: the model picks up the generator's worst habits. Cause: no quality gate on generated responses. Safeguard: multi-response generation plus reward-model threshold filtering (§3).
- **A reward model you cannot read.** Symptom: the model games the reward (e.g., verbosity creep) and you find out too late. Cause: an opaque scalar reward. Safeguard: five interpretable, calibrated attributes you can threshold and audit (§4).
- **Training on a coin-flip judge.** Symptom: preference data that is half noise. Cause: trusting an unvalidated LLM-as-judge. Safeguard: validate against human agreement, use verifiers where checkable, and use the reward model (0.87) over the LLM judge (0.54) on subjective cases (§5).
- **DPO over-suppressing good responses.** Symptom: the policy degrades on near-tie pairs. Cause: DPO's saturating, sign-only objective. Safeguard: RPO's reward-gap-calibrated target (§7).
- **Iterative self-generation collapse.** Symptom: diversity and quality erode over rounds. Cause: a loop with no external anchor. Safeguard: a human-trained reward model as governor, injected diversity, and verifiers as a correctness floor (§8).
- **Catastrophic forgetting across stages.** Symptom: code skill vanishes after general SFT. Cause: three epochs of general data overwriting the code phase. Safeguard: retain ~2% code data in General SFT (§6).

The meta-lesson is identical to the compression story: none of these are exotic, and every one is a place where the obvious shortcut (generate blindly, skip the filter, trust the LLM judge, run DPO once and call it done) quietly degrades the result. The recipe is a disciplined list of "don't"s.

## 10. Case studies from the report

### 1. The 98% number, and what the 2% is

The headline — over 98% of alignment data is synthetic — is precise, and the 2% is as interesting as the 98%. The entire human contribution is ~20K annotations: ~10K for the SFT mix and the 10K HelpSteer2 examples that train the reward model and seed preference tuning. Everything else — millions of prompts, responses, and preference pairs — is machine-made. The lesson is *where* the human effort went: not into volume, but into the **two highest-leverage places** — a small, high-quality SFT seed and the reward-model training set. Human judgment was spent building the *judge*, and the judge scaled the rest. If you have a limited annotation budget, this is the allocation to copy: pour it into the reward model, not into raw demonstrations.

### 2. HelpSteer2 and the RewardBench crown

Nemotron-4-340B-Reward topped RewardBench at 92.0 overall and 87.1 on Chat-Hard, beating GPT-4o and Gemini 1.5 Pro as judges — and it did so trained on only **10K** human preference examples (HelpSteer2). The efficiency comes from the multi-attribute regression formulation: by asking annotators to rate five concrete attributes on a 0–4 scale rather than make holistic pairwise choices, each annotation carries far more structured signal, and the model learns a richer notion of quality from fewer labels. The case study is a refutation of "you need hundreds of thousands of preferences" — you need a well-designed annotation schema and a strong base, and 10K goes a long way.

### 3. The judge swap that unlocked the data

Midway through building preference data, the team measured their LLM-as-judge against human agreement on hard chat cases and found it at **0.54 — barely better than chance**. Switching to reward-model-as-judge took agreement to **0.87**. This single measurement-and-swap is arguably the pivotal engineering decision in the report, because preference data judged at 0.54 accuracy is poison — you would be training the model on noise half the time. The lesson is to **validate your judge against human agreement before you trust it to label data at scale**, and to treat the reward model as a judge in its own right, not just an RL reward.

### 4. Genetic Instruct: evolving code problems

The Code SFT stage used 800K samples generated by **Genetic Instruct**, an evolutionary loop that mutates and recombines seed coding problems to evolve new ones — harder, more varied, covering more of the language surface. It is a domain-specific answer to the coverage problem of §2: in code, you can mechanically verify correctness, so you can afford an aggressive generate-mutate-filter loop that would be risky in domains without verifiers. The broader lesson: where you have a cheap correctness oracle, lean into synthetic generation hard, because the oracle keeps the evolved data honest.

### 5. Why alignment starts with code

A surprising recipe choice: the *first* SFT stage is code, before general instruction tuning. The hypothesis the report tests is that code data teaches **precise, structured, constraint-following behavior** — close brackets, respect types, follow the spec — that transfers to general instruction-following. The benchmark progression supports it: the model enters General SFT already disciplined. The lesson for anyone designing an SFT curriculum is that **stage ordering matters**, and front-loading a domain that instills precision can lift everything downstream. (The 2% code retention in General SFT then protects that investment from catastrophic forgetting.)

### 6. RPO's three iterations and the IFEval jump

The clearest demonstration that RPO earns its complexity is the IFEval numbers: instruction-following prompt-strict accuracy climbs from ~61 after DPO to **~80** after three RPO iterations, and instruction-strict to ~86. DPO alone barely moved IFEval (61.4 → 61.7). The reason RPO can iterate three times without collapsing is exactly its reward-gap calibration — because it does not over-suppress good responses, the policy stays well-behaved across rounds, and each round using the previous as reference compounds the gains. A method that blew up margins (like aggressive DPO) could not survive three iterations.

### 7. The permissive license as a strategy

Nemotron-4 340B shipped under the NVIDIA Open Model License — permitting commercial use, modification, and redistribution — *and* the team open-sourced the synthetic data generation pipeline. This is not incidental; it is the point. The model's primary advertised use case is **generating synthetic data to train other models**, which requires a license clean enough that the generated data is yours to use commercially. The whole release is designed to seed *other people's* flywheels. The strategic lesson: in the synthetic-data era, a permissively-licensed strong model is a data-generation factory, and the license is as much a part of the product as the weights.

### 8. Fitting 340B on one DGX H100

A quiet but consequential constraint: the model was sized to deploy on a **single DGX H100 (8 GPUs) in FP8**. The 340B parameter count was not chosen for a round number — it was chosen as the largest model that fits the target hardware in FP8, which is why the report pairs naturally with the [Minitron compression work](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) and the broader [quantization](/blog/machine-learning/large-language-model/quantization-in-llm) story. The lesson is that frontier-model design is co-designed with deployment hardware from the start; the architecture serves the serving constraint, not the other way around.

### 9. The continued-pretraining phase that sharpened the base

A detail easy to skip: the 9T-token budget is 8T of main pretraining plus **1T of continued pretraining** on a higher-quality data blend. This final 1T is a deliberate "annealing" phase that shifts the data distribution toward cleaner, more useful tokens right before the model is frozen as a base. The effect mirrors the Minitron best practice of compressing from the *final* training-stage checkpoint: the last phase shapes the most valuable behaviors, and you want your downstream work — alignment here, compression there — to start from the model that has them. The transferable insight is that the *order* of data matters, not just the total: ending pretraining on your best data leaves the base in a better starting position for everything that follows, at a marginal cost (1T of 9T) that is cheap relative to the benefit.

### 10. The multilingual and code blend as a capability bet

The pretraining mix — 70% English, 15% multilingual across 53 languages, 15% code across 43 languages — is a capability bet placed at pretraining time that pays off through the whole pipeline. The 15% code is what makes the Code-SFT-first strategy (§6) possible: a base with weak code competence could not be sharpened into a precise instruction-follower by code SFT alone. The 15% multilingual is what lets the synthetic-data pipeline generate and align in many languages without a separate multilingual training program. The lesson is that **alignment cannot add a capability the base lacks** — it can only elicit and shape what pretraining laid down — so the pretraining blend is, in effect, a pre-commitment about which capabilities the aligned model will be *able* to have. Decide your capability portfolio at pretraining, because alignment is downstream of it.

### 11. RPO versus a PPO-based RLHF loop

It is instructive to contrast what NVIDIA *did not* do. The InstructGPT-lineage approach would be a PPO-based RLHF loop: train a reward model, then run on-policy RL with an actor and a critic, sampling from the policy, scoring with the reward model, and updating with a clipped policy-gradient. That works, but it is infrastructure-heavy (you maintain a sampling loop, a critic, and careful KL control) and notoriously unstable at scale. RPO gets much of the benefit — a reward-model-informed policy update — with none of the RL machinery, because it is a closed-form offline objective on pre-collected pairs, like DPO. The case study is a reminder that "use the reward model's signal" does not have to mean "run PPO"; RPO is evidence that you can fold calibrated reward information into a stable offline objective, which is far easier to operate at 340B scale. For teams without a battle-tested RL stack, this is the pragmatic path.

### 12. LMSYS anchoring against synthetic drift

Amid millions of synthetic prompts, the pipeline deliberately retains ~1M real-world conversations from LMSYS (safety-filtered). This is a small fraction by volume but a large one by importance: it keeps the synthetic prompt distribution *anchored to how people actually talk*, which synthetic generation drifts away from (synthetic prompts trend cleaner, more formal, and more "benchmark-shaped" than real user messages). The real prompts re-introduce the messiness — typos, ambiguity, context-switching, weird requests — that the model must handle in deployment. The lesson generalizes: when you synthesize a training distribution, keep a real-data anchor in the mix to stop the synthetic distribution from drifting into an unrealistically tidy corner of input space. A model trained only on clean synthetic prompts is brittle on the messy real ones.

## The bigger picture: alignment becomes an engineering discipline

Step back from the components and the report makes a larger claim about where alignment is going. In the InstructGPT era, alignment was a *data-collection* problem — the quality of your model was bounded by the quality and quantity of human labels you could afford, and improving the model meant collecting more labels. Nemotron-4 340B reframes alignment as an *engineering* problem: the quality of your model is bounded by the quality of your **reward model** and your **data pipeline**, both of which you can improve with engineering rather than annotation budget. This is a profound shift in where the leverage lives.

Concretely, it relocates human effort to its highest-value use. The ~20K human annotations did not go into demonstrations that the model would imitate; they went into the reward model's training set and a small SFT seed — the two artifacts that then *manufacture* millions of training examples. Human judgment was spent building the *measurement instrument*, and the instrument scaled the rest. This is the same move that made software testing scalable: you do not manually check every output, you build an oracle that checks them for you. The reward model is an oracle for response quality, and once you have a good oracle, generating and filtering data is cheap.

The flywheel property is what makes this compound. Because each round's aligned model becomes the next round's generator, and because the reward model anchors quality, the system improves *itself* — better base and better data reinforcing each other — without proportionally more human input. That is the dream of self-improvement, made safe by the external governor of a human-anchored reward model. It is not unbounded (the base model's latent capability is still the ceiling, and the reward model's quality still gates everything), but within those bounds the loop turns on its own.

For practitioners, the lesson is to invest where the leverage actually is. If you are aligning a model and your instinct is to collect more preference pairs, stop and ask whether your annotation budget would do more good *building a better reward model* — because a reward model that judges at 0.87 instead of 0.54 does not improve one dataset, it improves *every* dataset you will ever filter with it. The reward model is the multiplier; the demonstrations are linear. Spend on the multiplier.

### 13. The head-to-head against GPT-4

The human-evaluation result is the proof that 98% synthetic data does not mean 98% as good. Across 136 prompts spanning 10 task categories, human raters preferred Nemotron-4-340B-Instruct over GPT-4-1106-preview **28.19%** of the time, tied **46.57%**, and preferred GPT-4 **25.24%** — meaning the synthetically-aligned model was *at least as good as GPT-4* on roughly three-quarters of the comparisons (wins plus ties), and actually won slightly more often than it lost. For a model aligned almost entirely on its own machine-generated data, going toe-to-toe with the leading proprietary model of the moment is the headline that validated the whole approach. The lesson is not that synthetic data is "good enough as a compromise" — it is that, with a strong reward model anchoring quality, synthetic-data alignment is *competitive with the best human-data alignment in the world*. The compromise framing is wrong; this is a different and arguably better way to align, not a budget substitute.

### 14. The HumanEval dip and the value of measuring

A small but instructive detail in the stage table: HumanEval *drops* from 70.7 after Code SFT to 66.5 after General SFT, then recovers to 73.2 by the end. This transient regression is exactly the kind of thing that staged measurement catches and blended training hides. General SFT's broad data slightly eroded the sharp code competence from the code-first phase — the 2% code retention softened but did not fully prevent it — and only the later stages recovered and surpassed it. If the team had run SFT as one opaque blend, they would have seen the final 73.2 and never known that a regression-and-recovery happened in the middle, nor that the code-first phase was carrying the model to a higher starting point. The case study is a concrete argument for the "measure every stage" discipline: the interesting failures live *between* stages, invisible to anyone who only looks at the endpoints.

## When to reach for synthetic-data alignment — and when not to

**Reach for it when:**

- **You can build or access a strong reward model.** The reward model is the linchpin; with a good one, synthetic data is cheap and clean. Without one, you have no quality gate.
- **You have a strong base model.** Weak-to-strong only works because the base has latent capability to elicit. Synthetic alignment data shapes behavior; it does not install knowledge a weak base lacks.
- **Your tasks are partly verifiable.** Math, code, and format-constrained instructions give you ground-truth judges that are better than any model. Lean into those.
- **Human annotation budget is scarce.** Spend it on the reward model's training set and a small SFT seed, then let the pipeline scale.
- **You need diversity you can engineer.** Topic scaffolds, keyword seeds, and format constraints let you manufacture coverage deliberately.

**Skip it (or be careful) when:**

- **You have no reliable judge.** An unvalidated LLM-as-judge at 0.54 agreement will poison your data; do not build a pipeline on a coin flip.
- **The capability is genuinely missing from the base.** Synthetic data elicits; it does not create. If the base cannot do the task at all, no amount of self-generated data will conjure the ability — that is a pretraining or [continued-pretraining](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques) problem.
- **You cannot anchor the loop.** An iterative self-generation loop without a quality governor (a reward model, verifiers) drifts toward model collapse. Never spin the flywheel without the reward model holding the bar.
- **Your domain has no diversity scaffold.** If you cannot enumerate the axes of variation, unconditioned synthetic generation will give you near-duplicates and a narrow model.

The one-sentence version:

> Stop trying to *collect* alignment data and start trying to *build the judge* that lets you manufacture it — once your reward model is trustworthy, a strong base model will align itself on its own output, and even surpass the model that seeded the loop.

That inversion — from data-collection to judge-building — is the single most portable idea in the report. It applies whether you are aligning a 340B frontier model or a 3B in-house assistant: the reward model is the lever, the data pipeline is the multiplier, and the human annotation you can afford is best spent making the judge trustworthy rather than making the demonstrations numerous.

## Further reading

- [Nemotron-4 340B Technical Report](https://arxiv.org/abs/2406.11704) — the full report, with the RPO derivation, the data pipeline, and all benchmark tables.
- [Minitron: pruning and distillation](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) — the compression-side sibling in this NVIDIA series.
- [Fine-tuning LLMs with DPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) — the preference method RPO improves on.
- [GRPO vs DPO vs PPO decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) — where RPO sits in the preference-optimization landscape.
- [Knowledge distillation in LLMs](/blog/machine-learning/large-language-model/distillation-in-llm) — the weak-to-strong principle behind the flywheel.
- [Effective LLM fine-tuning techniques](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques) — the broader SFT and alignment toolkit.

*Next in the series: Llama-Nemotron and the efficient-reasoning recipe — RLVR, the reasoning on/off toggle, and the Puzzle neural-architecture-search that makes reasoning models fast.*

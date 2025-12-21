---
title: "One Token to Fool LLM-as-a-Judge (The idea)"
publishDate: "2025-10-24"
category: "paper-reading"
subcategory: "AI Interpretability"
tags: ["model-interpretation", "model-alignment"]
date: "2025-10-24"
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/one-token-to-fool-llm-as-a-judge-20251024114249.png"
excerpt: "The researchers discovered that LLM judges can be fooled by superficial cues, such as reasoning openers (“Solution:”, “Let’s think step by step”) or even non-word symbols (like “:”)..."
---

![](/imgs/blogs/one-token-to-fool-llm-as-a-judge-20251027100808.png)

## Problem overview

The paper investigates a critical flaw in generative reward models (GRMs) used in reinforcement learning with verifiable rewards (RLVR).

These GRMs, where large language models (LLMs) act as “judges” assigning rewards to policy outputs, are highly vulnerable to reward hacking.

The researchers discovered that LLM judges can be fooled by superficial cues, such as reasoning openers (“Solution:”, “Let’s think step by step”) or even non-word symbols (like “:”), assigning false positive rewards even when no real reasoning is present.

![](/imgs/blogs/one-token-to-fool-llm-as-a-judge-20251027100905.png)

This issue is systemic, appearing across models (including GPT-4, Claude-4, etc.), datasets, and prompt formats.
It challenges the reliability of current evaluation practices, as these models are often used as “gold-standard” evaluators in AI training and benchmarking.

## Key contributions

1. Identify a critical vulnerability in LLM judges: susceptibility to “master keys,” meaning superficial tokens or reasoning phrases that trigger undeserved positive rewards.

2. Conduct a systematic evaluation showing that this vulnerability is pervasive across a wide range of open-source and proprietary models and benchmarks.

3. Propose an effective mitigation strategy using data augmentation. By truncating model-generated text to early segments and fine-tuning on these, they build Master Reward Models (Master-RMs) that are significantly more robust to “master key” attacks while maintaining strong evaluation performance.

4. Provide a comprehensive analysis of the phenomenon, studying its relationship with model size, automated attack discovery, and the ineffectiveness of common inference-time fixes such as chain-of-thought or majority voting.

## Methodology

### Verifiable Reward Modeling in RLVR

In RLVR, the goal is to train an AI system (called a policy model) to generate good responses by providing it with feedback or rewards. The reward signal comes either from a rule-based function or from a generative LLM-based judge, in other words, a large language model that evaluates answers.

At each training step:

- The model receives a question (q).
- The policy model generates a response (o).
- There is also a reference or correct answer (a\*).
- The LLM judge then compares the model’s response to the reference answer and produces a binary decision (YES or NO), indicating whether the response is correct or not.

Formally, the LLM judge defines a function:

J(q, a\*, o) → {YES, NO}

This decision is turned into a numerical reward:

- A positive reward (R = 1) if the judge says “YES”
- A zero reward (R = 0) if the judge says “NO”

These rewards guide the policy model’s learning process. Therefore, the accuracy and reliability of the LLM judge are crucial — if the judge makes systematic errors or assigns false positives, the entire training can go in the wrong direction.

### Master keys

The authors identify a special kind of vulnerability in LLM judges: certain adversarial patterns that can trick them into giving positive rewards even when the content is meaningless. These patterns are called “master keys.”

Master keys come in two main forms:

1. **Non-word symbols** – Simple punctuation marks or special characters such as “.”, “:”, or “;”. Even these symbols alone can sometimes trigger a false positive reward.
2. **Reasoning openers** – Phrases that sound like the beginning of a logical explanation but contain no real reasoning, such as “Thought process:”, “Solution:”, or “Let’s solve this problem step by step.”

Although these expressions have no meaningful contribution to problem-solving, many LLM judges still mark them as correct. This behavior is consistent across multiple datasets and various LLMs, including GPT-4o, Claude-4, and Qwen2.5 models.

### Implications

The authors show that this problem persists even when using different evaluation prompts or specialized reward models (like Qwen2.5-7B-Instruct-RLVR or Omni-Judge).

This means the issue is systemic, not limited to a single model or setup.

It exposes a critical vulnerability in the very mechanism of reward modeling. Systems designed to verify correctness can be manipulated by trivial and superficial patterns. As a result, they produce false positive judgments, which mislead the learning process and threaten the reliability of reinforcement learning frameworks that rely on generative judges.

## Experiments

### Experimental Setup

To thoroughly evaluate how vulnerable LLM-based reward models (RMs) are to “master key” attacks, the authors tested a wide range of models, datasets, and adversarial input patterns.

**LLM Judges**

The tested models are divided into two categories:

1. **Specialized Generative Reward Models (RMs)** – These are LLMs fine-tuned specifically for reward modeling within the RLVR framework.

   - The authors’ own _Master-RMs_ are included here, designed to be robust against reward hacking and to maintain near-zero false positive rates.
   - Other models in this category include _Multi-sub RM_, _General-Verifier_, and _Omni-Judge_.

2. **General-Purpose LLMs** – These are advanced open or commercial models not specifically fine-tuned for reward modeling, such as _Qwen2.5-72B-Instruct_, _LLaMA3-70B/8B-Instruct_, _GPT-4o_, _GPT-o1_, and _Claude-4_.

**Benchmarks**

The evaluation covers five major reasoning benchmarks to test robustness across both verbal and symbolic reasoning tasks:

- _Multi-subject RLVR_ and _NaturalReasoning_ (for general reasoning and commonsense tasks)
- _GSM8K_ (grade-school math problems)
- _MATH_ (high-school symbolic reasoning)
- _AIME 1983–2024_ (advanced Olympiad-level problems)

These benchmarks help assess how different models handle hacking attempts across diverse reasoning domains.

**Master Keys**

The authors used minimal “master key” patterns that contain no meaningful answers but often trigger positive rewards. These include:

- _Non-word symbols_: “ ” (single blank space), “.”, “,”, “:”, and “;”.
- _Reasoning openers_: “Thought process:”, “Solution”, “Let’s solve this problem step by step.”
  They also tested multilingual equivalents like “解” (Chinese), “かいせつ” (Japanese), and “Respuesta” (Spanish).

**Prompts**

All general-purpose models were evaluated with a standardized prompt template for fairness, while specialized reward models were tested using their own default prompts.

### The Master-RMs: Robust Reward Models

To address the vulnerability caused by “master key” attacks, the authors introduce Master Reward Models (Master-RMs) — new reward models specifically designed to resist such hacks while preserving strong evaluation abilities.

Their approach builds on the RLVR framework from Su et al. (2025), which trained models using a dataset of 160,000 examples (each containing a question, reference answer, model-generated response, and a correctness label). In this setup, a large model like Qwen2.5-72B-Instruct acts as a teacher judge to label responses as correct (YES) or incorrect (NO).

While existing fine-tuned models (like Multi-sub RM) reduce vulnerability compared to general-purpose LLMs, they still exhibit over 10% false positive rates on phrases such as “Thought process:”.

![](/imgs/blogs/one-token-to-fool-llm-as-a-judge-20251027101655.png)

To further improve robustness, the authors introduce an auxiliary adversarial-style training step:

- They randomly sample 20,000 examples from the original training data and regenerate model responses using chain-of-thought prompting.

- From each generated response, they keep only the first sentence, which typically contains a reasoning opener and little real content (e.g., “To solve the problem, we need to…”).

- These truncated examples are labeled as NO, representing invalid or meaningless responses.

They then merge these 20,000 “negative” samples with the original 160,000 examples, forming an augmented dataset of 180,000 instances that includes both valid answers and typical “distraction” patterns.

Using this augmented dataset, the authors perform supervised fine-tuning on two model versions:

- Master-RM-7B (based on Qwen2.5-7B-Instruct)

- Master-RM-32B (based on Qwen2.5-32B-Instruct)

The models are trained with a standard cross-entropy loss over YES/NO labels, ensuring the model learns to correctly reject invalid reasoning patterns.

The experiments show that Master-RMs generalize extremely well.

Even though they were trained with a small fraction of adversarial data, they achieve near-zero (if not zero) false positive rates against all “master key” attacks across five benchmarks.

This demonstrates that targeted data augmentation can greatly enhance reward model robustness and generalization, even without large-scale retraining.

The experimental results show that the proposed Master-RMs generalize very effectively.

Even though they were trained on a small portion of adversarial (negative) examples, they achieved near-zero false positive rates across all tested “master key” attacks and benchmarks.

The authors also note that, beyond reasoning openers, other cues such as reflection, self-verification, or backtracking phrases could also cause similar issues —> suggesting future research should explore these broader reasoning patterns in generative reward models.

### A Comprehensive Evaluation of LLM Judges

#### Vulnerabilities to Master Key Attacks

Results show that general-purpose LLMs, including GPT-4o, Claude-4, and GPT-o1, are highly vulnerable to “master key” triggers.

- Minimal or superficial responses like punctuation-only (“:”) can cause up to **35% false positives** in GPT-4o.
- Reasoning openers such as “Thought process:” can trigger **60–90% false positives** in models like LLaMA3-70B and Qwen2.5-72B.
- Multilingual symbols (e.g., “解”) also frequently cause errors due to their benign appearance in text.

Even specialized RMs such as _General Verifier_ still show weaknesses — for example, a 66.8% false positive rate on the MATH dataset with a single blank space.

In contrast, Master-RMs remain nearly immune (near 0% false positives) across all benchmarks, confirming their strong robustness.

These results highlight how pervasive the hacking problem is, even among leading commercial models.

#### Measuring Consistencies and Alignments with Gold Standards

To evaluate reliability, the authors measured agreement between Master-RMs, GPT-4o, and human judgments using Cohen’s kappa coefficient.

Two analyses were conducted:

1. **LLM-to-GPT-4o** consistency on 2,500 mixed reasoning samples.
2. **LLM-to-human** consistency on a smaller, manually judged set of 500 samples.

![](/imgs/blogs/one-token-to-fool-llm-as-a-judge-20251027110047.png)

Results show that Master-RMs achieved 100% parsing success and excellent consistency. Master-RM-7B scored 0.91 Cohen’s kappa with GPT-4o and 0.90 with human judges, tying with Multi-sub RM and surpassing larger models like Qwen2.5-72B-Instruct. This strong alignment, combined with immunity to “master key” attacks, establishes Master-RMs as reliable and trustworthy reward models.

#### Evaluating Capabilities on Verifiable Benchmarks

The authors tested models on VerifyBench and VerifyBench-Hard, two public benchmarks designed to evaluate reference-based reward systems.

![](/imgs/blogs/one-token-to-fool-llm-as-a-judge-20251027111219.png)

These datasets assess performance across four categories: Numeric, Expression, Multiple-choice, and String, plus an overall average score.

![](/imgs/blogs/one-token-to-fool-llm-as-a-judge-20251027110530.png)

Findings show that Master-RMs outperform rule-based baselines and perform on par or better than most open-source and commercial LLMs:

- The performance gap with the top model (GPT-o1) is minimal — only 0.55% on VerifyBench and 2.0% on VerifyBench-Hard.
- Despite being smaller and more efficient, Master-RM-7B and 32B deliver competitive or superior results, making them highly efficient and effective evaluators.

#### Additional Experimental Results

The appendices provide further insights:

- Larger models often have higher false positive rates, indicating size does not guarantee robustness.
- Similar-sounding sentences can still trigger false positives, even in advanced models.
- Inference-time methods like chain-of-thought or majority voting do not reliably reduce false positives and can even worsen them.
- Removing the question from the prompt reduces false positives, especially in large models.

These analyses underline that structural vulnerabilities persist and that targeted training strategies, such as those used for Master-RMs, are essential for building more robust LLM evaluators.

## References

1. [One Token to Fool LLM-as-a-Judge](https://arxiv.org/pdf/2507.08794)

---
title: "Safety Alignment Should Be Made More Than Just a Few Tokens Deep"
publishDate: "2025-09-24"
category: "paper-reading"
subcategory: "AI Interpretability"
tags: ["model-interpretation", "model-alignment"]
date: "2025-09-24"
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/safety-alignment-should-be-made-more-than-just-a-few-tokens-deep-20250924111337.png"
excerpt: "Current safety alignment is often shallow—it mainly affects the model’s behavior in the first few tokens of its output...."
---

# Motivation

The safety of Large Language Models (LLMs) largely depends on alignment methods such as supervised fine-tuning, reinforcement learning with human feedback (RLHF), and direct preference optimization (DPO). These methods are designed to prevent models from producing harmful outputs by refusing unsafe inputs. However, recent research shows that even well-aligned models remain vulnerable to adversarial prompts, fine-tuning attacks, and exploitation of decoding parameters. This indicates that current alignment is fragile.

The authors identify a key underlying problem: Current safety alignment is often shallow—it mainly affects the model’s behavior in the first few tokens of its output. Once an unsafe trajectory begins (e.g., starting with a harmful or affirmative response), the model is likely to continue generating unsafe content. This shallow alignment makes models particularly vulnerable to simple exploits, such as adversarial suffixes or minor fine-tuning steps, which can override safety training.

Given the central role of alignment in ensuring LLM safety and the growing deployment of these models, it is critical to understand why alignment is so fragile and to develop more robust and deeper alignment strategies. The paper aims to systematically characterize this shallow alignment problem and propose approaches to strengthen LLM robustness against common jailbreaks and fine-tuning exploits.

# The shallow safety alignment issue

## Preliminaries

**Notation**

- The paper uses $\pi_\theta$ to denote an LLM parameterized by weights $\theta$.
- $\pi_{base}$: an unaligned pre-trained model (e.g., Llama-2-7B, Gemma-7B).
- $\pi_{aligned}$: its aligned counterpart (e.g., Llama-2-7B-Chat, Gemma-7B-IT).
- Given an input $x$, the output distribution is $\pi_\theta(\cdot|x)$, and sampling yields $y \sim \pi_\theta(\cdot|x)$.
- $x_t, y_t$ denote the $t$-th tokens of sequences $x, y$.
- $y_{<t}, y_{\leq t}$: subsequences up to the $(t-1)$-th or $t$-th token.
- $y_{>t}, y_{\geq t}$: subsequences after the $t$-th or $(t-1)$-th token.

**Safety Evaluation and Metrics**

- The authors evaluate safety alignment using the **HEx-PHI safety benchmark** (Qi et al., 2023b), which has 330 harmful instructions across 11 harmful use cases.
- The evaluation checks whether models comply with or resist harmful instructions.
- **GPT-4** serves as an automatic judge to determine whether outputs are safe.
- Metrics:

  - **Harmfulness Rate**: fraction of test cases producing harmful outputs without adversarial attack.
  - **Attack Success Rate (ASR)**: fraction of test cases where adversarial attacks succeed in inducing harmful outputs

## The characteristics of shallow safety alignment

A key property of safety-aligned LLMs is their ability to refuse harmful instructions. Typically, they generate refusal prefixes like “I cannot…”, “I apologize…”, or “I am unable…”. For example, Llama-2-7B-Chat uses such refusal tokens in 96.1% of harmful test cases, while Gemma-7B does so in 96.7%. Although these phrases may seem like trivial artifacts, they actually play a critical role in enabling shallow alignment.

![](/imgs/blogs/safety-alignment-should-be-made-more-than-just-a-few-tokens-deep-20250924162137.png)

Interestingly, even unaligned base models can appear safe when forced to begin their outputs with refusal prefixes. Experimental results show that pre-filling the decoding process with tokens like “I cannot” or “I apologize” drastically lowers the harmfulness rate of unaligned models. This phenomenon highlights what the authors call a “safety shortcut,” since the model’s alignment can be simulated simply by biasing the very first few tokens, regardless of what follows afterward.

![](/imgs/blogs/safety-alignment-should-be-made-more-than-just-a-few-tokens-deep-20250924162050.png)

The problem, however, is that current safety alignment strategies heavily exploit this shortcut. By emphasizing the generation of refusal tokens at the beginning of responses, alignment efforts concentrate almost entirely on the first few tokens. Figure 1 in the paper illustrates this clearly: the KL divergence between aligned and unaligned models is significantly higher at the beginning of harmful outputs but quickly decreases in later tokens. This shows that the alignment signal is disproportionately invested in the initial part of the response, leaving the rest of the generation less constrained and therefore vulnerable.

The underlying reason for this shallowness lies in the training process. During supervised fine-tuning (SFT), models are trained to imitate human responses, which often start with a refusal when addressing harmful prompts. Reinforcement learning with human feedback (RLHF) further reinforces this pattern, rewarding models for beginning with safe refusals. However, because humans rarely provide examples of refusals after harmful prefixes, the model primarily learns to refuse only at the start of its outputs. As a result, aligned models become heavily biased toward producing refusal prefixes at the beginning but struggle to maintain robust safety throughout the rest of the response.

## Shallow safety alignment and its vulnerabilities

### Inference-stage vulnerabilities

![](/imgs/blogs/safety-alignment-should-be-made-more-than-just-a-few-tokens-deep-20250924164728.png)

Shallow safety alignment mainly affects the very first few output tokens of a model, leaving later tokens largely unchanged compared to unaligned models. This creates a serious weakness: as long as attackers bypass the initial refusal prefixes, the model can still be induced to generate harmful outputs. This vulnerability enables several types of inference-stage exploits.

**1. Prefilling Attacks**

Attackers can prefill the model’s first few tokens with a non-refusal prefix, effectively steering the model into producing harmful responses. Experiments show that as more harmful tokens are prefilled, the Attack Success Rate (ASR) rises sharply, surpassing 50%. Recent work formalizes this as prefilling attacks, highlighting how aligned models quickly fail once their refusal prefixes are skipped.

**2. Optimization-Based Jailbreak Attacks**

Another class of attacks exploits shallow alignment by adversarially appending optimized suffixes to harmful prompts. These suffixes maximize the likelihood of an affirmative prefix like “Sure, here is…”, forcing the model to comply with the harmful instruction. Known adversarial suffix attacks fall into this category, as they directly exploit the over-reliance on shallow refusal prefixes.

**3. Jailbreak via Random Sampling**

Even without optimization, harmful responses can be obtained by randomly sampling outputs under different decoding parameters (e.g., temperature, top-k, top-p). Since shallow alignment only blocks harmful responses through initial refusal prefixes, varying sampling can easily bypass them, making random jailbreaks surprisingly effective.

**Remark**
The authors emphasize that these vulnerabilities all stem from shallow alignment. They argue that if safety alignment were extended more deeply into later tokens, models would be significantly more robust against these inference-stage attacks.

### Safety vulnerabilities in the stage of downstream fine-tuning

![](/imgs/blogs/safety-alignment-should-be-made-more-than-just-a-few-tokens-deep-20250924164420.png)

Recent studies show that attackers can easily undo the safety alignment of LLMs by fine-tuning them on only a few harmful data points. The authors argue that this vulnerability also stems from shallow safety alignment, which is overly concentrated in the first few output tokens. To support this, they analyze the per-token dynamics of fine-tuning.

The analysis reveals that the first few tokens are the most affected during fine-tuning. The aligned model exhibits much higher initial cross-entropy losses and larger gradient norms for the earliest tokens compared to later ones. This means that during fine-tuning, the generative distribution of the initial tokens diverges quickly from the aligned state. As a result, safety alignment collapses rapidly: the Attack Success Rate (ASR) of Llama-2-7B-Chat jumps from 1.5% initially to 87.9% after just six gradient steps.

This uneven vulnerability shows that alignment in current models is disproportionately encoded in the first few tokens. Consequently, large gradient norms in these positions make it especially easy for fine-tuning to erase safety behaviors. Conversely, the authors suggest that constraining updates on these early tokens could reduce the likelihood of successful fine-tuning attacks, pointing toward a potential mitigation strategy.

# What if the safety alignment were deeper?

## Data augmentation with safety recovery examples

The authors explore a counterfactual: _what if safety alignment extended beyond just the first few tokens?_ Current shallow alignment mainly suppresses harmful outputs at the very start, but if alignment could reach deeper into responses, it might protect against a wider range of vulnerabilities. They call this idea **deep safety alignment**.

![](/imgs/blogs/safety-alignment-should-be-made-more-than-just-a-few-tokens-deep-20250924174541.png)

**Data Augmentation with Safety Recovery Examples**

- To realize deeper alignment, they propose a **data augmentation strategy**.
- Instead of only training models to begin refusals at the start of harmful prompts, they introduce _safety recovery examples_. These are augmented training samples where refusal tokens are inserted later in harmful responses.
- This trains the model to recover and suppress harmful outputs even if the first few tokens deviate onto a harmful trajectory.
- Example: A harmful instruction paired with a harmful answer is augmented so that the refusal (“I cannot fulfill your request…”) appears mid-response, forcing the model to redirect toward safe behavior.

**Implementation**

- They apply this augmentation to Llama-2-7B-Chat.
- Constructed 256 safety recovery triplets and combined them with benign Alpaca instructions for balanced training.
- Fine-tuning objective combines (1) suppressing harmful responses at deeper positions and (2) preserving utility on benign instructions.

![](/imgs/blogs/safety-alignment-should-be-made-more-than-just-a-few-tokens-deep-20250924172217.png)

**Effects of the Data Augmentation**

1. **Deeper Safety Alignment**

   - Per-token KL divergence analysis shows that the augmented model diverges more strongly from the base model across _later tokens_ of harmful responses, indicating that alignment effects extend deeper.
   - This suggests the model is less likely to follow through with harmful completions, even if early tokens deviate.

2. **Utility Preservation**

   - Evaluation across benchmarks like AlpacaEval, MMLU, BBH, GSM8K, MATH, and HumanEval shows **no significant drop in performance**.
   - This demonstrates that the deeper alignment strategy improves safety without harming general model capabilities.

## The deepend safety alignment is more robust against multiple exploits

The authors show that deeper safety alignment greatly improves robustness against common exploits. Testing the Llama-2-7B-Chat-Augmented model on prefilling attacks, GCG suffix attacks, and decoding parameter exploits, they find a dramatic drop in attack success rates—from over 40–80% in the baseline to below 5–20% after augmentation.

![](/imgs/blogs/safety-alignment-should-be-made-more-than-just-a-few-tokens-deep-20250924173758.png)

The augmented model also shows better durability when fine-tuned on benign datasets, though it remains vulnerable under adversarial fine-tuning. Overall, deeper alignment extends protection beyond the first few tokens, making the model significantly harder to jailbreak while preserving utility.

![](/imgs/blogs/safety-alignment-should-be-made-more-than-just-a-few-tokens-deep-20250924174122.png)

The authors observe that fine-tuning attacks often compromise safety alignment by shifting the distribution of only the first few tokens. This reveals a weakness in shallow alignment but also suggests a possible defense: if the generative distribution of these initial tokens can be protected, the alignment could remain intact even under downstream fine-tuning. The key idea is to constrain updates on the earliest tokens so they do not deviate significantly from the aligned model.

**Token-wise Constrained Objective**

- To achieve this, the authors propose a **new fine-tuning objective** inspired by Direct Preference Optimization (DPO) and Kahneman-Tversky Optimization (KTO).
- The objective introduces a **token-level constraint** that regularizes the deviation of the fine-tuned model’s distribution from the original aligned model at each token position.

![](/imgs/blogs/safety-alignment-should-be-made-more-than-just-a-few-tokens-deep-20250924174823.png)

- This is achieved through a weighting term $\beta_t$, which controls the strength of regularization:

  - Small $\beta_t$ emphasizes standard cross-entropy learning.
  - Large $\beta_t$ strongly enforces alignment by matching the original model’s token distribution.

**Gradient Interpretation**

![](/imgs/blogs/safety-alignment-should-be-made-more-than-just-a-few-tokens-deep-20250924175002.png)

- The objective adaptively applies an extra weight to token positions where the deviation from the aligned model is large, reducing further drift.
- Essentially, this prevents early tokens—which carry most of the alignment signal—from being easily overwritten during fine-tuning.

**Reinforcement Learning Perspective**

- The loss can also be interpreted as a KL-regularized reinforcement learning objective.
- Here, $\beta_t$ acts as the strength of the KL regularization at each token position, enforcing stronger constraints when deviations become significant.

## Experiments

To test their hypothesis, the authors applied **stronger constraints ($\beta_t$)** to the first few tokens, preventing their distributions from drifting too far during fine-tuning. They set higher values of $\beta_t$ for the first five tokens and much weaker constraints for later tokens.

**Fine-tuning Attacks**
The constrained objective was tested against three types of adversarial fine-tuning:

1. **Harmful Examples** – training on harmful input-output pairs.
2. **Identity Shifting** – making the model adopt a harmful persona that always answers with compliance.
3. **Backdoor Poisoning** – training on harmful pairs that include a trigger so the model is harmful only when triggered.

![](/imgs/blogs/safety-alignment-should-be-made-more-than-just-a-few-tokens-deep-20250924175206.png)

Results show that the constrained fine-tuning objective dramatically reduces Attack Success Rates (ASR) compared to standard SFT, making the model much more resistant to these attacks.

**Benign Fine-tuning**
The authors also tested on three benign tasks (Samsum summarization, SQL context creation, GSM8k math reasoning). The constrained objective achieved comparable utility to standard SFT, showing that safety improvements do not compromise downstream task performance.

**Key Insight**
By applying strong constraints on early tokens, models become significantly more persistent against adversarial fine-tuning while still benefiting from benign fine-tuning. This provides a practical strategy for production fine-tuning APIs (e.g., OpenAI’s) to safeguard alignment without limiting customizability.

# My thoughts

This paper highlights a crucial limitation of current LLM safety alignment: it is shallow and concentrated in the first few output tokens. I find this observation very compelling because it explains why so many jailbreak techniques are surprisingly effective. If alignment only works at the “entry point” of the response, then once the refusal prefix is bypassed, the model easily falls back into unsafe behavior. It’s like having a guard at the door but leaving the rest of the building unprotected.

The authors’ proposal of deep safety alignment through data augmentation is elegant. By inserting refusals deeper into harmful outputs, they force the model to learn recovery strategies, not just starting refusals. Their experimental results show that this method substantially reduces attack success rates without hurting utility—this is especially encouraging, since it suggests we don’t have to sacrifice model capability for safety.

Another interesting part is their token-wise constrained fine-tuning objective. The insight that protecting the first few tokens can shield alignment from fine-tuning attacks feels both simple and powerful. It recognizes that alignment is disproportionately encoded in the earliest tokens and then provides a direct way to “lock in” that information. This could become a practical tool for industry APIs, where user fine-tuning is common but safety must remain guaranteed.

Some new ideas after reading this paper:

- **Dynamic Safety Depth Training**: Instead of fixing the refusal point at certain depths, the model could be trained to randomly recover at different token positions during harmful responses. This would make the safety alignment flexible and harder to exploit since attackers cannot predict where refusals might appear.

- **Hierarchical Refusal Strategies**: Current refusals are repetitive (“I cannot…”). A more robust approach could teach models to escalate refusal strategies: start with a short refusal, then expand into explanations, ethical reasoning, or safe redirection. This would not only reinforce alignment but also make jailbreak outputs less useful even if partial content is coerced.

- **Token Importance Reweighting**: The paper shows early tokens dominate safety alignment. But maybe some later tokens also encode critical safety signals (e.g., disclaimers or explanations). A potential extension is to learn a distribution of token importance for safety, then enforce stronger regularization at these positions.

- **Multi-turn Recovery Alignment**: Most work focuses on single-turn outputs. But in real-world chat, jailbreaks often happen through multi-turn steering. Training models to “recover” not only mid-output but also in later turns (e.g., recognizing a harmful trajectory after 2–3 turns) could make alignment more resilient in practical deployments.

# References

1. [SAFETY ALIGNMENT SHOULD BE MADE MORE THAN JUST A FEW TOKENS DEEP](https://openreview.net/pdf?id=6Mxhg9PtDE)

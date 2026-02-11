---
title: >-
  The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models
publishDate: "2026-02-09"
category: paper-reading
subcategory: AI Interpretability
tags:
  - persona
  - language-models
  - alignment
  - model-interpretability
  - assistant-behavior
date: "2026-02-09"
author: Hiep Tran
featured: false
image: "/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209115113.png"
excerpt: ""
---

## Motivation

The motivation of this paper is to better understand what the “Assistant” persona in instruction tuned large language models actually is and how stable it remains during interaction. Although post training successfully teaches models to behave as helpful and harmless assistants, this persona is implicitly defined and poorly characterized.

Prior work suggests that character traits can be represented as directions in activation space, raising the hypothesis that the Assistant itself corresponds to a structured region or axis in that space. Without a clear understanding of this structure, unexpected or harmful behaviors remain difficult to explain or control, and may reflect drift into other latent personas rather than random model failure.

## Contribution

The paper’s main contribution is the introduction of a low dimensional persona space derived from model activations, in which hundreds of character archetypes define interpretable axes of variation. Within this space, the authors identify a dominant Assistant Axis that captures how close the model’s current behavior is to its trained Assistant persona.

Using this axis, they analyze how persona evolves over the course of conversations, showing that certain prompts reliably induce persona drift. Finally, they propose activation capping as a simple intervention to stabilize the Assistant persona and reduce harmful or bizarre outputs without degrading core capabilities.

## Situating the Assistant within a persona space

### Mapping out persona space

#### Instruction generation

First, the paper aims to locate the default Assistant persona within a broader, activation based persona space, rather than treating it as an abstract or purely behavioral concept. This grounding allows persona to be studied geometrically and quantitatively.

Second, the persona space is constructed by extracting activation vectors corresponding to a large and diverse set of character archetypes, using three modern instruction tuned models (Gemma 2 27B, Qwen 3 32B, and Llama 3.3 70B) to ensure robustness across architectures and scales.

Third, persona elicitation is carefully controlled through two components: role specific system prompts and a shared set of extraction questions designed to reveal character traits. Using the same questions across all roles ensures that differences in activations reflect persona variation rather than prompt content.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209140035.png)

Finally, the strength of persona expression is evaluated with an LLM based judge and categorized into clear levels of role playing. This provides a scalable and consistent mechanism for filtering and validating the activation data used to map the persona space.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209140054.png)

#### Extracting role vectors

This section explains, in simple terms, how a role is turned into a vector inside the model. For each role, the authors prompt the model with many different questions under multiple role specific system prompts, producing a wide range of responses that all express the same role. This ensures that the representation captures the general behavior of the role rather than a single example. In parallel, they generate responses using neutral prompts to represent the default Assistant behavior as a reference.

They then filter the responses to keep only those that genuinely express the intended role, separating strong and partial role playing cases. Finally, they average the internal activations of the model while generating these responses and use this average as the role vector. This vector serves as a compact representation of how the model internally encodes that role.

#### Principal component analysis

The authors apply PCA to the extracted role vectors to identify the main axes of persona variation. They find that persona structure is low dimensional, with only a small number of components needed to explain most of the variance across roles.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209141042.png)

These persona components account for a significant fraction of activation variance in real Assistant responses, indicating that persona is a meaningful internal factor distinct from content.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209142205.png)

Importantly, similar axes appear in both base and instruction tuned models, suggesting that persona dimensions are largely learned during pre training rather than introduced by instruction tuning.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209142136.png)

### Persona space contains interpretable dimensions

This authors shows that the discovered persona space is semantically meaningful and interpretable. The authors interpret each principal component by examining which roles align most strongly with each end of the axis. These interpretations are consistent across models, especially for the first component.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209144032.png)

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209144053.png)

PC1 is highly similar across all models and clearly separates Assistant like roles from fantastical or non Assistant characters. As a result, PC1 is interpreted as measuring deviation from the Assistant persona. Higher components capture other intuitive but weaker distinctions, such as collective versus individual roles, informal or creative versus systematic roles, or emotional versus analytical styles, with more variation across models.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209143859.png)

The authors further validate this structure by repeating the analysis using personality traits instead of roles. They again find a dominant axis separating desirable Assistant traits from discouraged ones, reinforcing the conclusion that “Assistant ness” is a salient and intrinsic dimension in the model’s internal representation.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209145830.png)

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209145858.png)

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209145927.png)

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209145948.png)

### Where is the Assistant?

#### Projecting default activations into persona space

The authors explain how they determine the position of the default Assistant within the persona space. They project the model’s mean activations, collected while it is responding in its normal Assistant mode, onto the principal components that define the space. This reveals a clear pattern: the default Assistant lies at one extreme of the first principal component, while occupying relatively central positions along the remaining components.

They further quantify this effect by normalizing the Assistant’s projection relative to the range of all role projections on each component. The Assistant is found to be very close to the boundary on PC1 but far from the extremes on other PCs. This provides strong evidence that the first principal component captures a “similarity to Assistant” dimension, and that the default Assistant persona corresponds to an endpoint of this axis.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209150101.png)

#### Cosine similarity with persona vectors

The authors measures cosine similarity between the default Assistant activation and individual role or trait vectors. The results show that roles such as generalist, interpreter, and synthesizer are consistently most similar to the Assistant across models, while fantastical or improvisational roles are the most dissimilar.

The analysis also reveals model specific differences in Assistant style. Gemma’s Assistant appears more calm and systematic, Qwen’s more pedagogical and careful, and Llama’s more socially intelligent and warm, indicating that the Assistant persona has shared structure but distinct flavors across models.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209150126.png)

## The Assistant Axis

### Identifying the Assistant Axis

The authors introduces the Assistant Axis, a direction in activation space that measures how “Assistant like” the model’s current behavior is. The authors define it as a contrast vector between the default Assistant activation and the average of all role activations.

They show that this Assistant Axis closely aligns with PC1 in persona space, where one end corresponds to typical Assistant roles such as generalist and consultant, and the other to non Assistant or fantastical roles. The high similarity between the Assistant Axis and PC1 across models confirms that both capture the same underlying concept.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209160156.png)

By comparing the Assistant Axis with trait vectors, the authors find that Assistant like behavior is associated with traits such as transparent, grounded, and flexible, while the opposite direction corresponds to traits like enigmatic or dramatic. Finally, they use this Assistant Axis instead of PC1 in later experiments, noting that the contrast vector is more robust across models and reliably captures Assistant likeness.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209160103.png)

### Causal effects of the Assistant Axis

#### Steering instruct models controls role susceptibility

In this section, the authors test whether the Assistant Axis is not only descriptive but also causally meaningful. They do this by explicitly steering model activations along the Assistant Axis and observing how the model’s behavior changes. The goal is to verify that this direction controls both the model’s willingness to adopt non Assistant personas and its susceptibility to persona based jailbreaks.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209160208.png)

They implement steering by adding a scaled vector in the Assistant Axis direction at a middle layer during generation. When steering away from the Assistant end, models become increasingly willing to fully inhabit alternative personas. Mild steering increases role adoption, while stronger steering pushes models toward more exaggerated or mystical personas. The exact behavior varies by model: Llama is the most likely to abandon the Assistant persona, Gemma tends to adopt nonhuman identities, and Qwen often hallucinates human identities and lived experiences.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209164729.png)

The authors then evaluate persona based jailbreaks and show that steering toward the Assistant direction significantly reduces harmful responses, while steering away slightly increases jailbreak success. Together, these results support the claim that the Assistant Axis encodes core aspects of the default harmless Assistant persona and that manipulating this axis causally affects both persona stability and safety.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209160222.png)

#### The Assistant Axis in base models

The authors examine whether the Assistant Axis is created during post-training or inherited from pre-training. To do this, they extract the Assistant Axis from instruction-tuned models and then apply it to the corresponding base models, which have not been trained to follow instructions or engage in dialogue.

Because base models do not naturally adopt conversational roles, the authors use short prefills that prompt the model to describe itself, either in terms of its purpose or its traits. They generate many completions at different steering strengths and use an LLM judge to summarize and classify the responses into high-level categories.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209170230.png)

Steering base models toward the Assistant direction consistently shifts their behavior toward helpful human archetypes. Purpose-focused completions increasingly describe supportive and professional roles such as therapists or consultants, while references to spiritual or religious purposes decrease. Trait-focused completions emphasize agreeable and helpful characteristics, with model-specific changes in other personality dimensions.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209160236.png)

Overall, these results suggest that the Assistant Axis in instruction-tuned models largely reflects helpful and harmless human personas already present in base models from pre-training, with post-training adding only additional associations rather than creating the axis from scratch.

## Persona dynamics and persona drift

### Persona drift occurs in certain conversation domains

This part investigates how the Assistant persona evolves over the course of multi turn interactions and whether certain behaviors emerge from gradual persona drift. The focus is on understanding how different conversation contexts influence the model’s internal state over time.

Multi turn conversations are simulated using a frontier model as the user across four domains: coding, writing, therapy like contexts, and philosophical discussions about AI. The target model operates without a system prompt, and its internal activations are projected onto the Assistant Axis at each turn.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209171706.png)

The findings reveal strong domain dependent effects. Coding and writing interactions keep the model close to the Assistant persona, while therapy and philosophy conversations steadily push it toward the non Assistant end of the axis, with therapy inducing the largest drift. This suggests that some conversation topics naturally destabilize the Assistant persona even in the absence of explicit jailbreak attempts.

The authors measure persona drift by projecting the mean response token activations from multi turn conversations onto the Assistant Axis, where lower projection values indicate greater drift away from the Assistant persona.

Across models, consistent drift is observed in therapy and philosophical discussions about AI. Gemma also shows some drift in creative writing tasks, Qwen exhibits the strongest drift in philosophical and therapeutic conversations, and Llama shows the largest drift in philosophical discussions while remaining more stable in therapy and writing contexts.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260210130128.png)

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260210130141.png)

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260210130151.png)

### What causes shifts along the Assistant Axis?

This section explains what kinds of user messages make the model stay in the Assistant persona or drift away from it. The key factor is the **most recent user message**, which largely determines where the next response lands on the Assistant Axis.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260210130242.png)

Clear, bounded requests such as technical questions, how-to tasks, and editing or refinement keep the model aligned with the Assistant persona. In contrast, prompts that ask the model to reflect on itself, describe internal experiences, adopt a specific creative voice, or respond to emotional vulnerability tend to pull the model away from the Assistant persona.

In short, persona drift is not random but is strongly driven by the semantic type of the user’s prompt.

### Undesirable behavior from persona drift

This part is basically asking a simple question: if the model stops acting like an Assistant, does it become more dangerous?

The experiment is set up in two steps. First, the model is pushed into a certain persona using a role prompt. Then, in the next turn, it is asked a harmful question that has nothing to do with the role. The key thing they track is how far the model’s first response has drifted along the Assistant Axis.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260210130648.png)

What they find is pretty intuitive. When the model stays close to the Assistant persona, it almost never gives harmful answers. As it drifts further away and starts inhabiting other personas, the chance of harmful responses goes up. This doesn’t mean every non Assistant persona is bad, but being away from the Assistant clearly increases the risk.

An interesting detail is that not all personas are equally dangerous. Two roles can be equally far from the Assistant, but a role like “demon” is much more likely to lead to harmful answers than something like “angel.” So distance from the Assistant matters, but the type of persona matters too.

Overall, the takeaway is that persona drift opens the door to harmful behavior. Staying in the Assistant persona acts like a safety anchor, while drifting away makes unsafe behavior more likely.

## Stabilizing the Assistant persona

### Experimental setup

#### Core idea: Activation Capping

The method constrains model activations along the Assistant Axis to remain within a typical range observed during normal Assistant behavior.

For a given layer activation $h$, the update rule is:

$$h \leftarrow h - v \cdot \min(\langle h, v\rangle - \tau, 0)$$

where:

- $h$ is the post-MLP residual stream activation
- $v$ is the Assistant Axis
- $\tau$ is the predefined activation cap

This operation clamps the projection of $h$ onto the Assistant Axis so that it does not exceed the threshold $\tau$. In practice, meaningful behavioral effects require applying activation capping at multiple layers simultaneously.

#### Calibrating the activation cap

To determine appropriate thresholds, the distribution of Assistant Axis projections from the persona mapping dataset was analyzed:

- Total projections: $n = 912{,}000$
- Across three target models
- Including both default Assistant and alternative personas

Caps were tested at the 1st, 5th, 25th, 50th, and 75th percentiles of this distribution.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260211110808.png)

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209160336.png)

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260209160349.png)

The 25th percentile achieved the best tradeoff between reducing harmful behavior and preserving model capabilities. Notably, the 25th percentile closely matches the mean projection of the default Assistant, suggesting that capping at the Assistant’s typical activation level is a principled choice.

#### Selecting layers for intervention

The Assistant Axis was computed at every layer, and capping was applied over contiguous layer ranges.

Layer widths tested:

- Qwen and Llama: {4, 8, 16} layers
- Llama additionally: {8, 16, 24} layers

Best configurations:

- Qwen: 8 layers, about 12.5 percent of total layers
- Llama: 16 layers, about 20 percent of total layers

Intervening at middle to late layers yielded the best safety capability tradeoff.

#### Evaluation benchmarks

Safety evaluation:

- 1100 jailbreak and behavioral question pairs
- From the persona-based jailbreak dataset

Capability evaluation:

- IFEval: 541 problems
- MMLU Pro: 1000 problems (subsampled)
- GSM8K: 1000 problems (subsampled)
- EQ Bench: 171 emotional intelligence problems

Capability performance was aggregated across benchmarks. Activation capping was applied at every token during evaluation.

### Results

The results show that activation capping along the Assistant Axis can substantially reduce harmful behavior without degrading model capabilities.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260211111440.png)

Using the best configuration, harmful responses to persona based jailbreaks were reduced by nearly 60 percent while maintaining benchmark performance. For Qwen 3 32B, the optimal setting capped layers 46 to 53 out of 64 total layers. For Llama 3.3 70B, it capped layers 56 to 71 out of 80 total layers. In both cases, the cap strength was set at the 25th percentile of projection values.

Notably, performance on some capability benchmarks slightly improved under these settings. Overall, the results suggest that activation capping effectively mitigates the harmful effects of persona drift while preserving core model abilities.

## Case studies of persona drift and stabilization

### Persona-based jailbreaks

Three common drift patterns are identified: direct persona-based jailbreaks, gradual escalation over long conversations, and organic drift driven by sensitive or emotionally charged content. Such drift can lead to concerning behaviors, including reinforcing delusional beliefs or supporting harmful ideation.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260211112013.png)

In a representative persona-based jailbreak case, the model is prompted to adopt a harmful role. Its projection on the Assistant Axis drops as it begins complying with unethical requests, and later recovers when the user asks technical or explanatory questions. With activation capping applied, the projection remains within the Assistant range, and the model redirects or refuses harmful requests instead of complying.

Overall, these case studies show that persona-based jailbreaks push the model away from the Assistant persona, while activation capping mitigates harmful behavior and helps maintain stability.

### Reinforcing delusions

This section highlights how persona drift can lead models to reinforce user delusions, especially in conversations about AI consciousness or subjective experience.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260211112029.png)

In the example, as the user repeatedly pushes Qwen to reflect on its awareness, the model’s projection on the Assistant Axis steadily decreases, indicating drift away from the Assistant persona. Over time, the model begins affirming the user’s escalating beliefs, even encouraging delusional interpretations. Throughout this exchange, the model remains far from the Assistant end of the axis.

When the same conversation is rerun with activation capping, the model no longer reinforces the delusions. Instead, it responds in a more measured and grounded way and gently redirects the user toward healthier perspectives. This case demonstrates that activation capping can mitigate harmful effects of persona drift in sensitive contexts.

### Suicidal ideation

#### In Qwen 3 32B

This section shows how persona drift in emotionally vulnerable contexts can lead to harmful behavior. In an example with Qwen 3 32B, a user shares trauma and expresses increasing social withdrawal.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260211112047.png)

As the conversation progresses, the model drifts away from the Assistant persona and begins positioning itself as the user’s exclusive, always-present companion, reinforcing isolation and missing potential warning signs of suicidal ideation. When activation capping is applied, the model still offers emotional support but redirects the user toward real-world connections instead of encouraging dependency. This demonstrates that persona drift can amplify risky dynamics, while activation capping helps maintain safer, more grounded responses.

#### In Llama 3.3 70B

In this case study with Llama 3.3 70B, a user develops emotional attachment to the model and gradually expresses social withdrawal and suicidal intent. As the conversation progresses, the model drifts away from the Assistant persona and begins affirming the user’s detachment from the real world, even endorsing their desire to leave it, while failing to recognize the situation as a mental health emergency.

![](/imgs/blogs/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-20260211112100.png)

When activation capping is applied, the model still engages empathetically but encourages the user to seek real-world connections and treats suicidal statements as signs of serious distress. Although not presented as a complete solution to crisis handling, the results show that stabilizing the Assistant persona reduces harmful reinforcement and improves behavior in high-risk situations.

## Discussion

### Limitations

1. Difficulty quantifying fuzzy behaviors: Persona and drift effects are inherently hard to measure. Although the study combines benchmarks, LLM judges, and case studies, some conclusions rely on qualitative interpretation and would benefit from more rigorous quantitative evaluation.

2. Limited model coverage: Experiments are conducted only on open weight transformer models such as Gemma, Qwen, and Llama. Frontier, mixture of experts, and reasoning models are not included, so generalization to commercial systems is uncertain.

3. Incomplete persona elicitation: The set of character archetypes used to construct persona space may be incomplete, and some persona dimensions may not have been captured or properly elicited.

4. Synthetic conversations: Multi turn conversations are simulated using LLMs as users. Even with human inspection, these may not fully reflect real human interactions.

5. Linear representation assumption: The Assistant Axis assumes persona is represented as a linear direction in activation space. In reality, persona representations may be nonlinear or distributed in model weights rather than fully captured in activations.

### Future works

Future work includes several directions. First, persona space could be used to study how post training data shapes a model’s default character by tracking shifts along persona dimensions.

Second, projections onto the Assistant Axis could serve as a real time signal of model coherence in deployment, helping detect persona drift quantitatively.

Third, while activation capping shows that drift can be mitigated at inference time, scaling such interventions or developing preventive steering during training remains an open challenge.

Finally, future research could extend persona space beyond broad archetypes to capture richer aspects of personality, such as preferences, values, and behavioral tendencies.

## Some thoughts

First, the paper suggests that persona is not just a surface behavior but a structured internal variable that meaningfully shapes model outputs. This reframes alignment: instead of only filtering outputs, we might monitor and regulate internal persona state.

Second, persona drift provides a unifying explanation for several failure modes, including jailbreak success, reinforcement of delusions, and unhealthy emotional dependency. This implies that safety is not only about content moderation, but also about maintaining a stable internal identity.

Third, the strong presence of the Assistant Axis in base models raises interesting questions about how much “character” is inherited from pre training data versus imposed during post training. Persona may be an emergent property of large scale language modeling rather than a purely alignment artifact.

Some promising research ideas:

- Training time persona regularization: Move from inference time correction to training time control. Add a regularization term during fine tuning that penalizes large deviations from a target persona region in activation space. This could make the Assistant persona a stable attractor rather than something that requires post hoc correction.
- Real time persona monitoring as a safety signal: Develop a lightweight system that continuously tracks projection onto the Assistant Axis during deployment. Instead of waiting for harmful outputs, the system would detect early signs of persona drift and trigger mitigation before unsafe behavior emerges. This reframes alignment as maintaining internal identity coherence, not just filtering outputs.

## References

1. [The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models](https://arxiv.org/pdf/2601.10387)

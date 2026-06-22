---
title: "The LLM as Agent Brain: Why Next-Token Prediction Produces Planning"
date: "2026-06-22"
description: "Why large language models work as agent brains, how emergent planning arises from pretraining, and exactly where LLMs fail as reasoning engines."
tags: ["ai-agents", "llm", "agent-architecture", "machine-learning", "deep-learning", "nlp", "system-design", "production-ml"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 50
---

Here is the thing that nobody who builds agents will tell you up front: nobody designed the planning capability. Nobody wrote a planning module, no one encoded a search algorithm, no researcher sat down and specified "now produce a goal decomposition system." What we got instead is a statistical engine trained to predict the next token in a sequence — and when you scale that engine large enough, on enough text, aligned it with human preferences, and told it to use tools, it started planning.

That is the surprising truth at the center of every modern AI agent: the brain is a prediction machine that emergently learned to reason.

This post is about understanding exactly how that happened, what it means for the systems you build on top of it, and — just as importantly — where the prediction machine breaks down. Because it does break down, in specific and predictable ways, and if you understand the mechanism you can build around the failures instead of being surprised by them.

If you have not yet read [what an AI agent actually is](/blog/machine-learning/ai-agent/what-is-an-ai-agent), start there. This post assumes you know the perceive-reason-act loop; here we go one level deeper into the "reason" box.

## The Surprising Truth: Planning Wasn't Designed In

Let us start with the analogy that frames everything.

An LLM is like a chef who has read every cookbook, every restaurant review, every food science paper, and every recipe thread on the internet — but has never actually cooked. They have an extraordinarily detailed world model built from text. They can predict what happens if you add too much salt. They can describe the Maillard reaction in molecular detail. They can plan a seven-course tasting menu from ingredients. But they built all of that from descriptions of cooking, not from cooking itself.

That is both the power and the limitation. The world model is vast but vicarious. And the planning capability — the ability to decompose a goal into steps, recognize when a step failed, and try a different approach — was never written down anywhere. It emerged from the texture of the training data, which contains millions of examples of humans doing exactly that.

Consider what is in a pretraining corpus. There are debugging sessions where someone tried an approach, it failed, they diagnosed the problem, and tried again. There are research papers that describe hypothesis generation, experiment design, and result interpretation. There are chess analyses where a player considers move sequences. There are legal briefs that construct arguments step by step. There are Stack Overflow answers that decompose a complex problem into tractable sub-problems. The model did not learn planning by being told about planning — it learned it by absorbing billions of instances of humans planning.

The implication for agent builders: planning is a learned approximation, not a formal procedure. It works in the distribution of problems the training data covered. It fails outside that distribution, or when the problem structure deviates enough from patterns the model has seen. Understanding this is the first prerequisite for building reliable agents.

| Property | Designed system | LLM-based planning |
|---|---|---|
| Planning approach | Explicit search algorithm | Statistical approximation from pretraining |
| Behavior outside training distribution | Defined fallback | Unpredictable degradation |
| Correctness guarantees | Formal proofs possible | None — probabilistic |
| Interpretability | Fully inspectable | Opaque reasoning |
| Adaptability to new tasks | Requires reprogramming | Often generalizes |

## From Prediction to Reasoning: The Four-Stage Path

The jump from "predicts next tokens" to "decomposes goals into tool calls" did not happen in one step. There is a four-stage compounding process, and understanding each stage helps you understand both why agents work and where they break.

Before getting into the stages, consider why transformers specifically — not recurrent networks, not convolutional networks, but transformers — are the architecture that makes agents possible. The attention mechanism lets a transformer attend to any prior token in the sequence with equal ease. When you are running a multi-step reasoning chain and the current step depends on a fact established three steps ago, the model can attend directly to that fact. Recurrent architectures compress prior context into a fixed hidden state; by the time you are at step 5, the information from step 1 has been compressed through four layers of lossy updates. Transformers do not have this problem — they have direct access to the full context at every step, which is precisely what makes multi-step reasoning possible.

The second relevant architectural property is scale. Transformers exhibit scaling laws: as you add parameters and training compute, loss decreases predictably, and new capabilities emerge at specific scale thresholds. The capabilities relevant to agents — instruction following, tool use, multi-step planning, self-correction — are not present at small scales. They emerge. This is not a gradual improvement; it is a phase transition. A model trained on 1/10th of the compute of a frontier model is not 10% less capable at planning — it often has essentially zero planning capability.

![The four-stage emergence path from raw next-token prediction to multi-step planning](/imgs/blogs/llm-as-agent-brain-1.webp)

**Stage 1: Pretraining — the world model.**
A transformer trained on internet-scale text learns to compress a statistical model of the world into its weights. It learns syntax, semantics, factual associations, causal relationships, and — critically — procedural patterns. The model that can predict the next token in "first, install the dependencies; next, configure the environment; finally, run the tests" has implicitly learned something about sequential procedures.

This is not symbolic planning. It is pattern completion. But pattern completion over a corpus that contains millions of planning examples produces something that looks remarkably like planning.

**Stage 2: Instruction following — alignment to user intent.**
Pretraining produces a powerful but chaotic model. It will complete your text, but it might complete it as a continuation of the training document rather than a response to you. Instruction fine-tuning (SFT) and reinforcement learning from human feedback (RLHF) reshape the model's output distribution toward being helpful and responsive to requests.

This stage matters for agents because it changes what the model is optimizing for. Pre-RLHF, the model is optimizing for plausible text completion. Post-RLHF, it is optimizing for task completion according to human intent. That shift is what makes the model behave like an agent brain rather than an autocomplete engine.

**Stage 3: Tool calling — structured output enables action.**
Once you have a model that follows instructions, you can teach it to produce structured output that represents an action rather than natural language. Function calling — the ability to emit a JSON object representing a tool invocation — is this capability. It connects the model's planning to the physical world: the agent can now actually do things, not just describe what should be done.

Tool calling is where the rubber meets the road for agents. Without it, the model is a talking planning system. With it, it becomes an acting planning system.

**Stage 4: Multi-step planning — chaining to complete goals.**
Once a model can follow instructions and call tools, it can chain these capabilities over multiple steps. Given a goal, it can plan a step, execute it, observe the result, revise the plan, and continue. This is the emergent planning behavior that makes agents useful.

The key insight: none of Stage 4 was explicitly programmed. The chaining behavior emerged from Stages 1-3 in combination. And this means it is fragile in ways that designed planning systems are not.

There is a subtlety worth understanding about Stage 4. Multi-step planning as LLMs do it is not the same as classical AI planning (which involves explicit state representations, goal conditions, and search over a plan space). LLM planning is generative — the model generates the next action by predicting what an agent would do in this context given its training. It is closer to "what would a competent human do next?" than to "what is the provably optimal action according to a search algorithm?"

This generative nature is both its strength and its weakness. Strength: the model can handle ambiguous situations, make reasonable assumptions, and adapt to unexpected tool outputs without explicit re-planning. Weakness: the model can be confidently wrong in ways that a search-based planner would catch. A search-based planner knows when it has no valid plan and can report that. A generative model will produce a plan even when no good plan exists, and it will sound confident.

The engineering implication: build your agent infrastructure to detect when the LLM's plan is not working and trigger replanning or escalation. Do not assume the model knows when to stop.

## The Transformer Architecture as Agent Foundation

Before continuing, it is worth being precise about what "transformer" means in this context, because the architecture directly determines the capabilities and limitations of the agent brain.

A transformer processes its input — the full context window — in parallel using self-attention. Each token can attend to every other token in the window. This is what enables a model at step 7 of a reasoning chain to directly reference information from step 1: all of it is in the context window, all of it is equally accessible.

The attention mechanism computes, for each token, a weighted sum of value vectors from all other tokens, where the weights are determined by how "relevant" each token is to the current token. This relevance is learned during training. The model learns that the word "therefore" should attend heavily to the preceding premises. It learns that a tool call should attend to the system prompt's description of available tools. These attention patterns are the mechanism through which context influences generation.

What this means for agent behavior:

**Attention is where reasoning happens.** When the model appears to "think through" a problem, it is constructing attention weights that route information from relevant parts of the context to the tokens being generated. This is not symbolic reasoning — it is a learned pattern of information routing.

**The model has no internal state between turns.** Unlike a human who remembers what they were thinking, the transformer starts each forward pass from scratch, processing the full context window. There is no hidden state that carries information across turns except what is explicitly in the context. This is why context management is so critical: the context window IS the model's state.

**Attention sinks and position bias are not bugs — they are features of the training distribution.** Transformers attend strongly to the first few tokens because the first tokens in pretraining documents are often high-information (titles, introductory sentences). They attend strongly to recent tokens because recent context is most predictive of the next token in natural language. These learned biases are appropriate for language modeling but cause problems in agent contexts where you need uniform attention to a long context.

**The softmax bottleneck.** Attention uses softmax normalization, which means attention weights sum to 1. When the context is very long, each individual token gets a smaller share of the attention budget. Instructions buried in a 50K-token context receive less attention than the same instructions in a 2K-token context, all else being equal.

Understanding these architectural properties lets you make better engineering decisions. You do not fight the architecture — you work with it. You put critical instructions at positions the model naturally attends to. You keep contexts short. You structure tool schemas to take advantage of how the model processes structured text.

## Why Instruction Following Enables Tool Use

This connection is not obvious until you think about it carefully, so it is worth spending time here.

A pretraining-only model will produce text in the style of its training data. If it has seen function call patterns in the training data, it might produce function-call-shaped text. But it will not do so reliably, in the right format, with the right schema, in response to a user request.

Instruction following changes this. When a model learns to "respond to requests by producing the requested output format," and when the training data for instruction following includes examples of function calls, the model learns that a function call is a valid and appropriate response type. It learns the schema because schemas are in the training data. It learns when to use a tool versus when to answer directly because human labelers rewarded appropriate tool use during RLHF.

The second connection is more subtle: instruction following gives the model a consistent goal frame. A model that is told "answer the user's question" treats tool use as instrumental to that goal. It calls a search tool because search helps answer questions. It calls a calculator because the question involves arithmetic. The goal frame (satisfy the user request) makes the model productive use tools appropriately.

This is why you cannot easily add tool use to a model that has not been instruction-tuned. The raw pretraining model has no stable goal frame. It does not reliably interpret a tool schema as "these are actions you can take to help the user" — it might complete the schema as if it were a documentation file or an API response.

**The parallel function calling problem.** Once a model can call tools, a natural next capability is calling multiple tools in parallel — launching several searches simultaneously rather than sequentially. This requires the model to understand that it is producing multiple independent outputs that will be executed concurrently. This capability appears reliably only in frontier models, and even there requires explicit prompting or structured output. A model that does not understand parallel tool calls will serialize everything, producing an agent that is much slower than it needs to be.

**Tool error recovery.** When a tool returns an error (network timeout, invalid input, authentication failure), a capable agent should recognize the error, understand its cause, and either retry, use a fallback tool, or escalate to the user. This requires the model to:

1. Recognize that the tool output is an error, not a valid result.
2. Reason about what caused the error.
3. Choose an appropriate recovery action.

Step 1 sounds easy but is not: if the tool returns a plausible-looking response that is semantically an error message (e.g., "no results found" or "rate limit exceeded"), a model that is not trained to recognize these patterns may treat them as valid results and continue on a false premise. Building robust tool error recovery requires both model capability (the model must reason about errors) and infrastructure support (error responses must be clearly marked as errors, not ambiguous text).

![LLM capability profile — what works and what breaks in agent contexts](/imgs/blogs/llm-as-agent-brain-2.webp)

The profile above is the honest picture. Language understanding and code generation are strong. Tool use — learned through instruction tuning — is reliable in frontier models. Multi-step reasoning is inconsistent, depending on problem type and step count. Long-horizon planning is genuinely unreliable. Arithmetic fails without a tool. Factual recall carries hallucination risk.

| Capability | Why it works in LLMs | Mitigation when it fails |
|---|---|---|
| Language understanding | Directly trained on text | N/A — this is the foundation |
| Code generation | Huge code corpus + instruction tuning | Testing, static analysis |
| Simple tool use | Instruction fine-tuning with tool examples | Schema validation, retry logic |
| Multi-step reasoning | Pattern completion from training | Chain-of-thought prompting, verification |
| Long-horizon planning | Emergent — fragile | Task decomposition, verification gates |
| Arithmetic | Not reliably trained for this | Always use a calculator tool |
| Factual recall | Memorized patterns — not lookup | RAG, grounding |

## Context Window as Working Memory

The context window is the agent's working memory, and it has a structure that matters enormously for reliability. Understanding this structure is not optional if you are building production agents — it directly determines what the model can attend to and what it will forget.

![Context window anatomy showing the six layers and their priorities](/imgs/blogs/llm-as-agent-brain-3.webp)

The six layers serve different purposes and have different dynamics.

**System prompt** sits at the top and contains the most stable information: the agent's persona, rules about what it can and cannot do, output format requirements. In production, this is usually cached (via prompt caching) because it rarely changes. A well-written system prompt is the closest thing LLM agents have to source code.

**Tool schemas** define what actions the agent can take. These are either static (the same tools are always available) or dynamic (tools are injected based on the current task context). Every tool schema consumes tokens, and in a tool-heavy agent, schemas alone can occupy 15-25% of the context window.

**Few-shot examples** are the highest-value optional layer. A few carefully chosen examples of the desired behavior can dramatically reduce hallucination and improve format adherence. The cost is tokens — each example might consume 200-500 tokens. In production, the decision of whether to include few-shot examples is a latency-cost-quality tradeoff you will make explicitly.

**Retrieved context** is the RAG layer — documents, facts, or memories injected per-turn based on the current query. This is where [retrieval-augmented agents](/blog/machine-learning/ai-agent/basic-rag) do their work. The retrieved context is often the most variable layer: it might be empty (no relevant retrieval) or enormous (many relevant documents), and managing its size is a core engineering problem.

**Conversation history** accumulates as the agent runs. In a multi-turn agent, every prior turn adds tokens. Without active management, this layer will eventually overflow the context window. The engineering pattern here is sliding-window truncation or summarization of old turns.

**User message** is always at the bottom — the most recent and highest-priority input. A failure mode to watch for: when the context window is too full, the LLM's attention to the user message can degrade because the other layers are crowding it out.

The hard limits of context-as-working-memory:

**Position bias.** LLMs do not attend uniformly to the context window. They attend strongly to the beginning and end, and they underweight the middle. If your retrieved context is 8,000 tokens long, the facts in the middle may be effectively invisible to the model. This is the "lost in the middle" problem that has been extensively documented. The practical fix: for critical instructions, repeat them at both the beginning and end of the system prompt. For long retrieved contexts, chunk them into smaller pieces with explicit section headers.

**Token budget is zero-sum.** Every token you spend on tool schemas is a token not available for retrieved context or conversation history. In a 128K token context window, this sounds like plenty — until you realize that a complex system prompt, five tool schemas, three few-shot examples, and a RAG result can easily consume 50K tokens before a single user message. Budget your context window carefully: every component has a maximum allocation and you need a policy for what gets dropped when the limit is approached.

**No true memory consolidation.** Unlike biological working memory, there is no consolidation process that synthesizes context into durable long-term memory within a single context window. Each turn, the model processes the full context from scratch. The engineering pattern to work around this is external memory: after each multi-step task, have the agent write a summary of what it learned to an external store. Future tasks retrieve relevant memories as part of the RAG layer.

**Attention cost scales quadratically.** The self-attention mechanism has O(n²) computational cost with respect to sequence length. Doubling the context window roughly quadruples the attention computation (before hardware optimizations). Very long contexts are significantly more expensive per inference than short ones — another reason to actively manage context length.

**The effective context window is shorter than the nominal.** A model advertised as "200K context" can technically process 200K tokens, but it does not attend to all 200K with equal quality. Empirically, most models show degraded performance on tasks that require attending to information 50K+ tokens ago. Design your context management around an effective window of 50K-80K, using external memory for anything older. This is conservative but production-safe.

| Context layer | Typical token budget | Change frequency | Priority |
|---|---|---|---|
| System prompt | 500-2,000 | Deploy-time | Highest |
| Tool schemas | 200-500 per tool | Task-type | High |
| Few-shot examples | 200-500 each | Campaign-level | Medium |
| Retrieved context | 500-10,000 | Per-turn | High (variable) |
| Conversation history | Grows unbounded | Per-turn | Medium (must prune) |
| User message | 10-500 | Per-turn | Highest |

## Emergent Capabilities and Scale

The word "emergent" gets overloaded in AI discourse, but here it means something specific: capabilities that are absent or near-random at lower scales and then appear suddenly — as a step function — at higher scales.

![Emergent capabilities at each model scale tier](/imgs/blogs/llm-as-agent-brain-4.webp)

The scale tiers are real, and the capability differences between tiers are not gradual.

**Small models (< 7B parameters)** can handle basic question-answering, simple text completion, and extraction tasks with structured schemas. They struggle with instruction following on complex prompts and fail at multi-step reasoning. Their value in agents is as specialized workers for narrow, well-defined tasks — "extract the entity names from this paragraph" or "classify this customer message as positive/neutral/negative." Do not use them as the main reasoning brain.

**Medium models (7B-70B)** cross the instruction following threshold. They can reliably follow complex prompts, they can call tools, and they can handle single-step reasoning. Two-hop reasoning (A implies B, B implies C, therefore A implies C) starts to work. They are good agents for well-structured tasks with clear branching logic — customer support, form filling, simple research tasks. The cost-quality tradeoff at this tier is compelling: models like Llama 3 70B deliver 80% of frontier capability at 10% of frontier cost.

**Large models (70B+)** cross the multi-hop reasoning threshold. Three-hop and four-hop reasoning chains become reliable. Complex planning with multiple alternatives becomes possible. These models can handle ambiguous instructions and make reasonable choices when the right action is unclear. This is the tier where most "agentic" behavior first becomes reliable.

**Frontier models (GPT-4 class, Claude 3+ class)** cross the autonomous agent threshold. Self-correction — recognizing when a previous step was wrong and recovering — becomes reliable. Long-horizon planning holds together over 5+ steps. Complex tool use with error recovery works. These models can handle genuinely open-ended problems where the solution path is not known in advance.

The practical consequence: the capability gap between a 7B model and GPT-4 is not 2x. For multi-step agent reasoning, it is closer to 10x in terms of task success rates. This is not a continuous spectrum — it is a step function.

## The Hallucination Problem in Agent Chains

Hallucination in a conversational chatbot is annoying. Hallucination in an agent chain is catastrophic. The reason is compounding.

In a single-turn conversation, the model either produces a hallucinated answer or it does not. The user can verify it. Damage is bounded.

In a multi-step agent chain, a hallucination at step 2 becomes input to step 3. Step 3 builds on a false premise. Step 4 builds on step 3's output, which is now further distorted. By step 5, the error has propagated and amplified through the entire chain.

![Hallucination compounds in a 5-step agent chain — cumulative error probability grows faster than linearly](/imgs/blogs/llm-as-agent-brain-5.webp)

The math is simple but devastating. If each step has a 5% probability of producing an error (which is optimistic for complex reasoning steps), then a 5-step chain has a cumulative error probability of:

```
P(at least one error in n steps) = 1 - (1 - 0.05)^n
```

At n=5: 1 - (0.95)^5 ≈ 23%.

At n=10: 1 - (0.95)^10 ≈ 40%.

At n=20: 1 - (0.95)^20 ≈ 64%.

Now remember that 5% per step is optimistic. For complex reasoning steps that require multi-hop inference, the per-step error rate can be 10-20%. At 10% per step:

At n=5: 1 - (0.90)^5 ≈ 41%.

A five-step agent chain with 10% per-step error rate has a 41% chance of producing a wrong final answer. This is not a small problem. It is the central problem of agent reliability.

The second issue is error amplification. Not all errors compound equally. Some errors produce downstream nonsense that is obvious. Others produce downstream results that are wrong but plausible — the model confidently built a coherent story on a false foundation. These silent errors are the most dangerous: the agent appears to succeed, the user accepts the output, and the error propagates into the real world.

**Mitigation patterns:**

1. **Verification checkpoints.** After each step, verify the output before passing it to the next step. This can be as simple as schema validation (is the output the right shape?), a secondary LLM call to review the output, or a deterministic check (does this database ID actually exist?).

2. **Shorter chains.** Every additional step multiplies the error probability. Prefer 3-step chains over 6-step chains. Break a 10-step task into two verified 5-step sub-tasks.

3. **Ground the critical facts.** If step 2 extracts a fact that all subsequent steps depend on, verify that fact externally before continuing. A single extra tool call to confirm a number can prevent five downstream steps from building on a wrong premise.

4. **Human-in-the-loop gates.** For high-stakes operations, insert a human review step before irreversible actions. The cost is latency; the benefit is catching compounded errors before they cause real damage.

## Planning Horizons: How Far Ahead Can LLMs Reliably Reason?

This is the question that every agent architect eventually confronts: how many steps can I trust the model to plan before reliability degrades unacceptably?

The answer, based on empirical evaluation of frontier models: reliable planning holds up to roughly 3-5 steps. Beyond that, reliability degrades sharply. At 8+ steps, even the best frontier models produce correct final answers only about 60% of the time — and smaller models do much worse.

![Short-horizon tasks maintain 95% accuracy; long-horizon tasks degrade to 60% by step 8](/imgs/blogs/llm-as-agent-brain-6.webp)

Why does reliability degrade?

**Context accumulation.** As a multi-step task runs, the conversation history grows. Each step adds more context, and the model must attend to all of it. The position bias problem discussed earlier means that information from earlier steps — which is now in the middle of the context window — receives less attention. The model "forgets" earlier constraints and goals.

**Goal drift.** In long-horizon tasks, the model can gradually drift from the original goal. Each step is a slightly different framing of the task, and over many steps these small drifts accumulate. The model executing step 7 may be implicitly optimizing for a subtly different goal than the one specified at step 1.

**Error accumulation.** As discussed in the hallucination section, errors compound. Even without a dramatic hallucination, small inaccuracies at each step add up.

**Ambiguity resolution.** Real tasks are underspecified. The user says "research this topic and write a report." What counts as sufficient research? How long should the report be? How should conflicting sources be handled? Over many steps, the model makes implicit choices at each ambiguity, and these choices can diverge from what the user intended.

**Practical benchmarks on frontier models:**

| Task type | Steps | Success rate (GPT-4 class) | Success rate (7B models) |
|---|---|---|---|
| Information retrieval | 1-2 | ~96% | ~85% |
| Structured extraction | 2-3 | ~92% | ~72% |
| Research + synthesis | 4-6 | ~78% | ~45% |
| Multi-source analysis | 6-8 | ~63% | ~28% |
| Open-ended planning | 8+ | ~55% | ~15% |

The numbers vary by task complexity and model, but the trend is universal: longer horizons mean lower reliability.

**Design implication:** if you are building an agent that needs to accomplish a task requiring 10 steps, do not build a 10-step agent. Build two 5-step agents, verify the handoff between them, and treat the problem as two separate tasks. The extra architectural complexity pays for itself in reliability.

**The reliability ceiling.** There is a hard reliability ceiling for any LLM-based agent on a long-horizon task, and it is set by the per-step error rate and chain length. You cannot engineer around this ceiling through better prompting alone — you can only lower the per-step error rate (better model, grounding, verification) or shorten the chain (decomposition, checkpoints). If your task genuinely requires 20 steps with 5% per-step error, you will have a 64% failure rate regardless of how clever your prompting is. The math does not care about your prompt.

What does a 60% success rate mean in practice? It means that 4 out of 10 task executions require either human intervention or a retry from scratch. At 10,000 tasks/day, that is 4,000 daily interventions. At 5 minutes of human time per intervention, that is 333 hours per day. That is a full-time team of 40 people reviewing agent failures. At that scale, improving reliability from 60% to 75% is worth more than improving from 75% to 80% — the absolute reduction in interventions is larger.

This is why reliability engineering is not a nice-to-have for agents — it is the dominant cost driver for high-volume deployments.

## Model Capability Tiers and Their Agent Suitability

Choosing the right model tier for your agent is one of the most consequential architectural decisions you will make. Over-provisioning is expensive; under-provisioning causes failures that are hard to debug because the model does not error — it just produces confidently wrong output.

![Model tier vs task complexity fit matrix — green is optimal, amber is suboptimal, red is wrong choice](/imgs/blogs/llm-as-agent-brain-7.webp)

The matrix tells a clear story: there is a right model for each task type, and both over-provisioning and under-provisioning are costly. An overprovided frontier model extracting structured data from a form costs 50x more than a small model that does the same job correctly. An underprovided small model trying to plan a multi-hop research task will fail silently, wasting all the compute and producing wrong results.

Here is a practical framework for model selection:

**Simple tasks** (entity extraction, classification, format conversion): Small models (< 7B) are optimal. The task is well within their capability and the cost savings are massive. Use a small model here and save your budget for complex tasks.

**Moderate tasks** (single-step reasoning, simple QA with one tool call, instruction following): Medium models (7B-70B) are optimal. This is the sweet spot for most production workloads — cost-effective and capable enough for well-structured tasks.

**Complex tasks** (multi-hop reasoning, 3-5 step planning, complex tool orchestration): Large models (70B+) or the lower tier of frontier models. This is where the capability step function starts to matter.

**Expert tasks** (open-ended research, complex code generation, autonomous planning, self-correction): Frontier models only. Do not try to do this with smaller models — the failure modes are silent and expensive to debug.

One more consideration: **the consequences of failure.** A model that fails on a simple extraction task produces a missing value you can catch and retry. A model that fails on a complex planning task may have already executed irreversible actions (sent an email, modified a file, made an API call) before the failure is detected. Higher consequence actions warrant higher-tier models.

**Temperature and sampling in agents.** The model tier is one dimension; temperature is another. Temperature controls the randomness of the model's output. For tool calling and structured output, lower temperatures (0.0-0.2) produce more reliable and consistent results. For creative tasks (writing, brainstorming), higher temperatures (0.7-1.0) produce more varied output.

In agent contexts, the temperature choice has a non-obvious consequence: at higher temperatures, the model is more likely to recover from errors through creative problem-solving. At lower temperatures, the model is more likely to get stuck repeating the same failed approach. The practical recommendation: use low temperature for structured steps (tool calling, extraction, classification) and allow higher temperature for generative steps (drafting, reasoning, synthesis).

**Context caching for cost optimization.** Modern providers offer context caching that lets you pay once to process the static portion of your context (system prompt + tool schemas + few-shot examples) and reuse it across many inference calls. For a large system prompt, this can reduce input processing costs by 60-80%. If you are not using context caching for your agents, you are overpaying significantly. The tradeoff is that cached contexts cannot change between calls without incurring the full processing cost again.

## When to Use Smaller vs Larger Models in the Agent Loop

The most sophisticated production agent systems do not use a single model. They use model tiers strategically — a routing layer that dispatches tasks to the appropriate tier based on complexity.

**The case for smaller models in the loop:**

Cost is real. A 1,000-task-per-day agent using GPT-4 at $0.01/1K tokens and averaging 2,000 tokens per task costs $7,300 per year. The same workload on a hosted 70B model at $0.001/1K tokens costs $730. For extraction and classification steps that a 70B model handles as well as GPT-4, that 10x cost difference is pure waste.

Latency matters. Smaller models are faster. In interactive agents where the user is waiting, reducing the latency of intermediate steps can dramatically improve the experience. A small model that completes an extraction step in 200ms versus a frontier model that takes 2,000ms — that difference adds up over a 5-step chain.

Privacy and compliance. Some enterprise deployments cannot send data to cloud-hosted frontier models due to data residency requirements. A 13B model running on-premises may be the only legal option, even if it is less capable.

**The case for larger models in the loop:**

Silent failure is expensive. A cheaper model that fails 15% of the time on a task that costs $50 of engineering time to debug has an expected debugging cost of $7.50 per task — far more than the $0.001 saved on the model API call. The total cost of ownership includes failure cost.

Some tasks do not decompose. Not every multi-step task can be broken into simpler sub-tasks that smaller models can handle. Complex reasoning tasks require holding many constraints in mind simultaneously — that is a property of model scale that cannot be engineered around.

Quality signals matter. For tasks where the output quality is the primary value (a research report, a code review, a strategic analysis), the quality difference between a small model and a frontier model may be the entire value proposition. Cutting model quality here cuts the product value.

**The hybrid pattern (recommended for production):**

```
Router → Task classifier
├─→ Simple tasks → Small model (extraction, classification, formatting)
├─→ Moderate tasks → Medium model (single-step reasoning, QA)
├─→ Complex tasks → Large model (multi-step, tool orchestration)
└─→ Expert tasks → Frontier model (open-ended, high-stakes)
```

This pattern typically achieves 60-70% cost reduction versus using a frontier model for everything, while maintaining frontier-level quality for tasks that require it.

## The Prompt-as-Program Mental Model

There is a mental model that pays enormous dividends once you internalize it: **the system prompt is the source code; the user message is the runtime input.**

When you write a system prompt, you are programming the LLM. You are specifying its behavior, its constraints, its capabilities, and its default choices. The LLM does not have a stable personality or behavior independent of the prompt — it is a general-purpose execution engine that runs whatever specification you give it.

This reframing changes how you think about prompt engineering. "Writing a better prompt" becomes "writing better code." You start thinking about:

- **Specification completeness**: Have I specified behavior for all the inputs this agent will receive? What happens with edge cases?
- **Invariants**: What properties should always hold? Can I specify them explicitly?
- **Failure modes**: What happens when a tool returns an error? What happens when the user asks something out of scope?
- **Testability**: How do I test that the prompt behaves correctly across a representative sample of inputs?

![The prompt-as-program model — three inputs compose into an executable specification](/imgs/blogs/llm-as-agent-brain-9.webp)

The three inputs that compose the specification:

**System prompt (persona + rules)** is the most important. It defines identity, constraints, and defaults. A good system prompt specifies: what the agent is, what it can do, what it cannot do, what format outputs should take, and how to handle errors and edge cases. A bad system prompt is vague about any of these.

**Tool schemas (callable APIs)** are the capability declarations. Each schema says "I can do this." The schema must be precise — ambiguous schemas lead to the model calling the wrong tool or calling the right tool with wrong arguments. Think of tool schemas as function signatures: the name, the parameters, the types, and the description must all be accurate.

**Few-shot examples (behavior demonstrations)** are the most expensive per-token but the highest-value optional component. A few examples showing the agent successfully handling representative inputs gives the model a behavioral template to follow. Examples are particularly valuable for format adherence — showing exactly how the output should look is more reliable than describing the format in prose.

The practical implication: when your agent behaves wrongly, your first question should be "is this a specification failure?" Before debugging the model, before changing model tiers, examine whether the system prompt fully and accurately specifies the desired behavior. Most agent reliability issues are specification failures, not model capability failures.

**Prompt testing protocol:**

1. Define a set of representative test cases covering common inputs, edge cases, and adversarial inputs.
2. Run the agent on all test cases and categorize failures: specification failure (the prompt does not specify this case), capability failure (the model cannot do this regardless of prompting), or data failure (the retrieved context was wrong).
3. Fix specification failures first — they are free to fix and often account for 70%+ of failures.
4. Fix capability failures with better model selection or capability compensation (tools, grounding).
5. Fix data failures with better retrieval or context management.

**Version control your prompts.** The system prompt is code. It should be in version control. Changes should go through review. You should be able to roll back a prompt the same way you roll back a code deployment. This sounds obvious but most teams do not do it — they edit the system prompt in a config file and deploy without review. The result is prompt changes that silently break behavior in edge cases that were not tested.

**The prompt testing pyramid.** Borrow from software testing theory:

- **Unit tests** (fast, cheap): Test individual capabilities in isolation. Does the model correctly identify tool calls versus direct answers? Does it follow the output format? Run hundreds of these.
- **Integration tests** (medium speed/cost): Test complete task flows on representative inputs. Does the agent successfully complete a typical research task? Run dozens of these.
- **End-to-end tests** (slow, expensive): Test full user journeys on real-world edge cases. How does the agent handle ambiguous instructions? Conflicting tool results? Rate-limited APIs? Run a handful of these.

Most teams skip unit and integration tests and only notice failures in production. This is backwards — the earlier you catch a specification failure, the cheaper it is to fix.

## Failure Taxonomy: 5 Distinct Classes

Treating "the agent gave a wrong answer" as a single failure mode leads to wrong mitigations. There are five distinct failure classes, each with different causes, frequencies, and fixes.

![LLM failure taxonomy — five classes with different root causes and mitigations](/imgs/blogs/llm-as-agent-brain-8.webp)

**Class 1: Hallucination.**
The model generates plausible-sounding but factually incorrect content. This is the most discussed failure mode and for good reason — it is common (5-15% of factual claims in open-domain tasks for frontier models) and consequential (wrong facts propagate through agent chains).

Root cause: The model is a statistical pattern matcher. It generates the most likely continuation given the context, which is not the same as the most accurate continuation. For low-frequency or recent facts, the probability distribution is unreliable.

Mitigation: Ground all factual claims in retrieved evidence. Never trust the model's parametric knowledge for specific facts, numbers, dates, or names without external verification. Use RAG. Use tool calls to look up facts rather than asking the model to recall them.

**Class 2: Position Bias.**
The model systematically attends less to content in the middle of the context window. Instructions given in the middle of a long system prompt are ignored. Retrieved documents placed in the middle of a large context are underweighted.

Root cause: Attention mechanisms in transformers favor recent tokens (recency bias) and tokens near the beginning (primacy bias due to pretraining patterns). The middle gets lost.

Mitigation: Put critical instructions at the beginning and end of the system prompt. For long contexts, summarize or repeat key constraints at the end. Use chunking to break large retrieved contexts into shorter ones rather than passing a single massive block.

**Class 3: Repetition.**
The agent loops, re-executing steps it has already completed, or reiterating content without making progress. This can manifest as an infinite loop in a multi-step chain, or as a model that restates the same point in different words when asked to continue.

Root cause: The model's sampling process can enter high-probability cycles in the output distribution. Without an explicit loop detector, nothing stops the model from re-executing a step that looks like a good idea from the current context.

Mitigation: Explicit step tracking with the current step number and completed steps list in the context. A maximum step count with a hard stop. Sampling temperature adjustments when repetition is detected.

**Class 4: Context Overflow.**
The context window fills up. Early instructions and context are either dropped (for truncating context windows) or attended to less (for models with full-context attention but degraded performance on long inputs). The model "forgets" its original goal or early constraints.

Root cause: Context windows are finite. In a multi-step agent with retrieval, the context can easily grow to 50K-100K tokens per conversation. Even with a 200K-token context window, the position bias problem means that content 100K tokens ago is effectively invisible.

Mitigation: Active context management. Summarize completed steps rather than keeping the full transcript. Prune retrieved context that is no longer relevant. Consider a dedicated memory module (vector store + episodic memory) for long-running tasks.

**Class 5: Instruction Drift.**
Over a long multi-step task, the model's behavior gradually diverges from the original specification. It starts making choices that are internally consistent with its most recent actions but inconsistent with the original goal.

Root cause: In each step, the model is conditioning on the current context, which is dominated by recent turns. The original system prompt and goal specification are relatively diluted. The model optimizes for "what makes sense given everything that has happened" rather than "what achieves the original goal."

Mitigation: Periodically restate the original goal in the context. Use a "goal anchoring" pattern where the goal specification is repeated at each step. Build a task-tracking mechanism that explicitly checks whether each step advances the original objective. One practical pattern: include a short "current goal" field that gets updated at each step, reminding the model of the high-level objective. Agents that maintain explicit goal state drift significantly less than those that rely on the model to recall the goal from a long context.

One meta-point on failure classes: they interact. An agent suffering from context overflow (class 4) is more likely to hallucinate (class 1) because it cannot retrieve the grounding information it processed earlier. Instruction drift (class 5) is often triggered by position bias (class 2) when the original instructions fall into the middle zone of a long context. Treating these as independent problems leads to partial mitigations — the most reliable agents address all five classes systematically.

| Failure class | Frequency in production | Severity | Detection difficulty |
|---|---|---|---|
| Hallucination | High | High | Medium |
| Position bias | Medium | Medium | Low |
| Repetition | Low | Low | Low |
| Context overflow | Medium | High | Medium |
| Instruction drift | Medium | High | High |

**A note on failure detection.** The most dangerous failures are the ones that produce plausible-looking output. A model that produces obvious nonsense is easy to catch — validation fails, output is rejected, retry is triggered. A model that produces confidently wrong but plausible output passes validation, reaches the user, and may cause real-world harm before anyone notices.

This asymmetry means that your detection infrastructure needs to go beyond format validation. You need semantic validation — checking whether the output is internally consistent, whether it matches the input intent, whether the claimed facts can be verified. This is expensive to build but essential for high-stakes applications.

**Cascading failure signatures.** In a multi-step agent, different failure classes produce different cascading patterns:

- Hallucination in step 2 typically produces a semantically wrong but syntactically valid result in step 3, which produces an apparently confident but entirely made-up final answer.
- Position bias typically produces an agent that ignores constraints stated in the middle of a long system prompt, leading to outputs that are correct in isolation but violate the intended constraints.
- Repetition produces an agent that loops — you see the same tool call or the same text appear multiple times in the trace.
- Context overflow produces an agent that appears to "forget" earlier parts of the task — it finishes step 6 well but no longer references the goal specified in step 1.
- Instruction drift is the hardest to diagnose: the agent's outputs are reasonable responses to the most recent context but have subtly diverged from the original objective. The final output may look good but not actually serve the user's original need.

Building agent observability that logs full traces (not just final outputs) is the only reliable way to diagnose these cascading failure patterns. See [agent observability and tracing](/blog/machine-learning/ai-agent/agent-observability-and-tracing) for the infrastructure patterns.

## Multi-hop Reasoning: Where the Cliff Is

Multi-hop reasoning — chaining multiple inference steps where each step depends on the output of the previous — is the capability that most distinguishes frontier models from everything below.

![Multi-hop reasoning reliability by model family — smaller models collapse at 3 hops](/imgs/blogs/llm-as-agent-brain-10.webp)

The data is stark. At 1-hop reasoning (a single inference step), the capability gap between a frontier model (~98%) and a smaller model (~88%) is modest. At 2-hop, the gap widens. At 3-hop, smaller models fall below 50% — effectively coin-flip performance. At 4-hop, they are at 31%, which is worse than random guessing on most tasks.

Frontier models are not immune to multi-hop degradation — they fall from 98% at 1-hop to 74% at 4-hop — but they degrade more gracefully.

What counts as a "hop" in practice?

- A 2-hop task: "Who is the CEO of the company that makes [product]?" requires (1) identifying the company, then (2) finding the CEO.
- A 3-hop task: "What is the capital of the country whose currency is pegged to the euro that has the largest population?" requires (1) finding currencies pegged to the euro, (2) finding the largest-population country among them, (3) finding its capital.
- A 4-hop task: "What award did the author of [book] win that was later adapted into a film directed by the person who also directed [other film]?" — chains of references that each require a distinct lookup and inference step.

In agent contexts, hops correspond to sequential reasoning steps where the model must keep prior conclusions in mind. A research agent that identifies a company, finds its products, finds the company's main competitor, and then analyzes the competitor's strategy is executing a 4-hop chain.

**The practical implication:** when designing a multi-hop reasoning task, count the hops before choosing the model. If your task has 3+ hops, small models will fail most of the time. Budget for a frontier or large model and design the verification accordingly.

**Design rules from the data:**

1. Assume frontier models for anything requiring 3+ hops.
2. For 4+ hop chains, add intermediate verification steps to prevent error accumulation.
3. For smaller models, limit chains to 1-2 hops and use deterministic lookup tools for the rest.
4. When you find yourself designing a 5+ hop chain, stop and ask whether the chain can be decomposed into shorter verified sub-tasks.

**Why smaller models collapse at 3 hops.** The mechanism is attention dilution combined with insufficient world model depth. At 3+ hops, the model needs to simultaneously:

1. Remember the conclusion from hop 1.
2. Apply it correctly in hop 2.
3. Keep both hop 1 and hop 2 results active while computing hop 3.

For a frontier model with a rich internal representation of each concept, this is manageable. For a smaller model with a coarser internal representation, the concepts blur into each other. The model at hop 3 is reasoning with degraded representations of hops 1 and 2, and the errors compound.

This is not just about parameter count — it is about the quality of the concept representations encoded in the weights. Frontier models have seen enough training data to build rich, multi-faceted representations of most concepts. Smaller models have thinner representations that work for direct lookup but degrade under multi-hop composition.

**Chain-of-thought and multi-hop.** Chain-of-thought prompting — asking the model to "think step by step" and produce intermediate reasoning before the final answer — partially compensates for multi-hop degradation. By externalizing the reasoning chain into the context window, the model can "look back" at its prior conclusions rather than relying on internal representations. This is one reason CoT prompting improves multi-hop performance: it converts a parametric memory problem (remember what you concluded) into a context lookup problem (read what you wrote).

The implication: for any agent task requiring 2+ hops, use chain-of-thought explicitly. Do not let the model produce a final answer without producing visible intermediate reasoning. The intermediate reasoning serves both as a quality signal (you can evaluate whether the reasoning is sound) and as a reliability mechanism (the model can reference its prior conclusions instead of reconstructing them from degraded representations).

## Case Studies: When LLM Choice Was the Decisive Factor

### Case Study 1: GitHub Copilot's Model Routing

GitHub Copilot originally used a single model (Codex/GPT-3.5-class) for all tasks. The product team eventually built a routing layer that dispatches tasks to different models:

- **Simple completions** (completing a line of code): A small, fast model optimized for latency. Users experience this as "instant" completion.
- **Function-level generation** (writing a complete function from a docstring): A medium model that balances quality and speed.
- **Complex refactoring** (restructuring an entire module): A frontier model that can reason about architectural constraints.

The result: 65% reduction in inference cost with no user-perceived quality degradation. The key insight was that 80% of Copilot interactions are simple completions that do not require frontier model capability.

### Case Study 2: A Legal Research Agent's Hallucination Crisis

A legal research firm built an agent that used GPT-3.5 to search case databases, extract relevant precedents, and synthesize them into legal memos. In testing, the agent produced impressive-looking memos. In production, a partner noticed that several cited cases did not exist — the model had hallucinated plausible-sounding case names, judges, and holdings.

The fix was not to switch to a better model. The fix was to require every cited case to be verified against the actual database before inclusion in the memo. The model would propose citations; a deterministic lookup would verify them; unverifiable citations would trigger a re-generation. This reduced citation hallucination from 12% to 0.3%.

Lesson: hallucination is not primarily a model quality problem — it is an architecture problem. Add verification to the loop.

### Case Study 3: Customer Support Agent — 70B vs Frontier

A SaaS company was running a customer support agent using GPT-4. Monthly costs were $45,000. They ran an experiment: replace GPT-4 with a fine-tuned 70B model for the 70% of tickets that were classified as "routine" (password resets, billing questions, feature documentation requests).

The 70B model handled routine tickets at 94% user satisfaction versus 96% for GPT-4 — a 2 percentage point difference that was not statistically significant. Cost for the routine tier dropped from $31,500/month to $3,500/month. The remaining 30% complex tickets continued to use GPT-4.

Lesson: for high-volume, well-structured tasks, the quality difference between tiers is often smaller than the cost difference.

### Case Study 4: Research Agent Horizon Limit

A fintech startup built a research agent to analyze public company filings. The initial design had the agent plan and execute a 12-step analysis: download filing, extract financials, calculate ratios, compare to peers, identify anomalies, assess management discussion, check for red flags, compare to prior year, synthesize findings, draft summary, check for consistency, finalize.

The 12-step chain produced analyses that looked thorough but often contradicted themselves or lost track of the original question. The failure rate was 38% (required analyst review).

The redesigned system split the 12-step chain into three 4-step chains with human review gates between them. Each 4-step chain had a clear handoff specification. Failure rate dropped to 11%, analyst review time per report dropped by 60%, and the system could now provide a confidence estimate at each gate.

Lesson: long chains fail; verified sub-chains succeed.

### Case Study 5: Code Review Agent — Claude vs GPT-4

An engineering tools company tested Claude 3 Opus and GPT-4 for a code review agent. The agent was given a diff, asked to identify bugs, suggest improvements, and estimate effort.

Claude showed notably better performance on long diffs (> 400 lines) — likely due to better long-context coherence. GPT-4 showed better performance on short diffs with complex algorithmic reasoning. Both models failed equally often when the diff included unfamiliar internal patterns (the hallucination rate for unfamiliar codebases was similar).

The company deployed a hybrid: GPT-4 for short, algorithmically complex diffs (< 200 lines); Claude for long diffs (> 200 lines). Quality scores from engineer reviews improved 12% versus using a single model.

Lesson: model selection is not just about tier — different frontier models have different strength profiles.

### Case Study 6: Private Deployment with Llama

A healthcare company needed a clinical notes processing agent. HIPAA compliance requirements prohibited sending patient data to cloud APIs. The team fine-tuned Llama 3 70B on de-identified clinical notes and deployed it on-premises.

The fine-tuned 70B model achieved 89% accuracy on the extraction tasks (identifying diagnoses, medications, dates). A cloud-hosted GPT-4 baseline achieved 94%. The 5 percentage point gap translated to a $2.1M annual compliance cost savings versus building HIPAA BAA relationships with cloud providers, which justified the lower model quality.

Lesson: deployment constraints (privacy, latency, cost) often override pure capability considerations.

### Case Study 7: The 7B Extraction Specialist

A document processing company builds a pipeline that processes invoices, contracts, and receipts. The original pipeline used GPT-3.5 for everything. A team member noted that 90% of the processing volume was straightforward structured extraction — pull the vendor name, invoice number, date, line items.

They replaced the extraction steps with a fine-tuned 7B model, retaining GPT-3.5 for the exception handling (unusual formats, partial documents, conflicting fields). Fine-tuning cost: $3,200. Monthly API savings: $18,000. The fine-tuned model actually outperformed the general model on in-distribution extraction because the fine-tuning had specialized it perfectly.

Lesson: fine-tuned small models beat general frontier models on narrow, high-volume in-distribution tasks.

### Case Study 8: GPT-4 Turbo for Function Calling Reliability

An automation startup was building an agent that called 15 different tools. Early versions with GPT-3.5 had a tool-calling error rate of 12% — invalid JSON, wrong parameter types, calling tools that did not exist. Each error required a retry, doubling latency and cost for those calls.

They switched to GPT-4 Turbo for the planning and tool selection step (the "which tool and with which arguments" decision), while keeping GPT-3.5 for the post-processing step (interpreting tool results and formatting responses). Tool-calling errors dropped to 2.4%. Net effect: lower total cost (fewer retries) with better reliability.

Lesson: function calling accuracy varies dramatically across model tiers — budget for frontier models on the tool selection step.

## When to Use LLMs as Agent Brains (and When Not To)

### Use an LLM as agent brain when:

**The task is language-shaped.** If the core of the task involves understanding natural language input, generating natural language output, or both, an LLM is the right tool. No other technology handles unstructured text with comparable flexibility.

**The task requires generalization.** If you need the agent to handle a wide variety of task formulations, edge cases, and novel inputs without explicit programming, LLMs generalize better than any alternative. A rules-based system requires you to enumerate all cases; an LLM can handle cases you did not anticipate.

**The action space involves language.** Writing emails, drafting documents, generating code, summarizing content — if the agent's primary actions are language-producing, LLMs are ideal.

**You need flexible tool orchestration.** If the agent needs to dynamically select from a set of tools based on the task, LLMs handle this naturally. The tool schema + instruction-following mechanism is a reliable pattern for this.

**The problem is well within the training distribution.** Common business tasks (research, writing, coding, analysis) are well-represented in pretraining data. The model has seen thousands of examples of similar problems and learned good heuristics.

**You need to handle unexpected variations gracefully.** A customer support agent that handles 1,000 unique question types per day cannot be programmed for every variation. An LLM handles the long tail of edge cases with reasonable behavior even without explicit specification for each case — it generalizes from patterns learned during pretraining.

**The task benefits from reasoning about ambiguity.** When the right action depends on nuanced interpretation of the user's intent, an LLM's understanding of language context gives it a decisive advantage over rules-based systems. "Find me something similar to what I bought last time, but better" requires understanding "similar," "better," and user preference from context — a fundamentally language-understanding problem.

### Do not use an LLM as agent brain when:

**Exact computation is required.** If the task involves arithmetic, date calculations, financial computations, or any numerical precision, use a deterministic tool. Never trust the model's arithmetic. Add a calculator. Add a date library. The LLM should call the tool, not compute the answer.

**The decision requires real-time state.** LLMs have training cutoffs. Any task that requires knowing current market prices, live system state, or recent events must use tools that fetch real-time data. The model's parametric knowledge is stale.

**Formal correctness is required.** If the output must satisfy a formal specification (valid SQL, valid regular expression, valid code that passes a test suite), use a model plus a validator. Never assume the model produces formally correct output without verification.

**The cost-quality trade is not justified.** If the task is simple, structured, and high-volume, a rules-based system, a regex, or a fine-tuned small model is usually cheaper, faster, and more reliable than a frontier LLM. Do not use GPT-4 to check whether an email address contains an @ symbol.

**Auditability is required.** LLMs are opaque. If the task requires a traceable, auditable reasoning chain (legal decisions, financial approvals, medical recommendations), LLMs can assist but cannot be the sole decision-maker. Every LLM output in an audited context needs logging, review, and a human override path.

**Latency is under 100ms.** LLM inference takes 500ms to several seconds. If your agent needs to respond in under 100ms, the LLM cannot be in the critical path. Pre-compute, cache, or use a different architecture.

**The decision space is finite and enumerable.** If the agent must choose among a small set of well-defined actions based on well-defined conditions, a decision tree, a rules engine, or a classifier is more reliable, cheaper, and auditable. The power of LLMs is handling open-ended cases — not making binary decisions from structured inputs.

**Your failure modes are unacceptable.** Some failure modes are intolerable: a medical dosing agent that hallucinates a drug interaction is a patient safety issue, not a reliability metric. If the failure modes of statistical prediction are incompatible with your application's risk profile, LLMs must be supplementary (suggesting, flagging, assisting) rather than primary (deciding, acting, outputting). The LLM can surface options; a human or a deterministic system must make the final call.

| Criterion | Use LLM | Use alternative |
|---|---|---|
| Input type | Natural language | Structured/numerical |
| Output type | Natural language / code | Exact computation / formal |
| Generalization needed | Yes | No — narrow fixed task |
| Latency requirement | > 500ms acceptable | < 200ms required |
| Auditability | Logging sufficient | Formal audit trail required |
| Training distribution | Common business tasks | Specialized/novel domain |
| Task volume | Low to medium | Very high (>10K/day) |

## The Second-Order Consequences: What Emergent Planning Means for Teams

Understanding the emergent nature of LLM planning has second-order consequences beyond the technical. It changes how teams should be staffed, how the development process should work, and what success looks like.

**You cannot fully specify the system in advance.** With a traditional software system, you write the specification, implement it, and test against the specification. Behavior outside the specification is a bug. With an LLM agent, behavior is probabilistic. The same input can produce different outputs. Edge cases exist that you did not specify and cannot fully enumerate. The development process is more like iterative curriculum design than traditional software engineering — you shape behavior through data, prompts, and feedback rather than through deterministic logic.

**Your evaluation infrastructure matters more than your model choice.** In traditional software, you know when something is broken — the test fails, the exception is thrown, the assertion is violated. In LLM agents, you often do not know something is broken until a human reviews the output. Building evaluation infrastructure — automated metrics, human review workflows, trace analysis — is the highest-leverage investment you can make in agent reliability. Without it, you are flying blind.

**Failure is gradual, not binary.** A traditional API either works or throws an exception. An LLM agent degrades gracefully — as the task gets harder, the outputs get gradually worse until they cross the threshold from "acceptable" to "wrong." This gradual degradation is harder to detect in production than binary failure. You need evaluation metrics that track quality, not just success/failure.

**Retraining changes the system's behavior.** When you fine-tune a model on new data, you are not patching a specific bug — you are shifting the entire distribution of outputs. A fix to one failure mode may introduce regression in another. Fine-tuning requires the same care as a significant refactor of a large codebase. Treat model updates as production deployments, with regression testing and staged rollout.

**The business case must account for reliability costs.** When evaluating whether to build an LLM agent, the relevant costs are not just API costs — they are total costs including human review of failures, retry compute, debugging time, and the opportunity cost of failures that reach users. A $0.001/call model that fails 20% of the time may have a higher true cost than a $0.01/call model that fails 3% of the time, once you account for the cost of handling those failures.

**The organizational boundary matters.** LLM agents blur the boundary between "the model" and "the application." A failure in production could be a model quality issue, a prompt specification issue, a retrieval quality issue, or an infrastructure issue. Without clear ownership of each layer, failures get mis-attributed and mis-fixed. The team that owns the model (prompt engineering, fine-tuning) should be closely coupled to the team that owns the infrastructure (context management, tool reliability, evaluation), or the feedback loops needed to improve reliability will not close.

These organizational points might seem like implementation details, but in practice they determine whether a team can continuously improve their agent's reliability or whether they plateau at the first stable-enough version. The most capable teams treat agent development as a continuous improvement loop: deploy, measure, identify failure modes, improve, repeat.

---

The LLM as agent brain is not a designed planning system — it is an emergent approximation of planning, learned from a vast corpus of human problem-solving. That framing matters because it changes what you expect and what you build around it. You expect capability that is strong in the training distribution and fragile at the edges. You build verification, grounding, and decomposition to compensate for the fragility.

The engineers who build reliable agents are not those who trust the LLM most. They are the ones who understand its failure modes most precisely and engineer their way around each one.

From here, the next logical steps are understanding [how the full agent loop works in detail](/blog/machine-learning/ai-agent/agent-loop-anatomy) — the mechanics of what happens inside a single agent turn from prompt assembly to tool dispatch. Going deeper on [specific reasoning patterns](/blog/machine-learning/ai-agent/react-pattern-deep-dive) shows how frameworks like ReAct extract reliable planning from statistical prediction engines by making the reasoning chain explicit. For a broader systems perspective, [building effective agents hands-on guide](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide) covers the engineering patterns that make production agents reliable at scale.

The core message is this: LLMs are powerful agent brains precisely because they generalize, and fragile agent brains precisely because that generalization is statistical. Every architectural decision you make around an LLM agent is either compensating for the statistical fragility (verification, grounding, decomposition) or leveraging the statistical generalization (flexible tool use, open-domain reasoning, natural language understanding). Understanding which side of that tradeoff you are on for each decision is the mark of a principal engineer building production agents.

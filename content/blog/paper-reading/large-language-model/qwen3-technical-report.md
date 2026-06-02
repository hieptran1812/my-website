---
title: "Qwen3 Technical Report: One Model, Two Minds"
date: "2026-05-17"
publishDate: "2026-05-17"
description: "A close read of the Qwen3 technical report: how a single dual-mode model, a thinking budget, a four-stage post-training pipeline, and strong-to-weak distillation replace the usual two-model reasoning stack."
tags: ["qwen3", "large-language-model", "reasoning", "mixture-of-experts", "reinforcement-learning", "distillation", "post-training", "thinking-budget", "paper-reading"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: false
readTime: 30
---

For most of 2024 and early 2025, shipping a strong open-weight model meant shipping *two* of them. You trained an instruction-tuned model for chat, and you trained a separate reasoning model — DeepSeek-R1, QwQ, o1-style — that spent thousands of tokens thinking before it answered. The two had different latency profiles, different serving configs, different prompt templates, and different failure modes. Picking the wrong one for a request was a real production decision, and getting it wrong cost you either accuracy or money.

The Qwen3 technical report ([arXiv:2505.09388](https://arxiv.org/abs/2505.09388)) is the Qwen team's argument that this split was an accident of training pipelines, not a law of nature. Qwen3 ships a single model that holds both behaviors and switches between them with a token in the prompt. It adds a *thinking budget* so you can dial reasoning effort up and down at inference time without retraining. And it builds the whole 0.6B-to-235B family by post-training only the two largest models properly and distilling everything else.

![How Qwen3 turns into a model family](/imgs/blogs/qwen3-technical-report-1.png)

The diagram above is the mental model for the entire report: one corpus, one pre-training run per size, a four-stage post-training pipeline applied only to the flagship, and then *strong-to-weak distillation* to manufacture the rest of the lineup at roughly a tenth of the GPU cost. This post walks the report section by section — architecture, data, the thinking budget, the post-training pipeline, distillation, the benchmark numbers — and then takes the senior-engineer lens to what is load-bearing and what is marketing.

> [!tldr] TL;DR
> - **One model, two modes.** Qwen3 unifies "thinking" (long chain-of-thought) and "non-thinking" (direct answer) into a single set of weights, switched by `/think` and `/no_think` tags in the prompt or chat template.
> - **Thinking budget.** Inference-time control: cap the reasoning trace at *N* tokens, and the model is forced to stop reasoning and answer from whatever partial CoT it has. Accuracy scales smoothly with the budget.
> - **Four-stage post-training.** Long-CoT cold start (SFT) → reasoning RL (GRPO) → thinking-mode fusion (SFT) → general RL. Reasoning is built in isolation first, *then* fused with chat behavior.
> - **Strong-to-weak distillation.** Smaller models skip the four-stage pipeline entirely; they distill the flagship's outputs and logits at roughly **1/10 the GPU hours**, and still beat same-size predecessors.
> - **Scale.** 36T pre-training tokens across **119 languages**; the flagship Qwen3-235B-A22B is an MoE with 128 experts, 8 active, 22B activated of 235B total.
> - **Where it's thin.** The mode-fusion step is under-ablated, "thinking budget" accuracy curves are reported on a narrow slice of benchmarks, and the distillation comparison holds teacher quality fixed in a way that flatters the method.

## Context: what came before

To see why Qwen3's "one model" framing is a real contribution and not a packaging trick, you have to remember the state of open-weight models when it landed.

The Qwen2.5 generation (late 2024) was a conventional family: a base model, an instruction-tuned chat model, and specialized variants (Coder, Math). Reasoning, in the modern test-time-compute sense, was not a first-class capability. You got chain-of-thought if you prompted for it, but the model was not *trained* to spend a controllable amount of compute reasoning before answering.

Then DeepSeek-R1 reframed the field. R1 showed that large-scale reinforcement learning on verifiable problems — math with checkable answers, code with unit tests — could induce genuine long-form reasoning, and that the resulting traces could be distilled into smaller models. We covered the mechanics of that in [DeepSeek-R1: incentivizing reasoning via RL](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning). The catch: R1 was a *separate model* from DeepSeek-V3. The reasoning capability lived in different weights from the chat capability.

This produced an awkward deployment story across the whole industry. You had a "fast" model and a "smart" model. The fast model was cheap but failed hard on multi-step problems. The smart model burned thousands of tokens — and dollars — even on "what's the capital of France." Routers and classifiers sprang up to dispatch requests between them, which is its own engineering tax (see [adaptive LLM routing under budget constraints](/blog/paper-reading/large-language-model/adaptive-llm-routing-under-budget-constraints) for how deep that rabbit hole goes).

The other inherited problem was the *family*. Once you commit to RL-heavy post-training, applying it to every size in your lineup — 0.6B, 1.7B, 4B, 8B, 14B, 32B, plus MoE variants — is brutally expensive. RL needs rollouts, rollouts need a verifier loop, and the loop does not parallelize cheaply. Most labs either post-trained only the big models well and shipped weaker small ones, or distilled reasoning traces from a teacher in an ad-hoc way.

There is also a subtler cost asymmetry worth naming. The two-model world is not just inconvenient — it is *quadratically* inconvenient. Every capability you add (a new language, a tool-use skill, a safety patch) has to be added twice, validated twice, and regression-tested twice, and the two copies inevitably drift. The chat model gets a fix the reasoning model does not. A prompt that works in one degrades in the other. Anyone who has maintained parallel model lines knows the real expense is not the GPU hours; it is the human attention spent keeping two artifacts in sync. Collapsing them into one set of weights is, before anything else, an *operational* simplification.

Qwen3's stated gap, then, is two-fold. First: collapse the fast/smart split into one model with a *runtime* knob instead of a *model-selection* knob. Second: make the family economically coherent — post-train the flagship once, propagate the capability to every smaller model by distillation rather than by re-running RL. The report is the case that both are achievable without giving up benchmark position. The interesting question is not whether the unified model exists — it ships — but how much the unification *costs* in per-mode quality, and the report is more confident on that question than its ablations strictly support.

## Contributions

The report's contributions, tightened from the authors' framing:

1. **A unified dual-mode model.** Thinking and non-thinking behavior live in one set of weights. Mode is selected per-request by chat-template tags (`/think`, `/no_think`), not by loading a different checkpoint. This is the headline.
2. **The thinking budget.** An inference-time mechanism to bound the reasoning trace at an arbitrary token count. Accuracy degrades gracefully as the budget shrinks, giving a smooth compute-vs-quality dial that needs no retraining.
3. **A four-stage post-training pipeline.** Long-CoT cold start, reasoning RL, thinking-mode fusion, and general RL — explicitly ordered so reasoning is established before it is fused with general chat behavior.
4. **Strong-to-weak distillation.** The flagship and the largest MoE act as teachers; the remaining six dense sizes and the small MoE are produced by off-policy and on-policy distillation at roughly **1/10 the GPU hours** of the full pipeline, while still surpassing the previous generation at equal size.
5. **Scale and multilinguality.** Pre-training on ~36T tokens covering **119 languages and dialects** — roughly triple the language coverage of Qwen2.5 — with the flagship Qwen3-235B-A22B reported to beat DeepSeek-V3-Base on 14 of 15 benchmarks at about one-third the total parameters.

The rest of this post takes each in turn.

## Method

### Architecture

Qwen3 is, deliberately, not an architecture paper. The model is a standard decoder-only Transformer, and the team's choices are conservative. That is itself a signal: the report is betting that *data and post-training*, not architectural novelty, are where the gains are.

There are eight models: six dense (0.6B, 1.7B, 4B, 8B, 14B, 32B) and two mixture-of-experts (Qwen3-30B-A3B and the flagship Qwen3-235B-A22B). The `A` number is *activated* parameters per token — Qwen3-235B-A22B has 235B total parameters but routes only 22B of them for any given token.

![Dense block vs MoE block in Qwen3](/imgs/blogs/qwen3-technical-report-2.png)

The figure above is the architectural claim in one picture: **the dense and MoE models share an identical attention stack and differ only in the feed-forward layer.** Both use:

- **Grouped-Query Attention (GQA).** Multiple query heads share a smaller number of key/value heads, shrinking the KV cache without the quality hit of full multi-query attention. If GQA is unfamiliar, our [KV-cache deep dive](/blog/machine-learning/large-language-model/kv-cache) explains why the KV cache, not the parameter count, is what bounds your serving batch size.
- **SwiGLU** feed-forward activations.
- **Rotary Position Embeddings (RoPE)** — the same relative-position scheme analyzed in [RoFormer: rotary position embedding](/blog/paper-reading/large-language-model/roformer-enhanced-transformer-with-rotary-position-embedding).
- **RMSNorm with pre-normalization.**

Two changes from Qwen2.5 are worth naming because they are about *training stability*, not capability:

**QKV-bias is removed.** Qwen2 added a bias term to the query, key, and value projections. Qwen3 drops it.

**QK-Norm is added.** Before the attention dot product, the query and key vectors are passed through RMSNorm. The motivation is the well-documented failure mode where attention logits grow without bound deep into a long training run, the softmax saturates, gradients vanish, and the loss curve develops a slow upward drift or an outright spike. Normalizing $q$ and $k$ before the dot product caps the logit magnitude:

$$
\text{logit}_{ij} = \frac{\text{RMSNorm}(q_i) \cdot \text{RMSNorm}(k_j)}{\sqrt{d_h}}
$$

where $d_h$ is the per-head dimension. This is the kind of change you make after you have watched a 235B run diverge at 80% completion and had to roll back a week of compute. It buys nothing on a benchmark table and everything on whether the run finishes.

The MoE models use **128 experts with 8 activated per token** and drop the shared-expert design some earlier MoEs used. The router is trained with **global-batch load balancing**: the balancing loss is computed across the whole global batch rather than per-device, which the report argues encourages genuine expert *specialization* instead of the uniform, interchangeable experts a per-device loss tends to produce. We unpack why load balancing is the central difficulty of MoE training in [optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) and [MoE LLM architecture, training and finetuning](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies).

The economic argument for the flagship is the activation ratio. Qwen3-235B-A22B activates ~22B of 235B — roughly 9% — per token. You pay for 235B of memory (and the bandwidth to stream those weights) but only ~22B of FLOPs per token. That is the trade MoE always offers: capacity is cheap, but the *capacity has to fit in memory*, and the all-active dense models in the family exist precisely for deployments where it does not.

It is worth making the memory number concrete, because it is the one that decides whether you can serve a model at all. At FP16, 235B parameters is ~470 GB just for weights — four 80 GB H100s before you have allocated a single KV-cache block. The activated-parameter count does *not* shrink this; routing 22B per token still requires all 235B resident, because any token might route to any expert. This is why the dense 32B exists alongside the MoE 30B-A3B: the 30B-A3B is *cheaper per token* (3B activated) but needs ~60 GB resident, while the dense 32B needs ~64 GB resident and activates all 32B. On a single 80 GB card, the dense 32B is often the *easier* deployment despite being "bigger" in FLOPs, because there is no router, no expert-parallel sharding, and no load-balancing tail latency. The report shipping both is a recognition that "which model is cheaper" has no answer independent of your hardware — exactly the point we labor in [choosing a GPU for LLM serving](/blog/machine-learning/large-language-model/choosing-gpu-for-llm-serving-cost-throughput-latency).

One more architectural detail the report is quiet about but that matters for long-context users: with GQA, the KV cache per token is proportional to the *number of KV heads*, not query heads. For a 32K-context request, the KV cache can easily exceed the activated-weight footprint of a small model. The thinking mode makes this worse — a 4,000-token reasoning trace is 4,000 tokens of KV cache that exist only to be discarded once `</think>` closes. That interaction between thinking mode and KV pressure is, quietly, one of the strongest practical arguments *for* the thinking budget: every token you do not spend reasoning is a KV-cache block you do not allocate.

### Three-stage pre-training

Pre-training runs on approximately **36 trillion tokens** spanning **119 languages and dialects**. Two data-pipeline details in the report are worth flagging because they reflect where frontier data work has moved:

- **Vision-model PDF extraction.** A substantial chunk of the corpus is text recovered from PDFs using Qwen's own vision-language models to do layout-aware extraction. Naive PDF-to-text mangles tables, equations, and multi-column layouts; a VLM that "reads" the page recovers far more usable, high-quality text.
- **Synthetic data from specialist models.** Earlier Qwen Math and Qwen Coder models are used to *generate* additional STEM, math, and code data — synthetic text that is then filtered back into the pre-training mix.

The corpus is not consumed uniformly. Pre-training is a three-stage curriculum:

![Three-stage pre-training curriculum](/imgs/blogs/qwen3-technical-report-3.png)

**Stage S1 — General (>30T tokens, 4,096 context).** The bulk of pre-training. Broad coverage across languages and domains at a modest 4K sequence length. This builds the base language model: world knowledge, grammar, the long tail of facts.

**Stage S2 — Reasoning (~5T tokens, 4,096 context).** The data mix is deliberately shifted toward STEM, code, and reasoning-heavy text, and the learning-rate schedule is adjusted to make these tokens count. The bet is that reasoning ability is partly a *pre-training* property — you can prime the base model for it before any RL touches it — rather than something RL must conjure from nothing.

**Stage 3 — Long-context (hundreds of billions of tokens, 32,768 context).** Only at the very end is the sequence length stretched to 32K. RoPE frequencies are adjusted to cover the longer range. Long-context training is expensive — attention is quadratic in sequence length — so doing it last, on a small fraction of tokens, is a pure cost decision. The model learns *language* on cheap short sequences and learns to *use a long window* on expensive long ones.

The curriculum logic is the same one we discuss in [pre-training, mid-training, and RL interplay](/blog/paper-reading/large-language-model/pre-training-mid-training-and-rl-interplay): spend cheap compute on broad capability, spend expensive compute narrowly and late.

There is a second-order reason the reasoning stage (S2) sits *inside* pre-training rather than being deferred entirely to post-training. RL post-training can only amplify behaviors the base model can already produce with non-trivial probability — it reshapes the policy, it does not invent capabilities from zero. If the base model essentially never emits a correct multi-step derivation, the RL reward is sparse to the point of uselessness and exploration stalls. Stage S2 raises the *base rate* of correct reasoning traces so that when reasoning RL starts, there is a signal to climb. This is the same dependency that [does RL really incentivize reasoning beyond the base model](/blog/paper-reading/large-language-model/does-reinforcement-learning-really-incentivize-reasoning-capacity-in-llms-beyond-the-base-model) interrogates: RL is a sharpening operation, and S2 is what it sharpens.

The report also describes a *scaling-law-guided* hyperparameter process — fitting predictive curves on small runs to choose learning rate and batch size for the large ones — which is now table stakes for any run where a single failed configuration costs millions of dollars. The practical shape of this: you train a sweep of small models (say 0.6B through 4B), fit a power law of optimal learning rate and batch size against parameter count and token count, and *extrapolate* to the 235B configuration you only get to run once. Combined with the QK-Norm stability fix, this is the report's whole stance on de-risking the flagship run — predict the hyperparameters, cap the attention logits, and do not improvise at 235B scale.

The 119-language coverage deserves a note because it is not free. Tripling language coverage versus Qwen2.5 means the tokenizer must allocate vocabulary to scripts and morphologies it previously under-served, and every additional language competes for a slice of a fixed 36T-token budget. The report frames this as a capability win, and for low-resource languages it is — but there is an unstated tradeoff with English and Chinese token density that the benchmark tables, being mostly English, do not surface. We dig into exactly this tension in [training an LLM to adapt to a new language](/blog/machine-learning/large-language-model/training-llm-adapt-new-language).

### The thinking budget

Here is the part of the report that is genuinely a product feature, not just a training detail.

A Qwen3 model in thinking mode emits a reasoning trace wrapped in `<think>` ... `</think>` before its final answer. Left alone, the model decides for itself when the trace is complete and emits `</think>`. The **thinking budget** intervenes in that decision.

![The thinking budget mechanism](/imgs/blogs/qwen3-technical-report-4.png)

The mechanism is shown above. You set a budget — say 1,024 reasoning tokens. The inference harness counts tokens inside the `<think>` block. Two cases:

- **Within budget.** The trace stays under the cap. The model emits `</think>` on its own and answers from a complete chain of thought. Identical to default thinking-mode behavior.
- **Budget exceeded.** The trace reaches the cap. The harness *injects* a stop instruction — effectively forcing `</think>` into the stream — and the model must produce its final answer from whatever partial reasoning it has accumulated.

The non-obvious result the report reports: accuracy scales *smoothly and monotonically* with the budget. More budget, more accuracy, with diminishing returns — not a cliff where a truncated trace produces garbage. That smoothness is what makes the budget a usable knob. It means the model was trained such that partial reasoning is still *coherent* reasoning, and the forced stop does not derail it into an incoherent answer.

Why this matters operationally: it converts a model-selection decision into a per-request scalar. A latency-sensitive autocomplete call sets budget 0 (non-thinking). A "summarize this email" call sets a small budget. A competition-math query sets a large one. Same weights, same endpoint, same KV cache layout — only the budget changes. Compare that to maintaining a fast model and a smart model behind a router.

Here is roughly what the harness-side control loop looks like. This is not Qwen-internal code; it is the shape of what you implement on top of any inference server that lets you inspect and steer the token stream:

```python
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
FORCED_STOP = (
    "\nConsidering the limited time, I have to give the "
    "answer now.\n" + THINK_CLOSE + "\n"
)

def generate_with_budget(model, prompt, thinking_budget, max_new=4096):
    """Cap the reasoning trace at `thinking_budget` tokens."""
    tokens = model.tokenize(prompt + THINK_OPEN)
    think_tokens = 0
    in_think = True

    for _ in range(max_new):
        next_tok = model.sample_next(tokens)        # one decode step
        piece = model.detokenize([next_tok])

        if in_think:
            think_tokens += 1
            if piece.strip() == THINK_CLOSE:        # model stopped on its own
                in_think = False
            elif think_tokens >= thinking_budget:   # budget hit: force the stop
                tokens += model.tokenize(FORCED_STOP)
                in_think = False
                continue

        tokens.append(next_tok)
        if next_tok == model.eos_id:
            break

    return model.detokenize(tokens)
```

The two failure modes to watch when implementing this yourself: (1) injecting the forced stop *mid-word* if your tokenizer splits the boundary awkwardly — always inject on a clean token boundary; (2) setting the budget so low the model has not yet "decided" anything, which is the regime where the smooth-degradation property is weakest.

It helps to think about the budget as carving the accuracy-vs-cost curve into operating points rather than as a single setting. Roughly, the regimes look like this — the exact numbers depend on the task, but the *shape* is what the report claims is robust:

| Budget (think tokens) | Regime | What you get | When to use |
|---|---|---|---|
| 0 | Non-thinking | Direct answer, no `<think>` block | Latency-critical, easy queries, autocomplete |
| 256–512 | Shallow reasoning | A few reasoning steps, forced close | Routine multi-step tasks, classification with rationale |
| 1,024–2,048 | Working reasoning | Most problems get a complete trace | General assistant traffic, code, analysis |
| 4,096–8,192 | Deep reasoning | Long derivations, near-natural termination | Competition math, hard proofs, agentic planning |

The key property the report sells is that moving *down* this table costs accuracy *smoothly* — there is no budget below which the answer collapses into noise. That is a trained property, not a free one: a model that had only ever seen complete reasoning traces during training would, when truncated, often emit an answer that contradicts its own half-finished reasoning. Stage 3 of post-training (below) is what installs the graceful behavior.

A worked example makes the operational value concrete. Suppose you run an assistant with mixed traffic: 70% easy ("rephrase this", "what does this error mean"), 25% moderate ("write this function", "explain this diff"), 5% hard ("prove this", "debug this race condition"). In the two-model world you run a router, eat its latency and its misclassification rate, and keep two models hot. With Qwen3 you keep *one* model hot and emit budgets 0 / 1,024 / 8,192 by traffic class. The misrouted-easy-query-to-smart-model cost — thousands of wasted reasoning tokens — becomes, in the worst case, a slightly-too-large budget that the model simply does not spend, because within-budget traces terminate naturally. The failure mode is bounded in a way the router's is not.

### Four-stage post-training

The flagship models — Qwen3-235B-A22B and Qwen3-32B — go through a four-stage post-training pipeline. The ordering is the point.

![Four-stage post-training pipeline](/imgs/blogs/qwen3-technical-report-5.png)

**Stage 1 — Long-CoT cold start (SFT).** Supervised fine-tuning on a curated set of long chain-of-thought examples spanning math, code, logic, and STEM, every example carrying a verified answer. The data is filtered by **rejection sampling**: candidate traces are generated, only those reaching the correct answer are kept, and even among correct ones the set is pruned to avoid the model overfitting to a single trace style. The goal here is explicitly *not* peak performance — it is to give RL a sane starting policy. A cold RL start from a base model wastes enormous compute exploring obviously-wrong reasoning formats; the cold-start SFT front-loads the *format* so RL can spend its budget on *correctness*.

**Stage 2 — Reasoning RL.** Reinforcement learning with **GRPO** (Group Relative Policy Optimization) on verifiable problems. GRPO drops the separate value network of PPO and instead normalizes each response's reward against the mean reward of a *group* of responses sampled for the same prompt — substantially cheaper for the rollout-heavy RL that reasoning training demands. We walk through GRPO's mechanics and its sharp edges in [fine-tuning an LLM with GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo).

The report's striking detail: the reasoning-RL stage uses only **3,995 query-verifier pairs**, and on that small set Qwen3-235B-A22B's AIME'24 score climbs from 70.1 to 85.1. The leverage is real — but note *what* makes it work. Every pair has a programmatic verifier (a math-answer checker, a code unit-test runner). The reward is not a learned reward model that can be hacked; it is ground truth. RL is sample-efficient here precisely because the reward signal is clean. This is the same lesson as [JustRL: scaling a 1.5B LLM with a simple RL recipe](/blog/paper-reading/large-language-model/justrl-scaling-a-1-5b-llm-with-a-simple-rl-recipe) — recipe simplicity buys you nothing without a trustworthy reward.

**Stage 3 — Thinking-mode fusion.** This is the stage that makes "one model" true. The reasoning-RL'd model from Stage 2 is a thinking-only specialist. Stage 3 is an SFT pass on a dataset that *mixes* thinking-mode and non-thinking-mode data, using a chat template designed to make the mode explicit. The model learns to associate `/think` with producing a `<think>` trace and `/no_think` with answering directly — and, critically, learns to do both well from one weight set. The thinking budget is *also* trained here: by exposing the model to traces that are cut short, Stage 3 is what produces the graceful-degradation property the budget mechanism relies on.

**Stage 4 — General RL.** A final RL stage over **20+ task types** — instruction following, format adherence, tool use, agent behaviors, safety — using a mix of programmatic checks and reward models. This is the broad alignment pass that turns a reasoning-capable model into a usable assistant. It is the stage most exposed to reward hacking, since not every one of those 20+ tasks has a clean verifier.

A useful way to read the four stages is by *what kind of signal* each one trusts. Stage 1 trusts curated human-or-teacher trajectories — high quality, low coverage, no exploration. Stage 2 trusts programmatic verifiers — perfect reward, narrow domain, heavy exploration. Stage 3 trusts a hand-designed data mixture — it is the only stage whose job is *integration* rather than *capability*, and it deliberately does not push any single benchmark up. Stage 4 trusts a heterogeneous bag of checkers and reward models — broad coverage, imperfect reward, the most alignment-shaped of the four. Ordering them this way is not arbitrary: each stage hands the next a policy that is *already in the right behavioral basin*, so the next stage's noisier signal cannot pull it far. Run Stage 4's messy reward signal directly on a base model and you would get reward hacking and format collapse; run it after Stages 1–3 and it can only nudge an already-competent policy. The pipeline is a sequence of decreasing signal quality applied to a policy of increasing robustness — and that pairing is the actual design.

There is a real risk the report does not dwell on: *catastrophic forgetting across stages*. Stage 2's reasoning RL can erode instruction-following; Stage 3's fusion SFT can blunt the sharp reasoning Stage 2 installed; Stage 4 can regress both. Each stage is, in effect, betting that the previous stage's gains are robust enough to survive the next stage's optimization pressure. The report's benchmark numbers suggest the bet pays off, but the *intermediate* checkpoints — what Qwen3 looked like after Stage 2 versus after Stage 3 — would tell you how much was given back, and those are not shown.

The chat template that carries the mode switch looks roughly like this — note that it is a *template* change, not a model change:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def chat(user_msg, thinking: bool):
    messages = [{"role": "user", "content": user_msg}]
    # enable_thinking toggles whether the template primes a <think> block
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,          # the entire mode switch
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=2048)
    return tok.decode(out[0][inputs.input_ids.shape[1]:])

fast   = chat("What is 17 * 23?", thinking=False)   # direct answer
smart  = chat("What is 17 * 23?", thinking=True)    # think block, then answer
assert fast != smart                                # same weights, different mode
```

You can also flip mode inline with `/think` and `/no_think` tokens inside a user turn, which is what makes multi-turn control possible — a long conversation can think on the hard turns and answer directly on the easy ones.

### Strong-to-weak distillation

The four-stage pipeline is expensive — Stages 2 and 4 are RL, with rollout loops that do not parallelize cheaply. Running it eight times, once per model size, would dominate the entire project budget. So Qwen3 runs it *twice* — for Qwen3-235B-A22B and Qwen3-32B — and produces the other six models by distillation.

![Strong-to-weak distillation](/imgs/blogs/qwen3-technical-report-6.png)

The flagship is the teacher. Distillation happens in two modes, and the report uses both:

**Off-policy distillation.** The teacher generates responses — across both thinking and non-thinking modes — and the student is trained, ordinary SFT-style, to imitate those outputs. The student learns from text the *teacher* chose to produce. This is straightforward and gets the student into the right behavioral neighborhood, but it suffers the standard exposure-bias problem: the student is never trained on its own mistakes, only on the teacher's clean trajectories.

**On-policy distillation.** The student generates its *own* responses, and for each token the loss is the KL divergence between the student's output distribution and the teacher's distribution over the *same* student-generated prefix:

$$
\mathcal{L}_{\text{on}} = \mathbb{E}_{x \sim \pi_{\text{student}}}\big[\, D_{\text{KL}}(\pi_{\text{teacher}}(\cdot \mid x) \,\|\, \pi_{\text{student}}(\cdot \mid x)) \,\big]
$$

Because the prefixes are sampled from the *student's* own policy, the teacher is effectively correcting the student exactly where the student tends to go wrong. This is the difference between learning from a textbook and learning from a tutor who watches you work. We go deeper on the off-policy/on-policy distinction and its tradeoffs in [distillation in LLMs](/blog/machine-learning/large-language-model/distillation-in-llm).

The reported payoff: the distilled models reach the **~1/10 GPU-hours** figure relative to running the full four-stage pipeline, and — the part that matters — they *also* inherit the thinking-mode switch and the thinking-budget controllability. The student does not just get a number on a benchmark; it gets the *feature*. And the distilled small models still beat their Qwen2.5 same-size predecessors: the report notes Qwen3-8B surpassing Qwen2.5-14B on more than half the benchmarks despite having ~43% fewer parameters.

In practice the report uses off-policy distillation as a *warm-up* and on-policy distillation as the main course. The off-policy pass cheaply moves the student into the right region of behavior space — correct formats, the `<think>` convention, plausible reasoning shapes — so that when on-policy distillation starts, the student's own samples are already good enough that the teacher's per-token corrections are informative rather than overwhelming. Starting on-policy distillation from a cold student wastes the teacher's signal: the student's outputs are so far off-distribution that the KL gradient is dominated by gross errors and never gets to the fine corrections that matter. The ordering is the same intuition as the cold-start-before-RL ordering in the four-stage pipeline — establish the *format* cheaply, then spend the expensive signal on *quality*.

This is the quiet structural contribution of the report. The unified dual-mode model is the headline, but the thing that makes the *family* economically possible is that the dual-mode capability is *distillable*. If it were not — if you had to run RL on every size to get the thinking switch — the "one model, two modes" story would not scale below the flagship. It is also worth noticing what distillation does to *consistency*: because every small model is taught by the same flagship, the family behaves coherently. A prompt that elicits a careful, hedged answer from Qwen3-235B tends to elicit a recognizably similar (if weaker) answer from Qwen3-1.7B. In the two-model, train-each-size-separately world, that family coherence is something you hope for; here it is a structural consequence of the teacher being shared.

## Experiments

The headline comparison is the flagship base model against DeepSeek-V3-Base. Numbers below are as reported in the technical report; treat them as the authors' framing, not an independent reproduction.

| Model | Total params | Activated | AIME'24 | Reported benchmark wins |
|---|---|---|---|---|
| DeepSeek-V3-Base | 671B | 37B | — | baseline |
| **Qwen3-235B-A22B-Base** | 235B | 22B | — | 14 of 15 vs DeepSeek-V3-Base |
| Qwen3-235B-A22B (post-RL, thinking) | 235B | 22B | **85.1** (from 70.1) | — |
| Qwen3-32B (dense) | 32B | 32B | competitive w/ prior 70B-class | — |
| Qwen3-8B | 8B | 8B | beats Qwen2.5-14B on >50% | — |

The claims that carry weight:

- **Parameter efficiency at the top.** Qwen3-235B-A22B-Base beating DeepSeek-V3-Base on 14 of 15 benchmarks at roughly one-third the total parameters (and well under two-thirds the activated parameters) is a strong result *if the benchmark set is representative*. Fifteen benchmarks is a reasonable spread, but base-model benchmark suites are exactly where contamination and selection effects hide.
- **The 3,995-pair RL jump.** AIME'24 70.1 → 85.1 from a 3,995-pair reasoning-RL stage is the single most quoted number. It is real, and it is *conditional*: it works because every pair has a programmatic verifier. Do not read it as "RL is cheap." Read it as "RL is cheap when your reward is ground truth."
- **Small-model uplift.** Qwen3-8B > Qwen2.5-14B on a majority of benchmarks is the evidence that distillation actually transfers capability rather than just style.

What is load-bearing in the setup, and might not transfer to your own work:

1. **Verifiable-reward RL.** The reasoning-RL efficiency depends entirely on having checkable answers. If your domain's "correctness" needs a human or a learned reward model, you are back in expensive, hackable-reward territory and the 3,995-pair magic evaporates.
2. **A genuinely strong teacher.** Strong-to-weak distillation is only as good as the flagship. The 1/10-cost figure assumes the teacher already exists and is excellent — its cost is simply not counted in the student's budget.
3. **The 36T-token corpus.** The pre-training reasoning priming (Stage S2) assumes you can assemble ~5T tokens of high-quality STEM and code. That data pipeline — VLM PDF extraction, specialist-model synthesis, aggressive filtering — is itself a major undertaking that the architecture's simplicity can make you forget.

A note on reading the AIME numbers specifically. AIME'24 is fifteen problems; AIME'25, when it is reported, is another fifteen. A jump from 70.1 to 85.1 is roughly two more problems solved out of fifteen. On a fifteen-item benchmark, the confidence interval is wide, and the standard mitigation — reporting the mean of many sampled completions per problem (avg@k or pass@k) — smooths the variance but introduces its own choice of $k$. None of this means the gain is not real; the *direction* across the report's full benchmark suite is consistent enough to trust. It means you should not treat a single 15-point AIME delta as a precise measurement of anything. The honest summary of the experiments section is: the *family-level* story (smaller Qwen3 beating larger Qwen2.5; the flagship base beating a 3× larger base) is well-supported by breadth; the *individual headline numbers* are best read as indicative, because the benchmarks they sit on are small and the field's contamination problem is unsolved.

One more thing the table does not show: inference *cost* at equal quality. The most interesting comparison Qwen3 enables is not "Qwen3-235B vs DeepSeek-V3" but "Qwen3-235B at budget 512 vs Qwen3-235B at budget 8,192" — same weights, same benchmark, an order of magnitude difference in tokens generated. That curve is the actual product, and the report shows less of it than the one-shot benchmark tables.

## Critique

**What is strong.** The unification is real and the report is honest that it is a *training-pipeline* achievement, not an architecture one. The ordering of the four stages is well-argued: building reasoning in isolation before fusing it with chat behavior is the kind of decision that sounds obvious in retrospect and is easy to get wrong (fuse too early and the RL signal gets diluted by chat data; fuse too late and the modes never integrate). The thinking budget is a genuine product primitive — a smooth compute dial is worth more in production than two percentage points on a benchmark. And making the dual-mode capability *distillable* is the structural insight that makes the whole family coherent.

**What is weak or under-supported.**

- **Thinking-mode fusion is under-ablated.** Stage 3 is the load-bearing stage for the entire "one model" claim, yet the report gives relatively little ablation on *how much* non-thinking quality is sacrificed by fusion versus a non-thinking-only model, or how the thinking/non-thinking data ratio in Stage 3 was chosen. The honest experiment — fused model vs. two specialist models, on both modes — is exactly the one a skeptic wants and does not fully get.
- **Thinking-budget curves are narrow.** The graceful-degradation property is demonstrated on a slice of (mostly math) benchmarks. Whether accuracy degrades as smoothly on open-ended generation, agentic tool-use, or long-context retrieval — where "partial reasoning" is much harder to define — is not really shown. The smooth curve may be a property of *verifiable* tasks specifically.
- **The distillation comparison flatters itself.** "1/10 the GPU hours" excludes the teacher's training cost. That is a fair number for an org that already has the flagship, but it is not the cost of the *capability* — it is the marginal cost of *propagating* it. The report could be clearer that the comparison is full-pipeline-per-model vs. distill-from-an-existing-teacher, not full-pipeline vs. distillation in the abstract.
- **General RL (Stage 4) is a black box.** "20+ task types" with a mix of verifiers and reward models is precisely where reward hacking lives, and the report says little about how it was detected or contained.

**What would change my mind.** If an independent reproduction showed that thinking-mode fusion costs less than ~1–2 points of non-thinking quality versus a dedicated non-thinking model across a *broad* benchmark set — including open-ended and agentic tasks — I would treat the "one model, two modes" claim as fully settled rather than mostly settled. Conversely, if the thinking-budget degradation curve turned out to be sharp (not smooth) on non-verifiable tasks, the budget would drop from "product primitive" to "math-benchmark trick," and the operational story in this post would need a serious caveat.

## What I'd build with this

1. **A budget-aware request router.** Instead of routing between a fast and a smart *model*, route to a single Qwen3 endpoint and route the *budget*. A lightweight classifier estimates problem difficulty and emits a token budget; the serving layer applies it via the control loop above. One model in memory, one KV-cache layout, a scalar per request.
2. **Per-user budget SLAs.** Expose the thinking budget as a billing tier. Free tier gets budget 256, pro gets 4,096. This is a far cleaner cost-control lever than gating model access, because the latency and the price scale with the same knob the user can see.
3. **Domain distillation with your own teacher.** If you have a strong in-house model on a narrow domain, replicate the strong-to-weak setup: on-policy distill a 1.7B or 4B Qwen3 student from it. The report's evidence is that on-policy distillation transfers the *controllability*, not just the accuracy — so your small model keeps the thinking switch.
4. **A fusion ablation harness.** Before trusting dual-mode in production, build the ablation the report skimps on: fine-tune a non-thinking-only checkpoint and a fused checkpoint, evaluate both in non-thinking mode on *your* traffic distribution, and measure the fusion tax directly. If it is under a point, adopt; if not, you have quantified exactly what "one model" costs you.

5. **An adaptive-budget controller.** The static budget table above leaves value on the table. Build a controller that starts a request at a modest budget, inspects the partial reasoning trace at the cap, and *extends* the budget only if the trace shows signs of being mid-derivation (unresolved sub-goals, a dangling equation) rather than converging. This turns the budget from a fixed dial into a closed loop — most requests finish cheaply, and only the genuinely hard ones escalate. The smooth-degradation property is what makes this safe: a budget extension never destabilizes a trace, it only refines it. Done well, this recovers most of the accuracy of a large fixed budget at close to the average cost of a small one, which is the entire economic promise of test-time compute made schedulable.

## References

- **Qwen3 Technical Report** — [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
- **Qwen3 models and code** — [github.com/QwenLM/Qwen3](https://github.com/QwenLM) · weights on [Hugging Face](https://huggingface.co/Qwen)
- Related on this blog:
  - [DeepSeek-R1: incentivizing reasoning capability in LLMs via RL](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning)
  - [Modern LLM architectures: Qwen, Llama, Gemma, DeepSeek](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek)
  - [MoE LLM architecture, training and finetuning](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies)
  - [Distillation in LLMs](/blog/machine-learning/large-language-model/distillation-in-llm)
  - [Fine-tuning an LLM with GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo)

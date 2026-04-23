---
title: "Effective LLM Fine-Tuning Techniques: A Practical Guide"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "Large Language Model"
tags: ["llm", "fine-tuning", "lora", "qlora", "peft", "sft", "dpo", "instruction-tuning", "adapter", "deep-learning"]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "Fine-tuning is how you turn a generic LLM into a model that actually works for your problem. This guide walks through every major technique — full fine-tuning, LoRA, QLoRA, DoRA, prompt tuning, SFT, DPO, and more — with intuition, math, and practical advice on when to use each."
---

## Why Fine-Tune at All?

Pretrained LLMs like Llama, Qwen, Mistral, and Gemma are remarkable generalists. They can write code, summarize documents, translate languages, and hold conversations — all with the same weights. So why would you ever change them?

Three reasons, in roughly this order of importance:

1. **Behavior** — The base model talks in a way that doesn't match your product. Too verbose. Too cautious. Doesn't follow your JSON schema. Doesn't refuse the things it should refuse.
2. **Domain knowledge** — The model doesn't know enough about your specific domain (medical coding, legal clauses, your company's internal API).
3. **Cost** — A fine-tuned 7B model can match a prompted 70B model for a specific task, at 10× lower inference cost.

Before fine-tuning, always ask: **can a good prompt + RAG + few-shot examples solve this?** If yes, do that. Fine-tuning adds a whole training pipeline, a GPU bill, and a model you now have to maintain. It's the right answer often, but not always first.

This guide is the map of what to reach for when prompting isn't enough.

## The Landscape in One Picture

![Fine-tuning landscape on two axes: what you update (Full FT vs PEFT — LoRA/DoRA, QLoRA, IA³/Prefix) × what you train on (SFT, preference opt with DPO/GRPO/PPO-RLHF, distillation)](/imgs/blogs/finetune-01-landscape.png)

Fine-tuning techniques split along two independent axes:

**Axis 1 — What do you update?**

- **Full fine-tuning** — every weight is trainable (expensive, powerful)
- **Parameter-Efficient Fine-Tuning (PEFT)** — only a tiny slice is trainable (cheap, almost as good)

**Axis 2 — What signal do you train on?**

- **Supervised Fine-Tuning (SFT)** — mimic ideal outputs
- **Preference / RL methods** — learn from comparisons between outputs (DPO, PPO, GRPO, KTO, ORPO)
- **Continued pretraining** — predict next tokens on domain text (no labels)

A real project usually chains a couple of these: continued pretraining → SFT → DPO, each with a PEFT method layered in.

Let's walk through them.

## Part 1: What You Update — Full Fine-Tuning vs. PEFT

![LoRA vs QLoRA: LoRA freezes W and trains low-rank A,B (rank r) summed into output; QLoRA freezes W in 4-bit NF4 while LoRA A,B stay in bf16 — same math, much less GPU memory](/imgs/blogs/finetune-02-lora-qlora.png)

### Full Fine-Tuning: The Brute-Force Option

In full fine-tuning, you update **every parameter** in the model via standard gradient descent. A 7B model has 7 billion weights — all of them become trainable.

**Memory math, so you see the pain:**

For a model with $N$ parameters, training requires (roughly):

- Weights: $2N$ bytes (bf16)
- Gradients: $2N$ bytes (bf16)
- Optimizer state (Adam): $8N$ bytes (fp32 momentum + variance)
- Activations: depends on batch size and sequence length, often $2$–$10×$ weights

Total: **~16× model size** just for the training state, before activations. A 7B model needs ~112 GB — more than a single A100 80GB can hold. A 70B model needs over a terabyte of GPU memory. You need multi-GPU training with FSDP, DeepSpeed, or Megatron-LM.

**When it's worth it:** you have the compute, the data is large (100K+ high-quality examples), and you're building a product you'll keep for years.

**When it's not:** most of the time. PEFT almost always gets you 95-99% of full fine-tuning quality at a fraction of the cost.

### LoRA: The Idea That Changed Fine-Tuning

**LoRA (Low-Rank Adaptation)**, published in 2021, is the technique most production fine-tunes use today. Understanding it is worth 20 minutes.

**The insight:** when you fine-tune a large model on a narrow task, the *change* in weights has very low intrinsic dimensionality. You're not teaching the model a new alphabet — you're nudging it. Small nudges don't need millions of free parameters; they live in a low-rank subspace.

**The math:** for a linear layer with weight matrix $W \in \mathbb{R}^{d \times k}$, instead of learning an update $\Delta W$ of the same shape, LoRA factorizes it:

$$
\Delta W = B \cdot A, \quad A \in \mathbb{R}^{r \times k}, \quad B \in \mathbb{R}^{d \times r}
$$

Where $r$ is the **rank** — typically 4, 8, 16, 32, or 64. The forward pass becomes:

$$
h = W x + \Delta W x = W x + B A x
$$

**Why this saves so much:** if $d = k = 4096$ and $r = 8$:
- Full update: $4096 \times 4096 = 16{,}777{,}216$ parameters
- LoRA update: $4096 \times 8 + 8 \times 4096 = 65{,}536$ parameters

That's **256× fewer trainable parameters** per layer, with nearly identical downstream performance on most tasks.

**How to read a LoRA config:**

- `r` — rank. Higher = more capacity, more parameters. 16 is a common default; go higher (32-128) for harder tasks.
- `alpha` — scaling factor. The effective update is $\frac{\alpha}{r} B A x$. Rule of thumb: `alpha = 2 * r` or `alpha = r`.
- `target_modules` — which linear layers get LoRA. Attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) are the classic targets. Adding the MLP layers (`gate_proj`, `up_proj`, `down_proj`) is "all-linear LoRA" and usually helps.
- `dropout` — LoRA dropout, typically 0.0-0.1.

**At inference time**, you can either:
- Keep $A$ and $B$ separate (swap LoRA adapters without touching the base model — great for multi-tenant serving)
- **Merge** them into $W$: $W' = W + BA$. Zero inference overhead, but now your base model is modified.

> **Intuition:** imagine you have a pretrained sculptor. Full fine-tuning is teaching them an entirely new style. LoRA is handing them a small template they overlay on whatever they sculpt — same hands, same tools, just a filtered view.

### QLoRA: LoRA on a Budget GPU

**QLoRA (Quantized LoRA)** combines LoRA with 4-bit quantization. The key idea: you don't need the full-precision weights to compute LoRA gradients — you just need them forward-pass-accurate enough.

Here's the recipe:

1. **Quantize the base model to 4-bit** using NormalFloat 4-bit (NF4), a data type optimized for normally-distributed weights.
2. **Keep LoRA adapters in bf16** (they're tiny — a few million parameters).
3. During the forward pass, dequantize base weights on the fly, compute the output, then discard the fp16 versions.
4. Compute gradients **only for the LoRA adapters.**
5. Use **paged optimizers** (Adam state lives in CPU RAM, paged to GPU as needed) to handle memory spikes.

**The payoff:** you can fine-tune a **70B model on a single 48GB GPU**. That was unthinkable before QLoRA.

**Quality tradeoff:** near-zero. The original QLoRA paper showed it matches 16-bit LoRA on most benchmarks. Production teams routinely use QLoRA for 7B-70B models.

**Downside:** forward passes are a bit slower because of the dequantization, and the final merged model is worse than a fresh fp16 fine-tune. If you merge QLoRA adapters back into the 4-bit base, you keep the quantization error; if you merge into fp16 weights, you lose some of the "alignment" between the adapter and the quantized base.

### DoRA: Decomposing Magnitude and Direction

**DoRA (Weight-Decomposed Low-Rank Adaptation)**, from 2024, noticed that LoRA lumps magnitude and direction of weight updates together, while full fine-tuning naturally separates them. DoRA decomposes each weight as:

$$
W = m \cdot \frac{V}{\|V\|}
$$

where $m$ is a scalar magnitude and $V$ is the direction. It then applies LoRA to the direction and trains the magnitude separately:

$$
W' = m \cdot \frac{V + BA}{\|V + BA\|}
$$

In practice: DoRA closes much of the (small) gap between LoRA and full fine-tuning, at ~10-20% extra compute over LoRA. Use it when you have the budget and LoRA isn't quite getting you there.

### Other PEFT Methods Worth Knowing

- **Prefix Tuning / P-Tuning v2** — instead of modifying weights, prepend learnable "virtual tokens" to every layer's attention. Tiny parameter count, but harder to optimize and generally weaker than LoRA for modern models.
- **Prompt Tuning** — same idea as prefix tuning but only at the input layer. Even smaller, even weaker, rarely used for big models today.
- **Adapter Tuning** (Houlsby adapters) — insert small MLP bottleneck layers between transformer blocks. Predecessor to LoRA. Adds inference latency (can't be merged), so LoRA has mostly replaced it.
- **IA³** — learns three vectors per layer that scale keys, values, and FFN activations. Ultra-lightweight (hundreds of parameters per layer). Good for quick experiments, weaker for deep behavior changes.
- **BitFit** — train only the biases. Works surprisingly well for small adaptations. Useful as a sanity-check baseline.

**Practical ranking for most projects (2026):**

```
LoRA / QLoRA (default)  →  DoRA (when you need more juice)
                        →  Full fine-tuning (when data is abundant)
                        ↓
Everything else is either legacy or a niche tool.
```

## Part 2: What You Train On — SFT, Preference Optimization, and Beyond

Choosing how to update is only half the decision. The other half: **what is the training signal?**

### Continued Pretraining: Learning New Facts

**Continued pretraining** (sometimes "domain adaptive pretraining") is just... pretraining, continued. You take a base model and keep running next-token prediction on a corpus of text from your domain.

**When to use it:**
- You have a lot of unlabeled domain text (medical journals, legal filings, financial reports, code in an unusual language)
- The base model clearly lacks domain vocabulary or facts — not just style
- You're fine with 100GB+ of text and multi-GPU training

**Gotchas:**
- **Catastrophic forgetting.** The model will forget general capabilities if you only feed it domain text. Mix in 10-30% general-purpose text (like RedPajama, FineWeb, or a slice of the original pretraining data).
- **Loss curves are not enough.** A loss going down on medical text tells you the model is getting better at medical text — not that it's still good at coding or chat. Always evaluate on a held-out general benchmark.
- **Tokenizer mismatch.** If your domain uses non-English or a specialized script, the default tokenizer may be wildly inefficient. Consider extending the tokenizer vocabulary — but be warned, this creates randomly-initialized embeddings that need careful warmup.

### Supervised Fine-Tuning (SFT): Learning to Follow Instructions

**SFT** is the workhorse. You show the model examples of (prompt, ideal response) pairs and train it to produce the response given the prompt — using standard next-token cross-entropy loss, computed only on the response tokens.

**The training format:**

```
<|system|>You are a helpful assistant.
<|user|>How many planets are in the solar system?
<|assistant|>There are eight planets in the solar system: Mercury, Venus, Earth,
Mars, Jupiter, Saturn, Uranus, and Neptune.
```

During training, we mask the loss on system and user tokens and only compute cross-entropy on the assistant tokens. The model learns **what to produce, not what to consume.**

**The SFT loss:**

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{|y|} \sum_{t=1}^{|y|} \log P_\theta(y_t \mid x, y_{<t})
$$

Where $x$ is the prompt, $y$ is the assistant response, and the loss averages over response tokens only.

#### What Makes SFT Data Good

This is 80% of the battle. A mediocre SFT run on great data beats a meticulous SFT run on mediocre data, every time.

**Rules of thumb:**

1. **Quality over quantity.** 1,000 expertly-written examples beats 100,000 scraped ones. The LIMA paper (Meta, 2023) famously showed strong results with just 1,000 high-quality examples.
2. **Diversity matters.** 10,000 examples of "summarize this email" teach the model to summarize emails. They don't teach it to *be* an assistant. Cover the breadth of behaviors you actually want.
3. **Response length distribution matters.** If all training responses are 2 sentences, the model will refuse to write paragraphs. If they're all 10 paragraphs, it'll waffle forever. Match the distribution to what you'll actually serve.
4. **System prompts in training should match production.** If you don't include a system prompt in training but do in production (or vice versa), you're testing under distribution shift.
5. **Deduplicate aggressively.** Duplicated examples are pure overfitting fuel.

#### How Many Epochs?

For instruction tuning, **1-3 epochs** is almost always right. More, and you start memorizing training data at the cost of generalization. The loss going down on your training set means less than you think — always watch a held-out eval set.

#### Chat Templates: The Silent Killer

Every model family uses a different chat template:

- **Llama 3:** `<|start_header_id|>user<|end_header_id|>\n\n{msg}<|eot_id|>`
- **Qwen:** `<|im_start|>user\n{msg}<|im_end|>`
- **Mistral:** `[INST] {msg} [/INST]`

If you fine-tune with the wrong template (or no template), the model at inference time will see formatting it was never trained on and produce garbage. **Always use the tokenizer's built-in `apply_chat_template`** — don't format by hand.

### Preference Optimization: Learning from Comparisons

SFT teaches the model what's correct. But "correct" is only half of what you want. You also want the model to *prefer* one response over another when both are valid — e.g., prefer a concise, polite, on-format reply over a rambling, off-format one.

This is the domain of **preference optimization**. You start from pairs:

- A prompt $x$
- A "chosen" (preferred) response $y_w$
- A "rejected" response $y_l$

And you train the model to make $y_w$ more likely than $y_l$.

Several flavors exist — here's the overview:

#### RLHF with PPO — The Classical Recipe

Train a reward model on preference pairs, then use PPO to maximize the reward while staying close (via KL) to the SFT model. This is how ChatGPT was originally trained. Powerful, but complex: four models in memory, unstable training, many hyperparameters. Rarely the first choice today unless you need its specific strengths.

#### DPO — The Modern Default

**Direct Preference Optimization** (Rafailov et al., 2023) proves you can skip the reward model entirely. The DPO loss:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)
$$

In plain terms: push up the probability ratio of the chosen response (vs. the reference model), push down the probability ratio of the rejected response. $\beta$ controls how far the policy is allowed to drift from the reference.

**Why DPO is popular:** stable, simple, uses the same training infrastructure as SFT, no reward model needed. It's the right first try for preference training in 2026. See the dedicated [DPO deep-dive](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) for the full derivation and practical guide.

#### GRPO — When You Have Verifiable Rewards

**Group Relative Policy Optimization** (from DeepSeek) shines when you can **verifiably grade** outputs (math answers, code that compiles, structured outputs that validate). Instead of learning a reward model, GRPO generates multiple candidates per prompt, scores each with a programmatic reward, and optimizes the policy using the group-relative advantage.

Use GRPO for: math reasoning, coding tasks, any domain with a ground-truth checker. See the [GRPO article](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) for details.

#### ORPO — Collapsing SFT + Preference into One Step

**Odds Ratio Preference Optimization** (Hong et al., 2024) combines SFT and preference optimization into a single loss:

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \lambda \cdot \mathcal{L}_{\text{OR}}
$$

Where $\mathcal{L}_{\text{OR}}$ is a log-odds-ratio term that penalizes the rejected response. The clever bit: ORPO doesn't need a reference model — you save on memory and training time. For smaller models and tight budgets, ORPO is very attractive.

#### KTO — When You Have Thumbs-Up / Thumbs-Down

**Kahneman-Tversky Optimization** replaces preference *pairs* with preference *labels*: each response is marked "good" or "bad" independently. No need for matched (chosen, rejected) pairs from the same prompt. Useful when your feedback signal is thumbs-up/thumbs-down from users, not ranked comparisons.

#### When to Use Which

| Technique | When to use it |
|---|---|
| **SFT alone** | You have good demonstration data and don't have preference data yet. Start here. |
| **DPO** | You have (prompt, chosen, rejected) pairs and want a stable, simple preference step after SFT. |
| **GRPO** | Your task has an automatic verifier (math answers, passing tests, format validation). |
| **ORPO** | You want to merge SFT and preference into one phase, and you're on a tight memory budget. |
| **KTO** | Your feedback is binary thumbs-up/thumbs-down rather than pairwise rankings. |
| **PPO (RLHF)** | You have the team, compute, and a specific reason to believe a learned reward model will outperform DPO. Rare today. |

## Part 3: Special Techniques Worth Knowing

### Reasoning Fine-Tuning (Chain-of-Thought Distillation)

Models like DeepSeek-R1 and o1 showed that you can distill step-by-step reasoning into smaller models via SFT on curated (problem, long reasoning trace, answer) triples. The formula:

1. Generate long reasoning traces from a big reasoning model
2. Filter for correctness (use the verifier — this is key)
3. SFT a smaller model on the filtered traces

This is just SFT, but the **data curation** is the whole technique. Without the verification filter, the model learns to *sound* smart rather than *be* correct.

### Rejection Sampling Fine-Tuning (RFT)

Closely related: generate $N$ candidate responses per prompt, use a reward model or verifier to pick the best one, and SFT on (prompt, best response) pairs. Cheap, stable, and surprisingly effective. Used heavily in post-training pipelines of modern open models.

### Self-Play and Iterative Fine-Tuning

Generate responses with your current model, rank or filter them, SFT/DPO on the top responses, repeat. Known by many names: **STaR, ReST, iterative DPO, self-rewarding**. The concern is mode collapse — the model can drift into a narrow style. Mitigate with: strong filtering, fresh human data injected each round, KL regularization to the base model.

### Merging Techniques

You don't always have to *train* to combine behaviors. **Model merging** (SLERP, TIES, DARE) mathematically combines the weights of multiple fine-tuned models into one. It's free, fast, and often works shockingly well — especially for combining capabilities from different SFT runs.

Use merging when: you have multiple task-specific fine-tunes and want a single model. Skip it when: one of your models is much stronger than the others (you'll just drag it down).

### Unlearning and Safety Fine-Tuning

Sometimes you don't want to add behavior — you want to **remove** it. Techniques like **Negative Preference Optimization**, **task vectors subtraction**, or targeted SFT on refusal data can make a model unlearn specific capabilities (e.g., refuse requests about a narrow topic).

This is harder than it sounds. Models often comply with the *letter* of the refusal training but not the spirit, and unlearning one thing often damages unrelated capabilities. Evaluate broadly.

## Part 4: A Recommended Pipeline for Most Projects

![Recommended fine-tuning pipeline: Base → SFT + task eval → Preference (DPO/GRPO) + pairwise eval → Safety (red-team, refusals) → Deploy — tune LR 1e-5 to 2e-4, LoRA rank 8-64, batch + grad accum](/imgs/blogs/finetune-03-pipeline.png)

Here's a pragmatic order of operations for building a fine-tuned LLM in 2026:

**Stage 0 — Baseline with prompting and RAG.**
If this works, stop. Save yourself a GPU bill.

**Stage 1 — (Optional) Continued pretraining.**
Only if your domain vocabulary is truly foreign to the base model. Mix in general-purpose data. Evaluate on a general benchmark to catch forgetting.

**Stage 2 — SFT with LoRA or QLoRA.**
Start with 1,000-10,000 high-quality examples. 1-3 epochs. LoRA rank 16-64, alpha = 2×rank, target all linear layers. Watch held-out eval loss, not training loss.

**Stage 3 — Preference optimization with DPO.**
After SFT, if you have preference data (or can generate it via a bigger model + ranking), run DPO with the SFT model as both the reference and starting point. $\beta$ typically 0.1-0.5.

**Stage 4 — (Optional) GRPO or rejection sampling.**
If your task has verifiable rewards, add a GRPO pass. Otherwise, rejection-sample-and-SFT is a cheap way to polish further.

**Stage 5 — Evaluation and iteration.**
Hold out a real evaluation set from day one. Check not just your target task but also general capability regressions, safety behaviors, and output format compliance. Iterate on **data**, not hyperparameters — data fixes are almost always more impactful.

## Part 5: Things That Go Wrong

A greatest-hits list of fine-tuning failure modes:

- **Forgetting general capabilities.** Mitigation: mix general-purpose data, use lower learning rates, use LoRA instead of full fine-tuning, freeze more layers.
- **Overfitting on small datasets.** Symptom: train loss plummets, eval loss flat or up. Mitigation: fewer epochs, more data, higher LoRA rank (actually — sometimes *lower* rank fixes this too), more dropout.
- **Model refuses to follow format.** Usually a chat template mismatch. Verify with `tokenizer.apply_chat_template` and compare train/eval formatting byte-by-byte.
- **DPO collapses and outputs become weird.** Your $\beta$ is too low, or your chosen/rejected pairs are too similar (no learnable signal), or your SFT model was too weak to serve as a reference. Try $\beta = 0.3$-$0.5$.
- **Model learns to game the reward.** In GRPO / PPO, the model discovers a shortcut that maximizes reward without genuinely solving the task. Fix the verifier. Reward hacking is a data problem, not a training problem.
- **Loss is fine, outputs are broken.** The loss you're computing doesn't capture what you care about. Always eyeball actual generations, not just metrics.

## Closing Thought

Fine-tuning used to require a cluster. In 2026, you can meaningfully fine-tune a 70B-parameter model on a single consumer-grade GPU using QLoRA. The bottleneck is no longer compute — it's **data quality**, **evaluation design**, and **judgment about which technique to reach for.**

That judgment is what this guide tries to give you. Pick one technique, run one experiment, look hard at the outputs, and iterate. That beats picking the fanciest method and hoping.

## Further Reading

- [Fine-Tuning LLMs with DPO: A Practical Guide](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo)
- [Fine-Tuning LLMs with GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo)
- [LLM Training Playbook](/blog/machine-learning/large-language-model/llm-training-playbook)
- *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)
- *QLoRA: Efficient Finetuning of Quantized LLMs* (Dettmers et al., 2023)
- *DoRA: Weight-Decomposed Low-Rank Adaptation* (Liu et al., 2024)
- *Direct Preference Optimization* (Rafailov et al., 2023)
- *ORPO: Monolithic Preference Optimization without Reference Model* (Hong et al., 2024)
- *LIMA: Less Is More for Alignment* (Zhou et al., 2023)

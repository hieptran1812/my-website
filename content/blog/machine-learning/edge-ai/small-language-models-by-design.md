---
title: "Small language models by design: Phi, Gemma, MobileLLM, and TinyLlama"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why a model designed small from the start — better data, deeper-and-thinner shape, GQA, tied embeddings, over-trained for inference — can beat a quantized big model at the same memory budget, with the param math and runnable code to prove it."
tags:
  [
    "edge-ai",
    "model-optimization",
    "small-language-models",
    "on-device-llm",
    "phi",
    "gemma",
    "tinyllama",
    "mobilellm",
    "inference",
    "efficient-ml",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/small-language-models-by-design-1.png"
---

There are two ways to fit a language model onto a phone, and most of this series so far has been about the first one. You take a model that was trained at full size for the cloud — a 7B, say — and you *shrink* it. You quantize the weights to 4 bits, you prune the dead channels, you distill it into a smaller student. Every one of those levers takes a model that was *designed for a datacenter* and squeezes it through the door of a 4 GB phone. It works, and we have spent whole posts on doing it well (see the companion piece on [weight-only LLM quantization with GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq)).

But there is a second way, and it quietly wins more often than people expect: **don't shrink a big model — design a small one from the start.** Train it on better data instead of more data. Shape its layers for the regime where a billion parameters live, not the regime where seventy billion live. Tie its embeddings, group its attention heads, over-train it on far more tokens than the compute-optimal rule says it "deserves," because on a phone the training cost is paid once and the inference cost is paid a billion times. The result is a model that is *natively* the right size — and at the same memory budget, a model designed small can beat a big model squeezed down to fit.

That last sentence is the entire thesis of this post, and it is worth pinning down with numbers before we explain *why*. Figure 1 is the matched-budget comparison this whole post defends: at roughly 1.5 GB of phone memory, a 7B model quantized to int4 lands at about 56 on MMLU, while a 2.7B model that was *designed* small — Microsoft's Phi-2, trained on curated "textbook-quality" data and over-trained well past compute-optimal — lands around 61. Same memory. The small-by-design model is ahead. The squeezed big model is dragging along outlier weights and an architecture tuned for a scale it no longer occupies.

![Before and after comparison showing a 7B model quantized down to fit a phone budget versus a 2.7B model designed small from the start at the same memory, with the small model scoring higher on MMLU](/imgs/blogs/small-language-models-by-design-1.png)

By the end of this post you will be able to do four concrete things. You will be able to **read the parameter budget of any sub-3B model** and say where every megabyte goes — embeddings versus attention versus the feed-forward network — and predict which design choices buy back the most memory. You will be able to **explain and configure the small-scale architecture levers** that actually matter: deep-and-thin layouts, grouped-query attention, tied embeddings, the SwiGLU/RMSNorm stack, and the right vocabulary size. You will be able to **load Phi, Gemma, or TinyLlama, inspect its GQA and tied-embedding config, quantize it to 4 bits, and run it on a laptop**. And you will have a **decision rule** for when a small-by-design model is the right call versus quantizing a bigger one — plus an honest accounting of where small models hit a ceiling and how distillation lifts it. This is the small-language-model design playbook, and it slots into the [four-lever Pareto frame](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) as the lever we have so far under-used in this series: **efficient architecture, applied at design time, not retrofit time.**

## 1. The matched-memory framing: shrink-a-big versus design-a-small

Let me make the comparison fair before I make it favorable, because "small model beats big model" is the kind of claim that smells like a benchmark trick.

The honest axis is not parameter count. It is **on-device memory at deployment precision**, because that is what the phone actually has to hold. A 7B model in fp16 is about 13 GB of weights; no consumer phone holds that. Quantize it to a good 4-bit k-quant (Q4_K_M in `llama.cpp` terms) and you get roughly 4 GB, still too big for the working set on most phones once you account for the KV cache and the OS. Push harder — Q4_0 plus a small embedding/output table — and you can get a 7B down toward 1.5–1.8 GB, but now you are at the edge of where 4-bit quantization starts to cost real accuracy, because the largest LLMs carry a small number of **outlier activations and weights** that 4 bits represent badly.

A 2.7B model, by contrast, is about 5.4 GB in fp16 and roughly 1.4–1.6 GB at Q4. So *at the same ~1.5 GB budget*, you are choosing between a 7B that has been pushed to the painful end of quantization and a 2.7B sitting comfortably in its int4 sweet spot. That is the matched-memory comparison, and once you frame it that way the result in Figure 1 stops being surprising. You are not comparing 7B to 2.7B. You are comparing **"7B with its quality partly quantized away" to "2.7B at full int4 quality, trained on data that punches above its weight."**

There is a deeper reason the small-by-design model wins, and it is the thing this post is really about. The big model was optimized for a different objective. Its data mix, its layer aspect ratio, its head count, and its training-token budget were all chosen for the regime where it lives — tens of billions of parameters, trained roughly compute-optimally, served on GPUs where memory bandwidth is enormous. None of those choices are right for a billion-parameter model on a phone. When you quantize a 7B to fit, you inherit *all* of those wrong-for-the-target choices and merely change the number of bits. When you design a 2.7B for the phone, you get to make every one of those choices *correctly for the target* from scratch. The bits are the same; the design is not.

So the structure of the rest of this post follows the design decisions, one lever at a time: **data** (Phi), **architecture shape** (MobileLLM, deep-and-thin), **the small-scale param savers** (GQA, tied embeddings), **the training-token lever** (TinyLlama, over-training for inference), and the **vocabulary trade**. Then we make it practical — load, inspect, quantize, run — and we close with the decision rule and the limits.

A note on where this sits in the series' four-lever frame before we dive in, because it is easy to mistake SLM design for a competitor to quantization and pruning when it is really their foundation. The four levers are quantization, pruning/sparsity, distillation, and efficient architecture. The first three are mostly *post-hoc* — you take a trained model and reduce it. Efficient architecture is the one lever you pull at *design time*, before a single training step, and it is the one this post is about. The crucial thing is that it **composes** with the others rather than competing: you design the model small *and then* you quantize it to int4, *and* you distill a big teacher into it, *and* you can prune it further if needed. The matched-memory win in Figure 1 is not "SLM instead of quantization" — it is "a model designed small, *then* quantized" beating "a model designed big, *then* quantized," at the same final memory. The design lever and the compression levers stack; the whole point of this post is to make sure the *base* model you hand to the quantizer was the right shape and the right size to begin with, because no amount of clever 4-bit packing fixes a model that was built for the wrong scale.

## 2. The science of data quality: Phi and "textbooks are all you need"

The first and most counterintuitive lever is that **you can buy quality with data instead of parameters.** This is the central finding of the Phi line of work from Microsoft Research — Gunasekar et al., *Textbooks Are All You Need* (2023), and its successors Phi-1.5 and Phi-2.

The setup is this. Standard pretraining throws a giant pile of web text at a model and lets scale sort it out. The implicit bet is that signal is sparse in web text — most tokens are low-value boilerplate, SEO spam, navigation chrome, duplicated content — but if you read *enough* of it, the rare high-value tokens (a clear explanation, a correct code snippet, a well-reasoned argument) accumulate into competence. The Phi team asked the obvious follow-up: what if you raised the *density* of high-value tokens instead of the *volume* of all tokens? What if you trained on data that looks like a textbook — clear, correct, pedagogically ordered — rather than data that looks like the internet?

Concretely, Phi-1 (1.3B parameters, aimed at Python code) was trained on a mix of (1) a filtered subset of web code, where a classifier kept only files with high "educational value," and (2) a much larger quantity of **synthetic** textbook-style content and exercises generated by a stronger model. The result was a 1.3B model that scored about 50% pass@1 on HumanEval — beating models more than ten times its size that had been trained on raw web code. Phi-1.5 extended the idea to general reasoning; Phi-2 (2.7B) carried it to the MMLU/BBH/reasoning suite and reached scores competitive with 7B–13B models on many benchmarks.

![Matrix comparing web-scale data against curated textbook data at 1.3B and 7B parameters, showing the curated 1.3B model reaching HumanEval scores that web-trained models need ten times the parameters to match](/imgs/blogs/small-language-models-by-design-2.png)

Figure 2 is the trade as a 2×3 grid. Read down the columns: at 1.3B parameters, web data gives you a noisy gradient signal and a model that mostly learns the *shape* of code rather than correct code; the same 1.3B trained on filtered-plus-synthetic textbook data learns correct code, because nearly every token it sees is correct. Read across the rows: to match the textbook-trained 1.3B's HumanEval pass rate using raw web data, you need to scale parameters up by roughly an order of magnitude, because parameters are partly being spent to *average out the noise* in the data rather than to store useful capability.

### 2.1 Why this works, made rigorous

It is tempting to wave at "quality matters" and move on, but there is a real information-theoretic reason, and it ties directly to the scaling-laws literature (see [data-quality scaling laws](/blog/machine-learning/scaling-laws/data-quality-scaling-laws) for the formal treatment). The loss a model reaches is, to first order, set by how much *learnable signal* its training data contains, not by the raw token count. If a fraction $\rho$ of tokens carry the signal you care about and the rest are noise, then a token budget $N$ delivers an *effective* signal budget of roughly $\rho N$. Doubling $\rho$ — by filtering and synthesizing — is worth the same as doubling $N$, at a fraction of the compute.

There is a second effect that is specific to small models. A model's parameters do two jobs: they **store** capability and they **denoise** the training distribution. A 70B model has enough capacity to memorize the long tail of web junk *and* learn the useful patterns; a 1.3B model does not. When you feed a 1.3B model noisy data, a large slice of its tiny capacity is spent fitting noise it cannot afford to fit. Clean data frees that capacity for the thing you want. This is why the data-quality lever is *disproportionately* powerful at small scale — and why it is the first lever in the SLM playbook. The smaller the model, the more it matters that every token earns its place.

The honest caveats: the Phi results were strongest on the benchmarks that resemble the synthetic data (code, textbook-style reasoning), and there was reasonable debate about benchmark contamination given that the synthetic data was generated by models trained on the web. The robust, uncontroversial takeaway is the *direction*: at sub-3B scale, **data curation is a primary lever, not a hygiene step.** You will get more from a month of building a clean, dense corpus than from another month of architecture tuning.

### 2.2 How the "textbook-quality" corpus actually gets built

It is worth knowing the recipe, because "use better data" is uselessly vague until you see the moving parts. The Phi pipeline has two complementary halves, and both are reproducible.

The first half is **filtering** existing web data by educational value. You take a large web corpus (for code, something like The Stack), label a small subset of files by whether they read like good teaching material — clear, self-contained, well-commented, correct — and train a lightweight classifier to predict that label. Then you run the classifier over the whole corpus and keep only the high-scoring fraction. This is cheap (the classifier is small and runs once) and it throws away the majority of the data, which feels wasteful until you remember the Section 2.1 argument: for a small model, a smaller pile of high-signal tokens trains a *better* model than a larger pile of mixed-signal tokens, because the model isn't spending capacity fitting the junk.

The second half is **synthesis**: prompting a strong existing model to generate textbook-style explanations, worked examples, and exercises on a controlled spread of topics. The art here is *diversity* — naïve synthesis collapses to a narrow band of similar examples, so you deliberately vary the topic, the difficulty, the style, and the constraints in the prompts to spread the generated data across the space you want the small model to cover. Synthetic data has two big advantages for SLMs: it is *dense* (every generated token is on-topic and, with a good teacher, usually correct) and it is *targeted* (you generate exactly the distribution of skills you want, rather than hoping the web happens to contain them in the right proportions). It also has the obvious risk — it inherits the teacher's blind spots and can leak the teacher's benchmark exposure — which is the contamination concern above and the reason you validate on held-out, freshly-written evaluations rather than trusting public benchmarks alone.

The practical takeaway for someone building an SLM today: budget real engineering time for the data pipeline — a quality classifier, a synthesis loop with diversity controls, aggressive dedup, and a decontamination pass against your eval sets — and treat it as a *first-class* part of the model, on par with the architecture. At small scale, the corpus is not the thing you feed the model; the corpus *is* the model's capability, more directly than at any larger scale.

## 3. Architecture for small scale: deep-and-thin beats wide-and-shallow

The second lever is the *shape* of the network, and the headline result here is from Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases* (Meta, 2024): at a fixed sub-1B parameter budget, **a deep-and-thin model beats a wide-and-shallow one.** This runs against the folk wisdom that, past a point, depth and width are interchangeable knobs you can trade freely. At small scale they are not.

Set up the comparison precisely. The dominant parameter cost of a transformer block is two pieces. The attention projections cost about $4 d^2$ parameters per layer (the $W_Q, W_K, W_V, W_O$ matrices, each roughly $d \times d$). The feed-forward network costs about $2 d \cdot d_\text{ff}$ per layer, and with the common $d_\text{ff} \approx 4d$ (or its SwiGLU equivalent) that is roughly $8 d^2$. So a single block costs on the order of $12 d^2$ parameters, and a model of $L$ layers with hidden size $d$ has roughly

$$
P_\text{blocks} \approx 12 \, L \, d^2
$$

parameters in its transformer stack (embeddings are separate; we get to those in Section 5). Now hold $P_\text{blocks}$ fixed. You can spend it as **few wide layers** (large $d$, small $L$) or **many thin layers** (small $d$, large $L$). Because $P_\text{blocks}$ scales with $L$ but with $d^2$, going thinner is cheap in width-per-layer and lets you buy a lot of depth. The question MobileLLM answered empirically is: which spend gives more accuracy?

![Before and after comparison of a wide-shallow transformer layout versus a deep-thin one at the same sub-1B parameter budget, with the deep-thin model gaining several accuracy points](/imgs/blogs/small-language-models-by-design-3.png)

Figure 3 shows the contrast at the ~125M scale MobileLLM studied. The wide-and-shallow layout — say 12 layers at $d = 1024$ — and the deep-and-thin layout — say 30 layers at $d = 576$ — can be tuned to the same parameter count, but the deep-and-thin one wins by several points on zero-shot reasoning benchmarks. MobileLLM pushed this to its sub-billion family and showed consistent gains from depth.

### 3.1 The intuition for why depth wins at small scale

Here is the reasoning, and it is worth internalizing because it generalizes. A transformer layer is, roughly, **one round of "gather information from other tokens (attention), then transform it (FFN)."** The number of layers is the number of *sequential refinement steps* the model can perform — how many times it can re-read the context and update its representation. Width is the *capacity per step* — how much information each step can hold.

At large scale you have plenty of both, so the marginal layer and the marginal unit of width are similarly valuable. At small scale you are starved, and the question becomes which scarcity hurts more. The empirical answer is that **compositional reasoning — the thing benchmarks reward — needs sequential steps more than it needs per-step capacity.** A task like "resolve this pronoun, then use the resolved entity to answer a question about it" is two refinement steps; a model with too few layers literally cannot perform it no matter how wide each layer is, because there is no later layer to do the second step. Width can make each step a little richer, but it cannot manufacture a step that does not exist. So when forced to choose, depth buys you *new capabilities* (more composition) while width buys you *marginally better existing capabilities* — and new capabilities move benchmarks more.

This is not a license to make the model infinitely deep. Very deep, very thin models become hard to train (vanishing/exploding signal, optimization instability) and, importantly for the edge, **deep-and-thin can be slower** than wide-and-shallow at the same parameter count, because each layer is a sequential dependency — you cannot start layer $L+1$ until layer $L$ finishes — whereas width parallelizes within a layer. On a GPU with lots of parallelism this latency cost can dominate; on a memory-bound phone decode (batch=1, one token at a time) it matters less because you are bandwidth-limited anyway. So depth-versus-width is itself a *Pareto* choice between accuracy and latency, exactly the frame this series keeps returning to. MobileLLM's contribution is the measurement that, for accuracy per parameter at sub-1B scale, depth is underpriced.

MobileLLM also paired deep-and-thin with **block-wise weight sharing** — reusing the same transformer block weights across adjacent layers, so you get the depth (sequential steps) without paying for the parameters twice. That is a more aggressive lever with its own trade-offs, but it follows the same logic: at small scale, *sequential computation is the scarce resource worth buying cheaply.*

### 3.2 The rest of the modern small-model stack: SwiGLU, RMSNorm, RoPE

Deep-and-thin sets the *aspect ratio*; three more choices set the *building blocks*, and modern SLMs have converged on the same three because each one buys a little quality or a little efficiency at the small-scale margin where every bit counts. They are worth understanding rather than copying blindly, because each is a small, defensible engineering decision rather than a magic ingredient.

**SwiGLU for the feed-forward network.** The classic transformer FFN is $\text{FFN}(x) = W_2 \,\sigma(W_1 x)$ with a ReLU or GELU nonlinearity $\sigma$ — two matrices and one activation. SwiGLU (Shazeer, 2020) replaces this with a *gated* form, $\text{FFN}(x) = W_2 \big(\text{SiLU}(W_\text{gate}\, x) \odot (W_\text{up}\, x)\big)$, where $\odot$ is element-wise multiply and $\text{SiLU}(z) = z\,\sigma(z)$ is the sigmoid-weighted linear unit. The gate lets the network *modulate* each hidden unit multiplicatively instead of just thresholding it, which empirically gives a small but reliable quality bump at fixed parameter count. The catch is that the gated form uses **three** matrices instead of two, so to keep the parameter count matched you shrink the intermediate dimension: a vanilla FFN with $d_\text{ff} = 4d$ has the same parameters as a SwiGLU FFN with $d_\text{ff} \approx \tfrac{8}{3}d$, which is exactly why you see oddly-shaped numbers like $d_\text{ff} = 5632$ on a $d = 2048$ model ($5632 \approx \tfrac{8}{3}\cdot 2048 \cdot 1.03$) rather than a clean $8192$. SwiGLU is "buy a little accuracy with a slightly more complex block at the same param budget" — a small lever, but free.

**RMSNorm instead of LayerNorm.** LayerNorm subtracts the mean and divides by the standard deviation, then scales and shifts: $\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$. RMSNorm (Zhang & Sennrich, 2019) drops the mean-centering and the bias entirely, normalizing only by the root-mean-square: $\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}}$. It matches LayerNorm's quality in practice while doing *less arithmetic* (no mean, no subtraction, no bias) — and on an edge device, where you run the norm at every layer for every token and you are memory-bandwidth-bound, shaving a reduction and a vector of bias parameters is a genuine, if modest, latency and memory win. RMSNorm is "the same quality for less compute," which is exactly the trade an edge model wants.

**RoPE for positions.** Rotary position embeddings (Su et al., 2021) encode token position by *rotating* the query and key vectors by an angle proportional to their position before the attention dot product, so that the dot product between positions $m$ and $n$ depends only on the relative offset $m - n$. This matters for SLMs for two practical reasons. First, it adds *zero* parameters — unlike learned absolute position embeddings, which would be another table to store. Second, it extrapolates to longer contexts than it was trained on (especially with the now-standard scaling tricks), which lets a small phone model handle a longer document than its training context without retraining. RoPE is "relative positions, for free, with length-extrapolation as a bonus" — and "for free, in parameters" is the phrase that keeps coming up at small scale, because the parameter budget is the thing you are fighting for.

The upshot: a modern SLM is RMSNorm + SwiGLU + GQA + RoPE on a deep-and-thin trunk with tied embeddings. None of these is exotic, all of them are well-supported by on-device runtimes (which matters — Section 11 returns to op support), and each one is a small, *correct-for-small-scale* decision. The art is not inventing a new block; it is assembling the boring, well-trodden ones with the dials set for a billion parameters instead of seventy.

## 4. The KV-cache lever: grouped-query attention

If depth is the lever for accuracy-per-parameter, **grouped-query attention (GQA)** is the lever for *memory-per-token* at inference — and on a phone, memory-per-token is often the binding constraint, not weight size. To see why, we need the KV-cache math, which we cover in depth in the LLM-side companion on the [KV cache](/blog/machine-learning/large-language-model/kv-cache); here is the part that drives the design choice.

When a transformer decodes autoregressively, it keeps, for every past token and every layer, the **key and value vectors** so it does not have to recompute them. The size of this KV cache, in elements, is

$$
\text{KV elements} = 2 \cdot L \cdot n_\text{kv} \cdot d_\text{head} \cdot T
$$

where $L$ is layers, $n_\text{kv}$ is the number of key/value heads, $d_\text{head}$ is the per-head dimension, $T$ is the sequence length, and the $2$ is for keys *and* values. In standard multi-head attention (MHA), $n_\text{kv}$ equals the number of query heads $n_q$. The cache grows linearly with sequence length, and on a long-context decode it can rival or exceed the *weights* in memory footprint — which on a phone, where every megabyte of working set competes with the OS and the app, is exactly the wrong place to spend memory.

GQA, introduced by Ainslie et al. (2023) and now standard in Llama-2/3, Gemma, and most modern SLMs, breaks the assumption that $n_\text{kv} = n_q$. Instead it puts the query heads into $g$ **groups**, and all query heads in a group *share one key/value head*. So $n_\text{kv} = g$, with $g < n_q$. The query side keeps its full resolution (you still have $n_q$ distinct query projections, so the model can still attend in many different ways), but the key/value side is shared.

![Dataflow graph showing sixteen query heads mapped onto four shared key-value groups under grouped-query attention, shrinking the KV cache fourfold while quality drops less than one point](/imgs/blogs/small-language-models-by-design-5.png)

Figure 5 shows the mapping. Take a model with $n_q = 16$ query heads. Under MHA you keep 16 key/value heads and the cache is "16 wide." Under GQA with $g = 4$ groups, four query heads share each key/value head, you keep only 4 key/value heads, and the cache is **4× smaller** — a direct factor-$n_q/g$ reduction in KV memory and in the bandwidth you spend reading the cache every decode step. The remarkable empirical finding is that quality barely moves: across the literature, going from MHA to a sensible GQA grouping costs well under a point on most benchmarks, because the *query* side — where most of the attention expressivity lives — is untouched. (The extreme case $g=1$, where all heads share one KV head, is **multi-query attention**, MQA; it saves the most memory but is the riskiest for quality. Most SLMs pick $g$ in the 2–8 range as the sweet spot.)

#### Worked example: KV cache on a phone for a 1.1B SLM

Take a TinyLlama-class 1.1B model: $L = 22$ layers, $n_q = 32$ query heads, $d_\text{head} = 64$ (so $d = 2048$), and GQA with $n_\text{kv} = 4$. Decode a 2,048-token context in fp16 (2 bytes/element). The cache size is

$$
2 \cdot 22 \cdot 4 \cdot 64 \cdot 2048 \cdot 2\ \text{bytes} \approx 46\ \text{MB}.
$$

Now suppose the same model had shipped with full MHA, $n_\text{kv} = 32$. The cache becomes

$$
2 \cdot 22 \cdot 32 \cdot 64 \cdot 2048 \cdot 2\ \text{bytes} \approx 369\ \text{MB}.
$$

That is the difference between a 46 MB and a 369 MB working set *on top of* the ~640 MB of int4 weights. On a phone budgeting maybe 1.5 GB for the whole model process, the MHA version blows the budget the moment the context gets long, while the GQA version leaves comfortable headroom. GQA is not a quality lever; it is the lever that *makes long context affordable on device.* And because decode is memory-bandwidth-bound — each step reads the whole cache — a 8× smaller cache also means a meaningfully faster decode. (For the activation-side companion to this, quantizing the KV cache itself, see [activation quantization, SmoothQuant, and KV-cache quantization](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache).)

### 4.1 Why GQA speeds up decode, not just shrinks memory

It is worth being precise about *why* a smaller KV cache is faster, because it is a consequence of the roofline, the recurring hardware frame of this series (see [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives)). During autoregressive decode at batch=1, the model produces *one* token per step, which means it does a tiny amount of arithmetic but must *read* a large amount of state from memory: all the weights, plus the entire KV cache. The ratio of arithmetic to bytes-moved — the **arithmetic intensity** — is very low, so the step is **memory-bandwidth-bound**: its latency is set by how many bytes you move, not how many multiply-adds you do.

Per decode step, the attention sub-layer must read the full KV cache to compute attention over all past tokens. The bytes read for that are exactly the cache size, $2 \cdot L \cdot n_\text{kv} \cdot d_\text{head} \cdot T \cdot (\text{bytes/elem})$. Cut $n_\text{kv}$ by $8$ (GQA, $g=4$ from $n_q=32$) and you cut those bytes by $8$. Since the step is bandwidth-bound, *fewer bytes is directly less time* — the attention portion of decode gets up to $8\times$ cheaper on the cache-reading part. In practice the speedup is smaller than $8\times$ because the weight reads (which GQA also shrinks a little, but mostly leaves alone) and other layers still cost their fixed bandwidth, but at long context, where the cache rivals the weights, the end-to-end decode speedup from GQA is substantial. This is the mechanism behind the ~2× decode-speed gaps you will see in the results table: not fewer FLOPs, *fewer bytes moved per token.* On the edge, where you are almost always memory-bound at batch=1, "shrink the bytes you move per token" is the single most reliable way to go faster — and GQA, tied embeddings, and quantization are all, at bottom, the same move: **move fewer bytes.**

## 5. Where the parameters actually go: the embedding problem

Now the lever that surprises people the most when they first do the arithmetic: **at small scale, the embedding table is a huge fraction of the model.** This is not true at 70B, and that is exactly why design choices that "everybody knows don't matter" turn out to matter enormously when you shrink the model.

Do the accounting. The token embedding table has shape $V \times d$, where $V$ is the vocabulary size and $d$ is the hidden dimension. The output projection ("LM head" or "unembedding") that turns the final hidden state into logits over the vocabulary is *also* $V \times d$. If those two are separate weight matrices — **untied embeddings** — you pay $2 V d$ parameters for them. Compare that to the transformer stack's $\approx 12 L d^2$ from Section 3.

The ratio of embedding params to block params is

$$
\frac{P_\text{emb}}{P_\text{blocks}} \approx \frac{2 V d}{12 L d^2} = \frac{V}{6 L d}.
$$

Plug in numbers. A Llama-style tokenizer has $V \approx 32{,}000$. A 70B model has roughly $L = 80$, $d = 8192$, giving a ratio of about $32000 / (6 \cdot 80 \cdot 8192) \approx 0.008$ — the embeddings are well under 1% of the model, genuinely negligible. Now a 1.1B model: $L = 22$, $d = 2048$, ratio $\approx 32000 / (6 \cdot 22 \cdot 2048) \approx 0.12$ for a *single* table, so untied (two tables) it is roughly a quarter of the model. The exact same vocabulary that vanished at 70B eats a quarter of your billion-parameter budget.

![Vertical stack showing how a 1.1B parameter model splits its budget, with untied embeddings consuming about a quarter, feed-forward blocks about half, and attention projections the rest](/imgs/blogs/small-language-models-by-design-4.png)

Figure 4 lays out the full budget for a ~1.1B model with untied embeddings: roughly 26% in the two embedding tables, ~49% in the FFN blocks, ~24% in attention projections, and a rounding-error sliver in the norms. The headline is that the embeddings are the *second-largest* line item, ahead of attention. That is the structural fact that makes the next two design choices — tying embeddings and choosing the vocabulary size — first-order memory decisions rather than footnotes.

### 5.1 Tied embeddings: removing a whole copy of the biggest table

**Weight tying** (Press & Wolf, 2017; Inan et al., 2017) shares one matrix between the input embedding and the output projection. Instead of an input table $E \in \mathbb{R}^{V \times d}$ and a separate output table $U \in \mathbb{R}^{V \times d}$, you use $E$ for both: tokens come in via $E$, and logits go out as $h E^\top$. You have removed one full copy of the largest table.

![Before and after comparison of untied versus tied embeddings, showing two separate vocabulary tables collapsing into one shared table that saves about 145 million parameters in a 1.1B model](/imgs/blogs/small-language-models-by-design-6.png)

Figure 6 makes the saving concrete. With $V = 32{,}000$ and $d = 2048$, one table is $32000 \cdot 2048 \approx 65.5$M parameters; in a 1.1B model the *pair* is ~131M (and in a wider 2K-ish-$d$ design the per-table figure can be ~145M, as the figure labels). Tying removes one whole copy — on the order of 130–150M params, an ~12–13% cut to a billion-parameter model — for essentially **zero quality cost.** In fact tying often *helps* slightly, because the gradient signal from the output side regularizes the input embeddings and vice versa. This is why nearly every well-designed SLM (Gemma, TinyLlama, the Phi line, MobileLLM) ties its embeddings, while many large models leave them untied (at 70B the saving is a rounding error and the extra capacity of a separate head is mildly useful). Tying is the single highest-leverage "free" memory win at small scale.

### 5.2 The vocabulary-size trade

If embeddings cost $2Vd$ (or $Vd$ tied), then **vocabulary size $V$ is a direct memory dial** at small scale — and it pulls in two directions. A larger $V$ means each token covers more text on average (better *compression* of the input: fewer tokens per sentence, so faster generation and more effective context per KV-cache slot), but it grows the embedding table linearly. A smaller $V$ shrinks the table but means more tokens per sentence (slower generation, more KV-cache pressure, and the model has to spend layers re-composing sub-word fragments).

The trade is genuinely scale-dependent. At 70B, vocabulary is nearly free in parameters, so you push $V$ large (Llama-3 went to ~128K, Gemma to ~256K) to get the compression and multilingual coverage benefits. At 1B, a 256K vocabulary would be *catastrophic* — at $d=2048$ a single 256K table is $256000 \cdot 2048 \approx 524$M params, half your model in one table. This is the one place where SLMs sometimes can't simply copy the big model's choice: a phone-targeted 1B model often wants a *smaller, denser* vocabulary (32K–50K) precisely because the table cost is no longer negligible. Gemma-2B is the interesting counterexample — it inherits Gemma's 256K vocabulary, which is why a "2B" Gemma is really ~2.5B parameters with an enormous embedding table, and why its on-device footprint is heavier than its block count suggests. The lesson: **at small scale, choosing the tokenizer is choosing a chunk of your parameter budget**, and you should make that choice deliberately rather than inheriting it. (For the tokenizer design space itself, see the LLM-side treatment of [designing and choosing a tokenizer](/blog/machine-learning/large-language-model/designing-choosing-tokenizer-llm).)

There is a subtler second-order effect worth flagging, because it bites people who optimize vocabulary by table size alone. A larger vocabulary *compresses the input more* — the average number of tokens to encode a sentence falls — and that compression cascades through every downstream cost. Fewer tokens per sentence means fewer decode steps to generate a reply (faster, lower energy), fewer KV-cache entries per unit of text (less runtime memory), and more *effective* context fitting in a fixed token window. So a big vocabulary is not purely a parameter cost; it partly *pays for itself* at runtime by making each token do more work. The right way to evaluate the trade is in **bytes of useful behavior per megabyte of footprint**, not parameters in isolation: a 50K vocabulary that needs 1.3× as many tokens to say the same thing has effectively shrunk your context window and slowed your generation by 1.3×, and that runtime tax can outweigh the table savings on a chatty workload.

The practical calculus for a sub-2B on-device model usually lands here: pick the *smallest* vocabulary that still gives good compression on your actual target language and domain, then quantize the embedding table aggressively (it tolerates low precision better than the transformer weights, because a small embedding error on one token is easily corrected by later layers). A 32K vocabulary at 4-bit embeddings on a 1B model costs about $32000 \cdot 2048 \cdot 0.5\,\text{bytes} \approx 33$ MB on disk — small enough to forget about. A 256K vocabulary at the same 4 bits is $\approx 262$ MB, which is now a real line item competing with the rest of the model. The rule of thumb: **vocabulary size is a free choice at 70B and a budgeted choice at 1B; treat it like one.**

## 6. The inference-optimal training lever: TinyLlama and over-training

The fourth lever is not about the model's *shape* at all — it is about *how long you train it.* And it is the one that most directly reflects the fact that we are designing for the **edge**, where inference cost, not training cost, dominates the lifetime bill.

Start from the compute-optimal scaling result. The Chinchilla work (Hoffmann et al., 2022; see [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling)) found that, *for a fixed training-compute budget*, the best loss comes from scaling parameters and training tokens together at roughly $20$ tokens per parameter. By that rule, a 1.1B model "should" be trained on about 22B tokens, and training it on more than that is "wasteful" — you would have gotten lower loss by making the model bigger instead.

But Chinchilla optimizes the *training* bill. It says nothing about *inference*, and on the edge inference is the whole point. A model you ship to a million phones runs trillions of forward passes over its lifetime; the training compute is a one-time cost amortized to nothing. The relevant objective is not "lowest loss for a fixed training budget" but **"lowest loss for a fixed *inference* budget,"** and a fixed inference budget means a fixed model size (because size sets latency and memory on the target). Under *that* constraint, the right move is: **pick the model size your device can afford, then train it on as many tokens as you can — far past the compute-optimal 20×.** Every extra token of training buys a little more quality at *zero* inference cost. You happily overspend training compute to save inference compute forever after. This is exactly the regime formalized in the [inference-aware scaling laws](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) post: when inference demand is high, the optimal model is smaller and *over-trained* relative to Chinchilla.

TinyLlama (Zhang et al., 2024) is the cleanest demonstration of this lever. It is a 1.1B model — Chinchilla would budget it ~22B tokens — trained on **3 trillion tokens**, roughly **130× past compute-optimal.** That is wildly "wasteful" by the training-bill logic and exactly right by the inference-bill logic. The payoff is a 1.1B model whose quality is far better than a 22B-token 1.1B would be, packed into a footprint a phone can hold. It will not beat a 7B on broad knowledge — over-training has diminishing returns and cannot manufacture capacity the parameters don't have — but it extracts close to the maximum quality a 1.1B *can* hold, which is precisely what you want when 1.1B is the size you're stuck with.

Gemma-2B (Google, *Gemma Technical Report*, 2024) follows the same playbook from the other direction — a ~2.5B model trained on ~3T tokens of web and code, also far past compute-optimal — and it is why these small models feel so much more capable than the "small models" of a couple years ago. The models didn't get a clever new architecture so much as they got *trained much longer for their size*, because the field finally optimized for the inference bill that the edge actually pays.

One honest qualification keeps over-training from becoming a blank check. The returns *diminish*: the loss-versus-tokens curve flattens, so the thousandth billion tokens buys far less than the first. At some point the model is close enough to the best loss its parameters can hold that more tokens barely move it, and you would have been better off spending that training compute on a slightly bigger model — but only if the device can afford the bigger model at inference, which on the edge it usually can't. So the discipline is: fix the size to what the *target device* can run, then ride the token curve until the marginal quality per training-dollar drops below your threshold, and stop. TinyLlama's 3T tokens on a 1.1B model is well into the flat part of the curve for that size, which is exactly the point — it is extracting the *last* available quality from a 1.1B, because 1.1B is the size the target can hold and there is no point leaving quality on the table when the inference cost is already fixed. The mental shift is from "how big a model can I afford to train" to "how good can I make the model I can afford to *run*," and over-training is the lever that answers the second question.

## 7. Putting the levers together: load, inspect, quantize, run

Enough theory. Let's load a small model, *see* the design choices in its config, quantize it, and run it. We'll use Hugging Face `transformers` to inspect, `bitsandbytes` to quantize for a quick GPU/laptop run, and `llama.cpp` for the real on-device path. (Quantization mechanics live in the [GPTQ/AWQ post](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq); here we *compose* it with the small-by-design model.)

### 7.1 Inspecting the design choices in code

First, load TinyLlama and read its architecture straight off the config — the GQA grouping, the tied embeddings, the SwiGLU/RMSNorm choices are all right there.

```python
import torch
from transformers import AutoModelForCausalLM, AutoConfig

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
cfg = AutoConfig.from_pretrained(model_id)

print("layers (depth)        :", cfg.num_hidden_layers)        # 22 -> deep-and-thin
print("hidden size (width)   :", cfg.hidden_size)              # 2048
print("query heads  n_q      :", cfg.num_attention_heads)      # 32
print("kv heads     n_kv     :", cfg.num_key_value_heads)      # 4  -> GQA, group = 8
print("ffn intermediate      :", cfg.intermediate_size)        # 5632 (SwiGLU)
print("vocab size  V         :", cfg.vocab_size)               # 32000
print("tied embeddings?      :", cfg.tie_word_embeddings)      # True
print("norm                  :", type(cfg).__name__, "uses RMSNorm")
print("activation (mlp)      :", cfg.hidden_act)               # silu -> SwiGLU gate
```

Every line of that output is a design decision from the sections above. `num_key_value_heads = 4` against `num_attention_heads = 32` is GQA with a group size of 8 — the KV cache is 8× smaller than MHA. `tie_word_embeddings = True` means one shared table, saving ~65M params. `hidden_act = "silu"` plus the gated `intermediate_size` is SwiGLU. `num_hidden_layers = 22` at `hidden_size = 2048` is the deep-and-thin lean. You can read the whole SLM playbook off six config fields.

### 7.2 Verifying the parameter budget empirically

Don't trust my arithmetic — count the parameters and confirm where they go. This script reproduces the Figure-4 breakdown for any model.

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16
)

buckets = {"embed": 0, "attn": 0, "ffn": 0, "norm": 0, "other": 0}
for name, p in model.named_parameters():
    n = p.numel()
    if "embed_tokens" in name or "lm_head" in name:
        buckets["embed"] += n
    elif any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
        buckets["attn"] += n
    elif any(k in name for k in ("gate_proj", "up_proj", "down_proj")):
        buckets["ffn"] += n
    elif "norm" in name:
        buckets["norm"] += n
    else:
        buckets["other"] += n

total = sum(buckets.values())
for k, v in buckets.items():
    print(f"{k:6s}: {v/1e6:7.1f}M  ({100*v/total:4.1f}%)")
print(f"total : {total/1e6:7.1f}M")
```

On TinyLlama this prints embeddings as a single ~65.5M table (because tying means `embed_tokens` and `lm_head` *share* the same storage — PyTorch counts it once), the FFN as the dominant block, and attention noticeably smaller than the FFN. If you load a model with `tie_word_embeddings = False`, you will see the embedding bucket roughly double — that is the cost of *not* tying, visible in one number. This is the most convincing way to internalize Section 5: run it on a tied and an untied model and watch the embedding line change.

### 7.3 Quick quantized run with bitsandbytes

For a laptop with a GPU, the fastest path to a 4-bit run is `bitsandbytes` NF4. This composes the small-by-design model with the quantization lever in three lines.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # information-optimal 4-bit for normal weights
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,     # quantize the quant constants too
)

tok = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", quantization_config=quant_cfg, device_map="auto"
)

prompt = "Explain why a deeper, thinner model can beat a wider one at the same size:"
ids = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

Phi-2 at NF4 is ~1.5 GB of weights and runs comfortably on a laptop GPU or a recent MacBook. NF4 (the "normal float 4" datatype) is worth a sentence of *why*: it places its 16 quantization levels to match the quantiles of a normal distribution, which is the empirical shape of neural-net weights, so it spends bits where the weights actually are — strictly better than uniform int4 for the same 4 bits.

### 7.4 The real on-device path: llama.cpp and GGUF

For an actual phone or a CPU-only deploy, `bitsandbytes` is the wrong tool — it targets CUDA. The on-device standard is `llama.cpp` with GGUF k-quants, which run on ARM CPUs, Metal, and small GPUs. The flow is convert → quantize → run.

```bash
# 1. Convert the HF checkpoint to GGUF (fp16)
python convert_hf_to_gguf.py ./TinyLlama-1.1B-Chat-v1.0 \
    --outfile tinyllama-1.1b-f16.gguf --outtype f16

# 2. Quantize to a 4-bit k-quant. Q4_K_M is the standard quality/size sweet spot.
./llama-quantize tinyllama-1.1b-f16.gguf tinyllama-1.1b-q4km.gguf Q4_K_M

# 3. Run it. -ngl offloads layers to GPU/Metal; drop it for pure CPU.
./llama-cli -m tinyllama-1.1b-q4km.gguf -ngl 99 -c 2048 \
    -p "Write a haiku about a model that fits on a phone." -n 64
```

`Q4_K_M` keeps the most sensitive tensors (the attention `q`/`v` projections and the output, where errors hurt most) at a slightly higher bit-width and the rest at 4 bits — a mixed-precision scheme inside the quantizer that buys back quality cheaply (the principle is the same one in [mixed-precision and sensitivity analysis](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis)). The resulting `tinyllama-1.1b-q4km.gguf` is ~640 MB and decodes at tens of tokens/s on a modern phone CPU and well over a hundred on a laptop. This is the file you actually ship.

## 8. Worked examples with concrete numbers

#### Worked example: param budget of a 1.1B model and what tying plus GQA save

Let's account for a TinyLlama-class model end to end: $L = 22$, $d = 2048$, $V = 32{,}000$, FFN intermediate $d_\text{ff} = 5632$ (SwiGLU uses three matrices: gate, up, down), $n_q = 32$, $n_\text{kv} = 4$, $d_\text{head} = 64$.

**Per-layer attention** (GQA): query projection $d \times d = 2048 \times 2048 \approx 4.19$M; key and value each $d \times (n_\text{kv} \cdot d_\text{head}) = 2048 \times 256 \approx 0.52$M, so 1.05M together; output $d \times d \approx 4.19$M. Attention per layer $\approx 9.4$M. (Under MHA, K and V would each be $2048 \times 2048$, adding ~6.3M per layer — GQA already saved parameters *and* the KV cache.)

**Per-layer FFN** (SwiGLU, three matrices): gate and up each $d \times d_\text{ff} = 2048 \times 5632 \approx 11.5$M, down $d_\text{ff} \times d \approx 11.5$M, total $\approx 34.6$M per layer.

**Per layer total** $\approx 9.4 + 34.6 = 44.0$M. Over 22 layers: $\approx 968$M.

**Embeddings**: one table $V \times d = 32000 \times 2048 \approx 65.5$M. **Tied**, so we pay it once: 65.5M. **Untied** we would pay 131M.

**Grand total (tied)** $\approx 968 + 65.5 \approx 1.03$B — which rounds to the advertised 1.1B once you add norms and rounding. Now the savings ledger:

| Design choice | Parameters saved | % of model | Quality cost |
| --- | --- | --- | --- |
| Tie embeddings | ~65.5M | ~6% (of ~1.1B) | ~0 (often helps) |
| GQA vs MHA (this model) | ~140M in weights | ~13% | < 1 pt |
| GQA KV cache (2K ctx) | ~323 MB at runtime | 8× cache | < 1 pt |

Read it together: tying plus GQA removed roughly **200M parameters of weights** — about a fifth of a billion-parameter budget — for under a point of quality, *and* GQA shrank the runtime KV cache 8×. That freed budget is what got *spent on more layers* (depth) and *more training tokens* (over-training). The levers compound: the param-savers fund the accuracy-buyers.

#### Worked example: SLM at int4 versus a quantized 7B at matched memory on a laptop

The decision that started this post, made concrete on a named target — an **M2 MacBook Air, 16 GB**, running `llama.cpp` with Metal, batch=1, measured warm (after a 20-token warm-up to fill caches and let the GPU clock up), reporting median over 5 runs of a 256-token generation. These are representative figures consistent with published `llama.cpp` benchmarks; treat them as order-of-magnitude, not lab-certified.

| Model | Quant | Weights on disk | Peak RSS (2K ctx) | Decode tok/s | MMLU (approx) |
| --- | --- | --- | --- | --- | --- |
| Phi-2 (2.7B) | Q4_K_M | ~1.6 GB | ~2.0 GB | ~75 | ~58–61 |
| Llama-2-7B | Q4_K_M | ~4.1 GB | ~4.6 GB | ~38 | ~46 |
| Llama-2-7B | Q3_K_S | ~2.9 GB | ~3.4 GB | ~44 | ~43 (drops) |
| Mistral-7B | Q4_K_M | ~4.4 GB | ~4.9 GB | ~36 | ~62 |

Two readings. First, the *clean* matched-memory comparison: Phi-2 at ~1.6 GB beats Llama-2-7B by ~12 MMLU points while taking less than half the memory and decoding ~2× faster. The small-by-design model is strictly Pareto-superior to that particular 7B. Second, the *honest* counterexample: Mistral-7B is a *better-designed* 7B (it has its own architecture and data advantages) and at Q4 it edges Phi-2 on MMLU — but it costs ~3× the memory and decodes half as fast. So the rule is not "small always wins." The rule is **small-by-design wins decisively against a *naively* trained big model at matched memory, and trades memory for a few points against a *well-designed* big model.** When the device budget is the hard constraint — which on the edge it always is — that trade usually favors the SLM. If you must squeeze the 7B harder to fit (the Q3_K_S row), it loses points *and* you are now in the fragile end of quantization. The SLM never had to make that compromise.

#### How to measure this honestly

Those numbers are easy to fake without meaning to, so a word on how to produce them so they survive a code review. Decode throughput must be measured **warm** — the first few tokens of a fresh process pay for paging the weights in from disk, JIT-compiling Metal/CUDA kernels, and ramping the GPU clock, so a cold first run under-reports steady-state speed by a lot. Discard a ~20-token warm-up, then time a fixed-length generation and report the **median** of several runs, not the best (the best is luck; the median is what a user feels). Pin the **context length** and the **batch size** explicitly, because both change the answer: throughput at 2K context with a full KV cache is lower than at 128 tokens, and batch=1 (the on-device reality) is a different regime from the batched server numbers vendors love to quote. Watch for **thermal throttling** on a phone or a fanless laptop — a 30-second benchmark and a 10-minute benchmark can differ by 30% as the SoC heats up and down-clocks, so state your measurement window. And separate **prefill** (processing the prompt, which is compute-bound and fast per token) from **decode** (generating, which is memory-bound and slow per token); quoting a single "tokens/s" that blends them hides the number that actually governs interactive latency. Report **peak resident memory** (RSS), not just the weight file size, because the KV cache, the activation buffers, and the runtime's own overhead all count against the device budget — the weight file is the floor, not the footprint.

#### Worked example: choosing between Gemma-2B and a distilled 1B for an on-device summarizer

A concrete decision. You are shipping an on-device email summarizer to a mid-range phone with a hard **1 GB** model-process budget (the rest of the RAM belongs to the OS and the email app). The task is narrow — summarize a thread into three bullets — and latency-sensitive (the user is waiting). Two candidates.

*Gemma-2B at Q4_K_M:* weights ~1.5 GB, peak RSS at a 2K context ~1.9 GB. It is a strong general model, but it **does not fit** the 1 GB budget — full stop. You would have to push it to Q3 or lower, dropping it to ~1.0–1.1 GB of weights but at a quality cost and into fragile quantization territory, and the RSS with the KV cache would still threaten the budget on a long thread. Broad capability you cannot afford to load is worth nothing.

*A 1.1B model distilled on summarization, at Q4_K_M:* weights ~640 MB, peak RSS at 2K context ~870 MB — **fits with headroom.** On its own, a base 1.1B summarizes worse than Gemma-2B. But distilled from a strong teacher *on summarization specifically* (training the 1B on the teacher's summaries of a large thread corpus), it closes most of the gap *on this task* while staying in budget. It decodes ~2× faster than Gemma-2B would, so the user waits half as long, and it uses proportionally less battery per summary.

The decision is the SLM, and the reasoning is the whole post in miniature: the task is narrow (so breadth is wasted), the budget is hard and sub-1.5 GB (the SLM's home turf), latency matters (the small model is ~2× faster), and you can retrain (so you distill to lift the narrow-task quality). Every arrow in the Figure 8 decision tree points the same way. The only thing that would flip it is if the product scope widened from "summarize" to "answer arbitrary questions about the email," at which point the 1B's knowledge ceiling starts to bind and you would either pay for the memory of a bigger model or add a retrieval/escalation path — which is exactly the limit Section 11 takes up.

## 9. Case studies and real numbers from the literature

A few load-bearing results from the papers, stated as carefully as I can.

- **Phi-1 (Gunasekar et al., 2023):** a 1.3B model trained on ~7B tokens of filtered-plus-synthetic "textbook-quality" code data reached ~50.6% pass@1 on HumanEval — competitive with or beating models 10× larger trained on raw web code. The headline lesson, robust to the contamination debate, is that *data density is a primary lever at small scale.* Phi-1.5 (1.3B) and Phi-2 (2.7B) carried the recipe to general reasoning, with Phi-2 reaching scores competitive with 7B–13B models on several reasoning benchmarks.

- **MobileLLM (Liu et al., 2024):** at the 125M and 350M scales, deep-and-thin layouts plus embedding sharing and GQA gave consistent zero-shot accuracy gains over wider baselines of equal size; the 125M/350M models beat prior sub-billion models by ~2–4 points on average, and a block-weight-sharing variant extended depth without adding parameters. The lesson: *architecture shape is a real lever below 1B, where it is nearly flat above 1B.*

- **TinyLlama (Zhang et al., 2024):** a 1.1B model trained on ~3T tokens — ~130× past Chinchilla-optimal — demonstrating the over-training-for-inference lever in the open. It does not rival 7B models on broad knowledge but extracts strong quality for its size, validating that *you over-train a small model when inference, not training, is the bill you pay.*

- **Gemma (Google, 2024):** the 2B (~2.5B with its 256K vocabulary) and 7B models trained on 2–6T tokens with tied embeddings, GQA, RMSNorm, and GeGLU. Gemma-2B is the standing example that a phone-targeted SLM can ship with a *large* vocabulary if you accept the embedding-table cost — a deliberate trade in the opposite direction from TinyLlama's lean 32K vocab. It also shows how the same design family scales from a phone target to a server target by turning the depth/width and vocabulary dials.

The throughline across all four: they pull *different* levers (data, shape, training-tokens, vocabulary) but they all land in the same sub-3B, sub-1.5-GB-at-Q4 envelope that a phone can hold — which is exactly the family comparison in the next figure.

![Matrix comparing Phi-2, Gemma-2B, TinyLlama, and MobileLLM across parameters, training tokens, MMLU, and on-device memory at int4, showing each pulls a different design lever yet all fit a phone budget](/imgs/blogs/small-language-models-by-design-7.png)

Figure 7 is the family at a glance. Each row names the model's *primary lever* — Phi-2's data quality, Gemma-2B's balanced web-plus-code at large vocab, TinyLlama's over-training, MobileLLM's deep-thin-plus-GQA-plus-tied architecture — and each lands in the 600 MB–1.5 GB Q4 band. The numbers are approximate and version-dependent (treat MMLU figures as ballpark), but the *shape* of the table is the point: there is no single winning recipe, only a set of levers you pull to taste for your target. A model for a flagship phone with 8 GB can afford Gemma-2B's vocabulary; a model for a wearable or a tight memory budget wants TinyLlama's or MobileLLM's lean.

### A design-choice impact summary

To consolidate, here is each lever against what it buys, what it costs, and where it sits in the build:

| Lever | What it buys | What it costs | When to pull |
| --- | --- | --- | --- |
| Textbook-quality data | More quality per param; biggest at small scale | Corpus-building effort; benchmark-fit risk | Always at sub-3B |
| Deep-and-thin shape | +2–4 pts acc per param under 1B | Higher decode latency (sequential layers) | Sub-1B, when accuracy-bound |
| GQA | 4–8× smaller KV cache, faster decode | < 1 pt quality | Always; pick g=2–8 |
| Tied embeddings | ~6–13% fewer params, free | Negligible | Always at small scale |
| Over-training (TinyLlama) | Max quality for a fixed size | Large training compute (one-time) | Always for edge (inference-bound) |
| Smaller vocabulary | Smaller embedding table | More tokens/sentence, slower gen | Tight budgets (< ~1.5B) |
| Distillation into the SLM | Lifts the reasoning ceiling on target tasks | Needs a teacher + distill run | When the SLM hits its ceiling |

Notice every "buys" in that table is one of the four series levers (efficient architecture, plus quantization and distillation composing in), validated against the accuracy–efficiency Pareto frame from the [taxonomy post](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression). The SLM playbook is not a separate technique; it is the *architecture lever pulled at design time*, with the others stacked on top.

## 10. When to reach for an SLM (and when to quantize a big model instead)

Now the decision the whole post has been building toward. You have a sub-3B on-device target and two roads: ship a model designed small, or quantize a bigger one down to fit. Which?

![Decision tree routing an on-device model choice by memory budget and task breadth toward a native small model, a distilled small model, or a quantized large model](/imgs/blogs/small-language-models-by-design-8.png)

Figure 8 is the decision tree, and the two questions are **how tight is the budget** and **how broad is the task.**

**Reach for a small-by-design model when:**

- **The memory budget is hard and under ~1.5 GB.** Below this, quantizing a 7B forces you into Q3 or lower, where quantization gets fragile and you lose points. A native 1–2B at Q4 sits in its sweet spot. Budget-bound is the SLM's home turf.
- **The task is narrow.** If you are doing on-device summarization, a specific extraction task, command parsing, or a single vertical, you do not need a 7B's broad world knowledge. A small model — ideally one *distilled* from a big teacher on exactly your task distribution (see [distilling LLMs and reasoning](/blog/machine-learning/edge-ai/distilling-llms-and-reasoning)) — will match it on the task you care about at a fraction of the footprint.
- **Latency or energy is the binding constraint.** A 1B model decodes ~2–3× faster and uses proportionally less energy per token than a quantized 7B. On a battery, that is the difference between a feature and a footnote.
- **You control training and the model will run a lot.** If you can over-train a small model and amortize that one-time cost over billions of inferences, the inference-optimal math (Section 6) says do it.

**Quantize a bigger model instead when:**

- **The task needs broad knowledge or open-domain reasoning.** A 1–2B model has a real *knowledge ceiling* — it simply cannot store what a 7B stores. For a general assistant that must answer arbitrary questions, the big model's breadth wins even after quantization, provided you have the memory.
- **You have the memory headroom (≥ 4 GB for the model process).** If a Q4 7B fits comfortably, and you need its breadth, take it. The matched-memory argument only bites when memory is the constraint.
- **You can't retrain.** If you only have access to an existing big checkpoint and no training budget, quantization is your only lever; designing a small model from scratch is off the table.
- **A *well-designed* big model exists for your domain.** As the Mistral-7B row showed, a good 7B at Q4 can out-score a small model on broad benchmarks — at a memory cost. If you can pay it, that quality may be worth it.

The synthesis rule, and the honest one: **small-by-design dominates a naively-trained big model at matched memory, and trades a few points of broad-knowledge quality for large savings in memory, latency, and energy against a well-designed big model.** On the edge, where memory and energy are the hard walls, that trade usually favors the SLM — *unless the task genuinely needs breadth the small model cannot hold.*

## 11. The limits of small models, and how distillation lifts them

I have been bullish, so let me be honest about the ceiling, because every deployment that goes wrong with an SLM goes wrong for one of these reasons.

**Knowledge is capacity-bound.** A model stores facts in its weights, and a 1B model has ~1B weights' worth of room. It will not know the obscure tail of facts a 70B knows, and no amount of clever training fixes that — over-training raises the floor toward the model's capacity ceiling, it does not raise the ceiling. If your application asks open-domain factual questions across a long tail, a small model will confabulate where a big one would recall. The mitigation is architectural, not parametric: pair the SLM with **retrieval** so the facts live in an external store and the model only has to *reason over* retrieved text, not *memorize* it. A 1B model with good retrieval beats a 7B without it on knowledge-heavy tasks, because you moved the knowledge out of the weights.

**Reasoning has a depth ceiling.** Multi-step reasoning needs sequential refinement steps (Section 3), and a small model has a finite number of layers. Past some reasoning depth, it cannot compose further regardless of prompting. You see this as a sharp cliff on hard multi-step benchmarks: small models do fine to a certain difficulty and then fall off. Two mitigations. First, **test-time compute** — let the small model think longer (chain-of-thought, self-consistency, sampling) to trade inference time for effective depth, which works surprisingly well and is itself a scaling lever (see [test-time compute scaling](/blog/machine-learning/scaling-laws/test-time-compute-scaling)). Second, and more powerful, **distillation**: train the small model on the *reasoning traces* of a big teacher, so it learns the teacher's reasoning *patterns* even though it can't match the teacher's capacity. This is the single most effective way to lift a small model's reasoning on a target distribution, and it is exactly what the [distilling LLMs and reasoning](/blog/machine-learning/edge-ai/distilling-llms-and-reasoning) post covers — the natural next lever to stack once you have squeezed the architecture.

**Calibration and tail behavior get worse.** Small models are more confidently wrong, hallucinate more on out-of-distribution inputs, and have noisier instruction-following. This is partly capacity and partly that small models have less "slack" to represent uncertainty. Budget for guardrails: constrained decoding for structured outputs, a verifier or a small re-ranker, and a confidence gate that escalates hard cases to a server model when you have connectivity. A well-architected edge system is rarely "the SLM alone"; it is the SLM as the fast, private, always-available *first responder*, with a fallback path for the cases it cannot handle.

**The stress tests, run honestly.** What happens at int4? The SLM holds up well — small models with GQA and clean training quantize *better* than big models, because they have fewer pathological outlier activations to begin with. What happens when the context gets long? GQA keeps the KV cache affordable, but at very long context the cache still grows linearly and eventually dominates; that is when you reach for KV-cache quantization or sliding-window attention (Gemma uses local-global attention partly for this). What happens when the NPU doesn't support an op? SwiGLU and RMSNorm are well-supported on modern delegates, but a custom attention variant can fall back to CPU and tank latency — so favor *standard* building blocks for on-device portability, which is itself an argument for the boring, well-trodden SLM stack (RMSNorm + SwiGLU + GQA + RoPE) over anything exotic. What happens when the model is memory-bound, not compute-bound? On a phone decode it almost always is — batch=1 means you read all the weights to produce one token, so you are bandwidth-limited, which is *why* shrinking the model (and the KV cache) helps more than shaving FLOPs. The SLM design playbook is, at its core, a memory-bandwidth playbook.

## 12. Key takeaways

- **Design small, don't just shrink big.** At a matched on-device memory budget, a model designed small from the start — better data, deeper-thinner shape, GQA, tied embeddings, over-trained — beats a big model quantized to fit, *unless* the task genuinely needs the big model's broad knowledge.
- **Data quality is the first lever at small scale.** Curated and synthetic "textbook-quality" tokens (Phi) let a 1–3B model punch an order of magnitude above its size, because clean data spends the model's scarce capacity on capability instead of denoising.
- **Below 1B, deep-and-thin beats wide-and-shallow.** Sequential refinement steps (layers) buy compositional reasoning that per-layer width cannot manufacture; MobileLLM measured the +2–4 points.
- **GQA is the KV-cache lever, not a quality lever.** Sharing key/value heads across query groups shrinks the per-token cache 4–8× for under a point of quality — and on a phone the KV cache, not the weights, is often the binding memory constraint.
- **At small scale, embeddings are huge.** Untied embeddings can be a quarter of a 1B model; **tie them** for an ~6–13% free param cut, and **choose the vocabulary size deliberately** because the table cost is no longer negligible.
- **Over-train for the inference bill.** On the edge, inference dominates lifetime cost, so pick the size your device affords and train it far past Chinchilla-optimal (TinyLlama: ~130×). You overspend training once to save inference forever.
- **Quantization composes on top, it doesn't replace the design.** Ship the GGUF Q4_K_M of a small-by-design model; the small model quantizes *better* than a big one because it has fewer outliers to begin with.
- **Know the ceiling and lift it with distillation and retrieval.** Small models are capacity-bound on knowledge and depth-bound on reasoning; move knowledge to retrieval and lift reasoning by distilling a big teacher's traces into the student.

## Further reading

- Gunasekar et al., *Textbooks Are All You Need* (2023) — the Phi-1 data-quality thesis; and the Phi-1.5 / Phi-2 follow-ups extending it to reasoning.
- Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases* (2024) — deep-and-thin, weight sharing, GQA at sub-1B.
- Zhang et al., *TinyLlama: An Open-Source Small Language Model* (2024) — 1.1B over-trained on 3T tokens, the inference-optimal lever in the open.
- Google, *Gemma: Open Models Based on Gemini Research and Technology* (Gemma Technical Report, 2024) — the 2B/7B design choices, large vocabulary, tied embeddings, GQA.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023) — the grouped-query attention method and the KV-cache math.
- Press & Wolf, *Using the Output Embedding to Improve Language Models* (2017) — weight tying.
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022) — the compute-optimal baseline the edge deliberately overshoots.
- Within this series: the [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) (the four-lever Pareto frame), [weight-only LLM quantization with GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq) (the lever you stack on top), [distilling LLMs and reasoning](/blog/machine-learning/edge-ai/distilling-llms-and-reasoning) (lifting the small-model ceiling), [efficient attention and vision transformers for the edge](/blog/machine-learning/edge-ai/efficient-attention-and-vision-transformers-for-edge) (the attention-side companion), and the capstone [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).
- Out of series: the [KV cache](/blog/machine-learning/large-language-model/kv-cache) and [inference-aware scaling laws](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) for the memory and over-training arguments in full.

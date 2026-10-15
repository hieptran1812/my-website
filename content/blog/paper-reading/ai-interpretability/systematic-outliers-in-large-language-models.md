---
title: "Systematic Outliers in Large Language Models: Why Softmax Grows Giant Numbers"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How the three kinds of LLM outliers (weights, activations, attention) are one connected mechanism, why softmax's sum-to-one rule creates them, and how a tiny context-aware scaling factor removes them."
tags: ["paper-reading", "interpretability", "outliers", "attention-sink", "massive-activations", "softmax", "quantization", "large-language-model", "iclr-2025-poster", "iclr-2025"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 18
paper:
  title: "Systematic Outliers in Large Language Models"
  authors: "Yongqi An, Xu Zhao, Tao Yu, Ming Tang, Jinqiao Wang"
  venue: "ICLR 2025"
  url: "https://arxiv.org/abs/2502.06415v2"
---

> [!tldr]
> **Claim.** The huge values that wreck LLM quantization come in three flavors: weight outliers, activation outliers and attention outliers. An et al. show they are not three bugs but one mechanism, appearing at the same tokens and the same channels.
>
> **Why it matters.** The mechanism points at a single root cause: softmax attention must always hand out 100% of its weight, so a token that wants "no update" has to park its attention somewhere harmless. Building that parking spot takes enormous logits, and the network grows giant activations and weights to make them.
>
> **Most surprising result.** Multiplying each head's output by a learned per-token gate in (0, 2), costing under 0.1% extra parameters, makes the outliers vanish in GPT-2. Perplexity under 8-bit AbsMax weight quantization drops from 93.44 to 29.22, and under 50% magnitude pruning from 7235.68 to 39.47.
>
> **Where it is thin.** The causal story is argued, not proven; the intervention is tested on a 124M GPT-2 and a 120M TinyLLaMA trained for about 2B tokens, and final validation loss does not improve.

If you have ever quantized a LLaMA model to 8 bits and watched perplexity explode, you have met these numbers. A handful of hidden channels carry values in the thousands while everything else sits near 1. Quantization papers treat them as weather: something to route around with per-channel scales, rotations or mixed precision.

This paper asks the meteorologist's question instead. *Why* do these values exist, where exactly do they sit, and what job are they doing? The answer it builds is satisfying because it is mechanical. Once you see it, the outliers stop looking like noise and start looking like a workaround the model invented for a limitation in softmax.

## The three outliers, defined

The paper starts by giving each kind of outlier a precise definition. All three share one idea: a value is an outlier if its magnitude exceeds $\tau$ times the average magnitude of its population. The authors set $\tau = 1000$, so these are not mild deviations; they are values a thousand times the typical size.

**Activation outliers.** For a layer output $h_\ell \in \mathbb{R}^{B \times H}$ ($B$ tokens in the batch, $H$ hidden channels):

$$
O_{\text{activation}} = \{(i, j) \;:\; |h_{i,j}| > \tau \cdot \mu_h\}, \qquad \mu_h = \frac{1}{B \cdot H} \sum_{i,j} |h_{i,j}|
$$

Here $(i, j)$ is a (token, channel) pair and $\mu_h$ is the mean absolute activation over the whole matrix. The same definition is applied to $x^{\text{down}}_\ell$, the input to the MLP's down-projection.

**Weight outliers.** For a projection matrix $W \in \mathbb{R}^{O \times I}$, the threshold is row-relative:

$$
O_{\text{weight}} = \{(i, j) \;:\; |w_{i,j}| > \tau \cdot \mu_{w_i}\}, \qquad \mu_{w_i} = \frac{1}{I} \sum_{j} |w_{i,j}|
$$

**Attention outliers.** For an attention matrix $A \in \mathbb{R}^{L \times L}$ over $L$ tokens, sum each column to get how much total attention key $j$ receives, $\hat{A}_j = \sum_{i=1}^{L} A_{i,j}$. Key $j$ is an outlier if $\hat{A}_j > \tau \cdot \mu_A$, where $\mu_A$ is the mean of the $\hat{A}_j$.

A tiny example makes the attention version concrete. In a sequence of $L = 4$ tokens with causal attention, the four query rows each sum to 1, so the column sums total 4 and $\mu_A = 1$. If every query puts 0.9 of its weight on token 0, then $\hat{A}_0 \approx 3.6$, far above its fair share. That column is the "sink" the rest of this post keeps returning to.

![Figure 1 from An et al. (2025): systematic outliers in LLaMA2-7B in four places: (a) down-projection inputs, (b) down-projection weights, (c) layer outputs, (d) attention weights](/imgs/blogs/systematic-outliers-in-large-language-models-fig1.webp)

Figure 1 shows all four locations on the probe sentence "Summer is warm. Winter is cold." Panels (a) and (c) are tall spikes on a flat floor. Panel (b) is a weight matrix with a few bright columns. Panel (d) is an attention map where almost every query row lights up the same few columns.

## Where they sit: a few tokens, a few channels, a few layers

The authors profile LLaMA2-7B on 100 RedPajama sequences of length 2,048. Three questions, three answers.

| Question | Activation outliers in $h_\ell$ | In $x^{\text{down}}_\ell$ | Weight outliers | Attention outliers |
| --- | --- | --- | --- | --- |
| Which layers? | Appear in shallow layers, persist through the middle, fade at the end | Only a few shallow and deep layers | $W^{\text{down}}$ of layer 2 and the last two layers | Every layer |
| Which tokens? | Start token (50.0%), "." (40.5%), "\_" (9.5%) | Start and weak-semantic tokens | n/a | Start and weak-semantic tokens |
| Which channels? | A fixed handful (1415 and 2533) | A fixed handful | Specific columns | n/a (head-dependent) |

![Figure 3 from An et al. (2025): distribution of activation outliers in layer outputs across layers, tokens and channels in LLaMA2-7B](/imgs/blogs/systematic-outliers-in-large-language-models-fig3.webp)

The token breakdown is the striking part. Half of all activation outliers sit on the initial token, and almost all the rest sit on a period or a space. These are exactly the tokens that carry the least meaning. Whatever the model is doing with these numbers, it is not storing content in them.

The picture below is a cartoon of that structure. The values are illustrative, but the shape is the paper's: the grid is almost entirely flat, with a handful of enormous cells at the intersection of a start or weak token and one of two fixed channels.

![Illustrative heatmap of token by channel activations: almost all cells near 1, with values in the thousands only where the tokens Summer and the first period meet channels 1415 and 2533](/imgs/blogs/systematic-outliers-in-large-language-models-1.webp)

For weights, the paper uses an "extremal ratio": the maximum over the mean within each column. Columns matter because a large value in column $j$ of a projection directly scales output channel $j$. Across all seven projection types (q, k, v, o, gate, up, down), only `down_proj` in layer 2 and layers 31 and 32 spike, with ratios above 1,000.

The appendix repeats the profile on Phi-2, Mistral-7B, LLaMA2-13B, LLaMA3-8B, OPT-6.7B, MPT-7B and Falcon-7B, plus chat and instruct fine-tunes. The outliers appear in all of them. OPT's ReLU MLP spreads them more loosely than gated models do, and fine-tuning barely moves them, which the authors read as evidence of an architectural origin rather than a data artifact.

## They are one thing: the consistency table

If the three outlier kinds were independent, their locations would overlap only by chance. Table 1 of the paper measures the overlap directly.

| Outlier type 1 | Outlier type 2 | Shared dimension | Consistency |
| --- | --- | --- | --- |
| Weight outliers in $W^{\text{down}}_\ell$ | Activation outliers in $x^{\text{down}}_\ell$ | channel | 100% |
| Weight outliers in $W^{\text{down}}_\ell$ | Activation outliers in $h_\ell$ | channel | 100% |
| Activation outliers in $x^{\text{down}}_\ell$ | Activation outliers in $h_\ell$ | token | 100% |
| Activation outliers in $h_\ell$ | Attention outliers in $A^i_\ell$ | token | 95% |

The attention overlap is computed per head and averaged:

$$
\text{Overlap}_i = \frac{|O_{\text{activation}} \cap O^{i}_{\text{attention}}|}{|O_{\text{activation}}|}, \qquad \text{Overall} = \frac{1}{n_{\text{samples}} \cdot n_{\text{heads}}} \sum_{k=1}^{n_{\text{samples}}} \sum_{i=1}^{n_{\text{heads}}} \text{Overlap}_i
$$

Here $O_{\text{activation}}$ and $O^{i}_{\text{attention}}$ are sets of token positions, and the average runs over 100 samples and every head. A 95% overlap means: if a token has a giant activation, then in almost every head it is also a token that soaks up a disproportionate share of attention. The same tokens, the same channels, at every level of the stack.

## The lifecycle: born in layer 2, retired in layer 32

Correlation becomes mechanism when the authors trace a single outlier through the network. The story has three acts.

**Birth (layer 2 MLP).** LLaMA's MLP is gated: $\text{down}\big(\text{SiLU}(\text{gate}(x)) \odot \text{up}(x)\big)$. In layer 2, weight outliers in the gate and up projections produce an extreme neuron response on the start token. SiLU and the elementwise product amplify it, giving intermediate activations up to a thousand times the average. Then the down-projection's outlier columns route that energy into exactly two residual channels, 1415 and 2533, where it dominates the residual stream.

![Figure 7 from An et al. (2025): the emergence of activation outliers from weight outliers in the layer 2 MLP of LLaMA2-7B](/imgs/blogs/systematic-outliers-in-large-language-models-fig7.webp)

**Spread (layer 3 attention).** Those residual outliers enter attention through the query and key projections. In layer 3, the outlier tokens' queries and keys align in head dimensions 58 and 122, which inflates their dot products. Every query now pays heavy attention to them. The key twist: these tokens' *value* vectors are unusually small. They attract attention but contribute almost nothing to the output.

**Disappearance (final layers).** In the last layers, values of opposite sign are written into the same channels and gradually cancel the outlier. Attention outliers shrink with them, and some final-layer heads show none at all. The cancellation is progressive, not a cliff.

![Timeline across the 32 layers of LLaMA2-7B: outliers are born in layer 2, spread into attention in layer 3, persist through the residual stream, and are cancelled in layers 31 to 32](/imgs/blogs/systematic-outliers-in-large-language-models-4.webp)

That combination (high attention, tiny value) is the clue the rest of the paper chases. A token that everyone attends to but that carries nothing is not a content token. It is a no-op target.

## Three hypotheses, five attention variants

Why would a model want a no-op target? The authors propose three possible roles and design an experiment that separates them.

1. **Fixed bias.** Following Sun et al.'s *Massive Activations*, the outliers might act as a constant bias that every token receives.
2. **Context-aware bias.** Because attention outliers vary between heads and tokens (from 20% to 95% of a head's attention), the bias might depend on the input.
3. **Context-aware scaling factor.** Because the outlier tokens have small values, attending to them shrinks the attention output. The outliers would then be a dial that turns each token's update down.

Each hypothesis predicts something testable: if you hand the model the mechanism explicitly, it should stop growing outliers to fake it. The paper trains five GPT-2 models that differ only in the attention formula.

| ID | Variant | Formula | What it provides |
| --- | --- | --- | --- |
| (a) | Default | $\text{softmax}\big(QK^\top/\sqrt{d}\big)V$ | nothing extra |
| (b) | Explicit fixed bias | $\text{softmax}\big(QK^\top/\sqrt{d}\big)V + v'$ | fixed bias |
| (c) | Explicit context-aware bias | default $+\; \text{softmax}\big(Q[K;k']^\top/\sqrt{d}\big)[0; v']$ | context-aware bias |
| (d) | Attention bias | $\text{softmax}\big(Q[K;k']^\top/\sqrt{d}\big)[V; v']$ | context-aware bias **and** scaling |
| (e) | Explicit context-aware scaling | $S_c(x) \cdot \text{softmax}\big(QK^\top/\sqrt{d}\big)V$ | scaling only |

$Q, K, V \in \mathbb{R}^{T \times d}$ are the usual query, key and value matrices over $T$ tokens, $k', v' \in \mathbb{R}^d$ are learned vectors, and $S_c(x)$ is a learned scalar per token and head.

Variant (d) deserves a moment, because it is the one people already ship as a "sink token". Appending a learned key $k'$ with value $v'$ gives every query an extra slot to dump weight into. If a query puts weight $p$ on that slot, the real values get only ${1 - p}$ of the budget. So (d) mixes a bias ($p \cdot v'$) with a scale (${1 - p}$). Variant (c) is the control that isolates the bias half: it computes the same $p \cdot v'$ term but adds it on top of a full-strength default attention, so nothing gets scaled down.

The paper's main text does not spell out $S_c$. The released code ([`model_ca_softmax.py`](https://github.com/an-yongqi/systematic-outliers)) defines it as

$$
S_c(x) = 2\,\sigma(W_s x + b_s), \qquad W_s \in \mathbb{R}^{n_{\text{head}} \times d_{\text{model}}}
$$

where $x$ is the token's layer-normalized input to the attention block, $\sigma$ is the logistic sigmoid and the factor 2 lets the gate range over (0, 2). One linear layer per attention block, producing one scalar per head per token. For GPT-2 small, that is ${12 \times (768 \times 12 + 12) = 110{,}736}$ parameters, which matches the paper's reported growth from 123.59M to 123.70M.

![Figure 10 from An et al. (2025): activation magnitudes in GPT-2 trained with the five attention variants; only (d) attention bias and (e) context-aware scaling stay flat](/imgs/blogs/systematic-outliers-in-large-language-models-fig10.webp)

The result is clean. Default (a), fixed bias (b) and context-aware bias (c) all grow activation spikes in the thousands. Attention bias (d) and context-aware scaling (e) do not: per the paper's layer-wise plot, (e) keeps its top-3 activations at roughly 140 or below while the default model exceeds 1,000.

The logic of elimination goes like this:

- **Not the fixed bias.** (b) supplies one for free and outliers still form.
- **Not the context-aware bias.** (c) supplies one for free and outliers still form.
- **It is the scaling.** (d) and (e) both supply a way to shrink a token's update, and those are exactly the two that stay clean. (e) has no bias at all, so scaling alone is sufficient.

## Why softmax forces the issue

Now the central argument, which the paper gives in Section 5.3 and Appendix C. I will walk the ladder slowly, because this is the idea worth keeping.

**The problem.** Attention exists to mix context into each token. But some tokens need almost no update: a period, a padding token, the start token after it has done its job. For those, the ideal attention output is

$$
\Delta x = \text{MHA}(Q_x, K, V) \approx 0
$$

**The analogy.** Softmax is a budget that must be spent in full. You have exactly one unit of attention and you are not allowed to keep any. If you want to buy nothing, your only option is to spend the whole unit on an item that costs nothing: a key whose value vector is close to zero.

**The mechanism.** The attention weight from query $x$ to key $j$ is

$$
A_{xj} = \frac{\exp\!\big(Q_x K_j^\top / \sqrt{d_k}\big)}{\sum_{k=1}^{n} \exp\!\big(Q_x K_k^\top / \sqrt{d_k}\big)}, \qquad \sum_{j=1}^{n} A_{xj} = 1
$$

$Q_x$ is the query of token $x$, $K_j$ the key of token $j$, $d_k$ the key dimension and $n$ the number of keys. To make $\sum_j A_{xj} V_j \approx 0$ when most $V_j$ are ordinary, the model must concentrate almost all weight on one sink key $j^*$ with a near-zero value. That requires the sink's logit to dominate every other logit.

**The math.** Suppose the sink's scaled logit is $M$ and every other logit is 0. Then

$$
A_{xj^*} = \frac{e^{M}}{e^{M} + (n - 1)}, \qquad A_{xk} = \frac{1}{e^{M} + (n - 1)} \;\; (k \neq j^*)
$$

The weight on the sink approaches 1 only as $M$ grows, and the required $M$ grows with the log of the sequence length.

**The worked example.** Take $n = 2048$, the paper's sequence length. To put 99% of the weight on the sink, solve $e^M / (e^M + 2047) = 0.99$, giving $e^M = 0.99 \times 2047 / 0.01 \approx 202{,}650$ and $M \approx 12.2$. At $M = 8$ you only get 59%; at $M = 4$, under 3%.

![Line chart of attention weight on the sink key versus the logit gap M for 2048 keys: flat near zero until about M = 6, steep through M = 8 to 10, reaching 0.99 at M of about 12.2](/imgs/blogs/systematic-outliers-in-large-language-models-2.webp)

A logit gap of 12 is not cheap. The logit is a dot product of two vectors that have just passed through a normalization layer, which pins their scale. The only ways to get a large dot product out of normalized inputs are to make a few directions enormous before normalization, or to grow large weights in the projections. That is precisely what the paper observes: residual channels 1415 and 2533 in the thousands, and query and key alignment in dimensions 58 and 122.

The paper adds a second pressure. Because softmax outputs are strictly positive, no key ever gets exactly zero weight, so gradients keep flowing to every query and key. The model never "finishes" building the sink; it keeps pushing the gap wider. Large residual activations in turn mean large gradients for the projection weights that produce them, which is the paper's account of where the weight outliers come from.

**When the argument fails.** The derivation shows that a near-zero update *requires* a large logit gap. It does not show that gradient descent will build that gap through giant activations rather than some other route. The step from "large dynamic range needed" to "MLP outlier in channel 1415" is plausible and matches the measurements, but it is an argument, not a theorem. The appendix's claim that outliers "predominantly emerge in deeper layers" also sits awkwardly next to the main-text finding that they are born in layer 2.

## The fix: give attention a volume knob

Once you see the problem, the fix is obvious. If the model needs a way to say "update this token by almost nothing", give it one directly. Let softmax distribute attention normally, then multiply the result by a per-token gate.

$$
\text{Attn}(Q, K, V) = S_c(x) \cdot \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

Now a period can attend wherever it likes, with ordinary-sized logits, and set $S_c \approx 0.05$ to shrink the update. No sink, no logit gap of 12, no reason to grow giant activations.

![Before and after panels: on the left softmax puts 0.96 of attention on a sink key with a tiny value, requiring a large logit gap; on the right attention is spread evenly and a scaling factor of 0.05 shrinks the output](/imgs/blogs/systematic-outliers-in-large-language-models-3.webp)

A quick worked comparison, using toy numbers of my own. With five keys, where the sink's value norm is 0.1 and the others are 1.0:

| | Weights on (sink, 4 others) | Output size | Logit gap needed |
| --- | --- | --- | --- |
| Softmax only | 0.96, 0.01 each | ${0.96 \times 0.1 + 0.04 \times 1.0 \approx 0.14}$ | $\ln(0.96 / 0.01) \approx 4.6$, growing with $\ln n$ |
| With $S_c = 0.05$ | 0.20 each | $\le 0.05 \times 1.0 = 0.05$ | 0 |

The scaled version gets a smaller update with no extreme logits at all, and the gap it avoids grows with context length.

The paper backs the diagnosis with two more experiments in the appendix. GPT-2 trained with **sigmoid attention** (each weight independently in (0, 1), no sum-to-one constraint) also shows no systematic outliers. And a **TinyLLaMA-120M** with context-aware scaling is outlier-free too, so the effect is not specific to GPT-2's architecture.

The variant (d) result has a practical reading: a learned sink key is itself a valid volume knob, because weight sent to $k'$ is weight taken from the real values. That is my interpretation, not the paper's wording, but it explains why explicit sink tokens or learned sink logits, as shipped in some open-weight models such as [gpt-oss](/blog/machine-learning/frontier-models/openai-gpt-oss-open-weight-models-timeline), address the same pressure.

## Results: compression and convergence

The payoff experiments are on GPT-2 (124M), trained with nanoGPT for 50,000 iterations, about 2B tokens. Compression is evaluated as WikiText2 perplexity, with 8-bit AbsMax weight quantization and unstructured 50% magnitude pruning.

| Model | PPL (FP16) | PPL (AbsMax W8) | PPL (50% sparse) | Params |
| --- | --- | --- | --- | --- |
| GPT-2 default | 27.24 | 93.44 | 7235.68 | 123.59M |
| GPT-2 + context-aware scaling | 26.95 | 29.22 | 39.47 | 123.70M |

![Grouped log-scale bar chart of GPT-2 perplexity on WikiText2: FP16 27.24 vs 26.95, AbsMax W8 93.44 vs 29.22, 50% sparse 7235.68 vs 39.47](/imgs/blogs/systematic-outliers-in-large-language-models-5.webp)

The mechanism behind both numbers is the one from the [SmoothQuant post](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache). AbsMax quantization sets the step size from the largest weight in a tensor, so one giant value forces a coarse grid on everything else. Magnitude pruning keeps the largest weights by definition, so it happens to keep the outliers, but the network that relied on finely balanced cancellations around them breaks. Remove the outliers at training time and both methods behave.

On training, Figure 12 of the paper shows the scaled model's validation loss dropping faster in the first 200 steps, while the final losses are comparable. The FP16 perplexity difference (27.24 vs 26.95) is small, so the honest summary is: same quality, faster early convergence, far better compressibility.

**What might not transfer.** Everything above is GPT-2 small and a 120M LLaMA trained on 2B tokens. The 7B observational analysis and the training interventions are on different scales. A 7B model trained on trillions of tokens may find other routes to a near-zero update, and the gate might interact with grouped-query attention, long context or RoPE in ways a 1,024-token GPT-2 cannot reveal. The compression test also uses the simplest methods (AbsMax per-tensor, global magnitude pruning); the gap to modern methods like GPTQ or AWQ, which already handle outliers, would likely be much smaller.

## Critique

**What is strong.**

- **Unification.** Three literatures (LLM.int8 activation outliers, massive activations, attention sinks) are tied into one causal chain with measured 95 to 100% overlap. That alone is a useful map for anyone working on [activation outliers in an inference engine](/blog/machine-learning/inference-engineering/activation-outliers-calibration-and-measuring-quality-loss).
- **Elimination-style experiment.** The five variants are a good design: each hypothesis gets its mechanism handed to the model for free, and you watch which one removes the need for outliers. The (c) vs (d) contrast separating bias from scaling is the cleverest part.
- **Cheap, falsifiable fix.** A linear layer and a sigmoid per block is easy to reproduce, and the authors released code.

**What is weak.**

- **One seed, small scale.** The intervention results are single training runs of 124M and 120M models. No error bars, no scaling sweep.
- **Appendix C is an argument.** It derives that a near-zero update needs a large logit gap, then narrates the rest. There is no experiment that, say, ablates the normalization layer to test whether it is the thing forcing activations to grow.
- **The 1,000x threshold is a single cut.** With $\tau = 1000$ fixed, the per-layer counts and the "none in final heads" claims depend on one arbitrary line.
- **Missing ablation.** Variant (e) keeps sum-to-one softmax and adds a gate. A run that gates a *constant* (non-context-aware) scale would pin down whether context dependence matters or whether any learnable output scale suffices.

**What would change my mind.** If a 1B+ model trained with context-aware scaling for hundreds of billions of tokens still grew activation outliers above 1,000x the mean, the claim that softmax's sum-to-one rule is *the* root cause would weaken to "a" cause. Conversely, a clean result there with matched quality and better W8A8 quantization would make this a default architectural change.

## References

1. Yongqi An, Xu Zhao, Tao Yu, Ming Tang, Jinqiao Wang. [Systematic Outliers in Large Language Models](https://arxiv.org/abs/2502.06415v2). ICLR 2025. arXiv:2502.06415.
2. Code: [github.com/an-yongqi/systematic-outliers](https://github.com/an-yongqi/systematic-outliers) (the $S_c$ definition is in `gpt-2/model_ca_softmax.py`).
3. Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu. [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762). 2024.
4. Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453). 2023.
5. Tim Dettmers et al. [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339). NeurIPS 2022.
6. Jason Ramapuram et al. [Theory, Analysis, and Best Practices for Sigmoid Self-Attention](https://arxiv.org/abs/2409.04431). 2024.
7. On this site: [LLM quantization of activations: SmoothQuant and the KV cache](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache), [Activation outliers, calibration and measuring quality loss](/blog/machine-learning/inference-engineering/activation-outliers-calibration-and-measuring-quality-loss), [OpenAI gpt-oss open-weight models](/blog/machine-learning/frontier-models/openai-gpt-oss-open-weight-models-timeline).

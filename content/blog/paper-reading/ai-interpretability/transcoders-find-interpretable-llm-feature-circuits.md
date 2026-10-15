---
title: "Transcoders Find Interpretable LLM Feature Circuits: Seeing Through the MLP"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How transcoders replace a dense MLP with a wide, sparse imitation so that feature-to-feature circuits split cleanly into a per-prompt activation and a fixed weight term, and what that buys on GPT2-small's greater-than circuit."
tags: ["paper-reading", "transcoders", "feature-circuits", "mechanistic-interpretability", "sparse-autoencoders", "circuit-analysis", "large-language-model", "neurips-2024-poster"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 18
paper:
  title: "Transcoders Find Interpretable LLM Feature Circuits"
  authors: "Jacob Dunefsky, Philippe Chlenski, Neel Nanda"
  venue: "NeurIPS 2024"
  url: "https://arxiv.org/abs/2406.11944"
---

> [!tldr]
> **Claim.** If you swap each MLP sublayer for a *transcoder* (a wide, sparsely activating ReLU MLP trained to imitate the original MLP's input-to-output map), then the influence of one feature on a later feature factorizes into two parts: the earlier feature's activation on this prompt, times a dot product of weights that never changes.
> **Why it matters.** Sparse autoencoders give you readable features, but the MLP between two SAE features is still a dense tangle of neurons, so any connection you measure holds for one prompt only. Transcoders give you connections that hold for *every* prompt.
> **Most surprising result.** Transcoders cost nothing in quality: on GPT2-small, Pythia-410M and Pythia-1.4B they match or beat SAEs on the sparsity versus loss frontier, and a blind human rating found them just as interpretable (41 vs 38 interpretable features out of 50).
> **Where it fails.** The method treats attention patterns as fixed, approximates LayerNorm as a constant, and two of nine blind case studies got the feature wrong.

Mechanistic interpretability wants **circuits**: small subgraphs of a model that explain one behavior. Attention heads have been taken apart this way for years. MLP sublayers have resisted. This paper, by Dunefsky, Chlenski and Nanda, offers a clean way through them, and it is the direct ancestor of the "replacement model" style of circuit tracing that later work built on.

I will walk through the four ideas that carry the paper: the transcoder itself, the attribution factorization, the greedy path search, and the evaluations that check the whole thing is not an illusion.

## The problem: the MLP is a wall

A transformer's hidden state at each token is a **residual stream**. Every attention and MLP sublayer reads from it and adds its output back. The paper writes this as

$$
x^{(l,t)}_{\text{mid}} = x^{(l,t)}_{\text{pre}} + \sum_{h} \text{attn}^{(l,h)}\big(x^{(l,t)}_{\text{pre}};\, x^{(l,1:t)}_{\text{pre}}\big), \qquad
x^{(l+1,t)}_{\text{pre}} = x^{(l,t)}_{\text{mid}} + \text{MLP}^{(l)}\big(x^{(l,t)}_{\text{mid}}\big)
$$

Here $l$ is the layer, $t$ the token position, $h$ an attention head, and $x \in \mathbb{R}^{d_{\text{model}}}$ the hidden state. Because everything is *added*, the stream is a sum of all earlier outputs. That additivity is what makes circuit analysis possible at all.

The trouble is **polysemanticity**. A single MLP neuron fires on many unrelated things, so reading neurons directly gets you nowhere. Sparse autoencoders (SAEs) fix this by decomposing activations into many sparse **features**, each a direction in the residual stream. But an SAE feature before the MLP and an SAE feature after it are connected *through* the MLP, and each SAE feature is a dense mix of many neurons, each with its own nonlinearity.

![Why an SAE route through the MLP gives only per-prompt answers, while a transcoder route factorizes](/imgs/blogs/transcoders-find-interpretable-llm-feature-circuits-1.webp)

Concretely, let $z$ be the earlier feature's activation, $z'$ the later feature's activation, and $y$ the MLP's activation in between. The standard attribution, input-times-gradient, is

$$
z \,\frac{\partial z'}{\partial z} \;=\; z \,\frac{\partial z'}{\partial y}\,\frac{\partial y}{\partial z}
$$

Every factor depends on the input. $z$ obviously does, but so does $\partial y / \partial z$, because the MLP's gradient changes with which neurons are on. So the number you compute says "on this prompt, $z$ mattered this much". It cannot say whether $z$ matters on other prompts, or whether something else could trigger $z'$ instead.

The paper's own example makes this vivid. Patching shows a "Polish last names" feature drives some post-MLP feature on one prompt. Would an "English last names" feature also drive it? Is there a prompt where the Polish feature fires and the later feature does not? Per-prompt attribution cannot answer either question. The authors call the missing property **input-invariance**.

## Technique 1: the transcoder

**The problem it solves.** We want a version of the MLP whose internals are sparse and readable, so that we can trace connections through it.

**Analogy.** An SAE is a *transcriptionist*: it listens to a signal and writes down the same signal in a sparser alphabet. A transcoder is an *interpreter*: it listens to what goes *into* the MLP and says what the MLP would have said *out*, using a large vocabulary of simple phrases, only a few at a time.

![Figure 1 from Dunefsky, Chlenski and Nanda (2024): an SAE reconstructs activations at one point, while a transcoder reads the MLP input and imitates the MLP output](/imgs/blogs/transcoders-find-interpretable-llm-feature-circuits-fig1.webp)

**Mechanism.** A transcoder is a one-hidden-layer ReLU MLP, just much wider than the original, and trained so that its hidden layer is sparse:

$$
z_{\text{TC}}(x) = \text{ReLU}\big(W_{\text{enc}}\, x + b_{\text{enc}}\big), \qquad
\text{TC}(x) = W_{\text{dec}}\, z_{\text{TC}}(x) + b_{\text{dec}}
$$

The symbols:

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $x$ | $d_{\text{model}}$ | input to the MLP sublayer (post-LayerNorm) |
| $W_{\text{enc}}$ | $d_{\text{features}} \times d_{\text{model}}$ | row $i$ is the **encoder vector** $f_{\text{enc}}^{(l,i)}$ of feature $i$ |
| $b_{\text{enc}}$ | $d_{\text{features}}$ | encoder bias |
| $z_{\text{TC}}(x)$ | $d_{\text{features}}$ | feature activations, mostly zero |
| $W_{\text{dec}}$ | $d_{\text{model}} \times d_{\text{features}}$ | column $i$ is the **decoder vector** $f_{\text{dec}}^{(l,i)}$ |
| $b_{\text{dec}}$ | $d_{\text{model}}$ | decoder bias |

Each feature asks one question with its encoder vector ("how much of my pattern is in the input?") and, if the answer is positive, writes its decoder vector into the residual stream scaled by that amount.

**Training loss.** The only change from an SAE is the target of the reconstruction term:

$$
\mathcal{L}_{\text{TC}}(x) = \underbrace{\big\lVert \text{MLP}(x) - \text{TC}(x) \big\rVert_2^2}_{\text{faithfulness}} \;+\; \lambda_1 \underbrace{\big\lVert z_{\text{TC}}(x) \big\rVert_1}_{\text{sparsity}}
$$

An SAE would put $x$ where $\text{MLP}(x)$ is. The coefficient $\lambda_1$ trades faithfulness against sparsity. In the paper's runs every transcoder and SAE had $d_{\text{features}} = 32 \times d_{\text{model}}$, so ${24{,}576}$ features for GPT2-small ($d_{\text{model}} = 768$). Training used Adam at learning rate ${2 \times 10^{-5}}$, 60 million OpenWebText tokens, batch size 4096, context length 128, and no neuron resampling.

**Worked micro-example.** Take $d_{\text{model}} = 2$ and two transcoder features. Encoder rows are $f_{\text{enc}}^{(1)} = (1, 0)$ and $f_{\text{enc}}^{(2)} = (0, 1)$, biases zero. For input $x = (3, -1)$, the activations are $\text{ReLU}(3) = 3$ and $\text{ReLU}(-1) = 0$. Only feature 1 fires, so the output is just ${3}\, f_{\text{dec}}^{(1)}$. The whole MLP's effect on this token is "one feature, strength 3".

**When it fails.** A transcoder is an approximation. Whatever it gets wrong is invisible to circuit analysis built on it. The authors say this plainly in their limitations: they measured the size of the error (via cross-entropy) but did not study what the error contains.

## Technique 2: the attribution factorization

This is the heart of the paper.

**The problem.** Given a later transcoder feature $i'$ at layer $l'$, which earlier features caused it to fire, and would they cause it on *any* prompt?

**Mechanism.** Feature $i$ at layer $l$ writes $z_{\text{TC}}^{(l,i)}\, f_{\text{dec}}^{(l,i)}$ into the residual stream. The stream is additive, so that vector is still sitting in the input to layer $l'$, same token. Feature $i'$ reads its input with a dot product against $f_{\text{enc}}^{(l',i')}$. So the contribution of $i$ to $i'$ (pre-ReLU, ignoring LayerNorm for now) is

$$
\underbrace{z_{\text{TC}}^{(l,i)}\big(x_{\text{mid}}^{(l,t)}\big)}_{\text{input-dependent}} \;\cdot\; \underbrace{\big(f_{\text{dec}}^{(l,i)} \cdot f_{\text{enc}}^{(l',i')}\big)}_{\text{input-invariant}}
$$

The first factor is a single scalar you read off the forward pass. The second is a dot product of two learned weight vectors: compute it once after training and it holds for every prompt ever.

![The two factors of a transcoder edge, plus the attention-mediated and de-embedding variants](/imgs/blogs/transcoders-find-interpretable-llm-feature-circuits-2.webp)

**Worked micro-example.** Let $f_{\text{dec}}^{(l,i)} = (0.5, -1, 2)$ and $f_{\text{enc}}^{(l',i')} = (1, 0.5, 0.25)$. The weight term is ${0.5 - 0.5 + 0.5 = 0.5}$. On prompt A feature $i$ fires at 4, so the edge is ${4 \times 0.5 = 2}$. On prompt B it fires at 0, so the edge is 0. The *connection* is the same on both prompts; only the *usage* differs. That separation is exactly what the gradient formula above could not give you.

You can also read the weight term globally: sort all earlier features by $f_{\text{dec}} \cdot f_{\text{enc}}^{(l',i')}$ and you get the features that *could* drive $i'$, regardless of whether any prompt in your dataset exercised them.

### Through attention heads

Features at token $s$ can also influence a feature at token $t$ through an attention head. The paper extends the factorization through the head's OV circuit. The contribution of source token $s$ via head $h$ at layer $l$ to feature $i'$ at layer $l'$, token $t$, is

$$
\text{score}^{(l,h)}\big(x^{(l,t)}_{\text{pre}},\, x^{(l,s)}_{\text{pre}}\big)\;\Big(\big(W^{(l,h)}_{OV}\big)^{\!\top} f_{\text{enc}}^{(l',i')}\Big) \cdot x^{(l,s)}_{\text{pre}}
$$

Here $\text{score}^{(l,h)}$ is the attention weight from token $s$ to token $t$ (input-dependent) and $W_{OV}^{(l,h)}$ is the head's combined value-output matrix (fixed). Pulling $f_{\text{enc}}$ back through $W_{OV}^{\top}$ gives a new vector that you can dot against anything at the source token, including earlier transcoder decoder vectors. The attention pattern is taken as given; the paper does not explain *why* the head attends where it does.

### De-embeddings

A special case of the input-invariant term is the **de-embedding**: $W_E^{\top} f_{\text{enc}}^{(l,i)}$, where $W_E$ is the token embedding matrix. It gives one score per vocabulary token, saying how strongly that token, on its own, directly pushes feature $i$. No prompts needed.

For one GPT2-small MLP0 feature the top de-embedding tokens were *oglu*, *owsky*, *zyk*, *chenko* and *kowski*. You can name that feature "Polish and Eastern European surnames" without looking at a single activating example. That property turns out to be what makes blind case studies possible.

## Technique 3: greedy path search

**The problem.** Pairwise attributions are edges. A circuit is a graph. How do you grow one without enumerating every path through a 12-layer model?

**Analogy.** It is a beam search run backwards from the output: keep the $k$ most promising partial explanations, extend each by one step, prune back to $k$, repeat.

![Figure 2 from Dunefsky, Chlenski and Nanda (2024): the six steps of the greedy circuit-finding algorithm](/imgs/blogs/transcoders-find-interpretable-llm-feature-circuits-fig2.webp)

**Mechanism.** The paper's Algorithm 1 works like this:

1. Start from the target feature $i'$ at its layer and token.
2. Compute the direct attribution of every earlier transcoder feature and attention head to it.
3. Keep only the top $N$ paths.
4. For each kept node, find *its* top contributors, extending each path by one node.
5. Prune again to the top $N$ paths, and repeat for $L$ iterations.
6. Merge the surviving paths into one graph (Algorithm 2).

The trick that keeps attributions consistent along a path is to **propagate a feature vector backwards**. If node $c'$ has attribution $a' = f' \cdot y'$ to the target, we choose a new vector $f$ such that $f \cdot x' = a'$, where $x'$ is the input to $c'$. Then the attribution of any earlier component with output $y$ is just $f \cdot y$. For a transcoder feature this new vector is

$$
f = \big(f' \cdot f_{\text{dec}}^{(l,i)}\big)\, f_{\text{enc}}^{(l,i)}
$$

and for an attention head it is $\text{score}\,(W_{OV})^{\top} f'$. The authors note this is equivalent to input-times-gradient along the path, which is why it resembles backprop.

When paths are merged, a node's attribution is the sum over *distinct* paths beginning at it, and prefixes already seen are skipped so nothing is counted twice. Under zero transcoder error, each node's attribution equals the sum of its incoming edges. Error nodes, following Marks et al., can absorb the rest.

**When it fails.** Two approximations sit inside every step. LayerNorm is replaced by a constant scale, estimated as the ratio of pre- and post-LayerNorm norms on the prompt. Attention patterns are frozen. Both are reasonable for a single prompt and both hide real computation. Greedy pruning can also drop a path whose individual edges are small but whose sum matters.

## Are transcoders as good as SAEs?

A factorization is useless if the transcoder features are junk or the transcoder is unfaithful. The paper checks both.

### Interpretability: a blind rating

The authors took 50 random features from a Pythia-410M layer 15 transcoder and 50 from a layer 15 SAE trained on MLP *inputs*, shuffled them, and rated each from its top-activating OpenWebText examples without knowing the source. The SAE was trained on MLP inputs so that both encoders live in the same space, and the pair was chosen to sit at almost the same point on the frontier (transcoder $L_0 = 44.04$ at 3.35 nats; SAE $L_0 = 47.85$ at 3.36 nats).

| Rating | Transcoder | MLP-in SAE |
| --- | --- | --- |
| Interpretable | 41 | 38 |
| Maybe | 8 | 8 |
| Uninterpretable | 1 | 4 |
| Of the interpretable, "context-free" (single token) | 6 | 16 |

The headline is parity. The detail I find more interesting is the last row: the SAE spent far more of its interpretable features on single-token detectors. One rater, 50 features per side, so treat the gap as a hint, not a result.

### Faithfulness and sparsity

The quantitative test measures two numbers per trained dictionary:

- **Sparsity**: mean $L_0$, the average count of active features per token.
- **Fidelity**: next-token cross-entropy when the MLP output is replaced by the dictionary's output, compared with the original model (best case) and with the MLP mean-ablated (worst case).

Each point below is one transcoder or SAE trained with a different $\lambda_1$, evaluated on the same 3.2M OpenWebText tokens.

![Figure 3 from Dunefsky, Chlenski and Nanda (2024): mean loss against mean L0 for transcoders and SAEs on GPT2-small layer 8, Pythia-410M layer 15 and Pythia-1.4B layer 15](/imgs/blogs/transcoders-find-interpretable-llm-feature-circuits-fig3.webp)

| Model and layer | $d_{\text{model}}$ | Dictionary size | Finding |
| --- | --- | --- | --- |
| GPT2-small, layer 8 | 768 | 24,576 | transcoder frontier on par with or below SAE |
| Pythia-410M, layer 15 | 1024 | 32,768 | transcoder below SAE |
| Pythia-1.4B, layer 15 | 2048 | 65,536 | gap widest of the three |

In all three, transcoders are equal or better, and the authors observe the gap seems to widen with model size. They are careful about one caveat that cuts against them: the learning rate was tuned on transcoders and reused for SAEs, so SAEs may be under-tuned. Only $\lambda_1$ was swept separately.

**What is load-bearing here.** The SAE baseline is trained on MLP *outputs*, because that is the fair comparison when you splice the output into the model. But an SAE on the output has a different job (reconstruct a vector) from a transcoder (predict a vector from another vector), so "equal loss" is not obviously the same kind of success. The comparison is fair for the use case, splicing into the forward pass, and that is the use case that matters for circuits.

## Case study 1: blind reverse-engineering

The paper's most original evaluation is the **blind case study**. Pick a random feature in GPT2-small's layer 8 transcoder, then try to say what it does *without ever seeing the text that activates it*. The rules:

1. You may not look at the tokens of any prompt. Prompts are referenced by dataset index only.
2. You may run those prompts to compute activations and circuits.
3. You may use any input-invariant information, including de-embeddings.

The point is to stop confirmation bias. If you read the activating examples first, you will find the circuit you expect and ignore the one that disagrees.

![The circuit recovered for tc8[355] without seeing any prompt text](/imgs/blogs/transcoders-find-interpretable-llm-feature-circuits-3.webp)

Here is the run for feature `tc8[355]` (notation: `tcA[B]@C` is feature B of the layer A transcoder at token C).

- **The target.** Example 5701, token 37, where `tc8[355]` fires at 11.91.
- **Current token.** Strong contributions from `tc0[16632]` and `tc0[9188]`, whose de-embeddings are variants of `;`. Another current-token contributor, `tc6[11831]`, has those same semicolon features as its top input-invariant inputs. So token 37 is a semicolon.
- **Previous tokens, via attention.** `tc0[13196]@36` de-embeds to years such as 1973, 1971 and 1967. `tc0[10109]@31` de-embeds to an open parenthesis. `tc6[21046]@35` has top input-invariant inputs `tc0[16382]` (Eastern European surnames: *kowski*, *chenko*, *owicz*) and `tc0[5468]` (English surnames: *Burnett*, *Hawkins*, *Johnston*).

Put together, the pattern is "( ... surname, year ;". The hypothesis: semicolons inside parenthetical citations. A second prompt added a feature de-embedding to *Accessed*, *Retrieved* and *Springer*. When the authors finally looked, the top examples included "(Poeck, 1969; Rinn, 1984)", which is exactly prompt 5701.

Notice what the surname step used: input-invariant connections only. Nobody needed a prompt to learn that `tc6[21046]` listens to surname features.

### How often did it work?

Nine features were studied. Because layer 0 features are nearly single-token, their de-embeddings leak the prompt, so most studies were **restricted blind**: input-dependent MLP0 information was forbidden too.

| Outcome | Count | Examples |
| --- | --- | --- |
| Success | 5 | citation semicolon, "All" token, "told/said in an interview", clause-initial "though/although", mostly-uninterpretable Cyrillic |
| Mostly a success | 2 | "past tense of catch" (actual: forms of *caught*); "not offering/providing" (hypothesis added a legalese guess) |
| Failure | 2 | `tc8[479]` guessed "with a total estimated" (actual: a local context feature); `tc8[9030]` guessed "biology as a subject of study" (actual: any scientific subject) |

Seven of nine is encouraging, and the paper reports the failures in full, which I respect. It is also nine features, analyzed by the method's authors.

## Case study 2: the greater-than circuit

The second case revisits a known circuit. Given "The war lasted from 1737 to 17", GPT2-small should put probability on years after 37. Hanna et al. had found that MLP10 does much of the work, analyzed neuron by neuron.

**First, corroboration.** Path search on one prompt put high attribution on MLP10 transcoder features, which were in turn driven by earlier features through **attention head 1 in layer 9**. That matches the original analysis.

**Then the mechanism.** Over all 100 prompts "The war lasted from 17YY to 17", the highest-variance MLP10 features had top connections, through L9H1, from MLP0 features whose de-embeddings were two-digit numbers. The input-invariant connections showed the same. The authors flag this as unexpected, since only 100 of the 50k+ vocabulary tokens are two-digit numbers.

For the de-embedding curves, they took each MLP10 feature's top ten MLP0 inputs by the input-invariant score $\big(f_{\text{dec}}^{(0,i)}\big)^{\top} W_{OV}^{(9,1)} f_{\text{enc}}^{(10,j)}$ and summed those features' de-embedding scores per YY token, weighted by that score.

![Figure 5 from Dunefsky, Chlenski and Nanda (2024): for the three highest-variance MLP10 features, the logit boost on each YY token alongside its de-embedding score](/imgs/blogs/transcoders-find-interpretable-llm-feature-circuits-fig5.webp)

The picture is tidy. Each feature's de-embedding peaks on some range of YY values, and its direct logit attribution boosts years just *after* that range and suppresses years before it. In words: "a few features fire on years in certain ranges and boost the following years". That is a simpler story than the neuron-level one.

### Sparser than neurons?

To test sparsity directly, the authors kept only the top $k$ highest-variance transcoder features (or MLP10 neurons), zero-ablated the rest, and measured the **mean probability difference**:

$$
\text{PD}(y) = \sum_{y' > y} p_{y'} \;-\; \sum_{y' \le y} p_{y'}
$$

where $p_{y'}$ is the softmax probability over YY tokens and the mean is over the 100 prompts.

![Figure 4 from Dunefsky, Chlenski and Nanda (2024): probability difference as more features or neurons are kept (left) and the flat logit profile of tc10[5315] (right)](/imgs/blogs/transcoders-find-interpretable-llm-feature-circuits-fig4.webp)

- For fewer than 24 components, transcoder features beat neurons.
- At 24 the transcoder curve drops sharply. The culprit was `tc10[5315]`, the 24th-highest-variance feature, whose logit effect is nearly flat across years: it boosts everything, which hurts a "later minus earlier" metric.
- With that feature removed, a handful of features recovers most of the original model's performance; neurons need many more.
- Even the full transcoder does not recover the original model's score.

**My read.** The "bad feature" episode is a good lesson in itself. A transcoder can learn a feature that is faithful on average and actively unhelpful for a specific task. Finding it took one plot; finding the equivalent among neurons would not have been as easy.

## How this connects to the rest of the field

Transcoders sit between two older lines of work. SAEs gave readable features (see [scaling and evaluating sparse autoencoders](/blog/paper-reading/ai-interpretability/scaling-and-evaluating-sparse-autoencoders)). Feature-circuit methods such as [sparse feature circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models) used those features with patching and gradients, which is exactly the per-prompt approach this paper argues is not enough. A third line, bilinear MLPs, chases the same goal of weight-based interpretability by changing the architecture instead of approximating it.

The greedy path search itself comes from earlier work by Dunefsky and Cohan; the novelty here is that the transcoder makes every edge through an MLP factorize.

## Critique

**What is strong.**

- The core argument is algebraic, not empirical. Once the MLP is a sparse ReLU layer whose output is a sum of decoder vectors, the factorization follows from additivity of the residual stream. You do not have to trust a benchmark to believe it.
- The blind case study protocol is a genuinely better way to evaluate interpretability tools, and the paper publishes its failures.
- The SAE comparison is honest about its own biases (shared learning rate, limited sweeps).

**What is weak or missing.**

- **The error term is unexamined.** Equal cross-entropy to SAEs says how much is lost, not *what*. If the transcoder systematically drops rare but important computations, circuits built on it will be clean and wrong.
- **Attention is a black box.** The method explains what flows through an attention head, never why the head attends there. For a circuit like greater-than, the attention pattern is half the story.
- **Small samples.** One rater, 50 features per side; nine blind studies run by the authors. Neither supports a statistical claim.
- **Input-invariant does not mean global.** The weight term $f_{\text{dec}} \cdot f_{\text{enc}}$ is the *pre-activation* contribution. Whether it matters still depends on the ReLU threshold of the later feature and on what else is in the stream. A large dot product is a possible route, not a guaranteed one. (That last point is my extrapolation; the paper's framing is that the term describes "general behavior".)
- **Missing ablation.** I would want a direct check that top input-invariant connections predict causal effects across a held-out set of prompts, for example via patching, rather than only through case studies.

**What would change my mind.** If someone showed that, across many random features, the ranking of earlier features by $f_{\text{dec}} \cdot f_{\text{enc}}$ correlates poorly with their measured causal effect on held-out prompts, I would stop trusting input-invariant connections as a description of general behavior. Conversely, a strong correlation on a larger model would make me treat transcoders as the default way to cross an MLP.

## Practical takeaways

- If your circuit question has to cross an MLP, a transcoder lets you ask it once for all prompts, not once per prompt.
- De-embeddings are the cheapest interpretability tool in the paper: one matrix multiply per feature and you get a vocabulary profile.
- Check for "flat" features like `tc10[5315]` before trusting a sparse reconstruction on a task metric.
- Keep the blind protocol in mind for your own tools. It costs nothing and removes the easiest way to fool yourself.

## References

1. Dunefsky, J., Chlenski, P., and Nanda, N. (2024). *Transcoders Find Interpretable LLM Feature Circuits*. NeurIPS 2024. [arXiv:2406.11944](https://arxiv.org/abs/2406.11944)
2. Code: [github.com/jacobdunefsky/transcoder_circuits](https://github.com/jacobdunefsky/transcoder_circuits/)
3. Hanna, M., Liu, O., and Variengien, A. (2023). How does GPT-2 compute greater-than? (the original greater-than circuit analysis the paper revisits)
4. Related on this site: [Sparse Feature Circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models), [Scaling and Evaluating Sparse Autoencoders](/blog/paper-reading/ai-interpretability/scaling-and-evaluating-sparse-autoencoders), Bilinear MLPs Enable Weight-based Mechanistic Interpretability

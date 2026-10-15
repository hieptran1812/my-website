---
title: "Layer by Layer: Why the Middle of a Language Model Often Beats Its Last Layer"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "A close reading of Skean et al. (ICML 2025): how matrix-based entropy, curvature and augmentation invariance explain why mid-depth embeddings beat final-layer embeddings on 32 MTEB tasks, and where the argument is thinner than it looks."
tags:
  - paper-reading
  - hidden-representations
  - language-models
  - layer-analysis
  - matrix-based-entropy
  - representation-quality
  - embeddings
  - icml-2025-oral
  - icml-2025
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 18
image: "/imgs/blogs/layer-by-layer-uncovering-hidden-representations-in-language-models-fig1.webp"
excerpt: "Final-layer embeddings are the default, yet on 32 MTEB tasks a middle layer wins by 2% to 16%. The paper ties this to a mid-depth compression valley that appears in autoregressive models, text or image alike."
paper:
  title: "Layer by Layer: Uncovering Hidden Representations in Language Models"
  authors: "Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Patel, Jalal Naghiyev, Yann LeCun, Ravid Shwartz-Ziv"
  venue: "ICML 2025"
  url: "https://arxiv.org/abs/2502.02013v2"
---

> [!tldr]
> **Claim.** For embedding tasks, the last layer of a language model is usually not the best layer. On 32 tasks from the Massive Text Embedding Benchmark (MTEB), some intermediate layer beats the final layer, by 2% to 16% on average, across Pythia, Llama3, Mamba, BERT and LLM2Vec models.
>
> **Why it matters.** If you pool embeddings from the last hidden state by habit, you may be leaving accuracy on the table, and you can often find a better layer without any labels.
>
> **Most surprising result.** The mid-depth "compression valley" is tied to the *autoregressive objective*, not to text. An autoregressive image model (AIM) shows the same valley and the same mid-layer accuracy peak; masked and contrastive vision models do not.
>
> **Where it is thin.** The unifying "matrix-based entropy" framework is more a shared vocabulary than a predictive theory, the label-free layer picker helps Pythia but hurts LLM2Vec with one of its metrics, and one of the stated theorems appears to run the wrong way (details in the critique).

## The question nobody asks: which layer?

Almost every embedding pipeline mean-pools the final hidden state and moves on, on the theory that the last layer is the "most processed". Skean and colleagues test that assumption systematically.

Their answer is a plot you can read in five seconds. They take each layer of three models in turn, use its output as a sentence embedding, and score it on 32 MTEB tasks.

![Figure 1 from Skean et al. (2025): average MTEB score per layer depth for Pythia-410m, Mamba-370m and BERT-base](/imgs/blogs/layer-by-layer-uncovering-hidden-representations-in-language-models-fig1.webp)

Read the curves from left to right. Every model climbs out of its early layers, peaks somewhere in the middle, and then *drops* before the end. The x-axis is depth as a percentage rather than layer index, so a 24-layer and a 48-layer model line up.

Beyond the plot, the paper offers seven label-free "representation quality" metrics, all framed as views of one object, the Gram matrix of the embeddings. They track downstream score layer by layer, which makes them a cheap way to pick a layer. This post walks them from intuition to math, then checks how much of the explanation holds up.

## The common object: one prompt as a matrix

Feed a prompt of $L$ tokens through the model and stop at layer $\ell$. You get one hidden vector per token, each of width $D$.

Stack them as rows and you have a matrix $Z \in \mathbb{R}^{N \times D}$, where $N$ is the number of rows. For a single prompt the rows are tokens ($N = L$). For a whole dataset the paper mean-pools each prompt into one row, so $N$ is the number of prompts.

Every metric in the paper is a question about how the rows of $Z$ are spread out. The cleanest way to ask that question is through the **Gram matrix** $K = ZZ^\top$, an $N \times N$ table of dot products between rows. Its eigenvalues say how much of the total spread lives along each principal direction.

## Technique 1: matrix-based entropy

### The problem it solves

You want a single number that says "these token vectors use many directions" versus "these token vectors have collapsed onto a few". Variance alone will not do it, because a matrix can have huge variance concentrated along one axis.

### The analogy

Think of a budget split across departments: equal shares are maximally spread, one department taking 85% is concentrated. Entropy over the shares measures exactly this, and the eigenvalues of $K$ are the shares of variance.

### The mechanism

1. Compute $K = ZZ^\top$ for the layer you care about.
2. Take its eigenvalues $\lambda_1, \dots, \lambda_r$, where $r$ is the rank of $K$.
3. Normalize them by the trace so they sum to one: $p_i = \lambda_i / \operatorname{tr}(K)$.
4. Treat the $p_i$ as a probability distribution and compute its entropy.

### The math

The paper uses the α-order matrix-based entropy of Giraldo et al. (2014):

$$
S_\alpha(Z) \;=\; \frac{1}{1-\alpha}\,\log \sum_{i=1}^{r} \left(\frac{\lambda_i(K)}{\operatorname{tr}(K)}\right)^{\alpha}
$$

Here $K = ZZ^\top$ is the Gram matrix, $\lambda_i(K)$ its $i$-th eigenvalue, $\operatorname{tr}(K)$ the sum of all eigenvalues, $r = \operatorname{rank}(K) \le \min(N, D)$, and $\alpha \gt 0$ is an order parameter. Setting $\alpha = 2$ gives collision entropy. Letting $\alpha \to 1$ gives the Shannon (von Neumann) form, which is the one the paper uses by default:

$$
S_1(Z) \;=\; -\sum_{i=1}^{r} p_i \log p_i, \qquad p_i = \frac{\lambda_i(K)}{\operatorname{tr}(K)}
$$

The two extremes are easy to read. If all $r$ shares are equal, $S_1 = \log r$, the maximum. If one share is 1 and the rest are 0, $S_1 = 0$. A related quantity is $\exp(S_1)$, which you can read as "the effective number of directions in use".

### A worked example

Take a toy layer whose Gram matrix has four nonzero eigenvalues.

| Spectrum | Shares $p_i$ | $-p_i \ln p_i$ per term | $S_1$ | $\exp(S_1)$ |
| --- | --- | --- | --- | --- |
| Spread out | 0.25, 0.25, 0.25, 0.25 | 0.35 each | 1.39 | 4.0 |
| Compressed | 0.85, 0.05, 0.05, 0.05 | 0.14, 0.15, 0.15, 0.15 | 0.59 | 1.80 |

Both layers live in the same four dimensions. The compressed one behaves as if it used fewer than two of them.

![Two eigenvalue spectra: four equal shares give entropy 1.39, one dominant share gives 0.59](/imgs/blogs/layer-by-layer-uncovering-hidden-representations-in-language-models-1.webp)

### Two flavors: prompt entropy and dataset entropy

The same formula is applied to two different $Z$ matrices, and the difference matters.

- **Prompt entropy** builds $Z$ from the tokens of *one* prompt. High prompt entropy means the tokens inside a sentence stay distinct; low means the layer has squeezed them toward a few shared directions.
- **Dataset entropy** builds $Z$ from mean-pooled embeddings of *many* prompts. High dataset entropy means different inputs land in different places, which is what a retrieval or clustering task needs.

The paper links the two with an informal theorem (its Theorem 2): if prompt entropy stays near its maximum for every prompt, dataset entropy grows faster with $N$ than if prompt entropy stays near its minimum. In plain words, crushing tokens together inside a prompt limits how far apart different prompts can end up.

### When it fails

Entropy is blind to *which* directions survive. A layer can keep high entropy by preserving noise, or have low entropy because it kept exactly the feature a task needs. The paper's own test shows this: random tokens *raise* early-layer entropy, which is not a sign of better representations.

## Technique 2: effective rank and the theorem connecting it

Effective rank (Roy and Vetterli, 2007) is another way to count "directions in use". It normalizes the *singular values* of $Z$ to sum to one, takes their Shannon entropy, and exponentiates. RankMe and LiDAR, two popular label-free quality scores from self-supervised vision, are built on it.

The paper's Theorem 1 states a bound between the two counts:

$$
\operatorname{EffRank}(Z) \;\le\; \exp\!\big(S_1(Z)\big)
$$

The intended message is that the representation-quality literature built on effective rank is secretly measuring matrix-based entropy, so one lens covers both. I return to this inequality in the critique, because my own arithmetic on the toy spectrum above does not agree with its direction. The qualitative point, that effective rank and $S_1$ rise and fall together, survives either way.

## Technique 3: curvature of the token path

### The problem it solves

Entropy treats tokens as an unordered cloud. But a prompt is a *sequence*, and the order in which the hidden states move through space carries information. Hosseini and Fedorenko (2023) proposed measuring how sharply that path turns.

### The analogy

Imagine tracing a pen from token 1's vector to token 2's, then to token 3's, and so on. A smooth arc means each step continues roughly in the previous direction. A zigzag means each step reverses course. Curvature is the average size of those turns.

### The mechanism and the math

Let $z_k$ be the hidden vector of token $k$ at some layer, for $k = 1, \dots, L$. Define the step between consecutive tokens as $v_k = z_{k+1} - z_k$. Average curvature is the mean angle between consecutive steps:

$$
\bar{C} \;=\; \frac{1}{L-2}\sum_{k=1}^{L-2} \arccos\!\left(\frac{v_{k+1}^\top v_k}{\lVert v_{k+1}\rVert\,\lVert v_k\rVert}\right)
$$

Each term is the angle between step $k$ and step $k+1$, in radians, between 0 (no turn) and $\pi$ (full reversal). There are $L-1$ steps and therefore $L-2$ angles. The paper's reading: higher curvature means abrupt, local, token-to-token changes; lower curvature means a smoother path dominated by global, prompt-level structure.

### A worked example

Five points give four steps and three angles. In the smooth panel below, each step turns 15 degrees, so $\bar{C}$ is 15 degrees (about 0.26 rad). In the jagged panel the turns are roughly 113, 116 and 116 degrees, so $\bar{C}$ is about 115 degrees (about 2.0 rad).

![A smooth token path turning 15 degrees per step versus a jagged one turning about 115 degrees](/imgs/blogs/layer-by-layer-uncovering-hidden-representations-in-language-models-2.webp)

### When it fails

Curvature is sensitive to tokenization: more subword pieces means more, smaller steps. It also says nothing about *where* the path is, so two prompts with identical curvature can be indistinguishable or perfectly separated.

## Technique 4: augmentation invariance (InfoNCE, LiDAR, DiME)

### The problem it solves

A good embedding should not care about typos. If you perturb a sentence slightly, its representation should move less than the distance to a genuinely different sentence. This is the self-supervised-learning idea of invariance, borrowed into an evaluation tool.

### The mechanism

For every prompt the authors create **two** perturbed copies using the NLPAug library, each passing through three character-level augmenters with probability 0.3: word splitting (inserting a space), random character insert, swap or delete, and keyboard-neighbor substitution. The paper's example turns "The quick brown fox jumps over the lazy dog." into "The quDUk b rown fox wEmps o ver the l azy dog." They then embed both copies at every layer, giving row-matched matrices $Z_1$ and $Z_2$, and score how well the copies find each other.

- **InfoNCE** treats each matched pair as a positive and every other prompt in the batch as a negative. Lower loss means the right partner is easier to pick out.
- **LiDAR** treats each prompt as its own class and measures within-class scatter (the two copies) against between-class scatter (different prompts), in the style of linear discriminant analysis. Tight, well-separated clusters score well.
- **DiME** is itself built on matrix-based entropy: it compares the entropy of the true pairing with that of random pairings, estimating how uniquely the correct copies align.

### The math

The bridge to entropy is the paper's Theorem 3, a standard InfoNCE bound restated for this setting:

$$
\log N - \mathcal{L}_{\text{InfoNCE}} \;\le\; I(X; Z) \;\le\; H(Z)
$$

$N$ is the number of prompts in the batch, $\mathcal{L}_{\text{InfoNCE}}$ the contrastive loss, $I(X; Z)$ the mutual information between inputs and their embeddings, and $H(Z)$ the entropy of the embeddings, which the paper identifies with dataset-level matrix-based entropy. Reading right to left: a layer cannot achieve low InfoNCE loss unless its dataset entropy is high enough to hold that much information.

### A worked example

With $N = 256$ prompts, $\log N \approx 5.55$ nats. If a layer's InfoNCE loss is 1.55, the bound says that layer carries at least 4.0 nats of information about which prompt went in, so its dataset entropy must be at least 4.0 nats too. (These numbers are mine, chosen to show how the inequality is read; the paper does not report this computation.)

### When it fails

The perturbations are all character-level. A layer invariant to typos might still be fragile to paraphrase, the more relevant invariance for most embedding tasks.

## Do the metrics track downstream score?

A metric is only useful if it predicts something. The authors compute, for Pythia-410M, the distance correlation (dCor) between each metric's layer-wise curve and the layer-wise task score. Distance correlation lies in [0, 1], with 0 meaning statistical independence, and it catches nonlinear as well as linear relationships.

| Metric | Family | dCor with downstream score (Pythia-410M) |
| --- | --- | --- |
| DiME | invariance | strongest |
| Curvature | geometric | high |
| InfoNCE | invariance | high |
| Prompt entropy, dataset entropy, LiDAR | info / invariance | strong but lower |

The ordering holds under Spearman and Kendall correlations too. Note what this is: a correlation across roughly 24 layers of one model, showing the metric curves share the *shape* of the score curve, not that any metric guarantees performance.

## What the metrics reveal across architectures

With the toolkit in place, the paper measures every layer of Pythia-410M (decoder transformer), Mamba-370M (state-space model) and BERT-base (encoder) on WikiText-103 prompts.

![Figure 2 from Skean et al. (2025): prompt entropy, curvature and LiDAR across depth for Pythia, Mamba and BERT](/imgs/blogs/layer-by-layer-uncovering-hidden-representations-in-language-models-fig2.webp)

Three distinct behaviors appear:

- **Pythia (autoregressive)** shows a deep mid-layer dip in prompt entropy. The authors read this as the next-token objective forcing the network to prune non-local detail in the middle, which is also where its downstream score peaks.
- **BERT (masked, bidirectional)** keeps entropy high across depth. It sees the whole input at once and has less need to discard.
- **Mamba (state-space)** sits in between with flatter curves: less retention than BERT, less compression than Pythia.

### Scale, training time and sub-layers

Four further measurements sharpen the picture.

1. **Scale.** Across Pythia from 14M to 1B parameters, larger models show *deeper* mid-layer entropy dips, smoother token paths (lower curvature) and higher LiDAR.
2. **Training progression.** Tracking Pythia-410M from step 1 to step 143k, the mid-depth layers change the most: their prompt entropy falls steadily as training goes on. The earliest layers settle almost immediately, consistent with the "detokenization" view that early layers mainly turn raw tokens into a usable embedding space.
3. **Inside the block.** Measuring after each sub-layer (pre-attention norm, attention, MLP, residual adds), the strong entropy drop shows up at the *residual* sub-layers, while pre-residual outputs compress only mildly. The authors tie the valley to how the residual stream merges new signal with the existing state.
4. **Finetuning.** Converting Llama3-8B into LLM2Vec (bidirectional attention plus contrastive training) improves augmentation invariance. The unsupervised LLM2Vec variant *raises* prompt entropy relative to Llama3, while the supervised variant *lowers* it.

### Extreme inputs and chain of thought

On Pythia-410M, more repeated tokens sharply lower mid-layer entropy (redundancy is discarded), while random tokens inflate *early*-layer entropy.

For chain-of-thought, the paper compares Qwen 2.5 with Qwen 2.5-Math, which had extra math pretraining and CoT finetuning. Measured token by token along the sequence, the finetuned model keeps higher prompt entropy with lower variance across examples. The authors read this as CoT training preserving more context in the hidden layers. That interpretation is plausible but untested here: no reasoning accuracy is tied to the entropy difference.

## The domain test: is it text, or is it the objective?

If the mid-layer bottleneck were a property of language, vision transformers should not show it. The authors evaluate seven ViT-L style models (24 layers each) with attention probing on frozen layers: supervised ViT, CLIP, BEiT (masked patches), DINOv2, MAE, AIM (autoregressive next-patch prediction) and AIMv2 (multimodal autoregressive).

![Figure 14 from Skean et al. (2025): probing accuracy and metrics across the layers of seven vision models](/imgs/blogs/layer-by-layer-uncovering-hidden-representations-in-language-models-fig3.webp)

The result is clean. Every model except AIM gets steadily *better* toward its final layer, matching older observations on ResNets. AIM alone shows the language-model signature: an entropy valley and an accuracy peak at intermediate depth.

Two edge cases keep the story honest. AIMv2 produces an entropy valley but *no* mid-layer accuracy gain, which the authors attribute to its text-vision pretext task. BEiT, a masked model, shows a substantial intermediate dip in dataset entropy without the accuracy peak.

![Grid of models by training objective, mid-depth entropy dip and best downstream layer](/imgs/blogs/layer-by-layer-uncovering-hidden-representations-in-language-models-3.webp)

The summary grid makes the paper's conclusion visible: the column that best predicts a mid-depth peak is the training objective, not the modality. Note what the grid also shows, though. An entropy dip is *not sufficient* for a mid-layer peak (AIMv2, BEiT), and BERT peaks mid-depth with little dip. So "valley implies use the middle layer" is a tendency, not a rule.

## Technique 5: picking a layer without labels

### The problem it solves

Knowing that "some middle layer" is better is not actionable. You need to pick one, and for a new task you may have no labels to evaluate with.

### The mechanism

Because the metrics correlate with task score, the authors use them as a proxy. For each task, they run one forward pass over the task's unlabeled inputs, compute DiME, InfoNCE or dataset entropy at every layer, and pick the layer that minimizes the chosen metric. Then they report the MTEB score of that layer.

![Label-free layer selection: augment, score every layer with DiME, pick the minimum](/imgs/blogs/layer-by-layer-uncovering-hidden-representations-in-language-models-4.webp)

### The results

![Table 2 from Skean et al. (2025): average MTEB performance under different layer selection schemes](/imgs/blogs/layer-by-layer-uncovering-hidden-representations-in-language-models-fig4.webp)

| Model | Best layer (labels) | Last layer | min-DiME | min-InfoNCE | min-dataset entropy |
| --- | --- | --- | --- | --- | --- |
| Pythia-410M | 52.0 | 45.5 | **48.5** | 46.2 | 48.1 |
| LLM2Vec-8B | 66.3 | 63.9 | 60.0 | **64.3** | 50.4 |

For Pythia-410M the headline holds: DiME-based selection lifts the average MTEB score from 45.5 to 48.5, the 3% the paper quotes, and closes about 46% of the gap to the label-chosen best layer (3.0 of 6.5 points; the ratio is my arithmetic).

LLM2Vec-8B tells a different story. Only InfoNCE beats the last layer, and only by 0.4 points. DiME, the winner on Pythia, *drops* the score by 3.9 points, and dataset entropy drops it by 13.5. A model already finetuned for embeddings has a last layer that is close to optimal, and a mis-chosen proxy can do real damage.

### When it fails

My reading of Table 2: label-free selection is worth trying on a *raw pretrained* decoder, whose last layer is specialized to next-token prediction, not on a model already trained as an encoder. Which metric wins is model-dependent, and the paper gives no rule for choosing it in advance.

## What is load-bearing and might not transfer

- **Mean pooling.** Every embedding score assumes mean-pooled hidden states. Decoders are often used with last-token pooling, and the layer ranking could change under a different pooling rule.
- **Small models.** Llama3-8B is the largest model; assuming the same layer fraction wins at 70B is extrapolation.
- **Typo-level augmentations.** All invariance metrics use character noise. Paraphrase or translation invariance is untested.
- **Dataset mismatch.** The metric curves come from WikiText-103 while the scores come from MTEB, so the correlation is between two different data distributions.

## Critique

### What is strong

The core empirical finding is robust and useful. It replicates across three architecture families, a range of scales and a second modality, and the vision experiment is a genuinely good control: it separates "sequential data" from "sequential objective" and the objective wins. For an interpretability reader, the training-dynamics and sub-layer results are the most interesting part, because they localize *where* and *when* the compression happens (mid-depth, at the residual adds, progressively during training).

The paper also connects to a broader picture of the middle of the network as the most abstract part of the model, the same region the [semantic hub hypothesis](/blog/paper-reading/ai-interpretability/the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities) identifies as shared across languages and modalities.

### What is weak

- **The unification is mostly vocabulary.** Saying that entropy, curvature and invariance are "all facets of how the Gram matrix distributes variance" is true of almost any second-order statistic. The framework never predicts *which* layer will win before you measure, and the metrics disagree with each other on LLM2Vec.
- **Theorem 1 looks reversed.** My check on the toy spectrum above: eigenvalue shares (0.85, 0.05, 0.05, 0.05) give $\exp(S_1) = 1.80$. The singular values are the square roots of those eigenvalues; normalized, they are about (0.58, 0.14, 0.14, 0.14), whose entropy gives an effective rank of about 3.14. That is *larger* than $\exp(S_1)$, the opposite of the stated bound. In general the square-root spectrum is flatter, so effective rank should be the upper quantity. The proof in the appendix assumes $\sigma_i^2 = \lambda_i$ after normalizing both sets separately, which does not hold. This is my analysis, not something the paper acknowledges; the monotone relationship between the two quantities, which is what the argument needs, is unaffected.
- **Correlation across layers is weak evidence.** With about 24 points per model, a metric that merely has a hump in the middle will correlate with a score that has a hump in the middle. A stronger test would predict the best layer on held-out models.
- **The CoT claim is unsupported by an outcome.** Higher entropy in Qwen 2.5-Math is shown, better reasoning *because of* that entropy is not.
- **Small inconsistencies.** LiDAR is described as *minimal* in the mid layers during training yet *higher* in larger models as evidence of stronger invariance, and the vision dataset is called ImageNet-1k in the text but ImageNet-100 in the Figure 14 caption.

### Missing ablations

A pooling ablation, a paraphrase-based invariance metric, and a 30B-plus model would each test a load-bearing assumption. Most useful of all would be an intervention: if the valley is what makes mid layers good, a regularizer that prevents it should hurt them. The authors list "explicit finetuning to control compression" as future work.

### What would change my mind

I would drop the "objective, not modality" conclusion if a masked or contrastive model trained at the same scale as AIM showed a mid-depth accuracy peak, or if an autoregressive model with a large mid-depth valley showed accuracy rising monotonically to the last layer on a broad task suite. AIMv2 is already one partial counterexample on the second count. I would drop the practical recommendation if last-token pooling made the final layer best for raw decoders.

## Practical takeaways

- When using a **raw pretrained decoder** for embeddings, sweep a few layers around 40% to 70% depth before defaulting to the last one.
- When using a model **already trained for embeddings**, the last layer is a strong default; validate before switching.
- If you have no labels, DiME or InfoNCE-based selection is cheap (one forward pass), but treat its pick as a candidate, not an answer.
- For interpretability work, mid-depth layers are the natural place to look for [linear probes and steering directions](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis).

## References

1. Skean, O., Arefin, M. R., Zhao, D., Patel, N., Naghiyev, J., LeCun, Y., Shwartz-Ziv, R. *Layer by Layer: Uncovering Hidden Representations in Language Models.* ICML 2025. [arXiv:2502.02013v2](https://arxiv.org/abs/2502.02013v2)
2. Code: [github.com/OFSkean/information_flow](https://github.com/OFSkean/information_flow)
3. Related on this site: [The Semantic Hub Hypothesis](/blog/paper-reading/ai-interpretability/the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities) and [The Linear Representation Hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis).

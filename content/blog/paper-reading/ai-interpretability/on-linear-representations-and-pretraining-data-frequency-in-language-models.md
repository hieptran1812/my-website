---
title: "On Linear Representations and Pretraining Data Frequency: How Often a Fact Appears Decides Whether It Becomes Linear"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How Merullo et al. tie the formation of linear relational embeddings in OLMo and GPT-J to subject-object co-occurrence counts in pretraining, and then run the link backwards to estimate the training data of a model."
excerpt: "A fact seen together about two thousand times in pretraining tends to become a clean linear map inside the model. This ICLR 2025 paper measures that threshold across OLMo checkpoints and GPT-J, then uses linearity to estimate term counts in a model's hidden training set."
tags:
  - paper-reading
  - ai-interpretability
  - linear-representations
  - linear-relational-embeddings
  - data-frequency
  - pretraining
  - mechanistic-interpretability
  - olmo
  - iclr-2025-poster
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 19
paper:
  title: "On Linear Representations and Pretraining Data Frequency in Language Models"
  authors: "Jack Merullo, Noah A. Smith, Sarah Wiegreffe, Yanai Elazar"
  venue: "ICLR 2025"
  url: "https://arxiv.org/abs/2504.12459"
---

> [!tldr]
> **Claim.** Whether a language model stores a factual relation (country to capital, musician to instrument) as a clean linear map depends strongly on how often the subject and object co-occurred in pretraining. Correlation between average co-occurrence and LRE causality is r = 0.82 in OLMo-7B.
>
> **Why it matters.** The Linear Representation Hypothesis says models encode features linearly, but never said *when*. This paper gives a measurable, data-side answer: a count threshold.
>
> **Most surprising result.** The threshold does not care *when* in training the counts accumulate. And the arrow runs backwards: a regressor trained only on OLMo reads GPT-J's LRE scores and predicts object counts in the Pile within 10x about 65% of the time.
>
> **Where it fails.** Everything is correlational (no counterfactual pretraining), only three models are studied, and predicting subject-object co-occurrence counts barely beats a mean baseline.

This paper asks a data question: *why did this relation become linear, and that one not?* Hernandez et al. (2024) had already found that "country-largest-city" can be approximated by a single linear map for over 80% of its entities, while "star-in-constellation" manages under 30%. Merullo, Smith, Wiegreffe and Elazar suggest the difference is mostly about exposure.

The paper does two things. First, it counts every co-occurrence of about 10k terms inside OLMo's actual training batches and lines those counts up with LRE quality across eight checkpoints. Second, it turns the relationship around and trains a regressor that reads a model's linearity scores and guesses term frequencies in its training corpus.

![Figure 1 from Merullo et al. (2025): count subject-object co-occurrences in the pretraining corpus, measure how linearly the model maps subjects to objects, and correlate the two](/imgs/blogs/on-linear-representations-and-pretraining-data-frequency-in-language-models-fig1.webp)

Figure 1 is the whole paper in one picture. On the left, a counter scans the corpus and returns 2,128 for ("Miles Davis", "trumpet"). On the right, the model's mapping from musicians to instruments is tested for linearity. The scatter plot joins the two, with r = .82 across relations.

Background on the hypothesis being tested is in the [Linear Representation Hypothesis explainer](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis).

## Technique 1: Linear Relational Embeddings (LREs)

### The problem

A transformer answers "Miles Davis plays the" with "trumpet" after dozens of layers of attention and MLPs. That computation is deeply nonlinear. If we want to say "the model represents the plays-instrument relation linearly", we need a concrete test, not a vibe.

### The analogy

Picture a currency converter. Internally a bank may route your money through a dozen ledgers and fees, but for small amounts the whole thing behaves like "multiply by 25,000 and add a flat fee". An LRE asks whether the model's subject-to-object computation behaves like that converter: one multiply, one add, the same for every subject in the relation.

The "multiply" is the matrix $W$, the "flat fee" is the bias $b$, and "the same for every subject" is the point. One map per relation, not one per fact.

### The mechanism

Take a few-shot prompt such as four "[X] plays the [Y]" examples followed by "Miles Davis plays the". Read off two hidden states:

- $s$, the representation of the **subject's last token** at some middle layer.
- $o$, the hidden state at the **last token position and the last layer**, which is what the model decodes into "trumpet".

The real computation from $s$ to $o$ runs through every later layer and every token position. The LRE replaces it with a first-order Taylor approximation, averaged over a handful of examples so it works for the relation rather than one subject.

![Two lanes: the language model's real nonlinear path from subject state s to final state o, and the LRE shortcut that maps s to o with a single affine map W s plus b](/imgs/blogs/on-linear-representations-and-pretraining-data-frequency-in-language-models-1.webp)

The top lane in the figure is what the model does; the bottom lane is the claim being tested. If the bottom lane lands on the same token, the relation is "linear" in the sense this paper uses.

### The math

Let $F(s, c) = o$ be the forward pass from subject state $s \in \mathbb{R}^{d}$ to object state $o \in \mathbb{R}^{d}$, given a few-shot context $c$, where $d$ is the hidden size. Linearize around a specific subject $s_i$:

$$
F(s, c) \approx W s + b = F(s_i, c) + W (s - s_i)
$$

Hernandez et al. estimate $W \in \mathbb{R}^{d \times d}$ and $b \in \mathbb{R}^{d}$ by averaging over $n$ examples $(s_i, c_i)$ from the relation (the paper uses $n = 8$, each with a 5-shot prompt):

$$
W = \mathbb{E}_{s_i, c_i}\left[ \frac{\partial F}{\partial s}\Big|_{(s_i, c_i)} \right], \qquad
b = \mathbb{E}_{s_i, c_i}\left[ F(s, c) - \frac{\partial F}{\partial s}\Big|_{(s_i, c_i)} s \right]
$$

Here $\partial F / \partial s$ is the Jacobian, a $d \times d$ matrix saying how each coordinate of $o$ moves when each coordinate of $s$ moves. Averaging Jacobians over eight subjects is what makes $W$ a property of the *relation*.

Hernandez et al. found this average underestimates the slope, so $W$ is multiplied by a scalar $\beta$ before use. Merullo et al. tune one $\beta$ **per relation** instead of one per model, so no relation is handicapped by a shared setting.

### A worked micro-example

Shrink the model to $d = 2$. Suppose for the plays-instrument relation the averaged Jacobian and bias come out as

$$
W = \begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}, \quad b = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \beta = 1.5
$$

For a subject state $s = (1, 2)$, the LRE predicts $\beta W s + b = 1.5 \cdot (2, 1) + (1, 0) = (4, 1.5)$. If the unembedding maps $(4, 1.5)$ closest to "trumpet", and the real model also says "trumpet", that subject passes. The same $W$, $b$, $\beta$ must also send "Cat Stevens" to "guitar" and "Bill Evans" to "piano". That shared map is the whole test.

### Two departures from the original recipe

The authors change two things relative to Hernandez et al., and both matter for comparing checkpoints:

| Choice | Hernandez et al. (2024) | Merullo et al. (2025) | Why |
| --- | --- | --- | --- |
| Scale $\beta$ | One per model | One per relation | Avoids disadvantaging specific relations |
| Fitting examples | Only ones the model gets right | Any 8 examples, right or wrong | Lets every checkpoint use the *same* 8 examples |
| Layer | Swept | Swept, chosen by causality | One LRE serves both metrics |

The second change sounds risky: why fit a map on examples the model gets wrong? The appendix reports that LREs fit on all-correct, half-correct or zero-correct examples score about the same. That is convenient, because an early checkpoint gets many facts wrong and would otherwise have no clean fitting set.

### When it fails

An LRE is one affine map per relation. It cannot handle a relation where the answer depends on the subject in a way that is not roughly linear, and it says nothing about relations with many valid objects. The authors drop "landmark-on-continent" for a related reason: 74% of its answers are Antarctica, so a constant map would look deceptively good.

## Technique 2: Scoring an LRE with faithfulness and causality

### The problem

Fitting an LRE always produces *some* matrix. We need a score that says whether that matrix captures the relation or is just noise that happens to decode to common tokens.

### The analogy

Testing a translation dictionary has two levels. Weak: does a lookup match what a native speaker says? Strong: if you change the entry, does the speaker's answer change to match? The first is faithfulness; the second, causality, proves the dictionary is actually *in use*.

### The mechanism

- **Faithfulness.** Apply the LRE to $s$, decode the top token, and check it matches the model's own top prediction.
- **Causality.** Pick a target object from another example in the relation ("guitar", from Cat Stevens). Use the LRE in reverse to compute how $s$ for "Miles Davis" would have to change to produce "guitar". Write the edited state back into the real model and run it forward. Count a success if the model now predicts the new object.
- **Hard causality.** Same as causality, but only counts it when the target becomes the number-one prediction.
- **Faith prob.** The log probability the LRE assigns to the correct answer.

![Two panels: faithfulness checks whether the LRE decodes to the same token as the model; causality inverts W to edit the subject state and checks whether the real model switches to the target object](/imgs/blogs/on-linear-representations-and-pretraining-data-frequency-in-language-models-2.webp)

The right panel is the important one. Faithfulness only reads the LRE. Causality writes into the real model, so a high score means the model's own downstream layers respect the linear map.

### The math

The paper defers implementation details to Hernandez et al., but the appendix confirms the edit uses a low-rank pseudoinverse whose rank is a tuned hyperparameter. In that formulation, with $o'$ the object state for the target answer and $W_r^{+}$ the rank-$r$ pseudoinverse of $W$, the subject edit is

$$
\Delta s = W_r^{+} \left( o' - o \right), \qquad s' = s + \Delta s
$$

and causality over a relation with $N$ test pairs is the fraction of edits that flip the model:

$$
\text{Causality} = \frac{1}{N} \sum_{j=1}^{N} \mathbb{1}\left[ \text{LM}(s'_j) = \text{target}_j \right]
$$

(The exact form of $\Delta s$ is my reading of the Hernandez et al. method, not something this paper restates.) The rank sweep runs from 0 to the full hidden size at coarse intervals, and $\beta$ is swept over $[0, 5]$.

### Why the paper prefers causality

A young checkpoint that predicts one common token for every prompt will match its own LRE often, scoring high faithfulness while knowing nothing. Causality cannot be gamed that way: the edit must steer the model to a *specific* different object. The authors use causality as their main quality signal and call a relation "nearly perfectly linear" above 0.9.

## Technique 3: Counting co-occurrences inside training batches

### The problem

To link LRE quality to data, you need the count of how many times "Miles Davis" and "trumpet" appeared together *up to checkpoint t*. Existing index tools such as WIMBD (What's In My Big Data) count over the whole dataset and at document level. Neither matches what the model saw at step 10k.

### The mechanism

OLMo releases tooling to reconstruct its exact training batches from Dolma. The authors wrote **Batch Search**, which scans tokenized batches, each a `(batch-size, sequence-length)` array, and records every position where one of their ~10k terms appears. A co-occurrence is counted whenever two terms land in the **same training sequence**.

Following Elsahar et al. (2018) and Elazar et al. (2022), a subject-object co-occurrence is used as a proxy for the fact being stated. Elsahar et al. showed the proxy is quite accurate for factual triplets.

### The math

For a subject $x$, object $y$, and checkpoint $t$, let $\mathcal{S}_t$ be the set of training sequences seen up to step $t$. Then

$$
C_t(x, y) = \sum_{q \in \mathcal{S}_t} \mathbb{1}\left[ x \in q \wedge y \in q \right]
$$

The relation-level quantity in every plot is the average over the relation's pairs, $\bar{C}_t(R) = \frac{1}{|R|} \sum_{(x,y) \in R} C_t(x, y)$, and the axes use its log.

### A worked micro-example

Take one batch of 3 sequences:

| Sequence | Contains "Miles Davis" | Contains "trumpet" | Counts? |
| --- | --- | --- | --- |
| 1 | yes | yes | 1 |
| 2 | yes | no | 0 |
| 3 | no | yes | 0 |

$C = 1$. A document-level counter would see a single long article covering sequences 2 and 3 as one unit and count 1 more. That is exactly why WIMBD over-counts.

### Scale and validation

Searching ~10k terms (10,488 unique subjects and objects) through ~2T tokens of Dolma took 900 CPUs about a day. Against WIMBD on the final OLMo checkpoint, the best-fit slope is 0.94 (r = 0.99); the gap is the document-versus-sequence effect. For GPT-J, whose batches cannot be reconstructed, the authors fall back to WIMBD counts over the Pile.

## Result 1: a co-occurrence threshold for linearity

The setup uses 25 factual relations from the Hernandez et al. RELATIONS dataset (24 after dropping landmark-on-continent), three models (OLMo-7B 0424, OLMo-1B 0724, GPT-J 6B), and eight OLMo checkpoints at {41B, 104B, 209B, 419B, 628B, 838B, 1T, 2T} tokens.

![Figure 2 from Merullo et al. (2025): LRE causality against log subject-object co-occurrence for OLMo-7B, OLMo-1B and GPT-J, with the per-model threshold where mean causality exceeds 0.9](/imgs/blogs/on-linear-representations-and-pretraining-data-frequency-in-language-models-fig2.webp)

Each point is a relation at one checkpoint. Red points are the 41B-token checkpoint (10k steps), blue points the final model, gray points the checkpoints in between. The dashed line is the threshold.

### The threshold, defined

The paper defines the threshold as the average co-occurrence count above which mean causality exceeds 0.9:

$$
\tau = \min \left\{ c \;:\; \operatorname{mean}_{R \,:\, \bar{C}(R) \geq c} \text{Causality}(R) > 0.9 \right\}
$$

| Model | Threshold $\tau$ (mean causality > 0.9) | $\ln \tau$ |
| --- | --- | --- |
| GPT-J (6B) | 1,097 | 7.0 |
| OLMo-7B | 1,998 | 7.6 |
| OLMo-1B | 4,447 | 8.4 |

The $\ln \tau$ column is my arithmetic. It matches where the dashed lines sit on the plots, which suggests the axis is a natural log.

### What the numbers say

- **Co-occurrence beats single-term frequency.** Causality correlates with subject-object co-occurrence at r = 0.82, against r = 0.66 for subject frequency and r = 0.59 for object frequency.
- **Timing does not matter.** A relation that crosses the threshold early is already linear at 41B tokens. Red points above the threshold sit as high as blue ones.
- **Scale may lower the bar.** OLMo-1B needs about twice the exposure of OLMo-7B. With only three models, the authors are careful not to claim a scaling law.

A worked reading: "country-largest-city" sits at the top right of the OLMo-7B panel, high co-occurrence and causality near 1. "star-constellation-name" averages about 21 co-occurrences across the whole dataset, far below 1,998, and its causality stays at 0.44.

## Result 2: linearity tracks accuracy, but not perfectly

Is this just "the model knows common facts" again? The authors check by correlating causality with 5-shot accuracy.

| Comparison (OLMo-7B) | Value |
| --- | --- |
| Corr(causality, co-occurrence) | 0.82 |
| Corr(causality, 5-shot accuracy) | 0.74 |
| star-constellation-name | 84% accuracy, 44% causality |
| food-from-country (mid-training) | 65% causality, 42% accuracy |
| Final model, avg gap causality vs accuracy | within 11% |

Mismatches run both ways. Low-frequency relations like star-constellation-name and product-by-company reach high few-shot accuracy *without* a consistent LRE, while others become linear before they become accurate. Linearity and accuracy are linked, but frequency predicts linearity better.

## Technique 4: predicting term frequency from linearity

### The problem

If frequency shapes linearity, linearity should leak frequency. Most open-weight models ship without their data, so measuring LREs could estimate term counts in a hidden corpus.

### The mechanism

Train a regressor on a model whose data you can count (OLMo-7B, Dolma). Features per example come in two sets:

- **LM-only baseline:** log probability of the correct answer under a few-shot prompt, plus average accuracy across 5 trials.
- **LRE + LM:** the baseline plus faithfulness, causality, faith prob and hard causality.

The target is either the object's count or the subject-object co-occurrence count. The model is a random forest with 100 trees. Every relation is held out for evaluation, rotated across 4 random seeds, and held-out objects and relations are guaranteed absent from training.

![Pipeline: train a random forest on OLMo-7B LRE and LM features against true Dolma counts, then apply the same regressor to GPT-J features to predict Pile counts, scored by whether each prediction lands within 10x of the truth](/imgs/blogs/on-linear-representations-and-pretraining-data-frequency-in-language-models-3.webp)

### The math

Exact counts are hopeless, so the paper scores **within-magnitude accuracy**: the fraction of predictions within one order of magnitude of the truth.

$$
\text{Acc}_{10\times} = \frac{1}{M} \sum_{k=1}^{M} \mathbb{1}\left[ \left| \log_{10} \hat{y}_k - \log_{10} y_k \right| \leq 1 \right]
$$

where $y_k$ is the true count for example $k$, $\hat{y}_k$ the predicted count, and $M$ the number of held-out examples. It also reports mean absolute error in natural-log space.

### A worked micro-example

From the paper's Table 2: for (Menangle Park, Australia) the regressor predicts 2,986,989 against a true 3,582,602. The ratio is ${3{,}582{,}602 / 2{,}986{,}989} \approx 1.2$, so $|\log_{10}|$ error is about 0.08, a hit. For (Arcturus, Boötes) it predicts 974,550 against 2,817, ratio 346, error about 2.5 orders of magnitude, a miss.

### Results on held-out relations (OLMo)

![Figure 3 from Merullo et al. (2025): within-magnitude accuracy for predicting object occurrences and subject-object co-occurrences, LRE plus LM features versus LM-only features, with random and mean baselines](/imgs/blogs/on-linear-representations-and-pretraining-data-frequency-in-language-models-fig3.webp)

| Target | LRE+LM features | LM-only features |
| --- | --- | --- |
| Object counts, within-10x accuracy | ~0.70 | ~0.40 |
| Object counts, MAE (ln space) | 2.1 | 4.2 |
| Subject-object counts, MAE (ln space) | 1.9 | 2.3 |

LM-only features barely beat predicting the mean or a random count. LRE features add about 30 points on object frequency. Subject-object prediction is much harder: LRE features are above but within one standard deviation of the mean baseline.

### Which feature carries the signal

![Figure 12 from Merullo et al. (2025): permutation importances for the regression, with hard causality causing about a 0.15 drop in accuracy when shuffled, far above the other features](/imgs/blogs/on-linear-representations-and-pretraining-data-frequency-in-language-models-fig6.webp)

Faithfulness and faith prob are strongly correlated, so for this test they are collapsed into one PCA component that keeps 89% of their variance. Shuffling **hard causality** drops accuracy by about 15 points; the faithfulness component by about 5. Log probability and accuracy matter little. The signal is in the editing metric, the one that touches the model's actual computation.

## Result 3: transfer to a model whose data you pretend not to have

Now the real test. Fit the regressor on OLMo-7B (final checkpoint) with Dolma counts, then apply it to GPT-J features and score against Pile counts. Then do the reverse. Predictions are scaled by the ratio of total training tokens between the two models.

![Table 1 from Merullo et al. (2025): within-magnitude accuracy when a regression fit on one model is evaluated on the other, for LRE features, log-prob features and a mean-frequency baseline](/imgs/blogs/on-linear-representations-and-pretraining-data-frequency-in-language-models-fig4.webp)

| Features | Object: eval GPT-J | Object: eval OLMo | Subj-Obj: eval GPT-J | Subj-Obj: eval OLMo |
| --- | --- | --- | --- | --- |
| LRE features | **0.65 ± 0.12** | **0.49 ± 0.12** | **0.76 ± 0.12** | **0.68 ± 0.08** |
| LogProb features | 0.42 ± 0.10 | 0.41 ± 0.09 | 0.66 ± 0.09 | 0.60 ± 0.07 |
| Mean freq. baseline | 0.31 ± 0.15 | 0.41 ± 0.17 | 0.57 ± 0.15 | 0.67 ± 0.16 |

Read the object columns first. OLMo-trained, GPT-J-evaluated reaches 0.65, roughly 5 points below the within-model number from Figure 3. So LREs seem to encode frequency in a way that is similar across two models trained on different corpora.

The subject-object columns are weaker: on OLMo evaluation, 0.68 against a mean baseline of 0.67, because those targets cluster tightly around the mean.

### Error analysis

![Table 2 from Merullo et al. (2025): example object-frequency predictions on GPT-J from a regression fit on OLMo, with prediction, ground truth and error ratio](/imgs/blogs/on-linear-representations-and-pretraining-data-frequency-in-language-models-fig5.webp)

Three patterns stand out:

1. **Country-flavored relations transfer well.** Menangle Park to Australia is off by 1.2x.
2. **Low-frequency relations fail badly.** Arcturus to Boötes is off by 346x, consistent with star-constellation-name never forming a good LRE.
3. **The subject matters.** Prince William to Princess Diana predicts 5,826 (4.6x off), Prince Harry to the same object predicts 131 (207x off). Same object, same truth of 27,094, wildly different estimates.

The regressor does use its full range: it also predicts 59 occurrences for Caroline Bright (Will Smith's mother) against a true 48.

## What is load-bearing and might not transfer

- **Reconstructable batches.** The per-checkpoint result needs OLMo's exact batch order. Most model families do not publish this.
- **Factual triplets with single objects.** Co-occurrence is a good proxy for "Paris is the capital of France". It is a poor proxy for commonsense relations: on 8 commonsense relations the correlation drops to .42, and the authors caution that co-occurrence does not track how "historian" gets defined.
- **The LRE definition.** Thresholds are tied to one estimator and its sweeps; a different linear probe could move the line.

## Critique

### What is strong

The data work is the real contribution. Counting inside reconstructed training batches, at eight checkpoints, replaces hand-waving about "frequency in pretraining" with numbers. The finding that exposure, not training stage, decides linearity is clean. And the inverse direction is a new kind of data-inference signal drawn from representation geometry rather than membership inference on text.

### What is weak

- **Correlation, not causation.** Frequent relations are also simple, regular and well-formatted. Without counterfactual pretraining (adding or removing occurrences), "frequency causes linearity" is a hypothesis the data fits, not a demonstrated mechanism.
- **Three models.** The scale effect (1B needing ~2x the exposure) rests on one pair of OLMo models plus a GPT-J measured with a different counter.
- **A small reporting mismatch.** The abstract says OLMo-7B and GPT-J form LREs at "1k and 2k times, respectively", while the Figure 2 table gives OLMo-7B 1,998 and GPT-J 1,097. The table looks like the source of truth; the abstract appears to have the order flipped.
- **Table 2 arithmetic.** The Brazil to Portuguese row lists 845,406 against 561,005 with error "1x"; the ratio is about 1.5x. Still within 10x, so it does not change any accuracy number.
- **The co-occurrence regressor is near baseline.** For the harder target the headline gain over the mean is one point on OLMo.

### Missing ablations

A sweep over the 0.9 causality cutoff would show how sensitive the thresholds are. And since frequencies cluster by relation, I would want evidence the forest learns "how frequent" rather than "which relation".

### What would change my mind

If a controlled pretraining run that injects a relation at 3k co-occurrences, below and above the threshold, showed no jump in causality, I would stop believing frequency is causal here and treat it as a marker of something else (relation regularity, formatting). Conversely, if the OLMo-trained regressor kept ~0.65 within-10x accuracy on a third model family with a third corpus, I would treat LRE-based data inference as a practical audit tool rather than a two-model curiosity.

## What I would do with this

My extrapolations, not the paper's claims: before concluding that a concept is represented nonlinearly, check whether it plausibly crossed ~2k co-occurrences in pretraining. A missing LRE below that line is expected. And a weak LRE for a domain you care about may be a data-coverage signal that a better training mix can fix.

For a contrast with this paper's static picture of linearity, see [how linear representations can shift over a single conversation](/blog/paper-reading/ai-interpretability/linear-representations-in-language-models-can-change-dramatically-over-a-conversation). For the geometric side of how concepts are laid out once they are linear, see [the geometry of categorical and hierarchical concepts](/blog/paper-reading/ai-interpretability/the-geometry-of-categorical-and-hierarchical-concepts-in-large-language-models).

## References

1. Merullo, J., Smith, N. A., Wiegreffe, S., Elazar, Y. *On Linear Representations and Pretraining Data Frequency in Language Models.* ICLR 2025. [arXiv:2504.12459](https://arxiv.org/abs/2504.12459)
2. Code: [github.com/allenai/freq](https://github.com/allenai/freq) and Batch Search at [github.com/allenai/batchsearch](https://github.com/allenai/batchsearch)
3. Hernandez, E. et al. *Linearity of Relation Decoding in Transformer Language Models.* ICLR 2024.
4. Elazar, Y. et al. *What's In My Big Data?* ICLR 2024.
5. On this site: [The Linear Representation Hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis)
6. On this site: [Linear representations can change dramatically over a conversation](/blog/paper-reading/ai-interpretability/linear-representations-in-language-models-can-change-dramatically-over-a-conversation)

---
title: "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "Why 113 datasets, four hard probing regimes and a fair model-selection protocol found that SAE probes do not beat plain logistic regression, and where the apparent SAE wins came from."
category: "paper-reading"
subcategory: "AI Interpretability"
tags:
  - paper-reading
  - sparse-autoencoders
  - sparse-probing
  - linear-probes
  - evaluation
  - mechanistic-interpretability
  - gemma-scope
  - icml-2025-poster
author: "Hiep Tran"
featured: true
readTime: 19
image: "/imgs/blogs/are-sparse-autoencoders-useful-a-case-study-in-sparse-probing-1.webp"
excerpt: "Kantamneni, Engels et al. put SAE probes up against five ordinary baselines on 113 datasets in four hard regimes. The SAE never earns its place in the toolkit, and the paper shows exactly how earlier positive results were produced by weak baselines."
paper:
  title: "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing"
  authors: "Subhash Kantamneni, Joshua Engels, Senthooran Rajamanoharan, Max Tegmark, Neel Nanda"
  venue: "ICML 2025 (poster)"
  url: "https://arxiv.org/abs/2502.16681"
---

> [!tldr]
> **Claim.** If sparse autoencoder (SAE) latents really are a better, more meaningful basis for a model's activations, then probes built on them should win where ordinary probes struggle: little data, skewed classes, noisy labels, shifted inputs.
>
> **What they did.** 113 binary classification datasets, Gemma-2-9B activations, Gemma Scope SAEs, five strong baselines, and a selection protocol (the "quiver of arrows") that only lets a method in if it wins on *validation* data.
>
> **Result.** Adding SAE probes to the toolkit changes mean test AUC by -0.003 ± 0.002 in standard conditions and by nothing measurable in the three hard regimes. Under covariate shift SAE probes are *worse* than logistic regression.
>
> **Most surprising.** Every interpretability win they found (spotting mislabeled CoLA sentences, a spurious "ends in a period" cue in AI-text detection) was also reachable with a plain logistic-regression probe. And the earlier positive SAE-probing result looks largely like an unequal-pooling artifact.
>
> **Where it is limited.** One task, mostly one model, single-token top-k probes. A strong negative datapoint, not proof that SAEs are useless.

Interpretability research has a measurement problem. With no ground truth for the concepts a model uses, SAE papers mostly report proxies like reconstruction loss, which say little about whether the latents help anyone do anything.

This paper proposes a blunt alternative: give SAEs a fair shot at a real task and see if they beat the tools people already use. The task is **probing**, training a small classifier on hidden activations to read off a property of the input.

Below: the SAE probe, the protocol that keeps the comparison honest, the four hard regimes, the interpretability case studies, and the multi-token "illusion" behind earlier SAE wins. If you want the background on why SAEs exist at all, my post on [superposition](/blog/machine-learning/ai-interpretability/what-is-superposition) covers it.

## Why probing is the right test

The authors give two reasons probing is a good arena.

- **Probing is practically useful.** Probes study representations, detect safety-relevant quantities, and catch synthetic sleeper agents.
- **SAEs might plausibly help.** An SAE rewrites a dense activation as a sparse mix of candidate "concept" directions. A probe restricted to the few latents that carry a concept has a strong prior, which should matter most when data is scarce or messy.

This is the prediction an SAE optimist would actually make, and the paper tests it at scale.

| Ingredient | Choice in the paper |
| --- | --- |
| Main model | Gemma-2-9B, layer 20, last token |
| Replication model | Llama-3.1-8B, layer 16 (Appendix F) |
| SAEs | JumpReLU from Gemma Scope (widths 16k, 131k, 1M); TopK from Llama Scope (width 128k, L0 55) |
| Datasets | 113 binary classification tasks, IDs ranging from 5 to 163 |
| Prompt length | 5 tokens up to a left-truncated 1024 tokens |
| Metric | Test ROC AUC; average test set size 1945, minimum 100 |

## The SAE probe: pick a few latents, fit a sparse classifier

**The problem.** The Gemma Scope SAEs have $W$ = 16k, 131k or 1M latents. A probe on all of them throws away the sparsity that is supposed to help.

**The analogy.** A librarian has filed every document under thousands of subject tags. To detect "about basketball" you would pick the few tags that differ most between basketball and other documents, then train on just those.

**The mechanism.** For each dataset:

1. Run the model on every prompt and keep the residual-stream activation at layer $l$ on the last token, $X^{l}_{-1}$, shape `(num_prompts, d_model)`.
2. Pass it through the SAE encoder to get latents $Z = \mathrm{SAE}(X^{l}_{-1})$, shape `(num_prompts, W)`.
3. Rank each latent by how differently it fires on the two classes. Keep the top $k$.
4. Fit an L1-regularized logistic regression on those $k$ columns only.

![Figure 2 from Kantamneni et al. (2025): left, the SAE probe pipeline; right, the "quiver of arrows" of candidate methods](/imgs/blogs/are-sparse-autoencoders-useful-a-case-study-in-sparse-probing-fig2.webp)

**The math.** The SAE encoder itself is standard background, not something this paper changes. For a JumpReLU SAE (the Gemma Scope family) it looks like this; I state it from the JumpReLU paper, not from this one:

$$
z = \mathrm{JumpReLU}_{\theta}\!\left(W_{\text{enc}}\, x + b_{\text{enc}}\right), \qquad \mathrm{JumpReLU}_{\theta}(u)_i = u_i \cdot \mathbf{1}[u_i \gt \theta_i]
$$

Here $x \in \mathbb{R}^{d}$ is one activation, $W_{\text{enc}} \in \mathbb{R}^{W \times d}$ and $b_{\text{enc}} \in \mathbb{R}^{W}$ are encoder weights, and $\theta \in \mathbb{R}^{W}$ holds per-latent thresholds. A latent passes its value only above threshold, so $z$ is sparse; the **L0** is the average count of nonzeros.

The selection step is the paper's Equation 1. Let $T_1$ be the training prompts with target 1, $T_0$ those with target 0, and $Z_{j,i}$ latent $i$ on prompt $j$:

$$
I = \underset{i \in \{1,\dots,W\}}{\operatorname{arg\,top}k} \; \left|\, \frac{1}{|T_1|}\sum_{j \in T_1} Z_{j,i} \;-\; \frac{1}{|T_0|}\sum_{j \in T_0} Z_{j,i} \,\right|
$$

The probe $p_{\text{SAE}}$ then maps $Z[:, I] \mapsto t$, where $t$ is the target. Concretely it minimizes a standard L1 logistic loss (my notation; the paper only names the regularizer):

$$
\min_{w, b} \; \frac{1}{n}\sum_{j=1}^{n} \ell\!\left(t_j,\; \sigma\!\left(w^\top Z_{j,I} + b\right)\right) + \lambda \lVert w \rVert_1
$$

with $\sigma$ the sigmoid, $\ell$ the cross-entropy, $w \in \mathbb{R}^{k}$ the probe weights, and $\lambda$ the regularization strength picked on validation data.

**A tiny worked example.** Say $W = 5$ and we have two prompts per class. Latent means on class 1 are `[0.0, 2.1, 0.3, 0.0, 1.5]` and on class 0 are `[0.1, 0.2, 0.3, 0.0, 1.4]`. The absolute differences are `[0.1, 1.9, 0.0, 0.0, 0.1]`. With $k = 1$ the probe uses only latent 2. With $k = 2$ it adds latent 1 or latent 5 (a tie at 0.1), and the L1 penalty can still zero out the weaker one.

**When it fails.** Mean difference is a crude filter: it happily picks latents that track a spurious cue. It assumes a concept fits in $k$ latents, and it only sees one token.

A preliminary scan (Appendix D.2) found **width barely matters, higher L0 and larger $k$ help**, so the main runs use the largest L0 at widths 16k, 131k and 1M, with $k = 16$ for readability and $k = 128$ for performance.

![SAE probe vs baseline probes: the SAE lane selects top-k latents by class mean difference and fits L1 logistic regression; the baseline lane fits five methods on the raw activation](/imgs/blogs/are-sparse-autoencoders-useful-a-case-study-in-sparse-probing-1.webp)

## The baselines are not weak

The comparison set is what makes this paper credible. Every baseline is trained on the raw activation $X^{l}_{-1}$ and gets 10 hyperparameter settings:

| Method | Hyperparameter searched |
| --- | --- |
| Logistic regression | L2 strength (L1 for SAE probes) |
| PCA + logistic regression | number of PCA components |
| K-nearest neighbors | number of neighbors |
| XGBoost | n_estimators, depth, learning rate, subsampling, regularization |
| MLP | depth, width, learning rate, weight decay |

Hyperparameters are chosen on validation AUC, and the validation scheme adapts to dataset size (Table 4 in the paper): train-set AUC for at most 3 examples, leave-two-out up to 12, 6-fold cross validation up to 128, and an 80/20 split above that. This matters most when data is scarce.

**AUC in one line.** The probability that the probe scores a random positive above a random negative: 1.0 is perfect, 0.5 is chance, and no threshold is needed.

## The quiver of arrows: select on validation, score on test

**The problem.** With dozens of SAE configurations (three widths, several L0s, two values of $k$) and 113 datasets, *some* SAE probe will beat logistic regression on *some* dataset by luck. If you pick the winner by looking at test AUC, you have leaked the test set and manufactured a result.

**The analogy.** A practitioner has a quiver of arrows, the methods they know. They try each, ship the one that looks best on validation, and the fair question becomes: does carrying the SAE arrow raise what I ship?

**The mechanism and math.** Appendix C.4 defines it. Given a method set $P = \{p_1, \dots, p_n\}$, compute each method's validation AUC, pick the argmax, and report that method's test AUC:

$$
p^{*} = \arg\max_{p_i \in P} \mathrm{AUC}^{\text{val}}_{p_i}, \qquad \mathrm{AUC}^{\text{test}}_{P} = \mathrm{AUC}^{\text{test}}_{p^{*}}
$$

Let $P$ be the baselines and $P' = P \cup \{\text{SAE probes}\}$. The quantity of interest is

$$
\Delta = \mathrm{AUC}^{\text{test}}_{P'} - \mathrm{AUC}^{\text{test}}_{P}
$$

A positive mean $\Delta$ across datasets means the SAE arrow helps; zero means it is dead weight; negative means it hurts, because sometimes it wins validation and then loses on test.

![Quiver of arrows: the fair protocol picks the best method on validation AUC and reports its test AUC; the leaky protocol picks the best SAE probe on test AUC and inflates the result](/imgs/blogs/are-sparse-autoencoders-useful-a-case-study-in-sparse-probing-2.webp)

**A tiny worked example.** On one dataset, logistic regression has validation 0.91 and test 0.90; the SAE probe has validation 0.92 and test 0.88. The baseline quiver ships logistic regression (0.90). The quiver with SAE ships the SAE probe because it won validation, so it reports 0.88. Here $\Delta = -0.02$: carrying the SAE arrow cost you two points.

**When it fails.** The protocol assumes validation data looks like test data. That breaks for label noise (the validation labels are corrupted too) and covariate shift (the test distribution moves). For those two regimes the paper drops the quiver and runs a head-to-head between logistic regression and one fixed SAE probe, which is what a sensible practitioner would deploy.

The authors also admit one small leak (Appendix C.1): $k$ and the SAE configurations were chosen from test-AUC trends on the normal datasets. If anything that favors SAEs, which makes the null result stronger.

## Standard conditions: a slight loss

The first run uses layer 20 (where baselines perform best), 1024 training points per dataset (or all available), balanced classes. SAE probes win validation and get chosen on **14 of 113** datasets, yet the mean $\Delta$ is **-0.003 ± 0.002**.

That is unsurprising: with plenty of balanced data, logistic regression already finds the right direction. The real test is the hard regimes.

## Four hard regimes

Each regime has a knob that the authors sweep. The point of the sweep is to find *any* operating point where the SAE prior pays off.

![Four hard regimes: data scarcity, class imbalance and label noise show no gain from SAE probes; covariate shift makes SAE probes worse](/imgs/blogs/are-sparse-autoencoders-useful-a-case-study-in-sparse-probing-3.webp)

### Data scarcity

**Knob.** Number of training examples $n$, 20 values log-spaced from 2 to 1024. Standard test set.

**Why SAEs should win here.** With 5 examples, a dense probe over thousands of dimensions is nearly unconstrained, while a probe over a few concept latents has far fewer ways to overfit.

**Result.** No meaningful improvement at any $n$.

### Class imbalance

**Knob.** The positive fraction $n_1/n$, 19 values linearly spaced from 0.05 to 0.95. The test set uses the same ratio.

**Result.** No improvement at any ratio.

### Label noise

**Knob.** The fraction of training labels flipped, 11 values from 0 to 0.5. Clean test set.

**Comparison.** Head-to-head: logistic regression versus the width-16k, $k = 128$ SAE probe, because the corrupted validation set would make the quiver pick at random.

**Result.** No improvement at any noise level.

![Figure 6 from Kantamneni et al. (2025): SAE minus baseline test AUC across datasets for data scarcity, class imbalance and label noise, with 95% confidence intervals](/imgs/blogs/are-sparse-autoencoders-useful-a-case-study-in-sparse-probing-fig6.webp)

The curves hug zero at every knob setting. And the SAE is not being ignored: Appendix E shows the quiver picks it on up to 40 datasets per regime. It wins validation, then fails to convert that into test AUC.

### Covariate shift

**Knob.** The test prompts change while the training prompts do not. The authors use 8 out-of-distribution datasets, each tested on 300 shifted examples:

- two GLUE-X datasets, "extreme" versions of task 87 (grammaticality) and 90 (entailment);
- three datasets (66, 67, 73) translated into another language;
- three datasets (5, 6, 7) with altered names or cartoon characters in place of historical figures.

**Comparison.** Head-to-head against a single SAE probe from the width-131k, L0 = 114 SAE, chosen because Neuronpedia has latent descriptions for it.

**Result.** The one regime with a clear direction points the wrong way: baselines generalize better.

![Figure 7 from Kantamneni et al. (2025): out-of-distribution test AUC for baseline logistic regression versus SAE probes across the eight shifted datasets and their average](/imgs/blogs/are-sparse-autoencoders-useful-a-case-study-in-sparse-probing-fig7.webp)

## Why do SAE probes break under shift?

Here the SAE's readable basis earns its keep. The authors test two hypotheses.

1. **Spurious latents.** The probe picked latents that correlate with the label in training but for the wrong reason.
2. **Brittle latents.** The latents are on-topic but do not fire on the shifted inputs.

**Testing hypothesis 1 by pruning.** For three tasks where SAE probes did badly OOD (7 hist_fig_ispolitician, 66 living-room, 90 glue_qnli), they take the top $k = 8$ latents, label each one with autointerp (Claude-3.5-Sonnet writes a description from the latent's top activations), then ask o1 to rank each description's relevance to the task and to push down latents that look spurious given the shift. Then they build probes from the top $k$ ranked latents for $k$ from 1 to 8.

| Task | Effect of pruning from $k$=8 to $k$=1 |
| --- | --- |
| 66 living-room | OOD AUC up 0.024 |
| 90 glue_qnli | OOD AUC up 0.052 |
| 7 hist_fig_ispolitician | A 0.077 OOD AUC change, which the authors read as "this task needs all 8 latents" (the text says "increases", which looks like a typo for a drop, given that conclusion) |

Pruning helps a little, but the in-distribution to OOD gap is far larger, so spurious latents are not the main story.

**Testing hypothesis 2 on one latent.** On the living-room task, latent 122774 has an in-distribution test AUC of 0.99 and an OOD AUC of 0.64. The shift here is translation into French. The authors translate "living room" into 15 languages with GPT-4o and find that the latent fires most on English, less on other languages, and **not at all on French**. The latent is a real "living room" detector. It is just an *English* "living room" detector, so a probe built on it cannot generalize to French.

My extrapolation, not the paper's claim: an SAE can carve concepts along surface features like language, which is exactly the axis a covariate shift moves, while a dense probe can mix directions into something more language-invariant.

## Interpretability wins that baselines also get

If SAE probes do not raise accuracy, perhaps their value is that you can *read* them. The authors test three uses: explaining probes (above), cataloguing what individual top latents mean, and catching problems in the datasets themselves.

**Cataloguing top latents.** For each dataset they evaluate each of the top 128 latents of the 131k SAE as a $k = 1$ probe and autointerp the best one with GPT-4o. The catalogue is a useful sanity check on autointerp itself:

| Dataset | Latent | What happened |
| --- | --- | --- |
| 5 hist_fig_ismale | 81210 | Description fits: "references to female individuals" |
| 125 wold_country_Italy | 50817 | 0.989 AUC, clearly an Italy latent, but autointerp labels it "names of researchers and scientific authors" |
| 22 headline_isobama | 10555 | 0.782 AUC on Obama-era headlines from a latent described as "strings of numbers, often with mathematical notations": likely spurious |

### Case study: CoLA is partly mislabeled

CoLA labels English sentences as grammatical or not. Latent 369585 (1M-width SAE) scores 0.76 test AUC and fires on ungrammatical text. Where it disagrees with a "grammatical" label, the sentence often really is ungrammatical.

To check, the authors ask three LLM judges (GPT-4o, Claude-3.5-Sonnet-New, Llama-3.1-405B Instruct) to rate 1000 random CoLA sentences and take the majority vote as a cleaned label. Roughly **25%** of CoLA labels disagree with the ensemble, against the 13% human disagreement rate the original CoLA authors reported.

Then they train three classifiers on the original labels and test them on the cleaned ones:

![Figure 9 from Kantamneni et al. (2025): accuracy on clean labels and on the mislabeled CoLA subset for the original labels, a logistic-regression baseline, a k=128 SAE probe, and the single latent 369585](/imgs/blogs/are-sparse-autoencoders-useful-a-case-study-in-sparse-probing-fig9.webp)

On the full clean set all four (the original labels, the baseline, the dense SAE probe, the single latent) land at roughly the same accuracy. On the mislabeled subset the single latent beats the $k = 128$ SAE probe, which in turn beats the logistic-regression baseline by a wide margin. So the sparsest probe learned the *concept* rather than the annotators' mistakes.

The twist: a plain logistic-regression probe surfaces the same mislabeled sentences (their Table 7). The SAE got there first, but was not required.

### Case study: AI text ends in a period

Dataset 110 separates human from GPT-3.5 text. Latent 105150 reaches 0.82 AUC while firing mostly on punctuation. The cue is spurious: AI text tends to end in a period, human text in a space.

Again, the baseline gets there too. Running the logistic-regression probe over 2.8 million Pile tokens shows it is most active on punctuation tokens (their Table 8).

The authors grant SAEs one edge: the tooling (Neuronpedia, autointerp) already exists, and an SAE probe splits into several labeled latents where a dense probe gives one direction.

## The illusion: pooling only one side

Bricken et al. (2024) reported that SAE probes slightly beat baselines. Why does this paper not replicate that? The authors offer two explanations.

The first is dataset luck: the earlier result used one dataset, and SAE probes win on only 2.2% of datasets here.

The second is more interesting. The earlier work max-pooled each SAE latent across all tokens, and max-pooled the baseline's raw dimensions too. But raw activations probably lack privileged dimensions, so that handicapped the baseline.

To test this on 60 random datasets with $k = 128$, the authors build a stronger multi-token baseline, an attention-pooled probe (their Equation 2):

$$
\mathrm{probe}(X) = \mathrm{softmax}_{t \in [1, \mathrm{CtxLen}]}\!\left\{ X^{l}_{t} \cdot q \right\} \cdot X^{l} \cdot v, \qquad q, v \in \mathbb{R}^{d}
$$

Here $X^{l}_{t}$ is the activation of token $t$ at layer $l$, $q$ is a learned query that scores how much each token matters, and $v$ is a learned readout direction. The softmax turns token scores into weights, the weighted sum pools the sequence into one $d$-dimensional vector, and $v$ maps it to a logit.

**Tiny example.** Three tokens with scores $X_t \cdot q = [0, 0, 2]$ get softmax weights of about $[0.11, 0.11, 0.79]$. If their readouts $X_t \cdot v$ are $[-1, -1, 3]$, the pooled logit is about ${0.11} \cdot (-1) + {0.11} \cdot (-1) + {0.79} \cdot 3 \approx 2.1$. The probe learned to attend to the one token that carries the signal.

![The pooling illusion: SAE win rate is 2.2% when both sides use the last token, 19.6% when only the SAE is max-pooled, and 8.7% when both sides can choose a pooled or last-token probe](/imgs/blogs/are-sparse-autoencoders-useful-a-case-study-in-sparse-probing-4.webp)

The win rates (SAE beats baseline by more than 0.005 AUC) tell the story:

| Setup | SAE win rate |
| --- | --- |
| Both last-token | 2.2% |
| SAE max-pooled vs baseline last-token (the illusion) | 19.6% |
| Both choose pooled or last-token via the quiver, baseline gets attention pooling | 8.7% |

Most of the apparent gain came from giving only the SAE the extra tokens; a fair pooled baseline cuts the win rate by more than half. Appendix J also tests binarizing the pooled latents, a trick from Gallifant et al. (2025), and finds it makes SAE probes worse, not better.

## Do newer SAE architectures help?

The authors test eight architectures released from October 2023 to December 2024 (ReLU original and updated, Gated, TopK, JumpReLU, BatchTopK, p-annealing, Matryoshka), using width-16k Gemma-2-2B layer-12 SAEs from Karvonen et al. Taking each one's best L0, the $k = 16$ probe shows a slight upward trend with newer architectures, but it is **not statistically significant** and the spread is larger than the trend. If you want the background on one of those architectures, see my post on [scaling and evaluating sparse autoencoders](/blog/paper-reading/ai-interpretability/scaling-and-evaluating-sparse-autoencoders), the TopK paper.

## Results at a glance

| Setting | Model / SAE | Compared against | Outcome |
| --- | --- | --- | --- |
| Standard, 113 datasets | Gemma-2-9B L20, Gemma Scope | 5-method quiver | Mean $\Delta$ = -0.003 ± 0.002; SAE chosen 14/113 |
| Data scarcity, $n$ from 2 to 1024 | same | quiver | No gain at any $n$ |
| Class imbalance, ratio 0.05 to 0.95 | same | quiver | No gain |
| Label noise, 0 to 50% flipped | 16k, $k$=128 | logistic regression | No gain |
| Covariate shift, 8 datasets | 131k, L0 = 114 | logistic regression | SAE worse on average |
| Multi-token, 60 datasets | $k$=128 | last-token, attention-pooled | SAE win rate 2.2% / 19.6% (illusion) / 8.7% |
| Architectures, 8 SAE types | Gemma-2-2B L12, 16k | baseline | Slight, non-significant uptrend |
| Replication | Llama-3.1-8B L16, Llama Scope 128k L0 55 | quiver | Same null result, a few positive outliers |

**What is load-bearing and might not transfer.** Last-token probing (long-prompt tasks might favor pooled SAE probes more); mean-difference selection with small $k$ (Gallifant et al. probe on the full SAE vector and report wins, against single-token baselines only); and open Gemma Scope and Llama Scope SAEs rather than larger or task-adapted ones.

## Critique

**What is strong.**

- **The protocol.** The quiver of arrows is the lasting contribution: it asks the practitioner's question and removes test-set selection bias.
- **Breadth.** 113 datasets and four swept regimes make a cherry-picked null result very unlikely.
- **Honest debugging.** Several sections narrate an SAE finding that a better baseline later matched.

**What is weaker.**

- **One task.** Probing is still a proxy, as the authors note; steering or unlearning could differ.
- **The SAE probe is not fully optimized.** A more tuned SAE probe could do better, though the authors reasonably reply that baselines deserve equal tuning effort.
- **OOD analysis is thin.** Eight datasets and one SAE configuration give a direction, not a precise effect size.
- **LLM-judged labels.** The 25% CoLA mislabel estimate rests on three LLM judges. The direction is convincing given Table 7; the exact rate is softer than it reads.

**Missing ablation.** Probes on toy models with known true features, which the authors name as future work. If even true features beat linear probing only slightly, the null result says more about probing than about SAEs.

**What would change my mind.** A replication with the same quiver-of-arrows protocol where SAE probes win consistently on a regime with real practical value (say, very few examples of a rare safety-relevant behavior, with pooled baselines included), or where SAE probes generalize *better* than logistic regression under translation or paraphrase shift. Either would show the concept basis buys something a dense linear direction cannot.

The broader lesson, and the one I would carry to every interpretability paper, is the authors': **put as much effort into your baselines as into your method.** Feature absorption, covered in my post on [splitting and absorption in SAEs](/blog/paper-reading/ai-interpretability/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders), is one mechanistic reason a single latent may fail to represent a whole concept, which fits the hist_fig_ispolitician result above.

## References

1. Kantamneni, S., Engels, J., Rajamanoharan, S., Tegmark, M., Nanda, N. "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing." ICML 2025. [arXiv:2502.16681](https://arxiv.org/abs/2502.16681)
2. Lieberum, T. et al. "Gemma Scope: Open Sparse Autoencoders Everywhere All at Once on Gemma 2." 2024.
3. Rajamanoharan, S. et al. "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders." 2024.
4. Gallifant, J. et al. "Sparse Autoencoder Features for Classifications and Transferability." 2025.
5. Related posts on this site: [What is superposition](/blog/machine-learning/ai-interpretability/what-is-superposition), [Scaling and evaluating sparse autoencoders](/blog/paper-reading/ai-interpretability/scaling-and-evaluating-sparse-autoencoders), [A is for Absorption](/blog/paper-reading/ai-interpretability/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders).

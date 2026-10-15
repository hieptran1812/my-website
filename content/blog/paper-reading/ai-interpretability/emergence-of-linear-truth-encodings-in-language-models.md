---
title: "Emergence of Linear Truth Encodings in Language Models: How a Truth Direction Grows Out of Memory and Layer Norm"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "Why language models end up with a linear direction that separates true from false statements: the truth co-occurrence hypothesis, a one-layer toy transformer where the direction provably appears, the two-phase training dynamic, and what carries over to Llama-3-8B and Pythia."
tags: ["paper-reading", "truth-direction", "linear-probing", "truth-co-occurrence-hypothesis", "layer-normalization", "associative-memory", "training-dynamics", "activation-steering", "mechanistic-interpretability", "neurips-2025"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 19
image: /imgs/blogs/emergence-of-linear-truth-encodings-in-language-models-20251105205044.png
paper:
  title: "Emergence of Linear Truth Encodings in Language Models"
  authors: "Shauli Ravfogel, Gilad Yehudai, Tal Linzen, Joan Bruna, Alberto Bietti"
  venue: "NeurIPS 2025"
  url: "https://arxiv.org/abs/2510.15804"
---

> [!tldr]
> **The claim.** Language models carry a linear "truth direction" because false statements cluster together in real text. If one claim on a page is false, the next one is more likely false too, so a model that infers a hidden truth bit predicts the next token better. The authors call this the **Truth Co-occurrence Hypothesis (TCH)**.
> **The mechanism.** In a one-layer transformer, the model first memorizes subject to attribute facts. A true pair then *cancels itself* inside the attention average, leaving a short vector; a false pair leaves a long one. Layer norm turns that norm gap into a linearly separable signal and into sharper or flatter predictions.
> **Most surprising result.** Training is two-phase. Memorization finishes within about 1,000 batches; the truth direction appears abruptly around 7,500 batches, long after the facts are known. Without layer norm, the theory proves no linear separator exists at all.
> **Where it fails.** The proofs live in a toy with one relation, uniform attention and one-hot embeddings. In Pythia-6.9B the authors find no evidence that layer norm itself creates the separation; the truth signal builds up gradually across many layers instead.

Probing papers keep finding the same thing. Train a logistic regression on a model's hidden states, label the inputs "true" or "false", and the classifier works, often across domains. "2 + 2 = 4" and "The capital of France is Rome" land on opposite sides of one plane.

Two questions remained open. **Why** would next-token training produce a truth feature, and **how** is it computed? This paper answers both in a setting small enough to prove things about. For why "linear" is the interesting word, see the [linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis) post.

## The question: why would truth be linear at all?

Nothing in the next-token loss mentions truth, so a truth feature has to *pay rent*: it must lower the loss somewhere. The paper's answer, in four parts:

| Question | The paper's answer | Where it lives |
| --- | --- | --- |
| Why does a truth feature help? | False claims co-occur, so knowing "this context is lying" improves the next prediction (the TCH) | Section 3, Appendix B |
| How is it computed? | Compare the memorized attribute with the observed one; a match cancels, a mismatch does not | Section 4, Theorems 1 and 2 |
| When does it appear in training? | After memorization, in a second, slower phase | Theorem 3, Section 5 |
| Does it exist in real LMs? | Behavioral evidence in Llama-3-8B, gradual emergence in Pythia-6.9B | Section 5.3, Appendix E.4 |

## The Truth Co-occurrence Hypothesis

**The problem.** Earlier "persona" explanations lean on lexical cues: Wikipedia sounds different from a forum post. The authors want a reason that survives when true and false text look identical.

**The analogy.** A witness who lies once is more likely to lie again, so a juror discounts the next claim. The juror's "unreliable witness" note is the latent truth bit.

**The evidence.** The authors measure this on MAVEN-FACT, a news corpus where annotators label each event mention with a factuality tag. Keeping only certain judgments, they report:

- The overall certain-false rate is p = 0.0209.
- Two mentions from the same article are both certain-false with probability 0.0009. Under independence that would be 0.00044, so false claims co-occur about **2 times** more often than chance.
- The ratio of observed to binomial variance across articles is 1.23, which means 23% extra article-to-article heterogeneity.
- A chi-squared test of independence gives a statistic of 4.17 × 10³ with p ≈ 9 × 10⁻⁴⁹.

Falsehoods cluster by document, so tracking a hidden truth bit pays.

### The data generator that isolates the effect

Real text confounds truth with style and frequency, so the paper builds a toy world: each subject has one correct attribute, a controlled fraction of examples is corrupted uniformly, and the two facts in a sequence share their truthfulness.

![The toy data generator: one hidden bit T decides whether both attributes are correct or both random](/imgs/blogs/emergence-of-linear-truth-encodings-in-language-models-1.webp)

Each training example is a four-token sequence $x\,y\,x'\,y'$. Here $x, x' \in S$ are subjects drawn from a set $S$ (say, "France capital"), $y, y' \in A$ are attributes from a set $A$ (say, "Paris"), and $g : S \to A$ is the ground-truth map. One hidden bit $T \sim \text{Bernoulli}(\rho)$ is drawn per sequence, where $\rho$ is the probability that the sequence is true:

$$
T = 1:\; y = g(x),\; y' = g(x') \qquad\qquad T = 0:\; y, y' \overset{\text{i.i.d.}}{\sim} \text{Unif}(A)
$$

The key point: both attributes share the same $T$. If the model can tell from $(x, y)$ whether the first fact is true, it knows whether to trust its memory for $y'$.

### How much loss does the truth bit save?

A model that ignores $T$ must hedge. Using the prior $\rho$ for every prefix, its best prediction for the second attribute is:

$$
\Pr\!\left[y' = g(x') \mid x, y, x'\right] = \rho + \frac{1-\rho}{|A|}, \qquad \Pr\!\left[y' = a \mid x, y, x'\right] = \frac{1-\rho}{|A|} \;\text{ for } a \neq g(x')
$$

A model that *knows* $T$ instead pays zero loss when $T = 1$ and $\log |A|$ when $T = 0$, for an average of $(1-\rho)\log|A|$.

Appendix B subtracts the two cross-entropies, $L_{\neg T}$ (no access to the bit) and $L_T$ (with it), and takes the large-vocabulary limit:

$$
L_{\neg T} - L_T \;\xrightarrow{\;|A| \to \infty\;}\; H_2(\rho) = -\rho \log \rho - (1-\rho)\log(1-\rho)
$$

$H_2$ is the binary entropy. The saving per second-attribute token is exactly one bit's worth of uncertainty about $T$.

**Worked example.** The numbers below are my own arithmetic from the formula, in nats:

| $\rho$ (true rate) | $H_2(\rho)$ in nats | In bits |
| --- | --- | --- |
| 0.5 | 0.693 | 1.000 |
| 0.8 (toy model in Section 4) | 0.500 | 0.722 |
| 0.99 (synthetic experiments) | 0.056 | 0.081 |

**When it fails.** The incentive shrinks fast as $\rho \to 1$. At the paper's default of 0.99 the payoff is only 0.056 nats per token, which already hints at why the truth direction is slow to appear. At exactly $\rho = 1$ the incentive is zero, and Section 5 shows what happens then.

## The toy model: one attention layer and a layer norm

**The problem.** Probes show a separating plane exists, not which weights produce it. So the authors shrink the transformer until every weight can be read.

**The setup.** Section 4 uses one-hot vectors in $\mathbb{R}^d$ with $d = 4N + 3$. Input tokens $z \in [2N]$ (subjects in $[N]$, attributes in $[N+1, 2N]$) get embeddings $e_z$, the three positions get $p_t$, and every output token gets an unembedding $u_z$. Each lives in its own block of coordinates, so the weights can be read off directly.

The model is one attention layer with uniform causal attention and a normalization $\mathcal{N}(v) = v / \lVert v \rVert$:

$$
F_W(z_{1:t})_t = U \cdot \mathcal{N}\!\left( e_{z_t} + p_t + \frac{1}{t}\sum_{s=1}^{t} W\,(e_{z_s} + p_s) \right)
$$

Reading it term by term:

- $e_{z_t} + p_t$ is the residual stream at position $t$ (the current token plus its position).
- $\frac{1}{t}\sum_{s} W(e_{z_s} + p_s)$ is uniform attention: average the earlier tokens, then apply the value matrix $W \in \mathbb{R}^{d \times d}$, the only learned parameter.
- $\mathcal{N}$ divides by the norm, a stripped-down RMS norm.
- $U \in \mathbb{R}^{2N \times d}$ projects onto the unembedding coordinates to give logits, which go through a softmax with inverse temperature $\beta = \sqrt{d}$.

### What the value matrix learns

The authors train this with $N = 20$, $\rho = 0.8$, batch size 16 and SGD at learning rate 1.0, starting $W$ from zero. Then they plot $W$.

![Figure 1 from Ravfogel et al. (2025): the value matrix W of the one-layer model at training steps 20, 100 and 500, showing blocks that appear in sequence](/imgs/blogs/emergence-of-linear-truth-encodings-in-language-models-fig1.webp)

Blocks appear, in order:

1. First, $e_x \to u_{g(x)}$: subject embeddings map to their correct attribute's unembedding. This is the factual memory.
2. Along with it, the positional blocks learn whether the next token should be a subject or an attribute.
3. Later, $e_x \to -e_x$ and $e_y \to -u_y$: negative identity blocks.
4. Last, $e_y \to e_{g^{-1}(y)}$: an attribute maps back to the subject it belongs to.

The proofs assume an idealized version:

$$
W e_x = -\alpha_1 e_x + \beta_1 u_{g(x)}, \qquad W e_y = \alpha_2 e_{g^{-1}(y)} - \beta_2 u_y
$$

with all of $\alpha_1, \alpha_2, \beta_1, \beta_2 > 0$. A subject writes "my attribute is $g(x)$" and erases itself; an attribute writes "my subject is $g^{-1}(y)$" and erases itself. Positional columns get coefficients $\gamma_1, \gamma_2, \gamma_3$.

## The cancellation mechanism

**The analogy.** Two halves of a torn banknote: if they match, the tear lines fit flat; if not, edges stick out. The model does not check "is this true?"; it checks whether two pieces of evidence fit.

**The mechanism.** Any token that attends to both $x$ and $y$ (that is $y$ itself, or $x'$) receives the sum of their value vectors:

$$
\zeta(x, y) := W(e_x + e_y) = -\alpha_1 e_x + \alpha_2 e_{g^{-1}(y)} + \beta_1 u_{g(x)} - \beta_2 u_y
$$

Now look at what happens when the pair is true, so $y = g(x)$ and therefore $g^{-1}(y) = x$:

- $-\alpha_1 e_x$ and $+\alpha_2 e_{g^{-1}(y)} = +\alpha_2 e_x$ land on the **same coordinate** with opposite signs.
- $+\beta_1 u_{g(x)}$ and $-\beta_2 u_y = -\beta_2 u_{g(x)}$ also land on the same coordinate with opposite signs.

When the pair is false, all four terms sit on four different coordinates and nothing cancels. The paper turns this into one identity:

$$
\lVert \zeta(x, g(x)) \rVert^2 = \lVert \zeta(x, y) \rVert^2 - 2\alpha_1\alpha_2 - 2\beta_1\beta_2, \qquad y \neq g(x)
$$

Since every coefficient is positive, a true pair always produces a **shorter** vector than a false pair.

![Why layer norm turns a norm difference into a confidence difference](/imgs/blogs/emergence-of-linear-truth-encodings-in-language-models-2.webp)

**Worked example.** Set every coefficient to 1. For a true pair, $\zeta = -e_x + e_x + u_{g(x)} - u_{g(x)} = 0$, so its squared norm is 0. For a false pair, the four unit spikes are orthogonal, so the squared norm is 4. The identity checks: 4 − 2 − 2 = 0.

Figure 2 shows this on a trained model: a false sequence has large spikes at $x$ and $g^{-1}(y)$ in the input-embedding dimensions; a true one does not.

![Figure 2 from Ravfogel et al. (2025): representations on a true sequence (top) and a false sequence (bottom), before layer norm, after layer norm, and as predicted probabilities](/imgs/blogs/emergence-of-linear-truth-encodings-in-language-models-fig2.webp)

### Layer norm as a confidence knob

Dividing by the norm rescales the whole vector, including the part pointing at the next answer. A short vector gets scaled **up**, so its logits grow and the softmax sharpens. A long vector gets scaled **down**, so the softmax flattens. Layer norm acts as an input-dependent temperature.

Theorem 1 makes this precise for the prediction of $y'$ at position $x'$. On a true prefix, the logit for the correct answer $g(x')$ beats every other logit by a positive margin that depends on $\alpha$, $\beta$ and $\gamma$. On a false prefix, the gap is **exactly zero**:

$$
F(x, y, x')_{g(x')} - \max_{k \neq g(x')} F(x, y, x')_k = 0 \qquad \text{for } y \neq g(x)
$$

The zero comes from a tie: on a false prefix, the logits for $g(x')$ and $g(x)$ are equal, the two low spikes in Figure 2's bottom-right panel. This matches earlier work on "confidence neurons" that regulate entropy through the norm.

**When it fails.** The mechanism needs the attribute's back-pointer $e_y \to e_{g^{-1}(y)}$ to be learned. That block is the last one to appear in Figure 1, so any training run that stops early has memory but no truth signal.

## Why the truth direction is linear (and why it needs layer norm)

Theorem 2 is the result in the paper's title. It has two halves, and the first is the surprising one.

**Without layer norm, no linear separator exists.** The argument is a neat midpoint trick. Take two true pairs $(x_i, y_i)$ and $(x_j, y_j)$, and the two crossed false pairs $(x_i, y_j)$ and $(x_j, y_i)$. Before normalization, the representation is a sum of a part that depends on $x$ and a part that depends on $y$. So the two false vectors add up to exactly the same total as the two true vectors.

A linear classifier that scores both true points positive gives their sum a positive score, and that sum equals the sum of the two false points. So at least one false point also scores positive, whatever the weights.

**With layer norm, a separator exists.** After normalization, the probe can read the norm. The proof uses the positional embedding $p_2$ as the probe direction $w$. Its coefficient in the raw vector is 1 for every sample, so after dividing by the norm:

$$
\langle w, \mathcal{N}(v) \rangle = \frac{1}{\lVert v \rVert}
$$

The probe score is just the inverse norm. Since true vectors are shorter, they score higher. The squared norms at the $y$ token differ by

$$
\lVert v_F \rVert^2 - \lVert v_T \rVert^2 = 2\alpha_1\alpha_2 + 2\beta_1\beta_2
$$

so as long as that is non-zero, a threshold separates the classes.

**Worked example.** With $\alpha_1 = \alpha_2 = \alpha$, $\beta_1 = \beta_2 = \beta$ and $\gamma_1 = \gamma_2$, the proof gives $\lVert v_T \rVert^2 = 2$ and $\lVert v_F \rVert^2 = 2 + 2\alpha^2 + 2\beta^2$. Take $\alpha = \beta = 1$ (my choice):

| Class | Squared norm | Probe score ${1}/\lVert v \rVert$ |
| --- | --- | --- |
| True | 2 | 0.707 |
| False | 6 | 0.408 |

The threshold sits at the midpoint, 0.558, and the margin is $\delta = \frac{1}{2\sqrt{2}}\left(1 - \frac{1}{\sqrt{1 + \alpha^2 + \beta^2}}\right) \approx 0.149$, matching the theorem's bound. The paper proves the same separation at the $x'$ token, with the gap shrunk by a factor that comes from averaging over three positions.

My interpretation, not the authors' claim: the toy's "truth direction" is a **norm detector** read through a constant direction. A probe would report "truth" while really measuring how much the context cancelled.

**When it fails.** If $\alpha_1\alpha_2 + \beta_1\beta_2 = 0$ (memory but no cancelling blocks), there is no separation. And a truth signal computed by MLPs over many layers would need a different proof.

## The training dynamics: memorize first, then encode truth

**The problem.** A good solution existing does not mean gradient descent finds it; normalization makes the loss non-convex.

**The result.** Theorem 3 (stated informally in the main text, proved in Appendix E.1) studies a simplified model with no positional embeddings. Two gradient steps on the first-position loss followed by one step on the third-position loss, all with step size of order $N$, produce the block structure of $W$ up to entry-wise $O(1/N)$ terms.

Layer norm creates every useful block except the plain memory block $e_x \to u_{g(x)}$. And the structure appears **even when** $\rho = 1$: the toy model learns a truth direction from purely true data, then generalizes it to false sequences it never saw.

### The synthetic experiments: all parameters trained

Section 5 drops the idealizations. The model now learns dense embeddings and the attention module, with one attention head per layer, RMSNorm and no feed-forward block. The defaults are $|A| = |S| = 512$, $d_{\text{model}} = 256$, one layer, and $\rho = 0.99$. Training runs 50,000 batches of 128 with Adam, up to half an hour on four GTX 1080 GPUs. Logistic-regression probes are fit on a balanced set at $y$ and at $x'$, averaged over 5 seeds.

![Figure 3 from Ravfogel et al. (2025): (a) probe AUC for truth at the second subject over training, and (b) the probability the model assigns to the memorized attribute on false sequences](/imgs/blogs/emergence-of-linear-truth-encodings-in-language-models-fig3.webp)

Figure 3 shows two phases:

1. **Memorization.** Within the first 1,000 batches the model puts probability near 1 on $g(x')$ for both true and false sequences. It predicts the correct attribute on over 99% of true sequences.
2. **Truth encoding.** Around 7,500 batches, after roughly 1 million examples, the probe AUC jumps abruptly. At about the same moment, the probability on $g(x')$ for false sequences falls. The model has learned to distrust its own memory when the context is lying.

![Two phases: memorize first, learn the truth direction much later](/imgs/blogs/emergence-of-linear-truth-encodings-in-language-models-3.webp)

The ordering is what the mechanism predicts: cancellation needs the memory blocks before it can compare anything.

### Inside the learned circuit

![Figure 4 from Ravfogel et al. (2025): (a) PCA of layer-1 representations at x' before and after normalization, (b) the near-uniform attention pattern of the 1-layer model, (c) PCA of input embeddings of subjects and attributes](/imgs/blogs/emergence-of-linear-truth-encodings-in-language-models-fig4.webp)

Three observations match the theory:

- **Attention is nearly uniform** (panel b). The head computes something close to the mean of $x$ and $y$, as the toy assumed.
- **Embeddings are learned to cancel** (panel c). On the first principal component, $e_x \approx -e_{g(x)}$. The model built the cancellation into the embeddings rather than the value matrix.
- **Separation appears only after normalization** (panel a). Before RMSNorm the probe is at majority level. After it, the true class sits around the origin with larger variance and the two classes split.

### What changes the timing

From Appendix D:

| Change | Effect on the truth direction |
| --- | --- |
| Higher $\rho$ (0.5 up to 0.999) | Emergence is delayed but still happens, even at 0.999 |
| $\rho = 1.0$ in the fully trained model | No emergence, unlike the toy model |
| $d_{\text{model}} = 32$ | Final AUC at $x'$ about 0.5, no separation |
| $d_{\text{model}}$ of 64 to 256, $\lvert S \rvert$ of 512 to 4096 | Final AUC at $x'$ between 0.75 and 0.96 |
| More layers (3) | Sometimes truth is encoded at $y$ and copied to $x'$; sometimes $x'$ attends directly |

Why $\rho = 1$ fails: with learnable attention and no false data, the model attends only to the current token, which discards the evidence it would need to notice a false context.

## Testing the hypothesis in real models

Real LMs have MLPs, many heads and messy data, so the authors test the hypothesis's *predictions*, not the circuit.

### Natural-language facts in a small transformer

They take CounterFact (simple facts with relations like "speaks language"), build false versions by swapping in another attribute from the same relation, and concatenate pairs with the same truth label.

A small transformer with RMSNorm, 2 attention heads and one MLP per layer ($d = 256$, depth 2, 5 or 9, $\rho = 0.99$) is trained on one relation at a time. The dynamics mirror the synthetic case: fast memorization, then linear separation, then lower confidence on false sequences. The 1-layer model also shows epoch-wise double descent: probe accuracy rises, dips, and rises again.

### Llama-3-8B: false context lowers the right answer

For a pretrained model, the authors use Llama-3-8B and the CounterFact "speaks language" relation. They ask the model to complete the last word of a sentence when it is preceded by $n$ false sentences or by $n$ true sentences, over 128 tuples.

![Figure 6 from Ravfogel et al. (2025): negative log-probability of the correct attribute for Llama-3-8B under false (F) and true (T) preceding context, before (light) and after (dark) adding the truth steering vector at layer 11](/imgs/blogs/emergence-of-linear-truth-encodings-in-language-models-fig6.webp)

Two results:

- **Sensitivity to false context.** Preceding a fact with two false sentences (FF) instead of two true ones (TT) raises the negative log-likelihood of the correct answer by 1.52. That is a factor of $e^{1.52} \approx 4.55$ less probability on the right answer.
- **Steering along the truth direction helps.** A linear probe separates true from counterfactual facts with over 95% accuracy on all middle and last layers. The authors compute the class means $\mu_T$ and $\mu_F$ at layer 11, where probe accuracy peaked, and add $\alpha(\mu_T - \mu_F)$ with $\alpha = 3.0$ to every representation. The correct attribute becomes more likely, even after false context.

For how such directions drift over a long context, see [how linear representations can change over a conversation](/blog/paper-reading/ai-interpretability/linear-representations-in-language-models-can-change-dramatically-over-a-conversation).

### Pythia-6.9B: two phases, but blurred

Appendix E.4 checks the training checkpoints of Pythia-6.9B. Each input chains four CounterFact statements whose context is all true or all false. For each checkpoint they record memorization (top-1 accuracy on the last fact), the entropy gap $\Delta H$ between false and true context, and probe AUC.

| Step | $\Delta H$ | Memorization | Probe AUC |
| --- | --- | --- | --- |
| 0 | 0.001 | 0.000 | 0.383 |
| 1,000 | 0.005 | 0.006 | 0.467 |
| 3,000 | 0.219 | 0.242 | 0.587 |
| 10,000 | 0.286 | 0.547 | 0.667 |
| 40,000 | 0.329 | 0.727 | 0.759 |
| 80,000 | 0.419 | 0.822 | 0.799 |
| 120,000 | 0.536 | 0.842 | 0.835 |
| 143,000 | 0.518 | 0.875 | 0.831 |

*Selected rows from Table 1 of the paper.*

Memorization jumps first, while the entropy gap keeps widening after 80,000 steps. But probe AUC and entropy gap also rise **before** memorization stabilizes, which the authors attribute to new facts arriving throughout pretraining. And they find no evidence that layer norm creates the separation in Pythia; the signal builds gradually with depth.

## What is load-bearing, and what might not transfer

The toy results depend on a few choices that real training does not share:

- **One relation.** Real text mixes many, so a subject must live in several key-value slots.
- **Uniform corruption.** Real falsehoods are plausible (a wrong capital is another city), not random.
- **Perfect correlation.** Synthetic pairs are always both true or both false; MAVEN-FACT shows only about 2 times chance.
- **Normalization as the readout.** Exactly the part Pythia did not show.

## Critique

**What is strong.** A *reason* for truth encoding that needs no style cues, backed by corpus statistics. A concrete, checkable mechanism, and a clean negative result ("no separator without normalization"). The predicted ordering is confirmed in three settings.

**What is weak.** The bridge from toy to real model is thin. The Llama experiment shows sensitivity to false context and that steering helps, but persona or style accounts would predict both. The Pythia evidence for two phases is suggestive, yet the authors' own note that the separation does not come from layer norm means the specific mechanism is unconfirmed at scale. The toy model also learns truth at $\rho = 1$ while the fully trained model does not, and that gap is left open.

**Missing ablations.** I would want to see the Llama sensitivity test with false contexts that are stylistically matched to true ones, which would separate TCH from a persona effect. A sweep over the strength of the truth correlation in the synthetic data, rather than only perfect correlation, would also show how much co-occurrence is enough. Both are my suggestions, not experiments in the paper.

**What would change my mind.** If a pretrained model's truth probe worked equally well on facts drawn from documents with no truth clustering (for example, shuffled single-fact contexts), the TCH would not be the main driver. Conversely, if ablating the norm-dependent component of a real model's truth direction removed its behavioral effect, I would take the cancellation story far more seriously as the mechanism at scale.

## References

1. Ravfogel, S., Yehudai, G., Linzen, T., Bruna, J., Bietti, A. (2025). *Emergence of Linear Truth Encodings in Language Models*. NeurIPS 2025. [arXiv:2510.15804](https://arxiv.org/abs/2510.15804)
2. Code release: [github.com/shauli-ravfogel/truth-encoding-neurips](https://github.com/shauli-ravfogel/truth-encoding-neurips)
3. Related on this site: [The linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis)
4. Related on this site: [Linear representations in language models can change dramatically over a conversation](/blog/paper-reading/ai-interpretability/linear-representations-in-language-models-can-change-dramatically-over-a-conversation)
5. Related on this site: [Do I Know This Entity? Knowledge awareness and hallucinations](/blog/paper-reading/ai-interpretability/do-i-know-this-entity-knowledge-awareness-and-hallucinations-in-language-models)

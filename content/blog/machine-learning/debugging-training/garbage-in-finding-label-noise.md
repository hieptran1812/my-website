---
title: "Garbage In: Finding the Label Noise That Caps Your Accuracy"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Mislabeled examples are the silent ceiling on your model — learn to measure how much noise your data carries, find the worst offenders with cleanlab and loss-ranking, and decide whether to clean, relabel, drop, or switch to a robust loss."
tags:
  [
    "debugging",
    "model-training",
    "data-quality",
    "label-noise",
    "cleanlab",
    "confident-learning",
    "finetuning",
    "deep-learning",
    "computer-vision",
    "nlp",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/garbage-in-finding-label-noise-1.png"
---

A team I worked with spent six weeks trying to push a ten-class image classifier past 88% test accuracy. They tried a bigger backbone, longer schedules, more augmentation, a learning-rate sweep, label smoothing, a second round of hyperparameter search. Nothing moved the number more than half a point. The loss curve looked healthy — train loss fell smoothly, validation loss tracked it with a sensible gap — and yet the test accuracy sat there like it had hit a wall. It had. When we finally pulled the 200 highest-loss training examples and looked at them with our own eyes, roughly a third were simply mislabeled: a "cat" that was a dog, a "ship" that was a pier, three near-identical product photos labeled with three different SKUs. The model had not plateaued because it lacked capacity or because the optimizer was stuck. It had plateaued because we were *grading it against wrong answers*, and asking it to fit those wrong answers was actively making it worse.

This is label noise, and it is the most under-diagnosed bug in applied machine learning. It does not crash. It does not NaN. It does not even look like a bug — a run contaminated with mislabeled examples produces a perfectly plausible loss curve, a believable validation number, and a model that ships. The damage is invisible precisely because your instruments are reading the *same corrupted labels* the model trained on. Your validation accuracy is wrong in the same direction as your training, your "best" checkpoint is the one that best memorized the noise, and the ceiling you keep hitting is not a property of your model at all — it is a property of your data. The figure below shows the trap in its purest form: a perfect model, scored against a test set that is itself 3.4% wrong, cannot score higher than about 94% no matter how good it is.

![A before and after comparison showing how a perfect model scored against a noisy test set is capped near 94 percent while the same model scored against a cleaned test set reaches its true accuracy](/imgs/blogs/garbage-in-finding-label-noise-1.png)

By the end of this post you will be able to do four concrete things. First, *estimate* how much label noise your dataset carries, with a real number and a method to back it. Second, *find* the specific mislabeled examples — not "there is probably some noise" but a ranked list of the rows most likely to be wrong, produced by `cleanlab`'s confident learning, by loss-ranking, by cross-validated disagreement, and by two-model agreement. Third, *decide* what to do about each one: clean it, relabel it, drop it, or leave it and switch to a noise-robust loss — a decision that depends on your noise rate and on whether your test set is noisy too. And fourth, *prove* the fix worked with an honest before→after, because the one thing harder than finding label noise is convincing yourself you have actually removed it.

In the language of this series, label noise lives in the very first of the [six places a bug hides](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — **data** — and it is the most common resident there. The discipline is the same as everywhere else: read the instruments, make it fail small, and confirm with a test before you touch anything. We will treat mislabeled data as a hypothesis to be falsified, not an excuse to be assumed.

## 1. The symptom: a stubborn accuracy ceiling that no model change moves

Let us be precise about what label noise looks like from the outside, because its signature is what tells you to suspect it in the first place. Here is the dashboard from the run that opened this post — a ResNet-50 finetune on a 50,000-image, ten-class dataset, after a week of failed improvement attempts:

| Instrument | Reading | The naive read | The real story |
| --- | --- | --- | --- |
| Train loss (final) | 0.21 | "Converging fine" | Partly fitting noise |
| Val accuracy | 87.9% | "Decent, room to grow" | Capped by noisy val labels |
| Test accuracy | 88.1% | "Plateaued — need a better model" | Capped by noisy test labels |
| Train–val gap | small | "Not overfitting" | Misleading: both sets are noisy |
| Highest-loss train examples | loss 6–9 | "Hard examples" | ~30% are mislabeled |

Everything here says "your model is fine but the problem is hard," and that is exactly the wrong conclusion. The tell is the *combination*: a healthy-looking loss curve, a believable gap, and a hard ceiling that does not respond to capacity. When a bigger model, more data, and more compute all fail to move the number, you have almost certainly stopped measuring your model and started measuring your labels. A model bug responds to model changes. A data bug does not.

There is a second, subtler symptom that shows up if you watch training over time rather than at the end: the validation curve climbs, peaks, and then *degrades* if you keep training. That non-monotone shape — get better, then get worse — is the fingerprint of a network that learned the real pattern first and then started memorizing the mislabeled examples, dragging its honest performance back down. We will see exactly why that happens in Section 3, and it is one of the most reliable label-noise tells there is.

Notice what is *missing* from that dashboard, because the absence is itself diagnostic. There is no NaN, no loss spike, no gradient explosion, no out-of-memory crash, no divergence. The run is, by every numerical instrument, *healthy*. That is exactly why label noise is so dangerous: the bugs that announce themselves — numerics, systems failures, broken wiring — at least give you a stack trace to chase. Label noise gives you a *plausible, stable, believable* run that is quietly capped, and the only instrument that betrays it is the one most people never read: the per-example loss on the training set, and your own eyes on the examples it ranks highest. Pay special attention to the deceptively reassuring "small train–val gap" row. Under heavy noise that gap can stay *small* even as the model fails, because both the training metric and the validation metric are computed against the *same corrupted labels* — they are wrong together, in the same direction, so their difference looks fine while their absolute level is meaningless. A small gap is supposed to mean "not overfitting"; under label noise it can mean "consistently mismeasured." This is the deeper reason label noise resists ordinary debugging: it corrupts the instruments you would normally use to detect a problem, so you have to step outside the aggregate numbers entirely and inspect individual examples. Every diagnostic in this post is, at bottom, a principled way to decide *which* examples to inspect first.

The diagnostic stance for the rest of this post is simple. We have a hypothesis — "a meaningful fraction of my labels are wrong" — and three things we need from it: a *measurement* (how much?), a *localization* (which rows?), and a *confirmation* (does fixing them help?). That is the science-diagnostic-evidence triad this whole series runs on, applied to data.

## 2. The science, part one: how much label noise real datasets actually carry

The first thing to internalize is that label noise is not a rare pathology of sloppy datasets. It is the *baseline condition* of essentially every real dataset, including the famous benchmarks you have trusted your whole career.

The definitive measurement here is Northcutt, Athalye, and Mueller's 2021 work, *Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks*, which audited the **test sets** of ten of the most-cited ML benchmarks — MNIST, CIFAR-10, CIFAR-100, ImageNet, and several NLP and audio sets — using confident learning plus human verification on Mechanical Turk. The headline result: an estimated **3.4% average label-error rate across these test sets**. Concretely, they found roughly 5,000 errors in the ImageNet validation set (about 6% of it by their estimate after correction), about 0.5% in MNIST's test set, around 5.8% in QuickDraw, and material error rates in CIFAR, IMDB sentiment, Amazon Reviews, and the 20 Newsgroups text sets. These are the *test* sets — the gold-standard answer keys we use to rank every model in the field — and they are several percent wrong.

That number, 3.4%, is worth holding onto because it has a direct and brutal consequence for measurement, which we derive next. But first, absorb the implication: if the curated, audited, world-famous benchmarks carry 3–6% label error in their *test* sets, your hastily-labeled internal dataset — annotated by a rotating crew of contractors against an ambiguous rubric, with classes that genuinely overlap — is very likely carrying *more*. Industry datasets routinely sit in the 5–15% range, and weakly-supervised or programmatically-labeled sets can be far worse. The question is never "do I have label noise?" The question is "how much, and where?"

### 2.1 Where the noise comes from

Mislabeling is not one phenomenon; it is a family, and knowing the source helps you predict the pattern:

- **Genuine ambiguity.** Some inputs honestly belong to two classes, or to none cleanly. ImageNet famously has images containing both a "monitor" and a "desktop computer"; a single-label answer key has to pick one and is "wrong" half the time by construction. This is *aleatoric* label noise — irreducible without changing the labeling scheme.
- **Annotator error and fatigue.** Humans labeling thousands of items make mistakes, especially on rare classes, near-boundary cases, and after hour three. Error rates rise with class count and with rubric ambiguity.
- **Class-definition drift.** The rubric changes mid-project, or two annotators interpret "is this a complaint?" differently, and you get systematic disagreement that looks like noise.
- **Data-entry and pipeline errors.** A join goes wrong and labels get shifted by one row; a CSV's columns are misaligned; a unit is entered as 1500 instead of 15.00. These are common in tabular data and produce *correlated*, not random, noise.
- **Weak supervision.** Labels generated by heuristics, distant supervision, or another model inherit that source's systematic errors — which tend to be confident and class-conditional, the hardest kind to detect.

The first three dominate in vision and NLP; the fourth dominates in tabular; weak supervision shows up everywhere labels are expensive. Keep this taxonomy in mind, because the *structure* of the noise — is it random, or does it concentrate between two confusable classes? — determines both how much it hurts and how you find it.

There is a simple, useful model of why single-annotator labels carry the error rates they do, and it tells you when to pay for redundant labeling. Suppose a single annotator labels each item correctly with probability $a$ (their per-item accuracy on this task). A single label is then wrong with probability $1 - a$ — so an annotator who is 95% accurate produces a dataset that is 5% noisy by construction, before any pipeline error. Now label each item with $m$ independent annotators and take the majority vote. If the annotators are independent and each is right with probability $a > 0.5$, the probability the *majority* is wrong falls off fast: with three annotators it is roughly $3a^2(1-a)\cdot 0 + \ldots$ — concretely, three 90%-accurate annotators voting drop the error rate from 10% to about 2.8%, and five drop it to about 0.9%. That is the quantitative case for $m$-way redundant labeling on your hardest classes: a few extra labels on the ambiguous slice buys an order-of-magnitude reduction in noise *there*, where it matters, far more cost-effectively than relabeling everything. The catch is the independence assumption — when annotators share a misleading rubric or all get fooled by the same genuinely-ambiguous image, their errors *correlate*, the majority vote does not help, and you are back to needing an expert or a rubric fix. This is also why inter-annotator agreement (Cohen's or Fleiss' $\kappa$) is the leading indicator to watch: low $\kappa$ on a class is a near-certain predictor that that class's labels are noisy, and it tells you *before* you train where the ceiling will come from.

## 3. The science, part two: why noise caps accuracy and why models fit it anyway

Now the part that makes label noise a *quantifiable* bug rather than a vague worry. There are two mechanisms to make rigorous: how noise caps your measured accuracy, and how (and why) a model fits noise even though fitting it hurts.

### 3.1 The measurement ceiling: a perfect model cannot beat the answer key

Suppose your test set has a label-error rate $\eta$ — a fraction $\eta$ of its labels are wrong. Now take the *oracle* model: a hypothetical model that always predicts the true class, perfectly. What test accuracy does it score?

It scores correct on every example whose label is *right* (fraction $1 - \eta$), and it scores *wrong* on every example whose label is *wrong* (fraction $\eta$), because on those examples the oracle predicts the true class while the answer key holds a different one. Assuming, for the moment, that a wrong label is a different class than the truth (the usual case), the oracle's measured accuracy is:

$$
\text{Acc}_{\text{measured}}^{\text{oracle}} = 1 - \eta.
$$

With $\eta = 0.034$, the best possible measured top-1 accuracy is about **96.6%**, and that is the *optimistic* bound assuming every error is a clean swap. In practice some mislabeled examples are genuinely ambiguous and the model may predict the (wrong) answer-key class anyway, which softens the cap; but the load-bearing point stands: **your reported accuracy is bounded above by $1 - \eta$, and no model improvement can break that bound.** If your test set is 6% wrong, 94% is your ceiling, and the six weeks you spend chasing 95% are six weeks wasted. This is exactly the trap drawn in Figure 1 — the same flawless model scores ~94% against a noisy key and its true accuracy against a clean one.

There is a more insidious corollary. When you compare two models on a noisy test set, the noise adds variance to the comparison and can *reorder* them: Northcutt et al. showed that on corrected ImageNet/CIFAR test sets, the model ranking changes — some models that looked better on the noisy set were actually worse, because they were better at predicting the *noise* rather than the truth. So label noise does not just cap your number; it can make you ship the wrong model. We will return to this in the case studies.

Let us make that reordering quantitative, because it is the part people refuse to believe until they see the algebra. Take two models, A and B, with *true* accuracies $a_A$ and $a_B$ where $a_A > a_B$ — model A is genuinely better. Now score both on a test set with label-error rate $\eta$. On the $\eta$ fraction of examples whose label is wrong, each model gets "credit" only if it predicts the (wrong) answer-key class. A model that has *memorized the noise pattern* — fit the systematic mislabeling — will predict the wrong answer-key class on those examples more often, earning spurious credit. Call $c_A$ and $c_B$ the fraction of the noisy examples where each model happens to match the wrong key. The measured accuracies are approximately

$$
\text{Acc}^{\text{meas}}_A \approx (1-\eta)\,a_A + \eta\,c_A, \qquad
\text{Acc}^{\text{meas}}_B \approx (1-\eta)\,a_B + \eta\,c_B.
$$

Model B overtakes A on the *measured* set whenever $\text{Acc}^{\text{meas}}_B > \text{Acc}^{\text{meas}}_A$, i.e. whenever

$$
\eta\,(c_B - c_A) > (1-\eta)\,(a_A - a_B).
$$

Plug in numbers: if A is truly 1 point better ($a_A - a_B = 0.01$) and the test set is 6% noisy ($\eta = 0.06$), then B wins the measured comparison as soon as it predicts the wrong key on just $c_B - c_A > \frac{0.94 \times 0.01}{0.06} \approx 0.157$ — about 16 percentage points more of the noisy examples than A does. A higher-capacity model that overfits the test-set idiosyncrasies can easily clear that bar, which is precisely why correcting the labels reshuffled the benchmark leaderboard. The practical rule that falls out: **when the true accuracy gap between two models is smaller than the test-set noise rate, your comparison is dominated by who fit the noise better, not who is better.** If $a_A - a_B \lesssim \eta$, do not trust the ranking until you clean the test set.

### 3.2 The training-side damage: why a model fits noise, and why that hurts

The test-set story is about measurement. The training-set story is about *learning*, and it is the reason noise actively makes your model worse rather than merely capping your score.

Cross-entropy on a one-hot label is, for a single example with true-class probability $p$, the loss $-\log p$. A mislabeled example points the gradient toward the *wrong* class: it asks the model to raise probability on a class the input does not belong to and lower it on the class it does. Early in training, when the model is far from fitting anything, the mislabeled examples are a minority and their gradients are drowned out by the consistent signal from the correctly-labeled majority — the model learns the real pattern. This is the well-documented **memorization order** result (Arpit et al., 2017, *A Closer Look at Memorization in Deep Networks*): deep networks fit the *easy, consistent* patterns first and memorize the *hard, inconsistent* ones — which is exactly what mislabeled examples are — later.

That ordering is the whole reason the validation curve has its tell-tale shape. The figure below traces it: the network drives down loss on clean examples in the early epochs (val climbs), reaches its honest best around the point where it has learned the real pattern but not yet memorized the noise (val peaks), and then — if you keep training — it starts to fit the mislabeled examples one by one, contorting its decision boundary around them and degrading its generalization (val falls).

![A timeline showing a network learning clean patterns and climbing validation accuracy early, peaking in the middle, then memorizing noisy labels and degrading validation accuracy by the final epochs](/imgs/blogs/garbage-in-finding-label-noise-5.png)

Two consequences fall out of this and both are diagnostic gold:

1. **Mislabeled examples have persistently high loss.** Because the model fits them last and only by brute memorization, throughout most of training a mislabeled example sits at high loss while a clean-but-hard example's loss falls. That is the entire basis of loss-ranking detection (Section 5).
2. **Early stopping is a partial, accidental defense.** Stopping at the validation peak avoids the worst of the memorization — which is why noisy datasets often have a sharp, early "best" checkpoint and degrade afterward. But early stopping does not *remove* the cap; it just stops you walking further into the hole.

Memorizing noise also wrecks **calibration**. A model forced to output high confidence on examples whose labels contradict their content learns to be confidently wrong in those regions, and the contortions needed to memorize a handful of mislabeled points spill over onto nearby clean points, flattening or distorting the probability surface around them. Concretely: after fitting noise, a model's reliability diagram sags below the diagonal in the high-confidence bins — it says 95% and is right 88% of the time — and your expected calibration error climbs. If you are doing anything that depends on calibrated probabilities (thresholding, abstention, downstream Bayesian decisions), label noise corrupts it twice: once through the wrong labels and once through the calibration damage.

#### Worked example: how much accuracy can label noise cost?

Let us put numbers on the training-side damage with a concrete, reproducible scenario. Take CIFAR-10 and inject **20% symmetric label noise** (a standard benchmark setting: with probability 0.2, flip each training label to a uniformly-random other class). Train a standard ResNet to convergence with no noise handling.

- **Clean baseline:** a well-tuned ResNet reaches roughly **94–95%** test accuracy on clean CIFAR-10.
- **20% symmetric noise, train to convergence:** test accuracy typically falls to the **~76–83%** range, depending on architecture and regularization, because the network memorizes the 20% noise and the boundary distortions cost it on clean test points. This is the figure reported across the noisy-label literature (e.g., the GCE and co-teaching papers use exactly this setup).
- **Same 20% noise, early-stopped at the validation peak:** you recover a chunk of it — often back into the **~85–88%** range — because you stop before heavy memorization. Still well below the clean 94%.
- **Same 20% noise, with the worst examples cleaned/dropped (Section 6):** you can recover most of the gap, landing in the **low-90s**.

The lesson in dollars and hours: if your real data carries even 5–10% noise and you train to convergence, you are likely leaving **several points of accuracy** on the table — and on a problem where a point of accuracy is worth real money, that is the single highest-return bug you can fix. It also costs you twice on the test side, because your measurement of *all* of this is happening against a noisy key.

## 4. The science, part three: symmetric, asymmetric, and the noise-transition matrix

To detect and reason about noise precisely, you need its mathematical object: the **noise-transition matrix**. This is the single most useful formalism in the field, and it makes the difference between two kinds of noise that behave completely differently.

Define $T$, a $C \times C$ matrix over your $C$ classes, where

$$
T_{ij} = P(\tilde{y} = j \mid y^* = i)
$$

is the probability that an example whose *true* class is $i$ gets the *observed* (possibly wrong) label $j$. The diagonal $T_{ii}$ is the probability a class-$i$ example is labeled correctly; the off-diagonal mass is the noise. Two regimes:

- **Symmetric (uniform) noise.** With overall noise rate $\eta$, a label is flipped with probability $\eta$ to a *uniformly random* other class: $T_{ii} = 1 - \eta$ and $T_{ij} = \eta/(C-1)$ for $j \ne i$. This is the standard benchmark setting and it is *easy* noise — because the flips are random, they add no consistent signal, so they mostly just add variance and the model's clean-majority learning wins.
- **Asymmetric / class-conditional noise.** The off-diagonal mass concentrates between *specific, confusable* classes: cats get mislabeled as lynxes, "3"s as "8"s, "neutral" sentiment as "positive." Here $T$ has a few large off-diagonal entries. This is *hard* noise, because the flips are consistent — many cats are labeled lynx — so the wrong signal reinforces itself and the model can learn the *flip* as if it were the truth.

The figure below shows the structure, from the true label through both noise regimes to the observed label you actually train on.

![A layered stack showing a true label passing through either symmetric noise that flips to any class or class-conditional noise that confuses specific classes, summarized by a transition matrix, producing the observed label used in training](/imgs/blogs/garbage-in-finding-label-noise-4.png)

The distinction is not academic. It changes everything about detection and remedy:

- **Symmetric noise is detectable and survivable.** Because flips are random, the high-loss / confident-disagreement signal is clean, and a model trained on symmetric noise still mostly learns the truth — you can often clean it after the fact and recover.
- **Class-conditional noise is dangerous because it is *learnable*.** When 18% of cats are labeled lynx *consistently*, the model can fit a "these cats are lynxes" rule with low loss — so loss-ranking *misses* it (the examples aren't high-loss; the model agrees with the wrong label). You need methods that reason about the *joint* distribution of predicted vs given labels — which is exactly what confident learning does, and why it is the right tool when the noise has structure.

Real-world noise is almost always class-conditional, because the things that cause mislabeling (genuine visual similarity, ambiguous rubric, annotator confusion) are themselves class-specific. So while benchmarks use symmetric noise for clean experiments, your detection strategy must assume the harder, structured case. The transition-matrix view is also what makes a principled remedy — *loss correction* via an estimated $\hat{T}$ — possible, which we cover briefly in Section 7.

#### Worked example: reading a 3×3 transition matrix

Suppose a 3-class "complaint / question / praise" text classifier has the estimated transition matrix below (rows = true class, columns = observed label):

| true ↓ \ obs → | complaint | question | praise |
| --- | --- | --- | --- |
| complaint | 0.90 | 0.08 | 0.02 |
| question | 0.05 | 0.92 | 0.03 |
| praise | 0.10 | 0.04 | 0.86 |

Read it like a doctor reads a chart. The diagonal tells you per-class label quality: "praise" is the noisiest class at $T_{\text{praise,praise}} = 0.86$ — 14% of true-praise items are mislabeled, and the dominant confusion is **praise → complaint** at 0.10 (sarcastic praise read as complaint, perhaps). "Complaint" is cleanest at 0.90. The off-diagonal structure tells you *where to spend your annotation budget*: re-review the praise class, and specifically the praise-vs-complaint boundary, not a random sample of everything. The overall noise rate here, weighting classes equally, is about $1 - \frac{0.90+0.92+0.86}{3} \approx 0.107$, or roughly 11% — well above the benchmark 3.4% and squarely in "this is capping your accuracy" territory. That single matrix turns "the model is bad at praise" into "the *labels* for praise are 14% wrong, here is the confusion, go fix it."

## 5. The diagnostic: four ways to find the specific mislabeled examples

Estimating *how much* noise you have is useful; finding *which examples* are wrong is what lets you act. There are four practical detectors, they read different signals, and they trade recall against compute. The matrix below lays out the trade-off, and the rest of this section makes each one runnable.

![A matrix comparing four detection methods by the signal each reads, its compute cost, and what it catches, from cheap loss-ranking to expensive two-model agreement](/imgs/blogs/garbage-in-finding-label-noise-3.png)

### 5.1 Loss-ranking: the five-minute first pass

The cheapest detector falls straight out of Section 3.2: **mislabeled examples have persistently high loss.** So train your model normally, then rank every training example by its per-example loss and look at the top. The highest-loss examples are a mix of genuinely-hard-but-correct examples and outright mislabeled ones, and the mislabeled fraction in that top slice is far higher than the base rate. It is the first thing you should ever do, and it requires no special library.

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def per_example_losses(model, loader, device):
    """Return per-example CE loss for every item in `loader`, in order."""
    model.eval()
    losses, indices = [], []
    for batch in loader:
        x, y, idx = batch["x"].to(device), batch["y"].to(device), batch["idx"]
        logits = model(x)
        # reduction="none" gives one loss per example, not the batch mean.
        loss = F.cross_entropy(logits, y, reduction="none")
        losses.append(loss.cpu())
        indices.append(idx)
    losses = torch.cat(losses)
    indices = torch.cat(indices)
    return indices, losses

idx, losses = per_example_losses(model, train_loader, "cuda")
order = torch.argsort(losses, descending=True)
worst = idx[order][:200]            # the 200 highest-loss training examples
print("Highest-loss example ids:", worst[:20].tolist())
print("Loss at rank 1 / 50 / 200:",
      losses[order[0]].item(), losses[order[49]].item(), losses[order[199]].item())
```

Two implementation notes that matter. First, your dataset must return a stable `idx` per example (wrap it: `return x, y, index`) so you can map ranked positions back to specific rows and eyeball them. Second, `reduction="none"` is the whole trick — the default `mean` reduction throws away exactly the per-example signal you need. Once you have `worst`, **render those examples and look at them** (Section 8 of this series, *look at your data*, is built on this). For images, plot them with their labels; for text, print the text and the label; for tabular, print the row. You will immediately see the mislabeled ones.

Loss-ranking's weakness is the one we already named: it *misses class-conditional noise that the model has learned*. If 18% of cats are confidently labeled lynx and the model fit that flip, those examples have *low* loss — the model agrees with the wrong label. Loss-ranking is a great first pass and a poor last word. For structured noise you need confident learning.

### 5.2 Confident learning with `cleanlab`: the principled detector

Confident learning (Northcutt, Jiang, Chuang, 2021, *Confident Learning: Estimating Uncertainty in Dataset Labels*) is the method behind the 3.4% benchmark result, and `cleanlab` is its production implementation. The idea, in one sentence: use **out-of-sample predicted probabilities** to estimate the joint distribution of (given label, true label), then flag the examples where the model is *confident* the given label is wrong.

The mechanics, which the figure below traces, are:

1. Get **out-of-fold** predicted probabilities for every training example — i.e., for each example, the prediction comes from a model that did *not* train on it (K-fold cross-validation). This is essential: in-sample probabilities are contaminated by memorization, so a mislabeled example the model memorized would look correctly-classified. Out-of-fold predictions don't have that bias.
2. For each class $j$, compute a **self-confidence threshold** $t_j$ = the average predicted probability of class $j$ over examples *labeled* $j$. This per-class threshold adapts to classes the model is systematically less confident about.
3. Build the **confident joint**: count, for each pair $(i, j)$, the examples labeled $i$ whose predicted probability for class $j$ exceeds $t_j$ and is the argmax — these are examples confidently belonging to $j$ but labeled $i$. The off-diagonal of this count matrix estimates the mislabeling structure (it estimates $C \cdot T$ up to normalization).
4. Rank and return the flagged examples by a self-confidence / normalized-margin score, lowest first.

![A branching graph showing out-of-fold probabilities and per-class thresholds feeding a confident-joint count matrix whose off-diagonal entries are ranked by self-confidence to produce a list of likely-wrong labels](/imgs/blogs/garbage-in-finding-label-noise-2.png)

In code, `cleanlab` does all of this for you. The minimal, real API:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues

# X: (n, d) features; y: (n,) integer labels in [0, C).
# Step 1: out-of-fold predicted probabilities (the contamination-free signal).
clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
pred_probs = cross_val_predict(
    clf, X, y, cv=5, method="predict_proba"
)  # shape (n, C); each row predicted by a fold that didn't see that row

# Step 2: confident learning flags the likely-mislabeled rows, ranked worst-first.
issue_idx = find_label_issues(
    labels=y,
    pred_probs=pred_probs,
    return_indices_ranked_by="self_confidence",  # most-confidently-wrong first
)

print(f"Flagged {len(issue_idx)} of {len(y)} "
      f"({100*len(issue_idx)/len(y):.1f}%) as likely mislabeled")
print("Worst 20 row indices:", issue_idx[:20].tolist())
```

For deep models, the only change is *where the probabilities come from*: you run K-fold cross-validation with your actual network (train K models, each predicting its held-out fold) to get `pred_probs`, then hand them to the exact same `find_label_issues`. Here is the pattern in full, because the bookkeeping — getting *out-of-fold* probabilities in the original row order — is the part people botch:

```python
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from cleanlab.filter import find_label_issues

def out_of_fold_probs(dataset, labels, build_model, train_fn, n_splits=5):
    """Return (n, C) out-of-fold softmax probabilities aligned to dataset order."""
    labels = np.asarray(labels)
    n, C = len(labels), len(np.unique(labels))
    pred_probs = np.zeros((n, C), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), labels)):
        model = build_model()
        train_fn(model, dataset, train_idx)          # train ONLY on this fold's train rows
        model.eval()
        with torch.no_grad():
            for i in val_idx:                         # predict the held-out rows
                x = dataset[i][0].unsqueeze(0).cuda()
                pred_probs[i] = F.softmax(model(x), dim=1).cpu().numpy()[0]
        print(f"fold {fold}: filled {len(val_idx)} held-out rows")
    return pred_probs

pred_probs = out_of_fold_probs(train_ds, y, build_model, train_one_fold)
issues = find_label_issues(labels=y, pred_probs=pred_probs,
                           return_indices_ranked_by="self_confidence")
```

The invariant that makes this correct: every row's probability vector comes from a model that *never saw that row in training*. If you accidentally fill `pred_probs[i]` from a model that trained on row `i`, the memorized mislabeled rows look correctly classified and confident learning silently under-flags. `cleanlab`'s higher-level `Datalab` wraps this with extra checks (near-duplicates, outliers, non-IID), but `find_label_issues` is the core and the one to understand.

Why is this better than loss-ranking? Because it reasons about the *joint* — it asks "is this example confidently a member of some *other* class?" rather than just "is this example high-loss?" That is exactly what catches the confident class-conditional flips loss-ranking misses: a cat-labeled-lynx that the model confidently predicts as cat lands in the confident joint's (lynx, cat) cell and gets flagged, even though its *loss against the lynx label* might be low if the model partly memorized it. Out-of-fold prediction is the key that makes this hold up.

It is worth writing the confident joint down precisely, because the per-class threshold is the part that makes it robust and it is easy to get wrong if you reimplement it. Let $\tilde{y}_x$ be the given label of example $x$ and $\hat{p}(j; x)$ its out-of-fold predicted probability for class $j$. Define the per-class self-confidence threshold

$$
t_j = \frac{1}{|\{x : \tilde{y}_x = j\}|} \sum_{x : \tilde{y}_x = j} \hat{p}(j; x),
$$

the *average* predicted probability of class $j$ over the examples that *carry* label $j$. The confident joint $C_{\tilde{y}, y^*}$ is then a $K \times K$ count matrix whose entry $(i, j)$ counts the examples labeled $i$ that the model is confident actually belong to $j$:

$$
C_{i,j} = \Big| \big\{ x : \tilde{y}_x = i,\ \hat{p}(j; x) \ge t_j,\ j = \arg\max_{k:\, \hat{p}(k;x)\ge t_k} \hat{p}(k; x) \big\} \Big|.
$$

Two design choices make this far more robust than naive argmax-disagreement. First, the **per-class threshold $t_j$** means a class the model is systematically under-confident about (say it never exceeds 0.6 on "lynx") still gets its examples counted — you are asking "is this above what this class *usually* scores?" not "is this above 0.5?" That prevents the method from flagging an entire hard class as mislabeled. Second, the **argmax-within-confident restriction** means an example is assigned to at most one off-diagonal cell, so the counts are a genuine joint estimate, not double-counted. Normalizing $C$ and calibrating it to the dataset size yields an estimate of the joint distribution $Q_{\tilde{y}, y^*}$, whose off-diagonal mass is your noise; `cleanlab` then ranks the flagged examples by a normalized-margin / self-confidence score so you can clean the worst first. You do not implement this by hand — `find_label_issues` does — but knowing the threshold is *per-class* and the assignment is *argmax-restricted* tells you why it beats a global confidence cutoff, and why feeding it good out-of-fold probabilities matters more than any other knob.

### 5.3 Out-of-fold disagreement and embedding-neighborhood checks

Confident learning's out-of-fold predictions give you a second, simpler detector almost for free: **disagreement**. An example where the out-of-fold model *confidently predicts a different class than the given label* is a strong mislabel candidate. You can run this without `cleanlab`:

```python
from sklearn.model_selection import cross_val_predict

pred_probs = cross_val_predict(clf, X, y, cv=5, method="predict_proba")
pred = pred_probs.argmax(axis=1)
conf = pred_probs.max(axis=1)

# Confident disagreement: model is sure, and it disagrees with the label.
disagree = (pred != y) & (conf > 0.90)
print(f"{disagree.sum()} examples where an OOF model is >90% sure "
      f"the label is wrong")
suspects = np.where(disagree)[0]
```

This is coarser than the full confident-joint (it does not adapt the threshold per class), but it is trivially fast and a great sanity check that your `cleanlab` flags are sane.

The **embedding-neighborhood** check is the complementary, model-free version. Embed every example (a frozen pretrained backbone for images, a sentence encoder for text), then for each example look at its $k$ nearest neighbors in embedding space. If an example's label disagrees with the *majority* label of its neighbors, it is a mislabel candidate — the input *looks like* members of a different class. This catches noise that is independent of your classifier (it does not require the classifier to be good), and it is especially powerful for finding *duplicate-with-different-labels* — near-identical inputs given conflicting labels, which is both a leakage problem and a noise problem.

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

# emb: (n, d) normalized embeddings; y: (n,) labels.
nn = NearestNeighbors(n_neighbors=11).fit(emb)
_, neigh = nn.kneighbors(emb)               # (n, 11), col 0 is self
neigh = neigh[:, 1:]                         # drop self
neigh_labels = y[neigh]                      # (n, 10)
# Fraction of neighbors agreeing with this example's own label:
agree_frac = (neigh_labels == y[:, None]).mean(axis=1)
suspects = np.where(agree_frac < 0.2)[0]     # <20% of neighbors share my label
print(f"{len(suspects)} examples whose label disagrees with their neighborhood")
```

### 5.4 Two-model agreement: catching the hardest cases

The most expensive and most thorough detector trains **two different models** (different architectures, or different seeds / data orders) and flags examples where *both* models confidently disagree with the given label. The logic is co-teaching's: an example that two independently-trained models both insist is class $j$ while the label says $i$ is very unlikely to be a coincidence — it is a real mislabel, not one model's quirk. This is the highest-precision detector (few false positives) and it naturally surfaces the *hardest* cases, the ones a single model might rationalize. Its cost is literal: you pay for two training runs. Reserve it for high-stakes cleaning where a false positive (deleting a correctly-labeled hard example) is expensive.

For deep models, the two-model check is the same idea with networks instead of `clf`. You already trained K cross-validation models for `cleanlab`; train a second K with a *different* architecture (or just different seeds and augmentation), get a second set of out-of-fold probabilities, and intersect the confident-disagreement sets:

```python
import numpy as np

# probs_a, probs_b: (n, C) out-of-fold probs from two different model families.
pred_a, conf_a = probs_a.argmax(1), probs_a.max(1)
pred_b, conf_b = probs_b.argmax(1), probs_b.max(1)

# Both models confident, both disagree with the label, both agree with each other.
both_wrong = (
    (pred_a != y) & (conf_a > 0.85) &
    (pred_b != y) & (conf_b > 0.85) &
    (pred_a == pred_b)
)
high_precision_suspects = np.where(both_wrong)[0]
print(f"{both_wrong.sum()} examples two independent models both call mislabeled")
```

The four detectors are complementary, not competing. The practical recipe: **loss-rank first** (free, catches obvious errors), **confident-learning second** (principled, catches structured noise), and bring in **embedding-neighborhood** and **two-model agreement** when the noise is class-conditional and the stakes justify the compute.

#### Worked example: reading detector precision and recall

A detector is only useful if you know how much to trust its flags, so let us put numbers on it with a controlled experiment you can run yourself. Take a clean dataset, inject a *known* set of label flips, and measure how well each detector recovers them. Suppose you flip 2,000 of 50,000 labels (a 4% injected noise rate) and run confident learning:

- `find_label_issues` flags **2,300** rows.
- Of those, **1,750** are among your 2,000 injected flips (true positives), and **550** are not (false positives — mostly genuinely-hard or genuinely-ambiguous examples).
- It misses **250** of the injected flips (false negatives — typically the most ambiguous flips, where even a good model is unsure).

That gives **precision $= 1750/2300 \approx 0.76$** and **recall $= 1750/2000 \approx 0.875$**. Read those two numbers operationally. The 76% precision says: if you *drop* every flagged row blindly, about one in four deletions is a correct example you shouldn't have removed — fine when data is abundant, costly when it is scarce, and the reason you *audit a sample before dropping*. The 87.5% recall says: confident learning catches the large majority of real errors, but not all — the residual ~12.5% is why you pair cleaning with a mildly robust loss rather than trusting cleaning to be exhaustive. Crucially, both numbers *improve as your out-of-fold model improves*: a better-converged, better-featured model produces sharper probabilities, which tightens both precision and recall. The detector's quality is downstream of your model's quality, which is the single most important thing to remember about every method in this section.

## 6. The before→after: cleaning the worst 1–3% and proving it helped

Finding suspects is half the job. The other half is *acting* on them and *proving* the action helped — honestly, which is harder than it sounds because your test set is noisy too.

### 6.1 The honest evaluation problem

Here is the trap. You clean your training set, retrain, and your test accuracy goes *up by 0.3 points*. Did cleaning work? You cannot tell, because the test labels are *also* noisy by the same 3–6%, so your measurement is capped and confounded. The fix that actually matters might be hidden under the test-set noise. There are two honest ways out:

1. **Clean (a sample of) the test set too, and measure on the clean slice.** Northcutt et al. did exactly this: they hand-corrected benchmark test sets and re-measured. If you can afford to expert-relabel even a few hundred test examples, do it, and report accuracy on the corrected slice. This is the gold standard.
2. **Use a metric and protocol robust to test noise.** Report the *change* on the same noisy test set but with a confidence interval, and corroborate with a clean held-out probe set if you have one. Note that a true improvement on a noisy test set is *attenuated* — the real gain is larger than the measured one — so a measured +0.3 against a 5%-noisy key may be a real +0.7.

State which of these you did. "Test accuracy went up" is not evidence unless you say what you measured it against.

### 6.2 The cleaning recipe and the result

The actual cleaning, once you have ranked suspects, is a `cleanlab` one-liner plus a retrain:

```python
import numpy as np
from cleanlab.filter import find_label_issues

# pred_probs from K-fold CV on your real model (Section 5.2).
issue_mask = find_label_issues(
    labels=y, pred_probs=pred_probs, return_indices_ranked_by=None
)  # boolean mask, True = likely mislabeled

# Conservative cleaning: drop only the worst K%, not everything flagged.
order = np.argsort(  # rank flagged rows by self-confidence (most-wrong first)
    pred_probs[np.arange(len(y)), y]
)
n_drop = int(0.02 * len(y))                 # drop the worst 2%
drop_idx = order[:n_drop]
keep = np.ones(len(y), dtype=bool)
keep[drop_idx] = False

X_clean, y_clean = X[keep], y[keep]
print(f"Dropped {n_drop} rows ({100*n_drop/len(y):.1f}%); "
      f"retraining on {keep.sum()} rows")
# ... retrain on (X_clean, y_clean), evaluate on a CLEANED test slice ...
```

Run on the ResNet finetune from the intro — dropping the worst 2% of training rows (about 1,000 of 50,000), retraining, and evaluating on a hand-corrected 1,000-image test slice — and you get the result in the figure below: honest test accuracy moves from **88.1% to 90.7%**, and just as importantly the validation curve stops sawtoothing and stabilizes, because the worst memorization targets are gone.

![A before and after comparison showing a model trained on noisy labels with a sawtoothing validation curve and 88.1 percent honest test accuracy versus the same model after dropping the worst 2 percent of rows with a stable validation curve and 90.7 percent honest test accuracy](/imgs/blogs/garbage-in-finding-label-noise-6.png)

#### Worked example: clean vs relabel vs drop, with numbers

You found 1,500 flagged examples in a 50,000-row dataset (3% flag rate). What do you *do* with them? Run the cost-benefit:

- **Relabel** all 1,500. At, say, \$0.30 per expert relabel, that is \$450 and a day of turnaround. Best for accuracy (you recover the example *and* fix its label) and mandatory if 1,500 examples is a meaningful fraction of a small dataset. Recovered accuracy on the worked run: the full ~2.6-point gain, because no data is lost.
- **Drop** all 1,500. Free and instant, costs you 3% of your data. On a 50,000-row set, losing 1,500 rows barely dents capacity, so you recover most of the gain (the ~2.6 points minus a small data-loss penalty). On a 2,000-row set, dropping 3% might hurt more than the noise did — *don't drop when data is scarce*.
- **Hybrid (recommended at scale):** relabel the top 0.5% (the most-confident, highest-value errors) and drop the rest. Captures most of the relabel benefit at a fraction of the cost.

The decision rule: **relabel when data is scarce or each label is high-value; drop when data is abundant and relabeling is too slow; never drop so much that you starve the model.** And always sanity-check a sample of what you are about to drop — `cleanlab` has false positives, and a confidently-flagged "error" that is actually a correct hard example is one you want to keep.

## 7. The full bisection: from "stuck at 88%" to root cause in one afternoon

Detectors and tables are the parts; the discipline is the sequence. Let me walk the ResNet finetune from the intro the way you would actually debug it, bisecting the [six places a bug can hide](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) before touching anything — because a label-noise hypothesis is a *hypothesis*, and the failure mode of junior debuggers is to assume it and start deleting data.

**Step 0 — read the instruments.** The dashboard (Section 1) says: smooth train loss to 0.21, val accuracy 87.9%, test 88.1%, small train–val gap, and a ceiling that ignored a week of model changes. The single most informative fact is the *last* one — capacity, data quantity, and compute were all tried and nothing moved. That immediately deprioritizes **model code** and **optimization** as suspects: a model bug or an LR bug responds to model and LR changes, and these didn't. Numerics is out too — there is no NaN, no spike, no divergence; the curve is smooth. That leaves **data**, **systems**, and **evaluation** in play.

**Step 1 — make it fail small.** Run the [overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test): grab 16 examples, kill augmentation and dropout, and train on them alone. The loss craters to ~0 in 150 steps. *The model can learn.* This rules out the mechanical bugs — frozen head, dead gradient, disconnected loss, LR of zero — and it is decisive: whatever is capping accuracy is not "the model can't learn," it is "the model is learning the wrong thing or being graded wrong." A subtle but important caveat we flagged earlier: overfitting a batch that happens to contain a mislabeled example is still easy (the model just memorizes it), so a *pass here does not exonerate the data* — it only clears the wiring. Good. Data and evaluation are still live.

**Step 2 — separate data from evaluation.** Is the ceiling a *training* problem (the model learned a worse function because it fit noise) or a *measurement* problem (the model is fine but the test labels are wrong)? Bisect by inspecting the test set directly. Pull the test examples the model gets "wrong" with high confidence — `pred != y` and `conf > 0.9` — and *look at them*. In our run, ~30% of those high-confidence "errors" were images where the model was right and the label was wrong. That single observation splits the bug: **both** training and evaluation are contaminated, and the measurement ceiling alone explains a chunk of the missing accuracy.

**Step 3 — confirm with a detector.** Run `cleanlab.find_label_issues` on out-of-fold probabilities over the *training* set. It flags ~3.1% (about 1,550 rows). Sample 100 of them, render them, and audit: ~80 are genuinely mislabeled, ~20 are correct-but-hard. An 80% precision on the flags is plenty to act on, and the 20% false-positive rate is exactly why you *audit before dropping*.

**Step 4 — fix and re-measure honestly.** Hand-correct a 1,000-image test slice (this is non-negotiable; see Section 6.1), drop the worst 2% of training rows, retrain. Honest test accuracy on the corrected slice goes from 88.1% to 90.7%, the val curve stabilizes, and — crucially — the *measured-on-the-noisy-set* number also rises but by less, exactly as the attenuation argument predicts. The bug is confirmed, localized, fixed, and proven, in an afternoon rather than another week of model tweaks.

### 7.1 Stress-testing the diagnosis

A good diagnosis survives "what if?" Here are the branches that change the playbook, because real runs rarely match the clean case:

- **What if the overfit-batch test had *failed*?** Then stop — it is not (primarily) label noise; it is a wiring bug. A model that cannot memorize 16 examples will not be saved by clean labels. Fix the mechanical fault first (frozen layer, detached graph, zero LR), *then* reassess noise.
- **What if `cleanlab` flagged 30%+ of the data?** That is almost never "a third of your labels are wrong." It is far more often a *bad out-of-fold model* — underfit, wrong features, too few folds, or a model that hasn't converged — producing garbage probabilities. Fix the probabilities (train longer, better features, check the CV is real) before trusting the flags. Confident learning is only as good as the predictions you feed it.
- **What if the noise is class-conditional and high?** Loss-ranking will *miss* it (the model agreed with the wrong label, so the loss is low) and aggressive cleaning will start deleting the examples that define a class boundary. Switch to the confident-joint view, lean on a robust loss, and re-review the specific confused-class pair the transition matrix points at — do not blanket-drop.
- **What if it only shows up at convergence, not early?** That is the memorization signature: clean early, degrades late. Early-stopping at the val peak is a free partial fix, but it does not remove the cap. Clean the data if you want the ceiling raised, not just avoided.
- **What if cleaning *doesn't* help the measured number?** Two innocent explanations before you conclude the data was fine: your test set is still noisy (you measured against a bad key — fix it), or the real gain is attenuated below your noise floor (a true +0.7 reads as +0.2 against a 5%-noisy test). Measure on a clean slice with a confidence interval before deciding cleaning was useless.

The throughline: **label noise is a leaf of the bisection tree, reached only after you have ruled out wiring and numerics, and confirmed by looking — never assumed.**

## 8. When to clean, relabel, drop, or switch to a robust loss

Cleaning is not always the answer. The right remedy is a function of your **noise rate** and **whether your test set is noisy too**, and the decision tree below encodes the rule.

![A decision tree branching on whether noise is below five percent with a clean test set toward relabeling or dropping, versus above fifteen percent or a noisy test set toward robust losses and fixing the test set first](/imgs/blogs/garbage-in-finding-label-noise-7.png)

Walk the branches:

- **Low noise (< ~5%) and a clean test set.** This is the cleaning regime. Find the worst examples, relabel them if cheap, drop them if relabeling is costly, and move on. The cap is low and removing a small fraction is safe.
- **High noise (> ~15%) or a noisy test set.** Cleaning becomes unreliable — your detector itself is trained on bad labels, false-positive rates climb, and you may not be able to afford to clean enough. Here, **noise-robust training** earns its keep: methods that *tolerate* noise rather than removing it.
- **A noisy test set, always.** Whatever else you do, *fix the test set first* — even a partial hand-correction — because an uncorrected test set means you cannot measure whether anything you did to training helped. Cleaning training while leaving the test set noisy is debugging blind.

### 8.1 The robust-loss toolbox (when you cannot clean)

When you choose to *tolerate* noise, the main tools are:

- **Robust losses.** Standard cross-entropy is highly sensitive to noisy labels because $-\log p$ is unbounded — a single confidently-wrong label produces a huge gradient. **Generalized Cross Entropy** (Zhang & Sabato, 2018) interpolates between CE and the noise-robust mean-absolute-error loss via a parameter $q$, bounding the per-example loss and capping the damage any one mislabeled example can do. **Symmetric Cross Entropy** and the **mean-absolute-error** loss are in the same family. These need no detection step — they degrade gracefully under noise.
- **Loss correction with an estimated $\hat{T}$.** If you can estimate the transition matrix (Section 4), you can *correct* the loss: the **forward correction** (Patrini et al., 2017) multiplies the model's softmax by $\hat{T}$ before computing CE, so the model is trained to predict the *clean* distribution that, after passing through the known noise process, matches the observed labels. Elegant when $\hat{T}$ is estimable.
- **Sample reweighting / co-teaching.** Down-weight or filter high-loss examples during training (the small-loss trick), or train two networks that teach each other on their confidently-clean examples (co-teaching). These bake the detection into the training loop.
- **Label smoothing.** A blunt but real partial defense: smoothing softens the one-hot target, which caps the gradient magnitude on any single (possibly wrong) label and modestly improves robustness and calibration. It is not a substitute for cleaning, but it is a cheap baseline.

The honest framing: **cleaning and robust-loss methods are complementary, not rivals.** Clean what you can confidently find, then train with a mildly robust loss to soak up the noise you missed. And if your noise is structured and high, lean harder on robust methods, because aggressive cleaning of class-conditional noise risks deleting the very examples that define a class boundary.

## 9. Label noise across modalities: it looks different, the detectors are the same

The mechanics above are modality-agnostic, but the *source* and *tell* of noise differ across vision, NLP, tabular, and speech. The matrix below maps them; the prose makes each concrete.

![A matrix mapping label noise across vision, NLP, tabular, and speech by its noise source, its tell-tale signal, and the best detector for each modality](/imgs/blogs/garbage-in-finding-label-noise-8.png)

**Computer vision.** The classic case, and the one the benchmarks are built on. Noise comes from genuine visual ambiguity (the monitor-and-computer image), annotator error on fine-grained classes (dog breeds, bird species), and multi-object images forced into a single label. The tell is high loss plus confident disagreement; the detector is `cleanlab` on out-of-fold image-model probabilities, optionally backed by an embedding-neighborhood check on a frozen backbone's features. Vision is also where *looking at the data* pays off most — a grid of the 100 highest-loss images is the single most convincing label-noise diagnostic you can produce, because the errors are visible at a glance.

**NLP.** Sentiment, intent, and topic labels are riddled with noise because the categories are genuinely fuzzy (is a sarcastic review positive or negative?) and annotators disagree on boundaries. The IMDB and Amazon-review benchmarks Northcutt et al. audited carry real error. The tell is an out-of-fold model confidently flipping the polarity; the detector is `cleanlab` on a finetuned-encoder's probabilities. A specific NLP trap: *annotator-systematic* noise, where one labeler's idiosyncratic interpretation infects a whole shard of data — group your disagreement analysis by annotator if you have that metadata, and you will sometimes find that "the noise" is one person.

**Tabular.** Here the noise is rarely random — it is *data-entry and pipeline* error, which means it is *correlated*: a misaligned join mislabels a contiguous block, a unit error mislabels everything from one source, a fat-fingered threshold flips a region. The tell is that the top-loss rows *cluster* — same source, same time window, same data provider. The detector is loss-ranking plus *rules*: once you find a cluster of high-loss rows, check whether they share a metadata field, because that field is your bug. Tabular noise also overlaps heavily with leakage; the [data-leakage post](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer) is the companion here.

**Speech.** Transcription labels carry noise from mishearing, inconsistent normalization (numbers, punctuation, casing), and segment-boundary errors. The tell is high per-segment loss or high word-error-rate on specific clips; the detector is aligning the audio to the transcript (forced alignment / CTC) and ranking by alignment loss, which surfaces the clips where the transcript and audio genuinely disagree. A speech-specific subtlety: much "noise" is actually *normalization mismatch* — the transcript says "twenty twenty" and the model says "2020" — which is a metric bug masquerading as a label bug, and you fix it by normalizing both sides before scoring, not by relabeling.

The unifying lesson: across all four modalities, a mislabeled example is one where a model that *did not train on it* is confident the label is wrong. Get out-of-fold predictions, rank by confident disagreement, and look at the top. The pipeline is identical; only the embedding model and the "look at it" tooling change.

## 10. Case studies and real signatures

A few documented and well-known patterns, accurately framed, so you recognize them in the wild.

**Pervasive label errors in the benchmarks (Northcutt et al., 2021).** The foundational result: ~3.4% average label error across ten major test sets, ~6% in the ImageNet validation set after correction, with the striking downstream finding that **correcting the test set reorders the model leaderboard.** On the corrected ImageNet/CIFAR test sets, lower-capacity models that were "worse" on the noisy set were sometimes *better* on the clean one — because the higher-capacity models had partly fit the test-set noise's idiosyncrasies. The practical takeaway: when a leaderboard race is within a couple of points, the label noise may be larger than the gap you are fighting over, and you may be selecting models on noise.

**The validation-curve U-turn.** A recurring real signature: train and validation loss both fall, validation accuracy peaks early, then validation accuracy *declines* while training accuracy keeps rising toward 100%. Practitioners often misread this as ordinary overfitting and reach for more regularization. It is overfitting — but specifically *to mislabeled examples* — and the giveaway is that the highest-loss training examples at the peak are disproportionately mislabeled. The fix is not more dropout; it is cleaning the noise (or early-stopping, which is the accidental partial fix).

**The "hard example" that is actually wrong.** Every applied team has a version of this: a class of examples the model "can't get right," treated as inherently hard, that turns out to be mislabeled. The CIFAR-10 "cat/dog" and ImageNet fine-grained confusions are the canonical public ones. The diagnostic that cracks it is always the same — pull the high-loss / high-confidence-disagreement examples for that class and *look*. The number of times "our model is bad at class X" turns into "class X's labels are 20% wrong" is, in my experience, most of the time.

**Tabular leakage-noise hybrids.** In a Kaggle-style post-mortem pattern, a model's CV score and its leaderboard score diverge, and digging in reveals a subset of rows with both *leaked* and *mislabeled* targets from a broken upstream join. The two bugs travel together because they share a root cause (a bad data pipeline), and the fix — audit the join, recompute the labels — fixes both at once. The lesson: when your tabular noise is clustered and correlated, suspect the pipeline, not the annotators.

**The speech "noise" that was a normalization bug.** A recurring ASR signature: word-error-rate is stuck high, the high-WER clips look fine when you listen to them, and the "errors" are all things like "twenty twenty" vs "2020", "doctor" vs "Dr.", or trailing-punctuation differences. This is *not* label noise — it is a metric and normalization mismatch masquerading as it. The fix is to apply the same text normalizer to both the reference transcript and the hypothesis *before* scoring (lowercase, expand numbers and abbreviations, strip punctuation), at which point the WER drops sharply with no relabeling at all. The diagnostic discipline that saves you here is the one this whole post preaches: *look at the flagged examples*. The moment you read the high-loss transcripts instead of trusting the aggregate number, the normalization bug is obvious — and you avoid burning an annotation budget "fixing" labels that were correct.

**The relabeling recovery curve.** When teams instrument cleaning properly — relabel the worst $k\%$ flagged by confident learning, retrain, measure on a clean test slice, and sweep $k$ — the recovery curve has a characteristic shape: steep gains for the first 1–3% (you are removing the high-loss, high-impact errors), then diminishing returns, then a *flat or slightly negative* tail as you start "correcting" examples that were actually fine and trimming legitimate hard cases. The practical reading: the highest return is almost always in the first couple of percent, which is why "clean the worst 1–3%" is the default recipe rather than "clean everything flagged." Pushing past the knee of that curve spends annotation budget for little gain and risks net harm.

## 11. When this is (and isn't) your bug

Label noise is over-blamed by people who have just learned about it and under-suspected by everyone else. Here is how to tell.

**It probably IS label noise when:**

- A hard accuracy ceiling does not move with model capacity, data quantity, or compute. (Model bugs respond to model changes; data bugs don't.)
- Validation accuracy peaks early and then *degrades* with more training, while training accuracy keeps climbing.
- The highest-loss training examples, when you look at them, are visibly wrong — and *looking* is the confirming test, full stop.
- `cleanlab` flags a few percent and a manual audit of a sample confirms most of the flags are real errors.
- Your test set is on a benchmark or a hastily-labeled internal set you have never audited.

**It probably ISN'T label noise — look elsewhere — when:**

- The loss is *diverging* or going NaN. That is numerics or learning rate, not labels — see [the NaN-hunting post](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) and the optimization track. Label noise degrades smoothly; it does not blow up.
- Your [overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) *fails*. If the model can't memorize 16 examples, the wiring is broken; clean labels won't help a model that can't learn at all. (Note the subtlety: a model can overfit a *noisy* batch fine — overfitting noise is easy — so the overfit test passing does not rule noise *out*; it just rules out the mechanical bugs.)
- The model is *miscalibrated but accurate* and you have not changed labels — that is more likely a temperature/calibration issue than noise.
- Accuracy is fine on a clean holdout but bad in production — that is distribution shift (covariate or label shift), a different data bug. See [class imbalance and when accuracy lies](/blog/machine-learning/debugging-training/class-imbalance-and-when-accuracy-lies) and [your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying) for the metric-side traps that masquerade as noise.
- `cleanlab` flags a *huge* fraction (say 30%+) and the flags are mostly correct examples. That usually means your out-of-fold model is bad (underfit, wrong features), not that a third of your data is wrong — fix the probabilities first.

The one rule to carry: **label noise is a smooth, ceiling-shaped, late-memorizing failure that you confirm by looking at high-loss examples.** Anything sharp, sudden, or that fails the overfit test is somewhere else in the six places.

And one closing piece of discipline that distinguishes a senior debugger here: treat "it's the labels" as a *conclusion to be earned*, not a starting assumption. The trap on both sides is symmetric. The engineer who has never heard of confident learning spends a month tuning a model whose ceiling is set by 6% noisy labels and never thinks to look. The engineer who just discovered `cleanlab` runs it on their first plateau, drops everything it flags, and quietly deletes a class's hardest legitimate examples — making the model *worse* while feeling productive. The cure for both is the same three-step loop this series runs on, applied to data: **measure** the noise (an estimated rate and a transition matrix), **localize** it (a ranked list from out-of-fold confident learning, audited by eye), and **confirm** the fix against a *clean* test slice before you believe it. Do those three and label noise stops being a silent, invisible ceiling and becomes just another bug in the data branch — found in an afternoon, fixed with a 2% drop or a targeted relabel, and proven with a number you can defend.

## 12. Key takeaways

- **Your test set is wrong by a few percent — assume it.** Benchmarks average ~3.4% label error (Northcutt et al., 2021); your internal data is likely worse. A perfect model is capped at $1 - \eta$ accuracy, so a noisy answer key sets a ceiling no model change can break.
- **A ceiling that ignores model capacity is a data bug.** When bigger models, more data, and more compute all fail to move the number, stop measuring your model and start measuring your labels.
- **Loss-rank first, it's free.** Train, compute `cross_entropy(..., reduction="none")`, sort descending, and *look* at the top examples — the mislabeled fraction up there is far above base rate.
- **Confident learning catches what loss-ranking misses.** `cleanlab.filter.find_label_issues` on *out-of-fold* probabilities reasons about the joint distribution of given-vs-true labels, so it flags the confident class-conditional flips a single in-sample model has memorized.
- **Out-of-fold is non-negotiable.** In-sample probabilities are contaminated by memorization; a mislabeled example the model memorized looks correctly-classified. K-fold cross-validation is what makes every detector here honest.
- **Know your noise structure.** Symmetric noise is survivable and easy to detect; class-conditional noise is learnable, dangerous, and needs joint-distribution methods. The noise-transition matrix tells you which classes to re-review.
- **Clean, relabel, drop, or robust-loss — it depends on rate and scarcity.** Relabel when data is scarce or labels are high-value; drop when data is abundant and relabeling is slow; switch to a robust loss (GCE, loss correction) when noise is high or you cannot afford to clean. Never drop so much you starve the model.
- **Fix the test set before you trust any improvement.** Cleaning training while the test set stays noisy is debugging blind — hand-correct even a few hundred test labels and measure there.
- **Confirm by looking, always.** Every detector has false positives. Before you drop a row, render it and check that it is actually wrong — a confidently-flagged correct-but-hard example is one you want to keep.

## 13. Further reading

- **Northcutt, Athalye, Mueller (2021), *Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks*.** The ~3.4% benchmark measurement and the leaderboard-reordering result; the empirical backbone of this whole post.
- **Northcutt, Jiang, Chuang (2021), *Confident Learning: Estimating Uncertainty in Dataset Labels* (JAIR).** The theory and algorithm behind `cleanlab`'s `find_label_issues` — the confident joint, out-of-fold probabilities, per-class thresholds.
- **The `cleanlab` documentation and `Datalab` guide.** The production API for confident learning, plus near-duplicate, outlier, and non-IID checks; the fastest path from this post to a flagged list on your own data.
- **Arpit et al. (2017), *A Closer Look at Memorization in Deep Networks*.** Why networks fit clean patterns first and memorize noise last — the mechanism behind loss-ranking and the validation-curve U-turn.
- **Zhang & Sabato (2018), *Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels*.** The robust-loss family (GCE) for when you cannot clean.
- **Patrini et al. (2017), *Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach*.** Forward/backward loss correction with an estimated transition matrix.
- **Within this series:** the [taxonomy and decision tree](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) that places label noise in the data branch of the six; the companion data posts on [data leakage](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer), [class imbalance and when accuracy lies](/blog/machine-learning/debugging-training/class-imbalance-and-when-accuracy-lies), [looking at your data before you train](/blog/machine-learning/debugging-training/look-at-your-data-before-you-train), and [your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying); and the [capstone debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) that ties the symptom→suspect→test→fix loop together.

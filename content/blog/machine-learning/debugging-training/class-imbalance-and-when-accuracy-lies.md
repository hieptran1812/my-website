---
title: "Class Imbalance and When Accuracy Lies"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why a 99% accurate model can have 0% recall, the gradient math that makes it inevitable, and the metric-loss-threshold fixes that turn a useless classifier into a working one."
tags:
  [
    "debugging",
    "model-training",
    "class-imbalance",
    "metrics",
    "focal-loss",
    "finetuning",
    "deep-learning",
    "pytorch",
    "tabular",
    "evaluation",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/class-imbalance-and-when-accuracy-lies-1.png"
---

A defect-detection model lands on your desk with a number attached: **99.1% validation accuracy**. The team is delighted. The slide deck is written. Then someone in QA asks the only question that matters — "how many of the actual defects did it catch?" — and the answer comes back: **zero**. Not "a few." Not "most of the easy ones." Zero. The model has learned the single most profitable trick available to any classifier facing a rare positive class: it predicts "no defect" every single time, collects its 99.1% accuracy from the 99.1% of parts that genuinely have no defect, and never once fires on the thing it was built to find.

This is the failure mode this post is about, and it is one of the most reliably misdiagnosed bugs in all of applied machine learning. The team will spend the next two weeks treating it as an *optimization* problem — "the model won't learn the positive class, let's try a bigger network, more epochs, a different architecture" — when it is nothing of the sort. The model learned exactly what you asked it to. You asked it to minimize average cross-entropy on a dataset where 99 out of every 100 examples are negative, and it found the global structure of that objective: ignore the rare class, because the rare class barely moves the loss. The bug is not in the model. It is split across your **metric** (accuracy hides the failure), your **loss** (the majority gradient drowns the minority), and your **operating point** (a 0.5 threshold on a skewed score distribution predicts nothing). Fix those three things and the same model, same weights-class, goes from 0% recall to 71% recall on the defects it was always capable of finding.

![A two-panel before and after figure showing a model with 99 percent accuracy and 0 percent recall on the left, transformed by class weighting, focal loss, and threshold tuning into a model with PR-AUC 0.61 and 71 percent recall on the right](/imgs/blogs/class-imbalance-and-when-accuracy-lies-1.png)

By the end of this post you will be able to take any classifier that "won't learn the rare class" and, in about ten minutes and without retraining once, decide whether you are looking at a genuine model bug or — far more often — a metric-and-loss-weighting problem wearing a model bug's clothes. You will know why cross-entropy under imbalance is dominated by the majority gradient (we will derive the exact contribution ratio), why accuracy and even raw loss mislead, what PR-AUC reveals that ROC-AUC conceals, how the threshold trades precision for recall along a fixed score, and how resampling quietly breaks your model's probability calibration. Then we will go through the fixes — class weighting, focal loss, resampling, threshold moving, two-stage training — with the trade-off of each spelled out, plus runnable code that computes honest per-class metrics, a weighted `CrossEntropyLoss`, a focal-loss implementation, and a threshold sweep.

This sits in the **data** and **evaluation** corners of the six places a training bug hides — data, optimization, model code, numerics, systems, evaluation — though, as we will see, the symptom masquerades as an *optimization* bug, which is precisely why bisection matters. If you have not read the series' [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), it is the decision tree this post instantiates; the capstone [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) folds this in with everything else.

## 1. The symptom: a model that scores 99% and does nothing

Let us be precise about what "imbalance" means and why it breaks the things you instinctively trust. Call the positive class (the defect, the fraud, the rare intent, the keyword) the *minority*, with prevalence $\pi$ — the fraction of the dataset that is positive. In the cases that hurt, $\pi$ is small: 1% for many fraud problems, 0.1% for rare manufacturing defects, well under 0.01% for some intrusion-detection streams. The *imbalance ratio* is $\rho = (1-\pi)/\pi$, so a 1% prevalence is a 99-to-1 ratio.

The first thing imbalance does is make accuracy meaningless. The **majority-class baseline** — a model that ignores its input entirely and predicts negative for everything — achieves accuracy $1 - \pi$. At $\pi = 0.01$ that is 99%. At $\pi = 0.001$ it is 99.9%. So when your model reports 99% accuracy, the correct reaction is not "great" but "is that better than the constant `predict_negative` function, which also gets 99% while being worthless?" Often it is not. Accuracy has a *floor* that rises with imbalance, and that floor is exactly where a lazy model parks itself.

Here is the part that surprises people: it is not just accuracy. The **raw training loss** lies too. Cross-entropy averaged over a 99%-negative dataset is dominated, term for term, by the negatives. A model that gets every negative confidently right and every positive confidently *wrong* can still post a low average loss, because there are 99 cheap negatives subsidizing every 1 expensive positive. You will watch your loss curve descend smoothly, exactly the shape that says "training is going fine" in the [loss-curve field guide](/blog/machine-learning/debugging-training/reading-the-loss-curve-as-a-diagnostic), while the model quietly converges to a degenerate solution. The curve is not lying about whether the optimizer is working. It is lying about *what* the optimizer is working toward.

#### Worked example: the accuracy floor

A credit-card fraud dataset has 284,807 transactions, of which 492 are fraudulent — a prevalence of $\pi = 492 / 284807 \approx 0.00173$, about 0.17%, a 578-to-1 ratio. The constant `predict_legitimate` classifier achieves accuracy $1 - 0.00173 = 0.99827$, or **99.83%**. Now suppose your trained neural net reports 99.84% accuracy. That sounds like an improvement of 0.01 points. But 0.01 points of accuracy on 284,807 examples is about 28 transactions — and if those 28 are all *negatives* it flipped from wrong to right (some borderline-legitimate transactions it now classifies correctly) while still missing every single fraud, your "improvement" caught zero fraud. Accuracy cannot distinguish "caught 28 more frauds" from "got 28 more legitimate transactions right and still missed all the fraud." Those are wildly different models with identical accuracy. That single fact is why you must never report accuracy alone on an imbalanced problem; it is structurally incapable of seeing the thing you care about.

The fix begins not with the model but with what you *look at*. The trustworthy signals live at the bottom of the metric stack: per-class recall, the confusion matrix, and the precision–recall curve. The deceptive ones — accuracy and raw loss — live at the top.

![A vertical stack figure ranking metrics from least trustworthy at the top to most trustworthy at the bottom under heavy imbalance, with accuracy and raw loss shown as deceptive and PR-AUC and per-class recall shown as the signals that expose the failure](/imgs/blogs/class-imbalance-and-when-accuracy-lies-2.png)

The diagnostic discipline is one line: **never look at a single aggregate number on an imbalanced problem.** Look at the breakdown by class. The moment you split accuracy into per-class recall, the 99% model's secret is out — recall on the negative class is 100%, recall on the positive class is 0%, and the average that hid this was a weighted mean that gave the positive class a weight of 1%.

## 2. The science: why the majority gradient wins

Accuracy being a bad *metric* is easy to accept. The deeper and more important fact is that imbalance corrupts the *optimization itself* — the gradient that updates your weights is dominated by the majority class, so the model is actively pulled toward the degenerate solution. This is not a metric artifact you can paper over with a better evaluation. It is a force acting on every step. Let us derive it, because the derivation tells you exactly how to counteract it.

Consider binary classification with logistic output. The model produces a logit $z$ for an example; the predicted probability of the positive class is $p = \sigma(z) = 1/(1 + e^{-z})$. The binary cross-entropy loss for a single example with label $y \in \{0, 1\}$ is

$$
\ell(z, y) = -\,y \log p - (1 - y)\log(1 - p).
$$

The gradient of this loss with respect to the logit $z$ has a famously clean form:

$$
\frac{\partial \ell}{\partial z} = p - y.
$$

That is the whole story in one symbol. The gradient pushing on the logit is just the **error** $p - y$. For a negative example ($y = 0$) the gradient is $p$, pointing to push the logit down. For a positive example ($y = 1$) it is $p - 1$, pointing to push the logit up. Now the loss the optimizer actually minimizes is the *sum* (or mean) over the batch, so by linearity the gradient flowing into the shared parameters is the *sum of the per-example gradients*. The total push on any parameter $\theta$ is

$$
\frac{\partial L}{\partial \theta} = \sum_{i \in \text{neg}} p_i \frac{\partial z_i}{\partial \theta} \;-\; \sum_{j \in \text{pos}} (1 - p_j)\frac{\partial z_j}{\partial \theta}.
$$

Here is the imbalance bite. In a batch of 100 with 99 negatives and 1 positive, the first sum has **99 terms** and the second has **1 term**. Even before any weighting, the negative class contributes 99 times as many gradient terms as the positive class. The optimizer feels a force that is, in aggregate, dominated by "make the negatives more confidently negative," and the lone positive's plea to "push my logit up" is a 1% minority vote that gets outvoted on essentially every step.

![A branching graph figure showing a batch of 100 with 99 negatives and 1 positive, the negative gradients and positive gradient flowing into a summed gradient that carries 99 times the mass on negatives, an optimizer step pushing toward the negative class, and a class-weighting fix that equalizes the mass](/imgs/blogs/class-imbalance-and-when-accuracy-lies-4.png)

It gets worse as training proceeds, and this is the subtle part that explains why the model converges all the way to *zero* recall rather than merely poor recall. Early in training the negatives are not yet confidently classified, so each negative's $p_i$ is around 0.5 and its gradient $p_i$ is sizable. But the negatives are easy and plentiful, so the model quickly drives them to high confidence: $p_i \to 0$ for negatives, which means each negative's gradient $p_i \to 0$ too. You might hope that once the negatives are "solved" their gradient vanishes and the positive class finally gets its turn. The catch is the **sum**: 99 small gradients still add up. If each of 99 negatives contributes a residual gradient of even $0.01$, their total is $0.99$ — still comparable to a single positive contributing $1 - p_j$. The minority never gets a clear field. Meanwhile the model has discovered the lowest-loss region is "predict everything is negative with high confidence," and the few positives are not enough to drag it out.

#### Worked example: the gradient contribution ratio

Make it concrete. Suppose at some point in training every negative has $p_i = 0.02$ (confidently negative) and the single positive has $p_j = 0.30$ (the model is unsure, leaning negative). In a batch of 100 (99 neg, 1 pos), assume for simplicity each example's $\partial z_i / \partial \theta$ has comparable magnitude $g$. The negative gradient mass is $99 \times 0.02 \times g = 1.98\,g$. The positive gradient mass is $1 \times (1 - 0.30) \times g = 0.70\,g$. The ratio is $1.98 / 0.70 \approx 2.8$: the negatives push **2.8 times harder** than the positive, even when the negatives are mostly "done" and the positive is badly wrong. The model's incentive is clear — keep tidying up negatives, ignore the positive. Now apply a class weight of $\rho = 99$ to the positive term. The positive mass becomes $99 \times 0.70\,g = 69.3\,g$, the ratio flips to $1.98 / 69.3 \approx 0.029$, and suddenly the positive *dominates* the step. Weighting does not change the model's capacity; it changes which examples the gradient listens to. That is the entire mechanism, and it is why class weighting is the first knob you reach for.

There is one more reason raw loss misleads that the gradient view makes obvious. The average loss is $\frac{1}{N}\sum_i \ell_i$, and with 99% negatives the average is essentially the *negative-class* loss. You can drive that average down to, say, 0.04 nats by getting negatives near-perfect while the positive loss sits at 3.5 nats (badly wrong). The reported number is 0.04. It looks like a well-trained model. It is a model that has memorized "negative" and given up on "positive." Always log loss *per class* if you log loss at all on an imbalanced problem — the gap between negative-class loss and positive-class loss is the tell.

### Where the decision boundary actually moves

The gradient argument explains the *force*; it helps to see where that force pushes the model's final answer. For the constant-input degenerate case — a model with only a bias term $b$ and no useful features — the loss is minimized when the predicted probability equals the empirical prevalence, $\sigma(b) = \pi$. At $\pi = 0.01$ that is $b = \log(\pi / (1-\pi)) = \log(0.01/0.99) \approx -4.6$. The bias alone parks the model's output at 0.01 for *every* input, which is below any sensible threshold, so it predicts negative for everything. That is the analytic fixed point the optimizer is pulled toward before features even enter the picture: the bias term learns the prior, and the prior says "almost everything is negative."

When features *do* carry signal, the imbalance does not zero out the model — it *shifts the decision boundary toward the minority*. Intuitively, because each minority error is one vote against 99 majority votes, the loss-minimizing boundary sits closer to the minority cluster than a balanced boundary would, sacrificing minority recall to avoid majority false positives. This is why even a model that has clearly *learned* the classes (good ROC-AUC) still under-predicts the minority at $t = 0.5$: the learned boundary is biased by the prior. Reweighting the loss is, in this geometric view, a way to *un-bias* the boundary — pull it back toward where it would sit if the classes were balanced. Threshold moving achieves a related effect post-hoc: it slides the decision surface along the score axis without moving the learned boundary at all. Both reach the same destination (more minority predictions); weighting does it during training by changing the gradient, thresholding does it after training by changing the cutoff. Knowing they are two routes to the same place is what lets you choose: weight when you also want better *features* for the minority; threshold when the features are already fine and you just need the right operating point.

## 3. ROC-AUC stays optimistic, PR-AUC tells the truth

So accuracy lies and raw loss lies. The natural next move is a ranking metric that does not depend on the threshold — **AUC**, the area under a curve. But there are two such curves, and on heavily imbalanced data they tell you opposite stories. Choosing the wrong one is how teams ship the 0% recall model with a green dashboard.

The **ROC curve** plots true-positive rate (recall) on the y-axis against false-positive rate on the x-axis as you sweep the threshold. ROC-AUC is the probability that a randomly chosen positive is scored higher than a randomly chosen negative. The trouble with ROC under heavy imbalance is hidden in the x-axis. The false-positive rate is $\text{FP} / (\text{FP} + \text{TN})$, and when negatives are 99% of the data, $\text{TN}$ is enormous. You can let through hundreds of false positives and barely move the FPR, because the denominator is huge. ROC-AUC therefore stays comfortably high — 0.90, 0.95 — even when, in absolute terms, your positive predictions are mostly wrong. ROC measures ranking quality *relative to the vast negative pool*, and that pool is so large it cushions everything.

The **precision–recall curve** plots precision ($\text{TP} / (\text{TP} + \text{FP})$) against recall as you sweep the threshold. Precision's denominator is the predicted-positives, not the true-negatives, so it does *not* get cushioned by the huge negative pool. When your few positive predictions are swamped by false positives, precision craters, and PR-AUC craters with it. Crucially, the PR curve's baseline — the AUC of a random classifier — is exactly the prevalence $\pi$. So a PR-AUC of 0.07 on a 1%-prevalence problem ($\pi = 0.01$ baseline) is genuinely good (7× the baseline), while a PR-AUC of 0.07 reported next to a ROC-AUC of 0.92 immediately tells you the ROC number was lulling you to sleep.

![A two-panel before and after figure contrasting the ROC-AUC view, which stays at 0.92 and looks healthy because the true-negative pool dominates, against the PR-AUC view, which collapses to 0.07 against a base rate of 0.01 and exposes that precision falls off a cliff](/imgs/blogs/class-imbalance-and-when-accuracy-lies-5.png)

The rule of thumb, due to Davis and Goadrich (2006) who proved the deep connection between the two curves: **when you care about the positive class and it is rare, use PR-AUC; ROC-AUC will flatter you.** A point that dominates in PR space dominates in ROC space, but the converse fails dramatically under imbalance — a model can look excellent in ROC and terrible in PR. Saito and Rehman (2015) made the same point with a memorable demonstration on imbalanced data, showing ROC plots that looked nearly identical for good and bad classifiers while the PR plots cleanly separated them.

The full menu of metrics, and what each one hides or reveals under heavy imbalance, is worth keeping on a sticky note:

| Metric | Random baseline | What it hides | What it reveals | Trust under imbalance |
| --- | --- | --- | --- | --- |
| Accuracy | $1-\pi$ (e.g. 0.99) | the entire minority class | nothing useful | none — retire it |
| Raw loss (mean) | depends | minority loss inside the majority average | optimizer is running | low — log per class instead |
| ROC-AUC | 0.50 | false positives cushioned by the TN pool | ranking vs the negative pool | partial — stays optimistic |
| PR-AUC / avg precision | $\pi$ (e.g. 0.01) | nothing it shouldn't | precision–recall trade across thresholds | high — the scoreboard |
| Per-class recall | varies | nothing — it is per class | exactly which class is dead | high — read it first |
| Macro-F1 | varies | nothing — every class equal vote | a dying rare class | high — the single-number choice |

The pattern is consistent: the metrics whose baseline *moves with prevalence* (accuracy at $1-\pi$, ROC's cushioned FPR) are the deceptive ones, because their floor rises to meet a lazy model. The metrics whose baseline *is* the prevalence (PR-AUC at $\pi$) or which look at each class independently (per-class recall, macro-F1) are the honest ones, because a lazy model cannot hide in them.

#### Worked example: the same predictions, two stories

You score 10,000 test examples, 100 positive (1% prevalence). Your model ranks reasonably: most positives get higher scores than most negatives. ROC-AUC computes to **0.92** — looks shippable. Now you pick a threshold to actually make decisions. To catch 70 of the 100 positives (recall 0.70), the threshold is low enough that it also lets through 900 negatives. Precision at that point is $70 / (70 + 900) = 0.072$ — **7.2%**. For every defect you flag, you are wrong 13 times. PR-AUC, integrating precision across all recall levels, comes to about **0.11**. The ROC said 0.92; the PR said 0.11. Both are computed from the identical score vector. The ROC number is not *wrong* — the ranking really is decent — but it answers a question ("can the model rank a random positive above a random negative?") that is not the question your product asks ("of the things I flag, how many are real?"). PR-AUC answers the product's question. On imbalanced problems, that is the number that belongs on the dashboard.

This connects to the broader theme of [your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying): the metric is not buggy in the sense of a code error; it is *answering a different question than the one you think you asked*, and imbalance widens the gap between the two questions until it swallows your project.

## 4. Micro vs macro averaging: the hidden majority vote

There is a second, subtler way imbalance corrupts a metric, and it bites even people who have learned to avoid plain accuracy: **how you average across classes**. When you report a single F1 or a single precision number for a multi-class problem, you have made a choice about averaging, and the two common choices — *micro* and *macro* — disagree wildly under imbalance. Reporting the wrong one is the multi-class version of the accuracy trap.

**Micro-averaging** pools all the per-example decisions before computing the metric: it counts total true positives, total false positives, and total false negatives across *all* classes, then computes one precision/recall/F1 from those totals. Because the totals are dominated by the majority class (it has the most examples), micro-averaged F1 is essentially the majority class's F1 in disguise — and for a single-label problem, micro-F1 equals accuracy exactly. So if you reach for micro-F1 thinking you have escaped accuracy, you have not; you have computed accuracy with extra steps. The rare classes contribute a rounding error to the totals and vanish.

**Macro-averaging** computes the metric *per class* and then takes an unweighted mean. Each class gets an equal vote regardless of size. A model that is perfect on the 99% majority and useless on the 1% minority gets a macro-recall of $(1.00 + 0.00)/2 = 0.50$ — a 0.50 that screams "something is broken," exactly the alarm you want. Macro-averaging is the right default when the rare classes matter as much as (or more than) the common ones, which under imbalance is precisely the situation. There is also **weighted-macro** averaging (per-class metric, weighted by class support), which sits between the two and quietly re-inflates the majority's vote — useful for an overall sense of accuracy-like performance, useless for catching a dead minority class.

#### Worked example: the same model, three averages

A 3-class intent classifier handles 10,000 utterances: 9,500 "chitchat" (95%), 400 "billing" (4%), 100 "cancel" (1%). The model nails chitchat (recall 0.99), does okay on billing (recall 0.70), and completely misses cancel (recall 0.00 — it never predicts cancel). Now look at what each average reports for recall. **Micro-recall** pools the hits: $(9405 + 280 + 0) / 10000 = 0.969$ — a reassuring **96.9%**, indistinguishable from accuracy, hiding the dead class entirely. **Macro-recall** averages per class: $(0.99 + 0.70 + 0.00)/3 = 0.563$ — a **56.3%** that correctly flags a serious problem. **Weighted-macro** re-weights by support: $0.95(0.99) + 0.04(0.70) + 0.01(0.00) = 0.969$ — back to 96.9%, because the 1% cancel class is weighted 1%. Three numbers, same model, same predictions: 96.9%, 56.3%, 96.9%. Only the macro number sees the failure. If your dashboard shows the micro or weighted number, the dead "cancel" class is invisible until a customer cannot cancel their subscription.

```python
from sklearn.metrics import f1_score, recall_score, classification_report

# y_true, y_pred are multi-class label arrays
print("micro-F1   :", f1_score(y_true, y_pred, average="micro"))     # == accuracy
print("macro-F1   :", f1_score(y_true, y_pred, average="macro"))     # rare classes count
print("weighted-F1:", f1_score(y_true, y_pred, average="weighted"))  # support-weighted
# The per-class breakdown is what you actually want to read:
print(classification_report(y_true, y_pred, digits=3, zero_division=0))
```

The discipline mirrors section 1: **never trust a single averaged number on an imbalanced multi-class problem — read the per-class table.** When you must report one number, report *macro* (every class an equal vote) so a dying rare class cannot hide behind the majority's success. The `classification_report` per-class breakdown is the honest artifact; the averages are summaries you should distrust by default.

This averaging trap is especially vicious in long-tailed problems (hundreds of classes with a steep frequency curve), where the head classes can carry a micro-metric to near-1.0 while the entire tail — possibly the classes you built the system for — sits at zero recall. The fix is the same: macro-average, and plot per-class recall sorted by class frequency so the tail's collapse is visible at a glance.

## 5. The threshold is a dial, not a constant

Here is a fact that quietly resolves half of all "the model won't predict the positive class" complaints: **the default 0.5 threshold is almost never the right operating point under imbalance, and changing it requires no retraining at all.**

A classifier outputs a score (a probability, or a logit you can squash into one). To turn that score into a decision you compare it to a threshold $t$: predict positive if score $\ge t$. The default $t = 0.5$ is a convention, not a law. It is the Bayes-optimal threshold only when the two classes are equally prevalent and the costs of the two error types are equal — neither of which holds under imbalance. With 99% negatives, the model's scores for true positives may all sit at, say, 0.2–0.4, simply because the loss-minimizing solution kept positive scores low (refer back to the gradient argument: the model is rewarded for low positive scores). At $t = 0.5$, *nothing* clears the bar, recall is 0%, and it looks like the model learned nothing. Lower $t$ to 0.02 and the same scores, unchanged, now produce 71% recall. You did not retrain. You turned a dial.

The threshold trades precision against recall along the model's fixed score distribution. Lower it and you catch more positives (recall up) but admit more false positives (precision down). Raise it and the reverse. The whole precision–recall curve is just this dial swept from 0 to 1. Choosing the operating point is therefore a *business* decision encoded as a number: how many false alarms can you tolerate per true catch?

![A four-row matrix figure showing how lowering the decision threshold from 0.50 to 0.005 trades precision for recall along fixed scores, moving from 0 percent recall at the default through a 71 percent recall F1 peak to a 93 percent recall flood of false alarms](/imgs/blogs/class-imbalance-and-when-accuracy-lies-8.png)

How do you pick $t$ honestly? You sweep it on a **held-out tuning split** (never the test set — that is [overfitting to the validation set](/blog/machine-learning/debugging-training/overfitting-to-the-validation-set) by another name) and choose the point that maximizes whatever you actually care about: $F_1$ if precision and recall matter equally, $F_\beta$ with $\beta > 1$ if recall matters more (rare-defect detection where a miss is expensive), or the point where precision crosses a contractual floor. The one thing you must not do is pick $t$ on the test set and report the resulting recall as an honest estimate — you will have leaked the operating point.

Threshold moving is, in my experience, the **cleanest** fix for imbalance, and the one teams skip because it feels too simple. It changes no weights, breaks no calibration relationships you did not already break with weighting, and is trivially reversible. If your model already *ranks* positives above negatives reasonably well (good PR-AUC) but predicts none of them at $t = 0.5$, you do not have a model problem at all — you have a threshold set to the wrong value. Move it before you touch the loss function.

## 6. The diagnostic: per-class metrics, the confusion matrix, the predicted-positive rate

Enough theory. Here is the code I run, in order, the moment a classifier "won't learn the positive class." It takes under a minute and tells you whether you are looking at a metric problem (almost always) or a genuine model bug (rarely). The single most diagnostic number is the **predicted-positive rate**: the fraction of examples the model labels positive at the current threshold. If it is exactly 0%, the model is constant-negative and you have an imbalance/threshold problem, full stop.

```python
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, roc_auc_score, average_precision_score,
)

def imbalance_report(y_true, y_score, threshold=0.5):
    """Honest per-class diagnostics for an imbalanced binary classifier.
    y_true:  array of 0/1 ground truth
    y_score: array of model probabilities for the positive class
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    prevalence = y_true.mean()
    y_pred = (y_score >= threshold).astype(int)

    pred_pos_rate = y_pred.mean()
    print(f"prevalence (base rate)   : {prevalence:.4f}")
    print(f"majority-class accuracy  : {1 - prevalence:.4f}  (the lazy baseline)")
    print(f"predicted-positive rate  : {pred_pos_rate:.4f}  <-- 0.0000 means constant-negative")

    # The two AUCs side by side: the gap is the tell.
    print(f"ROC-AUC                  : {roc_auc_score(y_true, y_score):.4f}  (stays optimistic)")
    print(f"PR-AUC (avg precision)   : {average_precision_score(y_true, y_score):.4f}"
          f"  (random baseline = {prevalence:.4f})")

    # Per-class recall/precision -- this is where 0% recall on positives shows up.
    print("\n" + classification_report(y_true, y_pred, digits=3,
                                        target_names=["negative", "positive"],
                                        zero_division=0))

    # The confusion matrix: an empty positive-prediction column is the smoking gun.
    cm = confusion_matrix(y_true, y_pred)
    print("confusion matrix [rows=true, cols=pred]:")
    print(f"            pred_neg   pred_pos")
    print(f"true_neg   {cm[0,0]:8d}   {cm[0,1]:8d}")
    print(f"true_pos   {cm[1,0]:8d}   {cm[1,1]:8d}")
    return cm
```

Run this on the broken model and you get the unmistakable signature: `predicted-positive rate : 0.0000`, the positive row of the confusion matrix is `[all, 0]` (every true positive predicted negative, nothing predicted positive), and `classification_report` shows `recall 0.000` for the positive class. The `zero_division=0` argument keeps sklearn from throwing when precision is undefined (no positive predictions). The instant you see a predicted-positive rate of zero with a non-trivial ROC-AUC, you know the model *can rank* — it just is not *firing*. That is a threshold-and-loss story, not a model story.

The cleanest way to confirm "the model can rank, it just won't fire" is to ignore the threshold entirely and ask whether the positives are scored higher than the negatives on average:

```python
def ranking_is_ok(y_true, y_score):
    """Does the model rank positives above negatives, independent of threshold?
    If this is good but predicted-positive rate is 0, the bug is the threshold/loss,
    not the model's representation."""
    y_true = np.asarray(y_true)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    print(f"mean positive score: {pos.mean():.3f}")
    print(f"mean negative score: {neg.mean():.3f}")
    print(f"ROC-AUC            : {roc_auc_score(y_true, y_score):.3f}")
    # A clear gap with AUC > 0.8 means: ranking is fine, fix the operating point.
    if pos.mean() > neg.mean() and roc_auc_score(y_true, y_score) > 0.8:
        print(">> Model RANKS positives above negatives. Bug is threshold/loss, not model.")
    else:
        print(">> Model does NOT separate the classes. Suspect a real model/data bug.")
```

This is the bisection step that saves the most time. It routes the symptom to the right corner of the six places. If `ranking_is_ok` reports good separation, you do **not** rewrite the model — you fix the loss weighting and move the threshold. If it reports no separation, *then* you go run the [overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) to check whether the model can even fit the positives, and you start suspecting a genuine bug.

![A decision tree figure routing the complaint that a model will not learn positives through a check of the predicted-positive rate, branching to a real model bug if the rate varies with input or to a metric and loss-weighting fix with threshold moving if the rate is stuck near zero](/imgs/blogs/class-imbalance-and-when-accuracy-lies-6.png)

The whole diagnostic, in cost order, fits on one screen: read the confusion matrix, swap accuracy for PR-AUC, check whether the model ranks, and only then decide between a metric fix and a model investigation.

![A timeline figure showing the order to debug an imbalanced run, reading the confusion matrix first, swapping to PR-AUC, adding a class weight of 99, sweeping the threshold to the F1 peak, and finally recalibrating probabilities to undo weighting drift](/imgs/blogs/class-imbalance-and-when-accuracy-lies-7.png)

## 7. Fix 1: class weighting and weighted loss

The most direct counter to the majority gradient is to **reweight the loss** so the minority class carries proportionally more weight, undoing the 99-to-1 term-count imbalance we derived in section 2. In PyTorch this is a one-argument change to your loss.

For `nn.CrossEntropyLoss` (multi-class or binary-as-2-class), pass a `weight` tensor with one weight per class. The conventional choice is inverse-frequency weighting, $w_c \propto 1 / n_c$, often normalized so the weights sum to the number of classes:

```python
import torch
import torch.nn as nn

# class counts from your TRAINING split (never the val/test split)
counts = torch.tensor([98_000., 2_000.])         # [negatives, positives], 2% prevalence
weights = counts.sum() / (len(counts) * counts)  # inverse-frequency, mean-normalized
# weights == [0.5102, 25.0]  -- the positive class is weighted ~49x the negative

criterion = nn.CrossEntropyLoss(weight=weights.to(device))
# logits: [B, 2], targets: [B] of class indices in {0, 1}
loss = criterion(logits, targets)
```

For the binary one-logit setup with `BCEWithLogitsLoss`, the corresponding knob is `pos_weight` — a scalar multiplying the positive term, which you set to the imbalance ratio $\rho = n_\text{neg} / n_\text{pos}$:

```python
pos_weight = torch.tensor([98_000. / 2_000.])    # = 49.0, the neg/pos ratio
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
# logits: [B] or [B, 1], targets: [B] floats in {0., 1.}
loss = criterion(logits.squeeze(-1), targets.float())
```

What this does, mechanically, is multiply each positive example's gradient $p - 1$ by $\rho$, so the summed positive gradient mass now rivals the summed negative mass — exactly the equalization from the worked example in section 2. The model can no longer minimize loss by ignoring positives, because each ignored positive now costs $\rho$ times as much.

The trade-off — and this is the one people forget — is **calibration drift**. By inflating the positive term you have changed the loss the model minimizes, so its output probabilities no longer match true frequencies. A weighted model will be *over-confident* about positives: it outputs $p = 0.6$ for things that are positive only 8% of the time. If you only need a ranking or a thresholded decision, this is fine (the *ordering* of scores is preserved). If you need the probability itself — to feed an expected-value calculation, to set a risk-based premium, to combine with other models — you must recalibrate afterward (Platt scaling or isotonic regression on a held-out set) or correct the prior analytically. We will return to this in the calibration section; for now, file it: **weighting buys recall and costs calibration.**

#### Worked example: weighting flips the operating point

The fraud model from section 1 (0.17% prevalence, $\rho \approx 578$) trained with plain BCE outputs scores that top out around 0.08 for true frauds — nothing clears 0.5, recall 0%. Retrain with `pos_weight = 578`. The positive gradient is now 578× heavier, the model is forced to push fraud logits up, and true-fraud scores now spread across 0.3–0.95. At the default 0.5 threshold, recall jumps from 0% to about 76%, though precision is a noisy 9% (many false alarms). The PR-AUC, which does not depend on the threshold, rises from 0.05 to 0.41 — an 8× lift over the 0.0017 random baseline. The model's *capacity* never changed; we relit the gradient so it would actually use that capacity on the rare class.

## 8. Fix 2: focal loss for a sea of easy negatives

Class weighting treats every negative equally. But in many imbalanced problems — especially dense vision tasks like object detection, where every image has tens of thousands of background anchors and a handful of object anchors — most negatives are *easy*: the model is already correct and confident on them, yet their sheer number still floods the gradient. **Focal loss**, introduced by Lin et al. (2017) in "Focal Loss for Dense Object Detection," targets exactly this: it down-weights examples the model already gets right, so the gradient concentrates on the hard examples regardless of class.

The construction starts from cross-entropy and multiplies by a modulating factor. Let $p_t$ be the model's predicted probability of the *true* class ($p_t = p$ if $y = 1$, else $1 - p$). Cross-entropy is $-\log p_t$. Focal loss is

$$
\text{FL}(p_t) = -\,\alpha_t\,(1 - p_t)^{\gamma}\,\log p_t.
$$

The two new pieces: $(1 - p_t)^\gamma$ is the **modulating factor**, and $\alpha_t$ is an optional class-balancing weight (the same idea as class weighting). The focusing parameter $\gamma \ge 0$ controls how aggressively easy examples are suppressed. When $\gamma = 0$, focal loss *is* (weighted) cross-entropy. As $\gamma$ grows, the down-weighting of confident examples grows.

The mechanism is worth feeling in numbers. Take a well-classified negative with $p_t = 0.95$ (the model is 95% sure it is negative, correctly). The modulating factor at $\gamma = 2$ is $(1 - 0.95)^2 = 0.0025$. That easy example's loss is scaled to **0.25% of its cross-entropy value** — it has been almost entirely silenced. Now take a hard, misclassified positive with $p_t = 0.1$ (the model thinks it is probably negative, but it is positive). Its factor is $(1 - 0.1)^2 = 0.81$ — barely reduced. So the well-classified negative's contribution is suppressed roughly $0.81 / 0.0025 \approx 324\times$ relative to the hard positive. The gradient now flows almost entirely to the examples the model is getting wrong, which under imbalance are disproportionately the minority. Lin et al. found $\gamma = 2$ with $\alpha = 0.25$ a robust default, and reported it let a one-stage detector (RetinaNet) match the accuracy of slower two-stage detectors that had used hard-example mining to dodge the same problem.

Here is a clean, numerically safe implementation for the binary case:

```python
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """Binary focal loss (Lin et al. 2017).
    logits:  [B] raw scores (pre-sigmoid)
    targets: [B] floats in {0., 1.}
    """
    # BCE per element, no reduction; computed from logits for numerical stability.
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)          # prob of the true class
    modulating = (1 - p_t).pow(gamma)                    # (1 - p_t)^gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * modulating * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
```

The trade-off with focal loss is that $\gamma$ is a hyperparameter you have to tune, and it can over-focus: with $\gamma$ too high the loss almost ignores the easy examples entirely, which can hurt when some "easy" negatives are actually informative or when label noise makes a few "hard" examples just wrong (focal loss will obsess over mislabeled points — see [finding label noise](/blog/machine-learning/debugging-training/garbage-in-finding-label-noise), since focal loss and label noise interact badly). Focal loss shines when the imbalance is driven by an *overwhelming pool of easy negatives*; for moderate imbalance where negatives are not trivially easy, plain class weighting is simpler and often just as good.

It helps to see *why* the modulating factor changes the gradient and not just the loss value. Differentiating focal loss with respect to the logit gives a gradient that, like cross-entropy's clean $p - y$, is the error scaled by a factor that *shrinks as $p_t$ grows toward 1*. For a confident-correct example ($p_t \to 1$) both the loss and its gradient go to zero faster than cross-entropy's do — cross-entropy's gradient at $p_t = 0.95$ is still $0.05$, but focal loss's is suppressed by roughly the $(1 - p_t)^\gamma$ factor on top, so the easy example contributes almost nothing to the *update*, not just to the reported loss. That distinction matters: a loss reweighting that left the gradient unchanged would not help at all, because the optimizer only ever sees the gradient. Focal loss earns its keep precisely because it down-weights the *gradient* of the easy majority, which is the quantity that was drowning the minority in the first place. This is the same lever class weighting pulls, but data-dependent: instead of a fixed per-class constant, focal loss's down-weighting adapts per example to how confident-correct the model already is.

One practical caution that follows from this: focal loss and class weighting are *not* additive magic — stacking a large $\alpha$ imbalance weight on top of an aggressive $\gamma$ can over-correct and make the model over-predict the minority, flooding you with false positives. Tune one lever at a time, with PR-AUC as the scoreboard, and stop when the precision–recall trade is where the product needs it.

## 9. Fix 3: resampling, and why naive oversampling overfits

The other family of fixes attacks the *data* rather than the loss: change the class ratio the model sees by **oversampling** the minority (show its examples more often) or **undersampling** the majority (show fewer negatives). Both rebalance the gradient by changing term counts directly rather than by reweighting. Both have sharp caveats.

**Naive oversampling** — duplicating minority examples until the classes are balanced — has a specific, predictable failure: it **overfits the exact copies**. With deep nets that have the capacity to memorize, showing the same 492 fraud transactions 578 times each does not teach the model the *concept* of fraud; it teaches the model those 492 specific feature vectors. Training loss on the minority drops beautifully because the model has memorized the duplicates, and validation recall stays poor because the memorized points do not generalize to *new* frauds. The signature is a widening train–val gap on the minority class specifically — exactly the overfitting pattern, localized to the duplicated examples. If you oversample, oversample *with augmentation* (so the copies are not identical) and watch the minority-class val recall like a hawk.

**Undersampling** the majority avoids the duplication problem but throws away data: drop 95% of your negatives and you have discarded most of what the model could learn about the negative class, which can hurt the decision boundary. It is cheap and fast (smaller dataset, faster epochs) and sometimes the right call when negatives are abundant and redundant, but you are paying in information.

**SMOTE** (Synthetic Minority Over-sampling Technique, Chawla et al. 2002) tries to fix naive oversampling's duplication by *synthesizing* new minority points: for a minority example, pick one of its $k$ nearest minority neighbors and create a new point on the line segment between them. This generates novel-ish minority examples instead of exact copies, which reduces the memorization problem. The caveats are real and underappreciated:

- SMOTE interpolates in **feature space**, which only makes sense when linear interpolation between two minority points yields a plausible minority point. In low-dimensional, continuous **tabular** data this is often reasonable. In high-dimensional spaces (images, text embeddings) the midpoint of two examples is usually a meaningless point in no class's region — interpolating two face images pixel-wise gives a ghost, not a face.
- SMOTE can **blur the boundary** by synthesizing minority points into regions where they overlap with the majority, especially with noisy data, creating false positives the model then learns from.
- Applied **before** the train/test split, SMOTE leaks: synthetic points derived from test minorities contaminate training. SMOTE must live *inside* the cross-validation fold, fit only on training data — a classic [tabular data leakage](/blog/machine-learning/debugging-training/tabular-data-leakage) trap.

Here is the operating-point comparison across the resampling and weighting fixes, with the caveat that decides each one:

![A five-row matrix figure comparing class weighting, focal loss, naive oversampling, SMOTE, and threshold moving by their effect on the gradient, their cost or caveat such as broken calibration or overfitting copies, and when to reach for each](/imgs/blogs/class-imbalance-and-when-accuracy-lies-3.png)

My ranked default: **start with class weighting** (one line, no extra data, just watch calibration), **move the threshold** (free, reversible, often sufficient), reach for **focal loss** when easy negatives dominate, and treat **SMOTE/oversampling** as a tabular-specific tool used with care and always inside the CV fold. Undersampling is a speed optimization, not a correctness fix.

A note on *why* class weighting usually wins the head-to-head against resampling on deep nets: weighting and oversampling are, in expectation, the same intervention — both increase the minority's contribution to the loss by the same factor — but they differ in *variance* and *side effects*. Oversampling achieves the reweighting by physically repeating examples, which inflates the number of gradient steps that touch identical (or near-identical, with SMOTE) points; that extra repetition is what drives the memorization. Weighting achieves the identical expected gradient with no repeated data, so it cannot memorize copies it never made. Undersampling matches the ratio by discarding majority data, trading bias for a smaller, faster dataset — sometimes worth it when the majority is genuinely redundant, never worth it when every negative carries distinct boundary information. The practical upshot is a clean ordering for deep nets: prefer the reweighting that touches no data (class weight / `pos_weight` / focal $\alpha$), use synthetic resampling only in low-dimensional tabular settings where interpolation is meaningful, and reach for undersampling purely as a throughput lever. The three are routes to the same rebalanced gradient; they differ in what they break on the way there.

```python
# Resampling with imbalanced-learn, INSIDE a pipeline so it only touches train folds.
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline      # imblearn's Pipeline, not sklearn's
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

pipe = Pipeline([
    ("smote", SMOTE(k_neighbors=5, random_state=0)),   # fit on train fold ONLY
    ("clf", LogisticRegression(max_iter=1000)),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# average_precision == PR-AUC; the honest metric under imbalance
scores = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision")
print(f"PR-AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
```

The key discipline in that snippet is using `imblearn.pipeline.Pipeline`, which applies SMOTE *only to the training fold* of each split. If you SMOTE the whole dataset first and then cross-validate, you leak synthetic neighbors of test points into training and your PR-AUC will be optimistically inflated — the kind of [data leakage](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer) that looks like a great result and dies in production.

## 10. Two-stage training and gradient-boosting knobs

Two more fixes round out the toolkit, one for deep nets and one for the gradient-boosted trees that dominate tabular problems.

**Two-stage training** decouples the *representation* from the *decision*. You train the backbone first on a balanced or reweighted objective to learn good features, then *freeze* it and retrain only the classifier head (or just recalibrate the threshold) on the natural, imbalanced distribution. The idea, formalized in "Decoupling Representation and Classifier for Long-Tailed Recognition" (Kang et al. 2020), is that imbalance hurts the *classifier boundary* more than it hurts *feature learning* — the features learned even on imbalanced data are often fine; it is the final linear layer that gets dragged toward the majority. So you learn features however works, then adjust only the decision rule. In practice the cheap version of this is exactly what we have been building toward: train with weighting or focal loss to get a good ranking, then *move the threshold* on a held-out split. That is two-stage training with a one-parameter second stage.

For **gradient boosting** (XGBoost, LightGBM), the imbalance knob is `scale_pos_weight`, set to the negative-to-positive ratio — the tree analog of `pos_weight`:

```python
import xgboost as xgb

ratio = (y_train == 0).sum() / (y_train == 1).sum()   # neg / pos
model = xgb.XGBClassifier(
    scale_pos_weight=ratio,        # rebalance the gradient, like pos_weight
    eval_metric="aucpr",           # PR-AUC, NOT accuracy or error
    n_estimators=2000,
    early_stopping_rounds=50,      # stop on the honest metric
)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],   # val set must be a CLEAN, un-leaked split
          verbose=False)
```

Two traps live in that snippet, and they are the heart of [gradient boosting and imbalance debugging](/blog/machine-learning/debugging-training/gradient-boosting-and-imbalance-debugging). First, `eval_metric` must be `aucpr` (PR-AUC) and not the default `error` (accuracy-like) — early stopping on accuracy will happily stop at the constant-negative model because its "error" looks great. Second, `early_stopping_rounds` watches the validation set, so if that validation set is *leaked* (preprocessing fit on all data, duplicate rows across the split), early stopping reads a fantasy curve and you overfit to the leak. The eval set has to be a genuinely held-out, leak-free split or both the metric and the stopping rule are lying.

#### Worked example: the XGBoost fraud model, before and after

A team trains XGBoost on the fraud data with defaults: `scale_pos_weight=1`, `eval_metric="error"`, 2000 rounds, early stopping. It stops at round 31 (error already at the 0.17% floor), reports 99.83% accuracy, and catches 4 of 123 test frauds — recall 3%. The fix is three lines: set `scale_pos_weight` to the 578-to-1 ratio, change `eval_metric` to `"aucpr"`, and confirm the val split is clean. Now early stopping runs to round 640 because PR-AUC keeps improving, the model catches 94 of 123 frauds at a tuned threshold (recall 76%), precision is 41%, and PR-AUC on the test set is 0.68 versus the previous 0.21. The accuracy barely moved (99.83% → 99.86%) — which is the whole point: accuracy was never going to show the difference between a model that catches 3% of fraud and one that catches 76%.

## 11. Case studies and real signatures

The imbalance failure recurs across every modality with the same shape and the same fix. Naming the real instances builds the pattern-match.

**Computer vision — dense detection.** This is the canonical case that produced focal loss. A single-stage object detector evaluates on the order of $10^4$–$10^5$ candidate boxes per image, of which only a handful contain objects. The background anchors outnumber foreground by roughly 1000-to-1, and most backgrounds are easy. Before focal loss, the standard fix was *hard negative mining* — explicitly subsampling negatives to a manageable 3-to-1 ratio (as in the original SSD). Lin et al. (2017) showed focal loss with $\gamma = 2$, $\alpha = 0.25$ removed the need for mining entirely and let RetinaNet, a one-stage detector, match the COCO accuracy of the slower two-stage Faster R-CNN family. The signature of getting it wrong: the detector trains to a low loss and predicts background everywhere — recall on rare classes near zero — exactly the tabular signature, just spread across an image.

**Tabular — fraud and credit risk.** The Kaggle Credit Card Fraud dataset (284,807 transactions, 492 frauds, 0.17% prevalence) is the standard teaching example precisely because naive pipelines on it report 99.8% accuracy and catch almost no fraud. The community's hard-won lesson, repeated across hundreds of post-mortems: report **PR-AUC / average precision**, not accuracy or ROC-AUC; use `scale_pos_weight` or class weights; keep any resampling *inside* the CV fold. The before→after is dramatic on paper and invisible in accuracy — which is why so many tutorials that stop at accuracy ship a useless model.

**NLP — rare intent and toxic-content detection.** In intent classification or toxicity filtering, the target class can be well under 1% of traffic. A BERT finetune trained with plain cross-entropy will achieve high accuracy by predicting "not toxic" / "no intent" almost always. The fix is the same triad — weighted loss, PR-AUC for model selection, threshold tuned on a dev set to the precision the moderation policy requires — and the additional wrinkle that the *threshold is a policy lever*: a stricter precision requirement (fewer false flags on benign content) maps directly to a higher threshold. The [metric-is-lying](/blog/machine-learning/debugging-training/your-metric-is-lying) failure is especially common here because validation accuracy looks excellent right up until the model is deployed and flags nothing.

**Speech — rare keyword spotting.** A wake-word or keyword-spotting model hears the target word in a tiny fraction of audio frames; the "no keyword" class dominates by orders of magnitude. Trained naively it predicts "no keyword" and posts high frame accuracy while never waking. The fixes carry over (frame-level class weighting, PR-AUC / detection-error-tradeoff curves instead of accuracy, threshold set to the false-accept rate the product allows), with the modality-specific note that the operating point is often expressed as a false-alarm rate per hour — a threshold in different clothes.

**The confident-learning connection.** One adjacent caution: when you crank up minority weighting or focal loss, you make the model obsess over the hardest minority examples — and some of those "hard" examples are simply *mislabeled*. Confident learning (Northcutt et al., the basis of `cleanlab`) found label errors in the test sets of major benchmarks (on the order of 3% across ImageNet, MNIST, and others). Under heavy minority weighting, those few mislabeled minority points get enormous gradient and can dominate training. If your weighted model's loss is being held up by a handful of examples, inspect them before assuming the model is at fault — they are often label noise, not hard positives.

## 12. Calibration drift: the cost of every reweighting fix

Every fix that rebalances the gradient — class weighting, `pos_weight`, oversampling, focal loss — changes the objective the model minimizes, and therefore changes what its output probabilities *mean*. This is the most-skipped consequence in imbalance work, and it matters the moment anyone downstream treats your score as a probability.

The mechanism is clean. A model trained on the natural distribution outputs $p \approx P(y = 1 \mid x)$ — a calibrated posterior. When you upweight positives by $\rho$, you are effectively training on a *resampled* distribution where positives appear $\rho$ times more often, so the model learns the posterior *under that fake prior*. Its output is now $P_{\text{resampled}}(y = 1 \mid x)$, which is systematically too high relative to the real base rate. Concretely, a weighted model might output 0.6 for a transaction whose true fraud probability is 0.03. The *ranking* is unaffected (scores still order examples correctly), so thresholded decisions and PR-AUC are fine. But if you feed that 0.6 into an expected-loss calculation, set an insurance premium from it, or average it with another model's probability, you will be badly wrong.

There is an exact prior-correction formula. If you trained with the positive class upsampled by factor $\rho$ (or weighted, which is equivalent in the limit), the calibrated probability is recovered by

$$
p_{\text{cal}} = \frac{p}{p + (1 - p)\,\rho}.
$$

In practice the robust fix is to *fit* a calibrator on a held-out set rather than trust the formula: Platt scaling (a logistic regression on the logits) or isotonic regression maps the distorted scores back to calibrated probabilities.

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

# Option A: recalibrate an already-trained model on a held-out calibration split.
calibrated = CalibratedClassifierCV(base_estimator, method="isotonic", cv="prefit")
calibrated.fit(X_calib, y_calib)        # X_calib must be UNSEEN by base_estimator
p_calibrated = calibrated.predict_proba(X_test)[:, 1]

# Option B: isotonic on raw scores you already have (e.g. from a torch model).
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(y_score_calib, y_calib)         # fit score -> empirical frequency
p_calibrated = iso.transform(y_score_test)
```

The discipline: **if you only need a decision, weight freely and tune the threshold; if you need a probability, recalibrate after weighting on a held-out split.** Confusing the two — reporting a weighted model's raw `predict_proba` as a probability — is its own silent bug, the evaluation-corner sibling of the training-corner imbalance bug.

## 13. Putting it together: bisecting a real imbalanced run

Theory and isolated fixes are one thing; the skill is sequencing them under pressure on a run that *looks* like it learned nothing. Here is the full bisection on a realistic span-defect model, the kind that flags defective spans in a manufacturing line from sensor windows. The symptom: after 30 epochs, validation accuracy is **99.3%** and the line operators report it has never once flagged a real defect. The team's instinct — and it is the wrong instinct this whole post is built to correct — is "the model is too small / the features are bad / we need a transformer." Resist it. Run the cheap tests in cost order and let the instruments route you.

**Step 1 — read the confusion matrix (10 seconds).** You run `imbalance_report` on the validation predictions. Prevalence is 0.7% (defects are rare, as they should be). The predicted-positive rate is **0.0000** — the model predicts "no defect" for every single validation window. The positive column of the confusion matrix is all zeros. This is the textbook degenerate solution. Already you can cross *four* of the six places off the list: this is not a systems bug (it reproduces on one GPU), not a numerics bug (no NaN, the loss is a clean small number), not a reproducibility issue (it is deterministic). The suspects are now just two: model code, or data/evaluation (imbalance).

**Step 2 — does the model rank? (5 seconds).** You run `ranking_is_ok`. Mean positive score: 0.21. Mean negative score: 0.04. ROC-AUC: **0.86**. The positives are scored more than 5× higher than the negatives on average, and the AUC is well above chance. *The model has learned a discriminative representation.* It can tell defects from non-defects — it simply never crosses 0.5. This single result demolishes the "the model is too small / can't learn" hypothesis. A model that cannot learn the positive class does not produce ROC-AUC 0.86; it produces ROC-AUC ~0.5 with positive and negative scores on top of each other. So the bug is *not* model code. We are down to one suspect: the metric-and-threshold-and-loss story. We have localized the bug in 15 seconds without retraining.

**Step 3 — swap the metric (2 seconds).** Before fixing anything, fix what you *look at*, so you can measure the fix. PR-AUC (average precision) on the current model is **0.18** against a base rate of 0.007 — so the model is already 25× better than random at ranking, which confirms step 2 from a different angle and gives us the number to beat. Accuracy is now retired from the conversation; PR-AUC is the scoreboard.

**Step 4 — move the threshold (1 minute, no retraining).** Because the model already ranks, the cheapest possible fix is to slide the threshold. You sweep $t$ on a held-out tuning split and print recall/precision at each:

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_tune, y_score_tune)
# F1 at each threshold; guard the divide-by-zero where precision+recall == 0
f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0)
best = np.argmax(f1[:-1])          # last point has no threshold
t_star = thresholds[best]
print(f"best threshold: {t_star:.4f}  recall: {recall[best]:.3f}  precision: {precision[best]:.3f}")
# typical result here: best threshold 0.09, recall 0.63, precision 0.29
```

At $t^\star = 0.09$ the *same model, same weights*, now catches 63% of defects at 29% precision. Recall went from 0% to 63% by changing one number. For many products that is already shippable — and you got there with zero retraining, which is the entire reason threshold moving is the first fix to try once ranking is confirmed.

**Step 5 — add class weighting and retrain (one training run).** To push recall higher and improve the *features* for the minority (not just the cutoff), you retrain with `pos_weight = n_neg/n_pos ≈ 141`. This relights the gradient (section 2), the model now pushes defect logits up, and the positive scores spread from 0.21-centered to 0.55-centered. Re-sweeping the threshold on the tuning split, you find the F1 peak at $t = 0.31$ with recall 0.71 and precision 0.34. PR-AUC rises from 0.18 to **0.61** — the headline number from this post's opening figure. The features genuinely improved; the model is no longer merely ranking, it is *separating*.

**Step 6 — recalibrate (optional, one held-out fit).** The weighted model is now over-confident (section 12): it outputs 0.55 for windows that are defective only ~30% of the time. The line's downstream logic multiplies the defect probability by a repair cost to decide whether to halt the line, so it needs a *real* probability, not a ranking. You fit isotonic regression on a held-out calibration split, and the over-confident scores map back to calibrated probabilities. Now 0.30 means 30%. The decision math is correct again.

The whole bisection — symptom to shipped fix — touched the model architecture exactly *zero* times. The defect was never in the model code, the numerics, or the systems. It lived in the metric (accuracy hid it), the loss (the majority gradient drowned the minority), and the operating point (0.5 was wrong). Every step bought information before the next, and the most expensive step (retraining) came last, only after the cheap tests had proven it was worth doing. That ordering — confusion matrix, then ranking check, then metric swap, then threshold, then weighting, then calibration — is the playbook. Memorize the order; it converts a two-week "the model won't learn" goose chase into an afternoon.

## 14. When this is (and isn't) your bug

Bisection is only useful if you know when to *stop* blaming imbalance and look elsewhere. Here is how to tell.

**It IS an imbalance/metric bug when:** accuracy is high but suspiciously close to $1 - \pi$; the predicted-positive rate is near 0%; the confusion matrix has an empty (or nearly empty) positive-prediction column; ROC-AUC looks fine but PR-AUC is near the base rate; the model *ranks* positives above negatives (good AUC) but fires on none of them at $t = 0.5$. In this case the cure is metric + loss weighting + threshold, and you should not touch the architecture.

**It is NOT an imbalance bug — look elsewhere — when:** the model fails the [overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) (it cannot drive loss to zero even on a tiny balanced batch — that is a model-code or optimization bug, not imbalance); the `ranking_is_ok` check shows positives and negatives have *the same score distribution* (the model has learned nothing discriminative — suspect a [data pipeline](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you) bug, label scrambling, or features with no signal); or the positive-class *loss* is high *and not improving even with massive weighting* (weighting the gradient should always move recall — if it does not, the gradient is not reaching those parameters, which is a frozen-layer or [gradient-flow](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) bug). A useful one-liner: **if 10× more weight on the positive class does not change recall at all, the bug is not imbalance — the gradient is not getting there.**

Two more disambiguations. A *too-good* result on the val set that collapses in production is more likely [data leakage](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer) than an imbalance fix gone right — be suspicious if class weighting alone took you from 0% to 95% recall *and* 95% precision, because honest imbalance fixes trade precision for recall, they do not hand you both. And if the metric looks fine offline but the model degrades over weeks in production while the data looks balanced, that is [distribution shift](/blog/machine-learning/debugging-training/distribution-shift-train-vs-the-real-world), not imbalance — the base rate itself moved.

#### Worked example: ruling imbalance in, then out

A keyword-spotting model reports 99.4% frame accuracy and never wakes. You run `imbalance_report`: prevalence 0.6%, predicted-positive rate 0.0%, ROC-AUC 0.88, PR-AUC 0.09 (baseline 0.006). The empty positive column and the ROC/PR gap say *imbalance* loudly. You add `pos_weight = 165` and recall climbs to 58% — confirmed, it was imbalance, and weighting reached the parameters. Now a *second* keyword-spotting model, same symptom, but when you add `pos_weight = 165` recall stays at 0%. Weighting did not move it, so the bug is *not* imbalance. You run overfit-one-batch on a balanced mini-batch: it also fails to drive loss down. The gradient is not reaching the head — and you find the final layer was accidentally created with `requires_grad=False`. Two identical symptoms, two different corners of the six places, separated in three minutes by the question "does weighting move recall?"

The disambiguation, as a lookup table — the same symptom (high accuracy, no positive predictions) routes to different corners depending on the confirming test:

| Observation | Confirming test | Most likely suspect | Fix direction |
| --- | --- | --- | --- |
| Pred-positive rate 0%, ROC-AUC 0.86 | `ranking_is_ok` shows clear gap | Imbalance: metric + threshold | move threshold, then weight |
| Pred-positive rate 0%, ROC-AUC ~0.50 | overfit-one-batch also fails | Model code / dead features | fix grad flow, check pipeline |
| Recall flat even at `pos_weight=10×` | weighting does not move recall | Frozen layer / detached graph | un-freeze, check `requires_grad` |
| 0% to 95% recall AND 95% precision | dedup across splits, time split | Data leakage, not a real fix | GroupKFold / temporal split |
| Great offline, decays over weeks | re-measure the live base rate | Distribution shift | re-collect, monitor prevalence |
| High macro-F1, low micro-F1 (or vice versa) | read the per-class table | Averaging choice, not the model | report macro, read per class |

The table is the whole post compressed: the symptom is rarely diagnostic on its own, but the symptom *plus one cheap confirming test* points to exactly one of the six places. That is bisection — and it is why the test in the second column matters more than any number in the first.

## Key takeaways

- **Accuracy has a floor of $1 - \pi$ under imbalance; never report it alone.** A 99% accuracy at 1% prevalence is the lazy `predict_negative` baseline. Report per-class recall, the confusion matrix, and PR-AUC.
- **The majority gradient wins by term count.** Summed cross-entropy gives the 99% class 99× the gradient terms; the model is *optimized* toward the degenerate "predict negative" solution. This is an optimization force, not just a metric artifact.
- **Raw loss lies too.** A smoothly descending loss curve can be 99 cheap negatives subsidizing 1 expensive, badly-wrong positive. Log loss *per class* to see it.
- **ROC-AUC stays optimistic; PR-AUC tells the truth.** ROC's huge true-negative denominator cushions false positives. Use PR-AUC (baseline = prevalence) when the positive class is rare and you care about it.
- **The threshold is a free dial.** If the model *ranks* positives above negatives but fires on none at 0.5, you have a threshold bug, not a model bug — sweep $t$ on a held-out split, no retraining.
- **Class weighting / `pos_weight` is the first knob.** Set `pos_weight = n_neg/n_pos` to equalize gradient mass; it costs calibration, so recalibrate if you need probabilities.
- **Focal loss for a sea of easy negatives.** $(1-p_t)^\gamma$ with $\gamma = 2$ suppresses confident examples ~324× relative to hard ones; ideal for dense detection, overkill for moderate imbalance.
- **Naive oversampling overfits the copies; SMOTE only suits low-dim tabular and must live inside the CV fold.** Both leak if applied before the split.
- **The fastest disambiguation: does 10× weight move recall?** If yes, it was imbalance. If no, the gradient is not reaching the parameters — go hunt a frozen layer or a broken pipeline.

## Further reading

- Lin, Goyal, Girshick, He, Dollár. **"Focal Loss for Dense Object Detection"** (ICCV 2017) — the $(1-p_t)^\gamma$ derivation, $\gamma$/$\alpha$ defaults, and the dense-detection imbalance setting.
- Davis & Goadrich. **"The Relationship Between Precision-Recall and ROC Curves"** (ICML 2006) — why a point dominating in PR dominates in ROC but not conversely, and why PR is the right view under imbalance.
- Saito & Rehman. **"The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets"** (PLOS ONE 2015) — the demonstration that ROC plots look identical while PR plots separate good from bad.
- Chawla, Bowyer, Hall, Kegelmeyer. **"SMOTE: Synthetic Minority Over-sampling Technique"** (JAIR 2002) — the interpolation method and its assumptions.
- Kang et al. **"Decoupling Representation and Classifier for Long-Tailed Recognition"** (ICLR 2020) — the two-stage idea: features survive imbalance, the classifier does not.
- PyTorch docs — `torch.nn.BCEWithLogitsLoss` (`pos_weight`) and `torch.nn.CrossEntropyLoss` (`weight`); scikit-learn `precision_recall_curve`, `average_precision_score`, and `CalibratedClassifierCV`.
- Within this series: the [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the symptom→suspect→test→fix tree this post fills in), the capstone [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook), and the siblings [your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying), [loss-function bugs](/blog/machine-learning/debugging-training/loss-function-bugs), and [gradient boosting and imbalance debugging](/blog/machine-learning/debugging-training/gradient-boosting-and-imbalance-debugging).

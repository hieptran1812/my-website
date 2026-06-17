---
title: "Your Metric Is Lying: Micro vs Macro, Eval Skew, and the Offline-Online Gap"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A number that goes up is not proof the model got better; learn the eight ways a metric lies, the math behind each, and the runnable checks that tell you whether your headline number is real."
tags:
  [
    "debugging",
    "model-training",
    "metrics",
    "evaluation",
    "class-imbalance",
    "data-leakage",
    "finetuning",
    "deep-learning",
    "pytorch",
    "scikit-learn",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/your-metric-is-lying-1.png"
---

The dashboard says F1 went from 0.88 to 0.95. The standup celebrates. Someone schedules the production rollout. Three weeks later, support tickets pile up about the one thing the model was supposed to catch — the fraud, the outage, the toxic comment — and nobody can explain why a model with a 0.95 F1 is missing it constantly.

Here is the uncomfortable truth that this post is about: **a number that goes up is not proof the model got better.** The metric is a measurement instrument, and like any instrument it can be miscalibrated, pointed at the wrong thing, or read off the wrong scale. When that happens, the metric reports success while the model fails — and because the number looks good, nobody goes looking for the bug. A lying metric is the most dangerous bug in machine learning precisely because it hides itself behind good news.

This is the evaluation corner of the six places a bug can hide — data, optimization, model code, numerics, systems, and **evaluation**. It is the corner people instrument last and trust most, which is exactly backwards. In this post we will catalog the eight distinct ways a metric lies, derive the math that makes each one possible, and write the runnable checks that expose them. The figure below is the whole map: eight layers sit between your model's predictions and the number on the screen, and any one of them can flip a true value into a flattering one.

![A vertical stack of the eight distinct ways a training or eval metric can report success while the model is actually worse, from micro-versus-macro averaging down to silent dropped samples](/imgs/blogs/your-metric-is-lying-1.png)

By the end you will be able to take any suspicious metric — the eval number that beats training, the offline F1 that collapsed in production, the accuracy that is high while the rare class fails — and bisect it to one of these eight causes with a confirming test, before you touch a single line of model code. The discipline is the same one that runs through the whole series: **read the instruments, and make-it-fail-small**, except here the instrument *is* the thing under suspicion, so the first move is always to re-derive it by hand and assert your code agrees.

A word on why this corner of debugging is so treacherous. The other five places — data, optimization, model code, numerics, systems — announce their bugs. A NaN crashes the run. A shape mismatch throws an exception. A diverging loss draws an ugly curve. The bug *fights you*, which is unpleasant but honest: you know something is wrong because something is visibly wrong. Evaluation bugs do the opposite. They produce a clean run, a smooth curve, and a number that goes *up*. The bug is camouflaged as success. There is no exception to catch, no NaN to hunt, no divergence to explain — only a green dashboard and a model that quietly fails to do its job once it meets reality. Because the symptom is "good news", the natural response is to stop debugging and start celebrating, which is precisely why a lying metric can survive all the way to production while a NaN never makes it past lunch.

There is a second reason metric bugs are uniquely costly: **they corrupt every decision downstream of them.** A wrong loss value affects this run. A wrong metric affects model selection (you ship the wrong model), hyperparameter search (you tune toward the lie), early stopping (you stop at the wrong epoch), and the go/no-go call for production (you deploy on a number that was never real). One bad metric can quietly steer months of work in the wrong direction, because the metric is the compass and you trusted the compass. This post is about checking the compass.

To keep things concrete, we will carry a single running example through the whole post: a **fraud-detection classifier** on a tabular dataset that is 2% fraud. It is the worst-case habitat for metric lies — heavy imbalance, a tempting accuracy number, a static test set, a production feed that drifts, and a business decision (auto-block vs. human review) that hinges on getting the number right. We will watch the same model's reported quality swing from "0.99, ship it" to "0.78, ship it carefully" to "0.55 recall on fraud, do not auto-block" depending entirely on which lie we fail to catch — without the model ever changing.

If you want the surrounding framework, this post sits inside [the taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) and feeds into [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook). It also leans hard on its siblings: [class imbalance and when accuracy lies](/blog/machine-learning/debugging-training/class-imbalance-and-when-accuracy-lies), [data leakage the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer), and [distribution shift train vs the real world](/blog/machine-learning/debugging-training/distribution-shift-train-vs-the-real-world).

## 1. The first lie: micro vs macro averaging

Start with the most common, most quietly destructive metric bug in multi-class and multi-label problems: **you reported the average over classes, but you chose the wrong kind of average.**

There are two ways to average a per-class metric like F1 across `K` classes, and under imbalance they can disagree by 0.3 or more.

**Micro averaging** pools the raw counts across all classes first, then computes the metric once. For precision and recall:

$$\text{micro-precision} = \frac{\sum_k \text{TP}_k}{\sum_k (\text{TP}_k + \text{FP}_k)}, \qquad \text{micro-recall} = \frac{\sum_k \text{TP}_k}{\sum_k (\text{TP}_k + \text{FN}_k)}$$

Because the counts are pooled, the class with the most examples contributes the most counts, so micro is **dominated by the majority class**. (In single-label multi-class classification, micro-precision equals micro-recall equals accuracy — a useful identity to remember, and a tell that micro-F1 is just accuracy wearing a disguise.)

**Macro averaging** computes the metric per class and then takes an unweighted mean:

$$\text{macro-F1} = \frac{1}{K} \sum_{k=1}^{K} \text{F1}_k, \qquad \text{F1}_k = \frac{2\,\text{TP}_k}{2\,\text{TP}_k + \text{FP}_k + \text{FN}_k}$$

Every class gets weight `1/K` regardless of how rare it is. A class with five examples counts exactly as much as a class with five thousand.

The math of why they diverge is direct. Macro-F1 is the arithmetic mean of the per-class F1 scores; micro-F1 is, effectively, a count-weighted blend that collapses onto the majority. So if your majority class has F1 0.97 and your rare class has F1 0.27, macro-F1 is `(0.97 + 0.27) / 2 = 0.62`, while micro-F1 sits up near 0.95 because almost all the pooled counts come from the easy majority. **One number says "great", the other says "the rare class is broken", and they are computing on the identical predictions.**

To see *why* the gap can be arbitrarily large, push the imbalance. Let the majority class be a fraction `1 − ε` of the data and the rare class a fraction `ε`. As `ε → 0`, the pooled counts in the micro numerator and denominator are utterly dominated by the majority, so micro-F1 → majority-F1, which can sit near 1.0 even if the rare-class F1 is exactly 0. Macro-F1, meanwhile, is `(majority-F1 + rare-F1) / 2`, which is bounded above by `(1 + rare-F1) / 2` — so a rare-class F1 of 0 caps macro at 0.5 no matter how perfect the majority is. The divergence between the two averages is therefore *largest* exactly when you most need the truth: when the rare class is failing under heavy imbalance. Micro is built to hide that failure; macro is built to surface it. This is not a quirk of F1 — the same dominance argument applies to micro-precision, micro-recall, and accuracy, which are all pooled-count metrics. The general rule of thumb: **the heavier the imbalance, the wider the micro-macro gap, and the more dangerous it is to report only micro.**

There is a useful sanity identity to carry around. In single-label multi-class classification (every example has exactly one true class and gets exactly one predicted class), every false positive for one class is simultaneously a false negative for another, so the pooled `Σ FP = Σ FN`, which forces micro-precision = micro-recall = micro-F1 = accuracy. If you ever see "micro-F1" and "accuracy" reported as two different numbers on a single-label problem, one of them is computed wrong — that disagreement is itself a metric bug you can catch by inspection.

![A two-column comparison showing micro-F1 dominated by the majority class reporting 0.95 versus macro-F1 averaging per class and exposing a rare-class F1 of 0.27 for a true 0.62](/imgs/blogs/your-metric-is-lying-2.png)

#### Worked example: a 95-to-5 binary problem

Take 10,000 examples: 9,500 negatives, 500 positives (the thing you care about). The model predicts almost everything negative — it learned the prior, not the signal. Suppose it gets 9,400 of the negatives right and 50 of the positives right.

Confusion counts: TP = 50, FN = 450, FP = 100, TN = 9,400.

- Positive-class precision = `50 / (50 + 100) = 0.333`, recall = `50 / (50 + 450) = 0.10`, F1 = `2·0.333·0.10 / (0.333 + 0.10) = 0.154`.
- Negative-class precision = `9400 / (9400 + 450) = 0.954`, recall = `9400 / (9400 + 100) = 0.989`, F1 = `0.972`.
- **Macro-F1** = `(0.154 + 0.972) / 2 = 0.563`.
- **Micro-F1** (= accuracy here) = `(50 + 9400) / 10000 = 0.945`.

The micro number is **0.945**. If that is what your training loop logs, the dashboard says the model is excellent. The macro number is **0.563**, and the recall on the class you built the model for is **10%** — it is catching one positive in ten. Both numbers are arithmetically correct. Only one of them answers the question you actually have.

The fix is not "always use macro." It is **report per-class, and pick the average that matches the goal**. If you genuinely care about overall correctness in a balanced problem, micro is fine. If you care about every class equally, or about a rare class specifically, you need macro or per-class recall. The bug is reporting *one aggregate* under imbalance and calling it done. For the deeper treatment of why accuracy itself collapses here, see [class imbalance and when accuracy lies](/blog/machine-learning/debugging-training/class-imbalance-and-when-accuracy-lies).

```python
from sklearn.metrics import classification_report, f1_score
import numpy as np

# y_true: 9500 zeros, 500 ones; y_pred from a lazy model
y_true = np.array([0] * 9500 + [1] * 500)
y_pred = np.array([0] * 9400 + [1] * 100 + [1] * 50 + [0] * 450)
# (construction above is illustrative; in practice y_pred comes from your model)

print("micro F1:", f1_score(y_true, y_pred, average="micro"))
print("macro F1:", f1_score(y_true, y_pred, average="macro"))
print("weighted F1:", f1_score(y_true, y_pred, average="weighted"))
print(classification_report(y_true, y_pred, digits=3))
```

The single most important habit here: **never log a bare aggregate F1 without also logging `classification_report` (or the per-class recall vector).** The aggregate is a summary; the per-class breakdown is the evidence. If you only keep one, keep the breakdown.

There is a third averaging mode, `weighted`, which takes the per-class metric and weights each by its support (number of true instances). It is a trap of its own: under imbalance, weighted-F1 also collapses toward the majority, so it can look almost identical to micro and hide the same rare-class failure. If someone says "we used weighted to be fair to class sizes", that is exactly the wrong direction for a minority-class problem.

For multi-label problems (each example can have several labels — tags on an image, topics on a document) the same averaging choice exists and the stakes are higher, because the label cardinality varies wildly across examples and classes. Micro-F1 there pools all label decisions into one giant 2×2 table, so a few high-frequency labels dominate; macro-F1 averages the F1 of each label, so a rare-but-important label (say, a safety-critical tag that appears on 0.1% of documents) gets full weight. The decision is the same: if the rare labels matter, macro or per-label, never micro alone. And there is a multi-label-specific footgun: **sample-averaged F1** (sklearn's `average="samples"`) computes F1 per example over its label set and averages across examples — a fourth number again, useful when you care about per-document tagging quality but yet another way to report a different value on the same predictions. The lesson generalizes past F1: any time a metric offers an `average=` argument, that argument is a decision about *what you care about*, and choosing it by default or by what looks best is how the metric starts lying.

Connect this back to our running fraud example. Fraud is the rare positive class. If the eval harness logs micro-F1 (or, equivalently here, accuracy), it reports a number dominated by the 98% legitimate transactions the model trivially classifies. The fraud class — the entire reason the model exists — contributes almost nothing to the pooled counts. So the dashboard can read 0.95+ while fraud recall is in the single digits, and the only artifact that exposes it is the per-class line in `classification_report` showing `recall = 0.10` on the positive class. That one line is worth more than the headline.

## 2. The second lie: accuracy under imbalance, and the metrics that reveal it

Accuracy is the metric everyone reaches for first and the one that lies most readily, because **accuracy rewards predicting the majority.** The science is a one-line argument. If a fraction `p` of the data is the majority class, the constant classifier "always predict majority" achieves accuracy `p` with zero ability to discriminate anything. At 99-to-1 imbalance, the do-nothing baseline scores **0.99**, so a model reporting 0.99 accuracy might be doing nothing at all. Accuracy has no idea whether your 1% of errors are spread randomly or land entirely on the one class you care about.

The metrics that do not lie under imbalance are the ones that decouple the two classes:

- **Per-class recall** answers "of the actual positives, what fraction did we catch?" — immune to the negative class entirely.
- **Precision-recall AUC (PR-AUC, or average precision)** integrates precision over recall and, crucially, has a baseline equal to the positive rate `p`, not 0.5. A PR curve for a useless model on a 1%-positive problem hugs `y = 0.01`, so any real lift is visible. ROC-AUC, by contrast, has baseline 0.5 regardless of imbalance, which is why ROC-AUC can look reassuringly high (0.85+) on a model that is useless in production — the false-positive *rate* stays low simply because there are so many true negatives to divide by.

The distinction between ROC-AUC and PR-AUC is itself a "your metric is lying" trap. ROC-AUC uses the false-positive *rate* `FP / (FP + TN)`. Under heavy imbalance, `TN` is enormous, so even thousands of false positives barely move the rate, and ROC-AUC stays optimistic. PR-AUC uses precision `TP / (TP + FP)`, which has no `TN` term, so it feels every false positive. **On a rare-positive problem, trust PR-AUC over ROC-AUC.**

Make that quantitative. Suppose at some operating point your model flags 1,000 transactions as fraud, of which 200 are real fraud (precision 0.20) and 800 are false alarms. On our 2%-fraud dataset of 100,000 transactions there are 98,000 legitimate ones, so the false-positive *rate* is `800 / 98,000 = 0.008` — under 1%. ROC, which plots true-positive rate against that false-positive rate, sees a tiny x-coordinate and reports a flattering curve. Precision, which is `200 / 1,000 = 0.20`, sees that four out of five alarms are wrong and reports a grim curve. Both describe the same 1,000 flags. The reason the two metrics disagree so violently is the denominator: ROC divides false positives by the gigantic legitimate population (so they vanish), while precision divides them by the small flagged population (so they dominate). When the cost of a false alarm is borne per-alarm — a human reviews each flag, a customer gets a declined card — precision and PR-AUC are the honest accounting, and ROC-AUC is the lie that makes a 20%-precision model look deployable.

#### Worked example: ROC-AUC 0.91, PR-AUC 0.34

A churn model on a dataset that is 3% churners. You score ROC-AUC 0.91 and feel good. Then you compute PR-AUC and it is 0.34. The PR-AUC baseline (a random model) is the positive rate, 0.03, so 0.34 is real lift — but it is nowhere near the "0.91" your ROC number implied. At a threshold that catches 60% of churners (recall 0.60), precision is 0.22: nearly four out of five flagged customers are not actually churning. If the business sends a retention offer to every flagged customer, four-fifths of the offers are wasted. The ROC number said "deploy"; the PR number said "this will cost you." Same predictions. The metric you chose decided the business outcome.

```python
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
import numpy as np

# scores: model's predicted probability for the positive (churn) class
# y_true: 3% positives
auc = roc_auc_score(y_true, scores)
ap = average_precision_score(y_true, scores)      # PR-AUC / average precision
print(f"ROC-AUC {auc:.3f}   PR-AUC {ap:.3f}   base rate {y_true.mean():.3f}")

# threshold-dependent metrics: report recall at a fixed operating point
for thr in (0.3, 0.5, 0.7):
    pred = (scores >= thr).astype(int)
    print(f"thr {thr}: recall {recall_score(y_true, pred):.3f}")
```

The takeaway that ties this to averaging: **accuracy and micro-F1 are the same lie wearing two hats.** Both pool over the classes and both collapse onto the majority. The cure is the same: per-class recall plus a threshold-free curve (PR-AUC) whose baseline reflects the imbalance.

## 3. The third lie: eval-train skew

This one is subtle because the metric is computed correctly — the bug is that the *inputs* the metric sees were processed differently than the inputs the model trained on. The model is fine; the eval pipeline is feeding it something slightly off, and the number measures the mismatch, not the model.

The classic instances:

- **Normalization mismatch.** Training normalized images with ImageNet mean/std on a 0-1 scale; eval forgot to divide by 255 first, so it normalizes on a 0-255 scale. The model receives inputs ten to a hundred times larger than it ever saw. Accuracy on eval craters even though the model is perfect.
- **Tokenization mismatch.** Training added a BOS token; the eval harness loads the tokenizer with `add_bos_token=False`. Every eval sequence is shifted by one token from what the model expects. Perplexity inflates; downstream accuracy drops; the model is untouched. (This is a cousin of the [tokenization bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) family — verify the eval tokenizer config byte-for-byte against training.)
- **Format/template mismatch.** A chat model trained on `<|user|> ... <|assistant|> ...` is evaluated on raw prompts with no template. The model has never seen this format; it underperforms, and the eval blames the model.
- **Label-space mismatch.** Training used label index 0 = "cat"; the eval set was built from a different export where 0 = "dog". Every prediction is "wrong" by a fixed permutation, and accuracy is near-random regardless of how good the model is.

![A dataflow graph where a shared raw sample splits into a correct train normalization branch and a wrong eval normalization branch, feeding the same frozen weights to produce a train metric of 0.92 and an eval artifact of 0.71](/imgs/blogs/your-metric-is-lying-4.png)

The science here is just function composition. Your model is a function `f`, but the metric measures `f ∘ g_eval`, where `g_eval` is the eval preprocessing. Training optimized `f ∘ g_train`. If `g_eval ≠ g_train`, the metric is evaluating a *different composed function* than the one you trained, and the gap `metric(f ∘ g_train) − metric(f ∘ g_eval)` is pure skew — it has nothing to do with model quality. The number is not lying about `f ∘ g_eval`; it is lying about `f`, because you wanted to know about `f`.

The diagnostic is the most powerful single check in this whole post, and it is brutally simple: **feed one training example through the eval pipeline and assert the tensors are bit-identical to the training pipeline.**

```python
import torch

def assert_pipelines_match(raw_sample, train_pipeline, eval_pipeline, atol=1e-6):
    """Run one raw sample through both pipelines; assert identical tensors."""
    xt = train_pipeline(raw_sample)
    xe = eval_pipeline(raw_sample)
    assert xt.shape == xe.shape, f"shape skew: train {xt.shape} vs eval {xe.shape}"
    assert xt.dtype == xe.dtype, f"dtype skew: {xt.dtype} vs {xe.dtype}"
    max_diff = (xt.float() - xe.float()).abs().max().item()
    assert max_diff < atol, (
        f"value skew: max |train - eval| = {max_diff:.3e}\n"
        f"  train range [{xt.min():.3f}, {xt.max():.3f}]\n"
        f"  eval  range [{xe.min():.3f}, {xe.max():.3f}]"
    )
    print(f"pipelines match: max_diff {max_diff:.3e}")

# For LLMs, assert the exact token ids match what the trainer produced:
ids_train = train_tokenizer(text, add_special_tokens=True)["input_ids"]
ids_eval = eval_tokenizer(text, add_special_tokens=True)["input_ids"]
assert ids_train == ids_eval, f"token skew:\n train {ids_train}\n eval  {ids_eval}"
```

When `max_diff` comes back as `2.5e+02` instead of `0`, you have found your bug in one second, and you found it *before* retraining anything. The value-range printout in the assertion message usually tells you the exact cause: a 0-255 vs 0-1 scale, a sign flip from a wrong mean, a transposed channel order.

Why is this the single most powerful check in the post? Because it converts an abstract suspicion ("the eval number seems wrong") into a concrete, falsifiable claim ("these two tensors must be identical") that a one-line assertion settles in microseconds. Most metric debugging is slow because the feedback loop is slow — change something, retrain, wait, look at the number, guess again. The pipeline-match assertion has a feedback loop of *one sample and one comparison*. It needs no training, no GPU, and no eval set; it needs one raw input and both code paths. That is make-it-fail-small applied to the eval harness: shrink the problem until the bug is forced into the open on a single example. Run it first, before any of the statistical checks, because if the pipelines disagree, none of the downstream metrics mean anything anyway.

A practical note on what "the eval pipeline" includes, because skew hides in the parts people forget. It is not just normalization. It is resize interpolation (bilinear in training, nearest in eval), anti-aliasing on or off, channel order (RGB vs BGR — the cv2-vs-PIL classic), dtype and casting order (cast-then-normalize vs normalize-then-cast changes rounding), center-crop vs resize, the exact mean/std constants, and for text: the tokenizer version, the special-token settings, truncation side, max length, and whether a chat template is applied. Each of these is a silent transform between the raw input and the tensor the model sees, and any divergence between train and eval is skew. The assertion above catches all of them at once because it compares the final tensors, not the intermediate steps — you do not need to know *which* transform diverged to know *that* one did.

#### Worked example: the 0.71 that was really 0.92

A teammate reports a vision finetune at 0.71 top-1 and concludes the model is undertrained. You run `assert_pipelines_match` on a single image and it fails: `max_diff = 247.3`, train range `[-2.1, 2.6]`, eval range `[-115.0, 140.0]`. The eval transform normalized before scaling to 0-1. You fix the eval transform, rerun, and top-1 is **0.92** — the model was fine all along. Nobody retrained. The "undertrained model" was a one-line preprocessing bug in the eval harness, and three days of planned hyperparameter sweeps evaporated.

This is why eval-train skew belongs in the "before you touch model code" category. The symptom — eval worse than expected — screams "train the model harder", and that instinct will waste a week.

## 4. The fourth lie: leakage in the eval set

Eval skew makes the number too low. **Leakage makes it too high**, and it is more dangerous because nobody questions good news. Leakage means information that will not be available at prediction time has leaked into the eval set, so the metric is optimistic — sometimes wildly so.

Three mechanisms, each with a clean signature:

1. **Train-test contamination (duplicates).** Some eval rows are exact or near-duplicate copies of training rows. The model has memorized them, so it scores them perfectly. The eval metric measures memorization, not generalization. This is endemic in scraped datasets and in any pipeline that shuffles before splitting at the wrong granularity (splitting individual frames of the same video, or augmented copies of the same image, across train and test).
2. **A leaked feature.** A column in the eval set is a proxy for the label that would not exist at inference. The textbook case: a `account_closed_date` feature in a churn dataset — it is populated only for customers who already churned, so the model "predicts" churn by reading the answer. AUC goes to 0.99 and is meaningless.
3. **Eval contamination in pretraining.** For LLMs, the benchmark's test questions appear in the pretraining corpus. The model has effectively seen the answer key. The benchmark number is inflated, sometimes by tens of points.

The science of *how much* a leak inflates a metric is worth making precise, because it explains the telltale signature. Suppose a fraction `λ` of your eval rows are leaked (duplicated from train, or carry a leaked feature) and on those rows the model scores essentially perfectly, while on the honest `1 − λ` fraction it scores its true value `m_true`. The reported metric is approximately the mixture:

$$m_{\text{reported}} \approx \lambda \cdot 1.0 + (1 - \lambda) \cdot m_{\text{true}}$$

So if the true value is 0.78 and 20% of the eval set is leaked, you report `0.2·1.0 + 0.8·0.78 = 0.964`. **A modest 20% contamination turns a 0.78 model into a reported 0.96** — exactly the kind of "we beat the baseline by a lot" jump that gets celebrated instead of investigated. Rearranging gives you a back-of-envelope leak estimate: if your eval number is `m_reported` and you suspect the honest number is `m_true`, the implied leak fraction is `λ ≈ (m_reported − m_true) / (1 − m_true)`.

The diagnostic has three parts, one per mechanism.

```python
import hashlib, numpy as np

# (1) Exact/near-duplicate detection across splits.
def row_hash(row):  # hash the input features only, never the label
    return hashlib.md5(np.ascontiguousarray(row).tobytes()).hexdigest()

train_hashes = {row_hash(x) for x in X_train}
overlap = [i for i, x in enumerate(X_eval) if row_hash(x) in train_hashes]
print(f"exact eval rows also in train: {len(overlap)} / {len(X_eval)} "
      f"({100*len(overlap)/len(X_eval):.1f}%)")

# (2) Leaked-feature smell test: a single feature that is near-perfect alone.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
for j, name in enumerate(feature_names):
    clf = LogisticRegression(max_iter=1000).fit(X_train[:, [j]], y_train)
    auc = roc_auc_score(y_eval, clf.predict_proba(X_eval[:, [j]])[:, 1])
    if auc > 0.95 or auc < 0.05:
        print(f"SUSPICIOUS leak: feature '{name}' alone gives AUC {auc:.3f}")
```

For near-duplicates in images/text, hash on a perceptual or embedding representation (e.g. cosine similarity > 0.98 in an embedding space) rather than exact bytes, because the dangerous duplicates are the *almost* identical ones — a re-encoded JPEG, a paraphrase, a cropped frame.

#### Worked example: AUC 0.97 to 0.78 after dedup

A fraud model reports test AUC 0.97. The duplicate check finds that 14% of test transactions share an exact feature hash with training rows (the same merchant-batch was exported into both splits). You remove the leaked rows and rerun: AUC **0.78**. Plugging into the mixture formula, `λ ≈ (0.97 − 0.78) / (1 − 0.78) = 0.86` of the *gap* was leak — consistent with a heavily-memorized duplicate set. The honest model is still useful, but the decision changes completely: 0.97 said "ship and trust it for auto-blocking", 0.78 says "ship with a human-in-the-loop review queue." For the full taxonomy of leak types and how to plug each, this is the dedicated companion: [data leakage the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer).

The general rule: **whenever a metric jumps suspiciously, suspect leakage before you suspect genius.** A 5-point improvement from a sensible change is plausible. A 15-point jump from a small change is almost always a leak, a skew, or an averaging bug.

It is worth dwelling on the *signatures* that distinguish the leak types, because the fix differs. A **duplicate leak** has a tell: the model scores the leaked rows near-perfectly and the honest rows at its true rate, so the per-example score distribution is bimodal — a spike near 1.0 (the memorized duplicates) and a broader mode at the real performance. Plot the per-example loss histogram and a sharp spike at near-zero loss on eval, when the rest of eval is spread out, is the fingerprint of contamination. A **feature leak** has a different tell: a single feature, used alone, achieves near-perfect AUC (the smell test in the code above), and its feature importance dwarfs everything else by an implausible margin. A **temporal leak** (using future information to predict the past) shows up as a metric that is excellent on a random split and collapses on a time-ordered split — which is exactly why time-series problems demand `TimeSeriesSplit` and never a random shuffle.

The deeper reason leaks are so common is *granularity*. The unit you split on must be the unit that is independent in production. If you split individual transactions but fraud comes in bursts from the same compromised account, then splitting transactions puts the same account's transactions in both train and test, and the model memorizes the account, not the fraud pattern. The fix is `GroupKFold` on the account id so that no account spans the boundary. The same logic governs patients in medical data, users in recommendation, videos in frame-level vision, and documents in sentence-level NLP. A metric computed on a wrongly-granular split is honest about that split and a lie about deployment, where the groups it memorized will never recur. The cross-validation companion in this series goes deep on getting the split right; here the point is that the leak makes the *metric* the lie.

## 5. The fifth lie: the metric that doesn't match the goal

Now a different category of lie — the metric is computed correctly, on the right data, with no skew or leak, and it *still* misleads, because **it measures something adjacent to what you actually care about.** This is the most insidious lie because there is no bug to find. The code is right. The choice is wrong.

Three common mismatches:

**Optimizing a proxy you don't care about.** You train a recommender to minimize logloss (calibrated probability error), but the product only ever shows the top 5 items — what matters is **NDCG@5 or recall@5**, a ranking metric. A model can improve logloss by becoming better-calibrated on the long tail of items nobody sees while getting *worse* at ordering the top few. Logloss goes down, the metric you reported, while the metric the user feels goes down too. You optimized the wrong objective and the wrong report agreed with you.

**Threshold-dependent metrics reported at the wrong threshold.** F1, precision, recall, and accuracy all depend on a decision threshold applied to the model's scores. The default 0.5 is arbitrary and almost never optimal under imbalance. A model can have excellent ranking (high PR-AUC) but terrible F1 at 0.5 simply because the right threshold is 0.12. If you report F1 at 0.5 and compare it to another model whose scores happen to be better-calibrated around 0.5, you can conclude the wrong model is better — when the difference is entirely a threshold artifact. The honest comparison is either a threshold-free metric (PR-AUC, ROC-AUC) or F1 at the threshold tuned on a *separate* validation split.

**Surface-form metrics that don't track task quality.** In generation, BLEU and ROUGE measure n-gram overlap with a reference; perplexity measures next-token likelihood. None of them measures whether the answer is *correct, helpful, or coherent*. A summarization model can raise ROUGE by copying long spans verbatim (high overlap, low usefulness). A chat model can lower perplexity by becoming more confidently generic. Perplexity dropping does not mean the model got better at the task — it means it got better at predicting the next token in the *distribution it was trained on*, which can diverge from instruction-following quality after RLHF. These metrics are correlated with quality at coarse resolution and uncorrelated (or anti-correlated) at fine resolution, which is exactly where you make model-selection decisions.

#### Worked example: BLEU up, humans say worse

A translation finetune raises BLEU from 31.2 to 33.8 — a clear, publishable +2.6. A human side-by-side eval on 200 sentences prefers the *old* model 58% of the time. What happened? The finetune learned to match reference phrasing more literally (more n-gram overlap, higher BLEU) while producing stiffer, less fluent output that humans dislike. BLEU rewarded the literalism; humans penalized it. The metric went up; the model got worse on the only axis that matters. The fix is not to abandon BLEU but to **treat it as a cheap proxy and gate releases on a small human or LLM-judge eval** that measures the real goal. The 2.6 BLEU points were real and irrelevant.

```python
# Picking the metric that matches the goal: a ranking task wants a ranking metric.
from sklearn.metrics import ndcg_score, log_loss
import numpy as np

# y_true_rel: relevance grades per item; scores: model probabilities
ll = log_loss(y_true_binary, scores)                     # what we trained on
ndcg5 = ndcg_score(y_true_rel[None, :], scores[None, :], k=5)  # what we care about
print(f"logloss {ll:.4f}   NDCG@5 {ndcg5:.4f}")

# Threshold-free vs threshold-at-0.5: report both so the threshold artifact is visible
from sklearn.metrics import average_precision_score, f1_score
pr_auc = average_precision_score(y_true_binary, scores)
f1_at_half = f1_score(y_true_binary, (scores >= 0.5).astype(int))
# tune threshold on a SEPARATE split, then report f1 at that threshold
best_thr = max(np.linspace(0.05, 0.95, 19),
               key=lambda t: f1_score(y_val, (scores_val >= t).astype(int)))
f1_tuned = f1_score(y_true_binary, (scores >= best_thr).astype(int))
print(f"PR-AUC {pr_auc:.3f}   F1@0.5 {f1_at_half:.3f}   "
      f"F1@{best_thr:.2f} {f1_tuned:.3f}")
```

The principle: **the metric you optimize and the metric you report should both be the metric you care about — and if they can't be (because the real metric is expensive, like human eval), the cheap proxy must be validated against the expensive one before you trust it.**

The threshold trap deserves its own emphasis because it is responsible for an enormous fraction of "model B beats model A" conclusions that reverse on inspection. Here is the precise mechanism. Two models can have the *identical* PR curve — identical ranking ability — and yet report different F1 at threshold 0.5 simply because their score *calibration* differs: model A's scores cluster around 0.5, model B's cluster around 0.2, so the same 0.5 cut lands at different points on each model's curve. If you compare F1@0.5, you are comparing two arbitrary operating points, not two models. The honest comparison is threshold-free (compare the whole PR-AUC) or threshold-matched (tune each model's threshold on a held-out validation split, then report both at their own best threshold). Reporting F1@0.5 for a model selection decision is, in effect, letting score calibration cast the deciding vote — and calibration is the thing temperature scaling fixes in an afternoon, so it should never decide which model ships.

On our fraud example, this is not academic. At threshold 0.5 the fraud model might flag almost nothing (its fraud scores rarely exceed 0.5 because fraud is rare and the model is appropriately conservative), giving F1@0.5 near zero and the conclusion "the model is useless." Move the threshold to 0.08 — tuned on validation to balance precision and recall for the business — and the same model has a perfectly workable F1. The model did not change; the reported number swung from "useless" to "useful" purely on threshold. Reporting F1 at the default threshold on an imbalanced problem is one of the most common ways a good model gets killed by a lying metric.

## 6. The sixth lie: the offline-online gap

You did everything right offline — correct averaging, no leak, no skew, the right metric — and the model still underperforms in production. This is the offline-online gap, and it is the lie that survives all the previous checks because it lives in the gap between your static test set and the live world.

![A two-column contrast of an offline report built on a static test set with a proxy label giving F1 0.91 versus a live and holdout-replay evaluation on logged inputs with the real delayed outcome giving 0.74](/imgs/blogs/your-metric-is-lying-8.png)

There are three distinct sources, and they need different fixes.

**Distribution shift.** Your test set was sampled from yesterday's distribution; production sees today's. If the input distribution drifts (covariate shift) or the label distribution drifts (prior shift), the offline metric is measuring performance on a population that no longer exists. The offline number is honest about the old distribution and a lie about the current one. The science: a metric is an expectation over a distribution, `m = E_{x∼P}[score(f(x), y)]`. Offline you estimate `E_{x∼P_test}`; production realizes `E_{x∼P_prod}`. When `P_test ≠ P_prod`, the two expectations differ, and no amount of offline rigor closes the gap — you are estimating the wrong integral. The fix is monitoring drift and refreshing the test set; the dedicated treatment is [distribution shift train vs the real world](/blog/machine-learning/debugging-training/distribution-shift-train-vs-the-real-world).

**Train-serve skew.** The features are computed differently in the offline pipeline (batch, with full history) than in the online serving path (streaming, with whatever is available at request time). A feature that is trivial to compute offline — "average spend over the last 30 days" — might be computed with a different window, a different join, or stale data online. This is eval-train skew (section 3) relocated to the serving boundary, and the same diagnostic applies: **log the exact feature vector the model received in production, replay it through the offline pipeline, and assert they match.** When they don't, you have found train-serve skew and the offline metric was lying because it never saw the features production actually computes.

**The metric never captured the real outcome.** Offline you scored against a proxy label — "did the user click?" — but the business goal is "did the user stay subscribed?" A model can raise clicks while lowering retention (clickbait). The offline metric improves; the real outcome degrades. There is no preprocessing bug; the label itself was a proxy. The only fix is to instrument the *real* outcome (even if delayed) and validate that the proxy correlates with it before trusting offline gains.

The diagnostic that exposes all three is **holdout replay**: take a sample of real production traffic with its eventual true labels, run your model on the exact logged inputs, and compute the metric on that. This is the closest you can get to the truth offline.

```python
# Holdout replay: score the model on logged production inputs with eventual labels.
import pandas as pd
from sklearn.metrics import f1_score

logged = pd.read_parquet("prod_requests_with_outcomes.parquet")
# inputs are EXACTLY what serving sent the model; labels arrived later
X_live = featurize_offline(logged["raw_request"])      # offline pipeline
preds = model.predict(X_live)
live_f1 = f1_score(logged["true_outcome"], preds)
print(f"offline test F1 {offline_f1:.3f}   live-replay F1 {live_f1:.3f}   "
      f"gap {offline_f1 - live_f1:+.3f}")

# Train-serve skew check: do offline features match what serving logged?
served = logged[feature_cols].to_numpy()
recomputed = X_live[feature_cols].to_numpy()
skew = np.abs(served - recomputed).max(axis=0)
for name, s in zip(feature_cols, skew):
    if s > 1e-6:
        print(f"FEATURE SKEW '{name}': max |served - offline| = {s:.4g}")
```

#### Worked example: 0.91 offline, 0.74 live, decision reversed

A content-moderation model reports F1 0.91 offline. Holdout replay on a week of logged traffic gives F1 0.74. The feature-skew check finds nothing (features match), and the per-class report shows recall on the rare "harmful" class dropped from 0.88 offline to 0.55 live. The cause is distribution shift: the offline test set predates a new spam campaign with phrasing the model never saw. The decision flips: 0.91 said "raise the auto-action threshold and reduce human review", 0.74 with 0.55 harmful-recall says "keep humans in the loop and retrain on recent data weekly." The offline number was not buggy. It was answering a question about last month.

The science of why offline rigor cannot save you here is worth stating plainly, because it is the part people resist. A metric is an *expectation over a distribution*. Offline you compute a sample estimate of `E_{x∼P_test}[score]`; production realizes `E_{x∼P_prod}[score]`. These are estimates of two different integrals whenever `P_test ≠ P_prod`. No amount of careful averaging, leak removal, or threshold tuning on `P_test` changes the fact that you measured the wrong distribution — you can compute a perfect estimate of the wrong number. This is the formal reason the offline-online gap is not a "bug" you fix in the eval code; it is a mismatch you can only close by changing *which distribution* you evaluate on, which means continuously refreshing the test set from production and monitoring drift. The most honest offline metric you can build is holdout replay on recent logged traffic, because it pulls `P_test` as close to `P_prod` as the data latency allows.

The three sources also have different latencies of discovery, which matters operationally. **Train-serve skew** is discoverable *immediately* — the moment you have one logged production feature vector, you can replay it and assert it matches the offline pipeline; there is no excuse for shipping with feature skew because the check costs one request. **Distribution shift** is discoverable only as the world changes — it can be zero at launch and grow over weeks, which is why it needs a monitor, not a one-time check. **A proxy-label mismatch** is the slowest of all, because the real outcome (did the user stay subscribed? did the flagged transaction turn out fraudulent?) arrives with a delay of days or weeks, so the lie is invisible until the delayed labels come back and you can finally compute the metric you actually cared about. Build for all three: a serving-feature assertion at deploy, a drift monitor in production, and a delayed-label backfill that recomputes the real-outcome metric once the truth arrives.

## 7. The seventh lie: per-batch averaging of non-decomposable metrics

Here is a bug that hides in your own training loop and produces a number that is wrong by construction, no matter how clean your data is. **You averaged the metric over batches, but the metric is not decomposable, so the average of per-batch values is not the global value.**

The science is the heart of this section. A metric is **decomposable** (more precisely, an average of per-sample contributions) if it can be written as a mean over samples: accuracy is `(1/N) Σ 1[ŷᵢ = yᵢ]`, and mean-squared error is `(1/N) Σ (ŷᵢ − yᵢ)²`. For these, the mean of per-batch means *is* the global mean (as long as batches are equal-sized — even that breaks for unequal batches, see below). You can stream them safely.

F1, precision, recall, and AUC are **not** of this form. F1 is a *nonlinear function of pooled counts*:

$$\text{F1} = \frac{2\,\text{TP}}{2\,\text{TP} + \text{FP} + \text{FN}}$$

This is a ratio. The average of ratios is not the ratio of sums:

$$\frac{1}{B}\sum_{b=1}^{B} \frac{2\,\text{TP}_b}{2\,\text{TP}_b + \text{FP}_b + \text{FN}_b} \;\neq\; \frac{2\sum_b \text{TP}_b}{2\sum_b \text{TP}_b + \sum_b \text{FP}_b + \sum_b \text{FN}_b}$$

The left side (per-batch averaging) and the right side (global) are different numbers, and the gap grows when batches are small (each per-batch ratio is noisy) and when the positive rate varies across batches. The same is true for AUC, which depends on the *global ranking* of all scores — it is literally undefined within a single batch if that batch contains only one class, and averaging per-batch AUCs is meaningless because the ranking that matters is across the whole eval set, not within each mini-batch.

![A two-column comparison of computing F1 per batch and averaging giving an optimistic 0.81 versus accumulating true and false positive counts globally and computing one F1 at the end giving the correct 0.74](/imgs/blogs/your-metric-is-lying-5.png)

This bug is everywhere because the naive training loop *invites* it. You compute a metric inside the batch loop, append it to a list, and report the mean. For loss that is fine. For F1 it is wrong. The symptom is a metric that is **optimistically biased and jittery** — it changes when you change batch size, which is the tell, because a correct global metric is invariant to how you chunk the eval set.

```python
# WRONG: per-batch F1, then averaged. Optimistic and batch-size dependent.
from sklearn.metrics import f1_score
import numpy as np

batch_f1s = []
for xb, yb in eval_loader:
    pred = model(xb).argmax(1)
    batch_f1s.append(f1_score(yb.cpu(), pred.cpu(), average="macro"))
wrong = float(np.mean(batch_f1s))          # the lie

# RIGHT: accumulate predictions (or counts), compute F1 once at the end.
all_y, all_p = [], []
for xb, yb in eval_loader:
    pred = model(xb).argmax(1)
    all_y.append(yb.cpu()); all_p.append(pred.cpu())
right = f1_score(np.concatenate(all_y), np.concatenate(all_p), average="macro")
print(f"per-batch averaged F1 {wrong:.3f}   global F1 {right:.3f}")
```

The right way to do this at scale, without holding every prediction in memory, is a **stateful metric** that accumulates the sufficient statistics (TP, FP, FN per class for F1; or the full set of scores for AUC) across batches and computes the final value once. This is exactly what `torchmetrics` does, and why you should use it instead of rolling your own per-batch average:

```python
import torchmetrics

# Stateful: .update() accumulates counts per batch; .compute() returns the GLOBAL metric.
f1 = torchmetrics.classification.MulticlassF1Score(num_classes=K, average="macro")
auroc = torchmetrics.classification.MulticlassAUROC(num_classes=K, average="macro")

f1.reset(); auroc.reset()
for xb, yb in eval_loader:
    logits = model(xb)
    f1.update(logits.argmax(1), yb)        # accumulates TP/FP/FN
    auroc.update(logits.softmax(1), yb)    # accumulates scores for global ranking
print(f"global macro-F1 {f1.compute():.3f}   global macro-AUROC {auroc.compute():.3f}")
```

#### Worked example: the eval that improved by changing batch size

A team reports macro-F1 0.81 and a competing team, same model, reports 0.77. The "difference" is real — and it is entirely a bug. Team A averaged per-batch F1 with batch size 16; team B did it with batch size 64. Smaller batches make each per-batch F1 noisier and, because F1 is a concave-ish function of the counts near the operating point, the per-batch average is biased *upward* — so the smaller-batch team reported a higher number for the *identical model*. Recomputing globally with `torchmetrics`, both get **0.74**. The 0.07 spread was an artifact of how each team chunked the eval set. Nobody had a better model; one team had a more flattering bug.

There is a quieter cousin: **unequal-batch averaging even for decomposable metrics.** If your last batch is partial (say 7 samples when the rest are 64) and you average per-batch accuracy with equal weight, the partial batch is over-weighted. For accuracy the fix is a weighted mean by batch size, or — simpler and always correct — accumulate `correct` and `total` counters and divide once at the end. The rule generalizes: **accumulate sufficient statistics, compute the metric once.** Never average the metric itself across batches unless you have proven it is decomposable *and* your batches are equal-sized.

This bug also reappears, in a nastier form, under distributed evaluation. When you evaluate across N GPUs with `DistributedDataParallel`, each rank sees a shard of the eval set. If you compute the metric per-rank and then average the N per-rank metrics, you have committed the per-batch averaging bug at the rank level — for F1 or AUC that average is wrong, and it also silently ignores the fact that the distributed sampler often *pads* the last rank's shard with duplicate samples to make the shards equal length, double-counting some examples. The correct pattern is to `all_gather` the per-rank *sufficient statistics* (TP/FP/FN counts, or the raw predictions and scores), then compute the global metric once on rank 0 after de-duplicating the sampler padding. `torchmetrics` handles the gather-and-compute correctly when you pass `dist_sync_on_step` / use it inside a distributed context, which is the strongest argument for not hand-rolling distributed metric reduction. The signature of getting this wrong is a metric that changes with the number of GPUs — the same invariance violation as changing with batch size, just at a different axis. If your eval number depends on how you *parallelized* the eval, it is not measuring the model.

One more subtle case: **macro-averaging interacts with per-batch averaging multiplicatively.** Macro-F1 averages over classes; per-batch averages over batches. If you do both wrong — compute macro-F1 within each batch and average across batches — small batches may not even contain every class, so the per-batch macro-F1 is computed over a *different set of classes* each batch (or with `zero_division` filling in absent classes), and the across-batch average is doubly meaningless. The only correct order is: accumulate per-class TP/FP/FN across all batches, compute per-class F1 once globally, then average across classes. Compute globally first, average across classes last; never the reverse, never per batch.

## 8. The eighth lie: silent NaN and dropped samples

The final lie is the most mechanical and the easiest to miss: **some samples silently fell out of the average, or NaN/inf values poisoned it, so the number is computed over the wrong set.**

The mechanisms:

- **NaN propagation.** A single NaN in a per-sample loss or score makes `np.mean` return NaN — but if you use `np.nanmean` "to be safe", you have now *silently dropped* every NaN sample, and your average is over a smaller, possibly biased subset. The metric looks clean (no NaN) but it is no longer measuring what you think; it is measuring the samples that happened not to produce NaN, which are often the *easy* ones.
- **Dropped last batch.** `DataLoader(drop_last=True)` is correct for training (stable batch stats) and *wrong* for evaluation — it silently discards up to one batch worth of eval samples. If your eval set is small or the dropped batch is non-random (e.g. the last batch is the rare-class tail because you sorted by length), the metric is computed over a biased subset.
- **`ignore_index` and masking miscount.** In sequence tasks you mask padding with `ignore_index=-100`. If your metric counts the masked positions as correct (or includes them in the denominator), the number is inflated or deflated by however many pad tokens there are — which varies with batch composition. A token-accuracy that counts pad positions as "correct" can read 0.98 while real-token accuracy is 0.71.
- **Empty-class division.** Macro-F1 over a class that has zero true and zero predicted instances is `0/0`. sklearn's `zero_division` parameter decides whether that becomes 0, 1, or NaN, and the default (warn, treat as 0) silently drags macro-F1 down — or, set to 1, silently inflates it. Either way the reported macro-F1 depends on a parameter most people never set.

The science is just careful bookkeeping of the denominator. A metric is `numerator / denominator` over some set `S`. Every one of these bugs changes `S` (drops samples, adds pad tokens, removes NaN rows) without telling you, so the metric is honest about `S` and dishonest about the set you meant. The cure is to **assert the count of samples the metric saw equals the count you expected.**

The reason this lie is so easy to commit is that the tools are *trying to be helpful* and the help is silent. `np.nanmean` exists to "not crash on NaN", which sounds responsible, but its behavior is to drop the NaN entries and average the rest — a silent subset selection dressed up as robustness. `DataLoader`'s `drop_last=True` exists to keep batch shapes uniform, which is genuinely useful in training, but on eval it silently discards real samples. sklearn's `zero_division` exists so empty classes do not throw, but its choice (0 or 1) silently moves macro-F1. In every case the library made a defensible default choice and never told you it changed your denominator. The defense is not to distrust the libraries; it is to *make the denominator explicit and assert on it*, so that any silent change becomes a loud failure. A metric you cannot account for sample-by-sample is a metric you cannot trust.

The NaN case has a particularly nasty bias structure worth understanding. NaN-producing samples are rarely random — they correlate with whatever caused the NaN. In a loss, a NaN often comes from a confidently-wrong prediction (a `log(0)` when the model assigned probability ~0 to the true class) or a degenerate input. Those are exactly the *hard* examples. So `nanmean` does not drop a random sample; it drops the hardest samples, which biases the metric *upward* — the average looks better precisely because the cases the model fails on hardest fell out of the denominator. A metric that silently drops its own worst cases is the most flattering lie of all, and it is one line of `nanmean` away.

```python
import numpy as np

def safe_metric_with_audit(per_sample_scores, expected_n):
    arr = np.asarray(per_sample_scores, dtype=float)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    n_valid = int(np.isfinite(arr).sum())
    # The audit: did we silently lose samples?
    assert n_valid + n_nan + n_inf == len(arr), "length accounting is wrong"
    if n_nan or n_inf:
        print(f"WARNING: {n_nan} NaN, {n_inf} inf in metric inputs "
              f"(out of {len(arr)}) -- these will be DROPPED if you nanmean")
    assert len(arr) == expected_n, (
        f"sample count mismatch: metric saw {len(arr)} but expected "
        f"{expected_n} -- check drop_last / ignore_index / a crashed batch")
    return float(np.nanmean(arr)), {"n_valid": n_valid, "n_nan": n_nan, "n_inf": n_inf}

value, audit = safe_metric_with_audit(losses, expected_n=len(eval_dataset))
print(f"metric {value:.4f}  audit {audit}")
```

For token-level metrics, mask explicitly and assert the masked count is what you expect:

```python
import torch

def masked_token_accuracy(logits, labels, ignore_index=-100):
    preds = logits.argmax(-1)
    mask = labels != ignore_index
    n_real = int(mask.sum())
    n_pad = int((~mask).sum())
    correct = int((preds[mask] == labels[mask]).sum())
    # audit: real + pad must equal total, and we never count pad as correct
    assert n_real + n_pad == labels.numel(), "mask accounting wrong"
    acc = correct / max(n_real, 1)
    return acc, {"n_real": n_real, "n_pad": n_pad}

acc, info = masked_token_accuracy(logits, labels)
print(f"real-token acc {acc:.3f}  (counted {info['n_real']} real, "
      f"ignored {info['n_pad']} pad)")
```

#### Worked example: token accuracy 0.98 that was 0.71

A finetune logs token accuracy 0.98 and everyone is thrilled. The masked-accuracy audit reveals the metric was dividing correct predictions by *all* positions including padding — and 38% of positions were pad tokens the model trivially predicts. Counting only real tokens (`labels != -100`), accuracy is **0.71**. The 0.98 was 0.71 padded with free wins. The fix is one line — apply the mask before counting — and the honest 0.71 is what you compare against the baseline. This is the same `-100` masking that bites in [the loss masking bug](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs); the metric inherits the masking discipline of the loss.

## 9. The master diagnostic: re-derive the metric by hand

Every section above has its own confirming test, but they share one root technique that is the most reliable thing in this entire post: **compute the metric by hand on a tiny example with known labels, and assert your code reproduces it.** If your code disagrees with the hand calculation, you have found a bug in the metric computation itself — before any of the subtler skew/leak/averaging issues even matter.

![A left-to-right timeline of the definitive metric check: pick six labeled rows, hand-count true and false positives, compute F1 on paper, run the same rows through your code, and assert the two agree](/imgs/blogs/your-metric-is-lying-6.png)

The reason this works is that metric bugs hide in the gap between *the metric you think you computed* and *the metric your code computed*. A six-row example is small enough to compute on paper in 30 seconds and rich enough to expose the common bugs: wrong averaging mode, wrong threshold, swapped precision/recall, label-space off-by-one, an inverted positive class.

```python
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

# Tiny known example: 6 rows. POSITIVE class is 1.
y_true = np.array([1, 1, 1, 0, 0, 0])
y_pred = np.array([1, 1, 0, 1, 0, 0])
# By hand: TP=2 (rows 0,1), FN=1 (row 2), FP=1 (row 3), TN=2 (rows 4,5)
#   precision = 2/(2+1) = 0.6667
#   recall    = 2/(2+1) = 0.6667
#   F1        = 2*0.6667*0.6667/(0.6667+0.6667) = 0.6667
HAND_F1 = 2 * 2 / (2 * 2 + 1 + 1)   # = 0.6667

code_f1 = f1_score(y_true, y_pred, pos_label=1)
assert abs(code_f1 - HAND_F1) < 1e-9, (
    f"metric code disagrees with hand calc: code {code_f1:.4f} vs hand {HAND_F1:.4f}\n"
    f"  -> check average mode, pos_label, threshold, label encoding")
print(f"F1 verified: code {code_f1:.4f} == hand {HAND_F1:.4f}")

# Also pin precision and recall so a swap can't hide inside a correct F1.
assert abs(precision_score(y_true, y_pred) - 2/3) < 1e-9
assert abs(recall_score(y_true, y_pred) - 2/3) < 1e-9
```

Make this a **unit test that runs in CI**, not a one-off. Metric code rots: someone changes `average="macro"` to `average="weighted"` to silence a warning, someone flips the `pos_label`, someone refactors the threshold out. A hand-derived assertion catches all of it the moment it breaks. The test is six lines and it is the cheapest insurance in your whole pipeline.

The bisection tree below is how you go from "the number looks wrong" to "it's *this* lie" with a small number of yes/no questions — the same make-it-fail-small discipline applied to the metric instead of the model.

![A decision tree that bisects a suspicious metric by asking whether the aggregate is high while the rare class is low, whether eval beats train, or whether live is below offline, each leading to a specific root cause and fix](/imgs/blogs/your-metric-is-lying-7.png)

### Bisection in action: the fraud model, step by step

Let us run the whole tree on our fraud classifier and watch the number get smaller and truer at each step. This is the problem-solving narrative the series is built around: bisect to the suspect, confirm with a test, fix, then stress-test the fix.

**Step 0 — the symptom.** The eval script reports **accuracy 0.99**. The product owner wants to turn on auto-blocking. Your alarm goes off because 0.99 on a 2%-fraud problem is suspiciously close to the do-nothing baseline of 0.98.

**Step 1 — is the aggregate hiding the rare class?** Run `classification_report`. Fraud-class recall is **0.07**. The model catches seven fraud cases in a hundred. The 0.99 accuracy was the majority-class lie of sections 1–2. *Confirmed:* report per-class. The honest headline is not "0.99 accuracy" but "0.07 fraud recall" — and that number does not support auto-blocking. We switch the reported metric to PR-AUC and per-class recall.

**Step 2 — with the right metric, is the eval set even clean?** PR-AUC comes back **0.97**, which now looks *too good* given the 0.07 recall at the default threshold — a high PR-AUC means good ranking, so the low recall must be a threshold artifact (section 5), but 0.97 PR-AUC on a hard fraud problem is itself suspicious. Run the duplicate check: **14% of eval transactions share a feature hash with training rows.** Leakage (section 4). *Confirmed:* dedup across splits, rerun. PR-AUC drops to **0.78**. Plugging into the mixture formula, the leak accounted for most of the gap, exactly as the math predicted.

**Step 3 — is the eval pipeline the same as training?** Before trusting 0.78, run `assert_pipelines_match` on one transaction. It passes (`max_diff 0.0`) — no skew this time. Good; we can trust 0.78 as an honest *offline* number on this distribution.

**Step 4 — does offline survive contact with production?** Holdout replay on a week of logged traffic gives PR-AUC **0.71** and fraud recall (at the tuned threshold) **0.55**. Small drop from 0.78, attributable to mild distribution shift — fraud patterns evolve. No feature skew (the serving-feature assertion passes). *Confirmed:* the honest, deployable number is around 0.71 PR-AUC with 0.55 fraud recall at the chosen operating point.

**The decision, four times over.** Accuracy 0.99 said "auto-block everything, fire the review team." Per-class recall 0.07 said "this model is useless at the default threshold." Dedup'd PR-AUC 0.78 said "the model ranks fraud well, tune the threshold." Live replay 0.55 recall said "ship it as a *ranking* that routes the riskiest transactions to a human queue, do not auto-block, and retrain weekly as patterns drift." The model never changed across all four steps. Every change was in *how honestly we measured it*, and each correction moved the business decision. That is the entire thesis of this post in one worked run.

**Stress-testing the fix.** Now interrogate the conclusion the way you would interrogate any debugging fix. *What if it's data, not metric?* The overfit-one-batch test passes and the loss curve is healthy, so the model trains fine — the problem was never the model, it was the report. *What if the imbalance were 1000-to-1 instead of 50-to-1?* Then even PR-AUC's baseline (0.001) is so low that absolute PR-AUC values look tiny; you would report *lift over baseline* and recall at a fixed precision, because raw PR-AUC becomes hard to read. *What if the eval set were small (say 2,000 rows)?* Then per-class recall on 40 fraud cases has a wide confidence interval — a recall of 0.55 is `22/40`, and the 95% interval spans roughly 0.39 to 0.70, so you must report the interval, not just the point estimate, or you will over-react to noise. *What if the metric is computed across 8 GPUs?* Then verify it is gathered-then-computed, not per-rank-averaged, or you reintroduce the section-7 bug at the rank axis. Each stress test either confirms the fix holds or names the new condition under which a new lie appears.

## 10. The diagnostic table: symptom to cause to test to fix

When a metric looks wrong, do not guess. Match the signature to the table, run the confirming test, and only then change anything. The figure first, then the expanded version with fixes.

![A matrix mapping four metric symptoms to their root cause and the single confirming test that isolates each, from high aggregate with bad rare class to metric jitter across runs](/imgs/blogs/your-metric-is-lying-3.png)

| Symptom | Most likely cause | Confirming test | Fix |
|---|---|---|---|
| High aggregate, but the class you care about fails | Micro/weighted averaging under imbalance | `classification_report` — read per-class recall | Report macro + per-class recall; pick the average that matches the goal |
| 99% accuracy, model seems to do nothing useful | Accuracy rewarding the majority | Compute PR-AUC; compare to base rate, not 0.5 | Use PR-AUC + per-class recall; move the threshold |
| Eval is *worse* than training, unexpectedly | Eval-train skew (preprocessing/tokenization) | `assert_pipelines_match` on one sample | Make eval pipeline byte-identical to train |
| Metric jumps suspiciously after a small change | Leakage (duplicates or a leaked feature) | Cross-split hash overlap; single-feature AUC | Dedup across splits; drop leaked feature; group-split |
| Reported metric improves, users say it got worse | Metric doesn't match the goal | Add the real metric (ranking/human) alongside | Optimize/report the metric you care about; validate proxies |
| Great offline, worse in production | Offline-online gap (shift / serve skew / proxy label) | Holdout replay; feature-skew check | Monitor drift; match serving features; instrument real outcome |
| Metric changes when you change batch size | Per-batch averaging of a non-decomposable metric | Recompute globally (torchmetrics) and compare | Accumulate counts/scores; compute once globally |
| Metric suspiciously clean, but counts seem off | Silent NaN drop / dropped batch / pad miscount | Audit sample count vs expected; check mask | `drop_last=False` on eval; mask `-100`; set `zero_division` |

This table is the section to bookmark. Each row is a symptom you have probably already seen, a cause most people misattribute to "the model needs more training", and a one-command test that settles it.

## 11. Case studies and real signatures

These are well-documented patterns, not invented numbers — where a metric lied at scale and someone caught it.

**Label errors in benchmark test sets (confident learning).** The 2021 study by Northcutt, Athalye, and Mueller ("Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks") used confident learning to estimate label errors in the *test sets* of ten widely-used benchmarks — ImageNet, CIFAR, MNIST, and others — and found an average of roughly 3.4% errors, with ImageNet's validation set estimated around 6% mislabeled. The consequence is a metric lie of a specific kind: when the test labels themselves are wrong, the benchmark "accuracy" is measuring agreement with a noisy oracle, and the study showed that model *rankings* can flip once you correct the labels — the model that "won" the benchmark was partly winning by matching the test set's mistakes. The takeaway: even a correctly-computed accuracy is a lie if the ground truth is wrong, and the `cleanlab` library exists precisely to find those errors. This is the eval-set analog of [data leakage the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer) — the eval set is the source of the lie.

**Benchmark contamination in LLMs.** As pretraining corpora grew to include large fractions of the public web, benchmark test questions began appearing verbatim in training data — the GPT-3 paper (Brown et al., 2020) included an explicit contamination analysis, removing benchmark overlap and re-scoring, and the practice of reporting contamination-cleaned numbers became standard for exactly this reason. The lie is the mixture formula from section 4 at corpus scale: if even a few percent of a benchmark leaked into pretraining, the reported score is inflated by memorization, and the model looks more capable than it is on genuinely held-out questions. Modern eval reports decontaminate (n-gram overlap removal between train and eval) before trusting the number.

**The Kaggle leakage post-mortems.** Competition write-ups are a catalog of metric lies caught the hard way: an ID column correlated with the target because the data was exported sorted by label; a timestamp that leaked the future; a patient appearing in both train and test so a "diagnosis" model memorized patients instead of learning medicine. The recurring fix is the same — `GroupKFold` or `TimeSeriesSplit` so that no group (patient, user, time period) spans the train/test boundary. The cross-validation companion covers the splitting discipline; the lesson here is that the *leaderboard metric itself was the lie*, and the gap between public and private leaderboard was the offline-online gap made visible.

**The padding-counts-as-correct LLM eval bug.** A recurring real-world bug in LLM finetuning harnesses: token-level accuracy or perplexity computed over all positions including padding, inflating the number because pad tokens are trivially predicted. It is the section-8 lie, and it is common enough that careful eval code always asserts it is masking `-100` and reports the masked-token count alongside the metric.

**The recommendation-system offline-online disconnect.** A pattern documented repeatedly across industry post-mortems: a ranking model improves an offline metric (NDCG, AUC on logged clicks) but a live A/B test shows flat or negative engagement. The usual culprits are the section-5 and section-6 lies stacked together — the offline metric is computed against *logged* impressions, which were themselves chosen by the *previous* model, so the offline evaluation only ever sees items the old policy decided to show (the "feedback loop" or "presentation bias" problem). The new model is scored on a distribution the old model curated, so a higher offline number can reflect agreement with the old policy rather than genuine improvement. The only honest arbiter is the online A/B test, which is why mature recommendation teams treat offline metrics as a *filter* (reject obviously-worse models cheaply) and never as the final ship decision. The offline number is not lying about the logged data; it is lying about what users would do on items they were never shown.

**The metric that ranked the wrong model on a noisy test set.** Tying the confident-learning result to model selection: when test labels carry a few percent of noise, two models within a couple of points of each other are effectively tied within the label-noise floor, yet teams routinely declare the higher number the winner. The honest move is to estimate the test set's own error rate (via cleanlab or a small re-annotation) and treat differences smaller than that floor as noise, not signal. Picking a model on a difference smaller than the test set's label-noise is the metric lying by *false precision* — reporting more significant digits than the ground truth supports.

## 12. When this is (and isn't) your bug

Be decisive about when a symptom points at the metric versus somewhere else in the six places.

**It IS a metric lie when:** the *number* moved but a per-class or per-example look at the predictions doesn't support it; the metric changes when you change batch size, eval-set size, or threshold (those are all metric-computation artifacts, not model changes); eval beats train (almost always skew or leak, rarely a genuinely better-on-eval model); a small change produced a large jump; or the offline number and the live outcome disagree. In all of these, **the model may be unchanged and the metric is the variable.**

**It is NOT a metric lie when:** the loss curve itself is diverging or stuck (that is optimization or numerics — see [the loss curve as a diagnostic](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs)); the overfit-one-batch test fails (that is data pipeline or model code, and no metric subtlety will save you); or the model genuinely predicts the same wrong thing for everything (that is a model/optimization problem, and the metric is correctly reporting failure). A correctly-computed metric reporting a bad number is not lying — it is telling you the truth you don't want to hear, and the bug is upstream.

The discriminating move is the same throughout: **re-derive the metric by hand, then compute it per-class and globally.** If the hand calc, the per-class breakdown, and the global computation all agree and the number is still bad, the metric is honest and you should bisect into the other five places. If they disagree, you have found your lie. Don't retrain a model to fix a number that a one-line assertion would have explained.

A useful way to keep the categories straight is by the *direction* of the lie. Lies that make the number **too high** — leakage, per-batch averaging of F1, padding counted as correct, NaN-drop dropping hard cases, accuracy under imbalance, micro/weighted averaging hiding the rare class — are the dangerous ones, because they manufacture false good news and nobody investigates good news. Lies that make the number **too low** — eval-train skew, an untuned threshold, a too-strict `zero_division` — are annoying but self-correcting, because a bad number sends you looking and you eventually find the cause. The offline-online gap can go either way but in practice is almost always a *fall* in production, which makes it a too-high offline lie. The practical consequence: **be most suspicious of your best numbers.** A metric that looks great deserves more scrutiny than a metric that looks bad, because the great one is the one that will ship a broken model. Reverse your instinct — when the dashboard turns green, that is the moment to run the audits, not to stop.

The cleanest single sentence to carry: a correctly-computed metric on the right data with the right average at the right threshold reporting a number you can reproduce by hand on six rows is not lying — anything else might be, and you check in that order.

## 13. Key takeaways

- **A rising number is a hypothesis, not a result.** Confirm it with per-class breakdowns, a global recompute, and a hand-derived unit test before you believe it or ship on it.
- **Under imbalance, micro-F1 and accuracy are the same lie** — both collapse onto the majority. Report macro plus per-class recall, and prefer PR-AUC (baseline = positive rate) over ROC-AUC (baseline = 0.5).
- **Eval worse than expected? Suspect eval-train skew first.** Assert one sample is byte-identical through the eval and training pipelines before retraining anything.
- **A suspicious jump is a leak until proven otherwise.** Check cross-split duplicate overlap and single-feature AUC; a 15-point gain from a small change is almost never genius.
- **F1, precision, recall, and AUC are not decomposable** — never average them per batch. Accumulate counts/scores and compute once globally (use `torchmetrics`). If the metric changes with batch size, that is the bug.
- **Audit the denominator.** Dropped last batch, NaN-drop via `nanmean`, and pad tokens counted as correct all silently change the set the metric is computed over.
- **Optimize and report the metric you actually care about.** Logloss is not ranking; BLEU and perplexity are not task quality. Validate cheap proxies against the expensive real metric.
- **The offline number answers a question about the past.** Close the offline-online gap with holdout replay, a serving-feature skew check, and a real-outcome label — not with more offline rigor.
- **Re-derive by hand and assert in CI.** A six-row example with a hand-computed F1 is the cheapest, most reliable metric check you will ever write.

## 14. Further reading

- Northcutt, Athalye, Mueller, "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks" (2021) — confident learning, the `cleanlab` library, and why benchmark accuracy can rank the wrong model.
- Brown et al., "Language Models are Few-Shot Learners" (GPT-3, 2020) — the benchmark contamination analysis and the practice of decontaminated reporting.
- scikit-learn documentation, "Metrics and scoring" and `classification_report` — the precise definitions of micro/macro/weighted averaging, `zero_division`, and `pos_label`.
- `torchmetrics` documentation — stateful, distributed-correct metrics that accumulate sufficient statistics and compute the global value, the correct cure for per-batch averaging.
- Saito and Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets" (2015) — why PR-AUC beats ROC-AUC under imbalance.
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook), [class imbalance and when accuracy lies](/blog/machine-learning/debugging-training/class-imbalance-and-when-accuracy-lies), [data leakage the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer), and [distribution shift train vs the real world](/blog/machine-learning/debugging-training/distribution-shift-train-vs-the-real-world).

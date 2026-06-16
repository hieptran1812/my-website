---
title: "Look at Your Data Before You Train: The 30 Minutes That Save the Week"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Before you launch a single GPU-hour, run the eight-step data audit that catches label flips, sentinel values, broken decodes, and near-duplicate leaks — the cheapest debugging tool you own is your own eyes."
tags:
  [
    "debugging",
    "model-training",
    "data-quality",
    "label-noise",
    "data-leakage",
    "finetuning",
    "deep-learning",
    "computer-vision",
    "nlp",
    "tabular",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/look-at-your-data-before-you-train-1.png"
---

A team I sat with once spent six days chasing a model that trained beautifully and then refused to generalize. The loss curve was textbook — a clean exponential decay, validation tracking within a hair of train, no spikes, no NaNs. They tuned the learning rate, tried three optimizers, added regularization, swapped the architecture, and rented a bigger GPU. On the seventh day someone finally opened the raw CSV in a spreadsheet, sorted one numeric column descending, and found a wall of `-999` values sitting at the top. A sensor had been emitting `-999` as its "no reading" sentinel, the ingestion code had treated it as a real number, and the model had quietly learned that "a feature value near negative one thousand" was a strong, leaky signal that existed in the training logs but never at serving time. Thirty seconds of looking at the data — not the loss curve, the data — would have caught it on day zero. Instead it cost a week and roughly \$4,800 in compute and salary.

This post is about the single most underused debugging tool in machine learning: your own eyes, pointed at your own data, *before* you train. Andrej Karpathy calls this "become one with the data," and he is not being poetic. The discipline is mechanical and repeatable: per-class counts, per-feature distributions, raw examples decoded and viewed next to their labels, duplicate detection, sorting by extremes and inspecting the tails, an embedding scatter colored by label, a loss-ranked pass after a quick model, and a split audit. Eight steps, thirty focused minutes, and it catches roughly half the bugs the rest of this series teaches you to debug *after* they have already wasted your time. The figure below lays out the full ritual as a left-to-right sequence, with what each step rules in or out.

![The eight-step pre-training data audit as an ordered timeline, from per-class counts through the loss-ranked pass, each labeled with the bug class it catches](/imgs/blogs/look-at-your-data-before-you-train-1.png)

I want to make a strong claim and then defend it for the rest of this post: looking at data is not a soft, fuzzy, "good hygiene" activity that you do when you have spare time. It is a precise diagnostic with a known yield, and it dominates summary statistics for a provable reason — two datasets can have identical means, variances, and correlations and yet be completely different in ways that destroy a model. The mean does not see the `-999`. The histogram does. By the end of this post you will have a reusable data-audit script for three modalities (tabular, vision, NLP), you will know exactly which bug each of the eight steps catches, and you will have internalized the order — counts first because they are cheapest, loss-ranking last because it needs a model. You will also know when *not* to trust your eyes, because looking has its own failure modes.

This is the proactive twin of the whole [training-debugging taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs): instead of waiting for a symptom and bisecting across the six places a bug can hide — data, optimization, model code, numerics, systems, evaluation — you spend thirty minutes up front collapsing the most common branch, *data*, to near-zero. The audit is the cheapest form of **make-it-fail-small** there is: you shrink the dataset to a few hundred rows you can actually read and let your visual cortex, the best anomaly detector ever built, do the work the optimizer cannot.

## 1. The symptom: a clean loss curve that is secretly a lie

Let us be precise about the failure mode this audit prevents, because it is the most seductive one in the field: training that *works* and a model that is *wrong*.

A genuinely broken pipeline is merciful. It throws a shape error, returns a NaN at step 3, or fails to converge at all. You get a stack trace and you go fix it. The dangerous case is the pipeline that runs perfectly, drives the loss down on a smooth curve, reports a strong validation number, and ships a model that fails in production — because the bug is not in the *code*, it is in the *data the code is faithfully training on*. The loss curve cannot see a label flip. It cannot see that `-999` is a sentinel. It cannot see that 8% of your training rows also appear in your validation set. Every one of those bugs produces a beautiful curve and a broken model.

Here is the running example we will return to. We are finetuning a small image classifier on a ten-class dataset of product photos, and we are also, in parallel, training a tabular gradient-boosted model on customer churn — two modalities so I can show you how the same audit step manifests differently. The dashboards both look great:

| Instrument | Vision run reading | Tabular run reading | What it seems to say |
| --- | --- | --- | --- |
| Train loss (final) | 0.08 | logloss 0.21 | "Converged cleanly" |
| Val loss / metric | 0.19 / 94% acc | AUC 0.971 | "Strong generalization" |
| Loss curve shape | Smooth decay | Smooth decay | "Optimization is healthy" |
| Gradient norm | 1.8, stable | n/a | "Numerics are fine" |
| Production result | 71% acc | AUC 0.78 | "...what happened?" |

Both runs look healthy on every instrument except the one that matters: the production number, which collapses. The vision model dropped 23 accuracy points; the tabular model lost 0.19 of AUC. Neither failure is in the optimizer, the architecture, or the numerics. Both are in the data, and both would have been caught in the first ten minutes of the audit — the vision model has a near-duplicate leak across the train/val split, and the tabular model has the `-999` sentinel acting as a leaky feature. We will find both with our eyes, not our tuners.

The deep point is that **a smooth loss curve certifies that your optimizer can minimize the loss you gave it. It certifies nothing about whether that loss corresponds to the task you care about.** If your labels are 6% wrong, the optimizer happily learns to be 6% wrong. If a leaky feature is present, the optimizer happily exploits it. The instrument that checks whether the *data* encodes the *task* is not the loss curve — it is you, reading the data.

It is worth walking through *why each instrument on that dashboard is blind to data bugs*, because the blindness is structural, not a calibration issue you could fix with a better tool. The **loss** measures the discrepancy between predictions and the labels you supplied — if the labels are wrong, the loss faithfully reports agreement with the wrong answer, and a low loss means the model agrees with the corruption. The **gradient norm** measures how hard the optimizer is pushing — it is large when the model is far from minimizing the supplied loss and small when it is close, and it cannot distinguish "close to the right answer" from "close to the leaked shortcut." The **GPU utilization** and **throughput** measure how busy the hardware is — they are pegged at 95% whether the bytes flowing through are pristine or poisoned. The **validation metric** is the one instrument that *could* catch a data bug, but only if the validation set is itself clean and disjoint — and a leak contaminates the validation set in exactly the same motion that it contaminates training, so the validation number is corrupted *in the same direction*, which is why it looks great right up until production. Every dial on the dashboard is downstream of the data; none of them audits the data. That audit has to happen *before* the data enters the pipeline, with a different instrument entirely: your eyes.

This is also why the bug is so expensive to find *after* the fact. By the time the production number disagrees with the offline number, you have a trained model, a logged run, a dashboard full of healthy curves, and a strong prior that "the training worked." Every artifact you would naturally inspect points away from the data, because every artifact was generated *by faithfully training on the data*. The investigation that finally cracks it is always the one nobody wanted to do — open the raw rows and look — precisely because it is the one thing that sits *upstream* of all the misleading evidence. Doing it first, before any of that evidence exists, is both cheaper and clearer.

## 2. The science: why looking beats summary statistics

The reason "just look at it" is a rigorous diagnostic and not a vibe is that summary statistics are *lossy compressions* of a distribution, and the information they throw away is exactly the information that bugs live in. This is not hand-waving; it is a theorem you can demonstrate in four lines of code.

### 2.1 Anscombe's quartet and the Datasaurus: identical stats, different worlds

In 1973 the statistician Frank Anscombe published four small datasets, now called Anscombe's quartet, that are nearly identical in every standard summary statistic: each has the same mean of $x$ (9.0), the same mean of $y$ (7.5), the same variance of $x$ (11.0), the same variance of $y$ (≈4.13), the same correlation between $x$ and $y$ (0.816), and the same fitted regression line ($y = 3.00 + 0.500x$). To two decimal places, the numbers are the same. Yet one is a clean linear relationship, one is a perfect curve, one is a line with a single wild outlier, and one is a vertical stripe of points with one far-off leverage point that single-handedly creates the "correlation." If you only ever looked at the summary statistics, you would conclude all four were the same well-behaved linear dataset. You would be wrong about three of them.

The modern, more dramatic version is the *Datasaurus Dozen* (Matejka and Fitzmaurice, 2017): a collection of datasets that all share the same mean, standard deviation, and Pearson correlation to two decimal places, yet one of them, when plotted, is unmistakably a *dinosaur*. The summary statistics are blind to the dinosaur. The scatter plot sees it instantly.

The lesson generalizes far beyond toy data. A mean compresses an entire distribution to one number; it cannot tell you that the distribution is bimodal, that it has a spike of sentinel values at `-999`, that 3% of it is `NaN`, or that it is actually two classes stacked on top of each other. The formal statement is that the map from a dataset to its low-order moments is *many-to-one*: enormously many different datasets map to the same $(\mu, \sigma, \rho)$. Bugs change the dataset; they often do *not* change the moments enough to notice. So checking the moments is a weak test, and looking at the full distribution — the histogram, the scatter, the raw examples — is a strong one.

### 2.2 Why the tails dominate failure modes

There is a second, sharper reason to look rather than summarize: model failures are concentrated in the tails of the data, and summary statistics are dominated by the bulk. A model's average loss can be excellent while its worst-case behavior is catastrophic, and it is the worst case that gets you paged.

Make this concrete. Suppose your per-example loss $\ell_i$ has a mean of 0.2 — a healthy-looking number. Now suppose 1% of your examples are mislabeled, and on those the model is forced into a loss of around 5.0 (it is being punished for being right). The contribution of those bad examples to the *mean* is only $0.01 \times 5.0 = 0.05$, which barely moves the headline number from 0.2 to roughly 0.25. The mean essentially hides them. But those same examples are the entire top of the loss-ranked list — they sit at rank 1 through (0.01 × N), screaming, if you only sort and look. This is the mathematical justification for **step 7, loss-ranked inspection**: the tail of the loss distribution is where label noise, edge cases, and broken examples concentrate, and the tail is precisely what averages erase.

The same tail logic drives **step 5, sort-by-extreme**. The brightest and darkest images, the longest and shortest sequences, the largest and smallest feature values — these extremes are where decode bugs, truncation bugs, and sentinel values live. A sentinel of `-999` will never show up in the mean of a feature whose real values are around 50; the mean will read maybe 35 and look unremarkable. But sort the column descending or ascending and the `-999` block is the first thing you see. **The tails are cheap to inspect (you only look at a few extremes) and they have the highest density of bugs per example you read.** That is the best return on attention in all of debugging.

### 2.3 The arithmetic of a duplicate leak

There is a third piece of science worth making precise, because it explains *exactly how many points* a near-duplicate leak steals from honesty — and it is the bug that broke our vision run. Suppose your validation set has $N$ examples, of which a fraction $f$ are near-duplicates of training examples (the same product photographed from a slightly different angle, sitting in both splits). On the leaked fraction, the model is effectively being tested on data it memorized, so it scores close to its *training* accuracy $a_{\text{train}}$ there. On the clean fraction, it scores its *true generalization* accuracy $a_{\text{gen}}$. The reported validation accuracy is then a blend:

$$a_{\text{val}} \approx f \cdot a_{\text{train}} + (1 - f) \cdot a_{\text{gen}}.$$

The inflation — the gap between what you report and the truth — is

$$a_{\text{val}} - a_{\text{gen}} \approx f \cdot (a_{\text{train}} - a_{\text{gen}}).$$

Plug in our run's numbers. The leak fraction was $f = 0.082$, training accuracy was about $a_{\text{train}} = 0.99$, and honest generalization was about $a_{\text{gen}} = 0.79$. The predicted inflation is $0.082 \times (0.99 - 0.79) = 0.082 \times 0.20 \approx 0.016$, or roughly 1.6 points on the *blended* metric — but the gap we actually observed (94% reported versus 79% honest) was 15 points, much larger. The discrepancy is the tell that the leak was not a uniform 8.2% sprinkle of memorized points; the duplicated products were *also* the easiest, highest-confidence classes, so they pulled the blend far harder than a uniform leak would. The formula gives you the floor of the inflation; the real number is usually worse because leaks are not uniformly distributed across difficulty. Either way, the conclusion is the same: **any non-zero $f$ inflates your headline number, the inflation scales with both the leak fraction and the train-minus-generalization gap, and the only way to drive $f$ to zero is step 4 — you cannot estimate it from the loss curve.** This is the quantitative backbone of the split audit, and it is why a 94% you cannot trust is worse than a 79% you can.

> **The audit's governing principle.** Bugs hide in the parts of the distribution that summary statistics compress away — the shape, the tails, the duplicates, the raw bytes. Every step of the ritual is a different lens onto a part of the distribution your `df.describe()` cannot see. You are not replacing statistics; you are looking at what they discard.

The stack below shows what each layer of inspection recovers that the layer above it threw away — from a single mean at the top down to the raw decoded example at the bottom, where the bug actually lives.

![A stack showing how each deeper layer of data inspection recovers information that summary statistics discard, from the mean down to the raw decoded example](/imgs/blogs/look-at-your-data-before-you-train-6.png)

## 3. The ritual, step by step

Here is the eight-step audit in order. The order is not arbitrary: each step is placed by cost and by dependency. Counts come first because they are a one-line `value_counts()` and catch the most catastrophic bugs (empty classes, label typos). Loss-ranking comes last because it requires you to train a quick model first. The figure below maps each step to the bug class it catches and the modality where it bites hardest — this is the matrix to pin above your desk.

![A matrix mapping each of the eight audit steps to the primary bug it catches and the modality where it most often bites, from label typos to split contamination](/imgs/blogs/look-at-your-data-before-you-train-2.png)

**Step 1 — Per-class / per-label counts.** A single `value_counts()`. Catches: empty classes (a class with zero examples that you think you are training on), severe imbalance (one class is 98% of the data and accuracy is meaningless — see [class imbalance and when accuracy lies](/blog/machine-learning/debugging-training/class-imbalance-and-when-accuracy-lies)), and label typos (you find `"cat"`, `"Cat"`, `"cta"`, and `"catt"` as four "different" classes). This is the cheapest step and it catches the most embarrassing bugs.

**Step 2 — Per-feature distributions, ranges, %missing, cardinality.** For every column or feature channel: min, max, mean, the fraction missing, the number of unique values, and a histogram. Catches: scale bugs (one feature is in 0–1, another in 0–10,000, and your distance metric is dominated by the big one), constant features (a column with one unique value contributes nothing and signals an upstream join bug), NaN/inf, and sentinel values like `-999`, `-1`, `9999`, `1970-01-01`, or the empty string masquerading as data.

**Step 3 — Decode and VIEW raw examples next to their labels.** Actually render the data your model will see — not the file path, the decoded tensor turned back into something human. For images: a grid of decoded images with their labels printed underneath. For text: print the tokenized-then-detokenized string with its label. For audio: the waveform and spectrogram with the transcript. Catches: mislabels (the image is clearly a dog and the label says cat), wrong preprocessing (the image is BGR when the model expects RGB and everything looks blue-tinted — see [the input pipeline is lying to you](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you)), broken decode (a JPEG that decoded to gray mush), and double-normalization.

**Step 4 — Duplicate / near-duplicate detection.** Exact duplicates by hashing the raw bytes; near-duplicates by perceptual hash (images), MinHash/shingling (text), or embedding cosine similarity (anything). Catches: train/val contamination (the same example in both splits inflates your validation number), and a training set that is secretly much smaller than it looks because 30% of it is copies. This is the bug that broke our vision run.

**Step 5 — Sort by something and inspect the tails.** Sort by sequence length, image brightness, file size, or any extreme feature value, and look at the top and bottom 20. Catches: truncation bugs (the longest sequences are silently cut and lose the answer), the sentinel block, corrupt files (zero-byte images), and outliers that will dominate the gradient.

**Step 6 — Embedding scatter colored by label.** Embed the data (a pretrained encoder, or even TF-IDF + PCA for text) and plot a 2-D projection (UMAP, t-SNE, or PCA) colored by label. Catches: label noise (a red point sitting deep inside a blue cluster is probably mislabeled), leakage (two classes that should overlap form perfectly separated islands — too easy, something is leaking), and structure you did not know was there (the data is actually three sub-populations).

**Step 7 — Loss-ranked inspection after a quick model.** Train a fast, weak model (a few epochs, or an XGBoost, or a logistic regression on embeddings), then sort the *training* examples by their loss and look at the highest-loss 50. Catches: the hardest real examples (useful), but mostly the *noise* — mislabels and broken examples cluster at the top of the loss ranking because the model is being punished for being correct. This is confident learning's core idea, and it is the highest-yield step per example you read.

**Step 8 — The split audit.** Confirm train, validation, and test are genuinely disjoint (no shared IDs, no shared near-duplicates) and that they are *similarly distributed* (the feature histograms overlap; the class balance matches; the time ranges do not overlap when they should not). Catches: the contamination from step 4 viewed from the other side, plus distribution shift between splits that makes your validation number meaningless. This connects directly to [data leakage, the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer).

The discipline is to run all eight, in order, every time, before the first real training run. It takes thirty minutes. The rest of this post shows you the code for each, the bugs each one caught in real projects, and the math behind why they work.

### 3.1 Why this exact order — the cost-versus-yield argument

The ordering is engineered, not aesthetic. Think of each step as having a *cost* (how long it takes to run and read) and a *yield* (how catastrophic and common the bugs it catches are). The optimal order runs high-yield-per-cost steps first, so that if you only have five minutes you have still caught the worst, most common bugs. Walk the gradient:

- **Steps 1–2 (counts and ranges) are nearly free and catch the most catastrophic bugs.** A `value_counts()` and a `describe()` are one line each and run in milliseconds. The bugs they catch — an empty class you think you are training on, a constant feature from a broken join, a `-999` sentinel — are both common and total: they invalidate the entire run. Highest yield per second of any step. Always first.
- **Step 3 (view raw) costs a minute and catches the bugs no number can.** Rendering a grid or printing decoded strings takes a moment, but it is the *only* step that catches BGR/RGB swaps, double-normalization, and obvious mislabels, because those are invisible to every aggregate. Third because it costs slightly more attention than a `value_counts()`.
- **Steps 4–5 (dups and tails) scale automatically and catch the silent metric-inflators.** Hashing and sorting are cheap and run on the full set; the leak and truncation bugs they catch are the ones that make a model look better than it is. They come after the free steps but before the model-dependent ones.
- **Step 6 (embedding scatter) costs an encoder pass and reveals geometry.** Slightly more expensive (you need embeddings), so it sits in the middle; it catches what counts and ranges cannot — the *relational* bugs, where an example is wrong relative to its neighbors.
- **Step 7 (loss-ranked) is last among the data steps because it needs a model.** You cannot rank by loss until you have trained something, even a weak something. So it is deferred — but it is the highest-yield-per-example-read of all, which is why it is worth the wait.
- **Step 8 (split audit) is the gate before you trust any number.** It is placed last because it depends on the splits being finalized, and it is the step that validates the *measurement* rather than the data — the final check before you let a validation number into a decision.

The practical upshot: if a manager interrupts you four minutes into the audit, you have already run steps 1, 2, and probably 3, which together catch the majority of catastrophic data bugs. The order degrades gracefully. A random order does not — you might burn your four minutes on a t-SNE plot and miss the empty class.

There is one more reason the order matters: **later steps assume earlier steps passed.** A loss-ranked inspection (step 7) is meaningless if your labels have typos (step 1) — the model cannot learn a consistent mapping, so every example looks high-loss. An embedding scatter (step 6) is misleading if a feature is on the wrong scale (step 2) — the geometry is dominated by the un-normalized axis. The audit is a pipeline of increasingly sensitive instruments, and each one needs the coarser bugs cleared before its signal is trustworthy. Run them out of order and you will chase artifacts of the bugs you skipped.

## 4. The diagnostic: a reusable data-audit script

Let us build the audit as runnable code, one modality at a time. Everything here is copy-and-run with `pandas`, `numpy`, `scikit-learn`, `torch`, and `torchvision`.

### 4.1 Tabular: the pandas-profiling-style audit

The tabular audit is mostly steps 1, 2, 5, and 8. Here is a single function that produces the per-feature report that would have caught the `-999` sentinel in our churn model.

```python
import numpy as np
import pandas as pd

# Common sentinel values that masquerade as real data.
SENTINELS = {-999, -9999, -1, 999, 9999, 0, -1.0}

def audit_tabular(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Per-feature audit: range, %missing, cardinality, sentinel hits."""
    rows = []
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        n_unique = s.nunique(dropna=True)
        pct_missing = float(s.isna().mean() * 100.0)
        # Count how many rows match a known sentinel value.
        sentinel_hits = int(s.isin(SENTINELS).sum()) if is_num else 0
        rows.append({
            "feature": col,
            "dtype": str(s.dtype),
            "n_unique": n_unique,
            "constant": n_unique <= 1,            # Step 2: constant feature
            "pct_missing": round(pct_missing, 2),  # Step 2: missingness
            "min": float(s.min()) if is_num else None,
            "max": float(s.max()) if is_num else None,
            "sentinel_hits": sentinel_hits,        # Step 2/5: sentinel block
            "high_cardinality": is_num is False and n_unique > 0.5 * len(s),
        })
    report = pd.DataFrame(rows)
    # Flag anything suspicious so it sorts to the top.
    report["suspicious"] = (
        report["constant"]
        | (report["pct_missing"] > 20.0)
        | (report["sentinel_hits"] > 0)
        | report["high_cardinality"]
    )
    return report.sort_values("suspicious", ascending=False)

# Step 1 — class counts (run this FIRST, it is the cheapest):
print(df[target].value_counts(dropna=False))

report = audit_tabular(df.drop(columns=[target]), target)
print(report[report["suspicious"]].to_string(index=False))
```

On the churn data this immediately printed a row for `last_login_days` with `min = -999.0`, `sentinel_hits = 1,840`, and `suspicious = True`. The mean of that column read 31.2 and looked unremarkable; the *audit* read the min and the sentinel count and the bug was undeniable. The fix is to replace the sentinel with a proper missing marker before any scaling or imputation:

```python
# Replace the sentinel with NaN so imputation handles it honestly,
# and add an explicit "was-missing" indicator (often predictive on its own).
df["last_login_missing"] = (df["last_login_days"] == -999).astype("int8")
df["last_login_days"] = df["last_login_days"].replace(-999, np.nan)
```

### 4.2 Step 5 — sort and look at the tails (any modality)

Sorting is one line and it is the highest-leverage cheap step. For tabular, sort by the suspect feature; for NLP, sort by token length; for vision, sort by mean pixel value (brightness) or file size.

```python
# Tabular: the extreme rows of every flagged feature.
for col in report[report["suspicious"]]["feature"]:
    extremes = pd.concat([df.nsmallest(5, col), df.nlargest(5, col)])
    print(f"\n=== {col} tails ===")
    print(extremes[[col, target]].to_string(index=False))

# NLP: the longest and shortest sequences after tokenization.
lengths = [len(tok(x).input_ids) for x in texts]          # tok = your tokenizer
order = np.argsort(lengths)
for i in list(order[:5]) + list(order[-5:]):              # 5 shortest + 5 longest
    print(f"len={lengths[i]:>4} label={labels[i]} :: {texts[i][:120]!r}")
```

The NLP version is how you discover that your "longest" sequences are all hitting the 512-token truncation wall and losing the answer that lives at the end of the document — a truncation bug that no aggregate metric reveals but the tail makes obvious in seconds.

### 4.3 Vision: the grid-of-images-with-labels (step 3) and the decode check

For images, step 3 is non-negotiable and trivially cheap: render a grid of decoded images with their labels. This is where you catch BGR/RGB swaps, normalization that turned everything purple, and mislabels.

```python
import torch
import torchvision
import matplotlib.pyplot as plt

def show_grid(dataset, n=16, denorm_mean=None, denorm_std=None):
    """Render the EXACT tensors the model sees, decoded back to viewable."""
    imgs, labels = [], []
    for i in range(n):
        x, y = dataset[i]                 # x is the post-transform tensor
        imgs.append(x); labels.append(y)
    batch = torch.stack(imgs)
    if denorm_mean is not None:           # undo Normalize so it looks right
        mean = torch.tensor(denorm_mean).view(1, 3, 1, 1)
        std = torch.tensor(denorm_std).view(1, 3, 1, 1)
        batch = batch * std + mean
    grid = torchvision.utils.make_grid(batch, nrow=4).clamp(0, 1)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title(" | ".join(str(l) for l in labels[:4]))
    plt.axis("off"); plt.tight_layout(); plt.savefig("audit_grid.png", dpi=120)

# Critical: pass the SAME mean/std your transform used, or you will
# "fix" a bug that is not there and miss the one that is.
show_grid(train_ds, denorm_mean=[0.485, 0.456, 0.406],
                    denorm_std=[0.229, 0.224, 0.225])
```

If the grid comes out blue-tinted, your decode is BGR and the model is being trained on color-swapped data — a silent accuracy killer that the loss curve will never show you. If a tomato is rendered next to the label "apple," you have found a mislabel by eye in one second that loss-ranking would take a quick model to surface.

### 4.4 Step 4 — duplicate and near-duplicate detection

Exact duplicates are a hash. Near-duplicates need a perceptual signal. Here is the pattern that found the leak in our vision run: hash for exact dupes, then embedding cosine similarity for near-dupes across the train/val boundary.

```python
import hashlib
from collections import defaultdict

def exact_dupes(paths):
    """Hash raw bytes; same hash = byte-identical file."""
    by_hash = defaultdict(list)
    for p in paths:
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        by_hash[h].append(p)
    return {h: ps for h, ps in by_hash.items() if len(ps) > 1}

# Near-duplicates ACROSS splits via embedding cosine similarity.
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cross_split_near_dupes(train_emb, val_emb, train_ids, val_ids, thresh=0.98):
    """Any val item with a >thresh-similar train item is a leak."""
    sims = cosine_similarity(val_emb, train_emb)        # (n_val, n_train)
    leaks = []
    for vi in range(sims.shape[0]):
        ti = int(sims[vi].argmax())
        if sims[vi, ti] >= thresh:
            leaks.append((val_ids[vi], train_ids[ti], float(sims[vi, ti])))
    return leaks

leaks = cross_split_near_dupes(val_emb, train_emb, val_ids, train_ids)
print(f"{len(leaks)} validation items have a near-duplicate in train")
```

On our product-photo data this printed `412 validation items have a near-duplicate in train`, which is 8.2% of the validation set — the same products shot from slightly different angles, scattered across both splits. The validation accuracy of 94% was inflated by roughly the leak fraction; once we deduplicated by product ID and re-split, honest validation accuracy fell to 79%, which matched production within noise. **The 23-point production gap was not a generalization failure — it was a measurement error caused by a near-duplicate leak that ten lines of audit code found.**

### 4.5 Step 7 — loss-ranked inspection (confident learning, lite)

After a quick model, rank training examples by loss and read the top of the list. This is the single highest-yield step, and the principled version of it is *confident learning*, implemented in the `cleanlab` library.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

# 1) Get out-of-fold predicted probabilities (so we don't rank on memorized rows).
probs = cross_val_predict(LogisticRegression(max_iter=1000),
                          X, y, cv=5, method="predict_proba")

# 2) Per-example self-confidence loss: -log P(true label).
true_probs = probs[np.arange(len(y)), y]
example_loss = -np.log(np.clip(true_probs, 1e-12, 1.0))

# 3) The top of this list is where mislabels live.
worst = np.argsort(example_loss)[::-1][:50]
for i in worst[:10]:
    print(f"loss={example_loss[i]:.2f} label={y[i]} "
          f"pred={probs[i].argmax()} p_true={true_probs[i]:.3f}")

# The library version (uses confident learning to estimate label issues):
# from cleanlab.filter import find_label_issues
# issues = find_label_issues(labels=y, pred_probs=probs)  # boolean mask
```

The crucial detail is `cross_val_predict` with out-of-fold probabilities: if you score examples with a model that *trained on them*, the model has memorized the labels and the loss ranking is meaningless. Out-of-fold scoring is what makes the high-loss examples genuinely suspicious rather than merely under-fit. This connects directly to [garbage in, finding label noise](/blog/machine-learning/debugging-training/garbage-in-finding-label-noise), which goes deep on confident learning; here the point is simply that thirty seconds of reading the top-50 loss list catches the noise that the average loss hid.

### 4.6 NLP: the token-length and special-token audit

For text, the audit's center of gravity shifts to steps 2, 3, and 5, but the unit of inspection is the *tokenized* sequence, not the raw string — because the model never sees your string, it sees token IDs, and the gap between the two is where NLP data bugs live. The single most common silent NLP bug is truncation eating the answer: a tokenizer with `max_length=512` quietly drops everything past the limit, and if your label depends on text near the end of a long document, you are training on inputs that no longer contain the signal.

```python
import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

# Step 2 for text: the token-length distribution and the truncation rate.
lengths = np.array([len(tok(t, truncation=False).input_ids) for t in texts])
max_len = 512
trunc_rate = float((lengths > max_len).mean())
print(f"median={np.median(lengths):.0f}  p95={np.percentile(lengths, 95):.0f}  "
      f"max={lengths.max()}  truncated@{max_len}={trunc_rate:.1%}")

# Step 3 for text: decode what the model ACTUALLY sees after tokenize+truncate.
for i in np.argsort(lengths)[-3:]:                    # the 3 longest docs
    ids = tok(texts[i], truncation=True, max_length=max_len).input_ids
    seen = tok.decode(ids)
    print(f"\nlabel={labels[i]}  orig_tokens={lengths[i]}  kept={len(ids)}")
    print(f"  ...tail the model sees: {seen[-200:]!r}")

# Special-token sanity: is BOS/EOS present exactly once? Double-BOS is a real bug.
ids = tok("hello world").input_ids
print("first/last:", tok.convert_ids_to_tokens([ids[0], ids[-1]]))
```

When `truncated@512` came back at 14% on one document-classification project, it meant one in seven training examples had its tail amputated — and for a contract-classification task where the decisive clause was usually near the end, that 14% was nearly all of the hard, valuable examples. The fix (sliding-window chunking, or a longer-context model) only became obvious *after* the audit printed the truncation rate. No aggregate metric shows you this; the length distribution and a decoded tail show it in three lines. The special-token check catches the equally common double-BOS bug, where a chat template and the tokenizer each add a beginning-of-sequence token and the model trains on a malformed prefix.

### 4.7 Speech: the waveform, the spectrogram, and the transcript

For audio the audit is steps 2, 3, and 5 again, but you look *and listen*: render the waveform and the mel-spectrogram next to the transcript, and check the sample rate the way a tabular audit checks a feature range. The classic silent speech bug is a sample-rate mismatch — audio stored at 44.1 kHz fed to a model expecting 16 kHz, which either errors loudly (good) or silently resamples or misreads the duration (bad, and accuracy quietly tanks).

```python
import torch, torchaudio
import matplotlib.pyplot as plt

wav, sr = torchaudio.load("clip.wav")
print(f"sample_rate={sr}  shape={tuple(wav.shape)}  "
      f"duration={wav.shape[-1]/sr:.2f}s  peak={wav.abs().max():.3f}")

# Step 2 for audio: flag the wrong sample rate and clipping BEFORE training.
assert sr == 16000, f"expected 16k, got {sr} -- resample or your features are wrong"
if wav.abs().max() >= 0.999:
    print("WARNING: waveform is clipping (peak at full scale)")

# Step 3 for audio: spectrogram + transcript, the visual+text pairing.
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr, n_fft=400, hop_length=160, n_mels=80)(wav)
plt.figure(figsize=(9, 3))
plt.imshow(mel.log2()[0].numpy(), aspect="auto", origin="lower")
plt.title(f"transcript: {transcript[:60]!r}")
plt.savefig("audit_spec.png", dpi=110)
```

A flat or empty spectrogram next to a non-empty transcript means the audio failed to decode or is silence — a broken example that would otherwise teach the model that silence maps to words. The `n_fft`/`hop_length`/`n_mels` triple is the audio analogue of the ImageNet mean/std: if those feature-extractor parameters disagree with what the pretrained model expects, training proceeds and accuracy silently suffers, and only a side-by-side look at the spectrogram (does it have the right resolution and shape?) reveals it.

## 5. Before and after: what thirty minutes actually buys

Let me put hard numbers on both runs from the running example. The figure below contrasts the two paths — skipping the audit and losing the week, versus running it and catching the bugs cold — and the table beneath it gives the instrument readings.

![A two-column before-after figure contrasting a skipped audit that wasted a week against a thirty-minute audit that caught the bugs before training, with the cost of each path](/imgs/blogs/look-at-your-data-before-you-train-3.png)

| Signal | Skip the audit (what happened) | Run the audit (what would have) |
| --- | --- | --- |
| Vision val accuracy | 94% (leaked, looked great) | 79% honest after dedup re-split |
| Vision production accuracy | 71% (23-pt collapse) | 78% (matches honest val) |
| Tabular val AUC | 0.971 (sentinel leak) | 0.81 honest after sentinel fix |
| Tabular production AUC | 0.78 (0.19 collapse) | 0.80 (matches honest val) |
| Time to root cause | 6–7 days, post-hoc | ~30 minutes, pre-training |
| Wasted compute | ~750 GPU-hours of retraining | ~0 (caught before the real run) |
| Estimated cost | ≈ \$4,800 (compute + salary) | ≈ \$25 (half an hour of one engineer) |

Notice what the audit changes. It does *not* make the model better — the honest numbers (79% vision, 0.81 AUC) are *lower* than the leaked ones. That is the entire point. The audit replaces a flattering lie with an honest measurement, *before* you build a quarter's worth of plans on top of the lie. The 94% that became 71% in production was never real; the audit would have told you the true number was 79% on day zero, and you would have planned accordingly instead of discovering the truth in front of a stakeholder.

#### Worked example: the sentinel that inflated AUC by 0.16

Let us trace the tabular leak quantitatively so you can see exactly how a sentinel becomes a leaky feature. The `last_login_days` column had a `-999` sentinel for 1,840 of 20,000 training rows (9.2%). Crucially, those `-999` rows were *not random*: they were customers whose login data failed to join, and the join failed disproportionately for churned accounts (a deleted account has no recent login record). So in the training data, `last_login_days == -999` correlated with the churn label at roughly 0.62 — an enormous, free signal.

The model learned a near-rule: "if `last_login_days` is around `-999`, predict churn." On the training and validation sets (which shared the same broken join), this rule was gold and pushed AUC to 0.971. In production, the serving pipeline used a *different*, fixed join that never produced `-999` — it produced a proper `NaN` that the imputer filled with the median. The leaky signal vanished, and AUC collapsed to 0.78. The 0.16-AUC gap between the leaked 0.971 and a clean 0.81 is almost entirely attributable to that one sentinel.

The audit catches this at step 2 (the `min = -999` and the 1,840 sentinel hits) and confirms it at step 8 (the `-999` block exists in train and val but is absent from a fresh production sample — a glaring distribution mismatch between the offline data and the serving data). Two cheap steps, sixteen-percent of AUC saved from being a fiction. The timeline below traces the sentinel from where it was born to where it detonated, with the audit step that intercepts it at each stage.

![A timeline tracing the -999 sentinel from a failed data join through training and into the production collapse, marking the audit step that intercepts it at each stage](/imgs/blogs/look-at-your-data-before-you-train-7.png)

#### Worked example: a label flip that capped accuracy at 92%

The vision dataset also had a quieter bug the audit caught. Step 7's loss-ranked pass surfaced a cluster of 60 images, all labeled `"running_shoe"`, that were unambiguously sandals. A labeling vendor had merged two categories by mistake. Out of 10,000 training images, 60 mislabels is 0.6% — invisible in the aggregate loss (it shifts the mean from, say, 0.180 to 0.184). But here is the mathematics of why it still matters: those 60 images sit at the very top of the loss ranking with a mean per-example loss around 4.1 (the model is confidently *right* and being punished), so they are trivial to find by sorting, and they actively teach the model that some sandals are running shoes, blurring the decision boundary between the two most-confused classes.

After flipping those 60 labels back, the confusion between sandals and running shoes dropped and overall accuracy rose by 1.4 points — modest, but free, and found by reading 50 images for ninety seconds. The general law from Section 2.2 holds: the mislabels contributed almost nothing to the average loss (the bulk) but occupied the entire top of the loss-ranked tail, which is exactly where step 7 looks.

#### Worked example: how many examples must you read to find the noise?

A natural objection is that reading the data does not scale: "I have ten million rows, I cannot look at them all." You do not have to, and there is a clean piece of probability that says exactly how much you must read. The question is really: *if a fraction $p$ of my data is buggy, and I read a random sample of $k$ examples, what is the chance I see at least one bug?*

If bugs are sprinkled at rate $p$, the chance a single random example is *clean* is $(1-p)$, so the chance all $k$ sampled examples are clean is $(1-p)^k$, and the chance you catch *at least one* bug is

$$P(\text{catch} \ge 1) = 1 - (1-p)^k.$$

Suppose label noise is at the benchmark-typical rate of $p = 0.034$ (3.4%, from the confident-learning study). To be 95% sure of seeing at least one mislabel, you need $1 - (1-0.034)^k \ge 0.95$, which solves to $k \ge \ln(0.05)/\ln(0.966) \approx 87$ examples. **You need to read fewer than ninety random examples to be 95% confident you will spot the noise if it exists at the typical rate.** That is the rigorous answer to "the dataset is too big to look at": you were never going to look at all of it, and you do not need to. A random sample of about a hundred, plus the tails (steps 5 and 7, which find the *concentrated* noise far faster than random sampling), is enough. The loss-ranked step does even better than this random-sampling bound, because it does not sample randomly — it puts the noise at the top of the list, so the first 50 examples you read are enriched for bugs by orders of magnitude over the base rate.

## 6. The embedding scatter: seeing label noise and leakage geometrically

Step 6 deserves its own section because it is the one that *looks* like a luxury and is actually a workhorse. A 2-D embedding scatter, colored by label, turns label noise into something you can literally point at and turns leakage into a shape your eye refuses to accept. The figure below sketches the two diagnostic signatures.

![A graph showing how an embedding scatter reveals label noise as off-color points inside a cluster and leakage as suspiciously perfect class separation](/imgs/blogs/look-at-your-data-before-you-train-4.png)

The mechanism is simple. A good representation (a pretrained encoder, or even TF-IDF plus PCA for text) places semantically similar examples near each other. If you then color each point by its *label*, two diagnostic patterns jump out:

**Label noise looks like a wrong-colored point inside a cluster.** If you see a single red dot sitting deep inside a dense blue cluster, the data near it is all blue, so the red label is probably wrong. You can rank candidates by "fraction of k nearest neighbors with a different label" and read the top.

**Leakage looks like classes that are *too* separated.** If your two hardest-to-distinguish classes form perfectly clean, non-touching islands, be suspicious — real classes that are semantically close should overlap at the boundary. Perfect separation usually means a feature is leaking the label (an ID, a timestamp, a formatting artifact). This is the geometric twin of the "too-good val number" smell.

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# X = embeddings (n, d); y = integer labels.
emb2d = PCA(n_components=2).fit_transform(X)   # PCA is fast; UMAP if you have it

# Quantify label noise: fraction of neighbors that disagree with the label.
nn = NearestNeighbors(n_neighbors=11).fit(X)
_, idx = nn.kneighbors(X)
neigh_labels = y[idx[:, 1:]]                    # drop self
disagree = (neigh_labels != y[:, None]).mean(axis=1)

suspect = np.argsort(disagree)[::-1][:30]       # most-isolated-from-own-class
print("Most likely mislabeled (by neighbor disagreement):")
for i in suspect[:10]:
    print(f"  idx={i} label={y[i]} neighbor_disagreement={disagree[i]:.2f}")
```

A point whose ten nearest neighbors are 90% a *different* label is, with very high probability, either mislabeled or a genuinely ambiguous edge case — and both are worth your eyes. This neighbor-disagreement score is a poor-man's confident learning that needs no model training at all, only an encoder you already have.

There is a subtlety worth flagging so you read the scatter correctly. The 2-D projection (PCA, t-SNE, or UMAP) is itself a lossy compression — it can distort distances, create apparent clusters that are projection artifacts, and exaggerate or hide separation depending on its hyperparameters. t-SNE in particular is notorious for producing visually clean clusters whose *sizes* and *between-cluster distances* are not meaningful. So the rule is: use the *scatter* to spot candidates (the off-color point, the suspiciously clean island) but *confirm* in the original high-dimensional space with the neighbor-disagreement score, which never went through the projection. The plot generates hypotheses; the `kneighbors` query in full dimensionality tests them. Treating the projection as ground truth is the most common way people misread an embedding scatter, and it is why the code above computes disagreement on `X` (the full embeddings), not on `emb2d` (the projection).

Why does perfect separation indicate leakage rather than a great model? Because of a base-rate argument. For two genuinely confusable classes — say, two visually similar dog breeds, or two near-synonymous intents — some fraction of examples *must* live near the decision boundary; that is what "confusable" means. If your embedding shows them as two perfectly disjoint blobs with empty space between, the model is using a feature that is *more* discriminative than the semantic content itself. The usual culprit is a non-semantic artifact: a watermark that appears only on one class's images, a date range that perfectly splits the labels, a formatting quirk in how one class was scraped. The geometry is telling you that *something other than the meaning* separates the classes cleanly — which is the literal definition of a leak. The honest version of the same scatter has fuzzy, overlapping boundaries, and a model trained on it generalizes because it had to learn the hard, real signal rather than the easy, leaky one.

## 7. The problem-solving narrative: bisecting a "good model that fails in prod"

Now let us run the audit as a real debugging session, the way you would when the symptom has already happened and you are bisecting across the six places a bug can hide. The figure below is the decision graph: a strong-offline / weak-online gap routes you straight into the data branch, and the audit steps are the confirming tests.

![A decision graph that routes a strong-offline weak-online symptom into the data branch and assigns each candidate cause its confirming audit step](/imgs/blogs/look-at-your-data-before-you-train-5.png)

**The symptom.** Validation AUC 0.97, production AUC 0.78. The first bisection question is always the same: *is this a data problem, an optimization problem, a model-code problem, a numerics problem, a systems problem, or an evaluation problem?* The signature — clean training, strong offline, weak online, no NaN, no instability — points hard at data or evaluation, not at optimization or numerics. A smooth-then-degrading-in-prod curve is almost never a learning-rate bug. So we go into the data/eval branch and run the audit.

**Step 1 (counts):** class balance is 11% positive, 89% negative. Imbalanced but expected for churn; not the bug. Cross off.

**Step 2 (distributions):** the audit flags `last_login_days` with `min = -999` and 1,840 sentinel hits. This is a strong suspect. We do not stop here — a suspect is not a confirmation — but we mark it.

**Step 8 (split audit), targeted:** we pull a fresh sample of *production* feature rows and compare distributions to training. The `-999` block is present in train and val and *absent* in production. That is the confirming test: the feature behaves differently offline and online, which is the definition of train-serving skew. The bisection is complete — the bug is in the data, specifically a sentinel-driven leaky feature, and the production join differs from the training join.

**The fix and the proof.** Replace `-999` with `NaN`, add a missing-indicator, retrain, and re-measure. Honest validation AUC drops to 0.81 (the leak is gone) and production AUC rises to 0.80 (the offline and online numbers now agree). The gap closed not because the model got better but because we stopped measuring a fiction. The whole diagnosis — symptom to confirmed root cause — took two audit steps and under fifteen minutes.

### 7.1 Stress-testing the diagnosis across modalities and conditions

A good debugger does not stop at the first plausible cause; they ask what *else* could produce the same signature and rule it out.

*What if it were not the sentinel?* A strong-offline / weak-online gap can also come from (a) a near-duplicate leak across splits (step 4 would catch it — and did, in the vision twin of this story), (b) temporal leakage where the validation period overlaps the training period (step 8's time-range check catches it), or (c) an evaluation-set leak where the test labels themselves are contaminated. The audit's value is that it checks *all* of these cheaply, so you confirm the real cause instead of fixating on the first one.

*What if the modality were vision instead of tabular?* The same step numbers map to different code. Step 2's "sentinel" becomes step 3's "all-gray decoded image" or "blue-tinted BGR." Step 4's duplicate detection becomes perceptual hashing instead of value hashing. The ritual is invariant; the implementation per step changes. That is why the matrix figure in Section 3 has a modality axis.

*What if the dataset is enormous and you cannot look at all of it?* You do not look at all of it — you never did. You look at a few hundred rows for steps 1–3 and 5, you sample for the embedding scatter, and you let steps 4 and 7 (hashing, loss-ranking) scale to the full set automatically. The audit's cost is roughly constant in dataset size because the human-attention steps operate on samples and tails, not the bulk.

*What if looking introduces its own bias?* It can — see Section 10. The fix is to look at a *random* sample plus the *tails*, never just the convenient first rows, and to write down what you expected before you look so you do not rationalize whatever you find.

*What if the audit comes back clean but production still fails?* Then you have learned something genuinely valuable: the bug is *not* in the data, and you have ruled out the single most common branch of the bisection tree in thirty minutes. Now the remaining suspects — a train-serve code-path divergence, an evaluation-metric mismatch, a numerics issue at serving precision — get your full attention, undiluted by the nagging worry that you skipped the obvious data check. A clean audit is not a wasted thirty minutes; it is a confidently-eliminated hypothesis, which is exactly what bisection is made of. The cost of *not* running it is that you spend three days on optimization theories while a `-999` sits in the data, mocking you.

*What if two data bugs are present at once?* They usually are, and the audit handles it naturally because each step targets a different bug class. The sentinel (step 2) and the near-duplicate leak (step 4) in our running example were *both* present and *both* found, in different steps, in the same thirty-minute pass. This is an argument for running all eight steps even after the first one finds something: the first bug you find is rarely the only bug, and the marginal cost of the remaining steps is low. Stopping at the first red flag is how you fix one bug, declare victory, and get surprised again next week by the second one you never looked for.

## 8. Case studies: real datasets where looking found the bug

These are documented, public results — the audit discipline at industrial scale.

**Confident learning finds label errors in benchmark test sets (Northcutt, Athalye, Mueller, 2021).** The team behind `cleanlab` ran confident learning — the principled form of step 7 — across ten of the most-cited ML benchmarks, including ImageNet, MNIST, CIFAR-10/100, and several NLP datasets. They estimated an average of around 3.4% label errors in the *test* sets, with about 6% in ImageNet's validation set and roughly 10% in QuickDraw. The headline finding is sobering: for some benchmarks, correcting the label errors *changed which model was declared state of the art*. The bugs were not in anyone's training code; they were in the labels, sitting at the top of the loss ranking the whole time, waiting for someone to sort and look. (Numbers are from the published paper; treat the exact percentages as the authors' estimates.)

**The -999 / sentinel-as-feature pattern in tabular competitions.** Across many Kaggle post-mortems, a recurring leakage pattern is exactly the one in our worked example: a sentinel or magic value (`-1`, `-999`, an out-of-range date, a placeholder ID) that correlates with the target in the training data because of *how the data was collected*, and vanishes or changes at serving time. The signature is always the same — a suspiciously strong single feature, a large train/serve AUC gap, and a sentinel value that step 2's range check exposes in one line. The exact AUC deltas vary by competition, but the *mechanism* is universal and the *detector* is a `min`/`max` and a sentinel count.

**The BGR/RGB and normalization-mismatch class of vision bugs.** A perennial, well-documented failure: training images decoded with OpenCV (which defaults to BGR) and fed to a model whose preprocessing assumed RGB, or normalized with ImageNet statistics on the wrong scale (0–255 vs 0–1). The model still trains and the loss still falls — it learns the color-swapped distribution — but accuracy is left several points on the table, and the bug is invisible to every aggregate metric. Step 3, the decoded grid, makes it instantly visible: the images look wrong to a human. This is detailed further in [the input pipeline is lying to you](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you).

**Near-duplicate contamination across train/test in large image datasets.** Multiple analyses of popular image benchmarks have found non-trivial fractions of near-duplicate images straddling the train/test boundary (for example, CIFAR-10 and CIFAR-100 have documented duplicate and near-duplicate rates in the low single-digit percentages between train and test). Any such overlap inflates the reported test number, and the only reliable detector is step 4 — perceptual hashing or embedding similarity across the split — not a glance at aggregate statistics. The exact fractions are dataset-specific and reported in the respective analyses; the structural point is that you cannot trust a split you have not audited for duplicates.

## 9. Making the audit happen every time — turn it into a gate

The hardest part of the data audit is not the code; it is the discipline. Under deadline pressure, "look at your data first" is the step everyone skips, and it is skipped precisely on the runs that need it most — the rushed ones. The fix is to stop relying on willpower and turn the cheap, automatable parts of the audit into a *gate*: a script that runs in CI or as the first stage of your training job, and that *fails the run* if it finds a red flag. The expensive, human-eye parts (the grid, the scatter) stay manual, but the mechanical parts (sentinel detection, dup detection, split disjointness) become a wall the data must pass before a single GPU-hour is spent.

Here is the pattern: a single `assert_clean()` function that encodes the non-negotiable invariants and raises before training starts.

```python
import hashlib
import numpy as np
import pandas as pd

def assert_clean(train_df, val_df, target, id_col, sentinels=(-999, -9999)):
    """Fail loudly BEFORE training if the data violates a hard invariant."""

    # Invariant 1: no empty classes in train (step 1).
    counts = train_df[target].value_counts()
    assert (counts > 0).all(), f"empty class detected: {counts.to_dict()}"

    # Invariant 2: no known sentinel values survived preprocessing (step 2).
    num = train_df.select_dtypes("number")
    for s in sentinels:
        hits = int((num == s).sum().sum())
        assert hits == 0, f"{hits} sentinel({s}) values reached training data"

    # Invariant 3: train and val are disjoint by id (step 8).
    overlap = set(train_df[id_col]) & set(val_df[id_col])
    assert not overlap, f"{len(overlap)} ids appear in BOTH train and val"

    # Invariant 4: no exact-duplicate rows leaking across the split (step 4).
    def row_hash(df):
        return df.drop(columns=[target]).apply(
            lambda r: hashlib.md5(r.to_string().encode()).hexdigest(), axis=1)
    leaked = set(row_hash(train_df)) & set(row_hash(val_df))
    assert not leaked, f"{len(leaked)} byte-identical rows span train and val"

    print("data audit PASSED: no empty class, sentinel, id-overlap, or dup leak")

# First line of your training entrypoint, before the model is even built:
assert_clean(train_df, val_df, target="churn", id_col="customer_id")
```

This is the single most important habit-forming move in the whole post. An assert at the top of the training script costs nothing on a clean dataset and saves a week on a dirty one, and — crucially — it runs *whether or not the engineer remembered to look*. The manual steps still matter (an assert cannot see a BGR swap or read a loss-ranked list), so the gate does not replace the eyes; it backstops them. The right architecture is: automated invariants in CI as a hard gate, plus a scheduled human audit (the grid, the scatter, the loss-ranked read) before every major run. The gate catches the catastrophic mechanical bugs reliably; the human catches the subtle semantic ones occasionally — and together they cover the space.

A few field notes on running this in practice. Keep the sentinel list versioned and growing: every time a new magic value bites you, add it to `sentinels` so it can never bite twice. Run `assert_clean` on the *post-preprocessing* data, not the raw data — the whole point is to verify that your cleaning code actually removed the sentinel, not that the sentinel exists in the raw file (it often legitimately does). And make the assertions *loud and specific*: a failure that says `"1,840 sentinel(-999) values reached training data"` tells the on-call engineer exactly what to fix, whereas a generic `"data check failed"` sends them back into the dark. The cost of a precise assertion message is one f-string; the value is the difference between a five-minute fix and another afternoon of bisection.

This gate is also where the audit connects back to the rest of your pipeline hygiene: the disjoint-split invariant is the mechanical enforcement of [data leakage, the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer), and the empty-class invariant feeds directly into the imbalance handling of [class imbalance and when accuracy lies](/blog/machine-learning/debugging-training/class-imbalance-and-when-accuracy-lies). A good gate is the union of the cheap checks from every data-debugging post in this series, run automatically, every time.

One more refinement makes the gate dramatically more useful over time: log the audit's *outputs*, not just its pass/fail. Record the class histogram, the per-feature min/max/missing table, the truncation rate, and the count of cross-split duplicates as artifacts attached to every run. The first time you do this it is mildly interesting; the tenth time it is invaluable, because now you can *diff the data audit between two runs* and instantly see when a feature's range shifted, when a class disappeared, or when the duplicate count jumped — which is exactly the signature of an upstream data-pipeline change that someone made without telling you. Many of the worst production incidents are not a bug in *your* code but a silent change in *someone else's* data feed, and a logged, diffable audit turns those from week-long mysteries into a one-line "the `country` column went from 47 unique values to 3 between Tuesday and Wednesday." The audit you run once protects this run; the audit you log and diff protects every run after it. Treat the audit outputs as first-class run artifacts, version them next to your model checkpoints, and you convert a one-time discipline into a permanent early-warning system for your whole data supply chain.

## 10. When this is (and isn't) your bug

The audit is the first thing to run, but it is not a universal explanation. Here is when a symptom points *away* from the data and into another of the six places. The decision tree below routes a symptom to the right suspect so you do not run a data audit on a numerics bug.

![A decision tree routing a training symptom to the right suspect, separating data bugs from optimization, numerics, and evaluation bugs by their distinct signatures](/imgs/blogs/look-at-your-data-before-you-train-8.png)

**If overfit-one-batch fails, it is probably not the data — it is the model or optimization.** If your model cannot drive loss to zero on sixteen fixed examples (see [the overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test)), the audit will not save you; the path from output to label is broken in code, not corrupted in data. Run the overfit test first if you are unsure which side you are on.

**A smooth-then-NaN curve is numerics, not data.** If training is stable and then explodes into NaN at step 4,000, that is almost always a numerical or learning-rate problem, not a label issue. Data bugs tend to produce *quiet* failures (a clean curve, a wrong model), not *loud* ones (a NaN). Looking at the data is the wrong tool for a NaN; the right tools are in the numerics track.

**If the training and serving code paths genuinely match and the data is clean, look at evaluation.** When the audit comes back clean and overfit-one-batch passes, but offline still beats online, the bug may be in how you *measure* offline — an eval-set leak, a metric that does not match the goal, a threshold tuned on the test fold. That is an evaluation bug, not a data bug, though the line is blurry (the split audit, step 8, straddles both).

**Beware confirmation bias when you look.** Looking has a real failure mode: you see what you expect. If you eyeball the first ten rows (which are often sorted, or all from one class) you will conclude the data is fine when it is not. The defenses are mechanical: always look at a *random* sample, always look at the *tails* (sorted extremes), always look at the *high-loss* examples, and write down your prior expectation before you look so you cannot rationalize after the fact. The audit's eight steps are structured precisely to force you off the convenient first rows and onto the parts of the distribution where bugs actually live.

**Do not over-clean.** Finally, the audit can be misused in the other direction: deleting every high-loss example until the model looks perfect on a sanitized set. High-loss examples include genuinely hard, genuinely valuable cases, not just noise. The goal is to *fix* mislabels and *understand* edge cases, not to launder the dataset into something easy. Drop only what is provably broken; relabel what is fixable; keep the hard-but-correct.

## 11. Key takeaways

- **Run the audit before the first real training run, every time.** Eight steps, thirty minutes; it collapses the most common bug branch (data) before you spend a single GPU-hour. The audit is **make-it-fail-small** for data.
- **Summary statistics are lossy; looking is not.** Anscombe's quartet and the Datasaurus prove that identical means, variances, and correlations can hide wildly different — and broken — distributions. The mean cannot see the `-999`; the histogram and the sort can.
- **Order the steps by cost: counts first, loss-ranking last.** `value_counts()` is one line and catches empty classes and label typos; loss-ranking needs a quick model but yields the most bugs per example you read.
- **The tails dominate failures.** Mislabels and broken examples contribute almost nothing to the average loss but occupy the entire top of the loss ranking and the extremes of any sort — so sort, and read the tails.
- **Duplicate detection across splits is non-negotiable.** A near-duplicate leak makes a model look 15–23 points better offline than it is. Hash for exact dupes; use perceptual hash or embedding similarity for near-dupes; audit across the train/val boundary.
- **A sentinel is a leaky feature waiting to happen.** `-999`, `-1`, magic dates, placeholder IDs — if they correlate with the label offline and vanish online, they inflate your metric and collapse in production. Step 2's `min`/`max` and a sentinel count expose them in one line.
- **An embedding scatter colored by label turns noise and leakage into shapes.** Wrong-colored point inside a cluster = probable mislabel; suspiciously perfect class separation = probable leak.
- **Decode and look at raw examples.** A grid of decoded images, a printed detokenized string, a waveform with its transcript — this is where BGR/RGB, double-normalization, and obvious mislabels become visible to a human in seconds.
- **The audit replaces a flattering lie with an honest number.** The honest metric is usually *lower* than the leaked one. That is the win: you learn the truth on day zero instead of in front of a stakeholder.
- **Know when it is not the data.** If overfit-one-batch fails it is code; if it is smooth-then-NaN it is numerics; if the data is clean and offline still beats online it is evaluation. The audit is the first test, not the only one — feed the result back into the [taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs).

## 12. Further reading

- Andrej Karpathy, "A Recipe for Training Neural Networks" (2019) — the canonical "become one with the data" essay; step 1 of his recipe is, literally, look at your data.
- Curtis Northcutt, Anish Athalye, Jonas Mueller, "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks" (NeurIPS 2021) — the confident-learning study that found ~3.4% average label errors across ten benchmarks.
- Curtis Northcutt et al., "Confident Learning: Estimating Uncertainty in Dataset Labels" (JAIR 2021) — the theory behind step 7 and the `cleanlab` library.
- Justin Matejka, George Fitzmaurice, "Same Stats, Different Graphs: Generating Datasets with Varied Appearance and Identical Statistics" (CHI 2017) — the Datasaurus Dozen; the formal case that you must plot, not summarize.
- Frank Anscombe, "Graphs in Statistical Analysis" (The American Statistician, 1973) — the original quartet.
- `cleanlab` documentation — runnable confident-learning for finding label issues in any classification dataset.
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the six-places decision tree this audit short-circuits), [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) (the capstone checklist), [garbage in, finding label noise](/blog/machine-learning/debugging-training/garbage-in-finding-label-noise) (deep on confident learning), [data leakage, the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer) (the split-audit bug in depth), [the input pipeline is lying to you](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you) (decode and collation bugs), and [class imbalance and when accuracy lies](/blog/machine-learning/debugging-training/class-imbalance-and-when-accuracy-lies) (what step 1's counts feed into).

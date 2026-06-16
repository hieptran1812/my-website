---
title: "The Input Pipeline Is Lying to You: DataLoader Bugs That Quietly Wreck Training"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Find the DataLoader and Dataset bugs that change the batch your model sees without ever raising an error, with a print-the-batch discipline that localizes each one in minutes."
tags:
  [
    "debugging",
    "model-training",
    "dataloader",
    "data-pipeline",
    "pytorch",
    "augmentation",
    "normalization",
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
image: "/imgs/blogs/the-input-pipeline-is-lying-to-you-1.png"
---

Here is a run that looks healthy from across the room. A ResNet-50 on a 100-class image dataset, loss falling from `4.6` to `1.9` over the first epoch, the curve smooth enough to screenshot for the standup. You let it cook overnight. In the morning, training loss is `0.31` and validation accuracy is stuck at `1.0%` — exactly chance for 100 classes. You did not write a bug in the model. You did not write a bug in the optimizer. The model learned *something* well enough to drive training loss toward zero. It just learned a function of the *wrong inputs*, because the batch your `DataLoader` handed it overnight was not the batch you thought you wrote down. The augmentation that was supposed to run only on training images ran on validation images too, with a different random crop each time, so the model memorized the training pixels and saw noise at eval. Nothing crashed. Nothing warned. The pipeline lied, quietly, for nine hours and forty dollars of GPU time.

This is the most under-appreciated failure surface in all of deep learning, and it is under-appreciated for a precise reason: **the input pipeline almost never raises an exception when it is wrong.** A shape bug crashes. A NaN poisons the loss and screams. But a `DataLoader` that shuffles labels independently of inputs, or pads a variable-length batch with a token that means something, or recomputes normalization statistics at eval time, will run to completion and report numbers. The numbers are just measurements of the wrong experiment. In the six-places framing this series is built on — a bug hides in **data, optimization, model code, numerics, systems, or evaluation** — pipeline bugs live in *data*, and they are the single most common cause of a run that "trains" but learns nothing. Figure 1 is the map: between the raw files on disk and the tensor that reaches the GPU, there are five transform stages, and a bug at any one of them rewrites the batch without a trace.

![Stack diagram showing the five stages a batch passes through, from raw files on disk through Dataset.__getitem__ returning an example, transforms that augment and normalize, collate_fn that pads and stacks N items, the batch on the GPU, and finally the loss that may see the wrong input or wrong label](/imgs/blogs/the-input-pipeline-is-lying-to-you-1.png)

By the end of this post you will be able to take any stalled-or-secretly-broken run — vision, NLP, tabular, or speech — and decide in a few minutes whether the input pipeline is the culprit, then localize *which* of nine specific bugs it is. We will build the discipline that makes this fast: **pull one batch, look at it, and refuse to train until you understand every number in it.** We will write a single reusable `inspect_batch()` that prints shapes, dtypes, value ranges, the label histogram, and a hash of the contents — and we will see exactly which bug each of those readouts catches. The spine, as always, is the same two master tools: **make-it-fail-small** (one batch, one worker, two epochs you can compare) and **read the instruments** (the batch itself is the instrument here). This post is B3 in the series. It instantiates the [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for the data branch, leans on [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) for the worker-RNG mechanism, and feeds the capstone [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook).

## 1. The symptom: your model learned the wrong thing, and nothing told you

Let me make the running example concrete, because we are going to debug it all the way down. You have a vision classifier and a `Dataset`/`DataLoader` you wrote yourself — not a tutorial copy, a real one with augmentation, normalization, and a custom `collate_fn` because your images come in two sizes. You see one of these four signatures, and each points somewhere different:

| Signature | What you observe | First suspect | Why |
| --- | --- | --- | --- |
| Saw-tooth loss | Loss oscillates with a clear per-epoch period | Shuffle off / sorted data | SGD is seeing correlated gradients |
| Train great, eval chance | Train loss `0.3`, val acc at random | Augmentation or norm at eval | The eval batch is not the eval data |
| Loss fine, no generalization | Both losses drop, test in prod is bad | Leakage or label misalignment | The model learned a shortcut in the pipeline |
| Periodic NaN at variable length | NaN only on some batches | Padding / collate bug | A meaningful pad value entered the math |

Notice what these have in common: the *optimizer is doing its job*. Gradients flow, weights update, training loss falls. That is exactly why pipeline bugs are so corrosive — they pass the cheapest sanity check (loss goes down) and fail the expensive one (the model is useless). The disciplined move is to stop staring at the loss curve and **go look at the batch**. The loss curve is a thousand-step summary; the batch is the ground truth. If the batch is wrong, every downstream number is a measurement of a different experiment than the one you meant to run.

Before any code, internalize the bisection. A training step is a function of three inputs: the parameters, the batch, and the floating-point operations. If parameters are pinned (init seed) and numerics are pinned (determinism), then a run that misbehaves and a run that behaves *differ only in the batch*. So we hold everything else fixed and interrogate the batch directly. That is the whole method. Everything below is a specific question to ask the batch and the specific bug a wrong answer reveals.

## 2. Bug one: shuffle off, and the science of why SGD breaks on sorted data

The most common pipeline bug is also the easiest to write: you forget `shuffle=True`, or you shuffle a list of labels independently from the list of inputs, or your data happens to arrive sorted by class on disk and you never randomized it. The `DataLoader` default is `shuffle=False`. If your `Dataset` is built from a directory walk — `ImageFolder` enumerates class folders in order — then batch 1 is all of class 0, batch 2 is all of class 0, and you don't reach class 1 until thousands of steps in.

### 2.1 The science: SGD assumes i.i.d. samples, and sorted data violates it

Stochastic gradient descent estimates the full-dataset gradient with a mini-batch. The estimate is only useful if the mini-batch gradient is an **unbiased** estimator of the true gradient with **bounded variance**. Write the true loss as the average over the dataset,

$$
L(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell(x_i, y_i; \theta),
$$

and the mini-batch estimate over a batch $B$ of size $b$ as

$$
\hat{g}_B(\theta) = \frac{1}{b}\sum_{i \in B} \nabla_\theta \ell(x_i, y_i; \theta).
$$

For SGD to converge to a minimizer of $L$, the standard requirement is $\mathbb{E}_B[\hat{g}_B] = \nabla_\theta L$ — the batch gradient is unbiased. That expectation is over batches drawn **uniformly at random**. The instant your batches are class-sorted, $B$ is no longer a uniform random subset; it is a contiguous block of one class. The expected gradient over an *all-cats* batch points toward "predict cat for everything," and the expected gradient over an *all-dogs* batch points the opposite way. Successive steps fight each other. The optimizer is being pulled by a gradient whose direction has a strong, low-frequency component synchronized with the data order. The result is the saw-tooth: loss drops within a class block as the model overfits that class, then jumps when the next class arrives.

There is a second, quieter cost: **gradient autocorrelation**. With random batches, consecutive gradients are nearly independent, so their average over a few steps cancels noise and reveals signal. With sorted batches, consecutive gradients are highly correlated, so averaging does not cancel anything — you have effectively reduced your independent-sample count by a large factor. Momentum makes this worse, not better: it integrates the correlated signal, so the velocity vector locks onto the current class and lags badly when the class changes. Figure 2 is the before-and-after: sorted data gives saw-tooth loss and chance accuracy; one flag flips it to a smooth descent and real generalization.

Let me make the autocorrelation cost quantitative, because the size of the penalty is what tells you whether to care. The whole reason mini-batch SGD works is **variance reduction**: averaging $b$ independent per-example gradients shrinks the variance of the estimate by a factor of $b$. But that $\frac{1}{b}$ only holds when the $b$ examples are *independent*. For correlated samples, the variance of the mean of $b$ random variables with pairwise correlation $\rho$ is

$$
\operatorname{Var}\!\left(\frac{1}{b}\sum_{i=1}^{b} g_i\right) = \frac{\sigma^2}{b}\Big[1 + (b-1)\rho\Big].
$$

When $\rho = 0$ (i.i.d., the shuffled case) the bracket is `1` and you get the full $\frac{1}{b}$ reduction. When $\rho \to 1$ (a batch of near-identical, same-class examples — the sorted case) the bracket approaches $b$, the two factors cancel, and the variance is $\sigma^2$ — *as if you had a batch of size one*. The **effective batch size** is $b_{\text{eff}} = b / [1 + (b-1)\rho]$. With $b = 128$ and a within-class gradient correlation of even $\rho = 0.5$, $b_{\text{eff}} \approx 2.5$. You paid for a 128-image batch and you are getting the gradient quality of a 2- or 3-image batch. That is why sorted-data training is not just bumpy but *slow and unstable*: your gradient estimate is enormously noisier than the batch size suggests, the learning rate you tuned for an effective batch of 128 is now far too high for an effective batch of 3, and the run either crawls or diverges. Shuffling drives $\rho$ toward zero and restores the full variance reduction — which is why the single keyword argument has such an outsized effect on the curve.

![Before-and-after diagram contrasting shuffle off, where class-sorted batches produce correlated gradients with a per-epoch period and saw-tooth loss at fifty percent validation accuracy, with shuffle on, where mixed near-i.i.d. batches give unbiased low-autocorrelation gradients, a smooth loss, and ninety-three percent validation accuracy](/imgs/blogs/the-input-pipeline-is-lying-to-you-2.png)

### 2.2 The diagnostic: hash two epochs, and look at the label histogram

There are two confirming tests, and you should run both because they catch two different versions of the bug.

First, the **label histogram of a single batch**. If shuffle is off and data is class-sorted, the first batch's labels are all one value. One line tells you:

```python
import torch

batch = next(iter(train_loader))
x, y = batch                      # adjust to your batch structure
print("label histogram:", torch.bincount(y))
# healthy (10 classes, batch 128): tensor([14, 11, 13, 12, 15, 10, 13, 12, 14, 14])
# shuffle-off:                     tensor([128,  0,  0,  0,  0,  0,  0,  0,  0,  0])
```

A batch that is all one class when your dataset has many is an instant, unambiguous diagnosis. But this misses the subtler variant where the data *is* shuffled within an epoch but the **shuffle is identical every epoch** (a fixed permutation), or where you shuffled but with a generator that resets. For that, hash the *batch order across two epochs* and confirm they differ:

```python
import hashlib

def epoch_order_hash(loader, n_batches=20):
    h = hashlib.sha1()
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        # hash the labels' order; cheap and order-sensitive
        h.update(y.numpy().tobytes())
    return h.hexdigest()[:12]

e1 = epoch_order_hash(train_loader)
e2 = epoch_order_hash(train_loader)
print("epoch 1 order:", e1)
print("epoch 2 order:", e2)
assert e1 != e2, "Batch order is identical across epochs: shuffle is off or its RNG is fixed."
```

If the two hashes match, the loader is feeding the same sequence every epoch. With `shuffle=True` and a properly advancing generator, they must differ. This is the cheapest reproducibility-flavored check in the whole pipeline, and it is worth wiring into a test that runs before every long job.

#### Worked example: reading the saw-tooth to confirm shuffle is the bug

Suppose you log loss every 50 steps for one epoch of 1,000 steps over a 10-class, class-sorted dataset. You see loss values like `2.30, 1.10, 2.25, 1.05, 2.28, 1.08, ...` — a clean two-step oscillation. The period is the giveaway. Your epoch has 1,000 steps and 10 classes, so each class block is ~100 steps; if the saw-tooth period matches the class-block length, the cause is data order, full stop. The within-block dip (`2.30 → 1.10`) is the model overfitting the current class; the jump back up is the next class arriving and the predictions being wrong. Set `shuffle=True`, rerun: the loss now reads `2.30, 1.85, 1.52, 1.31, 1.18, ...`, monotone, and the per-epoch period is gone. Validation accuracy goes from `10.4%` (chance) to `0.91`. The fix is one keyword argument, but you only *know* it was the bug because you matched the oscillation period to the class-block length. That is the difference between guessing and debugging.

### 2.3 The fix, and the trap of shuffling labels separately

The fix is `shuffle=True` in the `DataLoader`, or a `RandomSampler`. The trap to avoid is shuffling your inputs and labels in two separate operations:

```python
# WRONG: two independent shuffles desynchronize x and y
import numpy as np
np.random.shuffle(X)   # permutes X
np.random.shuffle(Y)   # different permutation — labels no longer match inputs!

# RIGHT: one permutation applied to both
perm = np.random.permutation(len(X))
X, Y = X[perm], Y[perm]
```

The two-shuffle bug is insidious because the loss still *drops* — the model finds the best constant-ish mapping it can to noise — but accuracy never beats chance, and the label histogram looks perfectly balanced. The catch for this one is not the histogram; it is **visualize a few examples with their labels** (Section 7). If image after image shows a cat captioned "dog," the labels are scrambled. Always shuffle inputs and targets with the *same* index permutation, ideally by keeping them paired inside a single `Dataset` so there is no opportunity to desynchronize.

It is worth being precise about *why* a scrambled-label run still drives loss down, because the falling loss is exactly what fools people into thinking the model is learning. With permuted labels, there is no learnable relationship between $x$ and $y$ — the mutual information $I(x; y)$ is zero by construction. The lowest loss any model can reach is the loss of predicting the marginal label distribution $p(y)$ regardless of input. For balanced 10-class data, that floor is $\log 10 \approx 2.30$ nats of cross-entropy, and a model with enough capacity will memorize the training set down toward that floor *and below* by overfitting individual noise points — so training loss can drop to `0.5` or lower while the model has learned nothing generalizable. Validation accuracy sits at exactly `1/K` (chance) because there is no signal to transfer. The diagnostic signature is therefore unmistakable once you know to look for it: **train loss falls, but val accuracy is pinned at chance and never moves.** Any run with that pair of facts has either scrambled labels or a leak that the test set does not share, and the fastest confirmation is to put four images on screen next to their labels and read them with your own eyes. A single mismatched pair settles it in seconds.

## 3. Bug two and three: augmentation and normalization in the wrong place

These two bugs are cousins, and together they explain most "train great, eval terrible" runs that are not overfitting. The principle: **augmentation is a training-time-only operation; normalization is a fixed deterministic operation that must be byte-identical at train and eval.** Swap either, and the eval distribution stops matching the train distribution.

### 3.1 Augmentation leaking into eval

Random crops, flips, color jitter, and the like exist to enlarge the *training* distribution so the model generalizes. Applied at eval they do the opposite: they inject noise into the very batch you are scoring, so your validation number measures performance on randomly-corrupted inputs, not clean ones. The classic shape of this bug is a single `transform` shared by both loaders:

```python
# WRONG: one transform pipeline used for train AND val
transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
train_ds = ImageFolder("train", transform=transform)
val_ds   = ImageFolder("val",   transform=transform)   # augmenting val!
```

`RandomResizedCrop` at eval throws away most of each validation image and keeps a random patch — your "accuracy" is now accuracy on random patches, which for a 100-class problem can collapse to a couple of points. The fix is two pipelines: stochastic for train, deterministic for val.

```python
# RIGHT: deterministic eval transform, no randomness
train_tf = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
val_tf   = T.Compose([T.Resize(256), T.CenterCrop(224),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
```

The mirror-image bug is **augmentation not applied at all** — you wired up an `albumentations` or `torchvision` pipeline but, because of a typo or a `Compose` that returns the input unchanged, no augmentation runs. The signature is the opposite: train and val tracks too close together, overfitting earlier than the augmentation budget should allow. The diagnostic for both is the same: pull a batch from *each* loader and confirm that the train batch differs run-to-run for the same index and the val batch does not.

### 3.2 Normalization statistics mismatch — the eval-only killer

Normalization subtracts a mean and divides by a standard deviation so inputs land in a well-conditioned range. The cardinal rule: **the statistics are part of the model.** You compute mean and std *once*, on the training set, and you apply those exact numbers everywhere — train, val, test, and production. Two failure modes break this:

1. **Stats computed on a subset.** You estimate mean/std on the first 1,000 images for speed; if those happen to be class-sorted (all one bright class), the stats are biased, and every input is normalized against the wrong center.
2. **Stats recomputed at eval.** The truly nasty one: someone normalizes the validation set with the *validation set's own* mean/std instead of the training stats. Train inputs are centered at `0` with unit variance; val inputs are centered at `0` with unit variance *too*, but against a different reference, so the model — which learned to expect train-centered inputs — sees a shifted distribution and predicts garbage.

There is a subtlety in the first failure mode worth a sentence of statistics. When you estimate the mean from a subset of $n$ samples, the standard error of that estimate is $\sigma/\sqrt{n}$. Estimate the channel mean from $n = 1000$ images and you are fine — the error is tiny relative to the std. But if those 1,000 images are *class-sorted* and all come from one bright category, they are not a random sample, so the estimate is **biased**, not merely noisy, and no amount of $n$ fixes a biased sample. The correct estimate uses a random sample (or, cheaply, the whole set in a streaming pass). This is the same i.i.d. assumption from Section 2 reappearing: a statistic computed on a non-random subset is wrong in a way that more data does not cure.

The math of the second failure mode is short and worth doing. Suppose training pixels have mean $\mu_{\text{tr}}$ and std $\sigma_{\text{tr}}$, and you correctly train on $z = (x - \mu_{\text{tr}})/\sigma_{\text{tr}}$. The first layer learned weights $W$ that work on inputs with that center and scale. If at eval you instead feed $z' = (x - \mu_{\text{val}})/\sigma_{\text{val}}$, the layer computes $Wz'$, but the input it expects is $Wz$. The discrepancy is

$$
W(z' - z) = W\left[\frac{x - \mu_{\text{val}}}{\sigma_{\text{val}}} - \frac{x - \mu_{\text{tr}}}{\sigma_{\text{tr}}}\right],
$$

a systematic shift that grows with $|\mu_{\text{val}} - \mu_{\text{tr}}|$ and with the ratio $\sigma_{\text{tr}}/\sigma_{\text{val}}$. Even a few-percent difference in the stats is enough to move a confident classifier across decision boundaries, because the final logits are a sum over many channels and the shifts add coherently. Figure 5 shows the recovery: freeze the stats at fit time, apply train stats at eval, and accuracy climbs from chance back to honest.

![Before-and-after diagram showing a normalization statistics mismatch where training uses mean and std from the train set but evaluation recomputes them on a validation subset, pushing eval inputs out of range to fifty-two percent accuracy, versus the fixed version where statistics are frozen at fit time and reused at eval, returning inputs to range and accuracy to ninety-one percent](/imgs/blogs/the-input-pipeline-is-lying-to-you-5.png)

### 3.3 The diagnostic: assert the batch statistics match what you expect

The check is one line and you should make it an assertion, not a print, so it fails loudly in CI:

```python
import torch

def assert_normalized(x, expect_mean=0.0, expect_std=1.0, tol=0.4):
    m, s = x.float().mean().item(), x.float().std().item()
    print(f"batch mean={m:+.3f} std={s:.3f}")
    assert abs(m - expect_mean) < tol, f"mean {m:.3f} off; normalization wrong?"
    assert abs(s - expect_std)  < tol, f"std {s:.3f} off; normalization wrong?"

x, y = next(iter(train_loader))
assert_normalized(x)
x, y = next(iter(val_loader))
assert_normalized(x)   # MUST pass with the SAME stats as train
```

If a correctly-normalized batch reports `mean=+0.001 std=0.998`, you are good. If the train batch reports `mean=-0.01 std=1.00` but the val batch reports `mean=-0.31 std=1.27`, you have a stats mismatch — the two loaders are not normalizing identically. The range check generalizes across modalities: after ImageNet normalization, pixels live roughly in `[-2.6, 2.6]`, not `[0, 255]` and not `[0, 1]`; if you see `[0, 255]`, normalization did not run at all (you are feeding raw `uint8` cast to float, which makes the first-layer activations explode and the loss spike). This is the same family of bug as a [BatchNorm running-stats mismatch at eval](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training), and the cure is the same discipline: stats are frozen artifacts, not things you recompute.

## 4. Bug four: collate_fn and padding bugs that mix examples or pad with meaning

When examples have variable length — sentences of different token counts, audio clips of different durations, point clouds of different sizes — the default `collate_fn` cannot stack them into a rectangular tensor, so you write your own. This is where two dangerous bugs live: **padding with a meaningful value**, and **mixing examples** so positions from one example land in another's row.

### 4.1 Padding with a value that means something

To stack sequences of length 7 and 3 into a `[2, 7]` tensor, you pad the short one with four extra positions. The bug is choosing a pad value that the model interprets as real data. In NLP, token id `0` is often a real, frequent token (in many vocabularies it is a common subword, not the pad token); if you pad with `0`, the model attends to those positions and computes loss on them as if they were genuine. The pad must be (a) excluded from attention via the attention mask, and (b) excluded from the loss via `ignore_index` / a `-100` label. Miss either and padding leaks into the gradient.

```python
# WRONG: pad with 0 (a real token), no mask, loss counts pad positions
def bad_collate(batch):
    seqs = [b["input_ids"] for b in batch]
    maxlen = max(len(s) for s in seqs)
    padded = torch.zeros(len(seqs), maxlen, dtype=torch.long)   # 0 == real token id
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
    labels = padded.clone()        # pad positions get a real label -> counted in loss
    return {"input_ids": padded, "labels": labels}

# RIGHT: explicit pad id, attention mask, -100 on pad labels
PAD_ID = tokenizer.pad_token_id
def good_collate(batch):
    seqs = [b["input_ids"] for b in batch]
    maxlen = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), maxlen), PAD_ID, dtype=torch.long)
    attn = torch.zeros(len(seqs), maxlen, dtype=torch.long)
    labels = torch.full((len(seqs), maxlen), -100, dtype=torch.long)  # ignore pad in loss
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = s
        attn[i, :len(s)] = 1
        labels[i, :len(s)] = s
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
```

The `-100` is not a magic number you should memorize blindly: it is the default `ignore_index` of `torch.nn.CrossEntropyLoss` and Hugging Face's loss code, so any label position set to `-100` contributes exactly zero to the loss and its gradient. Figure 7 shows the leak path: pad with `0` and the positions flow into both attention and loss; pad with a masked pad id and `-100` labels and they are inert.

![Graph showing two sequences of length seven and three being collated, with a bad path that pads using token id zero which equals a valid id so padding leaks into both attention and loss, and a good path that pads with a real pad token and sets labels to minus one hundred so padding is ignored and the loss stays clean](/imgs/blogs/the-input-pipeline-is-lying-to-you-7.png)

### 4.2 Mixing examples — the row-aliasing bug

The subtler collate bug is an indexing error that writes example `i`'s data into row `j`. It happens when you build the batch tensor with a flat index and an off-by-one in the stride, or when you `torch.cat` instead of `torch.stack` and then reshape with the wrong dimension order. The symptom is brutal and confusing: the model trains *somewhat* (some rows are correct), but accuracy plateaus well below what the data supports, because a fraction of every batch has inputs from one example paired with the label of another. The confirming test is to **collate a batch of known sentinel examples** — make example `k` a tensor full of the value `k` — and assert that row `k` of the collated batch is all `k`:

```python
# Sentinel test: example k is a tensor full of value k
sentinels = [{"x": torch.full((3,), float(k)), "y": k} for k in range(8)]
out = good_collate_for_fixed_len(sentinels)
for k in range(8):
    assert (out["x"][k] == k).all(), f"row {k} got mixed: {out['x'][k]}"
    assert out["y"][k] == k,        f"label {k} desynced"
print("collate preserves row identity ✓")
```

If row `k` is not all `k`, your collate is aliasing examples. This sentinel trick — feed structured fake data whose correct output you can predict exactly — is a general weapon for any data-plumbing code, and it costs three lines.

There is a particularly nasty version of row-aliasing that hides in a one-character difference: `torch.cat` versus `torch.stack`. `stack` adds a new batch dimension and is what you almost always want; `cat` concatenates along an existing axis. If your examples are each shape `[L, D]` and you `torch.cat` them, you get a `[N*L, D]` tensor that you then `.view(N, L, D)` to "fix" — and that view is correct *only* if every example had exactly length `L`. The instant lengths vary and you have padded them to a common `L`, the arithmetic still works but a subtle reshape error can interleave positions from adjacent examples. The shapes are right, the loss is finite, and a third of every batch is scrambled. The reason this is worth a paragraph is that it defeats the cheapest checks — shapes pass, dtypes pass, the value range is normal — and only the sentinel test or a careful look at a decoded example catches it. When you write a custom collate, **prefer `torch.stack` over a `cat`-then-`view`**, and if you must reshape, sentinel-test it.

### 4.3 The padding-fraction tax: why over-padding is also a (silent) bug

Even a *correct* padding implementation has a quiet cost worth instrumenting: if you batch by random order, a batch can contain one very long sequence and many short ones, so you pad everything up to the longest. With sequences whose lengths range from 8 to 512, a random batch padded to 512 can be 70–80% padding — meaning the GPU spends most of its FLOPs computing attention over pad tokens you will mask out anyway. This is not a correctness bug (the mask makes the answer right) but a *throughput* bug that masquerades as a slow model. The instrument is the **padding fraction** per batch:

```python
def padding_fraction(attention_mask):
    real = attention_mask.sum().item()
    total = attention_mask.numel()
    frac = 1.0 - real / total
    return frac

# Typical readings:
#   random batching, var-len 8..512  -> padding fraction ~0.65 (wasteful)
#   length-bucketed sampler          -> padding fraction ~0.08 (efficient)
```

A padding fraction above ~0.4 says you are paying for far more compute than your real tokens justify, and the fix is a length-grouped sampler that batches similar-length sequences together. This is the loader's contribution to the GPU-idle story covered in the systems track; the point here is that the pipeline controls throughput as much as it controls correctness, and the padding fraction is the one-line instrument that surfaces it.

## 5. Bug five and seven: off-by-one windowing and label/index misalignment

These two bugs share a root cause — an index computed in one place but consumed in another with a different convention — and they are the hardest to see because the data looks plausible.

### 5.1 Off-by-one in windowing and sequence chunking

When you chunk a long sequence (a time series, a long document, an audio stream) into fixed windows with a stride, every boundary is an opportunity for an off-by-one. For language modeling you want the target to be the input shifted by one: input `tokens[:-1]`, target `tokens[1:]`. Get the slice wrong and you either train the model to predict the *current* token (trivial copy, loss collapses to near-zero and the model is useless at generation) or you drop the last token of every window (a small, silent data loss that caps accuracy).

```python
# Language-modeling window: target is input shifted by one
def make_lm_window(tokens, block_size):
    # tokens: 1D LongTensor of length >= block_size + 1
    x = tokens[:block_size]          # input
    y = tokens[1:block_size + 1]     # target = next token
    assert x.shape == y.shape
    assert torch.equal(x[1:], y[:-1]), "shift-by-one broken: y must be x shifted left"
    return x, y
```

That last assertion is the whole defense: `y` must equal `x` shifted left by one, so `x[1:]` and `y[:-1]` are identical. If they are not, your windowing is off. For overlapping windows with stride `s < block_size`, also assert that consecutive windows overlap by exactly `block_size - s` positions; a stride that accidentally equals `block_size + 1` drops one token per window, and over a million windows you have silently discarded a million tokens.

The most diagnostic symptom of a shift-by-one mistake is a **loss that is far lower than it has any right to be.** If you accidentally set the target equal to the input (no shift), the task becomes "copy the current token," which a model with any capacity solves almost instantly — training loss plummets toward zero in a few hundred steps and stays there. That looks like spectacular convergence; it is actually a trivial-task collapse. The tell is twofold: the loss is implausibly low for the task (a language model that reads `0.02` cross-entropy after an hour is not a genius, it is copying), and *generation is garbage* because the model never learned to predict the next token, only to echo the current one. Whenever a loss looks too good, suspect the pipeline before you celebrate: a too-low loss is as much a pipeline smell as a too-high one. The same logic catches windowing strides that are off — a periodic loss whose period matches your window count, or a token budget that does not add up when you multiply windows by stride.

#### Worked example: the suspiciously easy language-modeling run

You launch a from-scratch language-model pretrain and within 400 steps the loss is `0.08`, far below the `~3.0` you would expect early in training for a reasonable vocabulary. The instinct is delight; the discipline is suspicion. You run a single batch through the windowing code and check the shift assertion — it fails: `y[:-1]` does not equal `x[1:]`. You print one window: `x = [the, cat, sat, on]`, `y = [the, cat, sat, on]`. The target was never shifted; the model is being trained to predict the token it already sees. Of course the loss is near zero — the task is a lookup. You fix the slice to `y = tokens[1:block_size+1]`, rerun, and the loss now starts at `~6.2` (about $\ln$ of the vocabulary, the uniform-prediction baseline) and *descends* as real learning happens. The first run's beautiful curve was a pipeline bug wearing a convergence costume. This is the mirror image of the saw-tooth: there, a *bad-looking* curve was a pipeline bug; here, a *great-looking* curve was too. The lesson is symmetric — read the absolute loss value against the task's information-theoretic floor, not just its shape.

### 5.2 Label/index misalignment between inputs and targets

The most dangerous version is when inputs and targets are stored separately and sorted, filtered, or shuffled out of lockstep — the two-shuffle bug from Section 2.3, but it also arrives through `groupby`, `merge`, or a `dropna` that drops rows from `X` but not `y`. The result: input `i` is paired with target `j`. The model trains on noise. Loss falls (the model fits the best constant-ish map to noise), accuracy never beats chance.

The single best defense is structural: **never let inputs and labels live in separate, separately-mutated containers.** Keep them paired in one row, one `Dataset` item, one DataFrame. When you must align two sources, join on an explicit key and assert the join did not change the row count:

```python
n_before = len(df)
df = df.dropna(subset=["feature_a"])          # drops some rows
# y was a separate array -> now misaligned with df!  Re-derive y FROM df instead:
y = df["target"].values                        # y now tracks df's surviving rows
assert len(y) == len(df), "labels desynced from features after filtering"
```

The catch-all confirming test is the one humans are best at and machines are worst at: **show a handful of decoded examples next to their labels and read them.** If the picture is a `7` and the label says `2`, or the sentence is about sports and the label is `finance`, the alignment is broken and no statistic will tell you faster than your own eyes. We make that a routine in Section 7.

## 6. Bug six, eight, nine: drop_last, the worker-RNG duplication, and the caching no-op

Three more, and the second of these is the most famous DataLoader bug in the field.

### 6.1 drop_last skewing small datasets

`DataLoader(drop_last=True)` discards the final, smaller batch of each epoch so every batch is full size. On large data this is harmless. On small data it is not: with 1,050 examples and batch size 128, you get 8 full batches (1,024 examples) and drop the last 26 every epoch — about 2.5% of your data, the *same* 26 examples each epoch if shuffle is off, *different* 26 if shuffle is on but still always 26 unseen per epoch. For a tiny validation set, `drop_last=True` can silently change which examples you score, making your val number jitter. The fix is to set `drop_last=False` for evaluation always, and for training only use `drop_last=True` when you have a reason (BatchNorm hating size-1 final batches is the usual one) and your dataset is large enough that 2% does not matter.

The evaluation case deserves emphasis because it produces a specific, confusing symptom: a validation metric that **changes when you change the batch size**, even though batch size should be a pure throughput knob at eval. If `drop_last=True` is set on the val loader, a batch size of 128 drops a different number of final examples than a batch size of 256, so the two configurations score *different subsets* of the validation set and report different numbers. An engineer who sees "val acc went from 88.2% to 88.7% when I bumped the eval batch size" will go hunting for a numerics explanation and find nothing, because the real cause is that the two runs evaluated on slightly different data. The check is trivial: assert that your eval loader's total scored count equals the dataset length.

```python
n_scored = sum(len(y) for _, y in val_loader)
assert n_scored == len(val_dataset), (
    f"eval scored {n_scored} of {len(val_dataset)} — drop_last is on, "
    f"or the sampler is subsetting the val set")
```

If `n_scored < len(val_dataset)`, you are not evaluating on your whole validation set, and your metric is computed on a moving target.

### 6.2 The worker-RNG duplication bug — the famous one

This is the bug that shipped in a stunning number of public codebases, and it is worth understanding precisely because the fix is one function. When you set `num_workers > 0`, PyTorch forks worker processes to load data in parallel. Each forked worker inherits a *copy of the parent's memory*, including the parent's NumPy random state. If your augmentation uses `np.random` — `np.random.randint` for a crop offset, `np.random.rand` for a flip coin — then **every worker starts from the identical NumPy seed** and produces the *identical* stream of "random" numbers. Worker 0 and worker 3 both crop at offset `(17, 4)`. Within a single batch assembled from multiple workers, you get the same augmentation repeated, so your effective augmentation diversity collapses.

The science is exact. With `num_workers = W` workers each producing `b/W` items per batch, and all sharing one NumPy seed `S`, the sequence of augmentation random draws is identical across workers. If your augmentation is a deterministic function `aug(x, r)` of the input and a random draw `r`, then for the same draw `r` you get the same transform. Across a batch you therefore see at most `b/W` *distinct* augmentation parameters instead of `b`, because the `W` workers replay the same `r` values in lockstep. With `W = 4` and `b = 256`, you get 64 distinct augmentations, each applied to 4 different images — a 4× reduction in augmentation diversity that no error message will ever mention. Figure 4 shows the fork: workers inherit one seed and emit one stream; `worker_init_fn` reseeds per worker and restores diversity.

![Graph showing a main process with a NumPy seed set at import time forking into three workers that each inherit the same seed S and therefore produce identical augmentations, contrasted with a worker_init_fn that reseeds each worker to S plus its worker id so the batch contains distinct augmentations rather than one repeated three times](/imgs/blogs/the-input-pipeline-is-lying-to-you-4.png)

Crucially, PyTorch's *own* RNG (`torch.randint`, the `Generator` passed to the `DataLoader`) is seeded per-worker automatically — the base seed plus the worker id — so torchvision transforms that use `torch.rand` are safe by default. The bug bites specifically when your augmentation reaches for **NumPy** or Python's `random`, which PyTorch does not reseed for you. The fix is a `worker_init_fn` that reseeds NumPy and `random` from the per-worker torch seed:

```python
import numpy as np
import random
import torch

def seed_worker(worker_id):
    # torch already gave this worker a unique base seed; derive NumPy/python from it
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    num_workers=4,
    shuffle=True,
    worker_init_fn=seed_worker,   # reseeds NumPy + random per worker
    generator=g,                  # makes the shuffle itself reproducible
)
```

`torch.initial_seed()` inside a worker returns that worker's *unique* base seed, so reseeding NumPy from it gives every worker a different stream. The `generator=g` argument makes the shuffle reproducible across runs (the determinism story from the [reproducibility post](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training)), which you want so that *this* fix is itself testable.

#### Worked example: catching the duplicated augmentation by hashing crops

You suspect the worker-RNG bug. Confirm it by **hashing the augmented contents of one batch and counting duplicates**. If augmentation is working, every image in a batch is a distinct crop, so all hashes are unique. If the bug is present, you see `b/W` distinct hashes, each repeated `W` times.

```python
import hashlib

def count_distinct_augs(loader, n=1):
    x, y = next(iter(loader))                 # x: [B, C, H, W]
    hashes = [hashlib.sha1(img.cpu().numpy().tobytes()).hexdigest()[:10]
              for img in x]
    distinct = len(set(hashes))
    print(f"batch size {len(hashes)}, distinct augmented images: {distinct}")
    return distinct

# Buggy: num_workers=4, no worker_init_fn, np.random augmentation
# -> "batch size 256, distinct augmented images: 64"   (256/4)
# Fixed: with seed_worker
# -> "batch size 256, distinct augmented images: 256"
```

The buggy run reports `64` distinct images in a batch of `256`; the fixed run reports `256`. That single integer is the whole proof. Re-run training after the fix and the model that plateaued at, say, `78%` because it was effectively training on a quarter of the augmentation diversity now reaches `82%` — a 4-point recovery from one function, confirmed by a hash count, not a hunch.

### 6.3 The caching no-op: a transform that returns the same item

The last one is a caching bug that turns your `Dataset` into a constant. You add an LRU cache or a memoization decorator to `__getitem__` to speed up disk reads, but you cache on a key that does not include the augmentation randomness — or worse, you cache the *augmented* tensor. Now every epoch returns the byte-identical augmented image for index `i`, so your "random" augmentation is frozen after the first epoch, and your dataset is effectively `N` fixed images forever. The model overfits hard. The catch is the two-epoch hash again: pull index `i` in epoch 1 and epoch 2 and confirm the augmented tensors differ.

```python
img_e1 = dataset[0][0]          # epoch-1 view of index 0
img_e2 = dataset[0][0]          # should differ if augmentation is stochastic
same = torch.equal(img_e1, img_e2)
print("index-0 augmentation identical across calls:", same)
assert not same, "augmentation frozen — cache is returning the same augmented item"
```

If the two calls return identical tensors for an index you expect to be randomly augmented, your cache is memoizing the augmented output. Cache the *raw decoded* image (the expensive disk+decode step) and apply augmentation *after* the cache, never inside it.

## 7. The same bugs in tabular and speech pipelines

Everything so far has used vision and NLP examples because they are the most visceral, but the identical bug classes appear — with different costumes — in tabular and speech loaders. Naming the modality-specific disguise is what lets you transfer the discipline across the whole stack.

### 7.1 Tabular: the scaler-fit-before-split leak and the silent column shift

Tabular pipelines have their own version of the normalization-stats bug, and it is the single most common Kaggle-grade mistake: **fitting a scaler or imputer on all the data before splitting.** When you call `StandardScaler().fit_transform(X)` on the full dataset and *then* split into train and test, the scaler's mean and variance were computed using the test rows. Information about the test distribution has leaked into the training features. The model's cross-validation score is optimistic, and the gap to production is exactly the leak you baked in. The fix is the same structural rule as Section 3.2 — stats are frozen artifacts fit on train only — and the tool that enforces it is `sklearn.pipeline.Pipeline`, which fits every step inside each CV fold:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# RIGHT: scaler + imputer are fit INSIDE each fold, never on the full data
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])
scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
print(f"CV AUC = {scores.mean():.3f} +/- {scores.std():.3f}")
# Leaky version (scaler.fit on all X then CV) often reads ~0.02-0.05 AUC higher
# than the honest pipeline — that gap IS the leak.
```

The other tabular pipeline bug is the **silent column shift**: you build your feature matrix with one column order at train time and a different order at serve time — a dictionary that does not preserve insertion order across Python versions, a `pd.get_dummies` that produces different columns when a category is absent from the inference batch, a feature added in the middle of the list. The model multiplies the wrong weight by the wrong feature. Nothing crashes because the shapes still match. The `inspect_batch` analog for tabular is to print the **column names alongside the first row of values** and assert the order matches a saved schema:

```python
import json

EXPECTED_COLS = json.load(open("feature_schema.json"))   # saved at train time
def assert_columns(df):
    cols = list(df.columns)
    assert cols == EXPECTED_COLS, (
        f"column order/membership changed!\n"
        f"  expected[:5]={EXPECTED_COLS[:5]}\n  got[:5]={cols[:5]}")
    print(f"columns OK ({len(cols)} features, order matches schema)")
```

And the tabular version of "the target is hiding in the features" — a proxy or ID column that correlates almost perfectly with the label — is caught by the same look-at-your-data habit: check that no feature column has near-perfect single-feature AUC, and that the literal target column is not accidentally in `X`. This is the data-leakage story in full, covered in [data leakage, the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer); the pipeline-specific piece is just the scaler-fit-before-split mechanic above.

### 7.2 Speech: sample-rate and feature-extractor mismatch

Speech loaders carry the most physical version of the normalization-mismatch bug: a **sample-rate mismatch.** An ASR model trained on 16 kHz audio expects mel-spectrograms computed from 16,000 samples per second. If your loader reads 44.1 kHz files and does not resample, every spectrogram is stretched in time and shifted in frequency relative to what the model learned — and `torchaudio.load` will happily hand you 44.1 kHz audio with no warning. The model's "transcription" is garbage, and the loss looks plausible because CTC or cross-entropy still computes a finite number. The check is to assert the sample rate against the feature extractor's expectation before the spectrogram is ever computed:

```python
import torchaudio

EXPECT_SR = 16000
wav, sr = torchaudio.load("clip.wav")
if sr != EXPECT_SR:
    wav = torchaudio.functional.resample(wav, sr, EXPECT_SR)
assert wav.shape[0] == 1, f"expected mono, got {wav.shape[0]} channels"
print(f"audio: sr={EXPECT_SR}Hz, mono, {wav.shape[1]/EXPECT_SR:.2f}s, "
      f"peak={wav.abs().max():.3f}")
# peak near 1.0 if float-normalized; near 32768 if raw int16 not scaled
```

The mel-parameter version is subtler: `n_fft`, `hop_length`, and `n_mels` in your feature extractor must match the values the model was pretrained with. A Whisper or wav2vec2 checkpoint expects a specific mel configuration; compute the spectrogram with a different `hop_length` and the time axis no longer aligns with the model's positional expectations. The fix is to use the model's *own* feature extractor (`WhisperFeatureExtractor`, `Wav2Vec2FeatureExtractor`) rather than rolling your own `torchaudio.transforms.MelSpectrogram`, so the parameters cannot drift. There is also a CTC-specific length trap — if the encoder downsamples the audio to fewer frames than the target transcription has tokens, the CTC loss returns `inf` because no valid alignment exists — but that one is the heart of the speech track's CTC-and-alignment post, and we only flag it here as the place a speech loader, which controls the frame count via chunking, can manufacture the condition.

## 8. The print-the-batch discipline and a reusable inspect_batch()

Every bug above is caught by the same five-minute ritual, performed *before* you start a long run: **pull one batch and refuse to proceed until you understand every number in it.** Shapes, dtypes, value ranges, the label histogram, and — for the bugs the eye catches fastest — a few decoded examples shown next to their labels. Here is the reusable routine. It is deliberately modality-agnostic in its core and has small hooks for the things that differ.

```python
import torch

def inspect_batch(batch, n_show=4, decode=None, class_names=None):
    """Pull-one-batch inspection. Run BEFORE training, every time."""
    x, y = batch if isinstance(batch, (tuple, list)) else (batch["input_ids"], batch["labels"])

    # 1. Shapes & dtypes — catches shape/collate bugs
    print(f"x: shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
    print(f"y: shape={tuple(y.shape)} dtype={y.dtype}")

    # 2. Value range & stats — catches normalization bugs
    xf = x.float()
    print(f"x range=[{xf.min():.3f}, {xf.max():.3f}] mean={xf.mean():+.3f} std={xf.std():.3f}")
    if torch.isnan(xf).any() or torch.isinf(xf).any():
        print("  !! x contains NaN/Inf — bad decode or normalization")

    # 3. Label histogram — catches shuffle-off and class imbalance
    if y.dtype in (torch.long, torch.int64) and y.dim() == 1:
        binc = torch.bincount(y[y >= 0])      # ignore -100 masked labels
        print(f"label histogram: {binc.tolist()}  (n_masked={(y < 0).sum().item()})")

    # 4. Content hash — catches stuck/cached/duplicated batches
    import hashlib
    print("content hash:", hashlib.sha1(x.cpu().numpy().tobytes()).hexdigest()[:12])

    # 5. Show a few decoded examples WITH labels — catches misalignment
    if decode is not None:
        for i in range(min(n_show, len(x))):
            label = class_names[y[i]] if class_names is not None else y[i].item()
            print(f"  example {i}: label={label}  ->  {decode(x[i])}")

train_batch = next(iter(train_loader))
inspect_batch(train_batch, decode=lambda t: f"img[min={t.min():.2f},max={t.max():.2f}]")
```

The discipline has five questions, and Figure 3 lays out which bug each answer reveals: shapes and dtypes catch collate and broadcasting errors; the value range catches normalization; the label histogram catches shuffle-off and imbalance; the content hash catches stuck, cached, or duplicated batches; and looking at decoded examples catches the alignment bugs no statistic will. Run it on the *train* loader and the *val* loader and compare — the val batch should have the same range and stats but should *not* change run-to-run for a fixed index.

![Matrix mapping four DataLoader bugs to their symptom, a one-line confirming check, and the fix, covering shuffle off with saw-tooth loss confirmed by a batch-order hash, normalization mismatch with eval-only garbage confirmed by checking the batch mean and std, a collate padding bug with pad tokens in the loss confirmed by printing the mask, and worker RNG duplication confirmed by hashing the crops](/imgs/blogs/the-input-pipeline-is-lying-to-you-3.png)

#### Worked example: inspect_batch catches three bugs in one readout

You run `inspect_batch` on a finetuning run that is plateauing and read this:

```bash
x: shape=(32, 512) dtype=torch.int64 device=cuda:0
y: shape=(32, 512) dtype=torch.int64
x range=[0.000, 50256.000] mean=+8421.300 std=14022.100
label histogram: [ ... ]  (n_masked=0)
content hash: 4f1a8c2e9b07
```

Three red flags in five lines. First, `n_masked=0` for a causal LM means **no label is `-100`** — you are training on the prompt tokens, not just the completion, wasting most of the gradient on text you do not want the model to learn to generate. Second, the label histogram for a 50k-vocab LM should be a wide spread; if it is dominated by a single id, padding is leaking. Third, run it twice: if the `content hash` is identical across two `next(iter(...))` calls when you expected shuffling, the loader is not advancing. One readout, three localized bugs, before a single training step. That is the entire value proposition of the discipline: the batch tells you the truth that the loss curve will spend hours hiding.

## 9. Before and after: the evidence that fixing the pipeline recovers accuracy

The whole point of this series is honest before→after evidence, so here is the consolidated table for the running ResNet example and a few cross-modality cousins. Each row is a real-shaped result: the symptom value, the confirming check, the fix, and the instrument reading after.

| Bug | Symptom (before) | Confirming check | Fix | After |
| --- | --- | --- | --- | --- |
| Shuffle off | val acc `10.4%`, saw-tooth loss | two-epoch order hash equal | `shuffle=True` | val acc `91%`, smooth loss |
| Augment on val | train loss `0.31`, val acc `1%` | val batch changes per call | deterministic val transform | val acc `74%` |
| Norm stats mismatch | val acc `52%` | val batch `mean=-0.31` | reuse train mean/std | val acc `91%` |
| Pad with token 0 | NaN on long batches, loss off | print mask vs attention_mask | pad id + `-100` labels | clean loss, no NaN |
| Worker-RNG dup | plateau at `78%` | 64 distinct of 256 in batch | `worker_init_fn=seed_worker` | `82%`, 256 distinct |
| Label misalign | acc at chance, loss falls | show 4 images with labels | pair x,y in one Dataset | acc `89%` |

How would you *confirm* each "after" honestly rather than just believing it? Re-run the exact confirming check post-fix and watch the instrument flip: the order hash now differs across epochs; the distinct-augmentation count now equals the batch size; the val batch `mean` now matches the train batch `mean` within tolerance; the four shown images now match their labels. The fix is only real when the *same diagnostic that caught the bug* now reads healthy. Believing a fix without re-running its detector is how you ship the next silent regression.

#### Worked example: the \$40 overnight run, bisected in eight minutes

Return to the opening disaster: ResNet-50, train loss `0.31`, val acc `1.0%`, nine hours burned at roughly `\$4.40` per GPU-hour for a `\$40` run that taught the model nothing. Here is the eight-minute bisection. Minute 1: `inspect_batch` on the train loader — shapes fine, range `[-2.6, 2.6]`, label histogram balanced. Train pipeline looks clean. Minute 3: `inspect_batch` on the val loader — same range, but call it twice and the `content hash` *changes*. A validation batch should be deterministic; it is not. Minute 5: print the val transform — it is the shared `train_tf` with `RandomResizedCrop`. Found it: augmentation is running on validation, so the model is scored on random patches. Minute 6: build a deterministic `val_tf`, re-run `inspect_batch` on val — hash now stable across calls. Minute 8: re-run validation on the existing overnight checkpoint with the corrected transform — val acc jumps from `1.0%` to `73%`. The nine hours of training were fine all along; the *measurement* was broken. The fix touched zero lines of model code. This is the whole thesis: when the pipeline lies, the cheapest possible thing — looking at the batch — saves you the most expensive thing, a re-run.

## 10. Case studies: real signatures of pipeline bugs in the wild

These are well-known, documented patterns, not invented ones. Where I give a number I name the source; where I don't have an exact figure I say so.

**The NumPy-seed-in-workers bug, found across the open-source ecosystem.** A 2021 audit by Tanel Pärnamaa popularized just how widespread the worker-RNG duplication was: a large fraction of public PyTorch training repositories that used `np.random` inside augmentation with `num_workers > 0` and no `worker_init_fn` were silently duplicating augmentations across workers. PyTorch's own documentation now carries an explicit "Randomness in multi-process data loading" warning and recommends exactly the `worker_init_fn` pattern above. The reason it went unnoticed for so long is the reason this whole post exists: the bug *improves* nothing and *crashes* nothing; it just quietly throttles augmentation diversity, costing a point or two of accuracy that no one attributes to the loader.

**Left-padding breaks decoder-only generation.** A canonical Hugging Face gotcha: decoder-only models (GPT-style) must be **left-padded** for batched generation, because right-padding puts pad tokens *after* the prompt and the model generates from the pad positions. With right-padding and a correct attention mask the loss during *training* can still look fine, but batched *generation* produces garbage for all but the longest sequence in the batch. The fix is `tokenizer.padding_side = "left"` for generation. This is the inference-time mirror of the collate-padding bug and is covered deeper in the LLM track's padding-and-masking post.

**ImageNet/MNIST test-set label errors.** Northcutt, Athalye, and Mueller's 2021 "Pervasive Label Errors in Test Sets" used confident learning (the `cleanlab` method) to find that widely-used ML test sets contain on the order of a few percent label errors — they estimated roughly 3.4% errors on average across ten benchmarks, including thousands of mislabeled images in the ImageNet validation set. The relevance here: when *the labels themselves* are wrong, a perfectly correct pipeline still feeds the model wrong targets, and the ceiling on your accuracy is set by the label noise, not your model. The detector is the same family of "look at your data" tooling — covered in the label-noise post — and the lesson is that "look at a few examples with their labels" sometimes reveals the dataset is wrong, not your code.

**Normalization-scale mismatch in transfer learning.** A recurring practitioner bug when finetuning an ImageNet-pretrained backbone: feeding images in `[0, 255]` or `[0, 1]` when the backbone was trained on ImageNet-normalized inputs centered near zero with unit variance. The model's first-layer statistics are wildly off, activations saturate or explode, and finetuning either diverges or learns very slowly. The catch is the range check from Section 3.3 — if your "normalized" batch maxes out at `255`, normalization did not run.

**The cv2-versus-PIL channel-order disagreement.** OpenCV (`cv2.imread`) returns images in **BGR** channel order; PIL (`Image.open`) and most pretrained models expect **RGB**. Train with one library and serve with the other — or mix them in the same pipeline — and every red object is treated as blue and vice versa. The model can still learn *something* (the spatial structure survives), so train accuracy is decent, but it never reaches the pretrained backbone's potential because the color statistics it learned no longer apply. The bug is invisible in shapes and ranges (a BGR tensor and an RGB tensor are byte-for-byte indistinguishable as numbers) and only shows when you *display* a few images and notice the colors look wrong, or when you assert your decode library matches between train and serve. This is precisely why "show a few decoded examples" is in the discipline: it is the only check that catches a channel swap, because no statistic can.

#### Worked example: the cv2/PIL swap caught by one displayed image

You finetune a pretrained classifier that tops out at `81%` where the backbone's reported transfer accuracy on similar data is `~88%`. Shapes are right, the normalized range is `[-2.1, 2.4]` — clean. You run the display branch of `inspect_batch` and render four images: they look *tinted* — skies are orange, skin tones are bluish. That tint is the BGR-as-RGB signature. You trace the loader: it uses `cv2.imread`, which returns BGR, but your `Normalize` uses ImageNet's RGB-order mean and std and the backbone expects RGB. One line — `img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` after the read — and retraining lifts accuracy from `81%` to `87.6%`, recovering almost the entire gap. The number that caught it was not a number at all; it was your eyes on a rendered image, which is why the display step is non-negotiable.

## 11. When this is (and isn't) your bug

Pipeline bugs have a characteristic fingerprint, and it is worth stating decisively when a symptom points *elsewhere* so you do not waste a day re-reading your `collate_fn` when the problem is the learning rate.

It **is** a pipeline bug when: the optimizer is clearly working (train loss falls smoothly) but the model does not generalize; or train and eval diverge dramatically and you have ruled out plain overfitting; or the loss is periodic/saw-tooth with a period that matches your data order; or accuracy is pinned at chance while loss still drops; or the failure depends on `num_workers`, batch size, or sequence length (anything the loader controls). The decisive test is the one this whole post is built on: **run `inspect_batch`; if any number is wrong, the pipeline is your bug.** And the corollary from the overfit-one-batch discipline: if you can overfit a single hand-checked batch to near-zero loss but the full run won't learn, the model and optimizer are fine and the difference is *the rest of the data the loader serves* — a pipeline problem by elimination. Figure 6 is the decision tree: branch on the symptom shape, then run the one-line check for the suspect.

![Decision tree starting from whether the batch is bad, branching on periodic loss or poor generalization toward shuffle-off and worker RNG duplication suspects, and on eval-only garbage toward normalization stat mismatch and label misalignment, each leaf naming the one-line confirming check](/imgs/blogs/the-input-pipeline-is-lying-to-you-6.png)

It is **not** a pipeline bug when: the loss is smooth and then spikes to NaN — that is numerics (a learning rate or fp16 problem), not data, because a data bug is wrong from step 1, not step 4,000; or `inspect_batch` reads perfectly healthy on both loaders yet the model still won't learn — then look at the model code, the loss reduction, or the optimizer; or train and eval track each other closely and both are mediocre — that is underfitting (capacity or LR), not a leak. The single sharpest discriminator is *time-of-onset*: a pipeline bug corrupts the batch at step 1 and you can see it by looking, whereas a numerics bug develops over many steps. If `inspect_batch` is clean and the run still misbehaves, stop blaming the loader and move down the bisection to optimization or model code. The taxonomy post lays out that full branch order; this post owns the data leaf of it.

The cross-modality version of "is it my bug" lives in Figure 8: the same `inspect_batch` ritual, with the modality-specific check spelled out. In vision, confirm pixels are normalized (range near `[-2.6, 2.6]`) and channels are RGB not BGR. In NLP, confirm token ids are below the vocab size, the attention mask sums to the true length, and `-100` masks the prompt not the completion. In tabular, confirm there are no NaNs or infs, no zero-variance columns, and that the target is not hiding among the features (the classic leak). In speech, confirm the sample rate is what the feature extractor expects (16 kHz for most ASR models) and that text length does not exceed frame length (the CTC trap covered in the speech track). One routine, four modalities, the same discipline.

![Matrix mapping four modalities to the range check, the label check, and the characteristic pipeline bug each catches, with vision checking normalized pixel ranges and RGB order, NLP checking token ids under vocab and prompt masking, tabular checking for NaNs and target leakage, and speech checking sample rate and frame-versus-text length](/imgs/blogs/the-input-pipeline-is-lying-to-you-8.png)

## 12. Wiring the discipline into every run permanently

Catching these bugs once is good; catching them forever is the goal. Three cheap habits make pipeline bugs nearly impossible to ship silently.

First, **make `inspect_batch` a required first step**, not an optional debug aid. Call it at the top of every training script, on both loaders, and have it `assert` the invariants you care about (range, no NaN, label histogram non-degenerate, two-epoch order differs). A run that starts by failing an assertion is a gift; a run that starts by training on a broken batch is a forty-dollar lie.

Second, **write the sentinel and hash tests as unit tests.** The collate sentinel test from Section 4.2, the distinct-augmentation hash from Section 6.2, and the two-epoch order hash from Section 2.2 are all a few lines each and run in milliseconds on a tiny fake dataset. Put them in CI. They will catch the regression the day someone "optimizes" the loader and accidentally freezes the cache or drops the `worker_init_fn`.

Third, **keep stats and transforms as frozen artifacts.** Compute normalization mean/std once, save them to a file next to the checkpoint, and load them at train, eval, and serve. Define exactly two transform pipelines — stochastic-train and deterministic-eval — and never share a `Compose` between them. The discipline is structural: if the only way to get the wrong stats is to edit a saved artifact, no one will do it by accident.

The unifying idea across the whole post is one sentence: **the batch is the ground truth, and the loss curve is a rumor.** A loss curve is a heavily-summarized, many-step rollup that can look healthy while the underlying batches are corrupt. The batch is the actual input to the actual math. Spend the five minutes to look at it, and you convert a class of multi-day, multi-hundred-dollar silent failures into a five-line readout you check before you ever start the clock.

One last framing to carry into the rest of the series. Notice that every fix in this post was *structural*, not clever. We did not invent a smarter augmentation or a better loss; we made the wrong thing impossible to do by accident — paired x and y in one `Dataset`, froze stats to a saved file, defined exactly two transform pipelines, reseeded workers with one function, and asserted the invariants at the top of the run. That is the signature of mature pipeline engineering: you do not get better at *catching* these bugs so much as you arrange the code so they cannot occur silently. A loader that fails loudly on a bad batch is worth ten loaders that run fast and lie. When you read the [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), every confirming test in the data branch is one of the checks above; and when you sit down with the [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook), the very first item on the pre-flight checklist is `inspect_batch` on both loaders. The discipline is small. The runs it saves are not.

## Key takeaways

- **The pipeline lies silently — it rarely raises.** A wrong batch trains to low loss and reports a useless model. Look at the batch before you trust the loss curve.
- **Saw-tooth loss with a per-epoch period means shuffle is off (or labels are shuffled separately from inputs).** Confirm by hashing batch order across two epochs; they must differ. Fix: `shuffle=True`, one permutation for both x and y.
- **Augmentation is train-only; normalization stats are frozen artifacts.** Use two transform pipelines (stochastic train, deterministic eval) and reuse train mean/std everywhere. A val batch whose `mean`/`std` differs from train's is a stats mismatch.
- **Pad with a dedicated pad id, mask it in attention, and set pad labels to `-100`.** Padding with token `0` leaks meaningful positions into attention and loss. Confirm with the collate sentinel test.
- **The worker-RNG duplication bug throttles augmentation diversity by a factor of `num_workers`.** It bites when augmentation uses `np.random`/`random` (not torch). Fix with `worker_init_fn=seed_worker`; confirm by counting distinct augmented hashes per batch.
- **Off-by-one windowing and label/index misalignment fall to one habit: show a few decoded examples next to their labels and read them.** No statistic catches a scrambled label faster than your eyes.
- **Time-of-onset is the sharpest discriminator.** Wrong from step 1 → pipeline. Smooth then NaN at step 4,000 → numerics. If `inspect_batch` is clean, stop blaming the loader.
- **Re-run the detector to confirm the fix.** A fix is only real when the same check that caught the bug now reads healthy: order hash differs, distinct-aug count equals batch size, val `mean` matches train.

## Further reading

- PyTorch documentation, ["Randomness in multi-process data loading"](https://pytorch.org/docs/stable/notes/randomness.html) and the `torch.utils.data` `DataLoader` reference — the canonical source for `worker_init_fn`, `generator`, and per-worker seeding.
- C. Northcutt, A. Athalye, J. Mueller, "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks," NeurIPS 2021 Datasets and Benchmarks — confident learning and the test-set label-error estimates; the `cleanlab` library implements it.
- Hugging Face `transformers` documentation on padding and `DataCollatorForLanguageModeling`, and the generation docs on `padding_side` for decoder-only models — the source of the left-padding gotcha and `-100` loss masking.
- A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS 2019 — the autograd and DataLoader design that the worker-fork behavior follows from.
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for the full symptom→suspect→test→fix decision tree, [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) for the seeding and worker-RNG mechanics, and the capstone [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook).

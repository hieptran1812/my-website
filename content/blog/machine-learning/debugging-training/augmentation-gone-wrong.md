---
title: "Augmentation Gone Wrong: When Your Data Augmentation Destroys the Label"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Augmentation is supposed to be free regularization — until a horizontal flip turns a 6 into a 9, a too-strong policy injects label noise, or the validation set gets augmented and your metric starts lying. Here is how to catch all of it in one screen of code."
tags:
  [
    "debugging",
    "model-training",
    "data-augmentation",
    "computer-vision",
    "pytorch",
    "albumentations",
    "finetuning",
    "deep-learning",
    "regularization",
    "mixup",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/augmentation-gone-wrong-1.png"
---

A team I worked with spent a week convinced their handwritten-digit classifier had a model problem. They had a clean little ConvNet that hit 99.1% on MNIST in their first notebook. Then they "made it more robust" by bolting on a standard augmentation block copied from an ImageNet recipe — random horizontal flips, random rotations up to 30 degrees, a dash of color jitter — and retrained. Accuracy *dropped* to 96.4%. They added more capacity. It dropped further. They lowered the learning rate, added more epochs, swapped the optimizer. The number kept sliding, and the train loss refused to go as low as it used to. The instinct in the room was "the model is underpowered," and they were three commits into a bigger architecture when someone finally did the one thing nobody had done: they pulled fifty augmented training images up on screen *with their labels printed on top*.

There it was. A `6`, flipped horizontally, sitting under the label `6` — but it now looked exactly like a `9`. A `2` rotated 30 degrees looked like a `7`. The horizontal flip and the aggressive rotation were not regularizing the model; they were taking correctly-labeled images and turning them into images of a *different* class while keeping the original label. Every flipped `6`-labeled-as-`6`-that-looks-like-`9` is a mislabeled example. The augmentation pipeline was a label-noise generator, and the model was correctly learning a confused, contradictory mapping. The figure below is the whole bug in one picture: the same horizontal flip is harmless on a cat and catastrophic on a digit.

![A two-column comparison showing a horizontal flip preserving the label cat in the left column and destroying the label by turning a six into a nine in the right column](/imgs/blogs/augmentation-gone-wrong-1.png)

This post is about every way data augmentation quietly betrays you. Augmentation is one of the most reliable wins in deep learning — it is, in effect, free training data and a strong regularizer — but it has a sharp edge that the textbooks rarely mention: **an augmentation is only valid if it preserves the label**, and it is alarmingly easy to use a transform that does not. By the end of this post you will be able to take any run where you suspect augmentation — train loss that will not drop, validation that swings two points between runs, a detection model whose mAP is glued to zero — and localize the fault in minutes. You will have a single diagnostic habit (visualize augmented samples *with* labels) that catches the majority of these bugs before you spend a GPU-hour, plus runnable `torchvision` and `albumentations` code for the box-in-lockstep pattern and a correct mixup, and an honest before-and-after table for each fix.

This is, in the language of this series, a **data** bug that masquerades as an **optimization** or **model** bug. It sits squarely in the [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs): the symptom (loss floor, noisy val) points you at optimization, but the root cause is your input pipeline corrupting the supervision signal. We will bisect to it the disciplined way, and you will see why the [overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) — run once with augmentation *off* and once *on* — is the cleanest way to prove augmentation is the culprit.

## 1. The science: augmentation is a prior over label-invariant transforms

Before we hunt bugs, we need to be precise about what augmentation *is*, because the whole class of failures follows directly from the definition. When you understand the assumption augmentation rests on, every failure mode becomes obvious: each one is a place where the assumption is false.

### 1.1 The invariance assumption

Data augmentation encodes a belief about the world: that there is a set of transformations $T$ under which the label does not change. If $x$ is an input with label $y$, and $t \in T$ is a transformation, augmentation asserts

$$
\text{label}(t(x)) = \text{label}(x) = y \quad \text{for all } t \in T.
$$

This is the **invariance assumption**. When it holds, augmenting your data with $t(x)$ gives the model more examples of class $y$ — for free — and pushes it to learn a function that is invariant to $t$, which is usually exactly the kind of robustness you want. A cat is a cat whether the photo is flipped left-to-right, slightly brighter, or cropped a little. So the transformation group "horizontal flip, small brightness change, small crop" is a *correct* prior for natural-image classification: it tells the model "these nuisance variations do not change the answer," and the model generalizes better because it stops wasting capacity on them.

The entire failure surface of augmentation is this: **you applied a $t$ for which the invariance assumption is false**, so $\text{label}(t(x)) \neq y$ but you fed the model the pair $(t(x), y)$ anyway. You have manufactured a mislabeled example. Every "augmentation gone wrong" story reduces to that one sentence.

### 1.2 Why a bad transform raises the train-loss floor

Here is the mechanism that explains the most confusing symptom — train loss that *will not go to zero* even though the model has plenty of capacity. A label-destroying augmentation injects label noise, and label noise puts a hard floor under the achievable loss.

Suppose a fraction $\rho$ of your augmented examples have been flipped to look like a different class but still carry the original label. Consider a region of input space where the augmented image looks like class $b$ but is labeled $a$ with probability $\rho$ and is a genuine $a$ with probability $1-\rho$. The model sees, for inputs that look like $b$, the target distribution

$$
P(y = a \mid x \text{ looks like } b) = \rho, \qquad P(y = b \mid \cdots) = 1 - \rho .
$$

The model cannot do better than predict this conditional distribution. Even a perfect model, in that region, incurs the irreducible cross-entropy of the label distribution — the conditional entropy

$$
H(y \mid x) = -\big[\rho \log \rho + (1-\rho)\log(1-\rho)\big] .
$$

For a flip that confuses 6 and 9 affecting, say, 10% of digits in a heavily-flipped batch, that is a nonzero loss floor the optimizer can never cross, no matter how big you make the network or how long you train. That is why the team in the intro saw train loss stall: they had added an irreducible-entropy floor to their own loss landscape by injecting label noise through the flip. The deep, counterintuitive point is that **over-strong or label-breaking augmentation does not just slow learning — it makes a region of the target genuinely random, and a perfect model still loses there.** If you have read [reading the loss curve as a diagnostic](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), this is the "loss plateaus above zero on a memorizable problem" shape, and augmentation is one of its top causes.

#### Worked example: how much accuracy a flipped digit costs

Let us put numbers on it. MNIST has roughly equal class frequencies, about 10% per digit. The digits genuinely confused by a horizontal flip are mainly the pair (6, 9) — flipping a 6 yields a shape close to a 9 and vice versa — and to a lesser extent 2/5 and the open digits. Be honest about scope: a horizontal flip does *not* break every digit. A 0, 1, or 8 is roughly flip-symmetric, and a 3 stays a 3. So the corruption is concentrated.

Take the worst-affected slice: the (6, 9) pair is about 20% of the dataset (two of ten classes), and a horizontal flip maps essentially all of them to the other class's appearance. If you apply the flip with probability 0.5 (the default `p=0.5`), then on those classes roughly half the training signal points at the wrong-looking image. Empirically, teams report MNIST dropping from around 99.0% to around 96–97% when a horizontal flip is added — a 2 to 3 point regression that is *entirely self-inflicted*. The 6/9 confusion alone, if it pushes per-class accuracy on those two digits from 99% down to the low 90s, accounts for most of it: $0.2 \times (0.99 - 0.92) \approx 1.4$ points lost on those classes, plus collateral damage from rotation on 2/5/7. The fix — delete the horizontal flip, cap rotation at a few degrees — recovers all of it. We will see the confirming experiment in Section 7.

### 1.3 Augmentation as a bias–variance trade-off

The other side of the science is that augmentation is a *dial*, and both extremes hurt — for opposite reasons.

Think of augmentation strength as a regularization knob, exactly like weight decay or dropout. With **no augmentation**, the model sees a fixed, small set of points and can memorize them, including their noise — classic overfitting, a large train–val gap. With **mild, valid augmentation**, each epoch presents slightly different versions of each example, so the effective dataset is larger and the model is pushed toward the invariance you actually want — the train–val gap shrinks and generalization improves. With **too-strong augmentation**, you cross a threshold where the transform starts to destroy the class-discriminative signal itself (a digit cropped to a corner, a sign rotated past recognition, color jitter so wide a red light looks green) — now you are injecting noise and the model *underfits*, with train loss that cannot drop. The result is a U-shaped curve in validation error as a function of augmentation strength, and the bug is being on the wrong side of the minimum. The matrix below is the field guide we will build out: each failure mode, its mechanism, and the one check that catches it.

![A four-row matrix mapping each augmentation failure mode to its mechanism, its symptom, and the single confirming check that catches it before training](/imgs/blogs/augmentation-gone-wrong-2.png)

### 1.4 The group-theoretic view, and why it predicts which transforms are safe

There is a cleaner way to state the invariance assumption that makes the failure modes fall out almost mechanically, and it is worth a paragraph because it turns "I have a bad feeling about this flip" into a checkable property. A set of label-preserving transformations forms, ideally, a *group*: it contains the identity, it is closed under composition (applying two valid augmentations in sequence is still valid), and each transform has an inverse. The label is then an *invariant* of the group action — a function constant on each orbit. When you choose an augmentation policy, you are implicitly asserting that your label is invariant under the group generated by your transforms.

The bugs are precisely the cases where your label is *not* invariant under the group you chose. A horizontal flip generates a two-element group $\{e, \text{flip}\}$; the digit label is not constant on the orbit $\{6, \text{flip}(6)\}$ because $\text{flip}(6)$ looks like a 9, so the assertion is false and you have a bug. For ImageNet object classes, the label *is* roughly constant on flip orbits (a cat faces either way), so the assertion holds. This is why the validity question always comes back to the label: the *same* transform group is a correct prior for one labeling and a wrong prior for another, because invariance is a joint property of the transform and the label function, never of the transform alone. The practical upshot: for every transform you add, name an input whose label *could* change under it, and if you can name one, that transform is a bug for that slice of your data. The rest of this post is a catalog of the transforms whose orbits people most often forget to check.

### 1.5 The augmentation budget: how much label noise can you afford?

The conditional-entropy argument also tells you *how much* of a bad augmentation you can tolerate before it shows up, which matters because real policies apply transforms with probability $p < 1$. If a label-destroying transform fires with probability $p$ and affects a fraction $q$ of classes (the orientation-sensitive ones), then the induced label-noise rate over the whole dataset is roughly $p \cdot q$ on the affected slice. The accuracy ceiling drops by approximately the test-set fraction that resembles the corrupted region times the per-region confusion. For the MNIST flip with $p = 0.5$ and the (6, 9) pair at $q = 0.2$, that is a $0.1$ corruption rate on a fifth of the data — enough to cost a couple of points, exactly as observed, but not enough to crash the run. This is why these bugs hide: at typical augmentation probabilities the damage is a few points, never a catastrophe, so it reads as "the model is a bit weak" rather than "the pipeline is broken." Lower the probability and the damage shrinks proportionally, which is also a tempting *wrong* fix — you can mask a label-destroying transform by setting its probability low, but you are still injecting noise; the right fix is to remove the transform, not to dilute it.

## 2. The single best diagnostic: look at your augmented data with labels

I will give you the most important tool first, because it catches more augmentation bugs than every other technique in this post combined, and it takes thirty seconds. **Render a grid of augmented training samples with their labels printed on them, and look at it with your own eyes.** If a flipped 6 is labeled 6, you will see it. If a box has drifted off the object, you will see it. If the image is all gray because you normalized before converting to a displayable range, you will see it. This is the augmentation-specific instance of the series-wide discipline from [look at your data before you train](/blog/machine-learning/debugging-training/look-at-your-data-before-you-train): the cheapest, highest-yield debugging action in machine learning is to *actually look*.

Here is a reusable visualizer for a classification dataset using `torchvision`. It pulls a batch straight from your training `Dataset` — *after* augmentation, the way the model sees it — un-normalizes for display, and tiles it with labels.

```python
import torch
import torchvision
import matplotlib.pyplot as plt

# ImageNet stats are the usual normalize step; invert them only for display.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def show_augmented_batch(dataset, class_names, n=25, cols=5):
    """Render n augmented samples straight from the training Dataset, with labels."""
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        img, label = dataset[i]                 # img is post-augmentation
        img = img * STD + MEAN                   # invert normalization for display
        img = img.clamp(0, 1).permute(1, 2, 0)   # CHW -> HWC for matplotlib
        ax.imshow(img.numpy())
        ax.set_title(class_names[label], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("aug_check.png", dpi=120)        # save so you can zoom in
    print("wrote aug_check.png -- OPEN IT and read every label")

# Usage: show_augmented_batch(train_ds, ["0","1","2","3","4","5","6","7","8","9"])
```

Three rules make this diagnostic actually work. First, pull from the *training* dataset, not a fresh un-augmented one — you want to see what the model sees. Second, **invert the normalization** before displaying, or every image will look like gray static and you will learn nothing (this is itself a common confusion we cover in Section 6). Third, **print the label on every tile**, because the entire failure class is "image transformed, label unchanged," and you can only catch it by reading the two together. Spend ninety seconds reading that grid before any training run that uses augmentation. It is the discipline that would have saved the intro team a week.

There is a subtler version of this check that catches the bugs your eyes miss: assert an *invariant* programmatically. For a classification task you can assert that the set of class labels in a batch is unchanged by augmentation (augmentation should never change which class an image belongs to), and for detection you can assert that every transformed box still lies inside the image bounds. We will build those asserts as we hit each modality.

Two failure shapes the visual grid surfaces that nothing else does. The first is the **gray-static grid**: every tile is featureless noise. That is not an augmentation bug at all — it is the display bug of forgetting to invert normalization (the model wants zero-mean unit-variance inputs, but matplotlib expects $[0, 1]$), and it is worth recognizing instantly so you do not chase a phantom. The second is the **wrong-class-right-label** pattern that is the heart of this post: a tile that clearly shows one class with another class's label printed on it. Train your eye to scan for that specific mismatch, because it is the signature of the most damaging augmentation bug and it is invisible in any aggregate metric.

A complementary numeric check verifies that augmentation has not changed the *class composition* of a batch. Augmentation should permute and perturb images but never move an image from one class to another, so the multiset of labels must be identical before and after the transform. This catches the rare-but-real bug where a buggy custom transform (or a mislabeled augmentation that pastes a different class's patch and relabels) silently changes labels.

```python
from collections import Counter

def assert_label_composition_unchanged(raw_dataset, aug_dataset, n=512):
    """Augmentation must not change WHICH class an example belongs to."""
    raw_labels = Counter(raw_dataset[i][1] for i in range(n))
    aug_labels = Counter(aug_dataset[i][1] for i in range(n))
    assert raw_labels == aug_labels, (
        f"AUGMENTATION CHANGED LABEL COMPOSITION:\n"
        f"  raw: {dict(raw_labels)}\n  aug: {dict(aug_labels)}"
    )
    print("OK: augmentation preserves class composition")
```

If that assert fails, an augmentation is rewriting labels, which is a categorically worse bug than a label-destroying transform (which at least leaves the recorded label alone). Run it once at dataloader-construction time; it is cheap insurance.

## 3. Failure mode one: label-destroying transforms

This is the headline bug and the one with the most variants. A transform is **label-destroying** when applying it changes the true class but not the recorded label. The whole skill is knowing, for *your* task, which transforms break the invariance assumption — and that depends entirely on the labels, not the images.

### 3.1 The canonical cases

**Horizontal flip on orientation-sensitive classes.** The 6↔9 digit case is the cleanest example, but it generalizes. Any class whose identity depends on left–right orientation is broken by a horizontal flip: handwritten or printed text and characters (a `b` flips to a `d`, a `p` to a `q`), road signs with directional arrows ("turn left" becomes "turn right"), medical images where laterality matters (a left-lung finding flips to the right lung), and any "which way is the object facing" task. The default `RandomHorizontalFlip(p=0.5)` is *correct* for ImageNet-style object recognition (a cat faces either way) and *wrong* for all of the above.

**Vertical flip on natural scenes.** Most photographs have a gravity prior: sky is up, ground is down. A vertical flip puts the sky at the bottom and is almost never a valid augmentation for natural-scene classification or detection. It *is* valid for some domains with no canonical orientation — satellite/aerial imagery, microscopy, some medical slices — which is exactly the point: validity is a property of the *task*, not the transform.

**Rotation on "which way is up" tasks.** A large rotation breaks digit recognition (a rotated 6 becomes a 9, a rotated 2 looks like a 7), text, and any task where orientation is part of the class. The bug is usually a too-large `degrees` range copied from an ImageNet recipe: `RandomRotation(30)` is reasonable for some object recognition but ruinous for MNIST, where even `RandomRotation(15)` starts to confuse digits.

**Color jitter on a color-classification task.** If the *label* is a color — classifying traffic-light state (red/yellow/green), grading fruit ripeness by color, detecting a specific colored uniform — then `ColorJitter(hue=0.5)` literally changes the answer. Hue jitter on a red light can produce a green light, labeled red. This is the color-channel analog of the geometric flip, and it is easy to miss because color jitter is so standard for natural images.

The unifying lesson is in the figure from Section 1: **the validity of a transform is a property of the label semantics, not the pixels.** Before adding any augmentation, ask: "Is there a $t$ in this set that changes the true class of *some* input I have?" If yes, that $t$ is a bug for those inputs.

### 3.2 The mechanism, restated

Why does this produce the specific symptom of a raised train-loss floor and a few points of lost accuracy, rather than an outright crash? Because the corrupted examples are a *minority* and they are *consistent* — every flipped 6 looks the same wrong way. The model does the rational thing: it learns the dominant signal (most 6-labeled images really are 6-shaped) and treats the flipped minority as noise it cannot fit. Loss settles at the conditional entropy of the induced label noise (Section 1.2), accuracy drops by the fraction of test examples that resemble the corrupted minority, and nothing ever errors out. This is why it is so insidious: **it degrades gracefully**, looking for all the world like a model that is merely a bit weak. The next section's diagnostic turns that ambiguity into a yes/no answer.

## 4. The diagnostic flow: bisecting to augmentation

You suspect augmentation. Here is how to *prove* it, the same make-it-fail-small way the rest of this series bisects. The augmentation-specific version of the overfit-one-batch test is an A/B: overfit one batch with augmentation **off**, then again with it **on**.

```python
import copy
import torch

def overfit_one_batch(model, batch, optimizer_fn, steps=200, device="cuda"):
    """Train on a single fixed batch and return the final loss."""
    model = copy.deepcopy(model).to(device).train()
    opt = optimizer_fn(model.parameters())
    x, y = batch
    x, y = x.to(device), y.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    last = None
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        last = loss.item()
    return last

# Pull ONE batch with augmentation ON (default training transform).
aug_batch = next(iter(train_loader_aug))
# Pull the SAME indices with augmentation OFF (eval transform: resize+normalize only).
clean_batch = next(iter(train_loader_clean))

opt_fn = lambda p: torch.optim.Adam(p, lr=1e-3)
print("overfit, aug OFF :", overfit_one_batch(model, clean_batch, opt_fn))
print("overfit, aug ON  :", overfit_one_batch(model, aug_batch, opt_fn))
```

Read the two numbers like a diagnosis:

- **Aug off drives loss to ~0, aug on does not.** This is the signature of a *label-destroying or too-strong* augmentation. The model can trivially memorize the clean batch, proving the model, loss, and optimizer are all wired correctly — so the only thing that changed is the augmentation, and it is breaking the supervision. Go look at the augmented grid from Section 2 and find the offending transform.
- **Both drive loss to ~0.** Augmentation is *not* destroying the label on this batch (at least at this strength). Your bug is elsewhere — keep bisecting toward optimization, model code, or evaluation. A clean pass here is exactly as informative as a clean pass in the base overfit test: it rules augmentation *out*.
- **Neither drives loss to ~0.** Augmentation is not your only problem; the model cannot even fit the clean batch, so you have a more fundamental fault (frozen layer, wrong loss reduction, learning rate of zero) that you must fix first.

The beauty of the A/B is that it isolates one variable. You are holding the model, the optimizer, the learning rate, the batch contents, and the labels fixed; the *only* difference between the two runs is whether the augmentation transform ran. If the loss floor moves, the augmentation moved it. This is the cleanest possible attribution.

#### Worked example: reading an A/B that convicts the flip

Concrete numbers from the intro scenario. The ConvNet has about 1.2M parameters; the batch is 64 MNIST digits.

| Run | Final loss @ 200 steps | Interpretation |
| --- | --- | --- |
| Overfit, aug OFF | 0.0008 | Model memorizes 64 digits trivially — model/loss/optimizer all healthy |
| Overfit, aug ON (flip + rot30) | 0.61 | Loss floor is **nonzero** — augmentation is injecting unfittable label noise |
| Overfit, aug ON (rot5, no flip) | 0.004 | Removing the flip and capping rotation lets the batch memorize again |

The first two rows convict the augmentation: same model, same digits, same optimizer, and the only run that cannot reach zero is the one with the flip. The third row identifies *which* part of the policy was the culprit by ablating it. You did not retrain anything large; you ran three 200-step single-batch loops, each a few seconds, and you have a verdict. From here the fix is mechanical: delete `RandomHorizontalFlip`, change `RandomRotation(30)` to `RandomRotation(5)`, re-run the full training. Section 7 has the before→after on the real metric.

## 5. The correct ordering: decode, augment, normalize, transform the target

A whole sub-family of augmentation bugs is not about *which* transform you use but about the *order* and *scope* of the steps. Augmentation lives inside a small pipeline, and three ordering mistakes are common enough to deserve their own section. The figure below is the order you want.

![A vertical stack showing the correct augmentation order from decoding uint8 to geometric augmentation to color augmentation to normalize last with the target transform riding alongside](/imgs/blogs/augmentation-gone-wrong-3.png)

### 5.1 Normalize last, not first

The model wants normalized inputs — typically $(x - \mu)/\sigma$ with per-channel ImageNet statistics, leaving inputs roughly zero-mean unit-variance. But **augmentation must run on the un-normalized image**, for two reasons.

First, many color and intensity augmentations are *defined* in pixel space. `ColorJitter` brightness/contrast/saturation, posterize, solarize, and JPEG-compression augmentations all assume inputs in $[0, 255]$ or $[0, 1]$. If you normalize first, "increase brightness by 20%" multiplies a value that might be $-1.7$, producing a meaningless transform. Second, normalization is a deterministic, label-irrelevant rescaling — it belongs at the very end, as the last step before the tensor enters the model, so that everything upstream operates on interpretable pixel values. The correct order is: **decode (uint8) → geometric augmentation → color augmentation → to-float → normalize.** Getting this backwards usually shows up as augmentations that "do nothing" or produce garbage, and as the gray-static images you saw when you forgot to invert normalization in the visualizer.

```python
from torchvision import transforms

# CORRECT: geometric + color on PIL/uint8, ToTensor + Normalize LAST.
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),          # geometric, on PIL image
    transforms.RandomHorizontalFlip(p=0.5),     # geometric (valid for natural images!)
    transforms.ColorJitter(0.4, 0.4, 0.4),      # color, on uint8/PIL
    transforms.ToTensor(),                       # -> float [0,1]
    transforms.Normalize([0.485, 0.456, 0.406], # NORMALIZE LAST
                         [0.229, 0.224, 0.225]),
])

# WRONG: Normalize before ColorJitter -> jitter operates on normalized floats.
# transforms.Compose([ToTensor(), Normalize(...), ColorJitter(...)])  # bug
```

### 5.2 The target rides with the image

The deeper ordering bug is forgetting that the *target* sometimes has to be transformed too. For plain classification the label is invariant under geometry, so this never comes up. But the moment your label has spatial extent — a bounding box, a segmentation mask, a set of keypoints — a geometric augmentation on the image **must** apply the identical geometric transform to the target. This is failure mode five and it has its own section (Section 8) because it is the single most common detection/segmentation augmentation bug. The ordering principle is simply: geometry is applied jointly to (image, target), and only the image is normalized.

## 6. Failure mode: augmentation leaking into validation, and TTA skew

Now a bug that does not corrupt training at all — it corrupts your *measurement*. If random augmentation runs on the validation or test set, your metric becomes noisy and biased, and every decision you make from it (early stopping, model selection, the number you report) is built on sand. The fix is a one-line discipline, but the bug is shockingly common because of how transform code gets shared.

### 6.1 The mechanism: a random metric

Validation is supposed to estimate performance on the deployment distribution with a *fixed* evaluation protocol. If your val transform includes `RandomResizedCrop` or `RandomHorizontalFlip`, then evaluating the *same* model on the *same* val set twice gives two different numbers, because the inputs themselves are random. Worse, the augmentation usually makes images *harder* (a random crop can cut off the object), so the val metric is biased *downward* — you systematically under-report accuracy. And because early stopping and checkpoint selection compare val numbers across epochs, you may "select" the epoch that got a lucky easy crop, not the genuinely best model. The before→after is stark.

![A two-column comparison of a validation set that receives training augmentation and swings two points between runs versus a deterministic resize-and-normalize validation that is stable to a tenth of a point](/imgs/blogs/augmentation-gone-wrong-6.png)

The correct pattern is two separate transforms — a stochastic training transform and a *deterministic* evaluation transform (resize/center-crop + normalize, nothing random) — and an assertion that they never get crossed.

```python
from torchvision import transforms

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Validation: DETERMINISTIC. Resize + center crop + normalize. Nothing random.
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def assert_deterministic_eval(transform):
    """Fail loudly if any 'Random' transform leaked into eval."""
    names = [type(t).__name__ for t in transform.transforms]
    bad = [n for n in names if n.startswith("Random") or "Jitter" in n]
    assert not bad, f"VAL TRANSFORM HAS RANDOM AUG: {bad}"

assert_deterministic_eval(eval_tf)   # run this in your test suite
```

That `assert_deterministic_eval` belongs in your test suite or at dataloader-construction time. It is three lines and it permanently closes a bug that otherwise re-emerges every time someone refactors the transforms. This connects directly to [the input pipeline is lying to you](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you): the val pipeline silently differing from your intent is the same family of fault, and the same print-and-assert discipline catches it.

### 6.2 Test-time augmentation skew

There is a *legitimate* use of augmentation at evaluation: **test-time augmentation (TTA)**, where you deliberately run several augmented versions of each test image through the model and average the predictions to reduce variance. TTA can genuinely add a fraction of a point. But it introduces two skew bugs. First, if you report a TTA val number and then deploy *without* TTA, your production number will be lower than your reported one — a train/eval/serve mismatch. Second, if you use TTA with a label-destroying transform (averaging the prediction on a flipped 6, which the model reads as a 9), you actively corrupt the test prediction. The rule: TTA must use the *same* invariance assumptions as training — only label-preserving transforms — and you must report which number (TTA or single-crop) you are quoting and match it to how you deploy.

The mechanics of TTA make the second bug precise. TTA averages predictions over a set of transforms $\{t_1, \ldots, t_k\}$:

$$
\hat{p}(y \mid x) = \frac{1}{k} \sum_{i=1}^{k} f\big(t_i(x)\big) .
$$

This reduces variance *only if every $t_i$ is label-preserving*, because the average assumes all $k$ views agree on the true label. Slip a horizontal flip into the TTA set for a digit classifier and one of your $k$ views is asking the model to classify a 6 that now looks like a 9 — its prediction pulls the average toward the wrong class, and TTA *hurts*. The same group-theoretic discipline from Section 1.4 governs TTA: the test-time transform set must be a subset of the *valid* (label-preserving) transforms for your task, never a copy of an ImageNet TTA recipe applied blindly. The practical check: TTA should never lower single-crop accuracy on a held-out set; if it does, a label-destroying transform is in your TTA list, and the fix is to remove it — the identical fix as for the training-side flip, for the identical reason.

#### Worked example: the val-aug metric correction

A team reports 86.0% val accuracy and is thrilled. On inspection, the val loader uses the *training* transform with `RandomResizedCrop(224)`. Two corrections are needed. First, switch val to deterministic `Resize(256) + CenterCrop(224)`: the number moves to **84.2%** and, crucially, stops swinging — repeated evaluations now agree to ±0.1 instead of ±2.3. The 86.0% was not real; it was a noisy draw. Second, they realize they had been early-stopping on the noisy 86-ish numbers and had saved a checkpoint from a lucky-crop epoch; re-running model selection on the stable metric picks a *different*, genuinely better checkpoint that reaches **84.7%**. Net: the headline number went *down* by 1.3 points, but it is now honest, reproducible, and matches production — and they actually have a better model. A metric that lies in your favor is worse than no metric.

## 7. Failure mode: too-strong and too-weak policies

Augmentation strength is a U-shaped dial (Section 1.3). Both ends are bugs, and they have opposite signatures, which is what lets you tell them apart.

### 7.1 Too strong: the model underfits

A too-strong policy destroys class-discriminative signal even when no single transform is *categorically* label-destroying. `RandAugment` and `TrivialAugment` apply several random operations at a sampled magnitude; crank the magnitude or the number of operations and you produce images that even a human cannot classify — a digit cropped to a sliver, a sign rotated and sheared past recognition, contrast pushed until the object vanishes. The symptom is the one we have built up: **train loss has a floor it will not cross, and val accuracy is low because the model genuinely underfit.** This is distinguishable from a label-destroying transform by the A/B test (Section 4): too-strong augmentation also fails the aug-on overfit, but visualizing the grid shows degraded-but-correctly-labeled images rather than the specific "wrong class, original label" pattern.

The before→after is a magnitude sweep.

![A two-column comparison showing RandAugment at magnitude nine erasing signal and capping val accuracy at seventy-one percent versus magnitude five recovering the train loss and reaching eighty-four percent](/imgs/blogs/augmentation-gone-wrong-4.png)

```python
from torchvision.transforms import RandAugment, Compose, ToTensor, Normalize

# Sweep magnitude; watch train-loss floor and val acc. Tune toward the minimum.
def build_train_tf(magnitude, num_ops=2):
    return Compose([
        RandAugment(num_ops=num_ops, magnitude=magnitude),  # M in [0, 30]
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# Try magnitude in {3, 5, 7, 9}. Symptom of too-strong: train loss won't drop
# AND the augmented grid (Section 2) shows unrecognizable-but-correctly-labeled images.
```

### 7.2 Too weak: the model overfits

The opposite end is just as real but less dramatic: **no augmentation (or far too little) lets the model memorize the training set**, including its noise, producing a large train–val gap — train accuracy 99%+, val accuracy stuck and rising no further. This is the classic overfitting signature, and augmentation is one of the first regularizers to reach for. The diagnostic is the train–val gap on the loss curve, and the fix is to *add* mild, valid augmentation and watch the gap shrink. The trap is over-correcting straight past the minimum into the too-strong regime, which is why you sweep rather than guess.

The two ends share a deceptive overlap: low validation accuracy. What disambiguates them is the *train* loss. Too-weak: train loss → ~0, val lags (overfitting). Too-strong: train loss *floored above zero*, val also low (underfitting). One curve tells you which way to turn the dial.

#### Worked example: a magnitude sweep that finds the minimum

To make the U-shape concrete, here is a four-point sweep on a small image classifier (a ResNet-18 finetune on a 20,000-image dataset, 50 epochs each), holding everything else fixed and varying only `RandAugment` magnitude. The instrument that matters is the *pair* (final train loss, best val accuracy), because together they place you on the curve.

| Magnitude | Final train loss | Best val acc | Diagnosis |
| --- | --- | --- | --- |
| 0 (no aug) | 0.02 | 76.1% | overfitting — train ~0, large val gap |
| 3 | 0.18 | 82.4% | under-regularized, improving |
| 5 | 0.31 | 84.3% | near the minimum — best val |
| 7 | 0.74 | 82.0% | starting to underfit |
| 9 | 1.42 | 71.3% | too strong — train floored, val collapses |

The shape is unmistakable: validation accuracy rises from 76.1% to a peak of 84.3% at magnitude 5, then falls back to 71.3% at magnitude 9 — *below* the no-augmentation baseline. Read the train-loss column alongside it and the mechanism is plain: at magnitude 0 the train loss is essentially zero (the model memorized, hence the gap), and at magnitude 9 the train loss is floored at 1.42 (the augmentation injected so much noise that the model cannot even fit the training set). The minimum is where train loss is moderate (0.31) and the train–val gap is small. The cost of getting this wrong in either direction is 8–13 points of accuracy, which is why the four-run sweep — a few GPU-hours total — is one of the highest-return experiments you can run, far cheaper than the architecture search teams reach for instead.

The sweep is also a clean illustration of the series' make-it-fail-small philosophy applied to a hyperparameter: rather than reasoning about the "right" augmentation strength in the abstract, you vary one knob across a small grid, read two instruments, and let the curve tell you where the minimum is. You do not need to understand *why* magnitude 5 is optimal for this dataset to *find* that it is.

| Regime | Train loss | Val acc | Train–val gap | The fix |
| --- | --- | --- | --- | --- |
| Too weak / none | → 0.02 | 76% | large (overfit) | add mild valid aug |
| Tuned | → 0.30 | 84% | small | leave it |
| Too strong | floored 1.4 | 71% | small (underfit) | lower magnitude / drop bad transform |
| Label-destroying | floored 0.6 | 88% | small | remove the offending transform |

## 8. Failure mode: geometric augs that move the input but not the target

This is the most expensive augmentation bug because it silently caps your model at near-zero performance on a task that *looks* like it should work — and it only happens in detection, segmentation, and keypoint tasks, where the label has spatial structure. When you flip or rotate the image, the bounding box, mask, or keypoints must take the *identical* transform, or the supervision becomes a lie: the image says "object on the right," the box says "object on the left," and the model is asked to reconcile a contradiction on every example.

![A branching graph showing an image and box where flipping the box in lockstep yields an aligned target that trains while forgetting the box leaves a misaligned target and mAP near zero](/imgs/blogs/augmentation-gone-wrong-5.png)

### 8.1 The mechanism

In detection, the model learns to regress box coordinates from image features. If a horizontal flip moves the object's pixels to the mirrored location but the *target* box still points at the original location, then for every flipped example the model is told "the object whose features are now on the right is actually on the left." Averaged over a dataset where roughly half the images are flipped, the regression target becomes inconsistent — the same visual feature maps to two contradictory box positions — and the localization head learns to predict the *mean*, which is garbage. The symptom is unmistakable once you know it: **mAP is glued near zero (or to a low ceiling) and never improves, while the classification part of the loss may look fine.** This is precisely the bug the matrix in Section 1 routes to a single check: draw the boxes on the augmented image and *look*.

### 8.2 The fix: transform image and target together with albumentations

Do not hand-roll geometric augmentation for detection. Use a library that transforms the target in lockstep. `albumentations` is the standard: you declare the bounding-box format once, and every spatial transform updates the boxes automatically.

```python
import albumentations as A

# Transform image AND boxes together. bbox_params ties them in lockstep.
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),        # flips image AND every box's x-coords
        A.RandomRotate90(p=0.5),        # rotates image AND boxes together
        A.RandomBrightnessContrast(p=0.3),
        A.Resize(640, 640),
        A.Normalize(),                  # normalize LAST (Section 5)
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",            # [x_min, y_min, x_max, y_max]
        label_fields=["class_labels"],
        min_visibility=0.3,             # drop boxes a crop cut down too far
    ),
)

out = transform(image=img, bboxes=boxes, class_labels=labels)
aug_img, aug_boxes, aug_labels = out["image"], out["bboxes"], out["class_labels"]
```

Two details earn their keep. The `format` must match how your boxes are actually stored — `pascal_voc` is `[x_min, y_min, x_max, y_max]`, `coco` is `[x, y, width, height]`, `yolo` is normalized `[cx, cy, w, h]` — and getting it wrong is its own silent bug (the box transforms, but from the wrong coordinate convention, so it lands in the wrong place). The `min_visibility` parameter handles the case where a crop removes most of an object: the box should be *dropped*, not left pointing at a sliver. Masks and keypoints work identically (`A.Compose(..., keypoint_params=...)`, and masks pass through `transform(image=..., mask=...)` with nearest-neighbor interpolation so you do not blend class indices).

This is the detection/segmentation instantiation of [augmentation debugging for vision](/blog/machine-learning/debugging-training/augmentation-debugging-for-vision), which goes deeper on coordinate-format swaps and mask-interpolation traps. The confirming check is the same everywhere: render the augmented image with the augmented boxes drawn on it, and verify the boxes still bound the objects.

#### Worked example: the detector whose mAP would not move

A realistic detection scenario, because the numbers are so different from the classification cases. A team finetunes a detector on a 5,000-image custom dataset. The training loss decreases smoothly from 4.2 to 1.1 over 30 epochs, the GPUs are pinned, and everything *looks* like a healthy run — but validation mAP@0.5 is stuck at 0.04 and has not moved since epoch 2. Classification accuracy on the objectness head is fine; it is purely *localization* that is broken. The hand-rolled augmentation applied `RandomHorizontalFlip` to the image tensor but the box coordinates were stored in a separate array that the flip never touched. So for the ~50% of images that got flipped, the regression target pointed at the mirror-image location of the object. The localization head, asked to predict two contradictory positions for the same visual feature, learned to predict the dataset-mean box, which overlaps almost nothing — hence mAP near zero. The confirming test took two minutes: `draw_and_save` on ten augmented samples showed boxes sitting on empty background while the object was on the opposite side. The fix was to move to `albumentations` with `bbox_params`, after which mAP@0.5 climbed from 0.04 to 0.58 over the same 30 epochs — a 14× improvement from deleting one bug, no architecture or hyperparameter change. The lesson is the cost asymmetry: a label-destroying classification flip costs a few points, but an un-transformed detection target costs *almost everything*, because localization has no fallback signal.

```python
import cv2

def draw_and_save(image, boxes, labels, path="box_check.png"):
    """Draw augmented boxes on the augmented image. Boxes must hug the objects."""
    vis = image.copy()
    for (x0, y0, x1, y1), lab in zip(boxes, labels):
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
        cv2.putText(vis, str(lab), (int(x0), int(y0) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(path, vis)
    print(f"wrote {path} -- boxes must hug the objects after augmentation")
```

## 9. Failure mode: mixup and cutmix label-mixing bugs

Mixup and cutmix are powerful regularizers that break the simple "label is invariant" frame: they *deliberately* change the label, and the bug is changing it *wrong*. Understanding the loss formulation makes the bug obvious.

### 9.1 The mixup formulation

Mixup takes two examples $(x_a, y_a)$ and $(x_b, y_b)$, samples a mixing coefficient $\lambda \sim \text{Beta}(\alpha, \alpha)$, and forms a single training example by interpolating *both* the inputs and the labels:

$$
\tilde{x} = \lambda\, x_a + (1-\lambda)\, x_b, \qquad \tilde{y} = \lambda\, y_a + (1-\lambda)\, y_b .
$$

The label $\tilde{y}$ is a *soft* target: a blend of the two one-hot vectors. The loss is cross-entropy against that soft target, which is identical to mixing the two per-example losses:

$$
\mathcal{L} = \lambda \cdot \text{CE}(f(\tilde{x}), y_a) + (1-\lambda)\cdot \text{CE}(f(\tilde{x}), y_b) .
$$

The whole point is that the model is trained to predict a *combination* of classes for a *combination* of inputs, which smooths its decision boundaries. Cutmix is the same idea with a spatial twist: instead of blending pixels, it pastes a rectangular patch of image $b$ onto image $a$, and $\lambda$ is the *area fraction* of the original image that survives — so the label mix must use the patch's area ratio, not a separately-sampled scalar.

### 9.2 The bugs

The figure shows the failure: mix the pixels but leave the label one-hot, and the target is now inconsistent with the input.

![A branching graph showing mixup blending two images by lambda where mixing the labels by the same lambda yields a consistent target but keeping the one-hot label gives a wrong target with no benefit](/imgs/blogs/augmentation-gone-wrong-7.png)

The common mixup/cutmix bugs:

- **Mixing the image but not the label.** You interpolate pixels but train against the original one-hot $y_a$. Now you are telling the model "this 70% cat / 30% dog image is 100% cat," which is just label noise — the regularization benefit vanishes and you may even hurt accuracy.
- **Cutmix using the wrong $\lambda$ for the label.** You paste a patch covering 30% of the image but mix the label with a separately-sampled $\lambda$ instead of the actual area ratio. The label no longer matches the pixel composition.
- **Applying mixup at eval time.** Mixup is a *training-only* augmentation; it must be off for validation and inference, or you are evaluating on blended images that do not exist in deployment.
- **The BatchNorm interaction.** Mixup changes the input statistics each batch (blended images have different pixel distributions), which interacts with BatchNorm's running-statistics estimation. With mixup, the running mean/variance BN accumulates during training can drift from the statistics of clean eval images, contributing a small train/eval gap. This is usually minor, but on small batches it compounds with BN's existing small-batch fragility — a connection to [initialization and normalization bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs).

Here is a correct mixup that mixes both tensors with the *same* $\lambda$ and computes the mixed loss.

```python
import numpy as np
import torch
import torch.nn.functional as F

def mixup_batch(x, y, alpha=0.2):
    """Return mixed inputs and the (y_a, y_b, lam) needed for the mixed loss."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[perm]   # SAME lam mixes the pixels...
    y_a, y_b = y, y[perm]
    return mixed_x, y_a, y_b, lam               # ...and is returned for the label

def mixup_loss(logits, y_a, y_b, lam):
    """Mix the TWO losses with the SAME lam used to mix the inputs."""
    return lam * F.cross_entropy(logits, y_a) + (1.0 - lam) * F.cross_entropy(logits, y_b)

# Training step (mixup ON in train, OFF in eval):
mixed_x, y_a, y_b, lam = mixup_batch(x, y, alpha=0.2)
logits = model(mixed_x)
loss = mixup_loss(logits, y_a, y_b, lam)        # NOT cross_entropy(logits, y)!
loss.backward()
```

The single most common bug this code prevents is the last line: a tired engineer writes `F.cross_entropy(logits, y)` (the original label) instead of the mixed loss, and silently disables the entire technique. Confirm it works by checking that the loss responds to $\lambda$: at $\lambda = 0.5$ a correctly-mixed loss on two different classes cannot reach zero (the target is genuinely 50/50), whereas a buggy one-hot loss could. If your "mixup" run reaches near-zero train loss as easily as a non-mixup run, the label mixing is broken.

## 10. Modality-specific traps: speech, NLP, and tabular

Augmentation failures are not a vision-only phenomenon. The invariance assumption breaks in modality-specific ways, and the symptoms differ, but the diagnostic is the same: visualize/inspect the augmented example *with its label* before trusting it.

### 10.1 Speech: time-warp and pitch-shift that break labels

For speech recognition, **SpecAugment** (time masking, frequency masking, time warping on the spectrogram) is the standard, and it is mostly label-preserving — masking a few time/frequency bands does not change the words. But two augmentations cross the line. **Pitch-shift** can change a speaker's identity (a bug for speaker-ID tasks) and, pushed far, can alter phoneme perception. **Time-warp / speed perturbation**, if too aggressive, distorts the temporal alignment that CTC or attention relies on, and in extreme cases changes which phonemes are audible. The classic too-strong-SpecAugment symptom is a Whisper or wav2vec2 finetune whose word error rate refuses to drop — the masking is erasing so much of the signal that the model underfits, exactly the Section 7 mechanism in the audio domain. The connection runs to the CTC/alignment family of bugs: when augmentation distorts timing, alignment-based losses are the first to complain.

There is a subtler speech-specific trap worth naming: the **WER metric itself lies if normalization differs between training and evaluation**. SpecAugment is applied to the input, but the *output* — the transcription — has its own normalization (lowercasing, punctuation stripping, number formatting). If your eval applies a different text normalization than your training labels, WER can swing several points without any augmentation change. So a speech debugging session has two augmentation-adjacent suspects: the input masking is too strong (visualize the masked spectrogram — yes, the same "look at it" discipline, plotted as an image), and the output normalization is inconsistent (diff the normalized reference against the normalized hypothesis on a few examples). The masked-spectrogram visualization is the audio analog of the augmented-image grid:

```python
import torchaudio
import matplotlib.pyplot as plt

def show_specaugment(waveform, sample_rate, time_mask=80, freq_mask=27):
    """Plot a mel-spectrogram before and after SpecAugment masking. LOOK at it."""
    melspec = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=80)(waveform)
    log_mel = torch.log1p(melspec)                              # log scale for display
    masked = torchaudio.transforms.FrequencyMasking(freq_mask)(log_mel.clone())
    masked = torchaudio.transforms.TimeMasking(time_mask)(masked)
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 5))
    a.imshow(log_mel[0].numpy(), origin="lower", aspect="auto"); a.set_title("clean mel")
    b.imshow(masked[0].numpy(), origin="lower", aspect="auto"); b.set_title("after SpecAugment")
    plt.tight_layout(); plt.savefig("specaug_check.png", dpi=120)
    print("wrote specaug_check.png -- if the masks erase most of the signal, it's too strong")
```

If the masked spectrogram has so much blacked out that *you* could not transcribe it, the policy is too strong and the model will underfit — the exact too-strong signature from Section 7, made visible.

### 10.2 NLP: token-level augmentation breaking spans

Text augmentation (synonym replacement, random insertion/deletion/swap, back-translation) is treacherous because the *label* is often tied to specific tokens. For sentiment or topic classification, swapping a word might be fine. But for **named-entity recognition or span extraction**, where the label is a span of token indices, *any* insertion or deletion shifts the span boundaries — and if you do not update the label spans in lockstep (the NLP analog of the box-in-lockstep bug from Section 8), your entity labels now point at the wrong tokens. Synonym replacement can also flip the label outright: replacing "not" or negating a sentiment word inverts the class. The rule mirrors detection: token-level geometric edits to the text must update token-level labels in lockstep, and any augmentation that can change meaning is unsafe for meaning-dependent labels.

The span-shift bug is exactly the detection bug in another costume. Consider an NER example where the entity "Barack Obama" occupies token positions 3–4 and is labeled `PERSON`. A random-insertion augmentation drops a word at position 1; now the entity sits at positions 4–5, but the label still says 3–4. The model is trained to tag the *wrong* tokens as the person, and entity-level F1 drops. The fix is to treat token indices like box coordinates: any edit that inserts or deletes tokens must remap the label spans.

```python
def insert_token_safe(tokens, labels, position, new_token):
    """Insert a token AND shift every span label after it -- the NLP box-in-lockstep."""
    tokens = tokens[:position] + [new_token] + tokens[position:]
    # labels are (start, end, tag); any span at or after `position` shifts by +1.
    new_labels = []
    for start, end, tag in labels:
        if start >= position:
            start, end = start + 1, end + 1     # span moved right by the insertion
        elif start < position <= end:
            end = end + 1                        # insertion landed INSIDE the span
        new_labels.append((start, end, tag))
    return tokens, new_labels
```

If your augmentation library does not remap spans for you, this is the logic it must contain. The confirming check is the text analog of drawing boxes: print the augmented tokens with the labeled span bracketed, and read whether the bracket still surrounds the entity.

### 10.3 Tabular: shuffling columns and other category errors

Tabular augmentation is rarer but has a sharp trap: **shuffling or permuting feature columns**. Unlike images, where translation invariance is real, tabular features are *not* exchangeable — column 3 is "age" and column 7 is "income," and swapping them is not an augmentation, it is data corruption. Some tabular "augmentation" schemes (SMOTE-style interpolation, adding Gaussian noise to features) can also be label-destroying when a small feature change crosses a decision boundary — interpolating between a fraud and a non-fraud example can produce a feature vector whose true label is ambiguous, labeled with the majority. The safe tabular augmentations (feature dropout/masking, mixup on continuous features done carefully) require the same scrutiny: does this transform preserve the label for *my* features? Tabular is where the invariance assumption is least likely to hold by default, because tabular features rarely have a natural nuisance-transformation group at all.

The deeper reason tabular augmentation is dangerous is that the group-theoretic argument from Section 1.4 has almost nothing to act on. For images, there is a genuine nuisance group — translations, small rotations, brightness shifts are *real* transformations that leave the label invariant, grounded in the physics of how the image was captured. For a row of `[age, income, num_transactions, region_code]`, there is no analogous nuisance group: every feature is meaningful, none is exchangeable with another, and there is no continuous transform you can apply that you *know* preserves the label. So the default move — reach for augmentation because it helped on images — is itself the bug. When practitioners do augment tabular data successfully, it is almost always with a domain-specific, label-checked transform (e.g. adding noise within a feature's known measurement error), not a generic one. If you cannot name the nuisance transformation and argue it preserves your label, do not augment the table; the honest baseline is no augmentation plus the regularization that *does* transfer (dropout, weight decay, early stopping).

## 11. The decision tree: which augmentation bug is it?

You have the failure modes, the mechanisms, and the checks. Here is how to route a symptom to a suspect quickly — the augmentation-specific slice of the master taxonomy. The figure is the tree; the text is how to read it.

![A decision tree routing the symptom of stuck train loss to policy strength a noisy validation to validation augmentation and a stalled detection metric to untransformed targets](/imgs/blogs/augmentation-gone-wrong-8.png)

Start, always, by *visualizing fifty augmented samples with labels* (Section 2). Most bugs die there. If they do not, route by symptom:

- **Train loss will not drop (floored above zero), val also low (underfitting).** Augmentation is too strong or label-destroying. Run the aug-off/aug-on overfit A/B (Section 4) to confirm augmentation is the cause, then look at the grid: unrecognizable-but-correctly-labeled images mean too-strong (lower the magnitude); correctly-recognizable-but-wrong-class images (a flipped 6 labeled 6) mean label-destroying (remove that transform).
- **Validation accuracy swings between runs / is suspiciously low.** Augmentation is leaking into the val set (Section 6). Assert the eval transform is deterministic; switch val to resize + center-crop + normalize.
- **Detection mAP / segmentation IoU stuck near zero, classification loss looks okay.** Geometric augmentation is moving the image but not the target (Section 8). Draw the augmented boxes/masks on the augmented image; transform the target in lockstep with albumentations.
- **Mixup/cutmix added but accuracy did not improve (or got worse).** The label mixing is broken (Section 9). Check that the loss uses the mixed soft target with the same $\lambda$, not the original one-hot label.
- **Train loss → 0 easily, val lags badly (large gap).** This is the *opposite* problem — too little augmentation, the model is overfitting. *Add* mild, valid augmentation.

The discipline is the same one that runs through the whole series: the symptom narrows you to a suspect, a cheap test confirms it, and only then do you change code. You never "try turning off augmentation and see if it helps" as a shotgun; you read the loss-and-metric pattern, predict which augmentation fault it implies, and confirm with the matching check.

## 12. Case studies and real signatures

A few well-known, real or realistic signatures, so you recognize them in the wild.

**MNIST and the horizontal flip.** This is folklore precisely because it is so common: practitioners apply an ImageNet-style transform block — including `RandomHorizontalFlip` — to MNIST or to a custom digit/character dataset and watch accuracy drop a few points below the no-augmentation baseline. The mechanism is the 6↔9 (and to a lesser extent 2↔5) confusion from Section 1.2. The fix is one line (delete the flip), and accuracy returns to baseline. The general lesson — *a flip that helps natural images destroys orientation-dependent classes* — is the single most-repeated augmentation mistake, and it generalizes to text recognition, document classification, and any directional-sign task.

**CIFAR and the RandAugment magnitude.** The `RandAugment` paper (Cubuk et al., 2020) showed that augmentation magnitude is a tunable hyperparameter with a clear optimum, and that the optimal magnitude grows with model and dataset size. The practical signature: on a small model or small dataset, a magnitude tuned for a large ImageNet model is *too strong* and underfits — train loss floored, val accuracy below the lighter-augmentation baseline. The fix is the magnitude sweep from Section 7. This is the empirical backbone of the "U-shaped dial" claim: there is a real, measurable minimum.

**Detection and the un-flipped box.** Anyone who has hand-rolled a detection augmentation pipeline has hit this: flip the image, forget the box, and mAP sits near zero through an entire training run that otherwise looks healthy (loss decreasing, GPUs busy). It is the reason mature detection codebases (the COCO ecosystem, `albumentations`, `torchvision`'s detection transforms) all transform image and target jointly by construction — the bug was common enough that the libraries removed the opportunity to make it. If you see a detection run where classification metrics move but localization mAP is pinned low, the un-transformed (or wrong-format) box is the first suspect.

**Mixup's soft-label requirement.** The mixup paper (Zhang et al., 2018) is explicit that the technique requires mixing labels, not just inputs, and that the loss must be computed against the soft target. Reimplementations that mix pixels but keep the one-hot label are a known failure: they report "mixup didn't help" because they never actually did mixup. The confirming check (the loss cannot reach zero at $\lambda = 0.5$ on two classes) distinguishes a working implementation from a broken one.

| Case | Symptom | Confirming test | Fix |
| --- | --- | --- | --- |
| MNIST hflip | acc 99.0% → 96.4% | view aug grid: flipped 6 labeled 6 | remove `RandomHorizontalFlip` |
| RandAugment too strong | train loss floored, val below baseline | magnitude sweep {3,5,7,9} | lower magnitude |
| Detection un-flipped box | mAP ~0, cls loss fine | draw aug boxes on aug image | transform box in lockstep |
| Broken mixup | "mixup didn't help" | loss reaches 0 at λ=0.5 (shouldn't) | mix labels with same λ |
| Val gets train aug | val ±2 pts run-to-run | assert eval transform deterministic | resize + center-crop only |

## 13. When this is (and isn't) your augmentation bug

Augmentation is a frequent culprit, but it is not the answer to every stalled run, and pattern-matching too eagerly wastes time. Be decisive about when to look elsewhere.

**It probably IS augmentation when:** adding or strengthening augmentation made the metric *worse*; the train loss has a *floor it will not cross* on a problem the model used to fit; validation accuracy *swings* between identical re-runs; or detection/segmentation localization metrics are stuck near zero while classification looks fine. The cleanest confirmation is always the aug-off/aug-on overfit A/B (Section 4): if the loss floor moves when you toggle augmentation, augmentation owns the floor.

**It is probably NOT augmentation when:** the overfit-one-batch test fails *even with augmentation off* — then your problem is more fundamental (a frozen layer, a wrong loss reduction, a learning rate of zero), and augmentation is a red herring. A smooth-loss-curve-then-NaN is *numerics*, not augmentation. A too-good-to-be-true val number that *survives* a deterministic eval transform is likely *data leakage*, not augmentation (a different data bug — see [the taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs)). And a model that trains fine but degrades in production, while augmentation is identical, is usually distribution shift, not an augmentation fault.

The bisection rule that keeps you honest: run the overfit-one-batch test *with augmentation off* first. If it fails, augmentation is not your primary bug — fix the more basic fault. If it passes off but fails on, augmentation is your bug. This single ordering prevents the most common waste of time in augmentation debugging, which is staring at transforms when the real problem is a detached gradient.

### 13.1 Stress-testing the diagnosis

Once you believe augmentation is the culprit, stress the conclusion against the adjacent suspects before you commit to a fix — the discipline that separates a guess from a diagnosis.

**What if it is actually data, not augmentation?** A raised train-loss floor can also come from genuine label noise in the *source* data (mislabeled examples that were always there), not from augmentation injecting it. The discriminator is the aug-off overfit: if the loss floor persists with augmentation *off*, the noise is in your labels, not your transforms, and you should be running a confident-learning pass, not editing the augmentation stack. Augmentation owns the floor only if turning it off removes the floor.

**What if it is numerics?** A loss that is smooth and then suddenly spikes to NaN is not an augmentation bug — augmentation degrades loss *gracefully* (a higher floor), it does not produce a sudden divergence. If you see a clean curve then a NaN, you are in numerics territory (a too-high learning rate, an fp16 overflow), and the augmentation is innocent. The signature shapes are genuinely different: augmentation gives a *persistently elevated* floor from step one; numerics gives a *sudden break* after a healthy stretch.

**What if it only fails on multi-GPU?** Augmentation bugs are per-sample and reproduce on a single GPU, so if a run is healthy on one GPU and broken on eight, the fault is in the distributed layer (per-rank seeding making every rank augment identically, or a sharding bug), not in the transform itself. Reproduce the symptom on one GPU before blaming augmentation; if you cannot, the augmentation is fine and the systems layer is the suspect.

**What if the batch is tiny?** Mixup and BatchNorm interact badly at small batch sizes (Section 9), so a mixup run that is fine at batch 256 and broken at batch 8 may be a BatchNorm-statistics problem amplified by mixup's input-distribution shift, not a label-mixing bug. The discriminator is to disable mixup and check whether small-batch BN alone is unstable; if it is, fix the normalization first.

Each of these is the same move: hold augmentation as the hypothesis, find the one experiment that would *distinguish* it from the neighboring suspect, and run that experiment before you touch code. A diagnosis you have tried and failed to falsify is worth ten you merely pattern-matched.

## 14. Key takeaways

- **An augmentation is valid only if it preserves the label.** Validity is a property of your *label semantics*, not the pixels: a horizontal flip is correct for cats and label-destroying for digits, text, signs, and laterality-sensitive medical images.
- **Visualize fifty augmented samples with labels before every run.** This thirty-second check catches more augmentation bugs than every other technique combined. Invert the normalization for display, and read the label on every tile.
- **A label-destroying or too-strong augmentation raises the train-loss floor.** It injects label noise, so even a perfect model cannot reach zero loss in the corrupted region — the floor equals the conditional entropy of the induced noise. Train loss that will not drop on a once-fittable problem points at augmentation.
- **Prove it with the aug-off/aug-on overfit A/B.** Overfit one batch with augmentation off (should hit ~0) and on (floored if augmentation is breaking the label). If the floor moves, augmentation owns it.
- **Normalize last, augment on un-normalized pixels.** Color and intensity transforms are defined in pixel space; normalizing first produces meaningless augmentations and gray-static images.
- **Never augment the validation set.** Random transforms on val make the metric noisy and biased-low; assert the eval transform is deterministic (resize + center-crop + normalize). Match TTA to deployment or do not report it.
- **Geometric augmentation must transform the target in lockstep.** Flip the image, flip the box/mask/keypoints with it, or detection mAP and segmentation IoU stay near zero. Use `albumentations` with the correct box format and draw the augmented boxes to confirm.
- **Mixup/cutmix must mix the label with the same $\lambda$ as the input** and run only at training time; the most common bug is computing the loss against the original one-hot, which silently disables the technique.
- **Augmentation strength is a U-shaped dial.** Too weak overfits (train→0, val lags); too strong underfits (train floored, val low). The train loss tells you which way to turn it; sweep the magnitude rather than guessing.
- **Run overfit-one-batch with augmentation off first.** If it fails there, augmentation is not your bug — fix the more basic fault before touching the transform stack.

## 15. Further reading

- **"mixup: Beyond Empirical Risk Minimization"** — Zhang, Cisse, Dauphin, Lopez-Paz (2018). The original mixup paper, with the soft-label loss formulation that the broken reimplementations skip.
- **"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"** — Yun et al. (2019). The area-ratio label-mixing rule that the wrong-$\lambda$ bug violates.
- **"RandAugment: Practical Automated Data Augmentation with a Reduced Search Space"** — Cubuk, Zoph, Shlens, Le (2020). Establishes augmentation magnitude as a tunable hyperparameter with a clear, model-size-dependent optimum — the empirical basis for the U-shaped-dial claim.
- **"SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"** — Park et al. (2019). The standard speech augmentation and the too-strong-masking underfitting failure mode.
- **Albumentations documentation** — the library reference for transforming images, bounding boxes, masks, and keypoints in lockstep; the canonical fix for the un-transformed-target bug.
- **torchvision transforms documentation** — the API reference for the classification transform stack, including the v2 transforms that support joint image/target transforms.
- **[A taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs)** — the series' master decision tree; augmentation is a data bug that masquerades as optimization or model code.
- **[The training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook)** — the capstone that folds the augmentation checks into the full symptom → suspect → test → fix workflow.
- **[The overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test)** — the base sanity check whose aug-off/aug-on variant is the cleanest way to convict augmentation.
- **[Augmentation debugging for vision](/blog/machine-learning/debugging-training/augmentation-debugging-for-vision)** — the deeper dive on coordinate-format swaps, mask interpolation, and test-time-augmentation skew for detection and segmentation.

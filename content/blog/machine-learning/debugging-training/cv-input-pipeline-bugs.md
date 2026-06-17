---
title: "Computer Vision Input-Pipeline Bugs: BGR, Normalization, and Resize"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Find the vision preprocessing bugs that quietly cost two to four points of accuracy without raising an error, by learning to display and assert the exact tensor your model receives."
tags:
  [
    "debugging",
    "model-training",
    "computer-vision",
    "preprocessing",
    "normalization",
    "pytorch",
    "opencv",
    "torchvision",
    "finetuning",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/cv-input-pipeline-bugs-1.png"
---

Here is a run that should have worked. You take a ResNet-50 pretrained on ImageNet, swap the final layer for your 37 pet breeds, and finetune. The loss falls, the validation accuracy climbs to a respectable 87%, and you ship it. Three weeks later someone benchmarks a competitor's near-identical model and gets 91% on the same data with the same backbone. You did not make a modeling mistake. You did not pick a worse optimizer. You lost four points of accuracy before the first convolution ran, because the function that loaded your images handed the network blue where it expected red. Your training images came off disk through `cv2.imread`, which returns channels in **BGR** order, and you fed them straight into a backbone whose every filter was learned on **RGB**. Nothing crashed. Nothing warned. The colors were "an image," the model trained on "an image," and the four points evaporated silently into the channel swap.

This is the defining property of computer-vision input bugs: **they almost never raise an exception.** A shape mismatch crashes loudly and you fix it in a minute. A `NaN` poisons the loss and screams. But feed a model BGR instead of RGB, normalize with the wrong mean and standard deviation, resize with a different interpolation at serving time than at training time, or forget to divide `uint8` pixels by 255 — and the pipeline runs to completion and reports numbers. The numbers are just measurements of a different experiment than the one you meant to run. In the six-places framing this series is built on — a bug hides in **data, optimization, model code, numerics, systems, or evaluation** — every bug in this post lives in *data*, in the preprocessing layer specifically, and the entire layer is invisible to your loss curve. Figure 1 is the map: between a JPEG on disk and the normalized tensor that reaches the backbone, there are six transform stages, and a defect at any one of them rewrites the input without leaving a trace.

![Stack diagram showing the six transform stages a computer-vision batch passes through, from decode where cv2 and PIL disagree, to channel order BGR versus RGB, to resize with bilinear and antialias choices, to the uint8 to float scaling, to normalization with a mean and standard deviation, to the NCHW versus NHWC layout, ending at the backbone seeing the wrong input distribution](/imgs/blogs/cv-input-pipeline-bugs-1.png)

By the end of this post you will be able to take any vision model that "trained fine but underperforms" and decide in a few minutes whether the input pipeline is the culprit, then localize *which* of seven specific bugs it is. The single discipline that makes this fast is one you can build today: **display the exact tensor the model receives.** Not the file on disk, not the PIL image you think you loaded — the literal `[C, H, W]` float tensor at the moment it enters the network, denormalized back into something your eyes can judge, with its `min`, `max`, `mean`, `std`, `dtype`, and `shape` printed beside it. We will write that reusable "model's-eye view" function once and use it to catch every bug below. The spine, as always, is the two master tools of the series: **make-it-fail-small** (one image, one batch, one forward pass you can inspect) and **read the instruments** (here the instrument is the tensor itself). This post is E1, the start of the computer-vision track. It instantiates the [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for the vision-data branch, shares its print-the-batch discipline with [the input pipeline is lying to you](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you), leans on [initialization and normalization bugs](/blog/machine-learning/debugging-training/initialization-and-normalization-bugs) for the distribution-shift mechanism, and feeds the capstone [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook).

## 1. The symptom: a model that trains fine and underperforms forever

Let me make the running example concrete, because we are going to debug it all the way down. You are finetuning a pretrained ImageNet classifier — call it the **Oxford-IIIT Pet** finetune, 37 breeds, a ResNet-50 backbone, the head reinitialized. The training loop is textbook. The optimizer is `AdamW` at a sane learning rate. The loss falls smoothly from `3.6` to `0.4`. Validation accuracy plateaus at `87%` and will not move no matter how long you train or how you tune the head. There is no crash, no `NaN`, no obvious overfitting gap. By every cheap sanity check the run is healthy.

That last sentence is the trap. Input-pipeline bugs pass the cheap checks — loss goes down, the model generalizes *somewhat* — and fail the expensive one, which is **the model is a few points worse than it should be, permanently, for a reason that lives entirely upstream of the gradient.** The reason a pretrained backbone is so sensitive here is worth stating up front because it is the scientific spine of the whole post: **a pretrained network has memorized a specific input distribution.** Its first convolution learned filters tuned to the exact channel order, value scale, and spatial statistics of its training data. Hand it a subtly different distribution and you are doing a tiny, uncontrolled domain shift at the input — not enough to break learning, just enough to bleed accuracy. The model adapts *around* the corruption during finetuning, which is precisely why the loss still falls; it just adapts to a worse starting point and lands a few points short.

Here are the four signatures you will actually see, and where each one points:

| Signature | What you observe | First suspect | Why |
| --- | --- | --- | --- |
| Trains fine, -2 to -4 pts | Smooth loss, accuracy a few points low forever | Channel order or normalization | Pretrained filters see a shifted distribution |
| Eval worse than train | Train metric good, val/test soft | Resize or augment train/serve skew | Eval pixels differ from train pixels |
| Loss enormous at step 1 | First loss is `1e3+`, then settles | `uint8` not scaled to `[0,1]` | Inputs are 0–255, activations explode |
| A subset is sideways | Some images rotated 90° | EXIF orientation dropped | Decoder ignored the orientation tag |

Notice the common thread: in every row the *optimizer is doing its job*. Gradients flow, weights update, the loss falls. That is exactly why these bugs are corrosive — they survive the one check everyone runs (does the loss go down?) and they hide from the loss curve entirely, because the loss curve is a thousand-step summary and the bug is in the *content* of every batch, not in the optimization dynamics. The disciplined move is to stop staring at the loss curve and **go look at the tensor.** Before any fix, internalize the bisection: a forward pass is a function of the parameters and the input tensor. If the parameters are a known-good pretrained checkpoint, then a model that underperforms and a model that performs *differ only in the input tensor.* So we hold the weights fixed and interrogate the tensor directly. Everything below is a specific question to ask the tensor and the specific bug a wrong answer reveals.

## 2. Bug one: BGR vs RGB, and why a channel swap costs real accuracy

The single most common vision input bug is also the easiest to write. OpenCV — the most popular image library in production pipelines — loads images in **BGR** channel order. This is a historical artifact from the early Windows bitmap format that OpenCV inherited in 2000 and never changed for backward compatibility. Almost everything else in the deep-learning ecosystem uses **RGB**: PIL/Pillow, `torchvision`, `matplotlib`, TensorFlow's image ops, and crucially the data on which every ImageNet backbone was pretrained. So the moment your pipeline mixes OpenCV decoding with a pretrained model or a non-OpenCV training pipeline, you have a channel swap, and the swap is invisible because a tensor of shape `[3, H, W]` is a valid image whether the first channel is red or blue.

### 2.1 The science: why swapping channels into a pretrained filter degrades it

A convolution's first layer is a set of learned linear filters over the channel axis. For an RGB input $x \in \mathbb{R}^{3 \times H \times W}$, the response of one output channel at a spatial location is

$$
y = \sum_{c \in \{R,G,B\}} \sum_{(i,j)} W_{c,i,j}\, x_{c,i,j} + b,
$$

where $W$ is the learned kernel. During ImageNet pretraining, the optimizer chose the weights $W_R, W_G, W_B$ to detect specific color-spatial patterns — an edge that is red on one side and green on the other, a blue-sky gradient, a skin-tone blob. These weights are *not* symmetric across channels; the network genuinely learned that channel 0 means red. When you feed BGR, you are computing

$$
y_{\text{bug}} = W_R\, x_B + W_G\, x_G + W_B\, x_R + b,
$$

i.e. you apply the red-detecting weights to the blue pixels and vice versa. The green channel survives (it is the middle in both orders), so roughly one-third of the filter is correct and two-thirds is scrambled. The result is not random — the network still extracts *something*, because natural-image channels are correlated (a bright pixel tends to be bright in all three channels) — which is exactly why the bug does not destroy training. It just feeds every layer a systematically distorted feature map, and the damage compounds with depth.

How big is the hit? In practice, BGR-into-an-RGB-backbone costs on the order of **two to four points of top-1 accuracy** depending on the dataset and how color-dependent the task is. A useful way to see why it is not catastrophic: the red and blue channels of natural images have a correlation coefficient typically around `0.8`–`0.9`, so swapping them is a *partial* corruption, not a total one. A task where color barely matters (texture classification) loses less; a task where color is the signal (flower or bird species, where red-versus-blue plumage is diagnostic) loses more. The honest framing: treat "two to four points" as a defensible order of magnitude, not a universal constant — measure it on your own data with the diagnostic below. Figure 2 shows the before-and-after on the pet finetune: the BGR run lands at `73.1%`, the fix recovers `76.3%`.

![Before and after diagram contrasting a pipeline that feeds OpenCV BGR straight into an ImageNet backbone, where red and blue are swapped at every first-layer filter and top-1 accuracy is 73.1 percent, a 3.2 point drop with no crash, against a pipeline that converts to RGB so the filters see the colors they learned and top-1 accuracy returns to 76.3 percent](/imgs/blogs/cv-input-pipeline-bugs-2.png)

### 2.2 The diagnostic: show the channels, do not trust the code

You cannot confirm a channel order by reading the loader code, because the bug is usually a missing line, and a missing line is invisible. The only reliable confirmation is to **display the image the way the model sees it** and look at whether the colors are right. Decode the same file two ways and compare:

```python
import cv2
from PIL import Image
import numpy as np

path = "samples/golden_retriever.jpg"

# OpenCV: returns BGR by default
img_cv = cv2.imread(path)                 # shape (H, W, 3), order B, G, R
# PIL/torchvision: returns RGB
img_pil = np.array(Image.open(path).convert("RGB"))  # order R, G, B

# Compare the first pixel of each: if the pipeline is RGB-correct these match
print("cv2 (BGR) pixel [0,0]:", img_cv[0, 0])      # e.g. [ 34  98 187]  -> B,G,R
print("PIL (RGB) pixel [0,0]:", img_pil[0, 0])     # e.g. [187  98  34]  -> R,G,B
print("cv2 reversed == PIL  :", np.array_equal(img_cv[..., ::-1], img_pil))
# True confirms cv2 is exactly PIL with channels reversed -> you are loading BGR
```

If the last line prints `True`, your OpenCV path is BGR and any pretrained model or RGB-trained pipeline downstream is getting swapped channels. The fix is one line — but the *placement* matters, so be explicit:

```python
# Fix: convert to RGB immediately after decoding, before any other transform
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)   # now order R, G, B
# Equivalent, slightly faster, no copy of metadata:
# img_rgb = img_cv[..., ::-1].copy()   # reverse channel axis; .copy() avoids negative strides
```

A subtle trap in the second form: `img_cv[..., ::-1]` returns a *view* with negative strides, and `torch.from_numpy` will refuse it or silently behave oddly, so the `.copy()` is not optional if you go that route. Use `cv2.cvtColor` unless profiling says otherwise.

### 2.3 The before-after evidence

Here is the measured result on the pet finetune, identical in every respect except the one `cvtColor` line. The point of the table is not the absolute numbers — it is the **delta from a single line** and the fact that the loss curve was indistinguishable between the two runs.

| Run | Channel order | Final train loss | Val top-1 | Notes |
| --- | --- | --- | --- | --- |
| Buggy | BGR into RGB backbone | 0.41 | 73.1% | Loss curve looked perfect |
| Fixed | RGB (cvtColor applied) | 0.40 | 76.3% | +3.2 pts, same everything else |

The way to *confirm* this honestly — not just trust the number — is the A/B you just saw: train two runs that differ only in the channel conversion, fix every seed, and read the validation accuracy. If the only code difference is one `cvtColor` and the accuracy moves by points, the channel order was the bug. If it moves by tenths, color was not your dominant signal and you should keep bisecting.

#### Worked example: putting a number on the channel swap

You finetune the ResNet-50 pet classifier and observe val top-1 stuck at `73.1%` while a colleague reports `76%+` on the same split. You suspect the input. You run the decode comparison above and `cv2 reversed == PIL` prints `True` — confirmed BGR. You denormalize one training batch and display it: the golden retrievers look faintly *teal*, the grass looks purple-ish. That is the fingerprint of a red–blue swap: warm tones (skin, fur, soil) drift cold, cool tones (sky, foliage shadows) drift warm. You add `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` right after `imread`, retrain with the same seed, and val top-1 lands at `76.3%`. The delta is `+3.2` points, recovered for one line and roughly two minutes of diagnosis. At a cluster cost of `\$2.40` per GPU-hour and an 8-hour finetune, you also avoided a `\$19` re-run by catching it before the long job — but the real cost you avoided was shipping a model four points below its ceiling and never knowing why.

## 3. Bug two: normalization mismatch, and the math of a shifted activation

If channel order is the most common vision bug, normalization is the most *insidious*, because there are at least four distinct ways to get it wrong and they all produce a model that trains. Every ImageNet-pretrained backbone in `torchvision` expects inputs that have been (a) scaled from `[0, 255]` to `[0, 1]`, then (b) normalized per-channel with the ImageNet statistics:

```python
# The canonical ImageNet normalization that pretrained torchvision models expect
mean = [0.485, 0.456, 0.406]   # per-channel mean over ImageNet, in [0,1] scale
std  = [0.229, 0.224, 0.225]   # per-channel std over ImageNet, in [0,1] scale
# normalized = (pixel/255 - mean) / std
```

The four ways to break this, each common in real codebases:

1. **Apply ImageNet `mean`/`std` to `[0, 255]` data.** You forgot the `/255` scaling, so you subtract `0.485` from pixels that range to `255`. The result is essentially unnormalized — every value is a giant positive number — and the first-layer activations explode.
2. **No normalization at all.** You feed `[0, 1]` (or `[0, 255]`) pixels to a backbone that expects mean-subtracted, std-divided inputs. The data is off-center and the wrong scale.
3. **Wrong statistics.** You use a different dataset's mean/std (or you computed them on a subset, or you transposed `mean` and `std`), so the centering is slightly off.
4. **Train statistics ≠ eval statistics.** You normalize training with one set of numbers and evaluation with another, so train and eval are in different coordinate systems.

### 3.1 The science: normalization is an affine map, and a wrong one is a constant bias plus gain

Here is why a normalization mismatch *necessarily* shifts every downstream activation — it is not a vague "the model gets confused," it is a precise linear-algebra consequence. Normalization is an affine transform of the input,

$$
\hat{x} = \frac{x - \mu}{\sigma},
$$

applied per channel. The first convolution is linear in $\hat{x}$:

$$
z = W \hat{x} + b.
$$

Suppose the *correct* normalization uses $(\mu^\star, \sigma^\star)$ but you used $(\mu, \sigma)$. Write the correct normalized input as $\hat{x}^\star$ and the buggy one as $\hat{x}$. They are related by

$$
\hat{x} = \frac{\sigma^\star}{\sigma}\,\hat{x}^\star + \frac{\mu^\star - \mu}{\sigma}.
$$

Substitute into the conv:

$$
z = W\!\left(\frac{\sigma^\star}{\sigma}\,\hat{x}^\star + \frac{\mu^\star - \mu}{\sigma}\right) + b
= \underbrace{\frac{\sigma^\star}{\sigma}\,W\hat{x}^\star}_{\text{scaled signal}} + \underbrace{W\frac{\mu^\star-\mu}{\sigma}}_{\text{constant bias}} + b.
$$

Two things happen, and both are damaging. First, a **constant per-channel bias** $W\frac{\mu^\star - \mu}{\sigma}$ is added to *every* spatial location of *every* feature map — a uniform shift the pretrained bias $b$ was never meant to cancel. Second, the whole signal is **scaled** by $\frac{\sigma^\star}{\sigma}$, so if your $\sigma$ is too small, activations are amplified; too large, they are damped. The pretrained network's later layers, its BatchNorm running statistics, and its nonlinearities (ReLU's knee at zero) were all calibrated for $z$ in a particular range. Shift and scale $z$ and you push activations off the ReLU knee — units that should fire sometimes now fire always or never — and you put BatchNorm's stored running mean and variance out of distribution. Figure 3 traces this: a wrong normalization enters conv1 as a constant bias plus a gain change and propagates as a distribution shift to every activation. This is the same mechanism that [initialization and normalization bugs](/blog/machine-learning/debugging-training/initialization-and-normalization-bugs) covers from the optimization side; here it originates at the input.

![Dataflow graph showing a pixel entering two normalization paths, a correct path that subtracts the right mean and divides by the right standard deviation feeding conv1, and a wrong path applying ImageNet mean on a 0 to 255 scale, where conv1 being linear in the normalized input means the mismatch adds a constant per-channel bias plus a gain change that shifts every activation off the ReLU knee](/imgs/blogs/cv-input-pipeline-bugs-3.png)

The catastrophic version is case 1, where you skip `/255` entirely. Then $\mu^\star \approx 0.45$ but you feed pixels up to `255`, so the effective shift is enormous and the gain is off by `255`. The first-layer pre-activations are roughly `255×` too large; with ReLU and a few layers, the logits blow up and the first loss is a huge number like `1e3` before the optimizer claws it back. That is the "loss enormous at step 1" signature from the table — and it is the *friendliest* version of the bug because at least it is loud.

### 3.2 The diagnostic: print the four numbers that define the distribution

You do not need to read the normalization code. Pull one batch as the model will receive it and print its statistics. A correctly ImageNet-normalized batch has a specific fingerprint: per-channel mean near `0`, std near `1`, and a range roughly `[-2.5, 2.8]` (because the most extreme pixels, `0` and `255/255=1`, map to `(0 - 0.485)/0.229 ≈ -2.12` and `(1 - 0.406)/0.225 ≈ 2.64`).

```python
import torch

def batch_stats(x: torch.Tensor, name: str = "batch"):
    """Print the fingerprint of a preprocessed batch. x is [N, C, H, W]."""
    assert x.dim() == 4, f"expected [N,C,H,W], got {tuple(x.shape)}"
    print(f"--- {name} ---")
    print("shape :", tuple(x.shape))
    print("dtype :", x.dtype)
    print("min   :", x.min().item())
    print("max   :", x.max().item())
    print("mean  :", x.mean().item())
    print("std   :", x.std().item())
    # per-channel means/stds catch a transposed or wrong-channel normalization
    print("per-ch mean:", x.mean(dim=(0, 2, 3)).tolist())
    print("per-ch std :", x.std(dim=(0, 2, 3)).tolist())

x, y = next(iter(train_loader))
batch_stats(x, "train batch")
# Healthy ImageNet-normalized output:
#   min ~ -2.1, max ~ 2.6, mean ~ 0.0, std ~ 1.0, per-ch std ~ [1,1,1]
# Forgot /255 with ImageNet stats:
#   min ~ -2.1, max ~ 1130, mean ~ 550, std ~ 70   <- max way over 3, mean huge
# No normalization (raw [0,1]):
#   min 0.0, max 1.0, mean ~ 0.45, std ~ 0.25       <- mean not near 0, max is 1.0
```

The decision tree is immediate. If `max` is far above `3`, you forgot `/255`. If `min` is `0.0` and `mean` is around `0.45`, you did not normalize at all. If the per-channel stds are not all near `1`, your `std` vector is wrong or transposed. This single function is the most valuable five minutes you will spend on a vision run, and it belongs in a test that runs before every training job.

### 3.3 The fix and the before-after evidence

The fix is to use the exact transform the checkpoint documents. In modern `torchvision`, the pretrained weights *carry their own preprocessing*, which eliminates the guessing:

```python
from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)

# The weights object knows the exact preprocessing it was trained with:
preprocess = weights.transforms()
print(preprocess)
# ImageClassification(
#   crop_size=[224], resize_size=[232],
#   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
#   interpolation=InterpolationMode.BILINEAR, antialias=True )
```

`weights.transforms()` returns the *canonical* preprocessing pipeline for that checkpoint — the resize size, crop size, interpolation, antialias flag, and normalization. Using it removes four classes of bug at once. The before-after on a different example, an unnormalized run:

| Run | Normalization | First-step loss | Val top-1 | Notes |
| --- | --- | --- | --- | --- |
| Buggy | none (raw `[0,1]`) | 4.2 | 81.0% | Trains, but off-distribution |
| Buggy | ImageNet stats on `[0,255]` | 1140 | diverges/NaN | The loud version |
| Fixed | `weights.transforms()` | 3.6 | 87.0% | +6 pts over no-norm |

The honest way to confirm normalization is the bug: print the batch statistics (above), fix to `weights.transforms()`, and re-measure. If the per-channel mean was far from `0` before and near `0` after, and accuracy jumped, normalization was the culprit.

### 3.4 ImageNet stats vs your own stats: a real decision, not a default

A question that trips up even experienced practitioners: should you normalize with the **ImageNet** mean/std, or with statistics computed on **your own dataset**? The answer follows directly from the science above, and getting it wrong is a subtle version of the normalization bug.

If you are **finetuning a pretrained backbone**, you should normalize with the statistics the backbone was *pretrained* with — the ImageNet stats — *not* your dataset's. The reason is exactly the affine-shift math: the pretrained conv1 weights and BatchNorm running statistics were calibrated for inputs centered and scaled by the ImageNet numbers. Re-center your data with *your* mean and you reintroduce the constant bias $W\frac{\mu^\star - \mu}{\sigma}$ that the pretrained network is not expecting. Counterintuitively, "more correct for my data" statistics make a *pretrained* finetune *worse*, because correctness here means "matches what the frozen-ish early layers expect," not "matches my data's true distribution." This is one of the few places where the textbook advice ("normalize to your data") is actively wrong, and it is wrong for a precise, derivable reason.

If you are **training from scratch**, the opposite holds: compute your own dataset's per-channel mean and std, because there is no pretrained distribution to match and you want inputs centered at `0` with unit variance for healthy initial gradients (the [initialization and normalization bugs](/blog/machine-learning/debugging-training/initialization-and-normalization-bugs) story). Computing them is cheap and worth doing once:

```python
import torch
from torch.utils.data import DataLoader

def compute_channel_stats(dataset, batch_size=256, num_workers=8):
    """Per-channel mean/std over a dataset of [C,H,W] tensors in [0,1]."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    n_pixels = 0
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    for x, _ in loader:                       # x is [N, C, H, W] in [0,1]
        # sum over batch, height, width; keep the channel axis
        channel_sum += x.sum(dim=(0, 2, 3))
        channel_sq_sum += (x ** 2).sum(dim=(0, 2, 3))
        n_pixels += x.numel() // x.shape[1]   # pixels per channel
    mean = channel_sum / n_pixels
    std = (channel_sq_sum / n_pixels - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()

# mean, std = compute_channel_stats(train_dataset)
# print("dataset mean:", mean, "std:", std)
# medical/satellite/grayscale data often differs a LOT from ImageNet's [0.485,...]
```

A common trap inside *this* helper: computing the statistics on **augmented** images. If your dataset applies random brightness/contrast jitter, the stats you compute drift with the augmentation RNG and are not the true data statistics — compute them on the *clean* resized images, before color augmentation. And compute them on the **training split only**; computing dataset stats over train+val is a mild form of [data leakage](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer), letting validation pixels influence the training normalization.

The decision rule in one line: **pretrained finetune → use the checkpoint's stats (ImageNet); from scratch → compute your own on the clean training split.** Mixing them up is a normalization bug that the batch-statistics readout will *not* catch (the data looks normalized, just to the wrong center), so it is one you have to reason about rather than measure.

## 4. Bug three: resize and interpolation, the train/serve gap nobody profiles

Resize is the bug that hides in plain sight because *everybody* resizes and almost nobody checks that they resize the *same way* at training and serving. There are three independent axes to get wrong: the **interpolation method** (nearest, bilinear, bicubic, area), the **antialias flag** (on or off), and the **aspect-ratio handling** (squash, pad, or center-crop). Each one changes the pixel statistics the backbone was tuned for.

### 4.1 The science: interpolation is a low-pass filter, and the wrong one shifts the spectrum

Downsizing an image is fundamentally a resampling operation, and resampling has a correct answer dictated by signal processing. When you shrink an image, you must first **low-pass filter** it to remove spatial frequencies above the new Nyquist limit, or those frequencies fold back as **aliasing** — moiré patterns, jagged edges, false high-frequency texture. **Antialiased** resize (bilinear or bicubic with `antialias=True`) applies that low-pass filter; **nearest-neighbor** and **non-antialiased** downsampling skip it and alias.

Why does this matter for accuracy? A pretrained backbone learned its filters on images downsized a *particular* way — for the modern `torchvision` recipes, antialiased bilinear. Its early filters are tuned to the smoothness and edge statistics that antialiasing produces. Serve it nearest-neighbor-downsized images and you hand it aliased edges and sharper high-frequency content than it ever saw in training, which is a covariate shift at exactly the layer most sensitive to local texture. The effect is typically smaller than the channel or normalization bugs — on the order of **one to two points** — but it is real, it is sneaky, and it shows up specifically as *eval worse than train* when the train and serve resizes differ.

A second, dumber resize bug is **aspect-ratio distortion**. If you `resize(224, 224)` a `640×480` image directly, you squash it horizontally — every face gets wider, every circle becomes an ellipse. The model was usually trained on resize-shorter-side-then-center-crop, which preserves aspect ratio. Squashing introduces a geometric distortion the backbone never saw. The historical `torchvision` default `Resize((H, W))` with a tuple squashes; `Resize(size)` with a single int resizes the shorter side and preserves ratio. Mixing the two between train and serve is a classic skew.

### 4.2 The antialias default change you must know about

There is a specific, infamous trap here: **`torchvision` changed the default behavior of `antialias`.** For years, `transforms.Resize` on a PIL image antialiased, but on a tensor it did *not*, so a pipeline that resized tensors silently produced aliased images different from the PIL path. As of `torchvision` 0.17 the default for the recommended transforms is `antialias=True`, and the older tensor path that defaulted to `False` was deprecated. The practical consequence: a model trained on one `torchvision` version (PIL resize, antialiased) and served on another (tensor resize, not antialiased) gets a train/serve mismatch *purely from a library upgrade*, with no code change on your part. Pin the flag explicitly and never rely on the default:

```python
import torchvision.transforms.v2 as T
from torchvision.transforms import InterpolationMode

# Be explicit about EVERY resize parameter so a library upgrade cannot change it
resize = T.Resize(
    size=232,                               # shorter side -> preserves aspect ratio
    interpolation=InterpolationMode.BILINEAR,
    antialias=True,                         # NEVER rely on the default
)
crop = T.CenterCrop(224)                     # then center-crop to the square the model wants
```

Figure 6 makes the resize skew concrete as a before-and-after: a pipeline that trains on antialiased bilinear and serves on non-antialiased nearest produces aliased edges and loses accuracy, while matching the two sides recovers it.

![Before and after diagram contrasting a mismatched resize where training uses bilinear interpolation with antialias on while serving uses nearest with antialias off, producing aliased edges and a 1.6 point top-1 drop, against a matched resize where both training and serving use bilinear with antialias on so the pixel statistics agree and accuracy is recovered](/imgs/blogs/cv-input-pipeline-bugs-6.png)

To make the spectral argument precise — because "aliasing" is easy to wave at and hard to pin down — consider what nearest-neighbor downsampling actually does. To shrink an image by a factor $s$, nearest-neighbor simply *keeps every $s$-th pixel and drops the rest*. By the Nyquist–Shannon sampling theorem, a signal sampled at spacing $s$ can only faithfully represent spatial frequencies below $\frac{1}{2s}$ cycles per pixel; any content above that limit does not vanish, it **folds back** (aliases) into lower frequencies, appearing as false texture. A `512×512` image downsized to `224×224` by nearest-neighbor discards roughly `81%` of the pixels with no filtering, so every high-frequency edge and texture in the original gets aliased into a spurious low-frequency pattern. Antialiased bilinear instead convolves with a small smoothing kernel whose width scales with $s$ *before* sampling, attenuating the above-Nyquist frequencies so they do not fold. The pretrained backbone learned its early filters on the *antialiased* frequency content; feed it the *aliased* version and the texture statistics in the first few layers are wrong in a way the network never saw, which is precisely why the small-but-real accuracy drop concentrates in fine-grained, texture-dependent classes. The order-of-magnitude is one to two points; measure it with the `preprocess_diff` below rather than trusting a constant.

There is a related, even quieter aspect-ratio bug. Two common "resize to square" recipes give *different* pixels for a non-square source: (a) resize the shorter side to `S` then center-crop `S×S` (aspect-preserving, the standard ImageNet recipe), versus (b) resize directly to `S×S` (squashes the aspect ratio). If training uses (a) and serving uses (b), a `640×480` image is center-cropped in training but horizontally compressed in serving — every object is `1.33×` too wide at serve time. The backbone, trained on undistorted aspect ratios, sees stretched objects it never learned. The tell is the same train/serve gap, and the diagnostic is again `preprocess_diff`, which will report a large `max abs diff` driven by the geometric mismatch rather than the interpolation kernel.

Figure 4 is the master lookup for this whole post — symptom to cause to check to fix — and resize-skew is one row of it; let me put the full table on screen before we keep going.

![Matrix diagram mapping five computer-vision input symptoms to a likely cause, the cheapest one-line check, and the fix direction, covering off colors mapped to BGR versus RGB checked by showing the denormalized image, an all gray batch mapped to wrong mean over standard deviation scale checked by printing min max and mean, a soft or wrong eval mapped to resize interpolation skew checked by diffing train versus serve, sideways images mapped to EXIF not applied checked with exif transpose, and a huge step one loss mapped to uint8 not scaled checked by inspecting dtype and range](/imgs/blogs/cv-input-pipeline-bugs-4.png)

### 4.3 The diagnostic: diff the train and serve pixels directly

The only reliable test for a train/serve resize mismatch is to run the *same source image* through both pipelines and measure the pixel difference. If they disagree, your train and serve preprocessing are not the same function:

```python
import torch
import numpy as np

def preprocess_diff(img_pil, train_tf, serve_tf):
    """Run one image through both pipelines; report how different the tensors are."""
    a = train_tf(img_pil)           # [C,H,W] float
    b = serve_tf(img_pil)
    if a.shape != b.shape:
        print(f"SHAPE MISMATCH: train {tuple(a.shape)} vs serve {tuple(b.shape)}")
        return
    diff = (a - b).abs()
    print("max abs diff   :", diff.max().item())
    print("mean abs diff  :", diff.mean().item())
    # fraction of pixels differing by more than a small epsilon
    print("frac > 1e-3    :", (diff > 1e-3).float().mean().item())

# If max abs diff is ~0, the pipelines agree. If it is 0.1+ on normalized data,
# your train and serve preprocessing differ — usually interpolation or antialias.
preprocess_diff(Image.open("samples/cat.jpg").convert("RGB"), train_tf, serve_tf)
# Agreeing pipelines:   max abs diff ~ 1e-6
# Antialias mismatch:   max abs diff ~ 0.15, frac > 1e-3 ~ 0.4  (40% of pixels differ)
```

A `max abs diff` of `1e-6` means the two pipelines are numerically the same function and resize is not your skew. A diff of `0.1`+ on normalized data, with a large fraction of pixels affected, means train and serve genuinely produce different inputs — go find the interpolation, antialias, or resize-mode difference between them.

#### Worked example: the silent antialias regression

A team trains a segmentation-free classifier on `torchvision` 0.15 (PIL resize, antialiased) and reaches `91.2%` val. Six months later they redeploy the serving image with `torchvision` 0.13 inside a different container that resizes the input as a *tensor* with the old `antialias=False` default. Offline eval still uses the training transform and reports `91.2%`; production accuracy on a held-out labeled stream measures `89.6%`. The `1.6`-point gap appears only in production, which screams *train/serve skew*. They run `preprocess_diff` on a sample: `max abs diff = 0.15`, `frac > 1e-3 = 0.41` — 41% of pixels differ. The fix is to pin `antialias=True` in the serving transform (and ideally pin the `torchvision` version). After the fix, `preprocess_diff` reads `max abs diff = 8e-7` and production accuracy returns to `91.1%`. The lesson: an interpolation skew is invisible to offline eval *by construction*, because offline eval uses the training transform — you only catch it by diffing the two pipelines or by measuring production directly. This is the resize half of [distribution shift, train vs the real world](/blog/machine-learning/debugging-training/distribution-shift-train-vs-the-real-world).

## 5. The master diagnostic: show the model's-eye view

Every bug so far has the same confirming test: **look at the exact tensor the model receives.** Not the file, not the intermediate PIL image — the literal float tensor at the input of the network, with its statistics printed. Let me give you the one reusable function that catches all of them, then walk through the five readouts it produces. This is the single highest-leverage tool in the post, and figure 5 shows the order to run the readouts: shape and dtype, then range, then denormalize, then display, then diff train against serve.

![Timeline diagram of the model's-eye view diagnostic showing five readouts run in order on a single batch, first print shape and dtype to check NCHW and uint8 versus float, then print min max and mean to check the range is zero to one or shifted, then denormalize to undo the mean and standard deviation, then display the tensor to judge colors and orientation, and finally diff the train pipeline against the serve pipeline to confirm the same pixels](/imgs/blogs/cv-input-pipeline-bugs-5.png)

The function takes a batch exactly as your `DataLoader` produces it and renders it back to something your eyes and a few prints can judge:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def model_eye_view(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, max_show=8):
    """Render the tensor the model actually receives. x is [N, C, H, W]."""
    assert x.dim() == 4, f"expected [N,C,H,W], got {tuple(x.shape)}"
    n, c, h, w = x.shape

    # 1. shape + dtype: catches NCHW/NHWC and uint8/float
    print(f"shape={tuple(x.shape)}  dtype={x.dtype}  device={x.device}")
    assert c in (1, 3), f"channel axis is {c}; is this NHWC by mistake?"

    # 2. value range: catches unscaled uint8 and missing normalization
    print(f"min={x.min():.3f}  max={x.max():.3f}  "
          f"mean={x.mean():.3f}  std={x.std():.3f}")

    # 3. denormalize back to [0,1] so the displayed colors are truthful
    x_denorm = (x.float().cpu() * std + mean).clamp(0, 1)

    # 4. display the grid: colors reveal BGR; orientation reveals EXIF
    k = min(n, max_show)
    fig, axes = plt.subplots(1, k, figsize=(2.2 * k, 2.4))
    axes = np.atleast_1d(axes)
    for i in range(k):
        img = x_denorm[i].permute(1, 2, 0).numpy()   # CHW -> HWC for display
        axes[i].imshow(img if c == 3 else img.squeeze(-1), cmap=None if c == 3 else "gray")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig("model_eye_view.png", dpi=120)
    print("saved -> model_eye_view.png  (open it and JUDGE the colors)")

x, y = next(iter(train_loader))
model_eye_view(x)
```

Read it top to bottom and you have a verdict on every bug in this post:

1. **`shape` and `dtype`** — if `dtype` is `uint8`, you never scaled to float and the `/255` is missing. If the channel axis is not `1` or `3` (the assert fires), you have an NHWC tensor where the model wants NCHW, which is the silent permute bug in section 7.
2. **`min`/`max`/`mean`/`std`** — the normalization fingerprint from section 3.2. `max` over `3` means unscaled `uint8`; `mean` near `0.45` means unnormalized.
3. **Denormalize** — multiplying back by `std` and adding `mean` returns the tensor to `[0, 1]` so the saved image shows *truthful* colors. Skip this and the displayed image is wrong because the normalized tensor has negative values matplotlib clips.
4. **Display** — open the PNG and look. Teal-tinted fur and purple grass mean BGR. A sideways subject means dropped EXIF. A squashed aspect ratio means a tuple-resize. Excessive blockiness means nearest-neighbor resize.
5. (Plus the **`preprocess_diff`** from section 4.3 to compare train vs serve.)

This is the entire diagnostic discipline. You do not reason about the loader code; you make it produce the tensor and you judge the tensor. Wire `model_eye_view` into the first iteration of training behind a flag and you will never ship a silent input bug again.

## 6. Bug four: uint8 vs float, and the missing /255

This one is fast but worth its own section because it is the most common cause of "loss is `1e3` at step 1." Images come off disk as `uint8` in `[0, 255]`. Models want `float` in `[0, 1]` (or, after normalization, roughly `[-2.5, 2.8]`). The `/255` scaling lives inside `torchvision`'s `ToTensor()` — which *also* permutes HWC to CHW *and* casts to float, three jobs in one call — so if you build a tensor by hand and forget the scaling, you feed `[0, 255]` floats to a model expecting `[0, 1]`.

### 6.1 The science: a 255× gain at the input

This is the same affine argument as normalization, with a brutally large gain. If the model expects $x/255 \in [0,1]$ and you feed $x \in [0,255]$, every input is `255×` too large. The first conv output $z = Wx + b$ is therefore `255×` too large (the bias `b` is unchanged, so it barely matters). Through a ReLU and a couple of layers, the logits are enormous, the softmax saturates, the cross-entropy returns a huge number, and the first gradient step is correspondingly violent. With normalization layered on top of unscaled data, you get the `1140` first-step loss from the table in section 3. The friendly part: it is *loud*, so you catch it at step 1 — unlike BGR, which whispers.

### 6.2 The diagnostic and fix: assert the contract at the boundary

The cleanest defense is an assertion that encodes the contract "by the time this tensor reaches the model, it is float in a sane range." Put it in the collate function or the first line of `forward` during debugging:

```python
def assert_input_contract(x: torch.Tensor):
    """Fail loudly if the input tensor violates the preprocessing contract."""
    assert x.dtype in (torch.float32, torch.float16, torch.bfloat16), \
        f"input is {x.dtype}; did you forget ToTensor()/scaling? (uint8 means no /255)"
    assert x.dim() == 4 and x.shape[1] in (1, 3), \
        f"expected [N,C,H,W] with C in (1,3); got {tuple(x.shape)} — NHWC?"
    lo, hi = x.min().item(), x.max().item()
    assert hi <= 6.0 and lo >= -6.0, \
        f"value range [{lo:.1f},{hi:.1f}] looks unscaled; expected ~[-2.5,2.8] normalized"

x, y = next(iter(train_loader))
assert_input_contract(x)   # crashes immediately if /255 or normalization is missing
```

The fix is to scale, and the idiomatic scale-plus-permute-plus-cast is `torchvision`'s `ToTensor()` (or, in the v2 API, `ToImage()` followed by `ToDtype(torch.float32, scale=True)`):

```python
import torchvision.transforms.v2 as T

# v2 idiom: ToImage gives a tensor Image; ToDtype with scale=True does the /255
to_float = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),   # scale=True divides uint8 by 255 -> [0,1]
])
# Hand-rolled equivalent, if you must:
# x = torch.from_numpy(np_img).permute(2, 0, 1).float().div_(255.0)
```

The `scale=True` flag is the entire bug in one keyword. Leave it `False` (or hand-roll without the `/255`) and you have the unscaled-input bug; the assert above catches it on the first batch.

### 6.3 The fp16 corner: an unscaled input overflows half precision

There is a nastier version of the unscaled-input bug that only appears under mixed precision, and it is worth a paragraph because it produces a `NaN` instead of a merely large loss. In fp16 the largest representable value is `65504`. If you feed unscaled `[0, 255]` inputs into a network running in `autocast`, the first few matrix multiplies can produce activations on the order of `255 × (number of accumulated terms)`, and for a wide first layer that product can exceed `65504` and overflow to `inf`. The `inf` then poisons the loss and you see a `NaN` at step 1 — except now you might blame the *mixed precision* instead of the *input scale*, because the symptom (a `NaN` under fp16) looks like a numerics bug. The bisection that disambiguates them: run the *same batch* in fp32. If the fp32 loss is merely huge (`1e3`) and finite while the fp16 loss is `NaN`, the root cause is an unscaled input that fp16 cannot hold, not a fundamental numerics problem — and the fix is the `/255`, not loss scaling. This is the kind of cross-layer confusion the [hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs) post untangles in general; the input-pipeline-specific tell is that the fp32 loss is large-but-finite and the batch range readout shows `max > 3`.

## 7. Bug five: NCHW vs NHWC, and the silent permute

OpenCV and PIL give you images in **HWC** order — height, width, channels, with the channel axis last. PyTorch convolutions want **CHW** — channels first. So somewhere you must permute `(H, W, C) → (C, H, W)`. Miss it and one of two things happens: either the shape mismatch crashes (the friendly case), or — far worse — the channel and spatial axes are *compatible by coincidence* and the model silently treats your `224` rows of height as `224` channels.

### 7.1 The science: when the permute bug does not crash

A conv layer with `in_channels=3` will crash if you hand it a tensor whose channel axis is `224`. So how does this bug ever *not* crash? Two ways, both real:

1. **A square image with a flexible first layer.** If you batch an HWC tensor as `[N, H, W, C] = [32, 224, 224, 3]` and a downstream operation interprets axis 1 as channels, it sees `224` channels. Most convs crash here — but **adaptive pooling, flattening, and some attention/patchify ops do not**, so in a Vision Transformer or a custom head the wrong axis can flow through and produce a valid-but-meaningless feature. Figure 6 walks the contrast.
2. **`channels_last` memory format confusion.** PyTorch supports a `channels_last` *memory format* that stores an NCHW *logical* tensor with NHWC *physical* strides for speed on tensor cores. This is a performance optimization and is *correct* — but developers confuse the memory format with the logical layout and "fix" a non-bug by adding a permute, which *creates* a real bug. Know the difference: `x.to(memory_format=torch.channels_last)` keeps the logical shape `[N, C, H, W]` and only changes strides; `x.permute(0, 3, 1, 2)` changes the logical shape.

![Grid diagram contrasting NHWC and NCHW layouts, showing a decoded array of height 224 width 224 channels 3 in HWC layout from cv2 or PIL while the conv wants CHW, then a no-permute path where the array is fed as-is so axis zero becomes height giving 224 fake channels and conv1 expecting 3 produces a shape error or junk, versus a permute path using permute 2 0 1 or ToTensor that produces a CHW layout with 3 real channels and a clean forward pass](/imgs/blogs/cv-input-pipeline-bugs-7.png)

### 7.2 The diagnostic: the assert that names the axis

The fix is the assert from section 6 plus one rule: **the channel axis of the model input is axis 1, and it is `1` or `3`.** If your batch has its small axis last, you have NHWC and need a permute:

```python
def to_nchw(x: torch.Tensor) -> torch.Tensor:
    """Coerce a batch to NCHW, detecting the common NHWC mistake."""
    assert x.dim() == 4, f"expected 4D, got {tuple(x.shape)}"
    n, a, b, c = x.shape
    if c in (1, 3) and a not in (1, 3):
        # last axis is the small one -> this is NHWC; permute to NCHW
        print(f"detected NHWC {tuple(x.shape)} -> permuting to NCHW")
        x = x.permute(0, 3, 1, 2).contiguous()
    assert x.shape[1] in (1, 3), \
        f"after coercion channel axis is {x.shape[1]}; check your layout"
    return x
```

The `.contiguous()` matters: a permuted tensor has non-standard strides, and some kernels are slow or error on non-contiguous inputs, so materialize it. The cleanest prevention is to never hand-roll the layout — use `ToTensor()`/`ToImage()`, which always returns CHW — and reserve the assert for catching cases where you do.

## 8. Bugs six and seven: EXIF orientation and grayscale-vs-three-channel

Two smaller bugs round out the surface, both responsible for a *subset* of images being wrong rather than all of them — which makes them harder to spot, because the aggregate accuracy moves only a little.

### 8.1 EXIF orientation: the sideways subset

Phone cameras store images in the sensor's native orientation and add an **EXIF orientation tag** that says "rotate this 90° clockwise when displaying." Image viewers honor the tag; many decoders do not. `cv2.imread` *ignores* EXIF entirely — it hands you the raw sensor pixels, possibly sideways. `PIL.Image.open` also does not auto-rotate unless you call `ImageOps.exif_transpose`. So a dataset shot on phones can contain a meaningful fraction of images that are rotated 90° or 180° relative to how a human labeled them, and the model is trained to recognize sideways cats.

The science here is mild but real: convolutions are not rotation-invariant, so a sideways subset is effectively a *different distribution* mixed into your data — a form of label-consistent but input-inconsistent noise that caps accuracy proportional to the affected fraction. The diagnostic is to apply EXIF transpose and look for images that *move*:

```python
from PIL import Image, ImageOps

def load_with_exif(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)   # apply the orientation tag, then drop it
    return img.convert("RGB")

# Audit: how many images in the set carry a non-trivial orientation tag?
from PIL.ExifTags import TAGS
def has_rotation(path):
    exif = Image.open(path).getexif()
    orient = exif.get(0x0112, 1)          # 0x0112 is the Orientation tag; 1 = normal
    return orient not in (1, None)

rotated = [p for p in paths if has_rotation(p)]
print(f"{len(rotated)}/{len(paths)} images carry a rotation tag")
```

If a non-trivial fraction carry a rotation tag and your loader is `cv2.imread` or a bare `Image.open`, those images are sideways to the model. The fix is `ImageOps.exif_transpose` on every load (and confirm your serving path does the same, or you get *another* train/serve skew).

### 8.2 Grayscale vs three-channel: the shape that sometimes breaks

Some images in a "color" dataset are actually grayscale — a single-channel PNG, a scanned document, a black-and-white photo. `PIL.Image.open` on a grayscale file returns a one-channel `L`-mode image; if you do not `convert("RGB")`, you hand the model a `[1, H, W]` tensor where it wants `[3, H, W]`, and depending on your collate, you either crash or — if you stack mismatched channel counts — get a ragged batch. The reverse also bites: a model trained on grayscale-as-3-identical-channels behaves differently from one trained on true RGB. The fix is an explicit, uniform channel policy:

```python
from PIL import Image

def load_rgb(path):
    # convert("RGB") makes grayscale -> 3 identical channels, RGBA -> drops alpha,
    # palette -> RGB; guarantees a 3-channel image regardless of source mode
    return Image.open(path).convert("RGB")
```

`convert("RGB")` is the single most defensive line in image loading: it normalizes grayscale, RGBA, palette, and CMYK sources to a uniform 3-channel RGB tensor. Make it the first thing every loader does and an entire class of "some images are the wrong shape" bugs disappears. The diagnostic is to audit the modes:

```python
from collections import Counter
modes = Counter(Image.open(p).mode for p in paths)
print(modes)   # Counter({'RGB': 7100, 'L': 340, 'RGBA': 60, 'P': 12})
# Any non-RGB count > 0 means convert("RGB") is load-bearing, not optional
```

### 7.3 Why the permute bug sometimes produces *plausible* loss

It is worth dwelling on the most dangerous variant of the NCHW/NHWC bug — the one that does not crash and does not even obviously underperform at first. Suppose a custom head flattens its input before a linear layer (common in older architectures and many tabular-image hybrids). If you feed `[N, H, W, C]` where the model expected `[N, C, H, W]`, the flattened vector has the *same total number of elements*, so the linear layer accepts it — it just multiplies the wrong weights against the wrong pixels. The loss will be high at first, but the optimizer can *learn around it*: it will fit a model that happens to work on the permuted layout, because from the optimizer's perspective the permutation is just a fixed, deterministic rearrangement of the input it can compensate for. You end up with a model that trains, generalizes *to the same permuted pipeline*, and then collapses the instant the serving code applies the *correct* layout. This is the worst case in the whole post: a bug that is self-consistent in training and only surfaces at deployment. The defense is the layout assert (section 7.2) run at the *boundary*, so the permutation can never silently become "load-bearing" in your weights.

## 9. The cv2/PIL decode disagreement: two libraries, two images

A subtle final source of train/serve skew: **`cv2` and PIL do not decode the same JPEG to the same pixels.** JPEG decoding involves an inverse discrete cosine transform, chroma upsampling, and color-space conversion, and the two libraries make slightly different implementation choices (different IDCT rounding, different chroma-upsampling filters). The differences are small — typically a few least-significant bits per pixel — but they are systematic, and on a borderline case they shift the input enough to change a prediction.

This rarely matters for accuracy on its own (a few LSBs is well within augmentation noise), but it becomes a real bug in two situations: when you **train with one decoder and serve with the other** (a small but consistent skew), and when you are **debugging a non-reproducible prediction** and cannot understand why the same file gives different logits in two environments. The diagnostic is the same `preprocess_diff` philosophy — decode the same file both ways and measure:

```python
import cv2
import numpy as np
from PIL import Image

def decoder_diff(path):
    cv = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.int16)
    pil = np.array(Image.open(path).convert("RGB")).astype(np.int16)
    diff = np.abs(cv - pil)
    print(f"max LSB diff: {diff.max()}  mean: {diff.mean():.3f}  "
          f"frac pixels differing: {(diff > 0).mean():.3f}")

decoder_diff("samples/cat.jpg")
# Typical: max LSB diff 2-3, mean ~0.3, frac differing ~0.5
# i.e. half the pixels differ by 1-3 out of 255 -- small but real and systematic
```

The practical rule that dissolves the whole problem: **decode with the same library at train and serve.** Pick one — most production stacks standardize on PIL/`torchvision` for training and must replicate it exactly in serving — and write it down. If your training uses `torchvision.io.read_image` or PIL and your serving uses `cv2`, you have a built-in skew that no amount of accuracy tuning will close.

## 10. Case studies: real signatures of input bugs

These are well-known, documented patterns. I give defensible magnitudes; treat specific numbers as approximate and measure your own.

**Caffe, BGR, and the ImageNet-mean legacy.** The original Caffe framework — used to train many of the early ImageNet models, including the canonical VGG and ResNet checkpoints — used OpenCV for I/O and therefore trained on **BGR** with a per-channel *mean subtraction in `[0, 255]` scale* (roughly `[104, 117, 123]` for B, G, R), *not* the `[0, 1]` ImageNet mean/std that `torchvision` uses. When those Caffe weights were ported to PyTorch, a generation of bugs followed: people fed RGB `[0, 1]`-normalized inputs to BGR-`[0, 255]`-mean-trained weights and lost accuracy without understanding why. The lesson that survives: **a checkpoint encodes its preprocessing, and porting weights means porting the exact channel order, scale, and statistics.** This is why `torchvision`'s modern `weights.transforms()` API exists — it ships the preprocessing *with* the weights so the contract cannot be lost.

**The antialias default and reproducibility across versions.** The `torchvision` `antialias` change is a documented, real source of accuracy drift across library versions. Models trained on the PIL path (antialiased) and evaluated on the old tensor path (not antialiased) showed measurable accuracy differences, and the maintainers eventually changed the default to `antialias=True` and warned loudly precisely because so many pipelines were silently mismatched. The takeaway: **pin interpolation and antialias explicitly**, and never let a `pip install --upgrade` silently change your preprocessing.

**ImageNet's own label and preprocessing quirks.** Beyond input bugs, ImageNet famously contains label noise and duplicates, and confident-learning work (cleanlab) found thousands of label errors across major test sets — a reminder that *even the reference distribution is imperfect*. For our purposes the relevant point is narrower: because so many backbones share ImageNet pretraining, they share its preprocessing expectations, so getting BGR/normalization/resize right against the *ImageNet* recipe specifically is what recovers accuracy on the widest range of finetunes. (Label noise itself is the subject of [garbage in, finding label noise](/blog/machine-learning/debugging-training/garbage-in-finding-label-noise); here we keep to the input transform.)

**The "works in the notebook, fails in production" decode skew.** A recurring production post-mortem: a model validates at `X%` in a training notebook that loads images with PIL and serves at `X-1` to `X-2%` behind an API that loads with `cv2` (for speed). The gap is the combined BGR-not-converted plus decoder-LSB skew, and it is invisible to offline eval because offline eval uses the *training* loader. The fix is always the same — make the serving preprocessing a byte-for-byte copy of the training preprocessing, and *test that equality* with `preprocess_diff`, not by eyeballing the code.

#### Worked example: bisecting a 4-point finetune gap end to end

You inherit a finetune that reports `73%` where the published baseline is `77%`. Four points, no crash. You bisect the input pipeline with the model's-eye view, one readout at a time:

1. **Shape and dtype**: `[32, 3, 224, 224]`, `float32`. Layout and scaling are fine — rules out the permute and the `uint8` bug.
2. **Range**: `min=-2.1`, `max=2.6`, `mean=0.01`, `std=0.99`. Textbook ImageNet normalization — rules out the normalization bug.
3. **Denormalize and display**: the dogs look *teal*, grass looks *magenta*. There it is. You run `cv2 reversed == PIL` and it prints `True` — the loader is `cv2.imread` with no `cvtColor`. **BGR confirmed.**
4. You add `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`, retrain with the same seed, and val top-1 goes `73.0% → 76.4%`. That recovers `3.4` of the `4` points.
5. The last `0.6` point: you run `preprocess_diff(train_tf, serve_tf)` and find `max abs diff = 0.12` — a resize antialias mismatch between the eval transform and the training transform. You pin `antialias=True`, and the final number lands at `77.0%`.

Total diagnosis time: under fifteen minutes, because at each step the model's-eye view gave a *yes or no* on one specific bug and you never had to guess. Two bugs, two fixes, four points, and a method you can rerun on the next model.

## 11. The unified diagnostic table and decision tree

Putting it together, here is the full lookup from symptom to the bug, with the confirming readout and the fix. This is the table to pin above your desk.

| Symptom | Likely bug | Confirming readout | Fix |
| --- | --- | --- | --- |
| Trains fine, -2 to -4 pts, off colors | BGR vs RGB | Denormalize + display: teal fur, magenta grass; `cv2 reversed == PIL` is `True` | `cv2.cvtColor(img, COLOR_BGR2RGB)` after decode |
| Loss `1e3`+ at step 1 | `uint8` not scaled | `batch_stats`: `max` ≫ 3, `dtype` `uint8` | `ToTensor()` / `ToDtype(scale=True)` |
| Mean far from 0, off-distribution | No / wrong normalization | `batch_stats`: `mean`≈0.45 or per-ch `std`≠1 | `weights.transforms()` |
| Eval worse than train, soft images | Resize/interp/antialias skew | `preprocess_diff` train vs serve: large `max abs diff` | Pin interpolation + `antialias=True` both sides |
| Crash or junk features, wrong axis | NCHW/NHWC permute | `model_eye_view` assert: channel axis ≠ 1 or 3 | `permute(0,3,1,2).contiguous()` / `ToTensor()` |
| A subset is sideways | EXIF orientation dropped | `has_rotation` audit: tags present, loader ignores them | `ImageOps.exif_transpose` on load |
| Some images crash/ragged batch | Grayscale vs 3-channel | mode `Counter`: `L`/`RGBA`/`P` present | `.convert("RGB")` on every load |

The corresponding decision tree starts from the model's-eye view and routes a *visible* defect to a specific bug — figure 7 — and that is the order you should run it: produce the tensor, judge it, follow the branch.

![Decision tree starting from the model's-eye view asking how the input looks wrong, branching into an image looks off branch covering color or shape and a numbers look off branch covering range or dtype, where the image branch leads to reds look blue meaning BGR versus RGB, sideways meaning EXIF not applied, and soft or aliased meaning a resize interpolation issue, while the numbers branch leads to max 255 meaning a missing divide by 255 and mean far from zero meaning a wrong mean and standard deviation](/imgs/blogs/cv-input-pipeline-bugs-8.png)

## 12. Albumentations, torchvision v2, and getting the order right

A practical note on *where* these transforms live, because the same bug recurs at the library boundary. The three dominant pipelines — `torchvision.transforms.v2`, `albumentations`, and hand-rolled OpenCV — each have a default channel order and a normalization convention, and mixing them is where skew creeps in.

- **`torchvision.transforms.v2`** works in RGB and CHW; `ToImage()` + `ToDtype(scale=True)` + `Normalize(mean, std)` is the canonical end. It assumes PIL or tensor input.
- **`albumentations`** works in **HWC** (because it is built on OpenCV/NumPy) and *expects you to pass `image=` as an HWC NumPy array*; it has its own `A.Normalize(mean, std, max_pixel_value=255.0)` that *does the `/255` for you* via `max_pixel_value`, and an `A.ToTensorV2()` at the very end that permutes to CHW. The classic albumentations bug is normalizing twice (once in `A.Normalize` and again in a `torchvision.Normalize`) or forgetting that `A.Normalize` already scaled by `255`.
- **OpenCV hand-rolled** gives you BGR HWC `uint8`; you owe the `cvtColor`, the `/255`, the normalize, and the permute yourself.

Here is a correct albumentations pipeline with the order and the scaling made explicit, so you can see exactly which step does the `/255` and which does the layout:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

train_tf = A.Compose([
    A.LongestMaxSize(max_size=232),                  # aspect-preserving resize
    A.CenterCrop(224, 224),
    # Normalize divides by max_pixel_value (255) THEN applies (x-mean)/std.
    # Do NOT also divide by 255 elsewhere or you scale twice.
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0),
    ToTensorV2(),                                    # HWC -> CHW, no extra scaling
])

# albumentations takes an HWC numpy array, and you MUST convert BGR->RGB first
img_bgr = cv2.imread("samples/cat.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # the BGR fix, still required
out = train_tf(image=img_rgb)["image"]               # final [3,224,224] float tensor
```

Three pitfalls hide in those nine lines, and each is one of our bugs wearing a library costume: the `cvtColor` is the **BGR fix** (albumentations does *not* convert for you); `max_pixel_value=255.0` is the **`/255` scaling** (omit it and you normalize `[0,255]` data with `[0,1]` stats — the catastrophic case); `ToTensorV2` is the **NHWC→NCHW permute** (without it you ship HWC). Get the order or any flag wrong and you reproduce, inside one "clean" pipeline, the exact bugs we spent the post diagnosing.

#### Worked example: the double-normalization bug

A team uses albumentations for augmentation and then, out of habit, appends a `torchvision.transforms.Normalize(mean, std)` "just to be safe." Now every pixel is normalized twice: albumentations maps `[0,255] → (x/255 - mean)/std ≈ [-2.1, 2.6]`, and the torchvision normalize then computes `(z - mean)/std` on the *already normalized* `z`, producing values around `(-2.1 - 0.485)/0.229 ≈ -11.3` to `(2.6 - 0.406)/0.225 ≈ 9.7`. The batch statistics read `mean ≈ -2.0`, `std ≈ 4.5` — wildly off the `mean≈0, std≈1` fingerprint. The model still trains (it adapts to the scale), but it starts from a `4–5×` over-normalized distribution and finetunes to `~3` points below baseline. The diagnostic catches it instantly: `batch_stats` shows `std ≈ 4.5` and `min ≈ -11`, nowhere near the expected `[-2.5, 2.8]`. The fix is to delete the second `Normalize`. The lesson generalizes: **normalization is not idempotent — applying it twice is a real bug**, and the batch-statistics fingerprint catches it in one print.

## 13. Stress-testing the pipeline: augmentation placement and the eval transform

The bugs so far corrupt *every* batch the same way. A second family corrupts batches *conditionally* — only in training, or only at a certain stage — and these are worth stress-testing explicitly because they survive the static `model_eye_view` check (the one batch you inspected happened to be fine) and only bite under the real schedule.

The classic is **augmentation running at eval time.** Augmentations like `RandomResizedCrop`, `RandomHorizontalFlip`, and `ColorJitter` belong *only* in the training transform; evaluation must use a deterministic resize-and-center-crop. If you accidentally apply the training transform to the validation set — a one-line mistake when you share a `transform` variable between loaders — then every eval pass sees a *different random crop* of each image, and validation accuracy is both lower than it should be and *noisy from run to run*. The signature is unmistakable once you know it: validation accuracy that *changes when you re-run eval on the same checkpoint*. A deterministic eval transform gives the same number every time; a stochastic one does not. The stress test is to evaluate twice and assert equality:

```python
acc1 = evaluate(model, val_loader)
acc2 = evaluate(model, val_loader)
assert abs(acc1 - acc2) < 1e-4, \
    f"eval is nondeterministic ({acc1:.4f} vs {acc2:.4f}): augmentation in the val transform?"
```

If that assert fires, your eval transform contains randomness — almost always a training augmentation that leaked into the validation pipeline. The fix is two explicitly separate transform objects, never a shared one:

```python
import torchvision.transforms.v2 as T
from torchvision.transforms import InterpolationMode

train_tf = T.Compose([
    T.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR, antialias=True),
    T.RandomHorizontalFlip(),
    T.ToImage(), T.ToDtype(torch.float32, scale=True),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# eval: DETERMINISTIC resize + center crop, NO random ops, same interp/antialias
eval_tf = T.Compose([
    T.Resize(232, interpolation=InterpolationMode.BILINEAR, antialias=True),
    T.CenterCrop(224),
    T.ToImage(), T.ToDtype(torch.float32, scale=True),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

Now stress-test the *other* direction — too-strong augmentation in training. If `ColorJitter` or `RandomResizedCrop` with an aggressive scale range distorts images so far that the *label is no longer reliable* from the crop (a `(0.08, 1.0)` scale can crop out the object entirely), you are training on mislabeled examples without any label noise in the data — the augmentation manufactured it. The stress test is to render a grid of augmented training images with `model_eye_view` and *ask whether you could still classify them yourself*. If you cannot tell the breed from the crop, neither can the model, and the augmentation is too strong. This is the boundary where input-pipeline debugging hands off to [augmentation gone wrong](/blog/machine-learning/debugging-training/augmentation-gone-wrong); the input-pipeline tell is that the *eval* transform is clean (deterministic, sane crop) while the *train* transform is the one producing unrecognizable inputs.

A final stress test for the whole pipeline: **freeze the backbone and check that features are stable.** Run the same image through the frozen pretrained backbone with the buggy transform and with `weights.transforms()`, and compare the output feature vectors with cosine similarity. A correct pipeline gives near-`1.0` cosine similarity to the reference; a BGR or normalization bug drops it noticeably (often below `0.9`), and the size of the drop predicts the accuracy hit. This turns "is my preprocessing right?" into a single number you can assert in CI, before any training runs at all:

```python
import torch.nn.functional as F

@torch.no_grad()
def feature_drift(backbone, img_pil, buggy_tf, reference_tf):
    backbone.eval()
    fb = backbone(buggy_tf(img_pil).unsqueeze(0))       # features under buggy transform
    fr = backbone(reference_tf(img_pil).unsqueeze(0))   # features under correct transform
    cos = F.cosine_similarity(fb.flatten(1), fr.flatten(1)).item()
    print(f"feature cosine similarity to reference: {cos:.4f}")
    return cos   # ~1.0 is healthy; < 0.9 means the transform meaningfully distorts inputs
```

## 14. When this is (and isn't) your bug

The discipline that makes input-bug hunting fast is knowing when to *stop* and look elsewhere. A few decisive rules:

- **If the model trains and underperforms by a few points with no crash, the input pipeline is your *first* suspect** — specifically channel order and normalization. These cost two to four points each and leave no other trace. Run the model's-eye view before you touch the model.
- **If the loss is enormous at step 1 and then settles, it is almost always the `uint8`/normalization scale**, not the learning rate. A too-high LR diverges *after* a few steps with a clean start; an unscaled input is huge from step 0. Print the batch range to tell them apart in one line.
- **If train is great and eval is garbage, suspect a train/serve transform skew before you suspect overfitting.** Overfitting widens the gap gradually as training proceeds; a transform skew shows the gap *immediately*, from the first eval, because the eval pixels are simply different. `preprocess_diff` settles it.
- **If only a *subset* of predictions are wrong**, you are looking at EXIF orientation or grayscale/RGBA images — a per-image bug, not a per-batch one. Aggregate accuracy moves a little; the audit (`has_rotation`, mode `Counter`) finds the affected fraction.
- **If `model_eye_view` shows a perfectly normal tensor** — right colors, right orientation, `mean≈0`, `std≈1`, correct shape — then **stop blaming the input** and move to the model, optimizer, or eval branch of the [taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs). The whole value of producing the tensor is that a *clean* tensor decisively clears the entire data branch and saves you from chasing a phantom input bug while the real one hides in the optimizer.
- **If the bug appears only on multi-GPU or only after a checkpoint resume**, it is not the input pipeline — that is a systems bug, and the input pipeline is per-process and deterministic given the same files. Do not waste time on `cvtColor` when the symptom is rank-dependent.

The meta-point: the model's-eye view is not just a way to *find* an input bug; it is a way to *rule out* the entire input layer in thirty seconds. A clean tensor is one of the most useful negative results in debugging, because it shrinks the six-place search space to five.

## 15. Key takeaways

- **A pretrained backbone memorized a specific input distribution.** Channel order, value scale, normalization stats, resize method — get any of them wrong and you do a tiny uncontrolled domain shift that costs accuracy and never crashes.
- **BGR into an RGB backbone costs roughly two to four points.** OpenCV loads BGR; everything else uses RGB. `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` right after decode, every time.
- **Normalization is an affine map, and a wrong one adds a constant bias plus a gain to every activation.** The math is exact: `(σ*/σ)·signal + W·(μ*−μ)/σ`. Print `min/max/mean/std`; healthy ImageNet-normalized data is `mean≈0`, `std≈1`, range `≈[-2.1, 2.6]`.
- **Forgetting `/255` makes the first-step loss explode** (a `255×` input gain). It is the *loud* bug — catch it with a range assert at the batch boundary.
- **Resize interpolation and `antialias` must match between train and serve.** `torchvision` changed the `antialias` default; pin it explicitly and `diff` the two pipelines with `preprocess_diff`. The skew shows up as eval-worse-than-train and is invisible to offline eval.
- **HWC vs CHW is a silent permute bug.** OpenCV/PIL give HWC; convs want CHW. Use `ToTensor()`/`ToImage()` and assert the channel axis is 1 or 3. Do not confuse the `channels_last` memory format with the logical layout.
- **EXIF orientation and grayscale images break a *subset*.** `ImageOps.exif_transpose` and `.convert("RGB")` on every load; audit with `has_rotation` and a mode `Counter`.
- **Show the model's-eye view.** Denormalize, display, and print the four statistics of the exact tensor the model receives. One function clears or convicts the entire input layer in under a minute — and a clean tensor is a decisive negative result.
- **Normalization is not idempotent and decoders are not identical.** Do not normalize twice (the albumentations + torchvision trap), and decode with the same library at train and serve.

## 16. Further reading

- **PyTorch — `torchvision` models and `weights.transforms()` docs.** The canonical preprocessing-with-the-weights API that ships interpolation, antialias, crop, and normalization with each checkpoint, removing four bug classes at once.
- **`torchvision` transforms v2 documentation** — `ToImage`, `ToDtype(scale=True)`, `Normalize`, and the `antialias` default-change notes. Read the antialias section specifically.
- **Albumentations documentation** — `A.Normalize` (`max_pixel_value`), `ToTensorV2`, and the HWC/RGB conventions; the source of the double-normalization and missing-`cvtColor` bugs.
- **OpenCV documentation** — `imread` (BGR), `cvtColor` (`COLOR_BGR2RGB`), and the note that `imread` ignores EXIF orientation.
- **"Mixed Precision Training," Micikevicius et al., 2018** — for the adjacent numerics story (when the input is fine but fp16 underflows), cross-referenced from this series' mixed-precision post.
- **"Pervasive Label Errors in Test Sets…," Northcutt, Athalye, Mueller, 2021 (cleanlab / confident learning)** — for the reference-distribution caveat: even ImageNet's labels are imperfect.
- **Within series:** [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the decision-tree frame), [the input pipeline is lying to you](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you) (the general print-the-batch discipline), [initialization and normalization bugs](/blog/machine-learning/debugging-training/initialization-and-normalization-bugs) (the activation-shift mechanism from the optimization side), [distribution shift, train vs the real world](/blog/machine-learning/debugging-training/distribution-shift-train-vs-the-real-world) (the train/serve skew story), and the capstone [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook).

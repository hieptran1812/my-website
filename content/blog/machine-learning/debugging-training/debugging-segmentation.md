---
title: "Debugging Segmentation: Mask Alignment, Class Indices, and the IoU That Counts Background"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How to find the silent segmentation bugs that pass every assertion and still report a beautiful mean IoU, by overlaying masks, printing unique label values, fixing ignore_index, and computing a per-class IoU that refuses to count background."
tags:
  [
    "debugging",
    "model-training",
    "segmentation",
    "computer-vision",
    "iou",
    "pytorch",
    "albumentations",
    "finetuning",
    "deep-learning",
    "metrics",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/debugging-segmentation-1.png"
---

The dashboard said 0.85 mean IoU. The training loss had fallen smoothly from 1.9 to 0.18, the validation loss tracked it, and the number that the whole project had been chasing for three weeks — mean intersection-over-union, the standard segmentation metric — was sitting at 0.85 by the second epoch and creeping toward 0.87. By every instrument we trust, the model was excellent. So we exported it, ran it on a handful of real images, and overlaid the predicted masks. The roads were there. The buildings were there. The thing we actually cared about — the small class, the one with two percent of the pixels — was simply absent. The model had never learned it, and the metric had never told us, because the metric was averaging in a background class that filled ninety percent of every image and scored 0.98 all by itself. A 0.85 mean IoU was three near-perfect easy classes and one class the model never saw, blended into a single confident-looking number.

This is the defining property of segmentation bugs: they are quiet. A classifier that breaks gives you a confusion matrix you can read at a glance. A language model that breaks generates garbage you can see. But a segmentation model operates on a dense grid of per-pixel labels, hundreds of thousands of predictions per image, and almost every way it can go wrong produces a loss that descends and a metric that looks plausible. The mask was resized with bilinear interpolation and your class 2 is now a smear of phantom labels — the loss still goes down. The mask is shifted one pixel relative to the image because albumentations resized them with different alignment — the loss still goes down. The void label is 255 and your loss is happily training the model to predict "class 255" on every unlabeled pixel — the loss still goes down. You are flying on instruments, and the instruments are lying.

![A vertical stack showing image and mask, augmentation, target tensor, loss, and metric layers, with the augmentation stage flagged as the place masks silently corrupt while the IoU stays high](/imgs/blogs/debugging-segmentation-1.png)

In the six-places framework that runs through this series — a bug hides in *data*, *optimization*, *model code*, *numerics*, *systems*, or *evaluation* — segmentation is unusual because the same root cause often lands in two places at once. A bilinear-resized mask is a *data* bug that also poisons your *evaluation*; a missing `ignore_index` is an *optimization* bug (the loss optimizes a label that should not exist) that also corrupts the *metric*. The discipline that cuts through this is the one we return to every post: **make-it-fail-small** and **read the instruments**. For segmentation the "small" thing is a single image with its mask, and the instruments are not loss curves — they are *the mask overlaid on the image* and *the set of unique values in the label tensor*. Two checks, run before you train, that catch the overwhelming majority of segmentation bugs in the field.

By the end of this post you will be able to take any segmentation run — a U-Net that "converged," a DeepLab finetune that scores 0.85 and ships nothing useful, a boundary that comes out as noise — and localize the bug in minutes. You will know the six silent failures: mask–image misalignment, interpolation on label maps, class-index off-by-one and palette confusion, the mishandled void label and the `ignore_index` you forgot to set, the one-hot-versus-index target shape mistake, and the mean IoU that counts background. For each you will have the science of *why* it happens, a runnable diagnostic in PyTorch and albumentations and torchmetrics, and a before→after that turns the 0.85 mirage into an honest per-class story. Let us start where the bugs start: with the mask and the image not lining up.

## 1. Mask–image misalignment: the spatial off-by-one

Semantic segmentation has a hard invariant that classification does not: **pixel (i, j) of your mask must label pixel (i, j) of your image**. Not approximately — exactly. The supervision is a per-pixel correspondence, and the moment that correspondence slips by even a single pixel or a fraction of one, you are training the model to map an image patch to the label of a *different* patch. The model will still learn something, because most pixels are interior pixels far from any boundary and a one-pixel shift does not change their class. But every boundary pixel — exactly the pixels that decide IoU, exactly the pixels that matter for a crisp segmentation — is now mislabeled. The model learns blurry, hedged boundaries because the ground truth it sees *is* inconsistent at boundaries, and your IoU caps at a ceiling you cannot break no matter how long you train.

The reason this happens so often is that the image and the mask travel through your augmentation pipeline as two separate arrays, and any operation that treats them even slightly differently breaks the correspondence. Resize the image with bilinear interpolation and the mask with nearest-neighbor — fine, that is correct, as we will see. But resize them to subtly different shapes, or apply a random crop to one with a different random seed than the other, or flip one and not the other, or — the classic — let two different library calls disagree about whether a pixel coordinate refers to a pixel center or a pixel corner, and the mask drifts. Sub-pixel misalignment is the worst kind because it produces no error and no visible artifact in either array alone. You only see it when you put them on top of each other.

![A three-by-three grid showing background and object cells where the object label is shifted one column to the right, so the right-hand boundary cells carry the wrong class](/imgs/blogs/debugging-segmentation-4.png)

### The science: where a half-pixel comes from

The deepest source of misalignment is not a coding mistake at all — it is the *resize coordinate convention*, and it bites people who did everything else right. When you resize an image from width $W$ to width $W'$, you must decide where the new sample points fall relative to the old grid. There are two conventions. **Align-corners** maps the first and last pixel *centers* to each other: new pixel $i'$ samples old coordinate $x = i' \cdot \frac{W-1}{W'-1}$. **Half-pixel** (the modern default in OpenCV, PIL, and PyTorch's `interpolate` with `align_corners=False`) treats pixels as unit squares and maps *centers* to *centers* with a half-pixel offset: $x = (i' + 0.5)\cdot\frac{W}{W'} - 0.5$. These two formulas place the sample points at different locations, and the difference is largest at the edges of the image — exactly where object boundaries often live.

Here is the trap made concrete. Suppose you resize the *image* with `cv2.resize` (half-pixel convention) and the *mask* with `torch.nn.functional.interpolate(..., align_corners=True)` because you copied that line from a tutorial. The image samples old coordinate $x_{\text{img}} = (i'+0.5)\frac{W}{W'} - 0.5$ and the mask samples $x_{\text{mask}} = i'\frac{W-1}{W'-1}$. For $W = 512$, $W' = 256$, the offset between them at $i' = 255$ (the last column) is

$$\Delta = x_{\text{img}} - x_{\text{mask}} = \left(255.5 \cdot \frac{512}{256} - 0.5\right) - \left(255 \cdot \frac{511}{255}\right) = 510.5 - 511 = -0.5.$$

A clean half-pixel shift, accumulating toward the edges. Half a pixel sounds harmless, but at a downsampled resolution of 256 it corresponds to a full pixel in the original 512, and at a boundary that is the difference between "this pixel is road" and "this pixel is sidewalk." Multiply across millions of boundary pixels and your IoU loses several points it can never recover, with no error in sight. The fix is not to find the magic convention; it is to **resize the image and mask with the same library call and the same convention**, ideally inside a single transform that operates on both. That is the entire reason mask-aware augmentation libraries exist.

### The diagnostic: overlay the mask, at full and downsampled resolution

The definitive alignment check is to overlay the mask on the image as a translucent color layer and *look*. Not at one resolution — at two. At full resolution a one-pixel shift is nearly invisible to the eye; at a small downsampled resolution (say 64×64) the same shift becomes a visible band of misalignment along every boundary, because the relative error per displayed pixel grows. This is the single most valuable thing you can do before training a segmentation model, and it takes thirty seconds.

```python
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

def overlay_mask(image, mask, num_classes, alpha=0.5):
    """image: HxWx3 uint8 array. mask: HxW int array of class ids.
    Returns an RGB overlay you can imshow."""
    # A fixed colormap so class k is always the same color.
    rng = np.random.RandomState(0)
    palette = rng.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    color_mask = palette[np.clip(mask, 0, num_classes - 1)]
    blended = (1 - alpha) * image + alpha * color_mask
    return blended.astype(np.uint8)

def check_alignment(image, mask, num_classes):
    """Overlay at full res and at 64x64 to expose sub-pixel shifts."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(overlay_mask(image, mask, num_classes))
    ax[0].set_title("full resolution")

    # Downsample image (bilinear) and mask (nearest) the CORRECT way.
    img_t = torch.from_numpy(image).permute(2, 0, 1).float()[None]
    msk_t = torch.from_numpy(mask).float()[None, None]
    img_s = F.interpolate(img_t, size=(64, 64), mode="bilinear",
                          align_corners=False)[0].permute(1, 2, 0).numpy()
    msk_s = F.interpolate(msk_t, size=(64, 64), mode="nearest")[0, 0].numpy()
    ax[1].imshow(overlay_mask(img_s.astype(np.uint8), msk_s.astype(int),
                              num_classes))
    ax[1].set_title("downsampled 64x64")
    plt.show()
```

Run this on five random training samples. If the colored mask sits *on* the object at both resolutions, your alignment is correct. If at 64×64 you see a consistent fringe — the color leading or trailing the object edge by a band — you have a misalignment bug, and the direction of the fringe tells you the sign of the shift. The reason you check the downsampled view is that misalignment is a *boundary* phenomenon, and downsampling concentrates boundaries: a 512×512 image with a thin one-pixel error has that error spread across thousands of edge pixels, each barely off; the 64×64 version forces those errors into a handful of pixels where they become unmissable.

#### Worked example: the U-Net that capped at 0.71

A road-scene U-Net trained for two days and plateaued at 0.71 mean IoU. The loss was healthy, the curve smooth, no NaNs. The team assumed the model lacked capacity and started reaching for a bigger backbone. Instead we ran `check_alignment` on ten samples. At full resolution everything looked fine. At 64×64, every object's mask was shifted one pixel down and to the right — a clean, consistent diagonal fringe. The cause: the image loader used `cv2.resize` (half-pixel) and a hand-written mask resize used `align_corners=True`. We routed both through one albumentations `Resize` transform. The next run climbed past 0.71 within an epoch and settled at 0.83 — twelve points of IoU that had nothing to do with model capacity and everything to do with a half-pixel of geometry. The instrument that found it was not the loss curve; it was the overlay at low resolution, and it cost thirty seconds against two wasted days.

Random geometric augmentation is the second great source of misalignment, and it is sneakier than the resize convention because it only breaks *some* of the time. Consider a random crop applied independently to image and mask. If you sample the crop box once for the image and *again* — with a freshly drawn random number — for the mask, the two crops land at different offsets, and now the entire image is shifted relative to its labels by a random amount that changes every epoch. The model sees a different misalignment on every pass, so it can never even learn the consistent-but-wrong boundaries it would learn under a fixed shift; instead it learns mush, and the loss plateaus high with no clear signature. The same trap waits in random horizontal flip (flip the image with probability 0.5 using one coin, flip the mask with another coin, and half your batches have a left-right-reversed mask), random rotation (different sampled angle), and elastic deformation (different displacement field). Any augmentation that draws a random parameter must draw it *once* and apply it to both arrays. This is not a convenience that mask-aware libraries provide — it is a correctness requirement, and hand-rolling augmentation for segmentation is hand-rolling a misalignment generator.

The reason a *consistent* shift caps IoU at a stable ceiling while a *random* shift produces high-variance mush is worth stating precisely, because the two have different signatures on the loss curve. Under a fixed one-pixel shift, the ground truth is internally consistent — every image of a road edge is labeled one pixel off in the same direction — so the model learns the shifted boundary and converges to a stable, suboptimal IoU; the curve looks healthy and just plateaus low. Under a random shift, the ground truth is *inconsistent* — the same visual boundary is labeled differently on different epochs — so the gradient signal at boundaries partially cancels across batches, the boundary predictions stay blurry, and the loss is both higher and noisier. A plateaued-but-smooth curve points at a consistent bug (resize convention); a high-and-jittery curve at boundaries points at a random-augmentation desync. Reading which one you have narrows the suspect before you even open the augmentation code.

The fix is structural: never transform the image and mask independently. Use a mask-aware augmentation library that guarantees the two stay locked. In albumentations, you pass both to the *same* `__call__` and the library applies the *same* geometric parameters to each, with the correct interpolation per target:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# additional_targets lets one transform pipeline drive image + mask
# with identical geometry; interpolation per target is handled inside.
transform = A.Compose([
    A.Resize(256, 256),            # same geometry for both
    A.HorizontalFlip(p=0.5),       # flips image AND mask together
    A.RandomResizedCrop(224, 224, scale=(0.5, 1.0)),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

out = transform(image=image, mask=mask)   # ONE call, locked geometry
img_t, msk_t = out["image"], out["mask"]   # aligned by construction
```

The single most important word in that snippet is `mask=`. Albumentations knows that the `mask` target must be resized and cropped with the *same random parameters* as the `image`, and it knows — critically — that masks must use nearest-neighbor interpolation while images use bilinear. That brings us to the second bug, the one that hides inside the word "resize."

## 2. Interpolation on masks: why nearest is mandatory

Resizing an image is a continuous operation. The pixel values are samples of an underlying continuous intensity field, and interpolation — bilinear, bicubic — estimates that field at new sample points by averaging neighbors. Averaging is exactly the right thing to do for an image, because the average of two nearby gray values is a plausible gray value in between.

A mask is not a continuous field. It is a map of *categorical labels*. The integer 1 means "road" and the integer 3 means "building," and the value 2 does not mean "halfway between road and building" — it means "sidewalk," a different category entirely. When you bilinearly resize a label map and the interpolation averages a class-1 pixel and a class-3 pixel, it computes $(1+3)/2 = 2$ and writes "sidewalk" into a pixel that was on the boundary between road and building. The interpolation has *invented a class id that describes a category that is not present at that location*. This is not a rounding error you can tolerate; it is a categorical lie, and it happens at every boundary in the image.

![A before-and-after comparison contrasting bilinear interpolation that averages class one and class three into a non-existent class two against nearest-neighbor that copies the closest label untouched](/imgs/blogs/debugging-segmentation-2.png)

### The science: averaging integers is averaging nonsense

Make the math explicit, because it tells you exactly what corruption to expect. A bilinear resize computes each output pixel as a convex combination of up to four input pixels: $v_{\text{out}} = \sum_k w_k v_k$ with weights $w_k \ge 0$ summing to one. For an image, $v_k$ are intensities and the weighted average is meaningful. For a mask, $v_k$ are class indices, and the weighted average is a real number that gets cast back to an integer. Two failure modes follow directly.

First, **interpolation invents classes that exist nowhere in the source mask**. If a boundary separates class 0 (background) and class 4 (car), every bilinear output pixel along that boundary takes a value in $(0, 4)$ — you will find pixels labeled 1, 2, and 3 in the resized mask even though the source mask contained only 0 and 4. If 1, 2, 3 happen to be valid classes (person, road, sidewalk), your model is now being trained to predict *people and roads and sidewalks along the edges of every car*, pure noise supervision. If they are not valid classes, you get an out-of-range index that crashes your loss (`CUDA error: device-side assert triggered`, the dreaded index-out-of-bounds in `nll_loss`).

Second, **the void label poisons everything around it**. Suppose unlabeled pixels carry the value 255 (Cityscapes convention). A boundary between class 2 and void-255 produces bilinear values sweeping from 2 up to 255 — your resized mask now contains 2, 47, 130, 200, 255, a continuous ramp of garbage integer labels, every one of which is either an invalid class or a real class assigned to the wrong pixel. A single void region, bilinearly resized, can corrupt a wide band of labels around it. This is why printing the unique values of a resized mask is such a powerful detector: a correct nearest-resized mask contains *exactly* the class ids that were in the source; a bilinear-resized mask contains a spray of values that should not be there.

The correct operation is **nearest-neighbor interpolation**, which for each output pixel copies the value of the single closest input pixel. No averaging, no invention: the output contains only values that were in the input. It produces slightly blocky boundaries at large upscale factors, but blocky-and-correct beats smooth-and-fabricated every time for a label map. The rule is absolute: **images get bilinear/bicubic; masks get nearest. Always.**

It helps to make the corruption concrete with a one-dimensional example you can verify by hand. Take a tiny mask row of five pixels with class ids `[0, 0, 4, 4, 4]` — a clean boundary between background (0) and car (4) between positions 1 and 2. Downsample it to three pixels. Nearest-neighbor picks the closest source pixel for each of the three outputs and returns `[0, 4, 4]` — only 0 and 4, exactly the classes present, boundary preserved as cleanly as three pixels allow. Bilinear computes each output as a weighted average of its source neighbors: the middle output pixel falls between source positions that straddle the boundary, averaging a 0 and a 4 to produce $0.5\times0 + 0.5\times4 = 2.0$, and the resized row becomes `[0, 2, 4]`. Class 2 — say "person" — now appears in a mask that contained no people, sitting precisely on the car's edge. Scale that single invented pixel up to a 512×512 image with thousands of boundary pixels and you have thousands of phantom-class labels ringing every object, and the model dutifully learns to hallucinate a thin outline of class 2 around every class-4 region. That thin hallucinated outline is a *visible* signature in predictions, and it is a dead giveaway of bilinear-on-mask once you know to look for it.

The same nearest-only rule applies in a place people forget: **upsampling the model's predicted mask back to the original image resolution.** Your network often predicts at a lower resolution than the input (a stride-8 or stride-16 output, or a deliberately downscaled training resolution), and at inference you resize the predicted class map back up to compare against full-resolution ground truth or to save a result. If you resize the *argmax class map* with bilinear interpolation — averaging integer class ids — you re-introduce the exact phantom-class corruption on the output side, degrading your IoU for reasons that have nothing to do with the model. The correct pipeline either resizes the *logits* (which are continuous and can legitimately be bilinearly interpolated) and *then* takes the argmax, or takes the argmax first and resizes the resulting class map with nearest. Resizing logits-then-argmax is usually slightly better because it interpolates in the continuous space the network actually produces; resizing argmax-then-nearest is acceptable and cheaper. Resizing argmax with bilinear is always wrong. The category "label map" stays categorical from disk, through the loss, all the way to the prediction you finally write to a file.

### The diagnostic: print the unique mask values

This is the cheapest, highest-yield check in all of segmentation debugging, and you should run it on every mask the instant it leaves your dataloader.

```python
import torch
from collections import Counter

def audit_mask_values(loader, expected_ids, ignore_value=255, n_batches=20):
    """Walk the dataloader and assert masks contain only legal labels.
    expected_ids: set of valid class indices, e.g. set(range(num_classes)).
    Catches bilinear-invented ids and a misread ignore value."""
    seen = Counter()
    legal = set(expected_ids) | {ignore_value}
    for i, (_, masks) in enumerate(loader):
        if i >= n_batches:
            break
        vals, counts = torch.unique(masks, return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            seen[v] += c
            if v not in legal:
                raise ValueError(
                    f"ILLEGAL mask value {v} (count {c}). "
                    f"Legal ids are {sorted(legal)}. "
                    f"Likely bilinear interpolation on the mask, "
                    f"or a wrong ignore value."
                )
    total = sum(seen.values())
    print("mask value histogram (value: fraction):")
    for v in sorted(seen):
        print(f"  {v:>4}: {seen[v] / total:6.3f}")
    return seen
```

The output is diagnostic on two axes at once. If you see values that are not in your legal set — `ILLEGAL mask value 47` — you have bilinear interpolation on a mask (or a corrupt file). If the histogram shows a value like 255 carrying twelve percent of all pixels, that is your void label and you had better be ignoring it in the loss and the metric (section 4). And if you *expected* background to be index 0 but the dominant value is index 1, you have an indexing convention mismatch (section 3). One twelve-line function, three classes of bug.

#### Worked example: the boundary that came out as static

A medical segmentation model produced masks whose interiors were correct but whose boundaries were a band of speckled noise — a few pixels of "tumor" scattered into "healthy tissue" along every edge, and vice versa. The loss was low; the Dice score was a respectable 0.88. We ran `audit_mask_values` and the histogram contained class ids 0, 1, 2, 3, 4 — but the source masks were binary, only 0 and 1. The resize step (`Image.resize(size)` with the default `BICUBIC` for a PIL image opened as a label map) had bicubically interpolated the binary mask, and bicubic *overshoots*: it produced values below 0 and above 1 that clamped and quantized into spurious ids 2, 3, 4 along the boundary. We changed one argument — `Image.resize(size, Image.NEAREST)` — and re-ran the audit: only 0 and 1, as it should be. The boundary static vanished and the honest Dice (computed correctly, see section 6) rose to 0.91 with a much cleaner edge. The bug was a default interpolation argument on a single `resize` call.

| Operation | Image (continuous) | Mask (categorical) | What goes wrong on a mask |
| --- | --- | --- | --- |
| `cv2.resize` | `INTER_LINEAR` (default) | must use `INTER_NEAREST` | linear averaging invents ids between neighbors |
| `PIL.Image.resize` | `BICUBIC`/`BILINEAR` | must use `NEAREST` | bicubic overshoots, creates out-of-range ids |
| `F.interpolate` | `mode="bilinear"` | must use `mode="nearest"` | averages indices, corrupts void boundaries |
| `T.Resize` (torchvision) | bilinear default | use `InterpolationMode.NEAREST` | same averaging trap |
| albumentations `Resize` | bilinear for image | **auto-nearest for mask** | handled correctly if you pass `mask=` |

The last row is why mask-aware libraries are worth adopting: albumentations applies nearest to the `mask` target automatically, so you cannot make this mistake if you route masks through it. But the moment you hand-roll a resize — and people do, constantly, for "just this one preprocessing step" — you are one default argument away from poisoning every boundary in your dataset.

## 3. Class-index bugs: off-by-one, palettes, and what zero means

A segmentation mask is, at the byte level, an array of small integers. But "small integer" hides a remarkable number of conventions, and getting any of them wrong shifts your entire label space by one, silently. The model still trains — it just learns to predict "building" where you meant "road" because your class-1 pixels are being supervised as class 0. The loss descends, the metric looks fine on the *internal* indexing, and you only discover the off-by-one when you visualize predictions against the true semantic labels and find the whole color map rotated.

The first and most common ambiguity is **what index 0 means**. In many datasets and frameworks, index 0 is *background* and the real classes are 1, 2, …, C. In others, there is no background channel at all and index 0 is the *first real class*. If your dataset uses 0-as-background but you set up your model with `num_classes` equal to the count of foreground classes (forgetting to add one for background), every label is off by one relative to your network's output channels, and class C — the last one — maps to an output channel that does not exist, crashing the loss or wrapping around. Conversely, if your dataset has no background and you reserve channel 0 for one anyway, you train a dead channel and shift everything.

![A decision tree from the mask index zero question branching to background versus first class and then to the void label encoding and the off-by-one bug](/imgs/blogs/debugging-segmentation-5.png)

### Palette PNGs versus index PNGs: the colormap trap

The second ambiguity lives in how the mask is *stored on disk*. Segmentation masks are usually saved as PNGs, and there are two completely different ways to do it. An **index PNG** (also called a paletted or P-mode PNG) stores the class index directly as the pixel value: pixel value 3 means class 3, and the PNG's palette is just a display convenience that maps index 3 to some RGB color so humans can look at it. A **color PNG** stores the RGB color itself: the pixel is literally `(128, 64, 128)` (the Cityscapes road color), and you must map that RGB triple back to the class index 0 yourself before training.

The bug is reading one as if it were the other. If you open an index PNG with a library that *applies the palette* — for example `cv2.imread` on a paletted PNG, or `Image.open(...).convert("RGB")` — you get back the *colors*, not the indices, and now your "mask" is a three-channel RGB image whose values are 128 and 64, not class ids 0, 1, 2. Train on that and your targets are nonsense. Conversely, if you open a color PNG and treat the raw bytes as class indices, you get class ids like 128 and 64, far outside your valid range, and the loss crashes or trains garbage. The correct read for an index PNG is `np.array(Image.open(path))` *without* converting to RGB — that returns the raw palette indices — or `Image.open(path)` left in `P` mode. The single most useful habit is to assert, right after loading, that the mask is 2-D (H×W), not 3-D (H×W×3); a 3-D "mask" means you accidentally loaded colors.

```python
import numpy as np
from PIL import Image

def load_index_mask(path, num_classes, ignore_value=255):
    """Load a paletted/index PNG as raw class indices, NOT colors.
    Asserts the result is a 2-D integer map in the legal range."""
    m = np.array(Image.open(path))           # do NOT .convert("RGB")
    assert m.ndim == 2, (
        f"mask at {path} is {m.ndim}-D; you loaded COLORS not indices. "
        f"Open the paletted PNG without converting to RGB."
    )
    legal = set(range(num_classes)) | {ignore_value}
    bad = set(np.unique(m).tolist()) - legal
    assert not bad, f"illegal class ids {sorted(bad)} in {path}"
    return m.astype(np.int64)
```

### The science of the off-by-one: it is a contiguity requirement

There is a precise reason the indexing must be contiguous `0..C-1` (plus the ignore value), and it comes from how the loss is computed. `nn.CrossEntropyLoss` for a `[B, C, H, W]` logits tensor and a `[B, H, W]` target gathers, for each pixel, the logit at channel `target[b, i, j]`. That gather is a raw array index into the channel dimension. If a pixel's target is 5 but your logits have only 5 channels (indices 0–4), the gather reads out of bounds: on CPU you get a Python `IndexError`, on CUDA you get the infamous asynchronous `device-side assert triggered` that points at a line nowhere near the real problem. And if your labels are, say, `{0, 1, 3, 4}` — class 2 was dropped during annotation and the ids were never compacted — then channel 2 is trained to predict a class that never appears (its target frequency is zero, so it learns to output negative infinity logits) while the gather still works. The model wastes a channel and your per-class IoU table has a permanent 0.0 for a class that does not exist.

The fix is to **remap labels to a contiguous range exactly once**, at load time, with an explicit lookup, and to do the *same* remap for both training and metric computation so they never drift:

```python
import numpy as np

# Dataset's raw ids are sparse: {0, 7, 11, 21, 255-void}. Compact them.
RAW_TO_TRAIN = {0: 0, 7: 1, 11: 2, 21: 3}   # 4 real classes
IGNORE_INDEX = 255

def remap_labels(mask):
    """Vectorized remap of raw dataset ids to contiguous train ids.
    Unmapped values become IGNORE_INDEX so the loss skips them."""
    out = np.full_like(mask, IGNORE_INDEX)
    for raw, train in RAW_TO_TRAIN.items():
        out[mask == raw] = train
    return out
```

This is exactly how Cityscapes is handled in practice: its 34 raw label ids are remapped to 19 contiguous training ids, with every unused id sent to the ignore label. Do the remap once, store it in one place, and reuse that same map for loss *and* metric — the moment training and evaluation use different remaps, your IoU is computed on a different label space than the one the model learned, and the number is meaningless.

A practical hazard hides in *how* you apply the remap. The naive approach — a Python loop over `if mask == raw: out = train` — is what the snippet above does and it is correct, but it is slow on large masks and, worse, people sometimes "optimize" it into an in-place rewrite that corrupts itself. Consider rewriting ids in place: `mask[mask == 0] = 10; mask[mask == 10] = 1`. The first line turns every 0 into 10; the second line then turns *both the original 10s and the just-created 10s* into 1, double-mapping the zeros. In-place sequential remapping is a classic self-inflicted bug because each assignment sees the results of the previous ones. The fix is to write into a *fresh* output array (as `remap_labels` does with `np.full_like`), or to build a single lookup-table array `lut` of length 256 and index it once: `out = lut[mask]`. The lookup-table form is both fast and immune to the double-mapping trap, because every source value is translated exactly once from the original array:

```python
import numpy as np

# A 256-entry LUT: index by raw id, get train id. Build it once.
lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for raw, train in RAW_TO_TRAIN.items():
    lut[raw] = train

def remap_fast(mask):
    """O(1) per pixel, immune to sequential-overwrite double-mapping."""
    return lut[mask]          # one vectorized gather, no loops, no aliasing
```

The asymmetry between the loss's tolerance for index bugs and the metric's tolerance is worth internalizing, because it explains why off-by-one bugs survive so long. The loss will *crash loudly* if an index is out of range (the gather reads past the channel dimension), so a label of 5 in a 4-class model dies immediately and you fix it. But the loss is perfectly *silent* about a label that is in range but semantically wrong — a consistent off-by-one where your class 1 is the dataset's class 0. The model trains, the loss descends, and the bug only surfaces when a human looks at decoded predictions next to ground truth. This is why the index audit has two halves: an automatic range check (caught by the assert, cheap) and a manual semantic check (decode a few predictions to RGB and eyeball them against the originals, irreplaceable). The automatic half catches the crashes; only the manual half catches the rotation.

#### Worked example: the whole colormap rotated by one

A satellite land-cover model trained cleanly and reported 0.79 mean IoU, but the predicted maps looked subtly wrong — water where there should be forest, forest where there should be urban. Visualizing predictions next to ground truth, the entire color map was rotated by one class. The cause: the dataset's masks used 0 for the first real class (water), but the training code assumed 0 was background and started real classes at 1, so it had quietly added a background channel and shifted every label up by one during a "cleanup" step. The model had faithfully learned the *shifted* labels, so its internal IoU was 0.79 — but every prediction, decoded back to the original class space, was off by one. We removed the spurious background channel, set `num_classes` to the true count, and confirmed with `audit_mask_values` that targets ran `0..C-1`. The IoU was unchanged numerically (the model was always fine), but the *predictions now decoded correctly*, which is the only thing that mattered. The lesson: an off-by-one in indexing does not always hurt the loss — it can produce a perfectly trained model that predicts the wrong labels, and only a prediction-versus-ground-truth visualization reveals it.

## 4. The void label and ignore_index: training on pixels you cannot label

Almost every real segmentation dataset has pixels that are *not labeled*: regions the annotators skipped, ambiguous boundaries, the ego-vehicle hood in a driving dataset, the black bars from a letterboxed image. These pixels carry a special **void** or **ignore** label — conventionally 255 in Cityscapes and many PyTorch examples, or -1 in some codebases, or a dedicated "unlabeled" class id. The critical property of a void pixel is that it has *no correct answer*. You do not want the model to learn to predict anything there, and you do not want those pixels to count toward the metric. They are excluded supervision.

The bug is forgetting to exclude them. If your void label is 255 and you do not tell the loss to ignore it, then `CrossEntropyLoss` tries to gather logit channel 255 for every void pixel — which either crashes (if you have fewer than 256 classes, which you always do) or, if you have remapped void into a real-looking integer, *trains the model to confidently predict that integer on every unlabeled pixel*. Twelve percent of your pixels (a typical void fraction) are now teaching the model to output garbage, and the gradient from those pixels fights the gradient from the real labels. The loss still descends — there are enough real pixels to drive it down — but the model wastes capacity and its boundaries near void regions degrade.

![A horizontal timeline of four checks, overlay the mask, print unique values, confirm ignore index, and per-class IoU, in the order you run them before trusting a score](/imgs/blogs/debugging-segmentation-8.png)

### The science: ignore_index removes terms from the sum, not pixels from the image

PyTorch's `ignore_index` does something precise and worth understanding exactly. Cross-entropy over an image is a sum over pixels: $L = \frac{1}{N}\sum_{p} -\log \hat{p}_{p, y_p}$, where $p$ ranges over all $N$ pixels and $y_p$ is the target class at pixel $p$. When you set `ignore_index=255`, PyTorch does two things: it skips every pixel whose target equals 255 (no loss term, no gradient from that pixel), *and* — this is the part people miss — with the default `reduction="mean"` it **divides by the number of non-ignored pixels**, not by $N$. So the denominator is the count of real pixels, and the per-pixel loss scale stays correct regardless of how much void there is. If `ignore_index` instead just zeroed the void terms but still divided by $N$, your loss would be artificially deflated by the void fraction and your effective learning rate on the real pixels would shrink. The correct `ignore_index` machinery removes the terms *and* fixes the normalization.

You must set it in two independent places, and forgetting either is a bug:

```python
import torch.nn as nn

IGNORE_INDEX = 255

# 1) The LOSS must ignore void pixels (no gradient from them).
criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

# logits: [B, C, H, W] raw (no softmax). target: [B, H, W] long.
loss = criterion(logits, target)   # void pixels contribute nothing
```

```python
from torchmetrics import JaccardIndex

# 2) The METRIC must also ignore void pixels, or IoU is computed on
#    pixels the model was never trained on.
iou = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES,
                   ignore_index=IGNORE_INDEX, average="none")
iou.update(preds, target)          # preds: [B, H, W] argmax class ids
per_class = iou.compute()          # tensor of length NUM_CLASSES
```

Set `ignore_index` in the loss but not the metric, and your IoU is scored against void pixels the model never learned — those pixels are predicted as *something*, and they count as errors (or, worse, accidental successes) in the denominator of intersection-over-union, dragging the number in whichever direction the void happens to fall. Set it in the metric but not the loss, and you train on void garbage while scoring honestly — the loss fights itself and the model underperforms its potential. You need it in both, with the same value.

There is a third place the void label can leak that almost nobody checks: **augmentation that creates void where there was none.** Rotate an image by 15 degrees and the corners that rotate out of frame must be filled with *something* — and if you fill them with a constant, that constant had better be the ignore label, not class 0. The default `border_mode` of many rotation and affine transforms fills with zeros, and if class 0 is a real class (or even if it is background), you have just painted real-class labels into the rotated-in corners, fabricating supervision out of empty space. The correct setting fills the mask's out-of-frame regions with the ignore value so the loss skips them: in albumentations, `A.Rotate(border_mode=cv2.BORDER_CONSTANT, mask_value=255)` (paired with whatever fill you like for the image). The same applies to padding a non-square image to a square, to perspective warps, and to any transform that introduces pixels with no source. The rule generalizes: **any pixel your augmentation invents has no ground truth and must carry the ignore label.** A unique-value audit *after* augmentation (not just on the raw masks) catches this — if 255 appears only after you turn on rotation, your rotation is correctly voiding the corners; if class 0 mysteriously grows when you add rotation, it is wrongly filling them.

One more numeric consequence of the `mean` reduction is worth seeing because it predicts a real symptom. Suppose 12% of pixels are void and you forget `ignore_index` entirely, but the void value (say 0) is a valid class, so nothing crashes. Now your loss averages over *all* pixels including the 12% void, and the model spends gradient learning to predict class 0 on void regions. Two things happen that you can observe on the instruments: the training loss settles *lower* than it should (the void pixels are easy and consistent — always "predict 0" — so they pull the average loss down), and the boundary IoU of the classes adjacent to void regions is suppressed (the model is being pulled toward predicting void-class-0 near those boundaries). A suspiciously low loss combined with degraded boundaries near specific regions is the fingerprint of void leaking into the loss, and the unique-value histogram showing a fat 12% spike at value 0 confirms it.

#### Worked example: the loss that trained on the sky

A driving-scene model had a void label of 255 for unlabeled sky and the ego-hood. The original config built the loss as plain `nn.CrossEntropyLoss()` — no `ignore_index`. Because the labels had been remapped so that void became a *valid-looking* class id (the author had mapped 255 → 19, just past the 19 real classes, and set `num_classes=20` to "make room"), the loss did not crash. Instead, channel 19 learned to fire on sky and hood, twelve percent of every image. The mean IoU over 20 classes was 0.81 — and class 19 ("void"), being huge and easy, scored 0.97 and *inflated the mean*. We did three things: set `num_classes=19`, set `ignore_index=255` in both loss and metric, and remapped void back to 255 instead of a fake class. The loss now skipped void; the metric now excluded it. The honest mean IoU over the 19 real classes dropped to 0.73 — eight points lower and finally *true* — and, freed of the void-prediction objective, the real classes' boundaries near sky and hood sharpened, lifting the genuine classes by about a point each over the next epoch. The 0.81 had been partly a fictional void class scoring 0.97.

## 5. One-hot versus index targets: the shape that should crash but does not

PyTorch's segmentation losses expect a specific target shape, and there are two common formats that get confused: **index targets** of shape `[B, H, W]` (a long tensor where each pixel is its class id) and **one-hot targets** of shape `[B, C, H, W]` (a tensor with a 1 in the channel of the true class). `nn.CrossEntropyLoss` wants the *index* form: logits `[B, C, H, W]`, target `[B, H, W]` of dtype `long`. Hand it a one-hot `[B, C, H, W]` float target and, depending on your PyTorch version, it either errors clearly or — in recent versions that accept "soft" probabilistic targets — *silently interprets your one-hot as soft labels and computes a different, subtly wrong loss*. The newer behavior is the dangerous one: no crash, a plausible loss, a model that trains on a slightly wrong objective.

The mirror-image bug is the channel-versus-index confusion in the *spatial* dimensions. A target accidentally shaped `[B, 1, H, W]` (an extra singleton channel, common when you forget to `squeeze` after loading) will broadcast in surprising ways or trigger a shape error that points at the loss rather than the dataloader. And a logits tensor in the wrong layout — `[B, H, W, C]` (channels-last) passed to a loss expecting `[B, C, H, W]` — will treat your width as the class dimension, gather nonsense, and produce a loss that is finite and meaningless.

The defense is a single shape-and-dtype assert that runs on the first batch and never again, costing nothing after step one:

```python
def assert_seg_shapes(logits, target, num_classes):
    """Catch the shape/dtype bugs that produce a finite, wrong loss.
    Run once on the first training batch."""
    B, C, H, W = logits.shape
    assert C == num_classes, (
        f"logits have {C} channels but num_classes={num_classes}; "
        f"check channels-last vs channels-first layout."
    )
    assert target.shape == (B, H, W), (
        f"target shape {tuple(target.shape)} != expected {(B, H, W)}; "
        f"a [B,1,H,W] target needs .squeeze(1); a [B,C,H,W] target is "
        f"one-hot and CrossEntropyLoss wants class indices."
    )
    assert target.dtype == torch.long, (
        f"target dtype {target.dtype} != torch.long; float targets get "
        f"read as soft labels and silently change the loss."
    )
    assert target.max() < num_classes or (target == IGNORE_INDEX).any(), (
        f"target max {target.max().item()} >= num_classes; out-of-range "
        f"class id will assert in the loss gather."
    )
```

There is one legitimate place one-hot belongs: **Dice and soft-IoU losses**. Dice loss is computed by multiplying predicted probabilities (per channel) by a one-hot target and summing — it genuinely needs the `[B, C, H, W]` one-hot form, built with `F.one_hot(target, num_classes).permute(0, 3, 1, 2)`. The bug there is *forgetting to exclude the ignore pixels before one-hotting*: `F.one_hot` cannot represent 255, so you must mask void pixels out of both the prediction and the one-hot target first, or zero their contribution. Mixing up which loss wants which format — index for cross-entropy, one-hot for Dice — is its own small bug class, and the assert above plus a comment at each loss site is the cheapest guard.

| Target format | Shape | Dtype | Loss that wants it | Failure if mismatched |
| --- | --- | --- | --- | --- |
| Class indices | `[B, H, W]` | `long` | `CrossEntropyLoss`, `nll_loss` | one-hot read as soft labels (recent torch) |
| One-hot | `[B, C, H, W]` | `float` | Dice / soft-IoU | passed to CE: silent soft-label loss |
| Extra channel | `[B, 1, H, W]` | `long` | none (bug) | broadcast / shape error in loss |
| Channels-last logits | `[B, H, W, C]` | `float` | none (bug) | width treated as classes, finite garbage |

## 6. The IoU that lies: why background dominance inflates mean IoU

Now we arrive at the bug that wasted three weeks: the metric itself. Intersection-over-union is the right metric for segmentation, but *how you average it across classes* determines whether it tells the truth or a comfortable lie. Let us be precise about what IoU is and then about exactly how the mean of it deceives.

For a single class $c$, IoU (the Jaccard index) is the area of overlap divided by the area of union between the predicted and true masks for that class:

$$\text{IoU}_c = \frac{|P_c \cap G_c|}{|P_c \cup G_c|} = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c},$$

where $\text{TP}_c$ is pixels correctly predicted as class $c$, $\text{FP}_c$ is pixels wrongly predicted as $c$, and $\text{FN}_c$ is class-$c$ pixels the model missed. IoU is in $[0, 1]$; it is 1 only when the predicted and true regions coincide exactly, and it is harsh on both false positives and false negatives, which is why it is preferred over pixel accuracy. Dice (the F1 of segmentation) is the closely related $\text{Dice}_c = \frac{2\,\text{TP}_c}{2\,\text{TP}_c + \text{FP}_c + \text{FN}_c}$, always slightly higher than IoU for the same prediction.

![A graph showing predictions splitting into a high background IoU and a low rare-class IoU, both feeding a naive mean IoU, with a separate path that drops background to reach an honest mean](/imgs/blogs/debugging-segmentation-6.png)

### The science: the mean is dominated by the easy class

**Mean IoU** is the average of the per-class IoUs: $\text{mIoU} = \frac{1}{C}\sum_{c=1}^{C}\text{IoU}_c$. Notice the structure: it is an *unweighted* average over classes, and it includes background as one of the classes. This is where the lie lives. Background in a typical scene is enormous and easy — it might be 90% of all pixels, with a clean boundary the model nails — so $\text{IoU}_{\text{bg}} \approx 0.98$. A rare class might be 2% of pixels, hard, and the model misses it, so $\text{IoU}_{\text{rare}} \approx 0.10$. Suppose four classes: background 0.98, two easy classes at 0.85 and 0.80, and the rare class at 0.10. The mean is

$$\text{mIoU} = \frac{0.98 + 0.85 + 0.80 + 0.10}{4} = \frac{2.73}{4} = 0.6825.$$

That is already misleading — a 0.68 that includes a class scoring 0.10 — but it gets worse with more easy classes. The real failure is the *micro* or pixel-weighted variants and the inclusion of a giant background. If instead you (accidentally) compute a *pixel-weighted* mean IoU, background's 90% pixel share gives it a 0.9 weight, and $0.9 \times 0.98 + 0.1 \times (\text{the rest}) \approx 0.88 + \text{crumbs} = 0.89$. The rare class, the entire point of the project, contributes essentially nothing to the headline. **Background dominance inflates the score precisely because the metric rewards the class that is easiest and largest, and that class drowns out the classes you actually care about.**

The honest computation excludes background and the ignore label and reports a *per-class* table alongside the mean. Excluding background from the four-class example: $\text{mIoU}_{\text{fg}} = \frac{0.85 + 0.80 + 0.10}{3} = 0.583$, and — more importantly than the single number — the per-class table screams `rare: 0.10` at you. The number you ship is not the one with background in it; the number you ship is the per-class table, and the foreground mean as a summary.

There is a subtlety worth pinning down: the choice between *macro* (unweighted per-class) averaging and *micro* (pixel-weighted) averaging is not a style preference, it changes which question the metric answers. Macro mean IoU — the standard for segmentation benchmarks like Cityscapes — gives every class equal vote regardless of size, so a one-pixel class and a million-pixel class both count once; this is what you want when small classes matter, because it refuses to let the giant class dominate. Micro IoU pools all classes' true positives, false positives, and false negatives into one global ratio before dividing, which weights each class by its pixel count and therefore *guarantees* the background swamps the result. The single most common "my IoU looks too good" bug is silently computing micro when you meant macro: a torchmetrics `JaccardIndex` with `average="micro"` on a background-heavy dataset will report a number five to fifteen points higher than the macro mean and effectively measure "did you get the background right," which you always do. When you see two IoU numbers that disagree by ten points, the first suspect is macro-versus-micro, and the fix is to standardize on macro with background excluded and only ever quote micro as a secondary, clearly labeled figure.

A second subtlety: even macro IoU can flatter you if you average over a single image instead of accumulating intersection and union across the whole dataset. Per-image IoU averaged over images is *not* the same as dataset-level IoU, because a class absent from one image but present in another creates undefined or zero per-image terms that bias the average. The correct accumulation maintains running intersection and union *counters per class over the entire evaluation set* and divides only at the end — exactly what torchmetrics does when you `update` across batches and `compute` once. Computing IoU per batch and averaging the batch IoUs is a real bug that produces a number that wobbles with batch composition and disagrees with the benchmark; accumulate the counters, divide once.

![A before-and-after contrasting a mean IoU of 0.85 that counts background against an honest per-class report that drops background and void to expose the failing rare class](/imgs/blogs/debugging-segmentation-7.png)

### The diagnostic: compute per-class IoU, excluding background and ignore

Here is the reusable metric. It excludes the ignore label, optionally excludes background, and *always returns the per-class breakdown* so a single inflated class can never hide a failing one.

```python
import torch

def per_class_iou(preds, targets, num_classes, ignore_index=255,
                  exclude_background=True):
    """preds, targets: [B, H, W] long class-id tensors.
    Returns (per_class_iou_tensor, mean_iou_over_kept_classes).
    Excludes ignore_index pixels; optionally drops class 0 (background)."""
    valid = targets != ignore_index            # mask out void pixels
    preds = preds[valid]
    targets = targets[valid]

    ious = torch.full((num_classes,), float("nan"))
    for c in range(num_classes):
        pred_c = preds == c
        true_c = targets == c
        inter = (pred_c & true_c).sum().item()
        union = (pred_c | true_c).sum().item()
        if union > 0:                           # class present in this eval
            ious[c] = inter / union
        # union == 0 -> class absent -> leave NaN, do not count it

    start = 1 if exclude_background else 0       # drop background class 0
    kept = ious[start:]
    mean_iou = torch.nanmean(kept).item()       # NaN classes excluded
    return ious, mean_iou
```

Three design choices in that function are each a bug fixed: masking `ignore_index` *before* counting (void never enters TP/FP/FN); leaving absent classes as `NaN` and using `nanmean` (a class with no pixels in this batch does not contribute a spurious 0 or 1); and `exclude_background` to drop class 0 from the mean. The same logic is available in torchmetrics with `average="none"` to get the per-class tensor:

```python
from torchmetrics import JaccardIndex

# average="none" returns per-class IoU; average="macro" returns the
# unweighted mean. ALWAYS look at "none" first.
metric = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES,
                      ignore_index=255, average="none")
metric.update(preds, targets)
per_class = metric.compute()           # tensor[NUM_CLASSES]
foreground_miou = per_class[1:].nanmean()   # drop background, average rest
print({c: round(v.item(), 3) for c, v in enumerate(per_class)})
```

The discipline is simple and non-negotiable: **never report a single mean IoU without the per-class table next to it.** The mean is a summary; the table is the truth. The moment someone shows you a 0.85 with no breakdown, the first question is "what does the smallest class score?" — and nine times out of ten that is where the project is actually failing.

#### Worked example: 0.85 to 0.42, and the project that was secretly failing

This is the run from the opening. The dashboard read 0.85 mean IoU over four classes: background, road, building, and pole (the rare 2%-of-pixels class the project existed to detect). We swapped the metric for `per_class_iou` with `exclude_background=True` and printed the table: background 0.98, road 0.84, building 0.79, pole **0.05**. The naive mean *with* background was indeed 0.85; the honest foreground mean was $(0.84 + 0.79 + 0.05)/3 = 0.56$, and the number that actually mattered — pole IoU — was 0.05. The model had essentially never learned the class the whole effort was for, and a metric that counted a 0.98 background had been broadcasting "success" for three weeks. Fixing the metric did not fix the model, but it *redirected* the model: we added a Dice + weighted cross-entropy loss for the rare class (section 7), and watched pole IoU climb from 0.05 to 0.38 over the next two epochs while the foreground mean rose from 0.56 to 0.67 — a real, honest improvement we could only see *because the metric had stopped lying*.

| Averaging scheme | Background 0.98 | Easy classes | Rare class 0.10 | Reported number | Honest? |
| --- | --- | --- | --- | --- | --- |
| Naive mean (incl. bg) | counted | 0.85, 0.80 | counted | 0.6825 | inflated by bg |
| Pixel-weighted (micro) | weight 0.9 | small weight | weight 0.02 | ~0.89 | rare class invisible |
| Foreground macro | excluded | 0.85, 0.80 | counted | 0.583 | honest, rare class visible |
| Per-class table | 0.98 | 0.85, 0.80 | **0.10** | the table | the truth |

## 7. Loss choice for imbalance: when CE alone cannot find the small class

The metric being honest exposes the next problem: a rare class the model never learns. This is not a bug per se — it is the natural consequence of extreme class imbalance under cross-entropy, and understanding *why* tells you which loss to reach for. Plain cross-entropy minimizes the average per-pixel loss, and when 90% of pixels are background, the gradient is overwhelmingly a background gradient. The 2% of rare-class pixels contribute 2% of the loss terms; their gradient is a whisper against the background's roar, and the easiest way for the optimizer to reduce average loss is to predict background everywhere and eat the small penalty on the rare pixels. The model converges to a confident background-predictor with a beautiful loss curve and a 0.05 rare-class IoU.

Three loss tools address this, each with a different mechanism. **Weighted cross-entropy** multiplies each class's loss term by a weight $w_c$, typically inversely proportional to class frequency, so the rare class's gradient is scaled up to compete: a class at 2% frequency might get weight ~10× the background's, restoring balance to the gradient sum. **Focal loss** reshapes the per-pixel loss to down-weight *easy, well-classified* pixels regardless of class: it multiplies cross-entropy by $(1 - \hat{p}_y)^\gamma$, so a pixel the model already predicts confidently ($\hat{p}_y \to 1$) contributes almost nothing and the gradient concentrates on the hard pixels — which are disproportionately the rare class and the boundaries. **Dice loss** optimizes the overlap metric directly, $1 - \text{Dice}$, and because Dice is a *ratio* it is inherently scale-invariant to class size — a small class and a large class both range over $[0, 1]$, so the rare class is not drowned by pixel count.

The focal mechanism deserves a precise statement because the $\gamma$ exponent is doing real work. Standard cross-entropy for a pixel is $-\log \hat{p}_y$, and its gradient with respect to the logit is the familiar residual; the problem under imbalance is that a vast number of *easy* background pixels, each with a small but nonzero loss $-\log \hat{p}_y \approx -\log 0.99 \approx 0.01$, sum to a large aggregate gradient simply because there are so many of them. Focal loss multiplies each pixel's term by $(1 - \hat{p}_y)^\gamma$ with $\gamma$ typically 2. For an easy background pixel with $\hat{p}_y = 0.99$, the modulating factor is $(1 - 0.99)^2 = 0.0001$, shrinking that pixel's loss by ten-thousand-fold; for a hard rare-class pixel with $\hat{p}_y = 0.2$, the factor is $(0.8)^2 = 0.64$, barely reducing it. The aggregate gradient is thus *re-weighted away from the easy majority and toward the hard minority by the model's own confidence*, dynamically, without needing to know class frequencies in advance. This is why focal loss handles imbalance even within a class (easy versus hard pixels of the same class), where static class weights cannot. The trade-off is that focal loss can suppress the gradient too aggressively early in training when *everything* is "hard," so a brief warmup on plain cross-entropy before switching to focal is a common, defensible recipe.

Dice loss has its own quiet failure mode worth naming: it is unstable when a class is *absent* from a batch. If the rare class has zero pixels in the current batch, both its intersection and its union (before the epsilon) are zero, and the Dice term degenerates — the epsilon in the numerator and denominator is what keeps it finite, but the gradient signal for that class in that batch is meaningless. With extreme imbalance and small batches, the rare class is absent from most batches, so its Dice gradient is sporadic and noisy. This is the technical reason the combined Dice + cross-entropy recipe wins: cross-entropy provides a stable, dense, per-pixel gradient on every batch (including a useful gradient even when the rare class is absent, by penalizing false positives), while Dice provides the scale-invariant overlap signal on the batches where the class is present. Each covers the other's blind spot.

In practice the strongest recipe for imbalanced segmentation is a *combination*: Dice loss (for the scale-invariant overlap signal) plus weighted or focal cross-entropy (for stable per-pixel gradients and calibration), summed. Here is a compact, correct implementation that respects `ignore_index` — the single most-skipped detail, because a naive Dice implementation will happily include void pixels and corrupt the overlap.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    """Dice + weighted CE for imbalanced segmentation, ignore-aware.
    logits: [B,C,H,W]. target: [B,H,W] long with ignore_index for void."""
    def __init__(self, num_classes, ignore_index=255, ce_weight=None,
                 dice_w=1.0, ce_w=1.0, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=ce_weight,
                                      ignore_index=ignore_index)
        self.dice_w, self.ce_w, self.eps = dice_w, ce_w, eps

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)

        # Dice needs probabilities and a one-hot target with void masked.
        probs = F.softmax(logits, dim=1)                  # [B,C,H,W]
        valid = (target != self.ignore_index)             # [B,H,W]
        t = target.clone()
        t[~valid] = 0                                     # placeholder
        onehot = F.one_hot(t, self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1).float()                # [B,1,H,W]
        probs = probs * valid                             # zero void pixels
        onehot = onehot * valid

        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        dice_loss = 1 - dice.mean()

        return self.ce_w * ce_loss + self.dice_w * dice_loss
```

The two lines that matter most for correctness are `valid = (target != self.ignore_index)` and the masking `probs = probs * valid`. Without them, Dice computes overlap over void pixels — pixels with no ground truth — and the resulting gradient pushes the model to predict *something* on regions that should be excluded, reintroducing the exact bug section 4 fixed. A loss that handles imbalance but reintroduces void training is a net loss; the ignore-awareness is not optional.

#### Worked example: pole IoU from 0.05 to 0.38

Continuing the four-class road scene: with the metric honest, the pole class sat at 0.05 IoU under plain `CrossEntropyLoss`. We computed class frequencies on the training set — background 0.90, road 0.06, building 0.02, pole 0.02 — and set cross-entropy weights inversely (normalized): roughly `[0.3, 1.0, 2.5, 2.5]`, then wrapped it in `DiceCELoss` with equal Dice and CE weighting. The first thing the instruments showed was the *loss went up* in absolute terms, from 0.18 to 0.31 — expected and correct, because the model was now being penalized properly for missing the rare class instead of getting a free pass. Over two epochs the pole IoU climbed 0.05 → 0.19 → 0.38, road and building held steady (0.84, 0.79), and the honest foreground mean IoU rose from 0.56 to 0.67. The loss being *higher* and the honest metric being *higher* at the same time is the signature of a correctly rebalanced objective: you stopped optimizing the easy average and started optimizing the thing you cared about.

## 8. Putting it together: the five-minute segmentation audit

Every bug in this post is caught by a fixed sequence of cheap checks you run *before* you trust a single number, in an order that bisects the failure space. The order matters: alignment first (because a misaligned mask poisons everything downstream), then unique-value audit (catching interpolation and void-encoding bugs), then `ignore_index` confirmation (loss and metric), then the per-class IoU (catching the metric lie). Run these five in order and you have localized the bug to one of four causes before you have spent a GPU-minute.

The four silent bugs map cleanly onto a symptom, a single cheap check, and a one-line fix, and it is worth holding the whole table in your head as one object — because in practice the symptoms overlap (a low IoU could be any of them) and the *check* is what disambiguates, not the symptom. A boundary that comes out as garbage could be misalignment *or* bilinear interpolation; the overlay distinguishes the first and the unique-value print distinguishes the second. A suspiciously high mean IoU could be a counted background *or* a fake void class scoring high; the per-class table exposes both. The discipline is to let each check rule exactly one cause in or out, so that after four checks you have a single suspect, not a vague worry.

![A matrix mapping four segmentation bugs to their instrument symptom, the cheap check that confirms each, and the one-line fix that resolves it](/imgs/blogs/debugging-segmentation-3.png)

There is a deeper reason this particular audit is so effective, and it is the through-line of the whole series: **dense prediction multiplies the number of places a bug can hide but does not multiply the number of *instruments* you need.** A classifier has one prediction per image; a segmentation model has hundreds of thousands, and intuitively that should make debugging hundreds of thousands of times harder. It does not, because almost every segmentation bug is *spatially structured* — it lives at boundaries, or in one rare class, or in the void regions — and structure is exactly what the eye and a histogram catch instantly. The overlay turns a boundary bug into a visible fringe; the unique-value print turns an interpolation bug into an illegal integer; the per-class table turns an imbalance bug into a single small number. You are not inspecting hundreds of thousands of predictions; you are reading the *shape* of the error, and the shape points at the cause.

```python
def segmentation_audit(loader, model, num_classes, ignore_index=255):
    """The five-minute pre-flight: alignment, values, ignore, metric.
    Run on a handful of batches before trusting any training number."""
    images, masks = next(iter(loader))

    # 1) ALIGNMENT: overlay at full + low res (visual; see check_alignment).
    check_alignment(images[0].permute(1, 2, 0).numpy().astype("uint8"),
                    masks[0].numpy(), num_classes)

    # 2) VALUES: only legal ids present? catches bilinear + wrong void.
    audit_mask_values(loader, expected_ids=set(range(num_classes)),
                      ignore_value=ignore_index)

    # 3) IGNORE: loss and metric both honor the void label.
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    assert criterion.ignore_index == ignore_index

    # 4) SHAPES: index target, long dtype, in-range (first batch).
    logits = model(images)
    assert_seg_shapes(logits, masks, num_classes)

    # 5) METRIC: per-class, excluding background + void. The truth.
    preds = logits.argmax(1)
    ious, miou = per_class_iou(preds, masks, num_classes,
                               ignore_index=ignore_index,
                               exclude_background=True)
    print(f"per-class IoU: {ious.tolist()}")
    print(f"honest foreground mIoU: {miou:.3f}")
```

This is `make-it-fail-small` applied to dense prediction: instead of staring at a loss curve over a whole dataset, you take one batch, look at it five ways, and let each check rule a class of bug in or out. If the overlay is misaligned, stop and fix the pipeline before anything else. If the unique values contain illegal ids, you have bilinear-on-mask. If a value carries 12% of pixels and is not in your legal set, that is your void label and you must ignore it. If the per-class table has a class at 0.05, you have an imbalance problem, not a metric bug. The audit does not fix anything — it *localizes*, which is the whole game.

## Case studies and real signatures

These patterns are not hypothetical; they recur across published work and production systems, and naming them makes them findable.

**Cityscapes and the 19-of-34 remap.** The Cityscapes dataset ships masks with 34 raw label ids, of which only 19 are used for the standard benchmark; the rest (and an explicit "unlabeled" id) are mapped to the ignore label 255. Nearly every public training script for Cityscapes includes an explicit `labelIds → trainIds` lookup table for exactly this reason — train on the raw 34 ids and you both crash (id 34 has no channel) and pollute the metric with classes the benchmark does not score. The official evaluation computes per-class IoU and reports the mean over the 19 classes, *excluding* the void regions — the honest computation built into the benchmark. When someone reports a Cityscapes mIoU wildly above the state of the art, the first suspect is a remap or ignore mismatch, not a breakthrough.

**The ADE20K and Pascal VOC 255 convention.** Pascal VOC segmentation masks use 255 as the boundary/void label, and the standard practice — baked into the PyTorch `VOCSegmentation` ecosystem — is `ignore_index=255` in the loss. Forgetting it is one of the most common beginner bugs on VOC: the model trains on the white boundary outlines as if "255" were a class, the loss looks fine, and the boundaries come out fuzzy. The fix is the single argument, and it is documented precisely because so many people hit it.

**Confident learning and label errors in segmentation ground truth.** The same confident-learning methods that found pervasive label errors in ImageNet and MNIST test sets (Northcutt, Athalye, Mueller, 2021, who documented an estimated ~3.4% average label-error rate across ten benchmark *classification* test sets) extend to segmentation, where boundary annotations are even noisier — pixels near object edges are genuinely ambiguous and different annotators disagree. The practical lesson is that a segmentation IoU has a ceiling set by *annotation consistency at boundaries*, and a model stuck a few points below your target may be hitting the label-noise ceiling rather than a bug; the overlay check distinguishes "my masks are wrong" (a bug) from "the boundaries are genuinely ambiguous" (a data-quality limit).

**The U-Net and the resize-mask default.** The original U-Net (Ronneberger, Fischer, Brox, 2015) and the encoder-decoder lineage it spawned all output a dense `[B, C, H, W]` logit map, and the recurring integration bug across reimplementations is the *post-processing* resize of the predicted or target mask back to the original resolution with a default bilinear interpolation — silently averaging predicted class ids and reporting a degraded IoU that has nothing to do with the model. The fix is the same nearest-neighbor rule applied to *outputs* as well as inputs: a label map is categorical at every stage of the pipeline, from disk to loss to the resized prediction you finally save.

## When this is (and isn't) your bug

Segmentation has enough distinctive failure signatures that you can often rule it in or out by the *shape* of the symptom, before running any check.

**It is a mask/alignment bug when** the IoU caps at a frustrating ceiling no amount of training breaks, the interior of objects is correct but boundaries are noisy or hedged, and the loss is healthy. Boundary-localized error is the fingerprint of misalignment or interpolation — both corrupt edges and leave interiors alone. Run the overlay-at-low-resolution check; if the mask is shifted, you have it.

**It is an interpolation bug when** `audit_mask_values` returns ids that were never in your source masks, or your loss crashes with a `device-side assert` / index-out-of-bounds that points at the loss line. Invented ids come from averaging; the unique-value print is definitive.

**It is an index/ignore bug when** the loss trains fine but predictions decode to the wrong semantic labels (the whole colormap rotated), or a giant fraction of pixels carries an unexpected value, or your "mean IoU" includes a suspiciously high class you do not recognize. The remap-and-ignore audit settles it.

**It is a metric lie when** the mean IoU is high but the model visibly fails on a class you care about. This is *not* a model bug — the model is doing exactly what the metric rewarded. The per-class table is the test; if the small class is near zero, the metric was hiding a real failure and the *next* move is loss rebalancing, not metric fixing.

**It is NOT a segmentation-specific bug when** the loss goes to NaN (that is numerics — see `hunting-nans-and-infs`), when the model cannot even overfit a single image (that is the model or optimization — run the overfit-one-batch test first), or when training and validation IoU are both honest and simply low (that is capacity, data quantity, or the genuine label-noise ceiling). A smooth-then-NaN curve is never a mask bug; a model that fails the overfit-one-batch test has a problem upstream of anything in this post. Bisect to the right place before you start overlaying masks.

## Key takeaways

- **Overlay the mask on the image at full *and* low resolution before you train.** A one-pixel misalignment is invisible at full res and unmissable at 64×64; this thirty-second check finds the spatial off-by-one that caps your IoU.
- **Masks get nearest-neighbor interpolation, always; images get bilinear.** Bilinear on a label map averages class ids into ids that describe categories that are not there — a categorical lie at every boundary. Verify with a unique-value print.
- **Print the unique values of every mask out of your dataloader.** Illegal ids mean interpolation corruption; a high-frequency unexpected value is your void label, which you had better be ignoring.
- **Set `ignore_index` in both the loss and the metric, with the same value.** It removes void terms from the loss *and* fixes the normalization; missing it in the loss trains on garbage, missing it in the metric scores against pixels the model never learned.
- **Cross-entropy wants index targets `[B, H, W]` long; Dice wants one-hot `[B, C, H, W]`.** A float one-hot handed to cross-entropy is silently read as soft labels in recent PyTorch — a finite, wrong loss. Assert the shape and dtype on batch one.
- **Never report mean IoU without the per-class table.** Background dominance inflates the mean because the metric is an unweighted average that rewards the easy, giant class; exclude background and ignore, and look at the smallest class first.
- **A high mIoU with a class at 0.05 is a metric lie, not a model success.** The fix is not the metric — it is loss rebalancing (Dice + weighted/focal CE), and the signature of success is the loss going *up* while the honest per-class IoU goes *up*.
- **Bisect before you overlay.** NaN is numerics, a failed overfit-one-batch is the model, both-honest-but-low is capacity or label noise. Segmentation-specific checks come after you have localized to data or evaluation.

## Further reading

- **"U-Net: Convolutional Networks for Biomedical Image Segmentation"** — Ronneberger, Fischer, Brox (2015). The encoder-decoder architecture and the dense `[B, C, H, W]` output that every segmentation resize bug attaches to.
- **The Cityscapes Dataset** — Cordts et al. (2016), and the official `cityscapesScripts` `labelIds → trainIds` mapping and ignore-region evaluation. The canonical example of remapping sparse labels to a contiguous range and excluding void from the metric.
- **"Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks"** — Northcutt, Athalye, Mueller (2021). The confident-learning estimate of label-error rates in standard benchmarks; the basis for the boundary-noise ceiling discussion.
- **"Focal Loss for Dense Object Detection"** — Lin et al. (2017). The $(1-\hat{p}_y)^\gamma$ down-weighting of easy pixels that rebalances extreme foreground/background imbalance.
- **The albumentations documentation on masks and `additional_targets`** — the canonical mask-aware augmentation API that locks image and mask geometry and applies nearest interpolation to masks automatically.
- **PyTorch `nn.CrossEntropyLoss` and torchmetrics `JaccardIndex` docs** — the exact semantics of `ignore_index`, `reduction`, soft targets, and `average="none"` for the per-class breakdown.
- Within this series: the master decision tree in [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) and the capstone [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook); the closely related [loss function bugs](/blog/machine-learning/debugging-training/loss-function-bugs) (reduction, `ignore_index`, logits-vs-probabilities), the sibling vision posts [debugging object detection](/blog/machine-learning/debugging-training/debugging-object-detection) and [augmentation debugging for vision](/blog/machine-learning/debugging-training/augmentation-debugging-for-vision), and the metric-honesty companion [your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying).

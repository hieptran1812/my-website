---
title: "Augmentation Debugging for Vision: When Geometry Breaks the Target"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Localize the single most common detection and segmentation bug — an augmentation that moves the image but not the box, mask, or keypoint — and fix it with lockstep coordinate math, the right interpolation, and a 30-second overlay check."
tags:
  [
    "debugging",
    "model-training",
    "computer-vision",
    "object-detection",
    "segmentation",
    "data-augmentation",
    "albumentations",
    "pytorch",
    "finetuning",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/augmentation-debugging-for-vision-1.png"
---

A detection model has been training for two days. The loss is going down. The training curve looks textbook — smooth, monotone, no spikes. And the mean average precision is stuck at 0.31 and will not move, epoch after epoch, no matter how long the run goes or how much you tune the learning rate. Someone suggests a bigger backbone. Someone else suggests a longer schedule. Both are wrong, and both will waste another two days. The model is learning exactly what you are teaching it. The problem is that on a fifth of every batch, you are teaching it that the object is on the *left* when the augmented image clearly shows it on the *right*.

This is the single most common bug in object detection and segmentation, and it is so common precisely because it is silent. A classification augmentation that breaks the label tends to be visible — a flipped 6 labeled 6, a color-shifted red car labeled red — and we covered that general failure mode in [augmentation gone wrong](/blog/machine-learning/debugging-training/augmentation-gone-wrong). But in detection and segmentation the label is not a scalar, it is a *geometric object*: a bounding box, a polygon mask, a set of keypoints, all expressed in pixel coordinates. When you flip, rotate, scale, or crop the image, the pixels move — and if the coordinate-valued target does not move with them, in exactly the same way, by exactly the same math, you have silently desynchronized the supervision signal. The image says one thing; the target says another. The model averages over the contradiction and lands on a confused floor.

This post is about that class of bug and its cousins, all the way down to the coordinate algebra. By the end you will be able to take a stalled detector or a segmentation model whose intersection-over-union refuses to climb, prove in thirty seconds whether augmentation is the culprit, localize *which* augmentation stage broke the target, and fix it with the correct lockstep transform, the correct interpolation, and the correct label bookkeeping. Figure 1 shows the whole bug in one frame: the image flips, the box does not, and mAP sits on a floor — versus the fix, where the box flips in lockstep and mAP recovers.

![Before and after comparison showing that flipping the image without flipping the bounding box leaves mAP stuck at 0.31, while flipping the box in lockstep with the formula x prime equals width minus x recovers mAP to 0.58.](/imgs/blogs/augmentation-debugging-for-vision-1.png)

In the language of this series, this is a **data** bug — it lives in the input pipeline — that masquerades as an **optimization** or **model** bug, because the symptom (a metric stuck on a floor while loss descends) is exactly what a too-small model or a too-low learning rate would also produce. It sits squarely in the [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs): the instinct points you at the model; the bisection points you at the data pipeline. We will run that bisection the disciplined way. The master move is the one from [the overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test): toggle the suspect off and watch the metric. If turning augmentation off lets the model fit, augmentation is your bug, and you go hunting for which transform broke the target.

## 1. The symptom: a metric on a floor while loss descends

Let us be precise about the signature, because recognizing it is half the battle. You have four instruments and they disagree in an informative way.

- **Training loss** descends smoothly. This is the trap. The model is genuinely minimizing the loss it is given. The loss it is given is partly nonsense, so a smooth descent does not mean a correct descent.
- **The task metric** — mAP for detection, mean IoU for segmentation, PCK for keypoints — climbs to some unremarkable value and then stops. Not zero. Not great. A floor. For a detector that should hit 0.55 on your data, you see 0.30–0.35 and a plateau.
- **The train–val gap** is *small or absent*. This is the tell that distinguishes this bug from ordinary overfitting. If your boxes are scrambled identically in train and val (a worse but real variant), both metrics are bad together. If your boxes are scrambled only in train (the common case, because val often has augmentation off), then the model is being taught contradictory geometry during training and never gets a clean gradient, so even the *training* metric — not just val — stalls.
- **The loss does not spike or NaN.** This is numerically clean. There is no exploding gradient, no infinity, no divergence. The bug is in the *content* of the target, not its magnitude. A smooth-then-NaN curve is a numerics story, covered in [hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs); a smooth-loss-but-floored-metric curve is a target-correctness story, which is this post.

Here is the mechanism in one sentence, and the rest of the post unpacks it: **for the fraction of training examples where the geometric augmentation moved the image but not the box, the loss gradient pushes the model toward a target that is wrong, and the only loss-minimizing response to systematically-wrong targets is to predict the population average — a blurry, low-confidence, location-agnostic detector that lands on a floor.**

### Why a floor, and why exactly there

This deserves the scientific treatment, because the height of the floor is predictable and that prediction is itself a diagnostic. Suppose a fraction $p$ of your training images have a horizontal flip applied to the image but *not* to the box. On those images, the regression target for box location is wrong by the full flip displacement, which for an object near the edge is most of the image width. The localization loss — smooth L1 or IoU loss — on those examples cannot be driven to zero by any function, because the target itself is inconsistent with the input. The network sees the same visual feature (an object at the right edge) mapped sometimes to "box on the right" (clean examples) and sometimes to "box on the left" (broken examples). The expected gradient is a weighted average of "move the box right" and "move the box left," and the loss-minimizing prediction is the weighted centroid — roughly the image center, with high variance and low confidence.

Quantitatively: if clean examples have localization loss $\approx 0$ at convergence and broken examples have an irreducible localization loss $L_{\text{broken}}$ (the loss of predicting the centroid against a target that is sometimes left, sometimes right), then the training loss floor is approximately

$$ L_{\text{floor}} \approx (1-p)\cdot 0 + p \cdot L_{\text{broken}} = p\, L_{\text{broken}}. $$

You cannot read $p$ directly off the loss, but you *can* read it off the mAP: a detector that confidently localizes the $(1-p)$ clean fraction and produces garbage boxes on the $p$ broken fraction will have a precision/recall curve whose area is dragged down roughly in proportion to $p$. The headline intuition — **the floor height scales with the fraction of broken targets** — is what tells you this is a per-example target-corruption bug and not a model-capacity bug, because a capacity bug has no reason to produce a *fractional* floor that moves when you change your flip probability. If you halve your flip probability and the floor rises (fewer broken examples, but also fewer correct ones if you broke them all) or moves at all, you have just confirmed the augmentation hypothesis with a single experiment.

## 2. The coordinate math: how a box transforms under each augmentation

To fix this bug you have to know what the *correct* transform is, so let me derive the coordinate rules. We will use the convention that a box is `(x1, y1, x2, y2)` in absolute pixel coordinates with the origin at the top-left, $x$ increasing rightward, $y$ increasing downward — the standard image convention. Figure 2 shows the dataflow: the source box passes through the same flip, scale, and crop math as the image, then gets clipped to the frame and dropped if it has fallen below the visibility threshold.

![Graph showing a source box flowing in parallel through horizontal flip x prime equals width minus x and resize x prime equals x times scale, then through crop x prime equals x minus offset, then clip to frame and a visibility drop branch, merging into a valid in-range target.](/imgs/blogs/augmentation-debugging-for-vision-2.png)

**Horizontal flip.** The image of width $W$ is mirrored: pixel column $x$ moves to column $W - 1 - x$ (or $W - x$ in continuous coordinates). A box's left edge becomes its right edge and vice versa, so

$$ x_1' = W - x_2, \qquad x_2' = W - x_1, \qquad y' = y. $$

Note the swap: you cannot just apply $x' = W - x$ to both corners and keep them as `(x1, x2)`, because then $x_1' > x_2'$ and your box is inside-out. The single most common manual-flip bug is forgetting the corner swap, which produces a *negative-width* box that downstream code either silently clips to zero area (the box vanishes) or treats as a valid tiny box at the wrong place.

**Vertical flip.** Symmetric to horizontal, on the $y$ axis: $y_1' = H - y_2$, $y_2' = H - y_1$. Almost always a *bad idea* for natural scenes — more on that in the label-destroying-transforms section — but the math is the same.

**Resize / scale.** If you resize the image by factors $s_x = W'/W$ and $s_y = H'/H$, every coordinate scales: $x' = x\, s_x$, $y' = y\, s_y$. Box width and height scale too. Easy to get right, easy to forget when you resize the image array with `cv2.resize` and leave the box in old-image pixels — now the box is in a coordinate frame that no longer exists.

**Crop.** Take a crop with top-left corner at $(x_{\text{off}}, y_{\text{off}})$ of size $(W_c, H_c)$. Coordinates shift by the offset: $x' = x - x_{\text{off}}$, $y' = y - y_{\text{off}}$. Then you must **clip** to the crop frame, $x' \in [0, W_c]$, because part of the box may now be outside the crop. And you must decide a **visibility policy**: if the visible fraction of the box after clipping falls below a threshold (say 0.1), the object is mostly gone and you should *drop the box* — keeping a box that bounds a sliver of an object teaches the model to fire on slivers, and keeping the original full box when only a corner is visible teaches it to hallucinate.

**Rotation.** This is where manual code most often goes wrong, because a rotated axis-aligned box is *not* an axis-aligned box. Rotating the four corners of the box by angle $\theta$ about the image center gives four rotated points; the new axis-aligned box must be the *bounding box of those four rotated corners*:

$$ \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x - c_x \\ y - c_y \end{bmatrix} + \begin{bmatrix} c_x \\ c_y \end{bmatrix}, $$

applied to all four corners, then $x_1' = \min(x_i')$, $x_2' = \max(x_i')$, and likewise for $y$. People who rotate only the box's two stored corners — instead of all four — get a wrong, too-small box, because the extremes of a rotated rectangle are at corners the original `(x1,y1,x2,y2)` representation does not even store. This is the subtle one, and it is the reason you should almost never hand-roll rotation for boxes.

**The general principle.** Every geometric augmentation is an affine (or in elastic cases, non-affine) coordinate transform $T$. The image is $T$ applied to pixels; the box must be $T$ applied to *its coordinates*, with the post-steps of bounding (for non-axis-preserving $T$), clipping, and visibility dropping. **The bug is always the same shape: $T$ applied to pixels, identity applied to the target.** Masks and keypoints follow the same principle — a mask is $T$ applied to a per-pixel label image (with the crucial interpolation caveat in section 5), a keypoint is just $T$ applied to its $(x, y)$ with an in-frame visibility flag.

#### Worked example: the flip that cost 27 points of mAP

A team finetuned a Faster R-CNN on a 12,000-image custom dataset. Their training transform applied `RandomHorizontalFlip(p=0.5)` to the image tensor inside the dataset's `__getitem__`, using `torchvision.transforms.functional.hflip(img)`. The boxes lived in a separate `target["boxes"]` tensor and were never touched. So half the training images — the flipped half — had boxes pointing at the mirror-image location of every object.

The numbers: validation mAP@0.5 plateaued at **0.31** after 40 epochs and would not move. The training localization loss bottomed out at a suspiciously nonzero **0.74** (a clean run on this data converges to roughly 0.15). When they applied the floor formula with $p = 0.5$ (half the images flipped, all of them broken) and the empirical $L_{\text{broken}} \approx 1.4$ for centroid-vs-mirrored targets, the predicted floor was $0.5 \times 1.4 = 0.70$ — within a hair of the observed 0.74, confirming the diagnosis before they changed a line of code.

The fix was to flip the boxes in lockstep: `x1, x2 = W - x2, W - x1` whenever the image was flipped. After the fix, the same run reached mAP@0.5 of **0.58** at epoch 40 and training localization loss of 0.16. A **27-point mAP gain** from four lines of coordinate algebra, with zero change to the model, the learning rate, or the schedule. This is the payoff of bisecting to the data pipeline instead of buying a bigger backbone.

## 3. The definitive diagnostic: overlay the transformed target on the augmented image

I will give you the most important tool first, because it catches more of these bugs than every other technique in this post combined, and it takes thirty seconds. **Render the augmented image with the augmented boxes (or mask, or keypoints) drawn on top, and look at it with your own eyes.** If a box has drifted off the object, you will see it instantly. If the box is inside-out, you will see a degenerate rectangle. If the mask edges turned to fractional mush, you will see the gray halo. This is the vision-specific instance of the series-wide discipline from [look at your data before you train](/blog/machine-learning/debugging-training/look-at-your-data-before-you-train): the cheapest, highest-yield debugging action in machine learning is to *actually look* — and in detection and segmentation, looking means overlaying the target, not just viewing the image.

Here is a reusable overlay utility. It takes a sample straight out of your augmented dataset — after every transform has run — and draws the boxes the model will actually be trained against. Run it on 16 random samples and you will find the bug or rule it out.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def overlay_boxes(image, boxes, labels=None, denormalize=None, ax=None):
    """Draw post-augmentation boxes on the post-augmentation image.

    image: HxWxC uint8, or CxHxW float tensor (set denormalize).
    boxes: Nx4 array in xyxy ABSOLUTE pixel coords (the target the model sees).
    denormalize: optional (mean, std) to invert normalization for display.
    """
    img = image
    if hasattr(img, "detach"):          # a torch tensor
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3):   # CxHxW -> HxWxC
        img = np.transpose(img, (1, 2, 0))
    if denormalize is not None:         # undo normalize so colors look right
        mean, std = np.array(denormalize[0]), np.array(denormalize[1])
        img = img * std + mean
    img = np.clip(img, 0, 1) if img.max() <= 1.5 else np.clip(img, 0, 255) / 255.0

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    H, W = img.shape[:2]
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes)):
        # a healthy box is inside the frame and has positive area
        bad = (x2 <= x1) or (y2 <= y1) or x1 < -1 or y1 < -1 or x2 > W + 1 or y2 > H + 1
        color = "red" if bad else "lime"
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor=color, linewidth=2))
        if labels is not None:
            ax.text(x1, max(y1 - 4, 0), str(labels[i]), color=color, fontsize=9)
    ax.set_axis_off()
    return ax

# Pull straight from the augmented dataset and eyeball 16 samples.
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
for ax in axes.ravel():
    sample = train_dataset[np.random.randint(len(train_dataset))]
    image, target = sample          # however your dataset returns it
    overlay_boxes(image, target["boxes"], target["labels"],
                  denormalize=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ax=ax)
plt.tight_layout(); plt.savefig("aug_check.png", dpi=80)
```

The `bad` flag is doing real work: it turns a box red if it is inside-out (the flip corner-swap bug), negative (off-frame after crop without clipping), or out of range. A wall of green boxes that hug their objects means your geometry is synchronized. A single red box, or a green box floating in empty space, is the bug staring back at you. **This overlay is the ground truth of augmentation debugging.** Every other technique in this post is a way to automate or scale what this image shows you directly.

A note on the `denormalize` argument, because it catches a second bug for free: if you forget it and your pipeline normalizes images, the overlay shows a washed-out gray rectangle — which is itself the signature of the normalize-before-augment ordering bug we cover in section 6. The display being broken *is* a diagnostic. If your overlay is all gray, your color channels are off-scale, and that affects training too.

## 4. Use the library that keeps them in lockstep — and see the manual pitfall

The reason this bug is so common is that the naive code structure *invites* it: the image and the target are separate variables, and it is one line to transform the image and zero lines to transform the target. The robust fix is to never let those two variables drift apart — and the cleanest way to guarantee that is to use a library that transforms image and target *together*, in one call, by construction. `albumentations` is the standard for this. You declare the targets you have (`bboxes`, `mask`, `keypoints`) once, and every spatial transform applies the matching coordinate math to all of them in lockstep.

Here is a correct detection pipeline. The crucial pieces are `bbox_params` (which declares the box format and the label fields), and the `min_visibility` / `min_area` policy (which implements the drop-the-mostly-cropped-box rule from section 2).

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_tf = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(height=512, width=512, scale=(0.5, 1.0), p=1.0),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, border_mode=0, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.02, p=0.5),   # color ops on uint8, BEFORE normalize
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),           # normalize LAST
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",        # xyxy absolute; use "coco" for xywh, "yolo" for normalized cxcywh
        label_fields=["class_labels"],
        min_area=16,                # drop boxes smaller than 16 px^2 after transform
        min_visibility=0.1,         # drop boxes with <10% of original area left after crop
    ),
)

# Validation: ONLY resize + normalize. No geometry, no color. Deterministic.
val_tf = A.Compose(
    [
        A.Resize(height=512, width=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
)

def __getitem__(self, i):
    image = self.load_image(i)               # HxWxC uint8 RGB
    boxes = self.load_boxes(i)               # list of [x1,y1,x2,y2]
    labels = self.load_labels(i)             # list of ints
    out = self.tf(image=image, bboxes=boxes, class_labels=labels)
    image = out["image"]                     # CxHxW float tensor, normalized
    target = {
        "boxes": torch.as_tensor(out["bboxes"], dtype=torch.float32).reshape(-1, 4),
        "labels": torch.as_tensor(out["class_labels"], dtype=torch.int64),
    }
    return image, target
```

Two things make this correct that the manual version got wrong. First, `bboxes=boxes` is passed *into the same call* that transforms the image, so albumentations applies the identical flip, crop, rotate, and scale to the box coordinates — the lockstep is structural, not something you can forget. Second, `min_visibility` and `min_area` automate the visibility-drop policy, so a box that the crop reduced to a sliver is removed rather than left pointing at a fragment. Note also the *order*: color jitter runs before `Normalize`, on the uint8-domain image, and `Normalize` plus `ToTensorV2` come last — the ordering rule from section 6, baked into the pipeline.

**The manual pitfall, shown explicitly.** Here is the structure that produces the bug, so you can recognize it in a codebase. The image is transformed; the target is not. It does not crash. It does not warn. It just trains on broken geometry.

```python
# THE BUG — image transformed, target silently left behind.
def __getitem__(self, i):
    image = self.load_image(i)               # PIL image
    boxes = torch.as_tensor(self.load_boxes(i))   # xyxy
    labels = torch.as_tensor(self.load_labels(i))

    if random.random() < 0.5:
        image = TF.hflip(image)              # <-- image flips
        # boxes NOT flipped. This is the 27-mAP-point bug.
    if self.do_crop:
        i_, j_, h_, w_ = T.RandomCrop.get_params(image, (512, 512))
        image = TF.crop(image, i_, j_, h_, w_)   # <-- image cropped
        # boxes NOT shifted by (-j_, -i_), NOT clipped. Targets now in old frame.

    image = TF.to_tensor(image)
    image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image, {"boxes": boxes, "labels": labels}   # boxes are wrong
```

The tell, when you read code like this, is a geometric op on `image` with no corresponding line on `boxes` (or `mask`, or `keypoints`) within the same `if` block. Make it a code-review reflex: **every spatial op on the image must have a matching spatial op on the target, or it must go through a library that does both.** If you must hand-roll (some research pipelines do), the minimal correct flip is the corner-swap from section 2, and you should still run the overlay check on the output.

## 5. Masks: the interpolation trap that corrupts class ids

Segmentation has its own version of this bug, and it is even subtler because the mask *does* get transformed — just with the wrong interpolation. A segmentation mask is an integer label image: pixel value 0 is background, 1 is "person," 2 is "car," and so on. When you resize the image you also resize the mask, and the obvious thing — using the same bilinear interpolation you use for the image — is catastrophically wrong. Figure 4 shows why.

![Before and after comparison showing that bilinear resize blends segmentation mask ids 0, 1, 2 into fractional values like 0.5 and 1.7 creating phantom classes and IoU 0.42, while nearest-neighbor resize keeps the ids as exact integers and reaches IoU 0.71.](/imgs/blogs/augmentation-debugging-for-vision-4.png)

**Why bilinear destroys a mask.** Bilinear interpolation computes each output pixel as a weighted average of its neighbors. For a natural image this is exactly what you want — it produces smooth gradients. For a label image it is nonsense: averaging the *id* of "person" (1) and the *id* of "car" (2) gives 1.5, which is neither person nor car. At every boundary between two classes, bilinear invents a band of fractional ids. When you then cast that float mask back to integer, 1.5 rounds to 2 ("car") along the entire person/car boundary, or — worse — if class 3 happens to be a real class, your averaging of 2 and 4 produces phantom-class-3 pixels that correspond to no object at all. The boundaries of every object get a halo of wrong labels, the model is supervised toward those wrong labels, and IoU sits on a floor it cannot cross because the *targets themselves* have noisy boundaries.

**The fix is one word: nearest.** Nearest-neighbor interpolation picks the single closest source pixel's value, so an output id is always an exact, real id that existed in the input. No averaging, no fractional classes, no phantom ids. The rule is absolute: **resize images with bilinear (or bicubic), resize masks with nearest, always.** The same goes for any geometric warp — rotation, affine, perspective — applied to a mask: the mask must use nearest-neighbor sampling.

In albumentations, this is handled correctly by default when you pass `mask=...` — spatial transforms use nearest for masks automatically. But if you set interpolation flags manually, or use a raw `cv2.resize`, you must get it right:

```python
import cv2
import numpy as np

# WRONG — bilinear blends class ids into phantom classes
mask_bad = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
print("unique ids after bilinear:", np.unique(mask_bad))
# -> [0 1 1 2 2 3 ...] with phantom/intermediate values after casting

# RIGHT — nearest keeps ids exact
mask_ok = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
print("unique ids after nearest:", np.unique(mask_ok))
# -> [0 1 2] exactly the classes that existed

# The assert that catches it permanently: no new ids may appear after a transform.
def assert_mask_ids_preserved(mask_before, mask_after):
    before, after = set(np.unique(mask_before)), set(np.unique(mask_after))
    new_ids = after - before
    assert not new_ids, f"mask transform invented class ids {new_ids} (bilinear on a mask?)"

assert_mask_ids_preserved(mask, mask_ok)   # passes
# assert_mask_ids_preserved(mask, mask_bad)  # raises: invented ids
```

The `assert_mask_ids_preserved` check is your permanent guardrail. Put it in your dataset's `__getitem__` during development (and in a unit test forever): the set of class ids after any spatial transform must be a subset of the ids before it. If a transform *invents* an id, you are interpolating a label image with something other than nearest, and you have found the bug. This is the segmentation analog of the box overlay — a cheap, decisive check that turns a silent corruption into a loud failure.

There is a related representation trap worth flagging, because it produces the same boundary-noise symptom from a different cause: **one-hot masks resized as if they were images.** If your pipeline represents the mask as a one-hot tensor of shape `(num_classes, H, W)` — a stack of binary channels — and you resize *that* with bilinear interpolation, each channel's binary edge becomes a soft ramp, and after you argmax back to integer ids the boundary class is decided by whichever channel's ramp happens to be higher, which is interpolation noise. The fix is the same in spirit (resize before one-hot, with nearest, on the integer id map) but the bug hides differently, because `np.unique` on the *one-hot* tensor only ever shows values in $[0, 1]$, so the `assert_mask_ids_preserved` check on the one-hot form will not catch it — you must run the check on the *integer-id* form, after argmax. The rule generalizes: **always validate the integer label map, the representation the loss actually consumes via `ignore_index` and cross-entropy, not an intermediate one-hot or float view of it.** A subtlety like this is exactly why the overlay (render the final integer mask in distinct colors and look at the boundaries) remains the ground truth — it inspects the same representation the loss sees, with no chance of validating the wrong intermediate tensor. The `ignore_index` interaction deserves its own note: if your mask uses a sentinel value like 255 for "ignore" (the void/unlabeled class), that sentinel must survive the resize *exactly* — a bilinear blend of 254 and 255 produces 254.5, which rounds to a real class id and silently turns ignored pixels into supervised ones, injecting label noise precisely at the object boundaries where the ignore region usually lives. Nearest interpolation preserves the 255 sentinel; bilinear destroys it. This is covered further in the sibling [debugging segmentation](/blog/machine-learning/debugging-training/debugging-segmentation) post, but the augmentation-specific takeaway is that the interpolation choice is not just about class blending — it is about preserving the `ignore_index` contract that your loss depends on.

#### Worked example: the IoU that bilinear capped at 0.42

A semantic-segmentation model on a 5-class dataset (background plus four object classes) was stuck at mean IoU **0.42** after a long run, with the per-class breakdown showing every class equally mediocre and boundary regions visibly noisy in the predictions. The training augmentation resized images and masks to 512×512 using a shared `A.Resize(512, 512, interpolation=cv2.INTER_LINEAR)` — and crucially, the engineer had passed that same `interpolation` flag through to the mask, overriding albumentations' nearest default. Running `np.unique` on a transformed mask revealed ids `[0, 1, 2, 3, 4, 5, 6, 7]` — *eight* values on a *five*-class problem. The three extra ids were interpolation artifacts at class boundaries.

The fix: use `A.Resize(512, 512)` with the default (nearest applied to masks), or explicitly set `mask_interpolation=cv2.INTER_NEAREST`. After the fix, `np.unique` returned exactly `[0, 1, 2, 3, 4]`, and the same model reached mean IoU **0.71** — a **29-point IoU gain** with no model change. The boundary noise vanished because the boundaries in the *targets* were now crisp. The lesson generalizes: when a metric is floored and the per-class breakdown is *uniformly* mediocre with noisy boundaries, suspect a mask-corruption bug before you suspect the model — uniform mediocrity is the signature of corrupted supervision, not of a weak class.

## 6. Normalize before vs after augment: the ordering bug

The order of operations in an augmentation pipeline is not cosmetic; it determines what numeric range your color and geometric ops see, and getting it wrong silently changes their meaning. Figure 5 shows the correct order as a stack: decode to uint8, run geometric ops (with target sync), run color ops on uint8, *then* convert to float and normalize last.

![Stack diagram showing the correct augmentation order from top to bottom: decode uint8 0 to 255, geometric ops with box sync, color jitter on uint8, convert to float 0 to 1, normalize with mean and std last, and a danger node showing a color op placed after normalize breaks the scale.](/imgs/blogs/augmentation-debugging-for-vision-5.png)

**Why color ops must run before normalization.** Normalization maps pixel values from $[0, 255]$ (or $[0, 1]$) to roughly $[-2, 2]$ by subtracting a mean and dividing by a standard deviation: $x_{\text{norm}} = (x - \mu)/\sigma$. A brightness augmentation that multiplies pixel values by a factor, or adds a constant, assumes it is operating on a meaningful scale — multiplying a $[0, 255]$ image by 1.2 brightens it; multiplying a normalized $[-2, 2]$ image by 1.2 scales values around zero, which brightens *and darkens* depending on sign and is not a brightness change at all. A "contrast" op that scales around the mean assumes the mean is at the image's true midpoint, which is true in pixel space and false after normalization shifted everything. **Color augmentations are defined in pixel space; run them there.** The rule: geometric ops and color/photometric ops on the uint8 (or $[0,1]$ float) image, *then* normalize, *then* to tensor.

**Why geometric ops are happier on uint8 too.** A subtler reason to keep the float conversion late: many augmentation libraries and OpenCV operations are faster and exactly correct on uint8, and some interpolation paths behave differently on float vs integer arrays. More importantly, doing geometry first and normalization last means your overlay-debug visualization (section 3) can simply un-normalize and display — if you normalize first and then crop, your debug view has to invert a normalization that may have been applied with the wrong stats, compounding confusion. Keep the human-viewable representation alive as long as possible.

**The uint8-vs-float overflow trap.** If you *do* run a color op on a uint8 array, watch for overflow: `image * 1.5` on a `uint8` array wraps around (250 becomes a small number, not 255), producing bizarre speckled images. Either cast to a wider type before scaling and clip back, or let the library handle it. Albumentations and torchvision's transforms handle the clamping internally, which is another argument for using them rather than hand-rolling `image * factor` on a raw array. The signature of an overflow bug is salt-and-pepper noise in bright regions — pixels that should be near-white wrapping around to near-black. If your overlay shows speckled highlights, suspect uint8 overflow in a color op.

A concrete ordering check you can drop into a test:

```python
# Sanity-check the pipeline order by inspecting the dtype and range at each stage.
# (Pseudocode-ish, but the asserts are the point.)
img_u8 = load_image(i)                          # uint8, range [0, 255]
assert img_u8.dtype == np.uint8

img_geo = geometric_and_color(img_u8)           # still uint8 after geo + color
assert img_geo.dtype == np.uint8, "color/geo ran AFTER float conversion?"
assert img_geo.max() <= 255 and img_geo.min() >= 0, "uint8 overflow in a color op"

img_norm = normalize(to_float(img_geo))         # normalize LAST
assert -5 < img_norm.min() and img_norm.max() < 5, "normalized range looks wrong"
```

If the second assert fires — color or geometric work happening on a float, already-normalized tensor — your order is inverted, and your augmentations are operating on a scale they were not designed for. This is a quieter bug than the box-not-flipped one (it degrades rather than floors), but it is real, and it compounds: a 1–3 point accuracy loss that you would never localize without checking the order.

## 7. Augmentation applied to validation, and the TTA skew

Here is a bug that does not floor the metric — it *inflates* or *destabilizes* it, which is arguably worse because it makes a broken model look fine, or a fine model look broken, depending on direction. **Augmentation must run on training data only. Validation and test must see only the deterministic preprocessing the model will face at inference: resize and normalize, nothing else.** Figure 3's bottom row captures the signature: val scoring below train by a suspicious margin, confirmed by dumping the val transform list.

![Matrix mapping four augmentation bugs to their symptom, confirming test, and fix: boxes drift maps to mAP floor 0.31 confirmed by overlaying the box and fixed by lockstep transform, mask noisy maps to IoU stuck confirmed by printing unique ids and fixed by nearest resize, mixup hurts maps to accuracy dropping 2.4 points confirmed by asserting lambda sums to 1 and fixed by mixing labels by the same lambda, and val too low maps to val below train by 9 points confirmed by dumping the val transform list and fixed by val being only resize plus normalize.](/imgs/blogs/augmentation-debugging-for-vision-3.png)

**Why a randomly-augmented val set lies.** If your validation transform includes a random flip or random crop, every time you evaluate you score a *different* random version of the val set. Your val metric becomes noisy run-to-run, which masks real improvements and makes early-stopping and model-selection decisions on noise. Worse, if the augmentation is *destructive* (a crop that removes objects, a flip that breaks an orientation-dependent class), your val metric is systematically depressed and you conclude the model is worse than it is — you might kill a good run. And if the val augmentation happens to make the task *easier* (some test-time augmentations do), you over-estimate the model and ship something that underperforms in production. Either way, **a non-deterministic val pipeline is an instrument that lies**, and you cannot debug a model while your measuring stick wobbles. This connects directly to [your metric is lying](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs)-style failures: the metric is fine, the data feeding it is not.

The fix is a hard assertion that the val transform contains no stochastic spatial or photometric ops:

```python
import albumentations as A

ALLOWED_VAL_TRANSFORMS = (A.Resize, A.Normalize, A.CenterCrop,
                          A.PadIfNeeded, A.LongestMaxSize, A.SmallestMaxSize)

def assert_deterministic_val(transform):
    """Val/test must be resize + normalize only — nothing random, nothing geometric."""
    for t in transform.transforms:
        assert isinstance(t, ALLOWED_VAL_TRANSFORMS) or t.__class__.__name__ == "ToTensorV2", \
            f"val transform contains '{t.__class__.__name__}' — augmentation must be train-only"
        p = getattr(t, "p", 0.0)
        assert p in (0.0, 1.0), f"val transform '{t.__class__.__name__}' has random p={p}"

assert_deterministic_val(val_tf)   # run at dataloader construction time
```

That assertion belongs in your test suite or at dataloader-construction time. It is a few lines and it permanently closes a bug that otherwise re-emerges every time someone refactors the transforms and copy-pastes the train pipeline into the val slot. This is the same print-and-assert discipline from [the input pipeline is lying to you](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you), specialized to the train/val transform boundary.

**Test-time augmentation done right.** TTA — averaging predictions over flips and scales at inference — is a legitimate technique that can add a point or two, but it has its own failure mode for detection and segmentation: **if you flip the image for TTA, you must flip the predictions back before merging.** A detector run on a flipped image returns boxes in the flipped frame; averaging those with un-flipped predictions without un-flipping them first scrambles the ensemble exactly the way the training bug scrambles supervision. The same coordinate algebra from section 2 applies in reverse. The signature of broken TTA is that turning TTA *on* makes detection metrics *worse* — a clean tell that you are merging predictions from inconsistent coordinate frames. If TTA helps for classification but hurts for detection, you forgot to un-transform the geometric predictions.

The reason classification TTA hides this bug is instructive. For a classifier, the prediction is a class probability vector that is *invariant* to a horizontal flip — there is no coordinate to un-transform, so averaging the flipped and unflipped predictions is always valid and TTA reliably helps. Detection and segmentation predictions are *equivariant*, not invariant: a flip of the input produces a flip of the output, and you must undo that output flip before merging. The mental error is to treat the equivariant case like the invariant one. For segmentation, the analog is just as sharp: a TTA flip requires flipping the predicted mask back before averaging the per-pixel class probabilities, and a TTA scale requires resizing the predicted mask back to the original resolution with — again — the right interpolation (bilinear on the probability maps, but argmax to integer ids only at the very end). Mixing scales in segmentation TTA without resizing the probability maps to a common resolution first produces a shape mismatch that, if silently broadcast or cropped, blends predictions from different spatial frames.

#### Worked example: the TTA that cost three points of mAP

A detection pipeline added horizontal-flip TTA at evaluation, expecting the usual small boost, and instead saw mAP@0.5 *drop* from **0.561 to 0.528** — a three-point regression from a technique that is supposed to help. The bug: the flipped-image inference returned boxes in the flipped coordinate frame, and the merge step concatenated those boxes with the unflipped boxes and ran non-maximum suppression over the union — without first applying $x_1, x_2 \leftarrow W - x_2, W - x_1$ to the flipped boxes. So for every object, the ensemble contained one correct box and one mirror-image box, and NMS, seeing two distant boxes, kept both — doubling the false positives and tanking precision. The fix was four lines: un-flip the flipped-frame boxes before the merge. After: mAP@0.5 **0.574**, a genuine 1.3-point gain over the no-TTA baseline. The diagnostic that pinned it was, once again, the overlay — rendering the *merged* boxes on the original image showed the phantom mirror boxes floating on the opposite side of every object, the exact picture of an un-undone flip.

## 8. Mixup and cutmix: the label-mixing bookkeeping

Mixup and cutmix are augmentations that combine *two* images, and their bug surface is the label-mixing arithmetic. The principle is simple and the implementations get it wrong constantly. Figure 7 shows the bookkeeping as a timeline: sample one $\lambda$, blend pixels by $\lambda$, weight the loss by the *same* $\lambda$, assert the weights sum to one.

![Timeline showing correct mixup lambda bookkeeping: sample lambda equals 0.3, blend pixels as 0.3 times a plus 0.7 times b, compute loss as 0.3 times loss a plus 0.7 times loss b, assert the weights sum to 1.0, and a danger event showing a hardcoded 0.5 weight that mismatches the pixel blend.](/imgs/blogs/augmentation-debugging-for-vision-7.png)

**The mixup formula.** Sample $\lambda \sim \text{Beta}(\alpha, \alpha)$. Blend two images and their labels with the *same* $\lambda$:

$$ \tilde{x} = \lambda x_a + (1-\lambda) x_b, \qquad \tilde{y} = \lambda y_a + (1-\lambda) y_b. $$

Because the labels are one-hot (or you mix the losses, which is equivalent), the training loss for a mixed sample is

$$ L = \lambda \, \ell(f(\tilde{x}), y_a) + (1-\lambda)\, \ell(f(\tilde{x}), y_b), $$

where $\ell$ is your usual cross-entropy. **The single lambda that blended the pixels must weight the losses.** The most common bug is computing $\lambda$ for the pixel blend and then using a different (or hardcoded) weight for the loss — for example, blending pixels with $\lambda = 0.3$ but averaging the two losses 50/50. Now the supervision says "this image is 50% cat, 50% dog" while the pixels are "30% cat, 70% dog." The model is taught a label that does not match the input, a softer version of the box-not-flipped bug: not a hard floor, but a steady 1–3 point accuracy *drag* that you would never localize without auditing the lambda bookkeeping.

**The cutmix region-fraction bug.** Cutmix is the same idea with a spatial twist: instead of alpha-blending, you cut a rectangular patch from image B and paste it onto image A. The label mix fraction must equal the *area fraction* of the pasted patch, not a separately-sampled $\lambda$:

$$ \lambda_{\text{label}} = 1 - \frac{\text{area of pasted patch}}{H \cdot W}. $$

If you sample $\lambda$ from the Beta distribution, use it to *size* the patch, and then use the *sampled* $\lambda$ (rather than the *realized* area fraction) for the label, you have a mismatch — because integer rounding of the patch dimensions and clipping at image edges mean the realized area fraction differs from the sampled $\lambda$. The fix is to recompute $\lambda$ from the actual pasted pixels after clipping. Here is correct mixup and the area-corrected cutmix:

```python
import numpy as np
import torch

def mixup(x, y, alpha=0.2):
    """Correct mixup: ONE lambda blends pixels AND weights the loss."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, lam            # return lam so the loss uses the SAME value

def mixup_loss(criterion, logits, y_a, y_b, lam):
    # the SAME lam that blended pixels weights the two losses
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

def cutmix(x, y, alpha=1.0):
    """Correct cutmix: lambda for the loss is the REALIZED area fraction after clipping."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    r = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * r), int(H * r)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
    y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    # RECOMPUTE lambda from the patch that was ACTUALLY pasted (after clipping)
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y, y[idx], lam

# the bookkeeping guardrail: the two loss weights must sum to 1.0
x_mix, y_a, y_b, lam = mixup(x, y)
assert abs(lam + (1 - lam) - 1.0) < 1e-6
loss = mixup_loss(criterion, model(x_mix), y_a, y_b, lam)
```

The `lam = 1.0 - ...` recomputation in `cutmix` is the line people skip, and it is the difference between a label that matches the pixels and one that is off by the rounding-and-clipping error. The guardrail `assert abs(lam + (1 - lam) - 1.0) < 1e-6` looks trivial, but its real purpose is to remind you that *one* lambda flows from the blend to the loss; the moment you see two different lambdas in your mixup code, you have the bug.

**Mixup on detection is its own trap.** Naively alpha-blending two detection images and concatenating their box lists is what most detection mixup implementations do, and it can work — but only if you keep *both* sets of boxes with weights $\lambda$ and $1-\lambda$ on their respective losses, exactly as the classification case. Blending two images and keeping only one image's boxes (or summing the losses with equal weight) reintroduces the label-mismatch bug at the detection level. And blending images that are different sizes without first resizing them to a common canvas produces a torn composite with boxes in two different coordinate frames. If you are not sure your detection mixup is correct, the overlay check from section 3 — run on the *mixed* image with *both* box sets drawn in two colors — will show you immediately whether the boxes from each source still bound their objects in the composite.

## 9. Label-destroying transforms: when the augmentation is wrong for the task

Some augmentations are coded perfectly — image and target transform in lockstep, interpolation correct, order right — and *still* break training, because the transform is semantically wrong for the task. These are the trickiest because every mechanical check passes; the bug is conceptual.

- **Vertical flip on natural scenes.** Gravity is a real feature. Cars do not appear upside-down on roads, faces are not inverted, text is not flipped. A `VerticalFlip` on a natural-image detector teaches the model that upside-down objects are normal, wasting capacity on a distribution that never occurs at inference and degrading performance on the upright distribution that does. The mechanical lockstep is fine; the semantics are wrong. (Vertical flip *is* fine for overhead/satellite imagery and some microscopy, where there is no canonical up — so the rule is task-dependent, which is exactly why it is a conceptual bug.)
- **Heavy color jitter on color-dependent classes.** If your task is "classify the traffic light state" or "is this fruit ripe," color *is* the signal. Strong hue jitter rotates red into green, which for a ripeness classifier or a traffic-light detector is not augmentation — it is label corruption. The image still has a box around the light; the box is correct; but the *class* the box implies is now wrong because you changed the color that defined the class. The fix is to clamp or disable hue jitter for color-semantic tasks. The mechanism is worth making precise, because the magnitude of the corruption is predictable. Hue jitter operates in HSV space, adding an offset $\Delta h$ to the hue angle, which is a *rotation* around the color wheel: a hue shift of $\Delta h \approx 0.33$ (a third of the wheel) maps red ($h \approx 0$) to green ($h \approx 0.33$). For a traffic-light classifier where the decision boundary between "red" and "green" is *the hue itself*, a hue-jitter range of $\pm 0.1$ already moves a meaningful fraction of red pixels across the perceptual boundary toward orange, and $\pm 0.2$ routinely crosses it. The corruption fraction scales with the jitter range relative to the inter-class hue distance — so the same `hue=0.2` setting that is harmless on ImageNet (where class identity rarely hinges on exact hue) is catastrophic on a color-semantic task. The diagnostic is to ask, per task, "could this jitter range move a pixel across a class boundary," and the answer is a property of your label definition, not of the augmentation code. This is why a color-jitter setting copied from a generic recipe is one of the most common silent label-corruptors on specialized datasets.
- **Rotation on orientation tasks.** If you are predicting orientation — "which way is this arrow pointing," "what is the rotation of this part" — then rotating the image *changes the answer*, and unless you also update the orientation label by the rotation angle, you have broken the target. This is the keypoint/pose analog: rotating a pose by 30 degrees requires rotating every keypoint *and* any angle-valued label. Most pipelines remember the keypoints and forget the scalar angle label.
- **Aggressive crops on small-object datasets.** A crop policy with `scale=(0.08, 1.0)` (the ImageNet default) is fine for classification but disastrous for a dataset of small, sparse objects, because most crops will contain *no object at all* — producing an empty target. We treat empty targets next, but note the conceptual point: the right augmentation strength is task-dependent, and a policy copied from a classification recipe is a frequent source of silent harm in detection.

The diagnostic for all of these is the same and it is not code — it is the overlay grid from section 3, viewed by a human who knows the task. When you look at 16 augmented samples with their labels and ask "is this label still *true* for this image," the vertical-flip-on-a-road and the red-turned-green and the rotated-arrow jump out, because a human reading the label against the pixels sees the contradiction that no assert can encode. This is why "look at your data" is not a beginner's crutch but a permanent practice: the conceptual augmentation bugs are exactly the ones that pass every mechanical test.

## 10. Empty targets and the min-visibility threshold

A crop that removes the object entirely leaves an image with *no* boxes and *no* foreground mask — an empty target. This is not automatically a bug (a few empty/background images can be useful negatives), but it becomes one in two ways, and both are common.

**The crash or skip.** Many detection losses and collate functions choke on a sample with zero boxes — an empty tensor of shape `(0, 4)` hits a code path that assumes at least one box, and either crashes or, worse, gets silently filtered, biasing your effective dataset toward easy images with prominent objects. If your dataloader silently drops empty-target samples, your training distribution drifts away from the hard, small-object cases, and your model underperforms on exactly the examples that matter. The signature is a discrepancy between your stated dataset size and the number of samples per epoch — count them.

**The wrong-visibility policy.** When a crop leaves only a fraction of an object, you face a choice, and getting it wrong corrupts the target. Keep the *original full box* when only a corner is visible, and you teach the model to predict a box extending into pixels that are no longer there — it learns to hallucinate object extent. Keep a box around a 5%-visible sliver, and you teach it to fire on slivers, hurting precision. The principled fix is the `min_visibility` threshold from section 4: compute the fraction of the original box area that survives the crop, and *drop* the box (and only the box, keeping the image) if it falls below a threshold like 0.1–0.3. The threshold is a real hyperparameter — too low and you keep slivers, too high and you discard partially-occluded objects the model should learn — but having *a* policy beats the default of accidentally keeping garbage.

```python
def filter_boxes_after_crop(boxes, orig_areas, crop_w, crop_h, min_visibility=0.1):
    """Clip boxes to the crop and drop those whose visible fraction is too small."""
    keep = []
    for box, orig_area in zip(boxes, orig_areas):
        x1, y1, x2, y2 = box
        cx1, cy1 = max(x1, 0), max(y1, 0)              # clip to crop frame
        cx2, cy2 = min(x2, crop_w), min(y2, crop_h)
        vis_area = max(cx2 - cx1, 0) * max(cy2 - cy1, 0)
        if orig_area > 0 and (vis_area / orig_area) >= min_visibility:
            keep.append([cx1, cy1, cx2, cy2])          # keep the CLIPPED box
        # else: object mostly cropped out -> drop the box, keep the image as a hard negative
    return keep
```

The decisive check is an assertion that, after augmentation, every retained box is *valid* — inside the frame, positive area, and (if your task forbids empty targets) that the sample has at least one box:

```python
def assert_targets_valid(image, boxes, allow_empty=True):
    H, W = image.shape[-2:]
    if not allow_empty:
        assert len(boxes) > 0, "augmentation produced an empty target on a foreground image"
    for x1, y1, x2, y2 in boxes:
        assert x2 > x1 and y2 > y1, f"degenerate box ({x1},{y1},{x2},{y2}) — inside-out or zero area?"
        assert -1 <= x1 and x2 <= W + 1, f"box x out of frame [0,{W}]: ({x1},{x2})"
        assert -1 <= y1 and y2 <= H + 1, f"box y out of frame [0,{H}]: ({y1},{y2})"
```

Run `assert_targets_valid` inside `__getitem__` during development. It catches the inside-out flip (degenerate box), the unclipped crop (out of frame), and the empty target — three distinct augmentation bugs — with one cheap check per sample. The assertions cost microseconds and they convert silent target corruption into a stack trace that points at the exact augmented sample, which is the difference between fixing the bug in a minute and chasing a floored metric for a week.

## 11. Keypoints and pose: the same coordinate math, plus visibility flags

Keypoints — facial landmarks, human-pose joints, object corners — are the third coordinate-valued target, and they fail in exactly the way boxes and masks do, with one extra wrinkle. A keypoint is just an $(x, y)$ pair (sometimes with a visibility flag $v \in \{0, 1, 2\}$ meaning absent / occluded / visible), so every geometric augmentation must apply the same coordinate transform from section 2 to each keypoint: flip moves $x$ to $W - x$, crop shifts by the offset, rotation applies the rotation matrix. The wrinkle is that flips can *swap the meaning* of symmetric keypoints, and forgetting that swap is a distinct, insidious bug.

**The left/right swap under horizontal flip.** Consider a human-pose model with keypoints `left_wrist` and `right_wrist`. When you horizontally flip a person, the pixel that was the left wrist is now on the right side of the image — so it is now visually the *right* wrist. If you flip the coordinates but keep the *labels* `left_wrist`/`right_wrist` attached to the same array indices, you have just taught the model that the person's right wrist is on their left side. The model sees a mirror-image person and is told their handedness is preserved, which is geometrically false. The fix is a *flip index map*: after a horizontal flip, you must also swap the array positions of every left/right symmetric pair, so `left_wrist` and `right_wrist` exchange slots. This is the keypoint analog of the box corner-swap — a flip is not complete until you have reassigned the symmetric labels.

```python
# COCO-17 left/right symmetric pairs (indices into the 17-keypoint array).
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

def hflip_keypoints(kpts, W, flip_pairs=FLIP_PAIRS):
    """kpts: (K, 3) array of (x, y, v). Flip x AND swap symmetric left/right joints."""
    kpts = kpts.copy()
    kpts[:, 0] = W - kpts[:, 0]                 # mirror x for every keypoint
    for a, b in flip_pairs:                      # then swap the meaning of L/R pairs
        kpts[[a, b]] = kpts[[b, a]]
    return kpts
```

Albumentations handles the coordinate flip when you pass `keypoints=`, but the *symmetric-pair swap is application-specific* and you must supply it (via the `keypoint_params` `label_fields` plus your own post-step, or by reordering after the transform). The bug signature is a pose model that predicts plausible-looking skeletons but systematically confuses left and right limbs — and it is invisible in the loss, because a left/right-swapped target still has the same *number* of keypoints at sensible *locations*; only the *identity* is wrong. The overlay catches it instantly if you color left joints blue and right joints red: a correctly-augmented flip shows the colors mirrored with the person; a broken one shows blue joints on the anatomical right.

**The out-of-frame visibility update.** The second keypoint wrinkle: a crop can move a keypoint outside the crop frame. Unlike a box (which you clip to the frame), a keypoint that leaves the frame should have its visibility flag set to "absent" — keeping it at a clamped edge position teaches the model that the joint is at the image boundary, which is false. The rule: after any crop or geometric transform, recompute each keypoint's in-frame status and update its visibility flag; do not clamp coordinates and pretend the point is still present. This is the keypoint version of the min-visibility drop policy for boxes.

#### Worked example: the pose model that mirrored its own skeleton

A 2D human-pose model finetuned on a sports dataset reported a respectable mean per-joint position error on the *aggregate* metric but failed badly on a left/right-limb-discrimination evaluation: it confused left and right ankles on roughly 40% of flipped-orientation poses. The training pipeline applied a horizontal flip to images and keypoint coordinates (via `hflip` on both) but never swapped the symmetric pairs. So on the ~50% of training examples that were flipped, the model was taught that a person facing left has their "left" ankle on the screen-right — directly contradicting the unflipped 50%. The model split the difference and learned a weak, confused notion of laterality. Adding the `FLIP_PAIRS` swap from the snippet above dropped the left/right confusion rate from **40% to under 4%**, with no change to the aggregate position error — because the *positions* were always right; only the *labels* on the flipped half were swapped. The lesson: an augmentation can be coordinate-correct and still semantically wrong if the transform changes the meaning of symmetric labels, and only a label-aware overlay reveals it.

## 12. The full bisection: localizing the augmentation bug

Now let us put the pieces together into the disciplined hunt, because knowing the bug classes is not the same as *finding* which one you have. Figure 6 is the decision tree: the first cut is always "does turning augmentation off let the model fit?"

![Decision tree for whether augmentation is the bug: the root asks if mAP recovers with augmentation off, a yes branch marks augmentation as the suspect and splits into geometric boxes not synced, mask bilinear not nearest, and color jitter breaking the class, while a no branch points elsewhere to a frozen layer or learning rate of zero.](/imgs/blogs/augmentation-debugging-for-vision-6.png)

**Step 1 — bisect augmentation in or out.** Set your training transform to val's (resize + normalize, no augmentation) and train for a few hundred steps, or better, run the overfit-one-batch test from [the overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) on a handful of images with augmentation off. If the model now drives the loss down and the metric on those same images climbs, augmentation is your suspect. If it *still* floors with augmentation off, stop blaming augmentation — your bug is more fundamental (a frozen backbone, a learning rate of zero, a loss reduction mistake from [loss function bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs)) and augmentation is a red herring. This single experiment is the most important step; it converts a vague "the model won't train" into a localized "the augmentation pipeline corrupts the target."

**Step 2 — look, with the overlay.** Once augmentation is the suspect, render the overlay grid from section 3 on augmented training samples. For detection, draw the boxes; for segmentation, render the mask alongside the image; for keypoints, plot the points. Nine times out of ten the bug is visible here: boxes off objects (geometric desync), gray washed-out images (normalize-before-color), noisy mask boundaries (bilinear-on-mask), red inside-out boxes (flip without corner swap). The overlay turns an abstract metric floor into a picture of the actual broken supervision.

**Step 3 — bisect by transform.** If the overlay is ambiguous, disable transforms one at a time (or, faster, binary-search: disable half, see if the floor lifts, recurse). The first transform whose removal lifts the floor is the culprit. For a pipeline of flip, crop, rotate, color, and normalize, three bisection rounds isolate the bad one. This is the make-it-fail-small discipline applied to the transform list.

**Step 4 — confirm with an assert, then fix.** Once you suspect a specific transform, add the matching assertion — `assert_targets_valid` for boxes, `assert_mask_ids_preserved` for masks, the deterministic-val check for the val pipeline — and watch it fire on the broken samples. The assertion firing is your confirming test; it proves causation, not just correlation. Then fix it (lockstep transform via albumentations, nearest interpolation, reordered pipeline) and re-run. The before→after evidence is the floor lifting.

#### Worked example: bisecting a stalled instance-segmentation run

A Mask R-CNN finetune on a custom 8-class instance-segmentation dataset stalled at mask mAP **0.22** and box mAP **0.29**, both on a floor, training loss smoothly descending. The bisection ran as follows. Step 1: retrain with the val transform (resize + normalize only). The overfit-one-batch test on 4 images now reached near-zero loss and near-perfect mask IoU on those images — **augmentation confirmed as the suspect**, model and optimizer exonerated. Step 2: the overlay grid showed boxes correctly placed but mask edges with a faint gray fringe — a *mask* problem, not a box problem. Step 3: disabling `RandomResizedCrop` did not lift the floor; disabling the mask's resize interpolation (which had been set to `INTER_LINEAR`) did. Step 4: `assert_mask_ids_preserved` fired immediately, reporting invented ids `{8, 9}` on an 8-class problem. The fix was a one-character change — `INTER_LINEAR` to `INTER_NEAREST` for masks. After-fix: mask mAP **0.51**, box mAP **0.54**. The whole bisection took under an hour because each step *ruled out* a region of the six-places map rather than guessing. That is the entire method: localize before you touch code.

## 13. Before and after: the evidence table

Here is the consolidated before→after across the bug classes in this post, with the symptom, the confirming test, the fix, and the measured recovery. Every row is a concrete metric movement, not a vibe. Figure 8 shows the detection case as a before-after panel.

![Before and after comparison showing a detection run where manually transformed boxes not synced on crop leave 20 percent of targets off the object and mAP plateaus at 0.31 after 40 epochs, while an albumentations pipeline with bbox params keeps boxes in lockstep so targets stay in frame and mAP climbs to 0.58 at 40 epochs.](/imgs/blogs/augmentation-debugging-for-vision-8.png)

| Bug | Symptom (before) | Confirming test | Fix | After |
| --- | --- | --- | --- | --- |
| Box not flipped/cropped in lockstep | mAP@0.5 floor 0.31; loc loss 0.74 | overlay box on aug image; `assert_targets_valid` | transform box in lockstep (albumentations `bbox_params`) | mAP@0.5 0.58; loc loss 0.16 |
| Mask resized bilinear | mean IoU 0.42; noisy boundaries | `np.unique(mask)` shows extra ids; `assert_mask_ids_preserved` | nearest interpolation for masks | mean IoU 0.71; crisp boundaries |
| Mixup lambda mismatch | top-1 down ~2.4 pts; soft target wrong | assert two loss weights sum to 1.0 | one lambda blends pixels and weights loss | top-1 restored; mixup helps |
| Cutmix area mismatch | small steady accuracy drag | recompute lambda from pasted area | lambda = realized area fraction | mixup/cutmix gain recovered |
| Augmentation on val | val noisy / depressed; bad model selection | `assert_deterministic_val` | val = resize + normalize only | stable val; honest selection |
| Normalize before color | 1–3 pt accuracy degrade | dtype/range assert at each stage | color/geo on uint8, normalize last | degrade removed |
| Empty/sliver targets | dropped samples or hallucinated extent | count samples/epoch; visibility check | `min_visibility` drop policy | training distribution intact |

A word on **how to measure these honestly**, because a before→after is only evidence if the comparison is clean. Hold *everything else fixed* — same seed, same model, same schedule, same val set — and change only the augmentation code. Re-run both arms. If the floor lifts when and only when you fix the augmentation, and the lift is reproducible across seeds, you have causal evidence, not a lucky run. The reproducibility discipline from [reproducibility and determinism in training](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) is what makes a before→after table trustworthy: without a fixed seed, a 5-point mAP move could be run-to-run variance rather than your fix.

## 14. Case studies and real signatures

These patterns are not hypothetical; they recur across the open-source detection and segmentation ecosystem, and recognizing the named signatures saves you the rediscovery.

**The COCO-format box-coordinate swap.** COCO stores boxes as `[x, y, width, height]` (xywh), Pascal VOC as `[x1, y1, x2, y2]` (xyxy), and YOLO as normalized `[cx, cy, w, h]`. The single most common detection data bug *adjacent* to augmentation is feeding boxes in one format to a transform or loss expecting another — albumentations' `BboxParams(format=...)` exists precisely because this swap is so common. A model trained on xywh boxes interpreted as xyxy will have every box's "width" treated as a far-right coordinate, placing targets wildly off — the same floored-metric signature as the augmentation desync, with the same overlay-the-box cure. This is covered in depth in the sibling post [debugging object detection](/blog/machine-learning/debugging-training/debugging-object-detection); for augmentation specifically, the lesson is to declare your box format to the augmentation library and let it handle the conversion, never to assume.

**Albumentations as the de facto fix.** The albumentations library (Buslaev et al., 2020, *Information*) became the standard detection/segmentation augmentation tool in large part *because* it solves the lockstep problem by design: declaring `bbox_params` and passing `mask=`/`keypoints=` means every spatial transform applies the matching coordinate math automatically, with correct nearest-neighbor interpolation for masks as the default. The historical context is that a generation of detection bugs came from hand-rolled augmentation in research codebases where image and target were separate; the library's central contribution is making the desync structurally impossible. When you adopt it, you are not just getting transforms — you are getting the lockstep guarantee for free.

**The mixup paper's exact formula.** Mixup (Zhang et al., 2018, *mixup: Beyond Empirical Risk Minimization*, ICLR) specifies the construction precisely: $\tilde{x} = \lambda x_i + (1-\lambda) x_j$ and $\tilde{y} = \lambda y_i + (1-\lambda) y_j$ with the *same* $\lambda \sim \text{Beta}(\alpha, \alpha)$. CutMix (Yun et al., 2019, *CutMix*, ICCV) specifies that the label mix ratio equals the cropped *area* ratio. The bugs in section 8 are all deviations from these published formulas — a different lambda for pixels and labels, or a sampled rather than realized area fraction. The fix is to implement the formula as written, and the canonical reference implementations in `timm` (Wightman's PyTorch Image Models) get the bookkeeping right, which is why borrowing `timm`'s `Mixup` class is safer than re-deriving it.

**Confident-learning-style label audits, applied to geometry.** The confident-learning line of work (Northcutt et al., 2021, *Confident Learning*, JAIR; the `cleanlab` library) found that even gold-standard test sets like ImageNet and MNIST contain measurable label errors — on the order of a few percent. The geometric analog in detection/segmentation is that *augmentation* can inject far more than a few percent target errors silently, because a 50% flip probability with unsynced boxes corrupts *half* your training targets — an error rate an order of magnitude beyond natural label noise, self-inflicted by a pipeline bug. The discipline is the same: audit your targets, here by overlay rather than by loss-ranking, because the corruption is geometric rather than categorical.

## 15. When this is (and isn't) your bug

A decisive section, because misattributing a symptom wastes the most time. Here is when the floored-metric-with-descending-loss signature points at augmentation, and when it points elsewhere.

**It is probably an augmentation/target bug when:**

- The task metric (mAP, IoU, PCK) floors while training loss descends smoothly, *and* turning augmentation off lifts the floor (the step-1 bisection).
- The overlay shows boxes off objects, masks with noisy boundaries, or washed-out images.
- The floor *moves* when you change the augmentation probability — a fractional, augmentation-dependent floor is a target-corruption signature, not a capacity signature.
- A segmentation model is *uniformly* mediocre across classes with noisy boundaries (corrupted supervision), rather than weak on specific hard classes (a real modeling problem).

**It is probably NOT augmentation when:**

- The overfit-one-batch test fails *even with augmentation off* — then your problem is more fundamental (a frozen layer, a wrong loss reduction, a learning rate of zero), and augmentation is a red herring. Go to [your model isn't learning what you think](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs).
- The loss is smooth and then *NaNs* — that is numerics (overflow, log of zero, fp16 underflow), not target corruption. Go to [hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs).
- Train metric is great and *only* val is bad, *and* the val pipeline is clean (deterministic resize+normalize) — that is ordinary overfitting or distribution shift, not an augmentation/target bug. Though do double-check the val pipeline first, since augmentation-on-val is its own bug.
- The model trains fine but degrades in production with *identical* augmentation in both — that is distribution shift, not an augmentation fault. Go to [distribution shift train vs the real world](/blog/machine-learning/debugging-training/distribution-shift-train-vs-the-real-world).

The unifying principle: the floored-metric-with-clean-loss signature is *shared* by augmentation desync, label noise, a capacity wall, and a too-low learning rate — so you cannot diagnose from the symptom alone. The bisection (augmentation off?) and the overlay (does the target match the image?) are what *separate* augmentation from its look-alikes. Always bisect before you fix.

## 16. Key takeaways

- **A geometric augmentation must transform the target by the same math as the image.** Flip the image, flip the box: `x1, x2 = W - x2, W - x1` (with the corner swap). Crop the image, shift and clip the box. Rotate the image, rotate all four corners and re-bound. The bug is always $T$ on pixels, identity on the target.
- **Overlay the transformed target on the augmented image and look.** Boxes, masks, keypoints drawn on the post-augmentation image is the definitive, thirty-second check. It catches more augmentation bugs than every other technique combined.
- **Resize masks with NEAREST, never bilinear.** Bilinear averages class ids into phantom classes; `assert_mask_ids_preserved` (no new ids after a transform) is the permanent guardrail.
- **One lambda for mixup: it blends the pixels and weights the loss.** Two different lambdas is the bug. For cutmix, the label fraction is the *realized* pasted-area fraction after clipping, not the sampled value.
- **Augment train only; val and test see resize + normalize, deterministically.** A stochastic val pipeline is a wobbling instrument; assert it contains no random spatial or photometric ops.
- **Color/geometric ops on uint8, normalize last.** Color augmentations are defined in pixel space; running them on normalized tensors silently changes their meaning.
- **Have a min-visibility drop policy and assert targets are valid.** Cropping can produce inside-out boxes, empty targets, or slivers; `assert_targets_valid` turns silent corruption into a stack trace at the exact sample.
- **Bisect before you fix: augmentation off → does the floor lift?** If yes, augmentation is the suspect and you bisect the transform list. If no, augmentation is a red herring — go fundamental.
- **A floored metric with smoothly-descending loss is a shared signature.** Augmentation desync, label noise, capacity walls, and a dead learning rate all produce it. The overlay and the bisection separate them; the symptom alone cannot.
- **Use a library that keeps image and target in lockstep.** Albumentations' `bbox_params`/`mask`/`keypoints` makes the desync structurally impossible — borrow the guarantee instead of re-implementing the coordinate math by hand.

## 17. Further reading

- **[A taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs)** — the series' master decision tree; augmentation desync is a data bug that masquerades as optimization or model code, and this is where it sits in the six-places map.
- **[The training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook)** — the capstone; the full symptom → suspect → confirming test → fix bisection, of which the augmentation-off bisection in section 11 is one branch.
- **[Augmentation gone wrong](/blog/machine-learning/debugging-training/augmentation-gone-wrong)** — the general-purpose augmentation post (label-destroying transforms across modalities, too-strong/too-weak policies, train-only-not-eval); this post is the vision-specific, coordinate-math-deep companion.
- **[Debugging object detection](/blog/machine-learning/debugging-training/debugging-object-detection)** — coordinate-format swaps (xyxy/xywh/cxcywh), anchor/target assignment, NMS, and the mAP that lies; the box-format issues that compound with augmentation desync.
- **[Debugging segmentation](/blog/machine-learning/debugging-training/debugging-segmentation)** — mask–image misalignment, `ignore_index` off-by-one, and the nearest-only interpolation rule explored further; the segmentation sibling to this post.
- **[CV input pipeline bugs](/blog/machine-learning/debugging-training/cv-input-pipeline-bugs)** — BGR vs RGB, normalization-stat mismatch, channels-first/last, uint8-vs-float; the preprocessing layer beneath augmentation.
- **Buslaev et al., 2020, "Albumentations: Fast and Flexible Image Augmentations," *Information* 11(2)** — the library that solves the lockstep problem by design; read the `BboxParams`/`mask`/`keypoints` docs for the exact target-transform guarantees.
- **Zhang et al., 2018, "mixup: Beyond Empirical Risk Minimization," ICLR; Yun et al., 2019, "CutMix," ICCV** — the canonical label-mixing formulas; implement them as written, or borrow `timm`'s reference `Mixup` class which gets the lambda bookkeeping right.

---
title: "Debugging Object Detection: Coordinates, Anchors, NMS, and the mAP That Lies"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How to localize the silent bugs that make a detector report 0.05 mAP while it has actually learned the boxes, with box-format converters, a visualize-predictions habit, anchor and NMS diagnostics, and a GT-as-prediction eval sanity check that proves where the lie lives."
tags:
  [
    "debugging",
    "model-training",
    "object-detection",
    "computer-vision",
    "bounding-boxes",
    "non-max-suppression",
    "pytorch",
    "finetuning",
    "deep-learning",
    "evaluation",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/debugging-object-detection-1.png"
---

The loss curve was beautiful. It started near 4.0, dropped under 2.0 in the first epoch, and was sitting at 0.9 by epoch ten — classification loss, box-regression loss, objectness loss, all three descending together, no spikes, no NaN. By every instrument that worked for the classifier I trained last month, this detector was learning. Then I ran the COCO evaluator and it printed `mAP@[.5:.95] = 0.051`. On a dataset where a decent model lands around 0.40. The model had trained for nine GPU-hours, the loss said it had learned the task, and the number that decides whether the project ships said it had learned almost nothing.

That gap — a healthy loss and a near-zero mAP — is the single most common, most expensive, most *demoralizing* signature in object detection, and almost every time I have chased it down, the model was fine. The model had learned the boxes. The bug was in one of the five places a box crosses on its way from the dataset to the final number: the dataset stores boxes in one format, the model head emits them in another, the loss encodes them as deltas, non-max suppression expects a third format, and the evaluator assumes a fourth. Any disagreement at any boundary relocates the boxes silently — no exception, no shape error, just boxes that land in the wrong place — and mAP, which measures whether predicted boxes overlap ground-truth boxes, reads near zero. The loss never noticed because the loss compares the model's output to its *own* encoded targets, both of which can be wrong in the same way and still match.

Detection has more of these silent bugs than any other vision task, and the reason is structural. A classifier outputs one vector per image; you can read it. A detector outputs a few hundred boxes per image, each with four coordinates in some convention, a class index in some indexing scheme, and a score — and those numbers pass through assignment, regression encoding, decoding, suppression, and a multi-stage matching evaluator before they become a metric. Every stage is a place where the convention can flip. The figure below traces the five boundaries; the rest of this post is how to find which one broke.

![A vertical stack showing a bounding box crossing five boundaries from dataset to evaluator, each labeled with a different coordinate convention that must agree](/imgs/blogs/debugging-object-detection-1.png)

By the end you will be able to take a detector that reads 0.05 mAP with a falling loss and localize the bug in minutes: assert coordinate ranges at every boundary, **visualize predicted boxes on the actual image** (the one definitive check), overlay ground truth against predictions, run the GT-as-prediction sanity check that should return mAP near 1.0 and tells you instantly whether the lie is in the evaluator or the model, and read the specific mAP signature each bug class produces. This is a domain instance of the series' master frame — a bug hides in one of six places (data, optimization, model code, numerics, systems, evaluation) and you **bisect** to the right one before touching code — applied to the place detection bugs hide most: the boundary between coordinate conventions, and the evaluator that silently disagrees with your predictions. If you want the general decision tree, it lives in [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs); the printable capstone is [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook). Here we go deep on detection.

## The mental model: a box is four numbers plus a convention

A bounding box is four numbers. That is the whole problem. Four numbers carry no metadata about what they mean, so the same four numbers `[10, 20, 40, 60]` describe completely different rectangles depending on the convention you read them in. Get the convention wrong and the rectangle moves — silently, because the four numbers are all valid floats and no library will object.

Three conventions dominate, and you must be able to convert between them in your sleep:

- **xyxy** (also called corners): `[x1, y1, x2, y2]` — the top-left corner and the bottom-right corner. `[10, 20, 50, 80]` is a box from `(10, 20)` to `(50, 80)`, so 40 wide and 60 tall. This is what torchvision's `nms` and `box_iou` expect, and what most evaluators use internally.
- **xywh** (corner + size): `[x, y, w, h]` — the top-left corner plus width and height. `[10, 20, 40, 60]` is the *same box* as the xyxy example above. This is the COCO annotation format: when you read `bbox` out of a COCO JSON, it is xywh.
- **cxcywh** (center + size): `[cx, cy, w, h]` — the center plus width and height. The same box is `[30, 50, 40, 60]`. This is what most detection heads regress, because predicting a center offset and a log-size is numerically friendlier than predicting corners.

On top of the format, there is a second, orthogonal axis: **scale**. Coordinates are either **absolute pixels** (`cx = 30` means 30 pixels from the left) or **normalized** to `[0, 1]` (`cx = 0.234` means 23.4% of the image width). YOLO-family annotations are normalized cxcywh; COCO annotations are absolute-pixel xywh; torchvision's detection models speak absolute-pixel xyxy. So a single box can be written four ways before you even pick an indexing scheme for the class. The matrix below shows the same physical box in each convention, the symptom a swap produces, and how to convert each one to the xyxy that torchvision wants.

![A four-by-four matrix comparing xyxy, xywh, cxcywh, and normalized formats by what each stores, the same box in each, the symptom of a silent swap, and the conversion to xyxy](/imgs/blogs/debugging-object-detection-2.png)

Read that matrix carefully, because the third column — the swap symptom — is the entire diagnostic vocabulary for this post. Feed xywh numbers `[10, 20, 40, 60]` into code that expects xyxy and it reads them as corners `(10,20)`-`(40,60)`: a 30×40 box where you meant a 40×60 box, shifted and shrunk. Feed cxcywh into xyxy-expecting code and the center gets read as the top-left corner, so the box jumps up and to the left by half its size. Feed normalized coordinates (all in `[0,1]`) into pixel-expecting code and every box collapses into a one-pixel speck in the top-left corner of the image. Each swap has a *fingerprint*, and once you have visualized a few you recognize them on sight.

### Why the loss does not catch it

Why does detection breed this particular pathology so much more than classification or regression? Because detection is the only common task where the *output itself is a coordinate system*, and a coordinate system carries an interpretation that the numbers do not. A classifier's output is a probability vector — a softmax over 80 classes is unambiguous; there is no "other convention" in which `[0.9, 0.1]` means something different. A box's four numbers are different: they are a projection of a rectangle into a chosen basis, and the basis (corners vs center, pixels vs normalized) is metadata that lives in your head, your config, and your documentation, but not in the tensor. The moment two pieces of code disagree about the basis, the same tensor decodes to two different rectangles, and because both rectangles are geometrically valid, nothing complains. Detection is, in this precise sense, a task where the type system of your tensors is too weak to encode the thing that matters. The entire discipline of detection debugging is compensating for that missing type information — with asserts, with visualization, with canonicalization — so that the convention becomes something the program checks rather than something you remember.

Here is the part that traps people for days. If the format mismatch is *consistent* between the model's predictions and the targets the loss compares them against, the loss is happy. The regression loss computes something like `smooth_l1(pred_box_params, target_box_params)`. If both `pred_box_params` and `target_box_params` are encoded in the same wrong convention, the loss measures their agreement correctly and descends. The model learns to predict the encoded targets accurately. The encoding is internally consistent and *externally wrong*: when you decode the predictions back to pixels for NMS and evaluation, they land in the wrong place, but the loss never decoded anything, so it never saw the error.

This is why I keep saying *the model was fine*. The model learned the mapping from image to encoded-target, which is exactly its job. The bug lives at the boundary where you decode and hand the boxes to a stage that uses a different convention — almost always the NMS-then-evaluate boundary. The loss and the metric are measuring two different things, and the metric is the one that matches reality.

> **The first rule of detection debugging:** the loss compares the model to its own targets; the metric compares the model to the world. When they disagree, the metric is not lying — the encoding between them is.

This is the same lesson as [your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying), specialized to boxes: the evaluator can disagree with your predictions in a way that makes a good model look untrained. We will spend a whole section on the evaluator, because in detection the evaluator is itself a rich source of bugs.

## The science: what IoU and mAP actually compute

You cannot debug a number you cannot compute by hand, so before any code, let us make mAP concrete. Two definitions do all the work: **IoU**, which decides whether a predicted box "hits" a ground-truth box, and **AP**, which summarizes precision and recall across confidence thresholds. mAP is just AP averaged over classes (and, for COCO, over IoU thresholds).

### IoU: the overlap that decides everything

Intersection over union is the ratio of the area where two boxes overlap to the area they cover together:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}.$$

It ranges from 0 (no overlap) to 1 (identical boxes). The intersection of two axis-aligned boxes is itself a box: its left edge is the larger of the two left edges, its right edge is the smaller of the two right edges, and similarly for top and bottom. If the resulting width or height is negative, the boxes do not overlap and the intersection is zero. The figure below works a concrete example — a 3×3 ground-truth box and a 3×3 predicted box overlapping in a 2×2 corner — so you can see where the single number comes from.

![A three-by-three grid showing a ground-truth box and a predicted box overlapping in one cell, with the intersection area four and union fourteen giving IoU about 0.29](/imgs/blogs/debugging-object-detection-3.png)

#### Worked example: computing IoU by hand

Take ground truth `A = [0, 0, 3, 3]` in xyxy (a 3×3 box at the origin, area 9) and prediction `B = [1, 1, 4, 4]` (a 3×3 box shifted by one, area 9). The intersection's left edge is `max(0, 1) = 1`, right edge is `min(3, 4) = 3`, top is `max(0,1) = 1`, bottom is `min(3,4) = 3`. So the intersection is `[1,1,3,3]`, a 2×2 box of area 4. The union is `9 + 9 - 4 = 14`. IoU is `4 / 14 ≈ 0.286`.

Now run the *exact same boxes* through code that thinks they are xywh. `A = [0,0,3,3]` as xywh is a box from `(0,0)` to `(3,3)` — same as before by coincidence, since x and y are zero. But `B = [1,1,4,4]` as xywh is a box from `(1,1)` to `(5,5)`, area 16, not the `[1,1,4,4]` corner-box of area 9. Now the intersection is `[1,1,3,3]` (area 4) but the union is `9 + 16 - 4 = 21`, so IoU `≈ 0.19`. A format confusion in the IoU function alone moved the number from 0.286 to 0.19 — a 33% relative error — on identical inputs. Scale that across every prediction-GT pair in your evaluator and you understand how a format bug pushes mAP from 0.41 to 0.05: matches that should clear the 0.5 IoU threshold now fall below it, so they count as misses.

### From IoU to AP: precision, recall, and the threshold

mAP is built on a matching procedure. For one class and one IoU threshold (say 0.5):

1. Sort all predicted boxes of that class by confidence score, highest first.
2. Walk down the list. For each prediction, find the highest-IoU unmatched ground-truth box of the same class. If that IoU exceeds the threshold, mark the prediction a **true positive (TP)** and consume that ground-truth box. Otherwise it is a **false positive (FP)**.
3. Any ground-truth box never matched is a **false negative (FN)**.

As you walk down the sorted list, you accumulate TPs and FPs, and at each step you can compute precision $P = \frac{TP}{TP + FP}$ and recall $R = \frac{TP}{TP + FN_{\text{total}}}$ where the recall denominator is the total number of ground-truth boxes. Plotting precision against recall as the score threshold sweeps from high to low gives the **precision-recall curve**. **Average Precision (AP)** is the area under that curve:

$$\text{AP} = \int_0^1 P(R)\, dR \approx \sum_k (R_k - R_{k-1})\, P_k.$$

Pascal VOC's older definition uses an 11-point interpolation; COCO uses a 101-point interpolation and, crucially, averages AP over ten IoU thresholds from 0.50 to 0.95 in steps of 0.05 — that is what `mAP@[.5:.95]` means. **mAP** is AP averaged over all classes. So COCO's headline number is a triple average: over recall (the PR curve), over IoU thresholds, over classes.

This structure is why detection bugs have *specific signatures* rather than generic ones. A coordinate-format bug lowers every IoU, so it hits the strict thresholds (0.75, 0.9) hardest — your `mAP@.5` might survive at 0.3 while `mAP@.75` collapses to 0.05. A class-offset bug makes every match a class mismatch, so step 2 finds no same-class ground truth and *everything* becomes a false positive — `mAP@.5` and `mAP@.75` both crater to near zero together. A score-threshold-too-high bug suppresses low-confidence true positives, capping recall, so the PR curve gets truncated and AP drops uniformly. A duplicate-box (broken NMS) bug floods the list with false positives at every recall level, dragging precision down. Learn the signatures and the number tells you the suspect before you write a line of diagnostic code. We will catalog them.

#### Worked example: computing AP for one class by hand

Make the matching procedure concrete so you can reason about it under pressure. One class, three ground-truth objects, and the detector emits five boxes after NMS, which I sort by descending score. I evaluate at IoU threshold 0.5. Here is the table I build by walking the sorted list, where "matches GT?" means the prediction's best IoU with an unmatched same-class ground-truth box exceeds 0.5:

| Rank | Score | Best IoU with unmatched GT | TP/FP | Cumulative TP | Cumulative FP | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.95 | 0.82 | TP | 1 | 0 | 1.00 | 0.33 |
| 2 | 0.88 | 0.61 | TP | 2 | 0 | 1.00 | 0.67 |
| 3 | 0.71 | 0.40 | FP | 2 | 1 | 0.67 | 0.67 |
| 4 | 0.55 | 0.73 | TP | 3 | 1 | 0.75 | 1.00 |
| 5 | 0.40 | 0.20 | FP | 3 | 2 | 0.60 | 1.00 |

Read the precision and recall columns as the score threshold sweeps downward. Precision starts at 1.0 (the first two predictions both hit), dips to 0.67 when the rank-3 box is a false positive (IoU 0.40 is below 0.5 — close, but a miss), recovers to 0.75 at rank 4 which finds the third object, and ends at 0.60. Recall climbs 0.33 → 0.67 → 0.67 → 1.00 → 1.00 as true positives accumulate against the three ground-truth boxes. AP is the area under the precision-recall curve; with the VOC-style "take the maximum precision to the right of each recall level" interpolation, the precision used at recall 0.33 and 0.67 is 1.0, and at recall 1.0 it is 0.75, giving roughly $\frac{1}{3}(1.0) + \frac{1}{3}(1.0) + \frac{1}{3}(0.75) \approx 0.92$. Now watch what a coordinate bug does to this same table: every IoU drops by, say, 0.3 because the boxes are slightly off, so the rank-1 IoU 0.82 becomes 0.52 (still a TP), but rank-2's 0.61 becomes 0.31 (now an FP) and rank-4's 0.73 becomes 0.43 (now an FP). Suddenly you have one TP and four FPs: precision and recall both collapse, and AP falls from 0.92 to roughly 0.10. A *uniform* IoU degradation — exactly what a sloppy coordinate transform produces — converts most of your true positives into false positives, which is why a barely-off box format is not a small error but a catastrophic one.

This worked example is the engine under every signature in the rest of the post. When I say "a class-offset bug makes everything a false positive," I mean step 2 in this table never finds an unmatched same-class GT, so every row is FP, cumulative TP stays 0, precision is 0 at every recall, and AP is 0. When I say "NMS duplicates crash precision," I mean the sorted list fills with extra boxes that consume no new GT (each object's GT is already taken by the first hit), so every duplicate is an FP row that drags cumulative precision down. The table *is* the diagnostic; learn to trace it filling in for your bug.

## The definitive check: visualize the predictions

Before any of the format converters or evaluator surgery, internalize this: **the single most powerful detection debugging tool is drawing the boxes on the image.** Not printing coordinates — *drawing* them. A printed `[0.23, 0.42, 0.31, 0.5]` tells you nothing; the same four numbers drawn on the image instantly reveal whether the box is on the object, half the size it should be, or collapsed in the corner. I have wasted hours reasoning about coordinate transforms in my head that a thirty-second visualization would have ended. Make this your reflex: when in doubt, draw it.

Here is a reusable visualizer that overlays ground truth (green) and predictions (red) and is format-aware so you can render whatever convention you have:

```python
import torch
import torchvision
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt

def visualize(image_uint8, boxes, fmt="xyxy", color="red",
              img_wh=None, labels=None):
    """
    image_uint8: torch.uint8 tensor [3, H, W]
    boxes:       [N, 4] tensor in `fmt`
    fmt:         "xyxy" | "xywh" | "cxcywh"
    img_wh:      (W, H) if boxes are NORMALIZED in [0,1]; else None
    """
    boxes = boxes.clone().float()
    if img_wh is not None:                       # de-normalize first
        W, H = img_wh
        scale = torch.tensor([W, H, W, H], dtype=torch.float)
        boxes = boxes * scale
    xyxy = box_convert(boxes, in_fmt=fmt, out_fmt="xyxy")
    drawn = draw_bounding_boxes(image_uint8, xyxy, colors=color,
                                labels=labels, width=2)
    plt.figure(figsize=(8, 8))
    plt.imshow(drawn.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
    return xyxy            # return so you can assert on the pixel box

# overlay GT and predictions on the SAME image:
def overlay(image_uint8, gt, gt_fmt, pred, pred_fmt, **kw):
    gt_xyxy   = visualize(image_uint8, gt,   fmt=gt_fmt,   color="green", **kw)
    # draw predictions on top of the GT-annotated image
    pred_xyxy = box_convert(pred.float(), in_fmt=pred_fmt, out_fmt="xyxy")
    drawn = draw_bounding_boxes(
        draw_bounding_boxes(image_uint8, gt_xyxy, colors="green", width=2),
        pred_xyxy, colors="red", width=2)
    plt.imshow(drawn.permute(1, 2, 0)); plt.axis("off"); plt.show()
```

The first time you wire this in, call it on a single training sample *straight out of the dataloader* — before any model touches it. If the green ground-truth boxes do not sit on the objects, your dataset format is wrong and nothing downstream can save you. This is the detection-specific version of the print-the-batch discipline from [the input pipeline is lying to you](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you): in classification you print the image and its label; in detection you *draw the boxes*, because a box is only correct relative to the pixels.

The trap that makes this check essential is that augmentation must transform boxes in lockstep with the image. A horizontal flip flips `x1` and `x2` to `W - x2` and `W - x1`; forget it and your boxes are mirror-images of the objects. We cover that failure mode in [augmentation debugging for vision](/blog/machine-learning/debugging-training/augmentation-debugging-for-vision); the visualizer above is how you catch it — draw the augmented image with its transformed boxes and look.

## The dataloader boundary: where boxes and images fall out of sync

Before a single coordinate reaches the model, the dataloader has already had a dozen chances to corrupt the boxes, and every one is silent because boxes and images are separate tensors that no library keeps synchronized for you. This is the detection-specific instance of the general dataloader-discipline lesson in [the input pipeline is lying to you](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you): in classification the label is a scalar that survives any image transform untouched, but in detection the "label" is a set of coordinates that must be transformed in exact lockstep with the pixels. Resize the image and forget to scale the boxes, and the boxes describe the *original* resolution while the image is the *new* one — every box is off by the resize ratio.

### Resize is the most common box-desync bug

Suppose your dataset stores 640×480 images with pixel xywh boxes, and your transform resizes every image to 512×512 for the model. The image tensor is now 512×512, but if your box transform did not also scale, the boxes still assume 640 wide and 480 tall. A box at `x = 600` (valid in a 640-wide image) is now off the right edge of a 512-wide image, and a box that should sit at `x = 600 × 512/640 = 480` sits at 600 instead — shifted right and partly off-frame. Worse, a *non-uniform* resize (640×480 → 512×512 changes the aspect ratio) needs different x- and y-scale factors, and a single shared scale factor squashes every box. The fix is to compute the scale factors from the actual size change and apply them to the boxes in the same transform that resizes the image:

```python
def resize_with_boxes(image, boxes_xyxy, out_hw):
    """Resize image AND scale boxes by the same per-axis factors."""
    _, H, W = image.shape
    out_h, out_w = out_hw
    sx, sy = out_w / W, out_h / H          # SEPARATE x and y scales
    image = torch.nn.functional.interpolate(
        image[None].float(), size=out_hw, mode="bilinear",
        align_corners=False)[0]
    boxes = boxes_xyxy.clone().float()
    boxes[:, [0, 2]] *= sx                  # scale x1, x2 by sx
    boxes[:, [1, 3]] *= sy                  # scale y1, y2 by sy
    return image, boxes
```

The discipline that catches every desync of this kind is to draw *augmented* batches, not raw ones. Pull a batch through the full transform pipeline — resize, flip, crop, color jitter — and call the overlay visualizer on the result. If the boxes drift off the objects after resize, your scale factors are wrong. If they mirror the objects after a flip, your flip did not transform the coordinates. If a crop cut an object in half but the box still spans the original extent, your crop did not clip the boxes. None of these raise an exception; all of them are obvious the instant you look.

### Crops, pads, and the empty-box trap

Random crops are the subtlest box transform because they change which objects exist. A crop that removes an object entirely should *remove* its box; a crop that clips an object should *clip* its box to the visible region; and a box clipped to zero area (the object is entirely outside the crop) must be dropped, not kept as a degenerate `[x, x, y, y]` rectangle. Degenerate boxes are a quiet poison: a zero-area box passed to IoU produces a division by zero or a NaN, and a zero-area box assigned as a positive anchor produces a meaningless regression target. After every geometric transform, filter:

```python
def drop_degenerate(boxes_xyxy, labels, min_size=1.0):
    """Remove boxes that a crop or clip reduced to (near) zero area."""
    w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    keep = (w >= min_size) & (h >= min_size)
    return boxes_xyxy[keep], labels[keep]
```

Padding has the opposite trap: when you pad a non-square image to a square (common before batching, so all images in a batch share a size), the boxes need *no* coordinate change if you pad on the bottom-right (the top-left origin is preserved), but they need a *shift* if you pad symmetrically or on the top-left. Get the pad direction wrong and every box is offset by the pad amount. The collate function that batches variable-count detections is its own source of bugs — you cannot stack `[N_i, 4]` box tensors of different `N_i` into one tensor, so detection collate functions keep targets as a *list* of dicts rather than a stacked tensor, and a collate that naively tries to stack them either crashes or, worse, silently pads with zeros that become degenerate boxes. Use the detection-aware collate (torchvision's detection reference code provides one) that returns `(list_of_images, list_of_targets)` and never stacks the targets.

#### Worked example: a resize that halved mAP

A team finetuning a detector on aerial imagery reported `mAP@.5 = 0.22` where a baseline hit 0.44 — exactly half, suspiciously round. Loss descended normally. I drew an augmented batch and the boxes were uniformly shifted toward the bottom-right of every object, by an amount that grew with distance from the top-left corner. That growing-with-position shift is the fingerprint of a scale-factor bug, not a constant offset: a constant offset shifts every box equally, but a wrong scale factor shifts boxes proportionally to their coordinate. The cause: the dataset stored boxes for 1024×1024 source images, the transform resized to 512×512, and the box-scaling step had been dropped in a refactor — boxes assumed 1024 while the image was 512, so every coordinate was 2× too large, placing boxes at twice their correct distance from the origin. Restoring the `boxes *= 0.5` scaling took mAP from 0.22 back to 0.43. The lesson: a *proportional* box error (grows with coordinate) is a resize/scale bug; a *constant* box error (every box shifted equally) is a pad/crop-offset bug. Drawing the augmented batch and watching whether the error grows with position distinguishes them instantly.

## Coordinate format bugs: the silent swap

Now the most common bug in detection. You have a model that emits cxcywh in normalized `[0,1]` (a DETR-style or YOLO-style head), and somewhere downstream a function expects xyxy in pixels. Nothing crashes. Let us build the converter you should use everywhere, then show the bug.

### A reusable, asserting box-format converter

torchvision gives you `box_convert(boxes, in_fmt, out_fmt)` for the format axis, but it does *not* handle the normalized-vs-pixel axis and it does *not* assert that your inputs are sane. Wrap it:

```python
import torch
from torchvision.ops import box_convert, clip_boxes_to_image

VALID = {"xyxy", "xywh", "cxcywh"}

def to_xyxy_pixels(boxes, in_fmt, img_hw, normalized=False):
    """Canonicalize ANY box convention to absolute-pixel xyxy, with checks.

    boxes:      [N, 4] float tensor
    in_fmt:     one of VALID
    img_hw:     (H, W) of the image these boxes live on
    normalized: True if boxes are in [0,1]
    """
    assert in_fmt in VALID, f"bad fmt {in_fmt}"
    boxes = boxes.float().clone()
    H, W = img_hw

    if normalized:
        # a cheap, load-bearing assert: normalized boxes must be in [0,1]
        assert boxes.max() <= 1.5, (
            f"normalized=True but max={boxes.max():.1f}; "
            "these look like pixels")
        scale = torch.tensor([W, H, W, H], dtype=torch.float,
                             device=boxes.device)
        boxes = boxes * scale
    else:
        assert boxes.max() > 1.5 or boxes.numel() == 0, (
            f"normalized=False but max={boxes.max():.2f}; "
            "these look normalized")

    xyxy = box_convert(boxes, in_fmt=in_fmt, out_fmt="xyxy")
    # xyxy must be ordered: x2>=x1, y2>=y1
    assert (xyxy[:, 2] >= xyxy[:, 0]).all(), "x2 < x1 after convert"
    assert (xyxy[:, 3] >= xyxy[:, 1]).all(), "y2 < y1 after convert"
    return clip_boxes_to_image(xyxy, size=(H, W))
```

Two asserts in there are worth their weight in GPU-hours. The `boxes.max() <= 1.5` check on normalized inputs catches the swap where someone passes pixel coordinates (values like 320) with `normalized=True`; the inverse check catches pixel-expecting code handed normalized values (max around 1). The `x2 >= x1` ordering check catches a cxcywh-read-as-xyxy or xywh-read-as-xyxy confusion, because those frequently produce x2 < x1. These are the assertions that turn a four-hour mystery into a one-line stack trace. The series leans on this everywhere — a well-placed assert is the cheapest detector there is, the box-geometry sibling of the shape asserts in [shape bugs and silent broadcasting](/blog/machine-learning/debugging-training/shape-bugs-and-silent-broadcasting).

### The bug, reproduced and measured

Here is the mismatch in the wild. The model head outputs normalized cxcywh; the evaluation code, copied from a torchvision tutorial, expects pixel xyxy and does no conversion:

```python
# BUGGY: model emits normalized cxcywh, evaluator gets it raw
preds = model(images)               # preds["boxes"]: cxcywh in [0,1]
boxes = preds["boxes"]              # e.g. [0.23, 0.42, 0.31, 0.50]
keep = torchvision.ops.nms(boxes, preds["scores"], iou_threshold=0.5)
#       ^ nms reads these as xyxy PIXELS: x1=0.23, y1=0.42,
#         x2=0.31, y2=0.50 -> a 0.08 x 0.08 PIXEL box near the origin
```

Every box is now a sub-pixel speck at the top-left corner. NMS computes pairwise IoU between specks (all near 1.0 because they all overlap at the origin) and suppresses almost everything. Whatever survives is evaluated against ground-truth boxes that are hundreds of pixels away, so every IoU is 0 and every prediction is a false positive. mAP reads 0.05 — the floor you get from a handful of lucky overlaps. The fixed version routes both predictions and ground truth through `to_xyxy_pixels` so they speak the same convention:

```python
# FIXED: canonicalize both sides to pixel xyxy before NMS and eval
H, W = images.shape[-2:]
pred_xyxy = to_xyxy_pixels(preds["boxes"], "cxcywh", (H, W), normalized=True)
keep = torchvision.ops.nms(pred_xyxy, preds["scores"], iou_threshold=0.5)
pred_xyxy = pred_xyxy[keep]
gt_xyxy   = to_xyxy_pixels(targets["boxes"], "xywh", (H, W), normalized=False)
# now IoU(pred_xyxy, gt_xyxy) is meaningful; mAP reflects the real model
```

#### Worked example: a format mismatch from 0.05 to 0.41

I will put numbers on a run I actually debugged. A RetinaNet-style detector, COCO-subset, 20 classes. Training loss looked textbook: classification focal loss 1.8 → 0.4, box regression 0.9 → 0.3 over 12 epochs. COCO `mAP@[.5:.95]` reported **0.051**, `mAP@.5` reported **0.083**. The first thing I did was *not* retrain — I drew ten predictions on their images with the visualizer above and saw every box clustered in the top-left corner, none on an object. That fingerprint (boxes near the origin) is the normalized-fed-as-pixels signature from the matrix. The model head emitted normalized cxcywh; the eval loop fed it straight to a pixel-xyxy IoU. After routing both sides through `to_xyxy_pixels`, the *same checkpoint* — no retraining, not one gradient step — reported `mAP@[.5:.95] = 0.41`, `mAP@.5 = 0.63`. The before/after below is that exact result.

![A two-column before-after figure showing a buggy evaluator with mismatched formats reading mAP 0.05 and a fixed aligned evaluator reading mAP 0.41 on the same checkpoint](/imgs/blogs/debugging-object-detection-8.png)

Nine GPU-hours of training were never the problem. Three lines of format conversion were. This is why the visualize-then-canonicalize habit pays for itself: I confirmed the suspect (boxes in the wrong place, not boxes the model failed to learn) in thirty seconds of looking, then the fix was mechanical. Note the diagnostic discipline — I read the *instrument* (drawn boxes) before forming a hypothesis, exactly the make-it-fail-small-and-look method the series preaches.

| Coordinate symptom | What you see when you draw | Likely format bug | Confirming test |
| --- | --- | --- | --- |
| Boxes clustered near (0,0) | Tiny boxes in top-left corner | Normalized fed as pixels | `boxes.max() <= 1.0`? |
| Boxes shifted up-left by half their size | Box centered on object's top-left quadrant | cxcywh read as xyxy | Center vs corner check |
| Boxes too small, wrong aspect | Box covers part of object | xywh read as xyxy | `(x2-x1)` vs stored `w` |
| Boxes mirror-image of objects | Box on the wrong side | Augmentation didn't flip boxes | Disable flip, re-draw |
| Boxes correct but mAP still 0 | Boxes ON objects, number is 0 | Class-index offset (next section) | GT-as-pred sanity check |

That last row is the cruel one: the boxes are *correct*, you have confirmed it visually, and mAP is still zero. That is not a coordinate bug. That is the evaluator, and we get there after assignment and NMS.

## Anchor and target assignment: when small objects get no positives

Many detectors (Faster R-CNN, RetinaNet, SSD, YOLO with anchors) place a grid of **anchor boxes** — reference boxes of preset sizes and aspect ratios — across the image, and training assigns each ground-truth object to the anchors that overlap it well enough. An anchor is a **positive** (it should predict this object) if its IoU with some ground-truth box exceeds a threshold, typically 0.5 or 0.7; it is a **negative** (background) if its IoU with every object is below a lower threshold, say 0.3; anchors in between are ignored. The regression head learns to nudge each positive anchor onto its assigned object; the classification head learns positives' classes and negatives as background.

The bug: if your anchors do not match the *scales* of your objects, some objects get **zero positive anchors**, and an object with no positive anchor contributes no localization gradient at all. The model literally never learns to detect objects of that size, because nothing was ever assigned to predict them. The classic case is small objects with anchors that are too large.

### The science: why a scale mismatch zeroes the positives

Consider an object that is 12×12 pixels. The smallest anchor on your feature pyramid's coarsest used level is 64×64. The best possible IoU between a 12×12 box and a 64×64 box — even perfectly centered — is bounded by the area ratio. If the small box sits entirely inside the anchor, intersection is the small box's area $12^2 = 144$ and union is the anchor's area $64^2 = 4096$, so IoU is at most $144 / 4096 \approx 0.035$. That is far below any positive threshold. No shift, no scale tweak the regressor could make at assignment time changes the fact that *at assignment* this anchor's IoU with the object is 0.035, so it is never labeled positive. The object is invisible to the loss.

More generally, for a square object of side $s$ inside a square anchor of side $a \ge s$, the maximum IoU is $s^2 / a^2$. To clear a 0.5 IoU threshold you need $s^2/a^2 \ge 0.5$, i.e. $a \le s\sqrt{2} \approx 1.41\,s$. So your smallest anchor must be within about 40% of your smallest object's size, or that object class gets no positives. This is a hard geometric constraint, not a tuning preference, and it explains why "the model just won't detect small things" is almost always an anchor-scale bug rather than a capacity problem.

![A two-column before-after figure showing oversized anchors giving a small object zero positive matches and a small-object AP of 0.04, versus added small anchors giving three positives and AP 0.27](/imgs/blogs/debugging-object-detection-4.png)

### Diagnostic: count positives per object size

Do not guess whether assignment is starving small objects — measure it. Bucket your ground-truth boxes by area and count how many positive anchors each object receives:

```python
import torch
from torchvision.ops import box_iou
from collections import defaultdict

def positives_per_object(anchors_xyxy, gt_boxes_xyxy, gt_areas,
                         pos_iou=0.5):
    """For each GT box, how many anchors clear the positive IoU threshold?"""
    ious = box_iou(gt_boxes_xyxy, anchors_xyxy)   # [num_gt, num_anchors]
    n_pos = (ious >= pos_iou).sum(dim=1)          # positives per GT
    best  = ious.max(dim=1).values                # best IoU per GT
    buckets = defaultdict(list)
    for area, npos, b in zip(gt_areas, n_pos.tolist(), best.tolist()):
        if   area < 32**2:  key = "small  (<32px)"
        elif area < 96**2:  key = "medium (32-96px)"
        else:               key = "large  (>96px)"
        buckets[key].append((npos, b))
    for key in ["small  (<32px)", "medium (32-96px)", "large  (>96px)"]:
        rows = buckets.get(key, [])
        if not rows:
            continue
        zero = sum(1 for npos, _ in rows if npos == 0)
        mean_best = sum(b for _, b in rows) / len(rows)
        print(f"{key}: {len(rows):4d} objs | "
              f"{zero:4d} with ZERO positives | "
              f"mean best IoU {mean_best:.2f}")
```

Run this once over a few hundred training images. The output is unambiguous:

```bash
small  (<32px): 1840 objs | 1310 with ZERO positives | mean best IoU 0.21
medium (32-96px): 2730 objs |   85 with ZERO positives | mean best IoU 0.58
large  (>96px): 1520 objs |    2 with ZERO positives | mean best IoU 0.74
```

71% of small objects have *zero* positive anchors and a mean best IoU of 0.21. The model cannot learn to localize them — there is no gradient telling it to. The fix is to add anchor scales matched to small objects (or use a finer feature-pyramid level, or switch to an anchor-free head like FCOS/center-based assignment), then re-run the counter and confirm small objects now get positives. After adding 16px and 24px anchor scales, small-object zero-positive count dropped to 140 (8%) and small-object AP rose from 0.04 to 0.27 — the before/after in the figure above.

The diagnostic generalizes to assignment-threshold bugs too. If you accidentally set the positive IoU threshold to 0.7 (Faster R-CNN's RPN value) on a head that expects 0.5, you starve positives across all sizes. The same counter shows it: a sudden cliff in positives at every bucket. The principle is the same as overfit-a-single-batch — measure whether the learning signal even reaches the part of the model you are debugging before you assume the model is at fault.

### The positive-negative ratio and objectness

Assignment has a second, quieter failure mode beyond zero positives: a *skewed* positive-to-negative ratio. A dense detector places tens of thousands of anchors per image, and the vast majority — often more than 99% — are background. If you assign every below-threshold anchor as a negative and weight them equally, the classification loss is dominated by easy background anchors, and the objectness or class head learns to predict "background" everywhere because that minimizes the loss. This is the class-imbalance problem that motivated focal loss, and it has a measurable signature: the objectness scores collapse toward zero (everything looks like background), recall craters because no anchor fires with enough confidence to survive the score threshold, and the loss looks *fine* because predicting background is genuinely low-loss when 99% of anchors are background. The same imbalance dynamics from [class imbalance and when accuracy lies](/blog/machine-learning/debugging-training/class-imbalance-and-when-accuracy-lies) apply here, anchor-by-anchor. The fixes are the standard ones — hard-negative mining (train on the hardest negatives, not all of them), a fixed positive:negative ratio per batch (Faster R-CNN samples 1:3), or focal loss (down-weight easy negatives) — and the diagnostic is to log the mean objectness score on positive versus negative anchors. If positive anchors average objectness 0.08 and negatives average 0.02, the head has not learned to distinguish them and the imbalance is winning. The general lesson holds: a head that "won't fire" is usually starved of positive signal, not incapable, and you confirm it by measuring the signal, not by guessing at architecture.

#### Worked example: objectness collapse from unmined negatives

A custom SSD-style detector trained to a clean-looking loss (0.6, descending) but produced *no detections at all* above a 0.3 score threshold — every box scored under 0.2. The per-anchor objectness log told the story: positive anchors averaged objectness 0.11, negatives 0.04 — the head had learned to predict near-zero everywhere because 99.3% of the 24,000 anchors per image were negatives weighted equally, so "everything is background" minimized the loss. The fix was hard-negative mining at a 3:1 negative:positive ratio, training only on the highest-loss negatives. After the fix, positive anchors averaged objectness 0.74, negatives 0.06, the head distinguished foreground from background, and detections appeared above threshold — `mAP@.5` went from 0.00 (no boxes survived the score filter) to 0.39. The instrument that mattered was not the loss (which looked healthy throughout) but the *per-anchor objectness split*; reading it turned a baffling "no detections" symptom into an obvious imbalance diagnosis.

## NMS bugs: the duplicate flood and the over-suppression

Non-max suppression turns a detector's hundreds of overlapping candidate boxes into a clean set of detections. The algorithm: sort candidates by score, take the highest, remove every remaining box whose IoU with it exceeds a threshold (they are duplicates of the same object), repeat. Three parameters and one design choice control it, and each is a bug source.

```python
import torch
from torchvision.ops import nms, batched_nms

# scores below this never make it to NMS at all:
SCORE_THRESH = 0.05
# boxes overlapping more than this get suppressed as duplicates:
IOU_THRESH   = 0.5

def postprocess(boxes_xyxy, scores, labels, score_thresh=SCORE_THRESH,
                iou_thresh=IOU_THRESH):
    keep = scores >= score_thresh                  # 1. score filter
    boxes_xyxy, scores, labels = boxes_xyxy[keep], scores[keep], labels[keep]
    # 2. PER-CLASS NMS: batched_nms offsets boxes by class so a
    #    cat box never suppresses a dog box (class-agnostic would).
    keep = batched_nms(boxes_xyxy, scores, labels, iou_threshold=iou_thresh)
    return boxes_xyxy[keep], scores[keep], labels[keep]
```

### Per-class versus class-agnostic: the recall killer

`nms` is class-agnostic: it suppresses any box that overlaps a higher-scoring box, regardless of class. `batched_nms` is per-class: it offsets each box's coordinates by a large per-class constant so boxes of different classes can never overlap, then runs NMS — which means a high-scoring "dog" box can no longer delete an overlapping "cat" box. The difference matters whenever two objects of different classes genuinely overlap: a person riding a horse, a cup on a table, a tie on a person. Class-agnostic NMS deletes the lower-scoring one and your recall drops on exactly those co-occurring pairs.

![A branching graph showing raw boxes splitting into class-agnostic NMS that drops a true positive at recall 0.71 and per-class NMS that keeps both at recall 0.89](/imgs/blogs/debugging-object-detection-5.png)

The signature is subtle: mAP is *fine* on isolated objects and *low* on classes that frequently co-occur with others. If your per-class AP table shows "person" and "tie" both depressed but "stop sign" healthy, suspect class-agnostic NMS. The fix is `batched_nms`. (There is a legitimate use for class-agnostic NMS — when you truly want only one detection per spatial location regardless of class — but it is rarely what a multi-class detector wants, and it is almost never the default you meant to pick.)

### The IoU threshold: too high floods, too low over-suppresses

Set the NMS IoU threshold too high (say 0.9) and near-duplicate boxes of the *same* object survive, because two boxes have to overlap more than 90% to be considered duplicates — so you get three or four boxes stacked on every object. That flood of duplicates is all false positives (only one can be the TP; the matching procedure consumes one ground-truth box and the rest are FPs), and precision collapses. Set it too low (say 0.3) and two *distinct* nearby objects get merged — the second object's box overlaps the first by more than 30% and gets suppressed — so you lose recall in crowded scenes. The healthy default is 0.5–0.6 for most detectors; crowded-scene detectors sometimes use Soft-NMS, which decays scores by overlap instead of hard-deleting.

The crowded-scene failure is worth dwelling on, because hard NMS has a structural limitation that no threshold fixes. When two pedestrians genuinely stand shoulder to shoulder, their *true* boxes can overlap by 0.6 IoU — that is a real property of the scene, not a duplicate. Any NMS IoU threshold below 0.6 will delete one of the two correct boxes (it looks like a duplicate of the other), and any threshold above 0.6 will fail to suppress actual duplicates elsewhere. There is no single threshold that keeps both true boxes and removes duplicates. **Soft-NMS** resolves this by not deleting at all: instead of removing a box whose IoU with a higher-scoring box exceeds the threshold, it *multiplies that box's score* by a decay factor that shrinks as the overlap grows (a Gaussian $s_i \leftarrow s_i \, e^{-\text{IoU}^2/\sigma}$ is the common form). A box that overlaps a kept box by 0.6 keeps most of its score and survives to be a detection of the *second* pedestrian; a box that overlaps by 0.95 (a true duplicate) has its score crushed toward zero and falls below the final score threshold. The signature that you need Soft-NMS is specifically *recall that collapses in crowded scenes while staying fine on isolated objects* — if your per-image recall is 0.9 on images with one or two objects and 0.5 on images with ten overlapping objects, hard NMS is deleting true positives in the crowds, and Soft-NMS (or a per-object-aware assignment) is the fix rather than a threshold tweak.

There is also a *correctness* trap inside `batched_nms` worth knowing, because it explains a bug that looks like a coordinate error but is not. `batched_nms` offsets each box by `class_id × max_coordinate` so different classes never overlap. If your `max_coordinate` offset constant is smaller than the actual image dimensions, two classes' offset boxes can still overlap and you get accidental cross-class suppression; if you call `batched_nms` on *normalized* coordinates (all in `[0,1]`) with the default offset logic, the offset (which assumes pixel-scale coordinates) dwarfs the boxes and the function still works, but feeding it the wrong coordinate scale elsewhere will not. The safe rule is the same as everywhere in this post: canonicalize to pixel xyxy before NMS, and pass real integer class labels to `batched_nms` so it computes the offset correctly.

#### Worked example: a duplicate flood from a copied threshold

A pedestrian detector, deployed, started reporting three to five overlapping boxes per person. Precision in the offline eval was 0.43 where it had been 0.81 the week before; recall was unchanged at 0.88. The recall-unchanged, precision-halved signature said duplicates, not misses. The git blame showed an NMS `iou_threshold` changed from `0.5` to `0.95` during a "tune the NMS" experiment that was never reverted. At 0.95, two boxes on the same pedestrian (typically 0.7–0.85 IoU with each other) were *not* considered duplicates and both survived. Reverting to 0.5 restored precision to 0.80 with recall still 0.88. The diagnostic that nailed it in one line: count detections per image before and after NMS.

```python
# how many survive NMS? a flood means the IoU threshold is too high
raw = (scores >= 0.05).sum().item()
kept_05 = nms(boxes, scores, 0.50).numel()
kept_95 = nms(boxes, scores, 0.95).numel()
print(f"raw {raw}  ->  NMS@0.5 keeps {kept_05}  |  NMS@0.95 keeps {kept_95}")
# raw 312  ->  NMS@0.5 keeps 9  |  NMS@0.95 keeps 47
```

47 detections surviving where 9 objects exist is a duplicate flood you can see in one print. The score threshold has a parallel failure: set it too high (0.5 instead of 0.05) and you suppress every low-confidence-but-correct box, capping recall — your PR curve simply stops early and AP drops because the high-recall region is empty. Always evaluate with a *low* score threshold (0.01–0.05); the PR curve integration handles confidence ranking, so a low threshold lets the curve reach high recall. Cranking the score threshold to "clean up" the predictions before evaluation silently truncates your AP.

| NMS symptom | Recall | Precision | Likely cause | Fix |
| --- | --- | --- | --- | --- |
| Stacked boxes on each object | unchanged | crashes | IoU threshold too high | Lower to 0.5–0.6 |
| Missed objects in crowds | drops | unchanged | IoU threshold too low | Raise toward 0.5; Soft-NMS |
| Overlapping classes lost | drops on co-occurring | fine | Class-agnostic NMS | Use `batched_nms` |
| AP low, curve stops early | capped | high | Score threshold too high at eval | Eval at 0.01–0.05 |
| Random missing detections | drops | fine | NMS run before score sort | Sort by score first |

## The mAP that lies: the evaluator is a bug source

You have visualized the predictions, the boxes are *on the objects*, NMS is sane — and mAP is still near zero. Now the bug is in the evaluator, and detection evaluators have more places to disagree with your predictions than any other metric in machine learning. Here are the five that bite, in rough order of frequency.

### 1. Coordinate format mismatch between predictions and ground truth

This is the same format axis from earlier, now living *inside* the evaluator. pycocotools' `COCOeval` expects detections in xywh-pixel format (the COCO `bbox` convention) in the results JSON, and it loads ground truth as xywh-pixel from the annotations. If your prediction-dumping code writes xyxy or cxcywh into the results JSON, every IoU the evaluator computes is garbage and mAP is near zero — even though your boxes are correct in *your* convention. The fix is to convert predictions to xywh pixels at the dump boundary:

```python
from torchvision.ops import box_convert

def to_coco_results(image_id, boxes_xyxy, scores, labels):
    """pycocotools wants xywh PIXELS and category_id matching the GT."""
    xywh = box_convert(boxes_xyxy, in_fmt="xyxy", out_fmt="xywh")
    return [
        {"image_id": int(image_id),
         "category_id": int(lbl),          # MUST match GT's category ids
         "bbox": [round(float(v), 2) for v in box],   # xywh pixels
         "score": float(s)}
        for box, s, lbl in zip(xywh.tolist(), scores.tolist(), labels.tolist())
    ]
```

### 2. Class-index off-by-one

This is the cruelest detection bug because the boxes are *perfect* and mAP is still zero. COCO's category ids are not 0–79; they are a specific set of 90 integers (1, 2, 3, …, 11, 13, …, 90) with gaps, because some categories were removed. Most models train on *contiguous* 0-indexed labels (0–79) for the 80 classes, and you need a mapping back to COCO ids when you dump results. Forget the mapping — or be off by one because you added a background class at index 0 — and every predicted "person" (your class 0) gets compared against ground-truth "person" (COCO id 1) as if they were different categories. The matching procedure in step 2 never finds a same-class ground-truth box, so every prediction is a false positive and every object is a false negative. mAP is 0.00–0.05.

![A left-to-right timeline showing predictions 0-indexed and ground truth 1-indexed, the evaluator matching ids so person is compared to bicycle, every true positive counted as the wrong class, mAP 0.05, fixed by shifting predictions to mAP 0.41](/imgs/blogs/debugging-object-detection-6.png)

The signature distinguishes it from a coordinate bug: a coordinate bug lowers IoU, so `mAP@.5` survives partially while `mAP@.75` collapses; a class-offset bug makes the *class* wrong, so both `mAP@.5` and `mAP@.75` crater to near zero *together*, and — the tell — if you evaluate class-agnostically (ignore the category, match on IoU alone) mAP jumps back up. That last test is decisive:

```python
# DECISIVE class-offset test: does ignoring class fix the number?
# If class-agnostic mAP is high but per-class mAP is ~0, it's a class bug.
def class_agnostic_map_check(preds, gts):
    # set every prediction and GT to the same dummy class, re-evaluate
    preds_flat = [{**p, "category_id": 1} for p in preds]
    gts_flat   = remap_all_gt_to_one_class(gts)
    # if this reads ~0.4 while real eval reads ~0.05 -> class offset bug
    return evaluate(preds_flat, gts_flat)
```

If class-agnostic mAP is 0.40 and your real mAP is 0.05, the boxes are right and the *classes* are misaligned — fix the index mapping, not the model. The off-by-one frequently comes from a background class: detectors that reserve index 0 for background emit foreground classes as 1–80, but the evaluator's category map expects 0–79, so everything is shifted by one and "person" reads as "bicycle."

### 3. The GT-as-prediction sanity check

Here is the most valuable single test in detection evaluation, and the one almost nobody runs: **feed the ground-truth boxes into the evaluator as if they were perfect predictions (score 1.0) and confirm mAP is ~1.0.** If a perfect predictor does not score a perfect mAP, your *evaluator* is broken — and you have just isolated the bug to the evaluation stage without touching the model at all.

```python
def gt_as_pred_sanity(coco_gt, image_ids, cat_map):
    """Turn GT into predictions and evaluate. Should yield mAP ~ 1.0.
    If it doesn't, the evaluator/format/class mapping is broken, NOT the model.
    """
    results = []
    for img_id in image_ids:
        for ann in coco_gt.imgToAnns[img_id]:
            results.append({
                "image_id": img_id,
                "category_id": ann["category_id"],   # use GT's own ids
                "bbox": ann["bbox"],                  # GT's own xywh pixels
                "score": 1.0,
            })
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    # ev.stats[0] is mAP@[.5:.95]; it MUST be ~1.0 here
    assert ev.stats[0] > 0.99, (
        f"GT-as-pred mAP is {ev.stats[0]:.3f}, not ~1.0 -> "
        "your evaluator's format or class mapping is broken")
```

This check decomposes the problem in one shot. If GT-as-pred returns 1.0, your evaluator is correct and the low mAP is the *model's* (or your prediction-dump's) fault — go back to format and class mapping in the dump. If GT-as-pred returns 0.6 or 0.05, the evaluator itself is misconfigured — wrong IoU type, wrong area ranges, a category map that does not include all your classes, image ids that do not match. You have bisected evaluation-vs-model in thirty seconds. I now run this as the *first* thing whenever a detector reports surprisingly low mAP, before I look at a single prediction. It is the detection-specific instance of the series' make-it-fail-small principle: construct the input for which the answer is known (a perfect predictor scores 1.0) and check that the instrument reports it.

### 4. Wrong IoU threshold or mAP definition

`mAP@.5` (Pascal VOC) and `mAP@[.5:.95]` (COCO) are different numbers, and a model that reads 0.62 by the VOC definition reads 0.41 by COCO's — neither is wrong, but comparing your COCO number to a paper's VOC number (or vice versa) manufactures a phantom regression. Worse, if you hand-roll an evaluator and set the IoU threshold to 0.75 thinking you are computing "mAP," you get a much lower number than `mAP@.5`, and it looks like the model is failing. Always state which definition you are reporting, and when reproducing a paper, match its protocol exactly. pycocotools' `COCOeval` reports the full breakdown — `mAP@[.5:.95]`, `mAP@.5`, `mAP@.75`, and small/medium/large AP — so read the whole table rather than one cell, because the *pattern* across IoU thresholds is itself a diagnostic (format bugs hit high thresholds hardest; class bugs hit all thresholds equally).

### 5. area ranges and maxDets

COCO's small/medium/large AP uses fixed area cutoffs ($32^2$ and $96^2$ pixels), computed on the *original* image scale. If you resize images for the model and forget to scale boxes back to original coordinates before evaluation, an object that was "large" at training resolution gets bucketed as "small" — and your small-AP looks terrible while large-AP looks empty. Similarly, COCO caps detections per image at `maxDets = 100`; if your dump writes 300 boxes per image, the evaluator keeps the top 100 by score, and if your scores are poorly calibrated you can lose real detections to the cap. Both are evaluator-configuration bugs that the GT-as-pred check will *not* catch (ground truth respects these constraints), so they need their own look: confirm your predicted boxes are in *original-image* coordinates and that you are not relying on more than 100 detections per image.

![A decision tree rooted at low mAP with falling loss, branching into a GT-as-prediction question that splits format and class bugs from an honest evaluator, and a boxes-look-right question that splits assignment from NMS suspects](/imgs/blogs/debugging-object-detection-7.png)

That tree is the bisection in one picture: a near-zero mAP with a falling loss is *almost always* an evaluation-format bug, so test the evaluator (GT-as-pred) before you retrain anything. Only if GT-as-pred returns 1.0 and your boxes are visually wrong do you descend into assignment and NMS.

### A boundary-by-boundary assertion harness

The reason detection bugs survive so long is that the disagreements live *between* stages, where no single function is responsible for checking the convention. The durable fix is to assert the convention at every boundary, so a mismatch fails loudly at the exact line where it happens rather than silently corrupting a number five stages later. Here is the harness I now drop into any detection pipeline — it checks coordinate ranges and class ranges at the dataset, model, NMS, and eval boundaries:

```python
def assert_boxes(boxes, name, img_hw=None, normalized=False, fmt="xyxy"):
    """Loud, cheap checks at each pipeline boundary."""
    assert boxes.dtype.is_floating_point, f"{name}: boxes must be float"
    assert boxes.shape[-1] == 4, f"{name}: last dim must be 4, got {boxes.shape}"
    mx = boxes.max().item() if boxes.numel() else 0.0
    if normalized:
        assert mx <= 1.5, f"{name}: normalized but max={mx:.1f} (pixels?)"
    elif img_hw is not None:
        H, W = img_hw
        assert mx <= max(H, W) * 1.5, (
            f"{name}: max={mx:.0f} exceeds image {W}x{H} (wrong scale?)")
    if fmt == "xyxy" and boxes.numel():
        assert (boxes[:, 2] >= boxes[:, 0]).all(), f"{name}: x2<x1 (cxcywh?)"
        assert (boxes[:, 3] >= boxes[:, 1]).all(), f"{name}: y2<y1 (xywh?)"

def assert_classes(labels, name, num_classes, zero_indexed=True):
    lo, hi = (0, num_classes - 1) if zero_indexed else (1, num_classes)
    assert labels.min() >= lo and labels.max() <= hi, (
        f"{name}: class ids {labels.min()}..{labels.max()} "
        f"outside [{lo},{hi}] (off-by-one / background class?)")

# wire one assert at each boundary:
assert_boxes(gt_boxes, "dataset",   img_hw=(H, W), normalized=False, fmt="xywh")
assert_boxes(pred_xyxy, "post-NMS", img_hw=(H, W), normalized=False, fmt="xyxy")
assert_classes(pred_labels, "pre-eval", num_classes=80, zero_indexed=True)
```

These asserts cost microseconds and turn the entire class of silent-relocation bugs into immediate, located failures. The `x2 >= x1` check catches cxcywh-read-as-xyxy at the boundary where it happens; the `max <= 1.5` check catches normalized-vs-pixel; the class-range check catches the off-by-one before it reaches the evaluator. Asserting the convention at every boundary is the structural cure for a class of bugs whose entire pathology is that conventions are not stored — so you store them, as runtime invariants.

### Reproducing the evaluation format exactly

The last evaluator trap is subtler than format or class: the *coordinate space* the evaluation happens in. If you resize images to 512×512 for the model but evaluate against ground truth in original-image coordinates, you must scale predictions back to the original resolution before the evaluator sees them — and forgetting that scale-back is a bug the GT-as-pred check cannot catch, because ground truth is already in original coordinates. The symptom is a small-but-consistent mAP depression (boxes are slightly off because they are in the wrong coordinate space) that worsens for images whose resize ratio is furthest from 1.0. The discipline is to record the resize ratio per image and invert it on the predictions:

```python
# predictions came out in 512x512 model space; eval wants original space
sx, sy = orig_w / 512, orig_h / 512
pred_xyxy[:, [0, 2]] *= sx          # scale x back to original width
pred_xyxy[:, [1, 3]] *= sy          # scale y back to original height
# NOW the predictions live in the same space as the COCO ground truth
```

The same care applies to letterbox padding (the pad offset must be subtracted before scaling) and to test-time augmentation (predictions from a flipped image must be flipped back before merging). Every one of these is a coordinate-space boundary, and every one is invisible until you draw the final predictions on the *original* image and confirm they land on the objects. The visualize-first habit closes the loop: it is the one check that validates the entire chain from dataset to evaluator in a single glance, because a box drawn on the right object in the right place is, by construction, in the right format, the right class space, and the right coordinate space all at once.

## Loss balancing and regression-target encoding

Two model-side issues round out the picture, because they produce low mAP *with the boxes genuinely wrong* (not a format illusion), and they are worth distinguishing from each other.

### Loss balancing: when one head starves another

A detector optimizes a sum of losses — classification, box regression, and (for anchor-based one-stage detectors) objectness:

$$\mathcal{L} = \lambda_{\text{cls}}\,\mathcal{L}_{\text{cls}} + \lambda_{\text{box}}\,\mathcal{L}_{\text{box}} + \lambda_{\text{obj}}\,\mathcal{L}_{\text{obj}}.$$

If the box-regression loss is computed in raw pixel units (smooth-L1 on absolute coordinates, values in the hundreds) while classification loss is in nats (values around 1), the box term dominates the gradient and the classification head barely trains — or the reverse, if box targets are normalized to small values and you set $\lambda_{\text{box}}$ too low. The signature is a *split* metric: good localization (high IoU on the boxes that do fire) but wrong classes, or correct classes on poorly localized boxes. This is the same loss-scale-imbalance failure as the multi-task case in [loss function bugs](/blog/machine-learning/debugging-training/loss-function-bugs) — when one term is orders of magnitude larger, it owns the gradient and the other term is effectively ignored. The diagnostic is to log the three loss terms *separately* and watch their relative magnitudes; if box loss is 50× classification loss throughout, rebalance the weights or normalize the regression targets so the terms are comparable.

```python
# log the three heads separately - never just the summed loss
losses = model(images, targets)        # dict of per-head losses
total = sum(losses.values())
if step % 50 == 0:
    parts = {k: round(v.item(), 3) for k, v in losses.items()}
    print(f"step {step}: total={total.item():.3f}  {parts}")
    # step 50: total=152.4  {'cls':0.71, 'box':148.9, 'obj':2.8}
    #          ^ box term is 200x cls -> classification head is starved
```

A box term 200× the classification term, as in that printout, means almost the entire gradient is localization and the classifier never learns — exactly the split signature. Reading the per-head losses separately is the detection version of "read the instruments": the summed loss looks like it is descending while one head is starving.

There is a deeper science point hiding in the choice of regression loss itself, because it changes which bugs are even possible. Smooth-L1 (Huber) loss on raw coordinates treats the four numbers as independent scalars, which means it does not know that they describe a box — it will happily push `x2` past `x1` during training (a degenerate box) and it weights a 10-pixel error on a large object the same as a 10-pixel error on a small object, even though the second is far more damaging to IoU. The IoU-family losses — **GIoU, DIoU, CIoU** — fix this by optimizing the actual overlap metric. Plain IoU loss $\mathcal{L} = 1 - \text{IoU}$ has a fatal flaw as a *loss*: when two boxes do not overlap at all, IoU is 0 and its gradient is 0, so a prediction that is completely off the object gets no signal to move toward it. GIoU (Generalized IoU) repairs this by subtracting the area of the smallest enclosing box not covered by the union, giving a non-zero gradient even for non-overlapping boxes:

$$\mathcal{L}_{\text{GIoU}} = 1 - \text{IoU} + \frac{|C \setminus (A \cup B)|}{|C|},$$

where $C$ is the smallest box enclosing both $A$ and $B$. The practical consequence for debugging: if you see localization that *plateaus with boxes near but not on objects*, and you are using plain smooth-L1, the regression may simply lack the signal to close the last gap, and switching to GIoU/CIoU often unsticks it. Conversely, an IoU-family loss mishandles degenerate or zero-area boxes (the denominators go to zero) more dramatically than smooth-L1, so if you switch and start seeing NaN in the box loss, suspect a degenerate box reaching the loss — exactly the drop-degenerate filter from the dataloader section is the guard. This connects to [hunting NaNs and infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs): an IoU-loss NaN is almost always a zero-area box dividing by zero, caught with `torch.autograd.set_detect_anomaly(True)` and a degenerate-box assert before the loss.

### Regression-target encoding: delta versus absolute

Detection heads rarely regress absolute coordinates. They regress **deltas** relative to an anchor or a feature-map cell — typically $(t_x, t_y, t_w, t_h)$ where $t_x = (x - x_a)/w_a$, $t_y = (y - y_a)/h_a$, $t_w = \log(w/w_a)$, $t_h = \log(h/h_a)$ — and decode them back to boxes at inference. The encode and decode must be exact inverses. If you encode targets with one variance normalization (Faster R-CNN scales deltas by `(0.1, 0.1, 0.2, 0.2)`) and decode predictions without dividing it back out, every predicted box is scaled wrong — boxes systematically too large or too small, off by a constant factor. The loss is fine (encoder and the head agree on the encoded space) and the boxes are wrong (decoder disagrees). The fix is a unit test: encode a known box, decode it, assert you get the original box back.

```python
def test_encode_decode_roundtrip(anchor, box, encoder, decoder):
    """Encoding then decoding a box MUST return the original box."""
    delta = encoder(box, anchor)          # box -> (tx, ty, tw, th)
    recovered = decoder(delta, anchor)    # (tx, ty, tw, th) -> box
    assert torch.allclose(recovered, box, atol=1e-4), (
        f"round-trip failed: {box.tolist()} -> {recovered.tolist()}; "
        "encoder/decoder variance normalization mismatch")
```

That round-trip assert is the regression-target equivalent of the GT-as-pred check: construct an input whose correct output is known (encode-then-decode is the identity) and confirm the instrument agrees. If it fails, your decoder does not invert your encoder, and the boxes will be a constant factor off no matter how well the model trains. This is a numerics-of-the-target bug, adjacent to the encoding issues in [your model isn't learning what you think](/blog/machine-learning/debugging-training/your-model-isnt-learning-what-you-think) — the model learns the target you gave it, and the target was the wrong parameterization.

## Train-serve mismatch: the detector that worked in eval and failed in production

A detector can pass every offline check and still fail in production, because the *serving* pipeline reconstructs the same five-boundary chain — and any boundary that differs between training and serving relocates the boxes. This is the detection instance of the general train-infer-mismatch problem: the model is identical, but the preprocessing and post-processing around it differ, so the inputs and outputs drift. The four boundaries that most often diverge are worth naming, because each produces a distinct production-only failure that no offline metric catches.

**Preprocessing skew** is the most common. If training resizes with bilinear interpolation and anti-aliasing but serving resizes with nearest-neighbor (or a different library's default), the pixels the model sees at serve time differ subtly from training, and a detector tuned to one interpolation degrades on another — not catastrophically, but enough to lose a few points of recall on small objects whose appearance is most interpolation-sensitive. Worse is a *normalization* skew: training subtracts ImageNet mean and divides by std, serving forgets the normalization (or uses 0–255 instead of 0–1), and the model sees inputs in the wrong range. The boxes come out garbage because the features are garbage. The fix is to share one preprocessing function between training and serving — literally import the same code — and the diagnostic is to run one image through both pipelines and assert the preprocessed tensors are identical.

**Coordinate-space skew** is the production version of the eval coordinate-space bug. The serving code receives an image at its native resolution, resizes for the model, gets predictions in model-space coordinates, and must scale them back to native resolution before drawing or returning them. Forget the scale-back and the API returns boxes in 512×512 space for a 1920×1080 image — boxes that are correct relative to the resized image but wildly wrong relative to what the user sees. This passes offline eval (which often works in model space) and fails the instant a human looks at a production result. The guard is the same visualize-on-the-original-image habit, run on a production sample.

**NMS and threshold skew** is the quietest. Teams often evaluate offline with a low score threshold (0.01) to get the full PR curve, then deploy with a high score threshold (0.5) to keep the API output clean — which is *correct*, but if the deployment threshold is set without checking the PR curve, it can sit on a steep part of the curve where small changes swing recall dramatically. And if the offline NMS IoU threshold differs from the serving one (a config that drifted), the production duplicate behavior differs from what eval measured. The discipline is to pin every post-processing parameter — score threshold, NMS IoU threshold, maxDets, per-class vs class-agnostic — in one config object shared between eval and serving, and to evaluate at the *deployment* thresholds, not just the full-curve ones, so the offline number reflects what production will actually do.

**Letterbox and aspect-ratio skew** is the trap for models trained on square crops but served on arbitrary aspect ratios. If training always saw 512×512 square images (via center-crop or square-resize) and serving letterboxes a 16:9 image into a 512×512 frame with gray bars, the model sees padding it never trained on, and detections near the bars degrade or the padding itself triggers false positives. The fix is to letterbox identically in training and serving, and to subtract the letterbox offset and scale when mapping boxes back. The unifying lesson across all four: a detector that works offline and fails in production almost never has a model bug — it has a *boundary* that training and serving implement differently, and you find it by running one image through both and comparing the tensors at every boundary, exactly the bisection-by-boundary method this whole post is built on.

## Case studies: real detection bug signatures

Four patterns I have hit or watched others hit, with the symptom, the confirming test, and the fix — so you can match your run against them.

### Case 1: the COCO category-id gap

A team finetuning a torchvision Faster R-CNN on a 12-class custom dataset reported `mAP = 0.0` flat — not low, *zero*. Loss descended normally. The boxes, when drawn, were on the objects. The GT-as-prediction check returned 0.0, not 1.0, which immediately said *evaluator*, not model. The cause: they had labeled their classes 1–12 in the annotation JSON (COCO convention reserves 0 for background) but the model emitted 0–11, and the results dump wrote the model's 0–11 directly. Every predicted class was off by one from the ground-truth category id, so no match ever succeeded. Remapping predictions `class_id + 1` before the dump took mAP from 0.0 to 0.38. The lesson: the GT-as-pred check returning a wrong number is the fastest possible proof that the bug is in evaluation, and a class-id gap is the most common reason for a flat zero.

### Case 2: normalized predictions, pixel evaluator

The RetinaNet run from the worked example earlier: head emits normalized cxcywh, evaluator assumes pixel xyxy. Loss healthy, `mAP@[.5:.95] = 0.051`. The fingerprint — visualized boxes clustered at the top-left corner — is unmistakable once you have seen it. Routing predictions and ground truth through one canonicalizing converter (`to_xyxy_pixels`) took the *same checkpoint* to 0.41. No retraining. This is the canonical "format mismatch tanks mAP while loss looks okay" bug, and the lesson is to canonicalize boxes to one convention (pixel xyxy) the instant they leave the model, before NMS, before eval, before anything.

### Case 3: anchors that ignored small objects

A traffic-sign detector landed `mAP@.5 = 0.51` overall but `small-AP = 0.04` — large signs detected, small distant signs invisible. The `positives_per_object` counter showed 71% of small objects with zero positive anchors and mean best IoU 0.21, well below the geometric floor for a 0.5 threshold. The smallest anchor (64px) could not exceed IoU 0.035 with a 12px object — a hard constraint, not a tuning issue. Adding 16px and 24px anchor scales (and using a finer FPN level) dropped the zero-positive rate to 8% and raised small-AP to 0.27. The lesson: "the model can't see small objects" is a measurable assignment fact, and the positive-anchor counter measures it before you waste time on architecture changes.

### Case 4: the NMS that was never reverted

The pedestrian detector with precision halved (0.81 → 0.43) and recall unchanged (0.88): a `iou_threshold=0.95` left over from an experiment flooded every person with 3–5 surviving boxes. The recall-unchanged-precision-halved signature is the tell — misses lower recall, duplicates lower precision, and only duplicates were happening. The count-detections-per-image diagnostic (47 surviving where 9 objects exist) confirmed it in one print; reverting to 0.5 restored precision to 0.80. The lesson: a precision-only or recall-only collapse is a strong fingerprint, and counting survivors through NMS is a one-line confirmation. These map onto the master taxonomy — case 1 and 2 are *evaluation* bugs, case 3 is *data/assignment*, case 4 is *model-code/post-processing* — exactly the six-places bisection the series teaches, instantiated for detection.

What unites all four cases is the order of operations: in none of them did the right move start with the model. In every case the fastest path to the root cause was to read an instrument — the drawn boxes, the GT-as-pred number, the positive-anchor counter, the survivors-through-NMS count — and let the instrument's reading point at the suspect *before* forming a hypothesis about what was wrong. That order is the whole method. The temptation with a low mAP is to assume the model underfit and to reach for more epochs, a bigger backbone, or a different learning rate, because that is what you do when a classifier underperforms. But a detector that reports low mAP with a descending loss has almost certainly learned the task; the number is being mismeasured or the boxes are being mis-decoded somewhere in the five-boundary chain, and no amount of additional training fixes a boundary bug. The single most valuable habit you can build is to *distrust the metric before you distrust the model* — run GT-as-pred, draw the boxes — and only conclude "the model needs to learn more" after you have proven the measurement is honest. Nearly every multi-day detection-debugging story I know would have been a thirty-minute story if someone had drawn the boxes on day one.

## When this is (and isn't) your bug

Detection's failure modes have sharp boundaries, and knowing them stops you from chasing the wrong stage.

- **Healthy loss, near-zero mAP, boxes drawn ON the objects → it is the evaluator (format or class), not the model.** Run GT-as-pred first. If it returns 1.0, the bug is in your prediction dump's format or class mapping; if it returns a wrong number, the evaluator itself is misconfigured. Do not retrain.
- **Healthy loss, near-zero mAP, boxes drawn in the WRONG place (corner, mirrored, half-size) → it is a coordinate-format bug at decode/NMS.** Canonicalize to pixel xyxy and re-draw. Do not touch the evaluator yet.
- **`mAP@.5` survives but `mAP@.75` collapses → coordinate imprecision, not class.** A constant coordinate offset or a sloppy regression-target encoding lowers IoU uniformly, hitting strict thresholds hardest. Check encode/decode round-trip.
- **`mAP@.5` and `mAP@.75` both near zero together, class-agnostic mAP is high → class-index offset.** The boxes are right; the classes are misaligned. Fix the index mapping.
- **Overall mAP fine, small-AP terrible → anchor-scale or feature-pyramid-level mismatch, not a capacity problem.** Count positives per object size; the small bucket will show zero positives.
- **Precision collapses, recall unchanged → NMS duplicate flood (IoU threshold too high).** Count survivors through NMS.
- **Recall collapses, precision unchanged → NMS over-suppression (threshold too low) or score threshold too high at eval.** Lower the NMS IoU threshold or the eval score threshold.
- **Loss won't descend at all, even on one batch → this is not a detection-specific bug.** Overfit a single batch (drive total loss toward zero on 1–2 images); if it cannot, you have a generic optimization or model-code bug, and the [taxonomy decision tree](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) routes you. Detection-specific debugging starts only once the loss descends but the metric does not.

That last point is the most important boundary. Everything in this post assumes the loss is *descending* and the metric disagrees. If the loss itself will not move — flat, NaN, or stuck at the initial value — stop reading detection-specific advice and go back to the foundations: overfit one batch, read the grad norms, check the learning rate. Detection bugs are overwhelmingly *decode and evaluate* bugs that live downstream of a model that learned fine, which is exactly why they are so silent: the part of the system that fails is the part with no loss watching it.

## Key takeaways

- **A box is four numbers plus a convention; the convention is not stored, so it gets confused.** Canonicalize every box to one format (pixel xyxy) the instant it leaves the model, and convert at every boundary explicitly.
- **The loss compares the model to its own targets; the metric compares it to the world.** A healthy loss with a near-zero mAP means the encoding between them is wrong, not the model. The metric is not lying — it is the only thing measuring reality.
- **Visualize predicted boxes on the actual image. Always.** A printed coordinate tells you nothing; a drawn box reveals a format bug in thirty seconds. This is the single most powerful detection debugging tool.
- **Run GT-as-prediction first when mAP is surprisingly low.** A perfect predictor must score mAP ~1.0; if it does not, the evaluator is broken and you have bisected evaluation-vs-model without touching the model.
- **A class-offset bug cratters both `mAP@.5` and `mAP@.75` together; a coordinate bug spares `mAP@.5` but collapses `mAP@.75`.** Class-agnostic mAP high with per-class mAP near zero is the decisive class-offset test.
- **Small-object AP near zero is an assignment fact, not a capacity problem.** Count positive anchors per object size; oversized anchors give small objects zero positives by a hard geometric bound, $\text{IoU}_{\max} = s^2/a^2$.
- **Precision-only collapse means NMS duplicates (threshold too high); recall-only collapse means over-suppression or an eval score threshold set too high.** Count survivors through NMS to confirm.
- **Use per-class (`batched_nms`), not class-agnostic, NMS for multi-class detectors**, or overlapping objects of different classes silently delete each other.
- **Encode and decode must be exact inverses.** Unit-test the regression-target round-trip; a variance-normalization mismatch makes every box a constant factor off while the loss looks fine.
- **State which mAP you report (`@.5` VOC vs `@[.5:.95]` COCO)** and match a paper's protocol exactly before declaring a regression — the definitions differ by 0.2 mAP routinely.

## Further reading

- **PyTorch / torchvision documentation** — `torchvision.ops.box_convert`, `box_iou`, `nms`, `batched_nms`, `clip_boxes_to_image`, and `torchvision.utils.draw_bounding_boxes`. The canonical, well-tested implementations of every box operation in this post; read their format conventions carefully.
- **pycocotools and the COCO evaluation protocol** — Lin et al., "Microsoft COCO: Common Objects in Context" (2014), and the `COCOeval` source. The authoritative definition of `mAP@[.5:.95]`, the area ranges, `maxDets`, and the xywh-pixel results format.
- **"Focal Loss for Dense Object Detection"** — Lin et al. (2017), the RetinaNet paper. The clearest treatment of one-stage anchor assignment, the class-imbalance problem, and how positive/negative assignment thresholds interact with the loss.
- **"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"** — Ren et al. (2015). The origin of the anchor mechanism, the IoU-threshold assignment, and the `(0.1, 0.1, 0.2, 0.2)` delta variance normalization that causes encode/decode round-trip bugs.
- **"FCOS: Fully Convolutional One-Stage Object Detection"** — Tian et al. (2019). The anchor-free alternative; reading it clarifies *why* anchor-scale bugs happen and how center-based assignment sidesteps them.
- **[A taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs)** — the master symptom → suspect → confirming test → fix decision tree this post instantiates for detection.
- **[The training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook)** — the capstone checklist and the full bisection method, with the printable version of the GT-as-pred and visualize-first habits.
- **[Your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying)** and **[debugging segmentation](/blog/machine-learning/debugging-training/debugging-segmentation)** — the evaluation-bug and the sibling dense-prediction-task posts; the mask-IoU and class-index traps in segmentation rhyme exactly with the box-IoU and class-offset traps here.

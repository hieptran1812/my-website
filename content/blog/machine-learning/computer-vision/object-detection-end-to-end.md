---
title: "Object Detection, End to End: A Working Mental Model for Data, Training, Evaluation, and Modern Detectors"
date: "2026-06-04"
publishDate: "2026-06-04"
description: "A principal-engineer field guide to building object detectors in 2026: the set-prediction mental model, DETR vs YOLO, data and augmentation, the loss cocktail, finetuning recipes, and why mAP lies."
tags: ["object-detection", "computer-vision", "detr", "rt-detr", "vision-transformer", "yolo", "fpn", "mean-average-precision", "data-augmentation", "finetuning", "coco", "nms"]
category: "machine-learning"
subcategory: "Computer Vision"
author: "Hiep Tran"
featured: true
readTime: 51
---

Most engineers arrive at object detection through image classification, and that is exactly where the trouble starts. Classification trained a reflex: one image goes in, one label comes out, you compute cross-entropy, and accuracy tells you how you are doing. Detection breaks every clause of that sentence. One image produces a *variable-length, unordered set* of outputs. Each output is not a label but a tuple of a box, a class, and a confidence. There is no single loss you can read off without first deciding *which prediction is responsible for which object* — a matching problem that does not exist in classification at all. And the headline metric, mean average precision, is a multi-stage computation that can move in the opposite direction from the thing your users actually care about.

I have shipped detectors for retail shelf auditing, drone imagery, and document layout, and the same failures recur on every team: people port their classification habits, get a model that trains without errors, watch mAP climb to something respectable, and then discover in production that it misses every small object, hallucinates boxes on textured backgrounds, or collapses the moment the camera changes. None of those failures are mysterious once you hold the right mental model. This article is that mental model, built up one layer at a time, with the code, the numbers, and the war stories that make each layer stick.

## Why object detection breaks your classification intuitions

Before any architecture, internalize the structural differences. Every one of them is a place where a classification habit silently does the wrong thing.

| What you assume from classification | The naive port to detection | The reality |
|---|---|---|
| One image produces one output | The network has a fixed-size output head | Each image has an unknown number of objects (0 to hundreds); output length is data-dependent |
| The label is the supervision | Compute loss against the label directly | You must first *match* predictions to ground-truth boxes, then compute loss on matched pairs |
| Accuracy summarizes quality | Report top-1 / top-5 | mAP integrates precision over recall, over IoU thresholds, over classes — one number hides six failure modes |
| Classes are roughly balanced | Standard cross-entropy is fine | 99%+ of candidate locations are background; without rebalancing the model learns to predict nothing |
| The object fills the frame | Resize to 224x224 and go | A traffic sign can be 12 pixels wide; aggressive resizing destroys the signal you need most |
| Augmentation transforms the image | Random crop, flip, color jitter | Every geometric transform must also transform the boxes, or your labels silently rot |

Read that table as a list of traps. The rest of the article is a guided tour out of each one, in the order you hit them when you build a real system.

## The mental model

Hold one picture in your head and everything else hangs off it. A detector is a function from an image to a *set* of predictions, and training is the process of making that predicted set match a ground-truth set. The diagram below is the mental model: the image flows through a backbone that extracts features, a neck that fuses those features across scales, and a head that emits candidate boxes with class scores; the resulting set of predictions is then matched against the ground-truth set, and only after matching can a loss or a metric be computed.

![Object detection as set prediction: an image flows through backbone, neck, and head to produce a set of box-label-score tuples, which a matcher scores against the ground-truth set](/imgs/blogs/object-detection-end-to-end-1.png)

The two boxes on the bottom row — *match to ground truth* and *loss / mAP* — are the ones with no analogue in classification, and they are where most of the conceptual difficulty lives. The matching step is the hinge of the entire field. Different detector families differ mostly in *how* they generate candidates and *how* they match those candidates to objects. Anchor-based detectors match by intersection-over-union (IoU) overlap between hand-placed priors and ground-truth boxes. DETR-family detectors match by solving a bipartite assignment problem with the Hungarian algorithm. Pick a family and you have largely picked your matching strategy, your post-processing, and your failure modes.

> If you remember one sentence from this article, make it this: object detection is set prediction, and everything hard about it is a consequence of the set being unordered and variable-length.

Everything that follows — the architecture, the data work, the losses, the metric — is a tour of that diagram, going deep on each box and the two arrows that have no classification equivalent.

## 1. The approach landscape

**Senior rule of thumb: you are not choosing an architecture, you are choosing how many hand-crafted components you are willing to maintain.** The history of detection is a steady deletion of hand-engineered machinery. Each generation removed a component that required tuning — region proposals, then anchors, then non-maximum suppression — and pushed that responsibility into the learned part of the network.

![The detector family tree: object detectors split into two-stage, one-stage, and set-prediction branches, each deleting a hand-crafted component](/imgs/blogs/object-detection-end-to-end-2.png)

The family tree above is worth reading branch by branch, because the branch you pick determines which knobs you will spend the next month tuning.

**Two-stage detectors** (Faster R-CNN, 2015) split the problem: a Region Proposal Network (RPN) suggests a few thousand class-agnostic boxes, then a second head crops features from each proposal (RoI Align) and classifies and refines them. Two stages buy accuracy on small and crowded objects because the second stage sees a tighter crop, but they pay in latency and in the complexity of training two coupled heads. In 2026 you reach for two-stage detectors mainly when recall on tiny objects is paramount and latency is not — medical imaging, satellite analysis, defect inspection.

The mechanism behind two-stage accuracy is worth naming precisely, because it explains a recurring tradeoff. The second stage uses RoI Align to crop and resample a fixed-size feature grid (typically 7x7) from *each* proposal, so every object — whether it occupies 20 pixels or 2000 — gets the same feature budget. That per-object normalization is exactly what helps small and crowded objects: a tiny object that would be one or two cells on the shared feature map instead gets a full 7x7 grid of resampled features. The price is that the classification head now runs once per proposal rather than once per image, and with a few hundred proposals that is a few hundred forward passes through the head. This is the latency that one-stage detectors refuse to pay, and it is the entire reason the field split into two branches in the first place.

**One-stage detectors** drop the proposal stage and predict densely over the feature map in a single pass. The anchor-based branch (RetinaNet, the YOLO line through v8/v11) tiles each feature-map location with a set of reference boxes ("anchors") of fixed scales and aspect ratios, then for each anchor predicts a class and an offset. The anchor-free branch (FCOS, CenterNet) deletes the anchors too: FCOS regresses, from each feature-map point, the four distances to the object's edges plus a center-ness score that down-weights points near the boundary. One-stage detectors are what most teams actually ship, because they hit the latency-accuracy knee that production demands.

**Set-prediction detectors** (DETR, 2020, and its descendants) delete the last hand-crafted component: non-maximum suppression. Instead of thousands of dense candidates that must be de-duplicated afterward, DETR carries a small fixed set of learned *object queries* — 100 to 300 of them — through a transformer decoder, and trains them so that exactly one query fires per object. No anchors, no NMS, no post-processing heuristics. The cost is slower convergence and a hunger for data and training compute, which the descendants (Deformable DETR, DINO, RT-DETR) progressively fixed.

Here is the landscape with the numbers that actually drive the decision, on COCO `val2017`:

| Family | Representative | COCO AP | Latency profile | Hand-crafted components |
|---|---|---|---|---|
| Two-stage | Faster R-CNN R50-FPN | ~40 | slow (RoI per proposal) | anchors, RPN, NMS |
| One-stage anchor | RetinaNet R50 | ~39 | medium | anchors, NMS |
| One-stage anchor | YOLOv8-L | ~53 | fast (real-time) | anchors-ish, NMS |
| One-stage anchor-free | FCOS R50 | ~41 | medium | center sampling, NMS |
| Set prediction | DETR R50 (500 ep) | ~42 | medium, no NMS | none |
| Set prediction | DINO Swin-L | ~63 | slow, no NMS | none |
| Set prediction | RT-DETR R50 | ~53 | fast (real-time), no NMS | none |

The two rows worth staring at are YOLOv8-L and RT-DETR-R50: roughly the same accuracy, roughly the same latency class, but RT-DETR carries no NMS. That single difference — a learned de-duplication instead of a hand-tuned suppression threshold — is why I now default new projects to the DETR family and treat YOLO as the baseline I have to beat, rather than the other way around.

### Second-order consequence: NMS is a hyperparameter you forget you have

Non-maximum suppression has an IoU threshold, and that threshold is a silent accuracy ceiling. Set it too high and you keep duplicate boxes; set it too low and you suppress genuine adjacent objects in crowded scenes. Every anchor-based detector ships with this knob, and it interacts with your data: a model tuned for sparse street scenes will merge people in a dense crowd. Set-prediction models make this failure mode disappear by construction, which is worth more in production than a point of AP.

## 2. The model

A detector, regardless of family, decomposes into three parts you can reason about independently. Get this decomposition straight and you can read any detection paper in fifteen minutes, because every paper is proposing a swap of one of these three parts.

### Backbone, neck, head

**Senior rule of thumb: the backbone learns *what*, the neck distributes *where*, and the head decides *how many and which*.** The backbone is a general-purpose feature extractor — a ResNet, a ConvNeXt, a Swin Transformer, or a plain Vision Transformer — usually pretrained on ImageNet or on a large image-text corpus. The neck fuses the backbone's features across resolutions so that a small object (resolved only in the high-resolution early layers) and a large object (resolved in the deep, semantically rich layers) both have a feature map that can detect them. The head is the per-location predictor that emits boxes and class logits.

![Backbone, neck, head laid out across three scales: backbone stages C3-C5 feed a top-down feature-pyramid neck whose P3-P5 levels each drive a per-scale detection head](/imgs/blogs/object-detection-end-to-end-3.png)

The diagram shows the canonical arrangement, the Feature Pyramid Network (FPN). The backbone produces feature maps at strides 8, 16, and 32 — call them C3, C4, C5 — where stride 8 means each feature-map cell summarizes an 8x8 pixel region, so C3 is high-resolution and C5 is coarse but semantically deep. The neck's top-down pathway takes the coarse, semantic C5, upsamples it, and adds it to C4, then repeats into C3. The result, the pyramid levels P3 through P5, gives every scale both spatial precision and semantic depth. P3 (stride 8) is where you detect the 12-pixel traffic sign; P5 (stride 32) is where you detect the bus. A shared head runs on each level.

This is also the single most important figure for understanding *why detectors fail on small objects*. If your backbone's highest-resolution usable feature map is stride 8, then an object smaller than roughly 8 pixels has *no cell that clearly belongs to it*, and no amount of training will recover it. The fix is architectural — add a P2 (stride 4) level, use a higher input resolution, or tile the image — not a matter of training longer. I have watched teams burn a week on hyperparameters for a small-object problem that was a stride-8 floor the whole time.

On backbones specifically: the field has largely moved to transformer and hybrid backbones, and the tradeoffs there deserve their own treatment. If you want the intuition for why a Vision Transformer sees an image differently from a ConvNet, and why DINO-pretrained features transfer so well to detection, I wrote a companion piece on [ViT, SigLIP, and DINO](/blog/machine-learning/computer-vision/vit-siglip-dino-explained); for the opposite end — why convolution's inductive biases made detection learnable in the first place — see [Why Convolution Works](/blog/machine-learning/computer-vision/why-convolution-works).

### Input resolution and the small-object floor

**Senior rule of thumb: input resolution is a first-class accuracy knob, and on small-object problems it routinely beats any architecture change you could make.** The reason follows directly from the stride argument above. An object that is 16 pixels wide in a 640-pixel input is 2 cells on a stride-8 feature map — barely detectable. Double the input to 1280 and that same object is now 32 pixels and 4 cells, with twice the spatial signal the head can use. Resolution does not change the network; it changes how many feature-map cells your smallest objects occupy, and below a floor of roughly 2 to 3 cells, detection collapses regardless of training.

The cost is quadratic: doubling each input dimension quadruples the activation memory and roughly quadruples the compute. So resolution is a budget you spend deliberately. A representative tradeoff on a small-object aerial dataset:

| Input size | AP_small | AP_all | Relative latency |
|---|---|---|---|
| 640 x 640 | 14 | 38 | 1.0x |
| 1024 x 1024 | 26 | 46 | 2.6x |
| 1536 x 1536 | 33 | 49 | 5.8x |

When even the largest input you can afford is not enough, the answer is **tiling**: split the high-resolution image into overlapping crops, detect on each crop at the model's native resolution, and merge the results back (the SAHI library packages this pattern). Tiling trades latency for an effective resolution far beyond what fits in memory, and on satellite and microscopy imagery it is frequently the difference between a working system and a useless one. The merge step needs care — an object split across a tile boundary appears in two crops and must be de-duplicated — but it is the standard tool for objects that are simply too small to resolve in one pass.

### Where the candidate boxes come from

The single biggest conceptual fork inside the head is *how candidate locations are generated*. There are three answers, and they determine your candidate count, your need for NMS, and whether you tune priors by hand.

![Comparison of candidate generation: anchor-based tiling, anchor-free per-point regression, and DETR's small set of learned object queries, with counts and NMS requirements](/imgs/blogs/object-detection-end-to-end-4.png)

The matrix lays out the three strategies. **Anchor-based** detectors place a fixed set of reference boxes at every feature-map location — typically 3 scales times 3 aspect ratios, so 9 anchors per cell. Across a feature pyramid that is on the order of 100,000 candidate boxes per image, the vast majority of which are background. Anchors are hand-set priors: you choose their scales and aspect ratios, ideally by clustering your dataset's box dimensions, and a mismatch between your anchors and your objects is a hard recall ceiling. **Anchor-free** detectors regress directly from each feature-map point to the object's extent, cutting the candidate count by roughly the per-cell anchor multiplier and removing the scale/aspect-ratio priors — but they still produce one prediction per point, so they still need NMS to collapse the many points that fire on one large object.

**Object queries** (DETR) are the qualitative break. Instead of tens of thousands of candidates derived from the geometry of the feature map, DETR carries a small, fixed number of learned query embeddings — 100 in the original, 300 in RT-DETR. Each query is a slot that learns, over training, to attend to a region of the image and either claim an object or declare itself empty. Because the count is small and the queries learn to specialize, there is no flood of duplicates to suppress.

### DETR: set prediction and bipartite matching

This is the mechanism that makes set prediction work, and it is worth slowing down for, because it is the part most people hand-wave and then misuse.

**Senior rule of thumb: DETR does not predict boxes and then remove duplicates; it makes producing a duplicate *expensive at training time*, via a loss that allows exactly one query per object.** The decoder emits, say, 100 predictions. Ground truth has, say, 3 objects. To compute a loss you must decide which 3 of the 100 predictions are responsible for the 3 objects, and the other 97 should predict "no object." DETR makes this assignment optimal rather than heuristic: it builds a cost matrix between every prediction and every ground-truth box — combining classification cost, an L1 box cost, and a generalized-IoU cost — and solves for the minimum-cost one-to-one assignment with the Hungarian algorithm.

![DETR bipartite matching: five object queries are each assigned to exactly one target by Hungarian matching, with surplus queries mapped to no-object so no NMS is required](/imgs/blogs/object-detection-end-to-end-5.png)

The figure makes the one-to-one structure concrete: five queries, three real objects, two "no-object" slots. The Hungarian algorithm finds the assignment that minimizes total cost, guaranteeing each ground-truth object is claimed by exactly one query. Surplus queries are pushed to the background class. Because the assignment is one-to-one and recomputed every step against the current predictions, the network is *penalized for redundancy*: if two queries both move toward the same object, only one can be matched, and the other takes a no-object penalty. Over training the queries spread out to cover the space of objects. That is why DETR needs no NMS — de-duplication is baked into the training objective.

Here is the matcher, which is the conceptual core, in runnable PyTorch:

```python
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou, box_convert

@torch.no_grad()
def hungarian_match(pred_logits, pred_boxes, tgt_labels, tgt_boxes,
                    w_cls=1.0, w_l1=5.0, w_giou=2.0):
    # pred_logits: [Q, C]   class logits for Q queries
    # pred_boxes:  [Q, 4]   (cx, cy, w, h), normalized to [0, 1]
    # tgt_*:       [T, ...] the T ground-truth objects in this image
    prob = pred_logits.softmax(-1)                      # [Q, C]
    cost_cls = -prob[:, tgt_labels]                     # [Q, T] want high prob for true class
    cost_l1 = torch.cdist(pred_boxes, tgt_boxes, p=1)   # [Q, T] L1 in box space
    cost_giou = -generalized_box_iou(                   # [Q, T] want high overlap
        box_convert(pred_boxes, "cxcywh", "xyxy"),
        box_convert(tgt_boxes, "cxcywh", "xyxy"),
    )
    cost = w_cls * cost_cls + w_l1 * cost_l1 + w_giou * cost_giou
    q_idx, t_idx = linear_sum_assignment(cost.cpu())    # one query per target
    return torch.as_tensor(q_idx), torch.as_tensor(t_idx)
```

The weights `w_l1=5.0` and `w_giou=2.0` are the DETR defaults and they matter: the L1 cost dominates early training (it has gradient everywhere), and the GIoU cost refines overlap once boxes are roughly right. Get the relative weighting wrong and the matcher becomes unstable, queries oscillate between objects, and convergence stalls.

And then in practice, you do not write this yourself. You stand on the Hugging Face implementation:

```python
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from PIL import Image

processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd").eval()

image = Image.open("street.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_object_detection(   # fixed 300-query set, no NMS step
    outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.5
)[0]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(f"{model.config.id2label[label.item()]:>14}  {score:.2f}  {[round(x) for x in box.tolist()]}")
```

Notice what is *absent*: there is no `nms_threshold`, no anchor configuration, no `max_detections` heuristic that secretly drops objects. The post-processing is a confidence threshold and nothing else. That simplicity is the deployment dividend of set prediction.

### Second-order consequence: DETR's slow convergence is a matching problem

The original DETR needed 500 epochs on COCO to converge, against ~36 for a Faster R-CNN. The cause is the matching: early in training the assignment is essentially random, so the supervision signal each query receives is noisy and inconsistent — a query matched to a person this step might be matched to a car the next. Deformable DETR fixed most of it by restricting attention to a small set of sampled points (cutting both compute and the search space), and DINO added denoising training and contrastive query selection to stabilize the assignment. If you ever find yourself training a vanilla DETR from scratch and despairing at epoch 50, the answer is not patience — it is to switch to a Deformable/DINO/RT-DETR variant whose matching converges in 12 to 50 epochs.

### Auxiliary losses and deep supervision

One more mechanism makes DETR-family training work, and skipping it will quietly cost you several points of AP: auxiliary losses on every decoder layer. The decoder is a stack of, typically, six layers, each refining the queries. Rather than computing the set loss only on the final layer's output, DETR runs the prediction head and the full Hungarian matching on the output of *every* decoder layer and sums the losses. This deep supervision gives every layer a direct gradient signal toward the correct set, which both speeds convergence and improves the final result, because the early layers learn to make useful coarse predictions that later layers refine rather than learning arbitrary intermediate representations.

RT-DETR adds two more pieces worth knowing. Its hybrid encoder decouples the expensive intra-scale attention from cheap cross-scale fusion, which is most of how it reaches real-time latency without losing the multi-scale features small objects need. And its IoU-aware query selection initializes the decoder queries from the encoder's most promising regions — picking initial query positions by predicted IoU rather than starting from learned-but-static embeddings — so the decoder begins from a warm start instead of cold queries. The practical takeaway when you finetune one of these models: leave the auxiliary losses and query-selection machinery on. They are not optional regularizers; they are load-bearing parts of why the model converges at all, and turning them off to "simplify" the loss is a classic self-inflicted wound.

## 3. The data is where the wins are

**Senior rule of thumb: after you have a reasonable architecture, every remaining point of AP is in the data, and you will get more from fixing labels and rebalancing classes than from any architecture swap.** I have never once been on a detection project where the model architecture was the bottleneck past the first week. The bottleneck was always annotation quality, class imbalance, or a domain gap nobody measured. This section is the one I wish every new hire read first.

### Getting your data into COCO format

Before any of the interesting data work, you have to load the data, and detection has a de-facto standard worth adopting rather than inventing your own: the COCO JSON format. It stores images, categories, and annotations as separate arrays, where each annotation carries an `image_id`, a `category_id`, a `bbox` as `[x_min, y_min, width, height]` in absolute pixels, and an `area`. Every tool in the ecosystem — `pycocotools` for evaluation, most training frameworks, most visualization tools — speaks it natively. Fighting the format costs you all of that interoperability.

A minimal but realistic PyTorch dataset that reads COCO and feeds the Albumentations pipeline from earlier looks like this:

```python
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

class CocoDetectionDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())   # stable order for reproducibility
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        info = self.coco.loadImgs(img_id)[0]
        img = np.array(Image.open(f"{self.img_dir}/{info['file_name']}").convert("RGB"))
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        boxes = [a["bbox"] for a in anns]           # [x, y, w, h], absolute pixels
        labels = [a["category_id"] for a in anns]
        out = self.transforms(image=img, bboxes=boxes, labels=labels)
        return out["image"], {
            "boxes": torch.as_tensor(out["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(out["labels"], dtype=torch.int64),
        }
```

Two details earn their place. Sorting `self.ids` makes the dataset order deterministic, which matters when you are debugging a reproducibility issue and need run N to match run N+1 exactly. And returning boxes and labels as a dict per image — rather than collating them into one ragged tensor — is the shape every detection loss expects, because the number of objects varies per image and cannot be stacked into a rectangular batch. Your `collate_fn` will keep the images as a batched tensor and the targets as a list of dicts. That list-of-dicts target shape is the detection-specific thing classification pipelines never prepare you for.

### The long tail and foreground-background imbalance

Detection data has two distributions working against you simultaneously, and they compound.

![The long tail of detection classes: a few head classes own most of the boxes while rare classes appear orders of magnitude less often yet count equally in mAP](/imgs/blogs/object-detection-end-to-end-7.png)

The first distribution is across classes, shown above. The counts are schematic — what is fixed across datasets is the *shape*. In COCO, `person` alone is roughly a quarter of all labeled instances, while the rare classes (`toaster`, `hair drier`) appear a few hundred to a few thousand times. Yet mAP averages over classes with equal weight: your `hair drier` AP counts exactly as much as your `person` AP. A model that is excellent on the head and hopeless on the tail posts a mediocre mAP, and the only way up is to fix the tail. More data for the head does nothing; you need targeted data, resampling, or augmentation for the rare classes.

The second distribution is *within* each image, between foreground and background. An anchor-based detector evaluates on the order of 100,000 candidate locations per image, of which perhaps a few dozen contain objects. That is a foreground-to-background ratio around 1:1000. Train with naive cross-entropy and the gradient is swamped by the easy background examples; the model's loss-minimizing move is to predict "background" everywhere, and it will. The two classic fixes:

- **Hard negative mining** (used by two-stage detectors): only backpropagate the highest-loss background examples, at a fixed positive:negative ratio like 1:3.
- **Focal loss** (RetinaNet's contribution): reshape cross-entropy with a modulating factor $(1 - p_t)^\gamma$ that down-weights well-classified examples, so easy backgrounds contribute almost nothing to the gradient while hard examples dominate. With $\gamma = 2$, an example the model already calls background with 0.9 confidence has its loss scaled by $(1 - 0.9)^2 = 0.01$.

The mathematics is worth writing once. Standard cross-entropy for the true-class probability $p_t$ is $\mathrm{CE}(p_t) = -\log(p_t)$. Focal loss is:

$$\mathrm{FL}(p_t) = -\alpha_t \, (1 - p_t)^{\gamma} \log(p_t)$$

The $(1 - p_t)^\gamma$ term is the whole idea: when $p_t \to 1$ (an easy, well-classified example), the factor goes to zero and the example stops dominating. DETR-family detectors sidestep foreground-background imbalance differently — their fixed query set has only ~100 predictions, so the imbalance is ~100:3 instead of 100,000:3, and a plain weighted cross-entropy handles it.

For the *class* imbalance, the loss is only half the fix; the sampler is the other half. Repeat-factor sampling (from the LVIS long-tail benchmark) oversamples images that contain rare classes, with a repeat factor computed from each class's image frequency so that rare classes appear often enough to learn from while common classes are not starved. It is a few lines wrapped around your sampler and it composes with copy-paste: copy-paste manufactures rare-class instances, and repeat-factor sampling makes sure the model sees them often. On long-tail problems I reach for both before touching the architecture, because together they move the rare-class AP far more than any backbone swap.

### Augmentation that respects boxes

**Senior rule of thumb: in detection, augmentation is not a regularizer you sprinkle on — it is your primary tool for fixing the data distribution, and every transform must carry the boxes along or it corrupts your labels.** This is the clause classification engineers forget. A random crop that cuts a person in half must update that person's box to the visible portion, or drop it if too little remains. The library that gets this right and that I reach for by default is Albumentations, because its `bbox_params` propagate every geometric transform to the boxes and drop boxes that fall below a visibility threshold:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_tf = A.Compose(
    [
        A.RandomResizedCrop(size=(640, 640), scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),
        A.ToGray(p=0.01),
        A.Normalize(),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="coco",                 # boxes as [x_min, y_min, width, height]
        label_fields=["labels"],
        min_visibility=0.2,            # drop a box once <20% of it survives a crop
    ),
)

out = train_tf(image=img, bboxes=boxes_xywh, labels=class_ids)
boxes, labels = out["bboxes"], out["labels"]   # stay consistent with out["image"]
```

Two detection-specific augmentations earn their keep beyond the standard geometric and photometric ones:

- **Mosaic** (from YOLOv4) stitches four images into one 2x2 grid, then crops. It quadruples the object density per training image, exposes the model to objects at the tile boundaries, and dramatically improves small-object and context robustness. It is so effective that most YOLO training schedules turn it *off* for the final 10 to 20 epochs, because its unnatural compositions start to hurt once the model has learned the easy gains.
- **Copy-paste** (Ghiasi et al., 2021) cuts object instances out of one image and pastes them into another, with proper box bookkeeping. It is the single most effective intervention I know for the long tail: take your 200 `hair drier` instances, paste them into thousands of varied scenes, and you have manufactured the rare-class data the annotation budget never bought you. On one shelf-auditing project, copy-paste of the rarest 8 SKUs moved their AP from the low 20s to the high 40s without a single new annotation.

### Second-order consequence: label quality sets a ceiling no model can pierce

Annotators disagree about box tightness. One person draws the box at the visible edge of the object; another includes the shadow; a third snaps to the bounding rectangle of a partially occluded shape. That disagreement is *noise on the regression target*, and it caps achievable IoU. If your annotators agree to within IoU 0.8 of each other, no model will reliably beat AP75 on that data, because the ground truth itself is only consistent to 0.8. Before you chase a stalled AP75, measure annotator agreement on a re-labeled subset. I have seen a "model problem" dissolve the instant we discovered the label problem.

## 4. Training and finetuning

With architecture and data in hand, training is where the set-prediction nature of the problem turns into a specific loss recipe and a specific schedule. Both have detection-only subtleties.

### The loss cocktail

A detection loss is never a single term. It is a weighted sum of a classification loss (focal or cross-entropy) and a box-regression loss, and for set-prediction models the whole thing is computed only on the matched pairs from the Hungarian assignment. The box-regression term has its own evolution, and understanding it explains a lot of otherwise-mysterious training behavior.

![Why box loss evolved past smooth-L1: coordinate regression ignores overlap and scale, while IoU-family losses optimize the overlap the evaluator actually measures](/imgs/blogs/object-detection-end-to-end-6.png)

The progression in the figure is a story of aligning the loss with the metric. The earliest detectors regressed box coordinates with **smooth-L1** loss on $(x, y, w, h)$. Two problems: it is *scale-dependent* (a 10-pixel error on a 500-pixel box matters as much as on a 20-pixel box, which is wrong), and it optimizes coordinates the evaluator never looks at — the evaluator measures IoU. So the field moved to **IoU loss**, which optimizes overlap directly and is scale-invariant. But plain IoU loss has a fatal gradient hole: when the predicted and ground-truth boxes do not overlap at all, IoU is zero *and so is its gradient*, so the loss cannot tell a near-miss from a wild miss and provides no signal to move the box closer.

**GIoU** (Generalized IoU) plugs the hole by adding a penalty based on the smallest enclosing box $C$ that contains both:

$$\mathrm{GIoU} = \mathrm{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}$$

When the boxes do not overlap, IoU is 0 but the enclosing-box term still has gradient — it pulls the prediction toward the target. **DIoU** adds a normalized center-distance term so boxes converge faster, and **CIoU** adds an aspect-ratio consistency term so the predicted box matches the target's shape, not just its center and area. In practice, RT-DETR and most modern detectors use an L1 term for early-training gradient plus a GIoU or CIoU term for metric alignment — the L1 gets you in the neighborhood, the IoU-family term tightens the fit.

A representative total loss for a DETR-style model, per matched pair, is:

$$\mathcal{L} = \lambda_{\text{cls}} \, \mathrm{FL}(\hat{c}, c) + \lambda_{\text{L1}} \, \lVert \hat{b} - b \rVert_1 + \lambda_{\text{giou}} \, (1 - \mathrm{GIoU}(\hat{b}, b))$$

with typical weights $\lambda_{\text{cls}} = 1$, $\lambda_{\text{L1}} = 5$, $\lambda_{\text{giou}} = 2$. The no-object queries contribute only the classification term, usually down-weighted so the abundant background does not dominate.

### Finetuning a pretrained detector

You will almost never train a detector from scratch. You will start from COCO-pretrained weights and adapt to your 12 classes. Doing this well is a *staged* process, and the most common mistake is to unfreeze everything at full learning rate and watch the pretrained backbone features get destroyed in the first hundred steps.

![Finetuning a pretrained detector as a staged unfreeze: freeze the backbone and warm up the head, then unfreeze with a smaller backbone learning rate and select the best checkpoint by EMA](/imgs/blogs/object-detection-end-to-end-10.png)

The schedule in the figure is the recipe that has never let me down:

1. **Freeze the backbone, warm up the new head.** Your new classification head is randomly initialized and produces enormous, noisy gradients. If the backbone is unfrozen, those gradients flow back and corrupt the very features you are trying to reuse. Freeze it for the first few epochs while a linear learning-rate warmup ramps from 0 to the target.
2. **Unfreeze with a differential learning rate.** Once the head is sane, unfreeze the backbone but give it a learning rate 10x smaller than the head's. The backbone needs gentle adaptation; the head needs aggressive learning. One global learning rate cannot serve both.
3. **Let EMA pick the checkpoint.** Maintain an exponential moving average of the weights and *always evaluate and deploy the EMA weights*, not the raw ones. The EMA is smoother, generalizes better, and is far less sensitive to the exact epoch you happen to stop on.

Here is the loop, compressed but runnable in shape:

```python
import torch
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

model = build_detector(num_classes=12).cuda()

for p in model.backbone.parameters():
    p.requires_grad = False   # Stage 1: freeze backbone so the random head can't corrupt it

opt = AdamW(                  # differential LR: backbone adapts gently, head learns fast
    [
        {"params": model.backbone.parameters(),    "lr": 1e-5},
        {"params": model.transformer.parameters(), "lr": 1e-4},
        {"params": model.head.parameters(),        "lr": 1e-4},
    ],
    weight_decay=1e-4,
)
ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

for epoch in range(40):
    if epoch == 3:                                   # Stage 2: unfreeze
        for p in model.backbone.parameters():
            p.requires_grad = True
    for images, targets in train_loader:
        loss = model(images.cuda(), targets)         # set loss computed inside
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        opt.step()
        opt.zero_grad()
        ema.update_parameters(model)
    evaluate(ema.module, val_loader)                 # validate the EMA, always
```

The `clip_grad_norm_(..., 0.1)` is not optional for DETR-family training — the matching makes the loss landscape spiky, and an unclipped gradient spike can blow up the queries' learned specialization in a single step. The value 0.1 is aggressive by classification standards and exactly right here.

### Second-order consequence: your learning rate must track your batch size and your hardware

Detectors are trained at small effective batch sizes because the images are large and the memory cost is high. If you scale from 8 GPUs to 2, your batch size drops, and a learning rate tuned for the larger batch will now be too high relative to the gradient noise. The linear scaling rule — halve the batch, halve the learning rate — is the right first move. And if your bottleneck is feeding those large images to the GPU fast enough, the data pipeline itself becomes the training-speed limiter; I wrote separately about [speeding up the CPU-to-GPU path](/blog/machine-learning/training-techniques/speeding-up-neural-network-training-4x-by-optimizing-cpu-to-gpu-data-transfer), which bites detection harder than most workloads because the inputs are big.

## Anatomy of a training run: what to monitor

**Senior rule of thumb: a falling total loss tells you almost nothing in detection; you must watch the components, the gradient norm, and the validation AP curve separately, because each fails in a different way.** A classification training run is largely summarized by its loss and accuracy curves. A detection run has at least four signals that move independently, and a problem in any one is invisible in the aggregate. These are the dashboards I wire up on day one of every project:

| Signal | What healthy looks like | What it catches |
|---|---|---|
| Classification loss | Falls steadily, then plateaus | Foreground-background imbalance (it falls while detections never appear) |
| Box / L1 loss | Falls fast early, then slow | A box-regression head that is not learning, or a label-scale bug |
| GIoU loss | Falls after the L1 loss settles | Localization quality stalling; pairs with a low AP75 |
| Gradient norm (pre-clip) | Stable, occasional spikes | Matching instability — frequent large spikes mean queries are oscillating |
| Validation AP curve | Rises, with EMA above raw | Overfitting (raw AP rises while EMA AP plateaus), or a leak (AP unrealistically high early) |

The most detection-specific of these is the gradient-norm trace. Because the bipartite matching reassigns predictions to objects every step, an unstable run shows up as a forest of gradient spikes long before the AP curve flattens — the queries are fighting over objects and the supervision target for each query keeps flipping. When I see that pattern, the fixes in order are: lower the learning rate, increase the gradient-clip aggressiveness, and if it persists, switch to a denoising-trained variant (DINO-style) whose matching is stabilized by construction.

The second non-obvious signal is the gap between raw-weight AP and EMA-weight AP. Early in training the EMA lags (it is averaging in the poor early weights), so raw AP is higher. As training matures the EMA should overtake the raw weights and stay ahead — that crossover is the sign your model has entered the stable regime where the EMA is denoising the optimization trajectory. If the EMA never overtakes the raw weights, your EMA decay is wrong for your schedule length: 0.999 wants thousands of steps to warm up, and on a short finetune you may need 0.999 or a warmup on the decay itself.

A worked example makes the imbalance signal concrete. Suppose after one epoch the total loss has dropped 40% but you have zero detections at inference. Break it down: if the classification loss accounts for 90% of that drop and the box losses are flat, the model is winning by predicting background better, not by finding objects. The total looks healthy; the components reveal the pathology. This is the single most common "training looks fine but the model does nothing" failure, and you can only see it by logging the loss terms separately.

## 5. Evaluation, and why mAP lies

**Senior rule of thumb: mAP is a research-leaderboard metric, and the number that goes up on the leaderboard is frequently not the number your users feel. Always pair mAP with an error decomposition and an operating-point analysis.** This is the section that separates engineers who ship working systems from engineers who ship high-mAP systems that fail in the field.

### How mAP is actually computed

You cannot reason about a metric you cannot compute by hand, so build it up. Start with one class. A prediction is a *true positive* if it matches a ground-truth box of that class with IoU above a threshold (and that ground-truth box has not already been claimed by a higher-scoring prediction); otherwise it is a *false positive*. Sort all predictions for the class by confidence, walk down the list, and at each point compute precision (fraction of predictions so far that are true positives) and recall (fraction of ground-truth objects found so far). Plotting precision against recall traces the precision-recall curve.

![How average precision reads off the precision-recall curve, and how mean average precision averages that area across classes and IoU thresholds](/imgs/blogs/object-detection-end-to-end-8.png)

**Average precision (AP)** for that class is the area under its precision-recall curve, shown shaded. It rewards a detector that maintains high precision as recall increases — that finds objects without flooding the output with false positives. **Mean average precision (mAP)** then averages AP in two directions: over all classes (so the rare ones count), and — this is the COCO-specific part that trips people up — over ten IoU thresholds from 0.50 to 0.95 in steps of 0.05. That last average is why COCO mAP is so much lower than the old PASCAL VOC numbers: PASCAL used a single loose IoU threshold of 0.5, where a sloppy box still counts; COCO demands tight localization at 0.75, 0.85, 0.95 as well.

This is the first place mAP lies. **AP50** (the loose threshold alone) almost always flatters your model, because it rewards finding the object even with a sloppy box. If you report AP50 and your users need tight boxes — for measuring, for cropping, for robotic grasping — you are reporting a number that does not reflect their experience. Report the full `AP@[.50:.95]`, and break out AP75 separately so you can see your *localization* quality distinct from your *detection* quality.

Computing this correctly by hand is a minefield of off-by-one and tie-breaking bugs, so use the reference implementation:

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_gt = COCO("instances_val2017.json")
coco_dt = coco_gt.loadRes(predictions_json)   # [{image_id, category_id, bbox, score}, ...]

ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
ev.evaluate()
ev.accumulate()
ev.summarize()

metrics = {
    "mAP@[.50:.95]": ev.stats[0],   # the headline number
    "AP50":          ev.stats[1],   # the flattering number
    "AP75":          ev.stats[2],   # localization quality
    "AP_small":      ev.stats[3],   # the one that hides small-object failure
    "AP_medium":     ev.stats[4],
    "AP_large":      ev.stats[5],
}
```

The per-size breakdown (`AP_small`, `AP_medium`, `AP_large`) is the second thing to wire into every dashboard. A single mAP can hide that your detector is fine on medium and large objects and catastrophic on small ones — which, given the stride floor discussed earlier, is the most common real-world failure.

There is a complementary metric COCO reports that gets ignored and should not: **average recall (AR)**, the recall averaged over IoU thresholds given a fixed budget of detections per image (AR@1, AR@10, AR@100). Where AP asks "of the boxes you emitted, how many were right and how well-ranked," AR asks "of the objects that exist, how many did you find at all." The two diverge in a diagnostic way. If your AP is mediocre but your AR@100 is high, your detector is *finding* the objects but ranking false positives above true ones or localizing loosely — a precision and ranking problem, fixable with better confidence calibration or box loss. If your AR@100 itself is low, you have a *recall ceiling*: objects the model never proposes at any confidence, which points back to anchors, query count, or resolution. Reading AP and AR together tells you which half of the problem you have before you spend a week on the wrong half. I treat a low AR@100 as the single most urgent signal on the dashboard, because no amount of threshold tuning recovers an object the model never detected.

### The six errors mAP hides

A scalar cannot tell you *why* you are losing points. For that you need an error decomposition, and the framework worth adopting is TIDE (Bolya et al., 2020), which attributes every lost point of AP to one of a small set of error types.

![The six errors a single mAP number hides: localization, classification, duplicate, background false positive, and missed detection, each with its symptom and usual fix](/imgs/blogs/object-detection-end-to-end-9.png)

The decomposition in the figure turns "mAP is 41, I want 48" into an actionable list:

- **Localization** error: right class, but IoU between 0.1 and 0.5 — the box is loosely placed. Symptom: AP75 far below AP50. Fix: box-loss weighting, and check label tightness.
- **Classification** error: well-placed box, wrong class. Symptom: confusion between visually similar classes. Fix: more data and hard negatives for the confused pair.
- **Duplicate** error: two predictions on one object. Symptom: precision drops. Fix: NMS threshold, or — better — move to a set-prediction model that cannot produce duplicates.
- **Background false positive**: a confident box on empty texture. Symptom: precision drops, often badly. Fix: focal loss, hard-negative mining, and harder background crops in training.
- **Missed detection**: an object with no box at all. Symptom: a recall ceiling you cannot train past. Fix: anchors/queries that cover the object's scale, better sampling, higher resolution.

Run TIDE on your validation set and the next month of work plans itself. If 60% of your lost AP is localization, you do not need more data — you need a better box loss or tighter labels. If it is background false positives, you need harder negatives, not a bigger backbone. I have watched a team spend a quarter collecting data for a problem TIDE would have diagnosed in an afternoon as a localization issue.

### Second-order consequence: the operating threshold is where mAP and reality diverge

mAP integrates over *all* confidence thresholds. Your production system runs at *one* threshold. A model with higher mAP can be worse at your chosen operating point, because mAP rewards the whole curve while you live at a single point on it. Always, before shipping, fix the confidence threshold your application will use, and measure precision and recall *at that threshold*. Then calibrate: a detector's raw confidence scores are usually not probabilities, and "0.5 confidence" rarely means "50% likely correct." If downstream logic thresholds on confidence, that miscalibration becomes silent errors.

### Calibration and per-class thresholds

Detectors are systematically overconfident, and the bias is not uniform across classes. A detector trained with focal loss in particular tends to produce scores that cluster near 0 and 1 rather than spreading across the range, because focal loss reshapes the objective away from probability calibration. The practical consequence: a single global confidence threshold is the wrong tool. The score that corresponds to 90% precision for `person` (a class with millions of training examples) is not the score that corresponds to 90% precision for `hair drier` (a class with a few hundred). Tune one threshold per class against your validation set, targeting a fixed precision or a fixed point on each class's PR curve, and store those thresholds alongside the model.

To recover trustworthy probabilities — which you need if any downstream logic reasons about confidence numerically, like a Bayesian fusion step or a cost-sensitive decision — fit a simple post-hoc calibrator. The cheapest effective choice is temperature scaling: learn a single scalar $T$ that divides the logits before the softmax, fit by minimizing negative log-likelihood on a held-out set. It does not change the ranking of detections (so mAP is unchanged) but it spreads the scores back toward calibrated probabilities. When per-class miscalibration is severe, fit per-class isotonic regression instead. Either way, the discipline is the same: measure the reliability of your scores before any system trusts them as probabilities, because a detector's raw confidence is a ranking signal, not a probability.

## Cross-cutting: shipping the detector

Training is half the job; the other half is making the thing run inside a latency budget, and detection has deployment subtleties that classification does not.

**End-to-end latency is not the backbone's FLOPs.** A profiler on a deployed anchor-based detector frequently shows that NMS — which runs on CPU, scales poorly with the number of candidates, and is hard to batch — is a serious fraction of wall-clock time, sometimes rivaling the network forward pass on crowded images. This is, again, a structural advantage of set-prediction models: with no NMS, the post-processing is a constant-time threshold, and the whole pipeline is a clean tensor-in, tensor-out graph that exports without a CPU detour.

Export to a serving runtime usually means ONNX, then a TensorRT engine:

```bash
python export_onnx.py --model rtdetr_r50vd --opset 17 --out rtdetr.onnx   # ONNX, opset 17

trtexec --onnx=rtdetr.onnx \
        --fp16 \
        --saveEngine=rtdetr_fp16.plan \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:8x3x640x640 \
        --maxShapes=images:16x3x640x640
```

For a YOLO-family model the same export has to bolt on an `EfficientNMS_TRT` plugin, configure its IoU and score thresholds, and keep those in sync with what you validated against — a step that is easy to get subtly wrong. RT-DETR's graph has none of that.

**The FP16 trap.** Half-precision export shifts confidence scores by small amounts, and small amounts matter near your operating threshold. A box that scored 0.51 in FP32 might score 0.49 in FP16 and vanish below your threshold. The discipline: re-run your full validation set *through the exported engine*, not the PyTorch model, and re-tune the confidence threshold against the engine's outputs. I have seen a "successful" deployment quietly drop 3 points of recall because nobody re-validated after the FP16 conversion.

**Throughput is a batching and preprocessing problem as much as a model problem.** A detector that runs in 8 ms on a single image does not necessarily serve 125 images per second, because preprocessing (decode, resize, normalize) and the host-to-device copy can dominate at batch size 1. Two levers usually matter more than shaving FLOPs off the backbone: batch inference so the GPU stays saturated (dynamic batching in Triton or a custom batcher), and move preprocessing onto the GPU (NVIDIA DALI or a CUDA resize) so the CPU stops being the bottleneck. On one pipeline, moving the resize and normalize from CPU NumPy to a GPU kernel lifted end-to-end throughput 2.3x with the model itself untouched. INT8 quantization is the next lever when you need it: it roughly halves latency again versus FP16 on tensor-core hardware, but it requires a calibration pass over representative images to set the activation ranges, and — like FP16, only more so — it shifts scores enough that you must re-validate and re-threshold against the quantized engine. Treat each precision step as producing a new model with its own operating point.

If your detector is one stage in a larger video or tracking pipeline, also consider whether you even need per-frame detection, or whether a detector every N frames plus a tracker is cheaper. The same scale-and-resolution thinking applies to the sibling task of [pose estimation](/blog/machine-learning/computer-vision/training-pose-estimation-models), which shares the backbone-neck-head decomposition and most of the data discipline described here.

## Case studies from production

These are real failure shapes, lightly anonymized. Each follows the same arc: the symptom, the wrong first hypothesis, the actual root cause, the fix, and the lesson. They are the fastest way to inoculate yourself against the mistakes that cost weeks.

### 1. The phantom 0.9 mAP

A team reported an astonishing 0.91 mAP on an internal dataset and wanted to ship immediately. In production the model was mediocre. The wrong first hypothesis was distribution shift between the lab and the field. The actual root cause was a data-splitting bug: the dataset contained near-duplicate frames from the same video clips, and the random split scattered frames from a single clip across both train and validation. The model had effectively seen the validation images during training. The fix was to split by *clip* (group-aware splitting), after which validation mAP fell to a believable 0.58 — which matched field performance exactly. The lesson: in detection, where data often comes from video, a random image-level split is almost always a leak. Split by the coarsest unit of correlation — clip, camera, location, day — and trust a lower honest number over a high dishonest one.

### 2. NMS ate the small objects

A pedestrian detector for a crowd-monitoring product worked beautifully in testing and failed in dense crowds, systematically undercounting. The wrong hypothesis was that the model could not detect small or distant people. The actual root cause was the NMS IoU threshold: at 0.45, two genuinely adjacent people whose boxes overlapped by more than 0.45 were collapsed into one detection. The model was *finding* both; NMS was *deleting* one. Raising the threshold helped the crowd case but reintroduced duplicates in sparse scenes — the classic NMS bind. The real fix was to switch to RT-DETR, whose set prediction has no suppression step and no threshold to trade off. Counting error in dense crowds dropped by half. The lesson: when objects of the same class legitimately overlap, NMS is a structural liability, and that is the strongest single argument for set-prediction detectors.

### 3. The anchor that never matched

A detector for elongated objects — pipes and cables in industrial imagery — had a hard recall ceiling around 0.7 that no amount of training moved. The wrong hypothesis was insufficient data. The actual root cause was anchor configuration: the default anchors had aspect ratios of 0.5, 1.0, and 2.0, but the cables had aspect ratios of 8 or more. No anchor ever achieved enough IoU with a cable to be assigned as a positive during training, so those objects contributed no positive gradient — they were *unlearnable* with that anchor set. The fix was to cluster the dataset's box dimensions with k-means and set anchors to the resulting aspect ratios, including the extreme ones. Recall jumped to 0.92. The lesson: anchor priors are part of your data, not your architecture. If your objects have unusual shapes, derive your anchors from your boxes, or use an anchor-free or query-based detector that has no shape priors to mismatch.

### 4. Background won the loss

A detector trained on a new dataset converged to predicting nothing — every image returned zero detections, and the loss looked like it was decreasing nicely. The wrong hypothesis was a learning-rate problem. The actual root cause was foreground-background imbalance with a plain cross-entropy classification loss: with a 1:1000 positive-to-negative ratio, "predict background everywhere" is a strong local minimum, and the model found it immediately. The decreasing loss was the background term shrinking while the foreground term was ignored. The fix was to swap in focal loss with $\gamma = 2$, which down-weights the easy backgrounds; detections appeared within two epochs. The lesson: if your detector predicts nothing and the loss still drops, you have an imbalance problem, not an optimization problem. Inspect the loss *per component* — a falling total can hide a foreground term that never moves.

### 5. mAP went up, complaints went up

An update improved validation mAP from 0.44 to 0.47, the team shipped it, and user complaints about false positives *increased*. The wrong hypothesis was a deployment or threshold bug. The actual root cause was that the new model gained its mAP in the high-recall, low-precision region of the curve — it found more objects, but at the operating threshold the product used (a relatively high-confidence cut), it also emitted more confident false positives. mAP, integrating over the whole curve, rewarded the gain; users, living at one threshold, felt the loss. The fix was to optimize and report precision and recall *at the production threshold*, and to choose the model that won there even though it had lower mAP. The lesson: mAP is the wrong objective for a product that runs at a fixed threshold. Define your operating point first, then select models by their behavior at that point.

### 6. The label-noise ceiling

A document-layout detector plateaued at AP75 around 0.6 despite a large, clean-looking dataset, while AP50 was a healthy 0.88. The wrong hypothesis was that the box-regression head needed more capacity. The actual root cause surfaced when we had three annotators re-label the same 200 pages: their boxes agreed with each other only to about IoU 0.82 on text blocks, because "where does a paragraph's box end" is genuinely ambiguous when there is trailing whitespace. The ground truth was inconsistent at exactly the IoU range AP75 measures, so no model could score well there — it would have to be more consistent than the labels. The fix was a labeling guideline that pinned box edges to ink, not whitespace, plus a re-label of the worst pages. AP75 rose to 0.78. The lesson: measure inter-annotator agreement before chasing high-IoU metrics. The labels set a ceiling, and you cannot train past the noise in your own targets.

### 7. DETR's slow convergence panic

A team adopted vanilla DETR, trained for 50 epochs, saw AP stuck in the low 20s, and concluded that transformer detectors "do not work for our data." The wrong hypothesis was a fundamental unsuitability of the architecture. The actual root cause was simply DETR's notorious convergence schedule: the original needs hundreds of epochs because early bipartite matching is unstable, and 50 epochs is barely past the warm-up phase. The fix was to switch to a Deformable-DETR-based variant (RT-DETR), which converges in roughly 50-72 epochs to competitive AP because its sparse attention and improved query selection stabilize the matching early. The same data hit AP 0.51 in 60 epochs. The lesson: know the convergence profile of your detector family before you judge it. Vanilla DETR's slowness is a known, solved property, not a verdict on set prediction.

### 8. TensorRT changed the numbers

A model validated at 0.49 mAP in PyTorch deployed at a measurably worse operating point, with recall down several points. The wrong hypothesis was a preprocessing mismatch between training and serving (a common and worthy suspect). The actual root cause was the FP16 TensorRT conversion: half precision shifted confidence scores by small margins, and because the production threshold sat at 0.5, a band of borderline true positives that scored just above 0.5 in FP32 now scored just below it in FP16 and were dropped. The fix was to re-run the full validation set through the TensorRT engine, observe the score shift, and lower the operating threshold to 0.45 to recover the recall. The lesson: an exported engine is a different model. Validate against the artifact you deploy, never against the framework you trained in, and expect to re-tune thresholds after every precision change.

### 9. Copy-paste saved the rare class

A retail detector had to recognize 80 product SKUs, eight of which were new and had fewer than 250 annotated instances each. Those eight sat at AP in the low 20s and dragged the mean down. The wrong hypothesis was that the only fix was a costly annotation campaign for the rare SKUs. The actual root cause was simply data volume for the tail, and it had a cheaper fix than new annotation: copy-paste augmentation. We segmented the existing rare-SKU instances, pasted them at varied scales and positions into thousands of real shelf backgrounds with correct box bookkeeping, and weighted those classes up in sampling. The eight SKUs rose to high-40s AP, lifting overall mAP by four points, with zero new labels. The lesson: for the long tail, manufactured data from copy-paste often beats collected data per dollar. Exhaust augmentation before you open the annotation budget.

### 10. The domain gap nobody measured

A detector pretrained on COCO and finetuned on a few thousand daytime street images was deployed on a fleet that also drove at night and in rain, and performance fell off a cliff after dark. The wrong hypothesis was that the model needed a bigger backbone to be "more robust." The actual root cause was a domain gap that was never in the validation set: the validation images were all daytime, so every metric looked fine while the nighttime distribution was completely unrepresented. The model was not weak; it was being tested on the wrong distribution. The fix was twofold: build a validation set stratified by lighting and weather so the gap became *visible*, then targeted data collection and heavy photometric augmentation (low-light simulation, synthetic rain) for the underrepresented conditions. The lesson: a metric is only as honest as the distribution it is computed over. If your validation set does not contain your deployment conditions, your mAP is measuring a world your users do not live in.

### 11. The resolution the preprocessing quietly threw away

A detector for small printed defects validated well but missed the smallest defects in production, and only the smallest ones. The wrong hypothesis was that the model lacked the capacity for fine detail. The actual root cause was a preprocessing mismatch: the training pipeline resized inputs to 1280 on the long side, but the serving wrapper, written by a different team, used a default that resized to 800. The smallest defects, marginal even at 1280, fell below the stride floor at 800 and became literally unresolvable — a 9-pixel defect at 1280 was a 5-pixel smear at 800, fewer than two cells on the feature map. The fix was a single line aligning the serving resize to the training resize, and the smallest-defect recall went from 0.3 to 0.8 instantly. The lesson: preprocessing is part of the model. The resize, the interpolation mode, the normalization constants, and the input size must be byte-for-byte identical between training and serving, and a resolution mismatch hits exactly the small objects that have no margin to spare. Pin these values in one shared config and assert them at serving startup.

### 12. The class that was three classes

A wildlife detector kept confusing a single animal class with the background and with two other species, and adding data barely helped. The wrong hypothesis was that the model needed more examples of the confused class. The actual root cause emerged from a TIDE analysis paired with a look at the raw annotations: the "deer" class actually contained three visually distinct sub-populations — adult deer, fawns, and deer seen only as a head behind foliage — that had been merged under one label. The model was being asked to map three different appearances to one class while keeping them separate from background, and the fawn and occluded-head modes were so different from the adult mode that the shared classifier could not cover all three. Splitting the label into `deer_adult`, `deer_fawn`, and `deer_occluded` for training (and optionally merging them back for reporting) let each mode get a coherent decision boundary; confusion dropped and AP rose six points. The lesson: a class is a modeling assumption, not a given. When one class is unusually hard, inspect whether it is secretly several appearance modes wearing one label, and let the model see the structure that is actually in the data.

## When to reach for DETR, and when YOLO is the right call

After all of this, the practical question is which family to start with on a new project. My defaults in 2026:

**Reach for the DETR family (RT-DETR, DINO) when:**

- Objects of the same class legitimately overlap or crowd — pedestrians, cells, packed retail shelves — where NMS is a structural liability.
- You want a clean export with no NMS plugin and no suppression threshold to maintain across training and serving.
- You have enough data and training compute to feed a transformer detector, or you can finetune from strong pretrained weights rather than training from scratch.
- Tight localization (AP75) matters and you want the box-quality benefits of IoU-aware query selection.

**Reach for the YOLO family when:**

- You need the absolute best latency on edge hardware today, with a mature, heavily optimized export and runtime ecosystem.
- Your team already operates a YOLO pipeline and the marginal accuracy of switching does not justify the migration cost.
- You are prototyping and want a single command from dataset to a deployable model, with the broadest community support and tooling.

**Skip heavy detectors entirely when:**

- The objects are large and always present — a whole-image classifier or a fixed-grid heuristic may be all you need, and a detector is overkill.
- You can constrain the problem geometrically (a fixed camera, known object positions) so that detection collapses to a much cheaper task.
- Latency and cost dominate and approximate counts suffice — a lightweight density-estimation or segmentation approach can be cheaper than per-object boxes.
- You only need detection occasionally in a video stream — detect every N frames and track in between, rather than running a detector on every frame.

The meta-point: the architecture is the least of your worries. Pick a sensible modern detector — I default to RT-DETR — and then spend your real effort on the data, the loss alignment, the staged finetune, and an honest evaluation that pairs mAP with an error decomposition and an operating-point analysis. That is where working detectors are actually made.

One forward-looking note, because it changes the calculus for some projects. The frontier is moving toward **open-vocabulary detection** — models like Grounding-DINO, OWL-ViT, and YOLO-World that take a text prompt and detect categories they were never explicitly trained on, by grounding a vision-language embedding into the detection head. For a fixed, known set of classes with a real annotation budget, a finetuned closed-set detector like RT-DETR still wins on accuracy and latency, and everything in this article applies unchanged. But when the class list is open-ended, changes often, or has classes with near-zero data, an open-vocabulary model used zero-shot — or used as an auto-labeler to bootstrap annotations for a closed-set detector — can collapse weeks of data work into an afternoon. It does not replace the discipline here; it changes *where* you spend it, shifting effort from per-class annotation toward prompt engineering and threshold calibration. Watch that space, but do not let it distract you: for most production problems in 2026, a well-trained closed-set detector with honest evaluation is still the right answer.

## Further reading

- **DETR** — Carion et al., *End-to-End Object Detection with Transformers* (2020): the set-prediction and bipartite-matching foundation.
- **Deformable DETR** — Zhu et al. (2021): the fix for DETR's slow convergence via sparse, multi-scale attention.
- **DINO** — Zhang et al. (2022): denoising training and contrastive query selection, the basis of current state-of-the-art.
- **RT-DETR** — Zhao et al. (2023): real-time DETR that matches and beats YOLO without NMS.
- **Focal Loss / RetinaNet** — Lin et al. (2017): the canonical treatment of foreground-background imbalance.
- **Feature Pyramid Networks** — Lin et al. (2017): the neck design every detector still uses.
- **GIoU** — Rezatofighi et al. (2019), and **DIoU/CIoU** — Zheng et al. (2020): the box-loss evolution.
- **TIDE** — Bolya et al. (2020): the error decomposition that should be in every detection dashboard.
- Sibling posts on this blog: [ViT, SigLIP, and DINO](/blog/machine-learning/computer-vision/vit-siglip-dino-explained), [Why Convolution Works](/blog/machine-learning/computer-vision/why-convolution-works), and [Training Pose Estimation Models](/blog/machine-learning/computer-vision/training-pose-estimation-models).

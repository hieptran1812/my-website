---
title: "ViT, SigLIP, and DINO: A Friendly Guide to Modern Vision Backbones"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "Computer Vision"
tags: ["vision-transformer", "vit", "siglip", "dino", "self-supervised-learning", "clip", "multimodal", "representation-learning"]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "ViT, SigLIP, and DINO are three pillars of modern computer vision. This guide explains what each one is, why it exists, and how they differ — using plain language, analogies, and step-by-step intuition."
---

## Introduction

For most of the 2010s, computer vision meant one thing: **Convolutional Neural Networks (CNNs)**. ResNet, VGG, EfficientNet — they all shared the same DNA: sliding filters across an image, building up features from edges to textures to objects.

Then, in 2020, everything changed. A paper from Google titled *"An Image is Worth 16×16 Words"* showed that you could throw away the convolutions entirely and just use a **Transformer** — the same architecture powering language models. That model is now called **ViT (Vision Transformer)**, and it became the new foundation for vision.

But ViT alone only tells half the story. A vision backbone is only as good as the **signal you train it with**. That's where two families of training recipes come in:

- **SigLIP** — trains a vision model by pairing images with text captions from the web. The model learns what a "cat on a skateboard" looks like because millions of captioned photos say so.
- **DINO** — trains a vision model with **no labels and no text at all**. The model learns by comparing different crops of the same image and figuring out that they should have similar representations.

Three models. Three recipes. One shared backbone (the Transformer). Together they form the foundation of almost every modern vision system — from medical imaging to robotics to the vision half of GPT-4 and Claude.

Let's build up each one from scratch.

## Part 1: Vision Transformer (ViT)

### The Core Insight

Before ViT, the working assumption in computer vision was: "Images have spatial structure, so we need architectures with built-in spatial priors." Convolutions have this built in — they only look at local neighborhoods, they share weights across positions, they have a natural hierarchy.

ViT asked a heretical question: **what if we don't need any of that?**

The architecture takes an image, chops it into small square patches (like tiles on a bathroom floor), flattens each patch into a vector, and feeds the sequence of patches into a standard Transformer — the exact same one used for text. No convolutions. No pooling. No inductive biases about spatial locality beyond positional embeddings.

It turns out: **with enough data, this works better than CNNs.**

### Step-by-Step: How ViT Processes an Image

Let's walk through what happens when ViT sees a 224×224 image.

**Step 1: Cut the image into patches.**

A 224×224 image is split into non-overlapping patches of, say, 16×16 pixels. This gives:

$$
\frac{224}{16} \times \frac{224}{16} = 14 \times 14 = 196 \text{ patches}
$$

Think of it as cutting a photograph into a grid of 196 postage stamps.

**Step 2: Flatten and embed each patch.**

Each 16×16×3 patch (RGB channels) has 768 raw pixel values. We flatten these into a vector and project it to the model's embedding dimension $d$ using a single linear layer:

$$
\mathbf{x}_i = W_E \cdot \text{flatten}(\text{patch}_i)
$$

This is equivalent to a single convolution with a 16×16 kernel and stride 16. So in practice, "patch embedding" is just one convolution. Don't let anyone tell you convolutions are dead.

**Step 3: Add a special `[CLS]` token.**

Borrowed from BERT, a learnable vector is prepended to the sequence. This `[CLS]` token's job is to aggregate information from the entire image — we'll use its final representation as the image's overall embedding for classification.

Our sequence length goes from 196 to 197.

**Step 4: Add positional embeddings.**

The Transformer itself doesn't know that patch 5 is to the right of patch 4 — it treats inputs as a set. We add learnable positional embeddings to each patch embedding so the model knows where each patch came from in the original grid.

$$
\mathbf{z}_0 = [\mathbf{x}_{\text{cls}}; \mathbf{x}_1; \mathbf{x}_2; \dots; \mathbf{x}_{196}] + \mathbf{P}
$$

**Step 5: Feed through Transformer layers.**

Now we run the sequence through $L$ identical Transformer blocks. Each block is:

$$
\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}
$$
$$
\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell
$$

Where **MSA** is multi-head self-attention and **LN** is layer normalization. Every patch gets to "look at" every other patch through attention — patch 3 (top-left) can directly attend to patch 190 (bottom-right) in a single layer. This is **global receptive field from layer one**, which is something CNNs need many layers to achieve.

**Step 6: Read out the result.**

For classification, we take the final `[CLS]` token's embedding and pass it through a single linear layer to get class logits:

$$
\hat{y} = W_C \cdot \mathbf{z}_L^{[\text{cls}]}
$$

That's the whole model.

### Why It Works (and Why It Took So Long)

ViT has **fewer built-in assumptions** than a CNN. A CNN assumes pixels nearby matter more than pixels far away, and that the same filter should work at every location. These are good assumptions for small datasets — they reduce the hypothesis space and prevent overfitting.

But with enough data, these assumptions become a ceiling rather than a floor. ViT has no such ceiling — it can learn whatever spatial patterns the data reveals. The tradeoff: ViT needs **a lot more data** to learn from scratch. The original paper showed ViT only beats CNNs once you pretrain on hundreds of millions of images (JFT-300M).

> **Analogy:** A CNN is like a student with strong priors about geometry — they solve basic problems fast but get stuck on weird ones. A ViT is a student with no priors — slow to start, but once they see enough problems, they outperform because they aren't wedded to assumptions.

### Variants Worth Knowing

- **ViT-B/16** — "Base" model with 16×16 patches. ~86M parameters. The default.
- **ViT-L/14** — "Large" model with 14×14 patches (more patches, finer detail). ~304M parameters.
- **ViT-H/14** — "Huge". ~632M parameters. Used in CLIP, SigLIP, and friends.
- **ViT-g/14, ViT-G/14** — Giant variants trained at extreme scale.

The `/N` number is the patch size. Smaller patches mean more tokens, more compute, but finer spatial detail.

## Part 2: SigLIP — Learning Vision from Text

### The Problem ViT Alone Doesn't Solve

ViT is an architecture. It tells you how to process an image, but not **what to learn**. The original ViT was trained on supervised classification — "this image is a golden retriever, this one is an espresso." That works, but it has two problems:

1. **Labels are expensive.** Labeling ImageNet took years and millions of dollars. Labels for 10 billion images are impossible.
2. **Labels are impoverished.** "Golden retriever" captures one concept. But the image also shows grass, sunlight, a blue collar, a tennis ball, a child's hand. A single class label throws all of that away.

The alternative: use the **text people already wrote** about images. Billions of image-caption pairs exist on the web. The caption "A golden retriever catching a frisbee on the beach at sunset" carries *vastly* more information than the label "dog."

This idea became **CLIP** (Contrastive Language-Image Pretraining), released by OpenAI in 2021. **SigLIP** (Sigmoid Loss for Language-Image Pretraining), released by Google in 2023, is CLIP's direct successor — same idea, better loss function.

### The Two-Tower Architecture

Both CLIP and SigLIP train **two separate encoders** at the same time:

- **Image encoder** — a ViT that takes an image and produces a single embedding vector.
- **Text encoder** — a Transformer that takes a caption and produces a single embedding vector.

The goal: **map matching image-caption pairs to nearby points in a shared embedding space.** After training, a cat photo and the caption "a fluffy cat on a sofa" should have embeddings with high cosine similarity.

```
  Image ──► ViT ──► image embedding ─┐
                                     ├── cosine similarity
  Text  ──► Transformer ──► text embedding ─┘
```

### CLIP's Loss: Softmax Contrastive

To train this, you need a **contrastive loss** — one that pulls matching pairs together and pushes non-matching pairs apart.

CLIP uses **softmax contrastive loss**. Take a batch of $N$ image-caption pairs. Compute all $N \times N$ similarities between every image and every caption. The diagonal holds the "correct" pairs; off-diagonal are mismatches. For each row (image), apply softmax and encourage the diagonal entry to be the largest:

$$
\mathcal{L}_{\text{CLIP}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(I_i, T_j) / \tau)}
$$

And symmetrically for each column (caption). $\tau$ is a temperature.

This works great — but it has a hidden cost: **the softmax denominator requires every process to know every other image's similarity to its text.** For a batch of size 32,768 spread across many GPUs, you need an all-gather operation on the full similarity matrix. As the batch grows, this becomes a bottleneck.

### SigLIP's Upgrade: Sigmoid Loss

SigLIP's single big idea: **replace softmax with sigmoid.** Instead of making each image's correct caption the most likely among $N$ options, treat every image-caption pair as an **independent binary classification**:

- Matching pair → predict 1
- Non-matching pair → predict 0

The loss:

$$
\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^N \log \sigma \left( z_{ij} \cdot (t \cdot \mathbf{x}_i \cdot \mathbf{y}_j + b) \right)
$$

Where:
- $\sigma$ is the sigmoid function
- $\mathbf{x}_i, \mathbf{y}_j$ are the normalized image and text embeddings
- $t$ is a learnable temperature, $b$ is a learnable bias
- $z_{ij} = +1$ if $i = j$ (matching pair), $-1$ otherwise

Because each pair contributes independently, you don't need global normalization. Each GPU can compute its chunk of the $N \times N$ matrix without waiting on others, then sum up local losses at the end.

### Why This Matters

Three practical wins:

**1. Easier to scale.** Softmax contrastive needs a giant all-gather. Sigmoid doesn't. SigLIP trains cleanly on batch sizes of 1M+ image-text pairs.

**2. Better performance at small batch sizes too.** Surprisingly, sigmoid loss also wins at batch sizes as small as 256. The softmax pressure to make one pair the "top-1" out of a small batch creates noisy gradients; sigmoid treats each pair on its own merits.

**3. Positive/negative imbalance is handled explicitly.** For a batch of $N$, there are $N$ positives and $N^2 - N$ negatives. The learnable bias $b$ lets the model calibrate for this imbalance.

> **Analogy for the loss change:** CLIP is a ranked tournament — every image competes against all other captions to be the "champion." SigLIP is a series of yes/no interviews — each image-caption pair is evaluated on its own, independently.

### What You Get After Training

After training, the image encoder (ViT) has learned:

- **Rich semantics** — it knows what "a vintage typewriter on a wooden desk" looks like because millions of web captions aligned images to language.
- **Zero-shot classification** — give it an image and a list of text labels ("photo of a cat", "photo of a dog"), and it picks the closest text by cosine similarity. No fine-tuning needed.
- **Multimodal bridging** — you can now plug the image encoder into a language model (like LLaVA, Qwen-VL, or PaliGemma do) because the embeddings already live in a language-aligned space.

SigLIP-family vision encoders are the **default vision backbone for modern VLMs (vision-language models).** When you hear about a new open-source VLM, there's a good chance its "eyes" are a SigLIP ViT.

## Part 3: DINO — Learning Vision Without Any Text

### The Motivation

CLIP and SigLIP need **image-text pairs**. But:

- Text captions are noisy ("IMG_4523.jpg", "my vacation!!", hashtag spam)
- Many domains have no text (microscopy, satellite, medical)
- Text is a bottleneck — it flattens a rich image into a short sentence

What if you could learn visual representations from **images alone**, with no labels and no text?

This is the domain of **self-supervised learning (SSL)**, and **DINO** (self-**DI**stillation with **NO** labels) from Meta AI is one of its most elegant and influential recipes.

### The Core Idea: A Student Copies a Teacher That Copies It Back

DINO has two networks: a **student** and a **teacher**. Both are ViTs with identical architectures but different weights.

Given one input image, we:

1. Create **several different crops** of it — some "global" crops that see most of the image, some "local" crops that zoom into small parts.
2. Feed global crops to both student and teacher.
3. Feed local crops only to the student.
4. Ask the student: **"Make your output for this local crop match the teacher's output for the global crop of the same image."**

The student sees a small piece of a dog (say, an ear) and must predict the same representation the teacher produces when it sees the full dog. To do this, the student must learn that the ear **belongs to the dog** — it must learn **semantic, object-level features** without ever being told "this is a dog."

### The Teacher's Weights: A Moving Average of the Student

Here's the twist: **the teacher is never trained by gradient descent.** Instead, its weights are an exponential moving average (EMA) of the student's weights:

$$
\theta_{\text{teacher}} \leftarrow \lambda \cdot \theta_{\text{teacher}} + (1 - \lambda) \cdot \theta_{\text{student}}
$$

With $\lambda$ typically around 0.996. The teacher is a slow-moving, smoothed version of the student. It is, in effect, the **student's past self**. This is sometimes called a **"momentum encoder."**

This bootstrapping dynamic is strange the first time you see it:

- The student learns to match the teacher
- The teacher is just a smoothed copy of the student
- So the student is effectively trying to match a lagged version of itself

Why doesn't this collapse to a trivial solution (e.g., both outputting a constant vector)? Two tricks:

**Centering.** Subtract a running mean from the teacher's outputs before softmax. This prevents any single dimension from dominating — it keeps the output distribution spread out.

**Sharpening.** Apply a low-temperature softmax to the teacher's output. This makes the teacher's "soft labels" confident and peaky. If centering without sharpening would lead to uniform collapse (everyone outputs the same thing), sharpening forces distinctions.

The combination — centering + sharpening — sits at exactly the right balance: distinctive enough to be informative, smooth enough to avoid trivial collapse.

### The Loss

Both networks output a probability distribution over $K$ dimensions (a learned "prototype" vocabulary, typically $K = 65{,}536$). The student's job is to match the teacher's distribution with cross-entropy:

$$
\mathcal{L}_{\text{DINO}} = -\sum_{k=1}^K P_t^{(k)} \log P_s^{(k)}
$$

Averaged over all (teacher view, student view) pairs where the views are different crops of the same image.

### What DINO Learns: A Surprise

The astonishing result from the original DINO paper: **the attention maps of a DINO-trained ViT segment objects** — without any segmentation labels. Visualize the attention from the `[CLS]` token to patches, and you'll see it light up on the foreground object (the dog, the car, the person) and ignore background.

```
    ViT + supervised → attention is spread, task-specific
    ViT + CLIP/SigLIP → attention focuses on text-relevant regions
    ViT + DINO → attention cleanly outlines objects, unsupervised
```

This wasn't designed in; it emerged. The model discovered that "parts of the same object should have similar representations" because that's what makes local and global crops map to the same target.

### DINOv2: Scaling the Recipe

**DINOv2** (2023) took the same core idea and scaled it:

- **Much larger, curated dataset** (142M carefully deduplicated images)
- **ViT-g/14 backbone** (1.1B parameters)
- Combined DINO loss with **iBOT loss** (masked image modeling — hide some patches and predict them back, like BERT for images)
- Added **KoLeo regularization** to spread features across the embedding space

The resulting DINOv2 features became arguably the **best general-purpose visual features available**. Without fine-tuning, they match or beat supervised models on:

- Image classification
- Depth estimation
- Semantic segmentation
- Instance retrieval
- Video tracking

One backbone. Many downstream tasks. No labels used in pretraining.

### DINOv3: The 2025 Upgrade

**DINOv3** pushed further — scaling to 7B parameters, 1.7B images, and introducing **Gram anchoring**, a technique that preserves the quality of dense patch-level features during very long training runs. Earlier DINO variants often saw dense features degrade even as global (`[CLS]`) features improved; Gram anchoring fixes this by regularizing the pairwise similarity structure of patch features against a frozen "anchor" teacher.

The result: a single frozen vision model that outperforms specialized state-of-the-art across dense prediction, classification, and retrieval.

## Part 4: How They Compare

All three can share the **exact same ViT architecture**. What differs is the training signal.

| Aspect | ViT (supervised) | SigLIP | DINO / DINOv2 |
|---|---|---|---|
| **Training data** | Labeled images (e.g., ImageNet, JFT-300M) | Image-text pairs from the web (billions) | Unlabeled images (no text, no labels) |
| **Training signal** | Ground-truth class labels | Contrastive image-text alignment | Self-distillation: student matches EMA teacher |
| **What it learns** | Classification categories | Language-aligned concepts | Object-level visual structure |
| **Zero-shot classification** | No | Yes (match to text labels) | No (no language grounding) |
| **Dense features (per patch)** | Moderate | Moderate | Excellent (DINOv2/v3) |
| **Best for** | Classification fine-tuning | VLMs, retrieval, zero-shot | Dense prediction, segmentation, frozen-feature tasks |
| **Key loss** | Cross-entropy | Sigmoid pairwise loss | Cross-entropy vs. EMA teacher (centering + sharpening) |

### When to Reach for Which

- **Building a VLM or needing zero-shot text-image retrieval?** Use **SigLIP** (SigLIP-2 is the latest). Its embeddings live in a language-aligned space, which is exactly what you need to bolt onto an LLM.
- **Doing segmentation, depth, tracking, or any task that cares about per-pixel/per-patch features?** Use **DINOv2** or **DINOv3**. Its dense features are unmatched.
- **Have a lot of labeled data for one specific task?** Fine-tune a **ViT** initialized from any of the above. SigLIP or DINOv2 pretraining almost always beats ImageNet pretraining.
- **Have no labels and no text?** **DINO** is your only choice — and it's a great one.

### They're Complementary, Not Competitive

In practice, cutting-edge systems often **combine** these. A modern VLM might:

- Use **SigLIP** features for language alignment (the `[CLS]`-like global representation that talks to the LLM)
- Use **DINOv2** features for spatial grounding (per-patch features that tell the model *where* things are)

Or even concatenate both as the visual input to the language model. The two signals — language-aligned semantics (SigLIP) and object-level spatial structure (DINO) — turn out to be genuinely complementary.

## Putting It All Together: A Mental Model

Think of a Vision Transformer as a **shared engine**. What fuels it determines where you can drive.

- Fuel it with **labels** → it learns to classify
- Fuel it with **captions** → it learns to speak the language of images
- Fuel it with **itself** (self-distillation) → it learns what visual structure *is*, independent of any human taxonomy

The architecture (ViT) was the key that unlocked modern vision. The training recipes (SigLIP, DINO, and their descendants) are what make specific vision models useful for specific jobs. Understanding this decoupling — **architecture vs. objective** — is the key to reading any new vision paper.

## Further Reading

**Original papers:**

- *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale* (Dosovitskiy et al., 2020) — the ViT paper
- *Learning Transferable Visual Models From Natural Language Supervision* (Radford et al., 2021) — CLIP
- *Sigmoid Loss for Language Image Pre-Training* (Zhai et al., 2023) — SigLIP
- *Emerging Properties in Self-Supervised Vision Transformers* (Caron et al., 2021) — DINO
- *DINOv2: Learning Robust Visual Features without Supervision* (Oquab et al., 2023)
- *DINOv3* (Meta AI, 2025) — the latest scaling of the DINO recipe

**Good blog posts and tutorials:**

- The Hugging Face blog posts on ViT, CLIP, and DINO for hands-on walkthroughs
- Lucas Beyer's talks on SigLIP and scalable vision pretraining
- Meta AI's DINOv2 and DINOv3 technical reports — worth skimming for the ablations alone

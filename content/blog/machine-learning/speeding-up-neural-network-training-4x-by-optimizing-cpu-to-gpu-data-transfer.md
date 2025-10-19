---
title: "Speeding up neural network training 4× by optimizing CPU-to-GPU data transfer"
publishDate: "2025-10-19"
category: "machine-learning"
subcategory: "Training Techniques"
tags: ["tips", "Training techniques"]
date: "2025-10-19"
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/speeding-up-neural-network-training-4x-by-optimizing-cpu-to-gpu-data-transfer-20251019194904.png"
excerpt: "When training neural networks, performance bottlenecks are not always inside the GPU kernels..."
---

When training neural networks, performance bottlenecks are not always inside the GPU kernels.
A surprisingly common bottleneck lies in data transfer between the CPU and GPU.

A simple yet powerful optimization can make this transfer up to four times faster:
move the data type conversion (from 8-bit integers to 32-bit floats) to the GPU.

In this article, we will explore:

- Why this optimization works
- How to implement it in PyTorch
- When it applies and when it doesn’t
- Related optimizations that can further improve training efficiency

## Understanding the Problem: Transferring Float32 Data Is Four Times Heavier

Consider a standard image classification pipeline:

1. Load an image in 8-bit integer format (`uint8`).
2. Apply transformations that convert it to 32-bit floating point (`float32`).
3. Transfer the batch to the GPU for training.

If you profile this pipeline, you will see two main components:

- **Kernel time**, representing actual training computation.
- **Data transfer time**, shown as `cudaMemcpyAsync`, representing CPU-to-GPU memory copy.

The issue arises because converting `uint8` to `float32` increases memory usage by 4× before the transfer.

For example, an image of size 224×224×3 in a batch of 64:

- As `uint8`: ~9.6 MB
- As `float32`: ~38.4 MB

This means you are sending four times as much data over PCIe or NVLink — wasting precious bandwidth.

## The Solution: Transfer First, Convert Later

The idea is simple:

- Keep data in `uint8` on the CPU.
- Transfer it to the GPU as is.
- Convert it to `float` and normalize it on the GPU.

By transferring smaller data (8-bit instead of 32-bit), the total data movement time decreases dramatically.

### Benefits

- Data transfer becomes roughly 4× faster.
- GPU performs type conversion efficiently.
- The entire data pipeline runs smoother with fewer stalls.

## Implementation in PyTorch

### Before: Conversion on CPU (Slower)

```python
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),            # Converts uint8 -> float32 on CPU
    T.Normalize(mean, std)
])

loader = DataLoader(dataset, batch_size=64, num_workers=8, pin_memory=True)

for x_cpu, y in loader:
    x = x_cpu.to(device, non_blocking=True)  # Transferring float32 (heavy)
    output = model(x)
```

### After: Conversion on GPU (Faster)

```python
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224)        # Keeps images as uint8
])

loader = DataLoader(dataset, batch_size=64, num_workers=8,
                    pin_memory=True, prefetch_factor=2, persistent_workers=True)

for x_cpu, y in loader:
    x = x_cpu.to(device, non_blocking=True)  # Transfer uint8 (lighter)
    x = x.to(torch.float16).div_(255)        # Convert and normalize on GPU
    x = (x - mean_gpu) / std_gpu
    output = model(x)
```

> Note: Use **torchvision v2** or transformations that preserve `uint8` tensors to prevent automatic conversion to `float`.

## When This Technique Does Not Apply

This optimization is not universal.
It mainly benefits tasks where raw data starts as 8-bit images.

You should not use it when:

- Data is inherently floating-point (e.g., embeddings, tabular data, or spectrograms).
- Preprocessing requires floating-point arithmetic on the CPU.

In those cases:

- Move augmentations to the GPU using libraries like Kornia or NVIDIA DALI.
- Or, accept the CPU conversion cost if necessary for numerical accuracy.

---
title: "NVIDIA Highlights Inkling-Small and NVFP4 Checkpoint"
publishDate: "2026-07-30"
date: "2026-07-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "quantization", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "NVIDIA AI promotes Inkling-Small for fine-tuning with NeMo on DGX Station and links an NVFP4 checkpoint."
---

> Source: [@NVIDIAAI on X](https://x.com/NVIDIAAI/status/2082887941626732664) · posted 2026-07-30

## Summary

NVIDIA AI highlights another open-weight release from Thinking Machines, **Inkling-Small**. It notes native reasoning over audio and images and variable thinking effort, and recommends fine-tuning with NVIDIA NeMo on DGX Station. It links an NVFP4 checkpoint on Hugging Face; the quoted post is the original Inkling-Small announcement.

## Background

NVFP4 is NVIDIA's 4-bit floating point format, supported on Blackwell GPUs, which shrinks memory and speeds up math. Variable thinking effort means you can choose how much the model reasons before answering. NeMo is NVIDIA's training framework.

## What I took away

- Quantized checkpoints like NVFP4 make large models cheaper to run.
- Variable thinking effort lets you trade cost for quality per request.
- Vendor posts pair models with their own hardware; compare options yourself.

## Reference

- Original post: [https://x.com/NVIDIAAI/status/2082887941626732664](https://x.com/NVIDIAAI/status/2082887941626732664)
- Author: [@NVIDIAAI](https://x.com/NVIDIAAI) on X, posted 2026-07-30

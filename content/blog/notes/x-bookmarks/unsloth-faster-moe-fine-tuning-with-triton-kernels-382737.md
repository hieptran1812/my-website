---
title: "Unsloth Faster MoE Fine-Tuning With Triton Kernels"
publishDate: "2026-02-11"
date: "2026-02-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "moe", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2021398069523382737-1.png"
excerpt: "NVIDIA highlights Unsloth's claim of 12x faster MoE training with 35% less VRAM."
---

> Source: [@NVIDIAAIDev on X](https://x.com/NVIDIAAIDev/status/2021398069523382737) · posted 2026-02-11

![Unsloth Faster MoE Fine-Tuning With Triton Kernels](/imgs/blogs/x-bookmarks/2021398069523382737-1.png)

## Summary

NVIDIA calls Unsloth's work a performance breakthrough. Unsloth says new Triton kernels train MoE models 12x faster with 35% less VRAM and no accuracy loss, enabling gpt-oss-20b fine-tuning on about 12.8GB of VRAM. It also works with DeepSeek, Qwen3 and GLM via a collab with Hugging Face.

## Background

Triton is a language for writing GPU kernels. MoE layers route tokens to experts, which is hard to run efficiently, so custom kernels can give large speedups. VRAM is GPU memory, the main limit for local fine-tuning.

## What I took away

- Custom kernels can cut both time and memory.
- MoE fine-tuning on consumer GPUs is becoming possible.
- Speedups are the vendor's claims, so benchmark yourself.

## Reference

- Original post: [https://x.com/NVIDIAAIDev/status/2021398069523382737](https://x.com/NVIDIAAIDev/status/2021398069523382737)
- Author: [@NVIDIAAIDev](https://x.com/NVIDIAAIDev) on X, posted 2026-02-11

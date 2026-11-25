---
title: "Step 3.7 Flash Available on NVIDIA Platforms"
publishDate: "2026-05-29"
date: "2026-05-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "moe", "serving"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2060158596730482985-1.jpg"
excerpt: "NVIDIA announces day-0 support for the Step 3.7 Flash MoE model across its build site, NIM and NeMo."
---

> Source: [@NVIDIAAI on X](https://x.com/NVIDIAAI/status/2060158596730482985) · posted 2026-05-29

![Step 3.7 Flash Available on NVIDIA Platforms](/imgs/blogs/x-bookmarks/2060158596730482985-1.jpg)

## Summary

NVIDIA says **Step 3.7 Flash** is a 198B MoE with 11B active parameters, 256K context and native image and video support. Day-0 support is live on build.nvidia.com with GPU-accelerated endpoints. It can be deployed with NVIDIA NIM and fine-tuned with NeMo.

## Background

NIM packages models as ready-to-run inference containers. NeMo is NVIDIA's toolkit for training and fine-tuning. Day-0 support means the model works on these tools the day it is released.

## What I took away

- Large MoE models can have small active parameter counts, keeping inference cost down.
- Day-0 vendor support lowers the barrier to trying new models.
- Native video input is becoming standard in open models.

## Reference

- Original post: [https://x.com/NVIDIAAI/status/2060158596730482985](https://x.com/NVIDIAAI/status/2060158596730482985)
- Author: [@NVIDIAAI](https://x.com/NVIDIAAI) on X, posted 2026-05-29

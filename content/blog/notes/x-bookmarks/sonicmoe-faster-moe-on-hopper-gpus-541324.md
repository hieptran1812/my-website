---
title: "SonicMoE Faster MoE On Hopper GPUs"
publishDate: "2025-12-18"
date: "2025-12-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "moe", "gpu-kernels", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2001773245318541324-1.jpg"
excerpt: "SonicMoE is an MoE implementation for NVIDIA Hopper GPUs that reportedly cuts activation memory 45% and runs 1.86x faster on H100."
---

> Source: [@WentaoGuo7 on X](https://x.com/WentaoGuo7/status/2001773245318541324) · posted 2025-12-18

![SonicMoE Faster MoE On Hopper GPUs](/imgs/blogs/x-bookmarks/2001773245318541324-1.jpg)

## Summary

The authors announce SonicMoE, a MoE implementation optimized for NVIDIA Hopper GPUs. They say it reduces activation memory by 45% and is 1.86x faster on H100 than the previous state of the art, with a paper on arXiv and co-authors including Tri Dao.

## Background

A mixture-of-experts layer sends each token to a few expert networks, which needs careful GPU kernels to be efficient. Activation memory is what is stored during the forward pass for training. Hopper is NVIDIA's H100 architecture. The numbers are the authors' claims.

## What I took away

- MoE speed depends heavily on kernel design.
- Reducing activation memory allows larger batches or models.
- Read the paper to check the benchmark settings.

## Reference

- Original post: [https://x.com/WentaoGuo7/status/2001773245318541324](https://x.com/WentaoGuo7/status/2001773245318541324)
- Author: [@WentaoGuo7](https://x.com/WentaoGuo7) on X, posted 2025-12-18

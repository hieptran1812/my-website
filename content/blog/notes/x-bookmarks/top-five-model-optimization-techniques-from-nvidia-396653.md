---
title: "Top Five Model Optimization Techniques From NVIDIA"
publishDate: "2025-12-23"
date: "2025-12-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "quantization", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2003572784677396653-1.png"
excerpt: "NVIDIA lists top techniques for faster inference, starting with post-training quantization and quantization-aware training."
---

> Source: [@NVIDIAAIDev on X](https://x.com/NVIDIAAIDev/status/2003572784677396653) · posted 2025-12-23

![Top Five Model Optimization Techniques From NVIDIA](/imgs/blogs/x-bookmarks/2003572784677396653-1.png)

## Summary

NVIDIA lists its top 5 AI model optimization techniques for faster, smarter inference. The first is post-training quantization, the fastest path to value, which compresses models without retraining for latency and throughput wins. The second is quantization-aware training, which fine-tunes for low precision. The rest of the list is cut off.

## Background

Quantization stores weights in fewer bits to save memory and speed things up. PTQ does this after training, while QAT includes it during training to keep accuracy. Post-training is easier, while QAT is more accurate at very low precision.

## What I took away

- Try PTQ first because it needs no retraining.
- Use QAT when PTQ loses too much accuracy.

## Reference

- Original post: [https://x.com/NVIDIAAIDev/status/2003572784677396653](https://x.com/NVIDIAAIDev/status/2003572784677396653)
- Author: [@NVIDIAAIDev](https://x.com/NVIDIAAIDev) on X, posted 2025-12-23

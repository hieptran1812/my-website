---
title: "LeapQuant Quantizes Recurrent States in Linear Attention"
publishDate: "2026-11-11"
date: "2026-11-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "llm-inference", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2120643704792989972-1.jpg"
excerpt: "LeapQuant is claimed to be the first training-free method for 8-bit or 4-bit recurrent state quantization in linear attention models."
---

> Source: [@conlesspan on X](https://x.com/conlesspan/status/2120643704792989972) · posted 2026-11-11

![LeapQuant Quantizes Recurrent States in Linear Attention](/imgs/blogs/x-bookmarks/2120643704792989972-1.jpg)

## Summary

The post says Qwen, Kimi and GLM are betting on linear attention, but it is still slow because the recurrent state is read and rewritten every token. **LeapQuant** is introduced as the first training-free method to make 8-bit or 4-bit recurrent states work, claiming to match FP32 across these models.

## Background

Linear attention keeps a fixed-size recurrent state instead of a growing KV cache. That state is touched every token, so memory bandwidth becomes the bottleneck. Quantizing it to fewer bits reduces traffic. Training-free means no retraining is needed.

## What I took away

- Linear attention still has memory-bandwidth costs.
- Quantizing state, not just weights, can speed decoding.
- Matching FP32 is the authors' claim to verify.

## Reference

- Original post: [https://x.com/conlesspan/status/2120643704792989972](https://x.com/conlesspan/status/2120643704792989972)
- Author: [@conlesspan](https://x.com/conlesspan) on X, posted 2026-11-11

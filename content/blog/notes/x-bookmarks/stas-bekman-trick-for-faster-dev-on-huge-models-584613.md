---
title: "Stas Bekman Trick For Faster Dev On Huge Models"
publishDate: "2025-11-16"
date: "2025-11-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "tools", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1989862347423584613-1.png"
excerpt: "A tip for developing with huge models by reducing hidden layers, making loading and running about 20x faster."
---

> Source: [@StasBekman on X](https://x.com/StasBekman/status/1989862347423584613) · posted 2025-11-16

![Stas Bekman Trick For Faster Dev On Huge Models](/imgs/blogs/x-bookmarks/1989862347423584613-1.png)

## Summary

Stas Bekman shares his favorite time saving recipe for dev work with huge models like Qwen3-Next-80B. He says loading and run time are about 20x faster. Setting num_hidden_layers=0 gives the overhead of the embedding and lm_head alone, so it is easy to get memory and flops per layer. The text is cut off.

## Background

Reducing the number of layers in a model config makes a small version with the same code paths, so you can debug fast without downloading or running all weights. Measuring a model with zero layers isolates fixed costs. The 20x figure is the author's claim.

## What I took away

- Shrink num_hidden_layers to debug quickly.
- Use zero layers to isolate embedding and head costs.
- Compute per-layer memory and FLOPs by difference.

## Reference

- Original post: [https://x.com/StasBekman/status/1989862347423584613](https://x.com/StasBekman/status/1989862347423584613)
- Author: [@StasBekman](https://x.com/StasBekman) on X, posted 2025-11-16

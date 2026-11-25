---
title: "JiT: Just Image Transformers On Raw Pixels"
publishDate: "2025-11-18"
date: "2025-11-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "llm-architecture", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1990700499327181247-1.jpg"
excerpt: "Introduction of JiT, simple large-patch Transformers for raw pixels with no tokenizer or extra losses, from Tianhong Li and Kaiming He."
---

> Source: [@jiqizhixin on X](https://x.com/jiqizhixin/status/1990700499327181247) · posted 2025-11-18

![JiT: Just Image Transformers On Raw Pixels](/imgs/blogs/x-bookmarks/1990700499327181247-1.jpg)

## Summary

The post says Tianhong Li and Kaiming He introduced JiT (Just image Transformers). These are simple large-patch Transformers that work on raw pixels and need no tokenizer, pre-training or extra losses. They predict clean data on the natural-data manifold.

## Background

Most image generators use a tokenizer or autoencoder to compress images first. Working directly on pixels is simpler but harder. Predicting clean data instead of noise is a choice in diffusion training that this work argues for.

## What I took away

- Simpler pipelines without tokenizers are a research direction.
- Predicting clean data may suit high-dimensional pixels.
- Read the paper for the evidence.

## Reference

- Original post: [https://x.com/jiqizhixin/status/1990700499327181247](https://x.com/jiqizhixin/status/1990700499327181247)
- Author: [@jiqizhixin](https://x.com/jiqizhixin) on X, posted 2025-11-18

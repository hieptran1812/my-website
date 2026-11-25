---
title: "NeMo AutoModel Builds On Transformers v5 For MoE"
publishDate: "2026-06-24"
date: "2026-06-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "moe", "pretraining", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2069813582825418828-1.jpg"
excerpt: "NVIDIA says NeMo AutoModel builds on Hugging Face Transformers v5 to address MoE training challenges."
---

> Source: [@NVIDIAAI on X](https://x.com/NVIDIAAI/status/2069813582825418828) · posted 2026-06-24

![NeMo AutoModel Builds On Transformers v5 For MoE](/imgs/blogs/x-bookmarks/2069813582825418828-1.jpg)

## Summary

@NVIDIAAI says the rise of MoE models brought training challenges, and Hugging Face's Transformers v5 added first-class support for solving them. NeMo AutoModel now builds on v5 and brings optimizations. The text is cut off.

## Background

MoE (mixture of experts) models route each token to a few expert sub-networks, which raises efficiency but complicates training and parallelism. NeMo is NVIDIA's framework for building models at scale. AutoModel loads Hugging Face models into it.

## What I took away

- Transformers v5 has native MoE support.
- NeMo AutoModel offers optimized training on top of it.

## Reference

- Original post: [https://x.com/NVIDIAAI/status/2069813582825418828](https://x.com/NVIDIAAI/status/2069813582825418828)
- Author: [@NVIDIAAI](https://x.com/NVIDIAAI) on X, posted 2026-06-24

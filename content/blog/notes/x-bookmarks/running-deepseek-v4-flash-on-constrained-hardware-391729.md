---
title: "Running DeepSeek V4 Flash On Constrained Hardware"
publishDate: "2026-06-26"
date: "2026-06-26"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "hardware", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2070584201020391729-1.jpg"
excerpt: "A quoted post shows DeepSeek V4 Flash in native FP4 and FP8 on two RTX Pro 6000 GPUs plus 256 GB RAM using KTransformers."
---

> Source: [@TheAhmadOsman on X](https://x.com/TheAhmadOsman/status/2070584201020391729) · posted 2026-06-26

![Running DeepSeek V4 Flash On Constrained Hardware](/imgs/blogs/x-bookmarks/2070584201020391729-1.jpg)

## Summary

@TheAhmadOsman says Yannick is underfollowed in the local AI space. The quoted post describes DeepSeek V4 Flash in native precision (FP4 + FP8) fitting on 2x RTX Pro 6000 GPUs and 256 GB DDR5 RAM with KTransformers, a KVCache-AI fork of SGLang for GPU/CPU memory inference. The author likes running apps on resource-constrained systems.

## Background

Hybrid GPU and CPU inference keeps part of a large model in system RAM so it can run on fewer GPUs. Native FP4 and FP8 weights are low precision formats that shrink memory needs. This makes big models more reachable for local setups.

## What I took away

- Large MoE models can run on a modest GPU plus big RAM setup.
- KTransformers is worth trying for CPU/GPU offload.

## Reference

- Original post: [https://x.com/TheAhmadOsman/status/2070584201020391729](https://x.com/TheAhmadOsman/status/2070584201020391729)
- Author: [@TheAhmadOsman](https://x.com/TheAhmadOsman) on X, posted 2026-06-26

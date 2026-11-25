---
title: "Techniques For When GPU Memory Is The Bottleneck"
publishDate: "2025-12-19"
date: "2025-12-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "kv-cache", "quantization", "gpu-kernels"]
author: "Hiep Tran"
featured: false
excerpt: "athletic_coder lists topics to master if GPU memory limits you, from KV-cache math to FlashAttention."
---

> Source: [@athletic_coder on X](https://x.com/athletic_coder/status/2001993522472267946) · posted 2025-12-19

## Summary

The author lists techniques to learn if GPU memory is your real bottleneck. They include KV-cache memory math, PagedAttention, context length versus VRAM curves, INT8 and INT4 quantization, AWQ versus GPTQ tradeoffs, activation offloading, CPU and GPU swapping, and FlashAttention.

## Background

LLM memory use is dominated by weights and the KV cache, which grows with context length. Quantization stores numbers in fewer bits, PagedAttention manages cache memory in blocks, and FlashAttention avoids storing large attention matrices. Together they let larger models or longer contexts fit.

## What I took away

- Learn to calculate KV-cache size for your model and context length.
- Quantization and paged caches are the main levers when memory is tight.
- Use the list as a study checklist for inference optimization.

## Reference

- Original post: [https://x.com/athletic_coder/status/2001993522472267946](https://x.com/athletic_coder/status/2001993522472267946)
- Author: [@athletic_coder](https://x.com/athletic_coder) on X, posted 2025-12-19

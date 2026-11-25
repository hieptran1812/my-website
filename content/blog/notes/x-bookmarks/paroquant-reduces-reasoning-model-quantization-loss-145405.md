---
title: "ParoQuant Reduces Reasoning Model Quantization Loss"
publishDate: "2026-02-24"
date: "2026-02-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "llm-inference", "gpu-kernels"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2026151986052145405-1.jpg"
excerpt: "ParoQuant is presented as a fix for quantization errors that accumulate in long reasoning chains."
---

> Source: [@zhijianliu on X](https://x.com/zhijianliu/status/2026151986052145405) · posted 2026-02-24

![ParoQuant Reduces Reasoning Model Quantization Loss](/imgs/blogs/x-bookmarks/2026151986052145405-1.jpg)

## Summary

The author says reasoning LLMs write very long chains of thought, so small quantization errors add up. With AWQ, Qwen3-4B drops from 71.0 to 68.2 on MMLU-Pro, about 4% relative. ParoQuant keeps only the critical rotation pairs and fuses everything into a single kernel.

## Background

Quantization stores model weights in fewer bits to save memory and speed up inference. AWQ is a common method. Long generations amplify small errors, so reasoning models are more sensitive. Rotations are used to spread out outlier values before quantizing.

## What I took away

- Quantization quality should be checked on long reasoning tasks, not only short ones.
- Fusing steps into one kernel is how a method stays fast.
- The numbers are the author's reported results.

## Reference

- Original post: [https://x.com/zhijianliu/status/2026151986052145405](https://x.com/zhijianliu/status/2026151986052145405)
- Author: [@zhijianliu](https://x.com/zhijianliu) on X, posted 2026-02-24

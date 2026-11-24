---
title: "FastH3-Omni Real-Time Video Model Served by vLLM-Omni"
publishDate: "2026-10-21"
date: "2026-10-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "audio-video", "serving", "open-models", "quantization"]
author: "Hiep Tran"
featured: false
excerpt: "vLLM-Omni announces day-zero serving for FastH3-Omni, an open real-time video generation model with weights and training code."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2112704330335945064) · posted 2026-10-21

## Summary

@vllm_project says FastH3-Omni is out with weights and training code, and vLLM-Omni serves it from day one, including on consumer RTX GPUs. The quoted FastVideo post says it open-sources MiniMax's FastH3 Omni, claims real-time generation 12x faster on NVIDIA Blackwell, supports NVFP4 and FP8 for Spark and RTX, and has ComfyUI support.

## Background

Video generation models are very compute-heavy, so real-time speed is hard. NVFP4 and FP8 are low-precision number formats that make models smaller and faster at some quality cost. Day-zero support means a serving engine works with the model on release day.

## What I took away

- Low-precision formats are what make video models run on consumer GPUs.
- Open training code is more useful than weights alone for research.
- Speed figures are the authors' claims for specific hardware.

## Reference

- Original post: [https://x.com/vllm_project/status/2112704330335945064](https://x.com/vllm_project/status/2112704330335945064)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-10-21

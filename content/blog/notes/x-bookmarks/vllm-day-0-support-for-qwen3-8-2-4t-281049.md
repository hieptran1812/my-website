---
title: "vLLM Day-0 Support for Qwen3.8-2.4T-A95B"
publishDate: "2026-08-12"
date: "2026-08-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "open-models", "moe", "quantization"]
author: "Hiep Tran"
featured: false
excerpt: "vLLM announced day-0 support for Qwen3.8-2.4T-A95B on NVIDIA and AMD hardware with 4-bit checkpoints."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2087571359413281049) · posted 2026-08-12

## Summary

vLLM congratulates Alibaba Qwen on Qwen3.8-2.4T-A95B, one of the largest open-weight models released so far, with 2.4T parameters, 95B active and 512 experts. vLLM has day-0 support verified on NVIDIA and AMD hardware. A ready-made 4-bit checkpoint is provided per vendor.

## Background

With 512 experts, only a small set is used per token, so compute is like a 95B model but all 2.4T parameters must be stored. 4-bit checkpoints shrink memory needs by storing weights in fewer bits, which is often required to fit such models on available hardware.

## What I took away

- Memory, not compute, is the main constraint for serving huge MoE models.
- Vendor-specific quantized checkpoints save setup effort on different hardware.
- Check quality of 4-bit versions against full precision for your tasks.

## Reference

- Original post: [https://x.com/vllm_project/status/2087571359413281049](https://x.com/vllm_project/status/2087571359413281049)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-08-12

---
title: "Qwen3.5 Update And Unsloth GGUF Benchmarks"
publishDate: "2026-02-28"
date: "2026-02-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "quantization", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2027553265915920666-1.jpg"
excerpt: "Qwen thanks Unsloth for GGUFs; Qwen3.5 got better tool-calling and coding, and 35B-A3B runs in 22GB RAM."
---

> Source: [@Alibaba_Qwen on X](https://x.com/Alibaba_Qwen/status/2027553265915920666) · posted 2026-02-28

![Qwen3.5 Update And Unsloth GGUF Benchmarks](/imgs/blogs/x-bookmarks/2027553265915920666-1.jpg)

## Summary

Alibaba Qwen thanks Unsloth for GGUFs so people can start with local models. The quoted post says Qwen3.5 is updated with improved tool-calling and coding, that Qwen3.5-35B-A3B runs on 22GB RAM, and that they benchmarked GGUFs and removed MXFP4 layers from 3 quants.

## Background

GGUF is a file format for running quantized models locally with tools like llama.cpp. A name like 35B-A3B suggests a mixture-of-experts model with a small active part. Quantized benchmarks show how much quality is lost.

## What I took away

- Local models can run on modest RAM when quantized.
- Benchmark quants before trusting them.
- Works with Claude Code and Codex per the post.

## Reference

- Original post: [https://x.com/Alibaba_Qwen/status/2027553265915920666](https://x.com/Alibaba_Qwen/status/2027553265915920666)
- Author: [@Alibaba_Qwen](https://x.com/Alibaba_Qwen) on X, posted 2026-02-28

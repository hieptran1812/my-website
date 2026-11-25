---
title: "Qwen3.5 35B Running On An iPhone"
publishDate: "2026-03-21"
date: "2026-03-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "quantization", "moe"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2035386645764006102-1.jpg"
excerpt: "The author reports running a 4-bit Qwen3.5 35B mixture-of-experts model fully on an iPhone at 5.6 tokens per second."
---

> Source: [@Alexintosh on X](https://x.com/Alexintosh/status/2035386645764006102) · posted 2026-03-21

![Qwen3.5 35B Running On An iPhone](/imgs/blogs/x-bookmarks/2035386645764006102-1.jpg)

## Summary

The author says they ran Qwen3.5 35B on an iPhone at 5.6 tokens per second, fully on-device. The model uses 4-bit weights and 256 experts, is 19.5GB in size and the phone has 12GB of RAM.

## Background

In a mixture-of-experts (MoE) model only a few experts run per token, so not all weights are used at once. That makes it possible to run a model bigger than RAM by loading parts as needed, probably from storage. 4-bit quantization shrinks weights to save memory.

## What I took away

- A model larger than RAM can still run when only some experts are active.
- Quantization plus MoE makes large models feasible on phones.
- 5.6 tok/s is usable for demos, but slow for heavy use.

## Reference

- Original post: [https://x.com/Alexintosh/status/2035386645764006102](https://x.com/Alexintosh/status/2035386645764006102)
- Author: [@Alexintosh](https://x.com/Alexintosh) on X, posted 2026-03-21

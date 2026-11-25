---
title: "DFlash Diffusion MTP For Apple Silicon Speedups"
publishDate: "2026-04-13"
date: "2026-04-13"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2043705038875328546-1.jpg"
excerpt: "A post praises DFlash speculative decoding for Apple Silicon and quotes speedups on Qwen3.5 models."
---

> Source: [@danveloper on X](https://x.com/danveloper/status/2043705038875328546) · posted 2026-04-13

![DFlash Diffusion MTP For Apple Silicon Speedups](/imgs/blogs/x-bookmarks/2043705038875328546-1.jpg)

## Summary

The author calls the work brilliant and says diffusion MTP is a great bolt-on. The quoted post presents DFlash with stock MLX at 2048 tokens on an M5 Max. Qwen3.5-4B goes from 53.74 to 219.83 tok/s, 9B from 30.96 to 127.07, and 27B-4bit from 32.35 to 62.78.

## Background

MTP means multi-token prediction. Speculative decoding drafts several tokens and verifies them in one pass. The numbers are the quoted author's benchmarks.

## What I took away

- Speedups were larger on small models.
- Benchmarks are self-reported.
- Works with stock MLX.

## Reference

- Original post: [https://x.com/danveloper/status/2043705038875328546](https://x.com/danveloper/status/2043705038875328546)
- Author: [@danveloper](https://x.com/danveloper) on X, posted 2026-04-13

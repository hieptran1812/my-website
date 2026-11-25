---
title: "DFlash Speculative Decoding Comes To MLX"
publishDate: "2026-04-15"
date: "2026-04-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2044267026080751751-1.jpg"
excerpt: "The DFlash project adds native MLX support and more models, claiming up to 4x faster lossless decoding."
---

> Source: [@zhijianliu on X](https://x.com/zhijianliu/status/2044267026080751751) · posted 2026-04-15

![DFlash Speculative Decoding Comes To MLX](/imgs/blogs/x-bookmarks/2044267026080751751-1.jpg)

## Summary

The author announces DFlash on MLX with native MLX support and more models such as Qwen3.5. They claim up to 4x faster generation that is lossless. They thank an early contributor.

## Background

Speculative decoding means a small model drafts tokens and the big model checks them in one pass. Lossless means the output is the same as normal decoding. MLX is Apple's machine learning framework.

## What I took away

- Speculative decoding speeds up local models on Apple Silicon.
- The 4x is the author's claim.
- Lossless means no quality loss.

## Reference

- Original post: [https://x.com/zhijianliu/status/2044267026080751751](https://x.com/zhijianliu/status/2044267026080751751)
- Author: [@zhijianliu](https://x.com/zhijianliu) on X, posted 2026-04-15
